# agent/graph.py
import os
import logging
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

import sys
sys.path.append('/app')

from vector_store import PineconeVectorStore
from text_processor import TextProcessor
from rag_pipeline import RAGPipeline

class CorrelateState(TypedDict):
    query: str
    document_context: str
    web_results: List[str]
    sources: List[str]
    correlation_report: str
    document_summary: str

def retrieve_document_context(state: CorrelateState) -> dict:
    query = state["query"]
    try:
        text_proc = TextProcessor()
        vector_db = PineconeVectorStore()
        query_embedding = text_proc.generate_embeddings([query])[0]
        docs = vector_db.search_similar(query_embedding, top_k=5)
        context = "\n".join([doc["metadata"]["text"] for doc in docs])

        summary_prompt = f"Summarize the main topic and key points of this document in one sentence:\n{context[:2000]}"
        rag = RAGPipeline()
        doc_summary = rag.generate_answer(summary_prompt)

        return {
            "document_context": context,
            "document_summary": doc_summary
        }
    except Exception as e:
        logging.error(f"Document retrieval error: {e}")
        return {
            "document_context": "No relevant document context found.",
            "document_summary": "Unknown document type."
        }

def search_web(state: CorrelateState) -> dict:
    user_query = state["query"]
    doc_summary = state.get("document_summary", "")

    if doc_summary and doc_summary != "Unknown document type.":
        search_query = f"{user_query} related to {doc_summary}"
    else:
        search_query = user_query

    logging.info(f"Web search query: {search_query}")

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")
    try:
        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(search_query, max_results=3, search_depth="basic")
        web_results = [result["content"] for result in response["results"]]
        sources = [result["url"] for result in response["results"]]
        return {
            "web_results": web_results,
            "sources": sources
        }
    except Exception as e:
        logging.error(f"Web search error: {e}")
        return {
            "web_results": ["Web search failed."],
            "sources": []
        }

def correlate_node(state: CorrelateState) -> dict:
    document_context = state["document_context"]
    web_results = "\n".join(state["web_results"])
    query = state["query"]
    doc_summary = state.get("document_summary", "")

    rag = RAGPipeline()
    prompt = f"""You are a research analyst. Compare the information from a specific document with the latest web search results on a related topic.

Document Summary: {doc_summary}
User Query: {query}

=== DOCUMENT EXCERPT ===
{document_context[:3000]}

=== WEB SEARCH RESULTS ===
{web_results[:3000]}

Instructions:
1. First, determine if the web results are directly relevant to the document's content. If not, state that the web results discuss a different topic and cannot be meaningfully correlated.
2. If they are relevant, identify:
   - Agreements (where the document aligns with current online information)
   - Contradictions (where the document differs from recent findings)
   - Gaps (what important information is present online but missing from the document)
3. Provide a concise, actionable analysis. If no meaningful correlation exists, suggest a better query or topic.

Analysis:"""

    report = rag.generate_answer(prompt)
    return {"correlation_report": report}

def build_correlate_graph():
    workflow = StateGraph(CorrelateState)

    workflow.add_node("retrieve_doc", retrieve_document_context)
    workflow.add_node("search_web", search_web)
    workflow.add_node("correlate", correlate_node)

    workflow.set_entry_point("retrieve_doc")
    workflow.add_edge("retrieve_doc", "search_web")
    workflow.add_edge("search_web", "correlate")
    workflow.add_edge("correlate", END)

    return workflow.compile()