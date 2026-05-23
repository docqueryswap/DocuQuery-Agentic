import logging
import os
import uuid
import io
import json
import fcntl
import asyncio
import concurrent.futures
from typing import Optional, List
from dotenv import load_dotenv
from docx import Document as DocxDocument

# FastAPI imports
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Import custom modules
from document_processor import DocumentProcessor
from text_processor import TextProcessor
from vector_store import PineconeVectorStore
from mcp_protocol import ModelContextProtocol
from rag_pipeline import RAGPipeline
from agent.graph import build_correlate_graph

# ============================================================
# App Initialization
# ============================================================
app = FastAPI(title="DocuQuery Agentic", version="3.0.0")

# ============================================================
# Configuration
# ============================================================
UPLOAD_FOLDER = 'uploads'
STATE_DIR = '/tmp/user_states'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

doc_proc = DocumentProcessor()
text_proc = TextProcessor()
rag = RAGPipeline()
mcp = ModelContextProtocol()

vector_db = None
correlate_agent = None

# ============================================================
# Pydantic Models
# ============================================================
class AskRequest(BaseModel):
    client_id: str
    question: str
    style: str = "default"

class CorrelateRequest(BaseModel):
    client_id: str
    query: str

class EditRequest(BaseModel):
    client_id: str
    instruction: str

class ReportRequest(BaseModel):
    client_id: str

class SummarizeRequest(BaseModel):
    client_id: str

class ClearRequest(BaseModel):
    client_id: str

# ============================================================
# Helper Functions (same as Flask version)
# ============================================================
def get_state_file_path(client_id: str) -> str:
    safe_id = "".join(c for c in client_id if c.isalnum() or c in '-_')
    return os.path.join(STATE_DIR, f"{safe_id}.json")

def read_state(client_id: str) -> dict:
    filepath = get_state_file_path(client_id)
    try:
        with open(filepath, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def write_state(client_id: str, state: dict):
    filepath = get_state_file_path(client_id)
    with open(filepath, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(state, f)
        fcntl.flock(f, fcntl.LOCK_UN)

# ============================================================
# Startup Event
# ============================================================
@app.on_event("startup")
async def startup_event():
    global vector_db, correlate_agent
    try:
        vector_db = PineconeVectorStore()
        logging.info("✅ Vector store initialized.")
    except Exception as e:
        logging.error(f"❌ Vector store init failed: {e}")
    try:
        correlate_agent = build_correlate_graph()
        logging.info("✅ Correlate agent initialized.")
    except Exception as e:
        logging.error(f"❌ Correlate agent init failed: {e}")

# ============================================================
# Frontend Route
# ============================================================
@app.get("/")
async def home():
    return FileResponse("templates/index.html")

# ============================================================
# Core Endpoints
# ============================================================
@app.post("/upload")
async def upload_file(client_id: str = Form(...), file: UploadFile = File(...)):
    global vector_db
    try:
        if vector_db is None:
            vector_db = PineconeVectorStore()

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        text = doc_proc.process_uploaded_file(file_path)
        chunks = text_proc.split_text(text)
        embeddings = text_proc.generate_embeddings(chunks)

        old_state = read_state(client_id)
        old_doc_id = old_state.get('doc_id')
        if old_doc_id:
            try:
                vector_db.delete_by_metadata({'doc_id': old_doc_id, 'client_id': client_id})
                logging.info(f"Deleted previous doc: {old_doc_id} for client {client_id}")
            except Exception as e:
                logging.warning(f"Delete error: {e}")

        new_doc_id = str(uuid.uuid4())
        vector_db.store_documents(chunks, embeddings, {'doc_id': new_doc_id, 'client_id': client_id})

        new_state = {
            'doc_id': new_doc_id,
            'document_text': text,
            'document_filename': file.filename
        }
        write_state(client_id, new_state)

        return JSONResponse({
            'message': f'✅ File processed: {file.filename}',
            'doc_id': new_doc_id
        })
    except Exception as e:
        logging.error(f'Upload error: {e}')
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    try:
        state = read_state(request.client_id)
        text = state.get('document_text')
        if not text:
            return JSONResponse({'error': 'No document uploaded yet.'}, status_code=400)

        prompt = f"""You are a helpful assistant. Provide a concise summary of the following document in 3-5 sentences.

DOCUMENT CONTENT:
{text[:5000]}

SUMMARY:"""
        summary = rag.generate_answer(prompt)
        return JSONResponse({'summary': summary})
    except Exception as e:
        logging.error(f'Summarize error: {e}')
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/ask")
async def ask(request: AskRequest):
    global vector_db
    try:
        state = read_state(request.client_id)
        doc_id = state.get('doc_id')
        if not doc_id:
            return JSONResponse({'error': 'No document uploaded yet.'}, status_code=400)

        q_embed = text_proc.generate_embeddings([request.question])[0]
        docs = vector_db.search_similar(q_embed, filter_doc_id=doc_id, top_k=5, client_id=request.client_id)

        if not docs:
            return JSONResponse({'answer': 'No relevant information found in the document.'})

        context = '\n'.join([d['metadata']['text'] for d in docs])
        prompt = mcp.get_context_prompt(request.style, request.question, context)
        answer = rag.generate_answer(prompt)

        return JSONResponse({'answer': answer})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/correlate")
async def correlate(request: CorrelateRequest):
    global correlate_agent
    try:
        state = read_state(request.client_id)
        doc_id = state.get('doc_id')
        if not doc_id:
            return JSONResponse({'error': 'No document uploaded yet.'}, status_code=400)
        if correlate_agent is None:
            return JSONResponse({'error': 'Correlate agent is not available.'}, status_code=500)

        result = correlate_agent.invoke({"query": request.query})
        return JSONResponse({
            'analysis': result.get('correlation_report', 'No analysis generated.'),
            'sources': result.get('sources', [])
        })
    except Exception as e:
        logging.error(f'Correlate error: {str(e)}')
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/report")
async def report(request: ReportRequest):
    try:
        state = read_state(request.client_id)
        text = state.get('document_text')
        if not text:
            return JSONResponse({'error': 'No document uploaded yet.'}, status_code=400)

        prompt = f"""You are a business analyst. Based on the following document content, generate an actionable report with:
- Executive Summary (2-3 sentences)
- Key Findings (bullet points)
- Recommendations (bullet points)

DOCUMENT CONTENT:
{text[:5000]}

REPORT:"""
        report_text = rag.generate_answer(prompt)
        return JSONResponse({'report': report_text})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/edit")
async def edit_document(request: EditRequest):
    try:
        state = read_state(request.client_id)
        text = state.get('document_text')
        if not text:
            return JSONResponse({'error': 'No document uploaded yet.'}, status_code=400)

        filename = state.get('document_filename', 'document')

        system_prompt = (
            "You are an expert document editor. "
            "Apply the user's editing instruction to the provided document text. "
            "Return ONLY the complete revised text, with no extra commentary, explanations, or markdown. "
            "Preserve the original structure (paragraphs, line breaks, bullet points) as much as possible."
        )

        user_prompt = f"""Original Document Text:
---
{text}
---

Editing Instruction: {request.instruction}

Please provide the fully revised document text:"""

        response = rag.client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        edited_text = response.choices[0].message.content.strip()

        state['document_text'] = edited_text
        write_state(request.client_id, state)

        doc = DocxDocument()
        for para in edited_text.split('\n\n'):
            if para.strip():
                doc.add_paragraph(para.strip())

        file_stream = io.BytesIO()
        doc.save(file_stream)
        file_stream.seek(0)

        return StreamingResponse(
            file_stream,
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            headers={'Content-Disposition': f'attachment; filename="edited_{filename.rsplit(".", 1)[0]}.docx"'}
        )
    except Exception as e:
        logging.error(f'Edit error: {str(e)}')
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/clear")
async def clear_state(request: ClearRequest):
    write_state(request.client_id, {})
    return JSONResponse({'message': 'Session cleared'})

# ============================================================
# Streaming Correlation Endpoint (SSE)
# ============================================================
@app.get("/correlate/stream")
async def correlate_stream(request: Request, query: str, client_id: str):
    state = read_state(client_id)
    if not state.get('doc_id'):
        return JSONResponse({'error': 'No document uploaded yet.'}, status_code=400)

    async def event_generator():
        yield {"event": "connected", "data": "Stream established."}
        try:
            yield {"event": "status", "data": "Starting correlation analysis..."}
            await asyncio.sleep(0.5)
            if correlate_agent is None:
                yield {"event": "error", "data": "Correlate agent not available."}
                return

            yield {"event": "status", "data": "Retrieving document context..."}
            await asyncio.sleep(0.5)
            yield {"event": "status", "data": "Searching the web for latest information..."}
            await asyncio.sleep(0.5)
            yield {"event": "status", "data": "Correlating findings and generating report..."}

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(
                    pool,
                    lambda: correlate_agent.invoke({"query": query})
                )

            analysis = result.get('correlation_report', 'No analysis generated.')
            sources = result.get('sources', [])

            yield {
                "event": "result",
                "data": json.dumps({
                    "analysis": analysis,
                    "sources": sources
                })
            }
        except Exception as e:
            logging.error(f"Streaming error: {e}")
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )