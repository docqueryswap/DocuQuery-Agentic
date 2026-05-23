import os
import time
import logging
from pinecone import Pinecone, ServerlessSpec

class PineconeVectorStore:
    def __init__(self):
        self.index_name = 'multiagent'
        api_key = os.getenv('PINECONE_API_KEY')
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        
        logging.info(f"Connecting to Pinecone index: {self.index_name}")
        self.pc = Pinecone(api_key=api_key)
        
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            logging.info(f"Index '{self.index_name}' not found. Creating it now...")
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            while not self.pc.describe_index(self.index_name).status.get('ready', False):
                time.sleep(2)
            logging.info(f"Index '{self.index_name}' created successfully.")
        else:
            logging.info(f"Index '{self.index_name}' already exists.")
        
        self.index = self.pc.Index(self.index_name)
        logging.info("PineconeVectorStore initialized successfully.")

    def store_documents(self, chunks, vectors, metadata):
        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            records.append({
                'id': f"{metadata['doc_id']}-{i}",
                'values': vector.tolist(),
                'metadata': {'text': chunk, **metadata}
            })
        self.index.upsert(vectors=records)
        logging.info(f"Stored {len(records)} vectors in Pinecone.")

    def delete_by_metadata(self, metadata_filter):
        try:
            self.index.delete(filter=metadata_filter)
            logging.info(f"Deleted vectors with filter: {metadata_filter}")
        except Exception as e:
            logging.error(f"Error deleting vectors: {e}")

    def search_similar(self, query_vector, top_k=5, filter_doc_id=None, client_id=None):
        filter_dict = {}
        if filter_doc_id:
            filter_dict['doc_id'] = {'$eq': filter_doc_id}
        if client_id:
            filter_dict['client_id'] = {'$eq': client_id}
        
        result = self.index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        return result.matches