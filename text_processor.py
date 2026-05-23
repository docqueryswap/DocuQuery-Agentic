from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class TextProcessor:
    def __init__(self):
        # 1024-dim model to match Pinecone index
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        return splitter.split_text(text)

    def generate_embeddings(self, chunks):
        return self.model.encode(chunks)