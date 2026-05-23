from pypdf import PdfReader
from docx import Document

class DocumentProcessor:
    def process_uploaded_file(self, file_path):
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            return ' '.join([p.extract_text() or '' for p in reader.pages])
        if file_path.endswith('.docx'):
            doc = Document(file_path)
            return '\n'.join([p.text for p in doc.paragraphs])
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()