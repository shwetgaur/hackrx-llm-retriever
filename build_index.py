import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DOCS_PATH = "./docs"
DB_FAISS_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

def create_vector_db():
    """Processes all documents in the docs folder and saves a single FAISS index."""
    print("Starting database creation from all documents...")
    loader = DirectoryLoader(DOCS_PATH, glob='**/*.pdf', loader_cls=PyMuPDFLoader, show_progress=True)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Creating and saving FAISS index to {DB_FAISS_PATH}...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    print("✅ Vector database created successfully.")

if __name__ == "__main__":
    create_vector_db()