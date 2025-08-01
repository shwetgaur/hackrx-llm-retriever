import os
import time
# UPDATED: Import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DOCS_PATH = "./docs" # UPDATED: Path to the documents folder
DB_FAISS_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

def create_vector_db():
    if not os.path.exists(DOCS_PATH):
        print(f"Error: The directory '{DOCS_PATH}' was not found.")
        return

    if os.path.exists(DB_FAISS_PATH):
        print(f"Database already exists at '{DB_FAISS_PATH}'. Skipping creation.")
        return

    start_time = time.time()
    print("Starting database creation from multiple documents...")

    # UPDATED: Use DirectoryLoader to load all files from the folder
    loader = DirectoryLoader(DOCS_PATH, glob='**/*.pdf', show_progress=True)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split the documents into {len(chunks)} chunks.")

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, 
        model_kwargs={'device': 'cpu'},
        cache_folder='/data/hf_cache'  # UPDATED: The correct persistent path
    )

    print("Creating vector database...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)

    end_time = time.time()
    print(f"Vector database created successfully in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    create_vector_db()