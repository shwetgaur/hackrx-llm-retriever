import os
from sentence_transformers import SentenceTransformer
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Define model names
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# Download and cache the embedding model
print(f"Caching embedding model: {EMBEDDING_MODEL}")
SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model cached.")

# Download and cache the re-ranker model
print(f"Caching re-ranker model: {RERANKER_MODEL}")
HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
print("Re-ranker model cached.")