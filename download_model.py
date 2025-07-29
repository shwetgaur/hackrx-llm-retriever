from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "./embedding_model" # We'll save it in a folder named 'embedding_model'

if not os.path.exists(MODEL_PATH):
    print(f"Downloading model '{MODEL_NAME}' to '{MODEL_PATH}'...")
    # This command downloads the model from Hugging Face
    model = SentenceTransformer(MODEL_NAME)
    model.save(MODEL_PATH)
    print("Model downloaded successfully!")
else:
    print(f"Model already exists at '{MODEL_PATH}'.")