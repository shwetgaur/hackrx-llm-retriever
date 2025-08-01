import os
import requests
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from urllib.parse import unquote

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
LLM_MODEL = "gemini-1.5-pro-latest"
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY")
CACHE_DIR = "/tmp/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# --- OPTIMIZATION: Initialize heavy models ONCE on startup ---
print("Loading AI models on startup...")
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, convert_system_message_to_human=True)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    cache_folder=CACHE_DIR
)
print("AI models loaded successfully.")

# --- Security ---
security_scheme = HTTPBearer()
def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != HACKATHON_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# --- Pydantic Models ---
class HackathonRequest(BaseModel):
    documents: str
    questions: list[str]

class HackathonResponse(BaseModel):
    answers: list[str]

# --- FastAPI Application ---
app = FastAPI(title="HackRx 6.0 Submission API (Optimized)")

# --- RAG Logic Components ---
qa_prompt_template = """You are an expert AI assistant... (rest of your prompt)"""
qa_prompt = ChatPromptTemplate.from_template(qa_prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackathonResponse)
async def process_documents(request: HackathonRequest, authorized: bool = Depends(get_current_user)):
    try:
        decoded_url = unquote(request.documents)
        print(f"Downloading document from: {decoded_url}")
        response = requests.get(decoded_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        loader = UnstructuredPDFLoader(file_path=tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, embeddings) # Re-uses the globally loaded embeddings
        retriever = vectorstore.as_retriever()

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | qa_prompt
            | llm # Re-uses the globally loaded LLM
            | StrOutputParser()
        )

        answers = []
        for question in request.questions:
            answer = rag_chain.invoke(question)
            answers.append(answer)

        os.unlink(tmp_file_path)
        return HackathonResponse(answers=answers)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "HackRx 6.0 API is running."}