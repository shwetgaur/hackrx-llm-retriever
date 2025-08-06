import os
import requests
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from urllib.parse import unquote_plus

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Load environment variables
load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
LLM_MODEL = "gemini-1.5-pro-latest"
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY")
CACHE_DIR = "/tmp/hf_cache" if os.path.exists("/tmp") else "./hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Initialize models ONCE on startup ---
print("Loading AI models on startup...")
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, convert_system_message_to_human=True)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'}, cache_folder=CACHE_DIR)
cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
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
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

# --- FastAPI App ---
app = FastAPI(title="HackRx 6.0 Submission API (High-Accuracy Version)")
router = APIRouter(prefix="/api/v1")

# --- RAG Components ---
qa_prompt_template = """You are a highly specialized AI assistant for processing insurance claims. Your ONLY function is to answer questions about an insurance policy based on the context provided.
**Instructions:**
1. You MUST answer the question using ONLY the provided CONTEXT.
2. Do not use any external knowledge or make assumptions.
3. If the information to answer the question is not in the CONTEXT, you MUST respond with "Information not found in the provided document."
4. Your response must be a direct and concise answer to the user's question, not a conversation.
**CONTEXT:**
{context}
**QUESTION:**
{question}
**Final Answer:**
"""
qa_prompt = ChatPromptTemplate.from_template(qa_prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- API Endpoint ---
@router.post("/hackrx/run", response_model=HackathonResponse)
async def process_documents(request: HackathonRequest, authorized: bool = Depends(get_current_user)):
    try:
        documents_url= request.documents
        print(f"Downloading document from: {documents_url}")
        response = requests.get(documents_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # --- Build Advanced Retriever for this request ---
        faiss_vectorstore = FAISS.from_documents(chunks, embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={'k': 10})
        
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 10
        
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
        
        rag_chain = (
            {"context": compression_retriever, "question": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        answers = [rag_chain.invoke(q) for q in request.questions]

        os.unlink(tmp_file_path)
        return HackathonResponse(answers=answers)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)
@app.get("/")
def read_root(): return {"message": "API is running."}