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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
LLM_MODEL = "gemini-1.5-flash" # Switched to a faster model
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# To guarantee a response under 30 seconds, we must cap the number of chunks we process.
# This prevents the embedding step from taking too long on very large documents.
# You may need to adjust this value based on the GPU performance in the execution environment.
MAX_CHUNKS_TO_PROCESS = 3500

# --- Initialize models ONCE on startup ---
print("Loading AI models on startup...")
# Using a faster LLM like gemini-1.5-flash can also reduce latency.
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
# **CRITICAL**: Set device to 'cuda' to use the GPU for embeddings.
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
print("AI models loaded successfully.")

# --- Security & Pydantic Models ---
security_scheme = HTTPBearer()
def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != HACKATHON_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

# --- FastAPI App ---
# Changed title to reflect the focus on speed
app = FastAPI(title="HackRx 6.0 Submission API (High-Speed Version)")
router = APIRouter(prefix="/api/v1")

# --- RAG Components ---
qa_prompt_template = """You are a specialized AI assistant for processing insurance claims. Your function is to answer questions about an insurance policy based ONLY on the context provided.
**Instructions:**
1. You MUST answer using ONLY the provided CONTEXT. Do not use external knowledge.
2. If the information is not in the CONTEXT, you MUST respond with "Information not found in the provided document."
3. Your response must be a direct and concise answer.
**CONTEXT:**
{context}
**QUESTION:**
{question}
**Final Answer:**
"""
qa_prompt = ChatPromptTemplate.from_template(qa_prompt_template)

def format_docs(docs):
    """Converts a list of Document objects into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- API Endpoint ---
@router.post("/hackrx/run", response_model=HackathonResponse)
async def process_documents(request: HackathonRequest, authorized: bool = Depends(get_current_user)):
    try:
        response = requests.get(request.documents)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # --- SPEED OPTIMIZATION: Cap the number of chunks ---
        if len(chunks) > MAX_CHUNKS_TO_PROCESS:
            print(f"Warning: Document is very large ({len(chunks)} chunks). Capping to the first {MAX_CHUNKS_TO_PROCESS} chunks to ensure speed.")
            chunks = chunks[:MAX_CHUNKS_TO_PROCESS]

        # --- SPEED OPTIMIZATION: Simplified & Fast Retriever ---
        # The original Ensemble + Reranker pipeline was accurate but too slow.
        # We now use a single, fast vector store. This is the main accuracy vs. speed trade-off.
        vectorstore = FAISS.from_documents(chunks, embeddings)
        # 'k' is the number of chunks sent to the LLM. 5-7 is a good balance for speed.
        retriever = vectorstore.as_retriever(search_kwargs={'k': 7})

        # --- SPEED OPTIMIZATION: Simplified RAG Chain ---
        rag_chain = (
            # The 'context' is now retrieved and formatted into a string in one step.
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        # --- SPEED OPTIMIZATION: Use .batch() for concurrent processing of questions ---
        answers = rag_chain.batch(request.questions)

        os.unlink(tmp_file_path)
        return HackathonResponse(answers=answers)

    except Exception as e:
        print(f"An error occurred: {e}")
        # It's good practice to log the full error for debugging.
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the document.")

app.include_router(router)
@app.get("/")
def read_root(): return {"message": "High-Speed API is running."}