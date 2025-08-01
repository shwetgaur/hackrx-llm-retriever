import os
import json
import requests
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, HttpUrl

# LangChain Imports
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

CACHE_DIR = os.path.join(os.getcwd(), "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)



# --- Configuration ---
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
LLM_MODEL = "gemini-1.5-pro-latest"
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY")

# --- Security ---
security_scheme = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    """Validates the bearer token."""
    if credentials.scheme != "Bearer" or credentials.credentials != HACKATHON_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# --- Pydantic Models for API ---
class HackathonRequest(BaseModel):
    documents: HttpUrl
    questions: list[str]

class HackathonResponse(BaseModel):
    answers: list[str]

# --- FastAPI Application ---
app = FastAPI(title="HackRx 6.0 Submission API")

# --- Core RAG Logic ---
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, convert_system_message_to_human=True)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL, 
    model_kwargs={'device': 'cpu'},
    cache_folder=CACHE_DIR  # Use the new, guaranteed-to-be-writable path
)

qa_prompt_template = """
You are an expert AI assistant for answering questions about insurance policies.
Answer the question based ONLY on the provided context.
If the information is not in the context, say "Information not found in the provided document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
qa_prompt = ChatPromptTemplate.from_template(qa_prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackathonResponse)
async def process_documents(request: HackathonRequest, authorized: bool = Depends(get_current_user)):
    """
    This endpoint downloads a document, processes it, and answers questions.
    """
    try:
        # 1. Download the document from the URL
        print(f"Downloading document from {request.documents}")
        response = requests.get(str(request.documents))
        response.raise_for_status() # Raise an exception for bad status codes

        # Save PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        # 2. Build a temporary RAG index in memory for this request
        print(f"Processing and indexing the document...")
        loader = UnstructuredPDFLoader(file_path=tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # Create a FAISS index in memory
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        # 3. Create the RAG chain for this request
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        print("RAG chain created for this request.")

        # 4. Loop through questions and get answers
        answers = []
        for question in request.questions:
            print(f"Answering question: {question}")
            answer = rag_chain.invoke(question)
            answers.append(answer)

        # Clean up the temporary file
        os.unlink(tmp_file_path)

        return HackathonResponse(answers=answers)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "HackRx 6.0 API is running."}