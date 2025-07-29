import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Core LangChain imports
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser

# Community and other necessary imports
from langchain_community.document_loaders import DirectoryLoader
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
DOCS_PATH = "./docs"
DB_FAISS_PATH = "faiss_index"
EMBEDDING_MODEL = "./embedding_model"
LLM_MODEL = "gemini-1.5-pro-latest"
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- FastAPI Application ---
app = FastAPI(title="Optimized Insurance QA System")

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    text: str

class QueryResponse(BaseModel):
    decision: str
    amount: int
    justification: list[str]

class TransformedQuery(BaseModel):
    search_query: str = Field(description="An optimized search query combining the original query and new keywords.")

# Global variables for our chains
qa_chain = None
query_transformer_chain = None

@app.on_event("startup")
def startup_event():
    """Initializes all components of the advanced RAG pipeline."""
    global qa_chain, query_transformer_chain
    print("Initializing Optimized RAG pipeline...")

    # --- Setup LLM and a shared retriever ---
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, convert_system_message_to_human=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    
    loader = DirectoryLoader(DOCS_PATH, glob='**/*.pdf')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    faiss_vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={'k': 5})
    
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
    
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
    print("Advanced retriever is ready.")

    # --- 1. SETUP QUERY TRANSFORMER CHAIN ---
    parser = PydanticOutputParser(pydantic_object=TransformedQuery)
    transform_prompt_template = """You are a search expert. Your task is to analyze a user's query and transform it into a more effective search query for a document retrieval system.
    Analyze the user's query for contextual clues like policy duration, location, or specific conditions. Based on these clues, add general keywords that are likely to appear in a policy document.
    For example:
    - If you see "3-month policy", add "waiting period" and "exclusions".
    - If you see a procedure like "knee surgery", add "joint replacement surgery" and "specified disease".
    Return a JSON object with a single key "search_query" containing the new, optimized query.
    USER QUERY: {query}
    {format_instructions}"""
    prompt = PromptTemplate(template=transform_prompt_template, input_variables=["query"], partial_variables={"format_instructions": parser.get_format_instructions()})
    query_transformer_chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    print("Query Transformer chain is ready.")

    # --- 2. SETUP THE MAIN QA CHAIN ---
    qa_prompt_template = """You are an expert insurance claims adjudicator. Your task is to analyze a user's query based ONLY on the provided policy document context.
    If the context does not contain the answer, state that the information is not found.
    CONTEXT: {context}
    QUERY: {question}
    INSTRUCTIONS: Your final answer MUST be a single, valid JSON object with the following keys: "decision", "amount", and "justification".
    - "decision" must be "Approved", "Rejected", or "Information Not Found".
    - "amount" must be an integer. If not applicable, use 0.
    - "justification" must be an array of strings, quoting the exact clauses from the context that support your decision.
    Do not add any text before or after the JSON object.
    FINAL JSON ANSWER:"""
    qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=compression_retriever,
        return_source_documents=False,
        chain_type_kwargs={'prompt': qa_prompt}
    )
    print("Main QA Chain is ready. Application startup complete.")

@app.post("/query", response_model=QueryResponse)
async def process_query(query: QueryRequest):
    if not qa_chain or not query_transformer_chain:
        raise HTTPException(status_code=503, detail="Chains are not initialized.")

    print(f"Original query: {query.text}")
    
    # Step 1: Transform the query
    try:
        transformed_query_obj = await query_transformer_chain.ainvoke({"query": query.text})
        search_query = transformed_query_obj['text'].search_query
        print(f"Transformed query: {search_query}")
    except Exception as e:
        print(f"Error during query transformation: {e}. Using original query.")
        search_query = query.text

    # Step 2: Run the main QA chain with the better query
    result = qa_chain.invoke(search_query)
    
    try:
        response_str = result.get('result', '{}')
        json_start = response_str.find('{')
        json_end = response_str.rfind('}') + 1
        clean_json_str = response_str[json_start:json_end]
        response_data = json.loads(clean_json_str)
        return QueryResponse(**response_data)
    except Exception as e:
        print(f"Error parsing LLM response: {e}\nRaw response: {result.get('result')}")
        raise HTTPException(status_code=500, detail="Failed to get a valid JSON response from the language model.")

@app.get("/")
def read_root():
    return {"message": "Welcome! Go to /docs to test the Optimized RAG endpoint."}