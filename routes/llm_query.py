from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
import tempfile
import requests

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str  # Direct URL of the document
    questions: list

@app.post("/query")
async def query(request: QueryRequest):
    try:
        # Decode URL (if needed)
        from urllib.parse import unquote
        url = unquote(request.documents)

        # 🟢 Step 1: Load PDF from URL
        loader = UnstructuredURLLoader(urls=[url], mode="elements")
        documents = loader.load()

        if not documents:
            raise HTTPException(status_code=400, detail="No content could be extracted from the document.")

        # 🧱 Step 2: Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # 🧠 Step 3: Create retriever
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Optional: add BM25 fallback retriever
        # bm25_retriever = BM25Retriever.from_documents(docs)

        # 🤖 Step 4: QA Chain
        llm = GoogleGenerativeAI(model="models/text-bison-001", temperature=0)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        # 📝 Step 5: Ask each question
        answers = []
        for question in request.questions:
            result = qa.run(question)
            if "Information not found" in result or not result.strip():
                result = "Sorry, I couldn’t find that in the document."
            answers.append(result)

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
