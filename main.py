from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.rag_pipeline import query_dynamic_rag

app = FastAPI(title="EnAI Dynamic RAG API")

@app.get("/query")
def query(url: str, q: str):
    """
    url: Wikipedia page URL
    q: user question
    """
    try:
        answer = query_dynamic_rag(url, q)
        return {"url": url, "query": q, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
