# api/app.py

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from generator import Generator

app = FastAPI(
    title="Delhi Traffic Rules RAG API",
    description="Hybrid RAG backend for Delhi traffic law queries.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = Generator()


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query in English or Hindi")
    top_k: int = Field(default=10, description="Top K retrieval count")


class QueryResponse(BaseModel):
    answer: str
    context: str = ""


@app.get("/")
def read_root():
    return {"message": "Delhi Traffic Rules RAG API is running."}


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = generator.ask(query, top_k=request.top_k)
        
        # Safely extract answer and context
        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
        context = result.get("context", "") if isinstance(result, dict) else ""
        
        return {"answer": answer, "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")