# api/app.py

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from generator import Generator

app = FastAPI(title="Delhi Traffic Rules RAG API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = Generator()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


class QueryResponse(BaseModel):
    answer: str
    context: str
    metrics: Optional[dict] = None


@app.get("/")
def read_root():
    return {"message": "Delhi Traffic Rules RAG API is running."}


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        result = generator.ask(request.query, top_k=request.top_k)
        return {
            "answer": result["answer"],
            "context": result["context"],
            "metrics": result.get("metrics")
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "context": "", "metrics": None}