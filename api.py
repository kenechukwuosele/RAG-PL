from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import ask as ask_question
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not os.path.exists("db"):
        logger.info("Indexing papers...")
        from parser import load_all_papers
        from retriever import index_papers
        papers = load_all_papers()
        index_papers(papers)
        logger.info("Indexing complete!")
    yield
    # Shutdown (nothing to clean up)

app = FastAPI(title="RAG Research Assistant", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str
    evaluate: bool = False

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    cached: bool
    evaluation: str | None = None

@app.get("/")
def root():
    return {"status": "RAG Research Assistant is running"}

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if len(request.question) > 500:
        raise HTTPException(status_code=400, detail="Question too long — max 500 characters")

    try:
        logger.info(f"Query received: {request.question}")
        result = ask_question(request.question, run_eval=request.evaluate)
        logger.info(f"Query answered. Cached: {result['cached']}")
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong processing your query")