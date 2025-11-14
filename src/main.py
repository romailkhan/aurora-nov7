from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from src.models.QueryModels import QueryRequest, QueryResponse
from src.qa_engine import query
from src.vector_store.embedding_init import load_embeddings

env_path1 = Path(__file__).parent.parent / ".env"
load_dotenv(env_path1)

vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store
    vector_store = load_embeddings()
    if vector_store is None:
        raise RuntimeError("Failed to initialize vector store")
    yield

app = FastAPI(lifespan=lifespan,
    title="Romail Khan - Aurora Q/A System",
    description="Aurora take home november 7th. Built by Romail Khan.",
)

@app.post(
    "/ask", 
    response_model=QueryResponse,
    summary="Ask a question about member data",
    description="Submit a question to get an answer based on Aurora member messages."
)
async def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        answer, _ = query(request.query, vector_store=vector_store)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

