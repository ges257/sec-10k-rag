# IBM 10-K RAG System - FastAPI Backend
# RESTful API for retrieval and answer generation

import time
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import system components
from src.chunker import DocumentChunker
from src.retriever import HybridRetriever
from src.generator import AnswerGenerator
from src.index_manager import VersionedIndexManager


# ============================================================================
# Pydantic Models
# ============================================================================

class QuestionRequest(BaseModel):
    """Request model for /ask endpoint."""
    question: str = Field(..., description="Question to ask about IBM 10-K")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources")
    use_rerank: bool = Field(default=True, description="Use cross-encoder re-ranking")
    use_hybrid: bool = Field(default=True, description="Use hybrid search")
    include_sources: bool = Field(default=True, description="Include source excerpts")


class Source(BaseModel):
    """Source citation model."""
    rank: int
    section: str
    score: float
    preview: str


class AnswerResponse(BaseModel):
    """Response model for /ask endpoint."""
    question: str
    answer: str
    sources: Optional[List[Source]] = None
    metrics: dict


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    system_ready: bool
    num_chunks: int
    index_version: Optional[str]
    model_info: dict


# ============================================================================
# Global State
# ============================================================================

class SystemState:
    """Manages global system state."""
    def __init__(self):
        self.retriever: Optional[HybridRetriever] = None
        self.generator: Optional[AnswerGenerator] = None
        self.chunks: List[dict] = []
        self.index_version: Optional[str] = None
        self.ready: bool = False

state = SystemState()


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize system on startup."""
    print("Initializing IBM 10-K RAG System...")

    try:
        # Check for pre-built index
        index_manager = VersionedIndexManager("indices")
        latest_version = index_manager.get_latest_version()

        if latest_version:
            print(f"Loading index version: {latest_version}")
            _, chunks, _, metadata = index_manager.load("latest")
            state.chunks = chunks
            state.index_version = latest_version
        else:
            # Build from scratch
            print("No pre-built index. Building from PDF...")
            chunker = DocumentChunker()
            state.chunks = chunker.process_pdf("data/ibm_10k_2024.pdf")
            state.index_version = "built_on_startup"

        # Initialize retriever
        print("Building retriever...")
        state.retriever = HybridRetriever(
            chunks=state.chunks,
            use_cross_encoder=True
        )

        # Initialize generator
        print("Loading generator...")
        state.generator = AnswerGenerator()

        state.ready = True
        print(f"System ready: {len(state.chunks)} chunks indexed")

    except Exception as e:
        print(f"Error during initialization: {e}")
        state.ready = False

    yield

    # Cleanup (if needed)
    print("Shutting down...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="IBM 10-K RAG API",
    description="Production-grade retrieval-augmented generation for SEC filing analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns system status and configuration info.
    """
    return HealthResponse(
        status="healthy" if state.ready else "initializing",
        system_ready=state.ready,
        num_chunks=len(state.chunks),
        index_version=state.index_version,
        model_info={
            "embedding_model": "all-MiniLM-L6-v2",
            "cross_encoder": "ms-marco-MiniLM-L-6-v2",
            "generator": "flan-t5-small"
        }
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about IBM's 10-K filing.

    Returns answer with optional source citations and performance metrics.
    """
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail="System not ready. Please wait for initialization."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        start_time = time.time()

        # Retrieve relevant chunks
        if request.use_hybrid:
            results, search_metrics = state.retriever.search(
                query=request.question,
                top_k=request.top_k,
                use_rerank=request.use_rerank,
                apply_section_boost=True
            )
        else:
            results = state.retriever.baseline_search(
                request.question,
                top_k=request.top_k
            )
            search_metrics = {"method": "dense_only"}

        retrieval_time = (time.time() - start_time) * 1000

        # Generate answer
        gen_start = time.time()
        response = state.generator.generate_with_sources(
            request.question,
            results,
            include_sources=request.include_sources
        )
        gen_time = (time.time() - gen_start) * 1000

        # Build sources list
        sources = None
        if request.include_sources:
            sources = []
            for i, (chunk, score) in enumerate(results, 1):
                sources.append(Source(
                    rank=i,
                    section=chunk.get('section', 'unknown'),
                    score=round(score, 4),
                    preview=chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text']
                ))

        # Build metrics
        total_time = retrieval_time + gen_time
        metrics = {
            "retrieval_ms": round(retrieval_time, 1),
            "generation_ms": round(gen_time, 1),
            "total_ms": round(total_time, 1),
            "method": search_metrics.get("method", "hybrid"),
            "reranked": search_metrics.get("reranked", False),
            "num_sources": len(results)
        }

        return AnswerResponse(
            question=request.question,
            answer=response['answer'],
            sources=sources,
            metrics=metrics
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "IBM 10-K RAG API",
        "version": "1.0.0",
        "description": "Production-grade retrieval-augmented generation for SEC filing analysis",
        "endpoints": {
            "/health": "GET - Health check and system status",
            "/ask": "POST - Ask questions about IBM 10-K",
            "/docs": "GET - Interactive API documentation"
        }
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
