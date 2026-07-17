"""
Islamic AI Assistant — FastAPI Application Entry Point

Main application dengan:
- FastAPI app initialization
- Middleware setup (CORS, logging)
- Lifespan events (startup/shutdown)
- Router registration
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, set_dependencies
from src.api.auth import RateLimiter
from src.core.vector_store import VectorStore
from src.core.rag_engine import create_rag_engine
from src.core.session_manager import SessionManager
from src.core.audit_logger import AuditLogger
from src.core.query_processor import create_query_processor
from src.core.embedding_model import get_embedding_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global instances
vector_store = None
rag_engine = None
session_manager = None
audit_logger = None
query_processor = None
rate_limiter = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager untuk startup dan shutdown events.
    """
    # Startup
    logger.info("Starting Islamic AI Assistant API...")

    global vector_store, rag_engine, session_manager, audit_logger, query_processor, rate_limiter

    try:
        # Initialize embedding model (pre-load untuk avoid cold start)
        logger.info("Loading embedding model...")
        embedding_model = get_embedding_model()
        logger.info(f"Embedding model loaded: {embedding_model.model_name}")

        # Initialize vector store
        logger.info("Initializing vector store...")
        chroma_mode = os.getenv("CHROMA_MODE", "persistent")
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

        vector_store = VectorStore(
            mode=chroma_mode,
            persist_directory="./data/chroma",
            host=chroma_host,
            port=chroma_port
        )

        # Health check vector store
        if not vector_store.health_check():
            logger.error("Vector store health check failed!")
            raise RuntimeError("Vector store is not available")

        logger.info("Vector store initialized and healthy")

        # Initialize RAG engine
        logger.info("Initializing RAG engine...")
        rag_engine = create_rag_engine(vector_store)

        # Initialize session manager
        logger.info("Initializing session manager...")
        session_manager = SessionManager(db_path="data/sessions.db")

        # Initialize audit logger
        logger.info("Initializing audit logger...")
        audit_logger = AuditLogger(db_path="data/audit.db")

        # Initialize query processor
        logger.info("Initializing query processor...")
        query_processor = create_query_processor(
            rag_engine=rag_engine,
            session_manager=session_manager,
            audit_logger=audit_logger
        )

        # Initialize rate limiter
        logger.info("Initializing rate limiter...")
        rate_limiter = RateLimiter(audit_logger=audit_logger)

        # Inject dependencies into routes
        set_dependencies(
            query_processor=query_processor,
            session_manager=session_manager,
            audit_logger=audit_logger,
            rate_limiter=rate_limiter
        )

        logger.info("✅ All components initialized successfully")
        logger.info("Islamic AI Assistant API is ready")

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        raise

    yield  # Application is running

    # Shutdown
    logger.info("Shutting down Islamic AI Assistant API...")
    logger.info("Cleanup completed")

# Create FastAPI app
app = FastAPI(
    title="Islamic AI Assistant",
    description="RAG-based Islamic Q&A Assistant with content filtering",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register router
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint dengan informasi API"""
    return {
        "name": "Islamic AI Assistant",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn

    # Run with uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Development mode
        log_level="info"
    )
