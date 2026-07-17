"""
Islamic AI Assistant — FastAPI Routes

API endpoints untuk Islamic AI Assistant:
- Health check
- Session management
- Chat endpoints
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.models.schemas import FinalResponse, SessionContext, GenerationConfig
from src.core.query_processor import QueryProcessor
from src.core.session_manager import SessionManager
from src.core.audit_logger import AuditLogger
from src.core.rag_engine import VectorStoreUnavailableError
from src.api.auth import get_current_user, RateLimiter
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class CreateSessionRequest(BaseModel):
    """Request untuk create session baru"""
    user_id: Optional[str] = None
    language: str = Field(default="id", description="Bahasa preferensi (id/ar/en)")
    madhab_preference: str = Field(default="shafii", description="Preferensi mazhab")

class CreateSessionResponse(BaseModel):
    """Response untuk create session"""
    session_id: UUID

class ChatRequest(BaseModel):
    """Request untuk chat endpoint"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: UUID = Field(..., description="Session UUID")
    stream: bool = Field(default=False, description="Enable streaming response")

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str

# ============================================================
# DEPENDENCY INJECTION
# ============================================================

# Global instances (akan di-inject dari main.py)
_query_processor: Optional[QueryProcessor] = None
_session_manager: Optional[SessionManager] = None
_audit_logger: Optional[AuditLogger] = None
_rate_limiter: Optional[RateLimiter] = None

def set_dependencies(
    query_processor: QueryProcessor,
    session_manager: SessionManager,
    audit_logger: AuditLogger,
    rate_limiter: RateLimiter
):
    """Set global dependencies (called from main.py)"""
    global _query_processor, _session_manager, _audit_logger, _rate_limiter
    _query_processor = query_processor
    _session_manager = session_manager
    _audit_logger = audit_logger
    _rate_limiter = rate_limiter

def get_query_processor() -> QueryProcessor:
    """Dependency untuk get QueryProcessor instance"""
    if _query_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    return _query_processor

def get_session_manager() -> SessionManager:
    """Dependency untuk get SessionManager instance"""
    if _session_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    return _session_manager

def get_audit_logger() -> AuditLogger:
    """Dependency untuk get AuditLogger instance"""
    if _audit_logger is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    return _audit_logger

def get_rate_limiter() -> RateLimiter:
    """Dependency untuk get RateLimiter instance"""
    if _rate_limiter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    return _rate_limiter

# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@router.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint (no authentication required).

    Returns:
        Status OK
    """
    return {"status": "ok"}

# ============================================================
# SESSION ENDPOINTS
# ============================================================

@router.post(
    "/api/v1/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Sessions"]
)
async def create_session(
    request: CreateSessionRequest,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Create new session.

    Args:
        request: CreateSessionRequest
        session_manager: SessionManager dependency

    Returns:
        CreateSessionResponse dengan session_id
    """
    try:
        session = session_manager.create_session(
            user_id=request.user_id,
            language=request.language,
            madhab_preference=request.madhab_preference
        )

        logger.info(f"Created new session: {session.session_id}")

        return CreateSessionResponse(session_id=session.session_id)

    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session"
        )

@router.get(
    "/api/v1/sessions/{session_id}",
    response_model=SessionContext,
    tags=["Sessions"]
)
async def get_session(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get session by ID.

    Args:
        session_id: Session UUID
        session_manager: SessionManager dependency

    Returns:
        SessionContext

    Raises:
        404 jika session tidak ditemukan
    """
    session = session_manager.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    return session

# TO BE CONTINUED in Part 2...
# (Chat endpoint akan ditambahkan di Part 2)

# ============================================================
# CHAT ENDPOINT
# ============================================================

@router.post(
    "/api/v1/chat",
    response_model=FinalResponse,
    tags=["Chat"]
)
async def chat(
    request: ChatRequest,
    fastapi_request: Request,
    query_processor: QueryProcessor = Depends(get_query_processor),
    session_manager: SessionManager = Depends(get_session_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Process chat message dan return response.

    Args:
        request: ChatRequest dengan message, session_id, stream
        fastapi_request: FastAPI Request untuk get client IP
        query_processor: QueryProcessor dependency
        session_manager: SessionManager dependency
        rate_limiter: RateLimiter dependency
        current_user: Current user dari JWT (None jika anonymous)

    Returns:
        FinalResponse dengan answer dan citations

    Raises:
        400: Invalid request (message length, etc)
        404: Session not found
        429: Rate limit exceeded
        503: Service unavailable (VectorStore error)
    """
    # Validate message length (sudah di-handle oleh Pydantic, tapi double check)
    if len(request.message) < 1 or len(request.message) > 2000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message length must be between 1 and 2000 characters"
        )

    # Check rate limit
    if current_user:
        # Authenticated user
        identifier = current_user.get("sub")
        is_authenticated = True
    else:
        # Anonymous user - use IP address
        identifier = fastapi_request.client.host
        is_authenticated = False

    rate_limiter.check_rate_limit(identifier, is_authenticated, fastapi_request)

    # Validate session exists
    session = session_manager.get_session(request.session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {request.session_id} not found"
        )

    # Handle streaming vs non-streaming
    if request.stream:
        # TODO: Implement streaming in future iteration
        # For now, return error
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Streaming not yet implemented"
        )

    # Process query
    try:
        logger.info(
            f"Processing chat request: session={request.session_id}, "
            f"message_length={len(request.message)}"
        )

        response = query_processor.process(
            query=request.message,
            session_id=request.session_id,
            config=GenerationConfig()
        )

        logger.info(
            f"Chat response generated: confidence={response.confidence:.2f}, "
            f"citations={len(response.citations)}"
        )

        return response

    except VectorStoreUnavailableError as e:
        logger.error(f"Vector store unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base temporarily unavailable. Please try again later."
        )

    except Exception as e:
        # Use repr to avoid loguru format string issues with curly braces
        logger.error("Unexpected error in chat endpoint: {}", repr(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again."
        )
