"""
Islamic AI Assistant — Authentication & Rate Limiting

JWT authentication dan rate limiting untuk API endpoints.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from cachetools import TTLCache

from src.core.audit_logger import AuditLogger
from src.utils.logger import get_logger

logger = get_logger(__name__)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

class RateLimiter:
    """
    Rate limiter menggunakan TTLCache.
    - Per user_id: 60 requests/minute
    - Per IP address (anonymous): 10 requests/minute
    """

    def __init__(self, audit_logger: AuditLogger):
        """
        Initialize rate limiter.

        Args:
            audit_logger: AuditLogger instance untuk logging rejections
        """
        self.audit_logger = audit_logger

        # Cache per user_id: max 60 req/min
        self.user_cache = TTLCache(maxsize=10000, ttl=60)

        # Cache per IP: max 10 req/min
        self.ip_cache = TTLCache(maxsize=10000, ttl=60)

        logger.info("Rate limiter initialized")

    def check_rate_limit(
        self,
        identifier: str,
        is_authenticated: bool,
        request: Request
    ) -> bool:
        """
        Check rate limit untuk user atau IP.

        Args:
            identifier: user_id (authenticated) atau IP address (anonymous)
            is_authenticated: True jika user authenticated
            request: FastAPI Request object

        Returns:
            True jika allowed, False jika rate limit exceeded

        Raises:
            HTTPException 429 jika rate limit exceeded
        """
        cache = self.user_cache if is_authenticated else self.ip_cache
        limit = 60 if is_authenticated else 10

        # Get current count
        current_count = cache.get(identifier, 0)

        if current_count >= limit:
            # Log rejection
            self.audit_logger.log_flagged_content(
                session_id=None,
                query=f"Rate limit exceeded: {identifier}",
                violation_category="RATE_LIMIT",
                confidence=1.0,
                reason=f"Exceeded {limit} requests per minute"
            )

            logger.warning(f"Rate limit exceeded for {identifier}")

            # Return 429 with Retry-After header
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": "60"}
            )

        # Increment counter
        cache[identifier] = current_count + 1
        return True

def create_token(user_id: str) -> str:
    """
    Create JWT access token.

    Args:
        user_id: User identifier

    Returns:
        JWT token string
    """
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.utcnow() + expires_delta

    to_encode = {
        "sub": user_id,
        "exp": expire
    }

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.debug(f"Created token for user: {user_id}")

    return encoded_jwt

def verify_token(token: str, audit_logger: Optional[AuditLogger] = None) -> dict:
    """
    Verify JWT token dan extract payload.

    Args:
        token: JWT token string
        audit_logger: AuditLogger untuk logging rejections (optional)

    Returns:
        Token payload dictionary

    Raises:
        HTTPException 401 jika token invalid atau expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")

        if user_id is None:
            if audit_logger:
                audit_logger.log_flagged_content(
                    session_id=None,
                    query="Invalid token: missing user_id",
                    violation_category="AUTH_FAILURE",
                    confidence=1.0,
                    reason="Token missing user_id claim"
                )
            raise credentials_exception

        logger.debug(f"Token verified for user: {user_id}")
        return payload

    except JWTError as e:
        if audit_logger:
            audit_logger.log_flagged_content(
                session_id=None,
                query=f"Token verification failed: {str(e)}",
                violation_category="AUTH_FAILURE",
                confidence=1.0,
                reason=str(e)
            )

        logger.warning(f"Token verification failed: {e}")
        raise credentials_exception

def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme)
) -> Optional[dict]:
    """
    FastAPI dependency untuk get current authenticated user.
    Jika token None → return None (anonymous user).

    Args:
        token: JWT token dari OAuth2 scheme

    Returns:
        User payload dict jika authenticated, None jika anonymous

    Raises:
        HTTPException 401 jika token provided tapi invalid
    """
    if token is None:
        # Anonymous user
        return None

    # Verify token (without audit logging in dependency)
    return verify_token(token, audit_logger=None)
