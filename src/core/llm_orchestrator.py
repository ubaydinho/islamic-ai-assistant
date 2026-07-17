"""
Islamic AI Assistant — LLM Orchestrator

Orchestrator untuk Groq API calls dengan:
- Model selection (llama-3.1-8b-instant, gemma2-9b-it)
- Token bucket rate limiting
- Retry logic dengan exponential backoff
- Streaming support
"""

import os
import time
from typing import Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timedelta

from groq import Groq, AsyncGroq
from groq.types.chat import ChatCompletion

from src.models.schemas import IntentCategory, GenerationConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LLMUnavailableError(Exception):
    """Raised when LLM is unavailable after retries on all models"""
    pass

class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded"""
    pass

@dataclass
class TokenBucket:
    """Token bucket untuk rate limiting"""
    capacity: int
    tokens: int
    last_refill: datetime
    refill_rate: float  # tokens per second

class RateLimiter:
    """
    Token bucket rate limiter untuk Groq API.

    Limits:
    - 30 requests/minute
    - 6000 tokens/minute
    - 14400 requests/day

    Thresholds (80% untuk delay):
    - 24 requests/minute
    - 4800 tokens/minute
    - 11520 requests/day
    """

    def __init__(self):
        """Initialize rate limiter dengan token buckets"""
        now = datetime.now()

        # Request rate: 30 req/min = 0.5 req/sec
        self.request_bucket = TokenBucket(
            capacity=30,
            tokens=30,
            last_refill=now,
            refill_rate=0.5
        )

        # Token rate: 6000 tokens/min = 100 tokens/sec
        self.token_bucket = TokenBucket(
            capacity=6000,
            tokens=6000,
            last_refill=now,
            refill_rate=100.0
        )

        # Daily request rate: 14400 req/day = 0.1667 req/sec
        self.daily_bucket = TokenBucket(
            capacity=14400,
            tokens=14400,
            last_refill=now,
            refill_rate=0.1667
        )

        logger.info("Rate limiter initialized")

    def _refill_bucket(self, bucket: TokenBucket):
        """Refill bucket berdasarkan elapsed time"""
        now = datetime.now()
        elapsed = (now - bucket.last_refill).total_seconds()

        # Calculate tokens to add
        tokens_to_add = elapsed * bucket.refill_rate
        bucket.tokens = min(bucket.capacity, bucket.tokens + tokens_to_add)
        bucket.last_refill = now

    def check_and_consume(
        self,
        estimated_tokens: int = 100
    ) -> tuple[bool, Optional[float]]:
        """
        Check rate limits dan consume tokens jika available.

        Args:
            estimated_tokens: Estimasi tokens untuk request ini

        Returns:
            (can_proceed, wait_time_seconds)
            - can_proceed: True jika bisa proceed, False jika harus wait
            - wait_time_seconds: Berapa detik harus wait (None jika can proceed)
        """
        # Refill buckets
        self._refill_bucket(self.request_bucket)
        self._refill_bucket(self.token_bucket)
        self._refill_bucket(self.daily_bucket)

        # Check 80% thresholds
        request_threshold = self.request_bucket.capacity * 0.8  # 24 req/min
        token_threshold = self.token_bucket.capacity * 0.8      # 4800 tokens/min
        daily_threshold = self.daily_bucket.capacity * 0.8      # 11520 req/day

        # Calculate wait times if needed
        wait_times = []

        if self.request_bucket.tokens < 1:
            wait_time = 1.0 / self.request_bucket.refill_rate
            wait_times.append(wait_time)
            logger.warning(f"Request rate limit reached, wait {wait_time:.2f}s")

        if self.token_bucket.tokens < estimated_tokens:
            wait_time = (estimated_tokens - self.token_bucket.tokens) / self.token_bucket.refill_rate
            wait_times.append(wait_time)
            logger.warning(f"Token rate limit reached, wait {wait_time:.2f}s")

        if self.daily_bucket.tokens < 1:
            wait_time = 1.0 / self.daily_bucket.refill_rate
            wait_times.append(wait_time)
            logger.warning(f"Daily rate limit reached, wait {wait_time:.2f}s")

        # If we need to wait
        if wait_times:
            return False, max(wait_times)

        # Check if we're near threshold (delay for safety)
        if (self.request_bucket.tokens < request_threshold or
            self.token_bucket.tokens < token_threshold or
            self.daily_bucket.tokens < daily_threshold):

            logger.info("Approaching rate limit threshold, adding safety delay")
            return False, 1.0  # 1 second safety delay

        # Consume tokens
        self.request_bucket.tokens -= 1
        self.token_bucket.tokens -= estimated_tokens
        self.daily_bucket.tokens -= 1

        logger.debug(
            f"Rate limit check passed. Remaining: "
            f"requests={self.request_bucket.tokens:.1f}, "
            f"tokens={self.token_bucket.tokens:.1f}, "
            f"daily={self.daily_bucket.tokens:.1f}"
        )

        return True, None

class LLMOrchestrator:
    """
    Orchestrator untuk Groq API calls dengan rate limiting dan retry logic.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM orchestrator.

        Args:
            api_key: Groq API key (default: dari env var GROQ_API_KEY)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)
        self.rate_limiter = RateLimiter()

        logger.info("LLM Orchestrator initialized")

    def select_model(
        self,
        intent: IntentCategory,
        complexity: str = "simple"
    ) -> str:
        """
        Select model berdasarkan intent dan complexity.

        Args:
            intent: IntentCategory dari query
            complexity: "simple" atau "complex"

        Returns:
            Model name (llama-3.1-8b-instant atau gemma2-9b-it)
        """
        # Default: llama-3.1-8b-instant (fast, efficient)
        default_model = "llama-3.1-8b-instant"

        # Gunakan gemma2-9b-it untuk kasus kompleks
        complex_model = "llama-3.3-70b-versatile"

        # Intent yang membutuhkan reasoning lebih dalam
        complex_intents = [
            IntentCategory.FIQH_QUESTION,
            IntentCategory.AQEEDAH,
            IntentCategory.QURAN_TAFSIR,
            IntentCategory.HADITH_LOOKUP,
        ]

        if complexity == "complex" or intent in complex_intents:
            logger.debug(f"Selected complex model: {complex_model}")
            return complex_model

        logger.debug(f"Selected default model: {default_model}")
        return default_model

    # TO BE CONTINUED in Part 2...
    # (generate, retry logic, streaming akan ditambahkan di Part 2)

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        config: Optional[GenerationConfig] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate response dari LLM dengan retry logic dan exponential backoff.

        Retry strategy:
        - HTTP 5xx atau timeout > 10s → retry dengan exponential backoff (1s, 2s, 4s)
        - HTTP 429 (rate limit) → tunggu sesuai Retry-After header
        - Max 3 retries pada primary model
        - Setelah 3 retries gagal → fallback ke secondary model (gemma2-9b-it)
        - Jika secondary juga gagal → raise LLMUnavailableError

        Args:
            system_prompt: System prompt
            user_message: User message
            config: GenerationConfig (optional)
            model: Model name override (optional)

        Returns:
            Generated response text

        Raises:
            LLMUnavailableError: Jika kedua model gagal setelah retries
        """
        if config is None:
            config = GenerationConfig()

        # Select model jika tidak di-override
        if model is None:
            model = "llama-3.1-8b-instant"

        primary_model = model
        secondary_model = "gemma2-9b-it" if model != "gemma2-9b-it" else "llama-3.1-8b-instant"

        # Try primary model first
        try:
            response = self._generate_with_retry(
                system_prompt=system_prompt,
                user_message=user_message,
                config=config,
                model=primary_model,
                max_retries=3
            )
            return response
        except Exception as e:
            logger.error(f"Primary model ({primary_model}) failed after retries: {e}")

            # Fallback ke secondary model
            logger.info(f"Falling back to secondary model: {secondary_model}")
            try:
                response = self._generate_with_retry(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    config=config,
                    model=secondary_model,
                    max_retries=3
                )
                return response
            except Exception as e2:
                logger.error(f"Secondary model ({secondary_model}) also failed: {e2}")
                raise LLMUnavailableError(
                    f"Both models failed: {primary_model}, {secondary_model}"
                ) from e2

    def _generate_with_retry(
        self,
        system_prompt: str,
        user_message: str,
        config: GenerationConfig,
        model: str,
        max_retries: int = 3
    ) -> str:
        """
        Internal method untuk generate dengan retry logic.

        Args:
            system_prompt: System prompt
            user_message: User message
            config: GenerationConfig
            model: Model name
            max_retries: Maximum retry attempts

        Returns:
            Generated response text

        Raises:
            Exception: Jika semua retries gagal
        """
        retry_delays = [1.0, 2.0, 4.0]  # Exponential backoff

        for attempt in range(max_retries):
            try:
                # Check rate limit
                can_proceed, wait_time = self.rate_limiter.check_and_consume(
                    estimated_tokens=config.max_tokens or 100
                )

                if not can_proceed and wait_time:
                    logger.info(f"Rate limit delay: waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                    # Recheck after waiting
                    can_proceed, wait_time = self.rate_limiter.check_and_consume(
                        estimated_tokens=config.max_tokens or 100
                    )

                # Make API call
                logger.info(f"Calling Groq API: model={model}, attempt={attempt + 1}")

                response: ChatCompletion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    timeout=10.0  # 10 second timeout
                )

                # Extract response text
                response_text = response.choices[0].message.content

                logger.info(
                    f"LLM response received: model={model}, "
                    f"tokens_used={response.usage.total_tokens if response.usage else 'unknown'}"
                )

                return response_text

            except Exception as e:
                error_type = type(e).__name__
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries}): "
                    f"{error_type} - {str(e)}"
                )

                # Check if we should retry
                should_retry = (
                    "5" in str(e) or  # HTTP 5xx errors
                    "timeout" in str(e).lower() or
                    "429" in str(e)  # Rate limit
                )

                if attempt < max_retries - 1 and should_retry:
                    delay = retry_delays[attempt]

                    # Check for Retry-After header (untuk 429)
                    if "429" in str(e):
                        # Extract retry-after if available (simplified parsing)
                        delay = 5.0  # Default 5 seconds untuk rate limit

                    logger.info(f"Retrying after {delay}s delay...")
                    time.sleep(delay)
                else:
                    # No more retries or non-retryable error
                    raise

        # Should not reach here, but just in case
        raise LLMUnavailableError(f"Max retries ({max_retries}) exceeded")

    async def stream_generate(
        self,
        system_prompt: str,
        user_message: str,
        config: Optional[GenerationConfig] = None,
        model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate response dengan streaming (async).
        Chunk pertama harus terkirim < 1 detik.

        Args:
            system_prompt: System prompt
            user_message: User message
            config: GenerationConfig (optional)
            model: Model name override (optional)

        Yields:
            String chunks dari LLM response

        Raises:
            LLMUnavailableError: Jika streaming gagal
        """
        if config is None:
            config = GenerationConfig()

        if model is None:
            model = "llama-3.1-8b-instant"

        try:
            # Check rate limit
            can_proceed, wait_time = self.rate_limiter.check_and_consume(
                estimated_tokens=config.max_tokens or 100
            )

            if not can_proceed and wait_time:
                logger.info(f"Rate limit delay before streaming: {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

            # Start streaming
            logger.info(f"Starting streaming: model={model}")
            start_time = time.time()
            first_chunk_received = False

            stream = await self.async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content

                    # Check first chunk timing
                    if not first_chunk_received:
                        first_chunk_time = time.time() - start_time
                        logger.info(f"First chunk received in {first_chunk_time:.3f}s")
                        first_chunk_received = True

                    yield content

            logger.info("Streaming completed successfully")

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise LLMUnavailableError(f"Streaming failed: {e}") from e

# Import asyncio for streaming
import asyncio

# Convenience function
def create_llm_orchestrator(api_key: Optional[str] = None) -> LLMOrchestrator:
    """
    Factory function untuk create LLMOrchestrator instance.

    Args:
        api_key: Groq API key (optional, dari env jika None)

    Returns:
        LLMOrchestrator instance

    Example:
        >>> from src.core.llm_orchestrator import create_llm_orchestrator
        >>> orchestrator = create_llm_orchestrator()
        >>> response = orchestrator.generate(
        ...     system_prompt="You are a helpful assistant",
        ...     user_message="What is Islam?"
        ... )
    """
    return LLMOrchestrator(api_key=api_key)
