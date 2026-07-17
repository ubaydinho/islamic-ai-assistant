"""
Islamic AI Assistant — Response Validator

Validasi response dari LLM untuk deteksi halusinasi.
Menggunakan keyword matching sederhana untuk memeriksa apakah
klaim dalam response didukung oleh context chunks.
"""

import re
from typing import Optional
from dataclasses import dataclass

from src.models.schemas import KnowledgeChunk, ValidationResult
from src.utils.logger import get_logger
from src.core.embedding_model import get_embedding_model

logger = get_logger(__name__)

class ResponseValidator:
    """
    Validator untuk deteksi halusinasi dalam LLM response.
    Membandingkan klaim dalam response dengan context chunks.
    """

    def __init__(self):
        """Initialize response validator"""
        logger.info("Response validator initialized")

    def _extract_declarative_sentences(self, text: str) -> list[str]:
        """
        Extract kalimat deklaratif dari text.
        Kalimat yang berisi klaim faktual (bukan pertanyaan atau salam).

        Args:
            text: Response text

        Returns:
            List kalimat deklaratif
        """
        # Split by sentence boundaries
        sentences = re.split(r'[.!]\s+', text)

        # Filter: hanya kalimat deklaratif (tidak dimulai dengan kata tanya, tidak salam)
        declarative = []
        question_words = ['apa', 'mengapa', 'bagaimana', 'kapan', 'dimana', 'siapa', 'berapa']
        greeting_words = ['assalamu', 'walaikum', 'bismillah', 'alhamdulillah']

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Skip if starts with question word
            first_word = sentence.lower().split()[0] if sentence.split() else ""
            if first_word in question_words:
                continue

            # Skip if greeting
            if any(greet in sentence.lower() for greet in greeting_words):
                continue

            # Skip very short sentences (< 10 chars)
            if len(sentence) < 10:
                continue

            declarative.append(sentence)

        return declarative

    def _is_supported_by_chunks(
        self,
        sentence: str,
        chunks: list[KnowledgeChunk]
    ) -> bool:
        """
        Check apakah kalimat didukung oleh salah satu chunk.
        Menggunakan semantic similarity (embedding cosine similarity),
        bukan keyword matching, supaya paraphrase dari LLM tetap
        terdeteksi sebagai "didukung" selama maknanya sama dengan chunk.

        Args:
            sentence: Kalimat yang akan dicek
            chunks: Knowledge chunks sebagai context

        Returns:
            True jika ada chunk yang mendukung kalimat ini (similarity > threshold)
        """
        if not chunks:
            return False

        embedding_model = get_embedding_model()
        sentence_vec = embedding_model.embed(sentence)

        SIMILARITY_THRESHOLD = 0.5

        for chunk in chunks:
            chunk_vec = embedding_model.embed(chunk.text_translated)
            similarity = sum(a * b for a, b in zip(sentence_vec, chunk_vec))

            if similarity > SIMILARITY_THRESHOLD:
                logger.debug(
                    f"Sentence supported by chunk (similarity={similarity:.2f}): "
                    f"{sentence[:60]}..."
                )
                return True

        logger.debug(f"Sentence NOT supported by any chunk: {sentence[:60]}...")
        return False

    def detect_hallucination(
        self,
        response: str,
        context_chunks: list[KnowledgeChunk]
    ) -> float:
        """
        Deteksi halusinasi dalam response.

        Algoritma:
        1. Extract kalimat deklaratif dari response
        2. Check setiap kalimat apakah didukung oleh context chunks
        3. Hitung proporsi kalimat yang tidak didukung
        4. Jika context_chunks kosong → return 1.0 (100% halusinasi)

        Args:
            response: Response text dari LLM
            context_chunks: Knowledge chunks yang digunakan sebagai context

        Returns:
            Hallucination score (0.0 - 1.0)
            - 0.0 = tidak ada halusinasi (semua didukung context)
            - 1.0 = semua halusinasi (tidak ada yang didukung)
        """
        # Jika tidak ada context chunks, assume 100% hallucination
        if not context_chunks:
            logger.warning("No context chunks provided, assuming 100% hallucination")
            return 1.0

        # Extract declarative sentences
        sentences = self._extract_declarative_sentences(response)

        if not sentences:
            # Jika tidak ada kalimat deklaratif, assume safe (0.0)
            logger.debug("No declarative sentences found in response")
            return 0.0

        # Count unsupported sentences
        unsupported_count = 0
        for sentence in sentences:
            if not self._is_supported_by_chunks(sentence, context_chunks):
                unsupported_count += 1
                logger.debug(f"Unsupported sentence: {sentence[:80]}...")

        # Calculate hallucination score
        hallucination_score = unsupported_count / len(sentences)

        # Clamp to [0.0, 1.0]
        hallucination_score = max(0.0, min(1.0, hallucination_score))

        logger.info(
            f"Hallucination detection: {unsupported_count}/{len(sentences)} "
            f"unsupported sentences (score: {hallucination_score:.2f})"
        )

        return hallucination_score

    def validate(
        self,
        response: str,
        query: str,
        chunks: list[KnowledgeChunk]
    ) -> ValidationResult:
        """
        Validate response terhadap query dan chunks.

        Args:
            response: Response text dari LLM
            query: Original user query
            chunks: Knowledge chunks yang digunakan

        Returns:
            ValidationResult dengan confidence dan hallucination_score
        """
        # Detect hallucination
        hallucination_score = self.detect_hallucination(response, chunks)

        # Calculate confidence (inverse of hallucination)
        confidence = 1.0 - hallucination_score

        # Clamp confidence to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))

        result = ValidationResult(
            confidence=confidence,
            hallucination_score=hallucination_score,
            needs_disclaimer=hallucination_score > 0.5
        )

        logger.info(
            f"Validation result: confidence={confidence:.2f}, "
            f"hallucination={hallucination_score:.2f}"
        )

        return result

# Convenience function
def create_response_validator() -> ResponseValidator:
    """
    Factory function untuk create ResponseValidator instance.

    Returns:
        ResponseValidator instance

    Example:
        >>> from src.core.response_validator import create_response_validator
        >>> validator = create_response_validator()
        >>> result = validator.validate(
        ...     response="Shalat wajib dilakukan 5 kali sehari...",
        ...     query="Berapa kali shalat wajib?",
        ...     chunks=[chunk1, chunk2]
        ... )
        >>> result.is_valid
        True
    """
    return ResponseValidator()
