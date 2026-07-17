"""
Islamic AI Assistant — Embedding Model Wrapper

Wrapper untuk sentence-transformers embedding model.
Menggunakan paraphrase-multilingual-MiniLM-L12-v2 (384 dimensi).
Singleton pattern untuk menghindari load model berulang kali.
"""

from typing import Optional
from sentence_transformers import SentenceTransformer

from src.utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingModel:
    """
    Singleton wrapper untuk SentenceTransformer embedding model.
    Menggunakan paraphrase-multilingual-MiniLM-L12-v2 (384 dimensi, CPU-friendly).
    """

    _instance: Optional['EmbeddingModel'] = None
    _model: Optional[SentenceTransformer] = None

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM = 384
    MAX_SEQ_LENGTH = 128  # Maksimal panjang token untuk model ini

    def __new__(cls):
        """Singleton pattern: hanya satu instance yang dibuat"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize embedding model jika belum di-load"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.MODEL_NAME}")
            self._model = SentenceTransformer(self.MODEL_NAME)
            logger.info(
                f"Embedding model loaded successfully. "
                f"Dimension: {self.EMBEDDING_DIM}, "
                f"Max sequence length: {self.MAX_SEQ_LENGTH}"
            )

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding vector dari text input.

        Args:
            text: Text yang akan di-embed (Arab, Indonesia, atau Inggris)

        Returns:
            List float dengan panjang tepat 384 elemen

        Raises:
            AssertionError: Jika output vector bukan 384 dimensi

        Note:
            - Jika text terlalu panjang, akan di-truncate otomatis dengan warning log
            - Output selalu 384 dimensi (validated)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding, returning zero vector")
            return [0.0] * self.EMBEDDING_DIM

        # Truncate jika terlalu panjang (soft limit)
        if len(text) > 512:
            logger.warning(
                f"Text too long ({len(text)} chars), truncating to 512 chars"
            )
            text = text[:512]

        try:
            # Generate embedding
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize to unit vector for cosine similarity
            )

            # Convert numpy array ke list
            vector = embedding.tolist()

            # Validasi dimensi output
            assert len(vector) == self.EMBEDDING_DIM, (
                f"Expected {self.EMBEDDING_DIM} dimensions, got {len(vector)}"
            )

            logger.debug(f"Generated embedding for text (length: {len(text)})")
            return vector

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector sebagai fallback
            return [0.0] * self.EMBEDDING_DIM

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors untuk batch texts (lebih efisien).

        Args:
            texts: List text yang akan di-embed

        Returns:
            List of embedding vectors, masing-masing 384 dimensi
        """
        if not texts:
            return []

        try:
            # Truncate texts yang terlalu panjang
            processed_texts = []
            for text in texts:
                if len(text) > 512:
                    logger.warning(f"Text too long, truncating")
                    processed_texts.append(text[:512])
                else:
                    processed_texts.append(text)

            # Generate embeddings batch
            embeddings = self._model.encode(
                processed_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True  # Normalize to unit vector for cosine similarity
            )

            # Convert ke list of lists
            vectors = [emb.tolist() for emb in embeddings]

            # Validasi semua dimensi
            for i, vector in enumerate(vectors):
                assert len(vector) == self.EMBEDDING_DIM, (
                    f"Vector {i}: Expected {self.EMBEDDING_DIM} dimensions, "
                    f"got {len(vector)}"
                )

            logger.debug(f"Generated {len(vectors)} embeddings in batch")
            return vectors

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            # Return zero vectors sebagai fallback
            return [[0.0] * self.EMBEDDING_DIM for _ in texts]

    @property
    def dimension(self) -> int:
        """Return dimensi embedding model"""
        return self.EMBEDDING_DIM

    @property
    def model_name(self) -> str:
        """Return nama model yang digunakan"""
        return self.MODEL_NAME

# Convenience function untuk akses langsung
def get_embedding_model() -> EmbeddingModel:
    """
    Get singleton instance dari EmbeddingModel.

    Returns:
        EmbeddingModel instance

    Example:
        >>> from src.core.embedding_model import get_embedding_model
        >>> model = get_embedding_model()
        >>> vector = model.embed("Bismillahirrahmanirrahim")
        >>> len(vector)
        384
    """
    return EmbeddingModel()
