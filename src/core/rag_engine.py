"""
Islamic AI Assistant — RAG Engine

Retrieval-Augmented Generation engine untuk semantic search.
Menggunakan ChromaDB vector store dan sentence-transformers embedding.

Pipeline:
1. Select sources berdasarkan intent
2. Embed query dengan caching
3. Search vector store per source type
4. Deduplicate results
5. Re-rank berdasarkan relevance + authority
6. Return top-K chunks
"""

from typing import Optional
from src.models.schemas import IntentCategory, SourceType, KnowledgeChunk
from src.core.embedding_model import get_embedding_model
from src.core.vector_store import VectorStore, ScoredChunk
from src.utils.cache import get_cache_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VectorStoreUnavailableError(Exception):
    """Raised when vector store is unavailable after retries"""
    pass

class RAGEngine:
    """
    RAG engine untuk semantic search knowledge chunks.
    Orchestrates embedding, vector search, deduplication, dan re-ranking.
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG engine.

        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
        self.embedding_model = get_embedding_model()
        self.cache_manager = get_cache_manager()
        logger.info("RAG Engine initialized")

    def select_sources(self, intent: IntentCategory) -> list[SourceType]:
        """
        Select SourceType yang relevan berdasarkan IntentCategory.
        Mapping sesuai requirements 3.2.

        Args:
            intent: IntentCategory dari query

        Returns:
            List SourceType yang akan di-search
        """
        # Mapping intent → source types (sesuai requirements 3.2)
        intent_source_mapping = {
            IntentCategory.QURAN_TAFSIR: [
                SourceType.QURAN,
                SourceType.SCHOLAR_OPINION
            ],
            IntentCategory.HADITH_LOOKUP: [
                SourceType.HADITH,
                SourceType.SCHOLAR_OPINION
            ],
            IntentCategory.FIQH_QUESTION: [
                SourceType.FIQH,
                SourceType.FATWA,
                SourceType.SCHOLAR_OPINION
            ],
            IntentCategory.IBADAH_GUIDE: [
                SourceType.FIQH,
                SourceType.HADITH,
                SourceType.QURAN
            ],
            IntentCategory.AQEEDAH: [
                SourceType.QURAN,
                SourceType.HADITH,
                SourceType.SCHOLAR_OPINION
            ],
            IntentCategory.DAWA_CONTENT: [
                SourceType.QURAN,
                SourceType.HADITH,
                SourceType.SCHOLAR_OPINION
            ],
            IntentCategory.GENERAL_ISLAMIC: [
                SourceType.QURAN,
                SourceType.HADITH,
                SourceType.FIQH,
                SourceType.SCHOLAR_OPINION
            ]
        }

        sources = intent_source_mapping.get(intent, [])

        if not sources:
            # Default fallback: search semua source types
            logger.warning(
                f"No source mapping for intent {intent.value}, using all sources"
            )
            sources = list(SourceType)

        logger.debug(f"Selected sources for {intent.value}: {[s.value for s in sources]}")
        return sources

    def embed_query(self, text: str) -> list[float]:
        """
        Embed query text dengan caching (TTL 1 jam).

        Args:
            text: Query text

        Returns:
            Embedding vector (384 dimensi)
        """
        # Generate cache key
        cache_key = self.cache_manager.get_cache_key(text)

        # Get or compute embedding with cache
        embedding = self.cache_manager.get_or_compute(
            cache=self.cache_manager.embedding_cache,
            lock=self.cache_manager._embedding_lock,
            key=cache_key,
            compute_fn=lambda: self.embedding_model.embed(text)
        )

        return embedding

    # TO BE CONTINUED in Part 2...
    # (Akan dilanjutkan dengan deduplicate, rerank, dan retrieve methods untuk Task 8.2 dan 8.3)

    def deduplicate(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """
        Deduplikasi chunks berdasarkan source_ref.
        Pertahankan urutan relatif, simpan hanya kemunculan pertama.

        Args:
            chunks: List ScoredChunk dari search results

        Returns:
            List ScoredChunk tanpa duplikat (urutan dipertahankan)
        """
        seen_refs = set()
        deduped = []

        for scored_chunk in chunks:
            source_ref = scored_chunk.chunk.source_ref
            if source_ref not in seen_refs:
                seen_refs.add(source_ref)
                deduped.append(scored_chunk)

        logger.debug(
            f"Deduplicated {len(chunks)} chunks → {len(deduped)} unique chunks"
        )
        return deduped

    def rerank(
        self,
        chunks: list[ScoredChunk],
        query_text: str
    ) -> list[KnowledgeChunk]:
        """
        Re-rank chunks berdasarkan combined score.
        final_score = (relevance_score * 0.7) + (authority_score * 0.3)

        Args:
            chunks: List ScoredChunk dari search results
            query_text: Query text untuk logging/debugging

        Returns:
            List KnowledgeChunk yang sudah diurutkan (score tertinggi pertama)
        """
        # Calculate final scores
        ranked = []
        for scored_chunk in chunks:
            relevance_score = scored_chunk.score
            authority_score = scored_chunk.chunk.authority_score

            # Combined score dengan weighting
            final_score = (relevance_score * 0.7) + (authority_score * 0.3)

            ranked.append((final_score, scored_chunk.chunk))

        # Sort descending by final_score
        ranked.sort(key=lambda x: x[0], reverse=True)

        # Extract chunks only (discard scores)
        reranked_chunks = [chunk for _, chunk in ranked]

        logger.debug(f"Re-ranked {len(reranked_chunks)} chunks")
        return reranked_chunks

    def retrieve(
        self,
        query_text: str,
        intent: IntentCategory,
        top_k: int = 5,
        min_score: float = 0.6
    ) -> list[KnowledgeChunk]:
        """
        Main retrieval method: orchestrates full RAG pipeline.

        Pipeline:
        1. Select sources berdasarkan intent
        2. Embed query dengan caching
        3. Search vector store per source type
        4. Deduplicate results
        5. Re-rank berdasarkan relevance + authority
        6. Return top-K chunks

        Args:
            query_text: User query text
            intent: IntentCategory dari query
            top_k: Jumlah maksimal chunks yang dikembalikan
            min_score: Minimum similarity score threshold

        Returns:
            List KnowledgeChunk (maksimal top_k items)

        Raises:
            VectorStoreUnavailableError: Jika vector store tidak tersedia setelah retries
        """
        # Check cache first
        cache_key = self.cache_manager.get_cache_key(
            query_text, intent.value, top_k, min_score
        )

        cached_result = None
        with self.cache_manager._rag_lock:
            if cache_key in self.cache_manager.rag_cache:
                cached_result = self.cache_manager.rag_cache[cache_key]

        if cached_result is not None:
            logger.info(f"RAG cache HIT for query: {query_text[:50]}...")
            return cached_result

        logger.info(f"RAG retrieve started: intent={intent.value}, top_k={top_k}")

        # Step 1: Select sources
        source_types = self.select_sources(intent)

        # Step 2: Embed query
        query_vector = self.embed_query(query_text)

        # Step 3: Search vector store per source type with retry
        all_results: list[ScoredChunk] = []
        max_retries = 3
        retry_delay = 5.0 / max_retries  # Total 5 seconds across retries

        for source_type in source_types:
            for attempt in range(max_retries):
                try:
                    # Health check vector store
                    if not self.vector_store.health_check(max_retries=1, timeout=2.0):
                        raise VectorStoreUnavailableError(
                            "Vector store health check failed"
                        )

                    # Search this source type
                    results = self.vector_store.search(
                        query_vector=query_vector,
                        source_type=source_type,
                        limit=top_k * 2,  # Get more results for better re-ranking
                        min_score=min_score
                    )

                    all_results.extend(results)
                    logger.debug(
                        f"Retrieved {len(results)} chunks from {source_type.value}"
                    )
                    break  # Success, exit retry loop

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Vector store search failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        import time
                        time.sleep(retry_delay)
                    else:
                        # Final retry failed
                        logger.error(
                            f"Vector store unavailable after {max_retries} retries"
                        )
                        raise VectorStoreUnavailableError(
                            f"Vector store unavailable after {max_retries} retries"
                        ) from e

        if not all_results:
            logger.warning("No chunks found for query")
            return []

        # Step 4: Deduplicate
        deduped_results = self.deduplicate(all_results)

        # Step 5: Re-rank
        reranked_chunks = self.rerank(deduped_results, query_text)

        # Step 6: Return top-K (atau semua jika kurang dari top_k)
        final_chunks = reranked_chunks[:top_k]

        logger.info(
            f"RAG retrieve completed: returned {len(final_chunks)} chunks "
            f"(requested top_k={top_k})"
        )

        # Cache result
        with self.cache_manager._rag_lock:
            self.cache_manager.rag_cache[cache_key] = final_chunks

        return final_chunks

# Convenience function untuk dependency injection
def create_rag_engine(vector_store: VectorStore) -> RAGEngine:
    """
    Factory function untuk create RAG engine instance.

    Args:
        vector_store: VectorStore instance

    Returns:
        RAGEngine instance

    Example:
        >>> from src.core.vector_store import VectorStore
        >>> from src.core.rag_engine import create_rag_engine
        >>> vector_store = VectorStore(mode="persistent")
        >>> rag_engine = create_rag_engine(vector_store)
        >>> chunks = rag_engine.retrieve(
        ...     query_text="Bagaimana cara wudhu yang benar?",
        ...     intent=IntentCategory.IBADAH_GUIDE,
        ...     top_k=5
        ... )
    """
    return RAGEngine(vector_store)
