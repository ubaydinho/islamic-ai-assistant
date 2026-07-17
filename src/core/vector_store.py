"""
Islamic AI Assistant — ChromaDB Vector Store Wrapper

Wrapper untuk ChromaDB vector database.
Menyimpan knowledge chunks dengan embedding untuk semantic search.
Support untuk multiple collections berdasarkan SourceType.
"""

import time
from typing import Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from src.models.schemas import KnowledgeChunk, SourceType, ChunkMetadata, MadhabType
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ScoredChunk:
    """Chunk dengan similarity score dari vector search"""
    chunk: KnowledgeChunk
    score: float  # Cosine similarity: 0.0 (tidak mirip) - 1.0 (identik)

class VectorStore:
    """
    ChromaDB wrapper untuk menyimpan dan mencari knowledge chunks.
    Setiap SourceType memiliki collection terpisah.
    """

    def __init__(
        self,
        mode: str = "persistent",
        persist_directory: str = "./data/chroma",
        host: str = "localhost",
        port: int = 8000
    ):
        """
        Initialize ChromaDB client.

        Args:
            mode: 'persistent' untuk development, 'http' untuk Docker deployment
            persist_directory: Directory untuk menyimpan data (mode persistent)
            host: ChromaDB server host (mode http)
            port: ChromaDB server port (mode http)
        """
        self.mode = mode

        if mode == "http":
            # Mode HTTP untuk production (ChromaDB sebagai service terpisah)
            logger.info(f"Initializing ChromaDB HTTP client: {host}:{port}")
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            # Mode persistent untuk development
            logger.info(f"Initializing ChromaDB persistent client: {persist_directory}")
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

        # Initialize collections untuk setiap SourceType
        self.collections = {}
        for source_type in SourceType:
            collection_name = f"islamic_knowledge_{source_type.value}"
            self.collections[source_type] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"source_type": source_type.value}
            )
            logger.debug(f"Collection ready: {collection_name}")

        logger.info(f"VectorStore initialized with {len(self.collections)} collections")

    def add_chunk(self, chunk: KnowledgeChunk, embedding: list[float]):
        """
        Tambahkan atau update knowledge chunk ke vector store.
        Upsert berdasarkan source_ref (update jika sudah ada).

        Args:
            chunk: KnowledgeChunk yang akan disimpan
            embedding: Embedding vector (384 dimensi)
        """
        try:
            collection = self.collections[chunk.source_type]

            # Prepare metadata
            metadata = {
                "source_ref": chunk.source_ref,
                "source_type": chunk.source_type.value,
                "language": "id",  # Default to Indonesian - can be extended later
                "authority_score": chunk.authority_score,
                "book_name": chunk.metadata.book_name,
                "author": chunk.metadata.author,
                "chapter": chunk.metadata.chapter,
                "verse_or_number": chunk.metadata.verse_or_number,
                "madhab": chunk.metadata.madhab.value
            }

            # Add optional metadata fields
            if chunk.text_arabic:
                metadata["text_arabic"] = chunk.text_arabic
            
            # Add topic tags as comma-separated string
            if chunk.metadata.topic_tags:
                metadata["topic_tags"] = ",".join(chunk.metadata.topic_tags)

            # Upsert ke collection (menggunakan source_ref sebagai ID unik)
            collection.upsert(
                ids=[chunk.source_ref],
                embeddings=[embedding],
                documents=[chunk.text_translated],
                metadatas=[metadata]
            )

            logger.debug(f"Added chunk: {chunk.source_ref}")

        except Exception as e:
            logger.error(f"Failed to add chunk {chunk.source_ref}: {e}")
            raise

    def add_chunks_batch(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: list[list[float]]
    ):
        """
        Tambahkan multiple chunks sekaligus (batch operation untuk efisiensi).

        Args:
            chunks: List KnowledgeChunk yang akan disimpan
            embeddings: List embedding vectors (masing-masing 384 dimensi)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) != embeddings count ({len(embeddings)})"
            )

        # Group chunks by source_type
        chunks_by_type: dict[SourceType, list[tuple[KnowledgeChunk, list[float]]]] = {}

        for chunk, embedding in zip(chunks, embeddings):
            if chunk.source_type not in chunks_by_type:
                chunks_by_type[chunk.source_type] = []
            chunks_by_type[chunk.source_type].append((chunk, embedding))

        # Batch insert per collection
        for source_type, chunk_emb_pairs in chunks_by_type.items():
            try:
                collection = self.collections[source_type]

                ids = [chunk.source_ref for chunk, _ in chunk_emb_pairs]
                documents = [chunk.text_translated for chunk, _ in chunk_emb_pairs]
                embeddings_list = [emb for _, emb in chunk_emb_pairs]
                metadatas = []
                for chunk, _ in chunk_emb_pairs:
                    meta = {
                        "source_ref": chunk.source_ref,
                        "source_type": chunk.source_type.value,
                        "language": "id",
                        "authority_score": chunk.authority_score,
                        "book_name": chunk.metadata.book_name,
                        "author": chunk.metadata.author,
                        "chapter": chunk.metadata.chapter,
                        "verse_or_number": chunk.metadata.verse_or_number,
                        "madhab": chunk.metadata.madhab.value
                    }
                    if chunk.text_arabic:
                        meta["text_arabic"] = chunk.text_arabic
                    if chunk.metadata.topic_tags:
                        meta["topic_tags"] = ",".join(chunk.metadata.topic_tags)
                    metadatas.append(meta)

                collection.upsert(
                    ids=ids,
                    embeddings=embeddings_list,
                    documents=documents,
                    metadatas=metadatas
                )

                logger.info(
                    f"Batch added {len(chunk_emb_pairs)} chunks to "
                    f"{source_type.value} collection"
                )

            except Exception as e:
                logger.error(f"Failed to batch add chunks to {source_type.value}: {e}")
                raise

    def search(
        self,
        query_vector: list[float],
        source_type: Optional[SourceType] = None,
        limit: int = 5,
        min_score: float = 0.5
    ) -> list[ScoredChunk]:
        """
        Semantic search berdasarkan query vector.

        Args:
            query_vector: Embedding vector dari query (384 dimensi)
            source_type: Filter berdasarkan SourceType tertentu (None = search semua)
            limit: Jumlah maksimal hasil
            min_score: Minimum similarity score threshold

        Returns:
            List ScoredChunk yang diurutkan berdasarkan similarity (tertinggi pertama)
        """
        results = []

        # Tentukan collections yang akan di-search
        collections_to_search = (
            [self.collections[source_type]] if source_type
            else list(self.collections.values())
        )

        for collection in collections_to_search:
            try:
                # Query ChromaDB
                search_results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"]
                )

                # Parse results
                if search_results["ids"] and search_results["ids"][0]:
                    for i, doc_id in enumerate(search_results["ids"][0]):
                        # With normalized embeddings, ChromaDB L2 distance relates to
                        # cosine similarity as: cos_sim = 1 - (dist^2 / 2)
                        # This gives a proper [0, 1] similarity score
                        distance = search_results["distances"][0][i]
                        similarity_score = max(0.0, 1.0 - (distance ** 2) / 2.0)

                        # Log all scores for debugging
                        logger.debug(f"Found chunk: id={doc_id}, distance={distance:.3f}, score={similarity_score:.3f}")

                        # Filter by minimum score
                        if similarity_score < min_score:
                            logger.debug(f"Filtered out chunk {doc_id}: score {similarity_score:.3f} < min_score {min_score}")
                            continue

                        metadata = search_results["metadatas"][0][i]
                        document = search_results["documents"][0][i]

                        # Reconstruct ChunkMetadata and KnowledgeChunk
                        chunk_metadata = ChunkMetadata(
                            book_name=metadata.get("book_name", "Unknown"),
                            author=metadata.get("author", "Unknown"),
                            chapter=metadata.get("chapter", ""),
                            verse_or_number=metadata.get("verse_or_number", ""),
                            madhab=MadhabType(metadata.get("madhab", "general")),
                            topic_tags=metadata.get("topic_tags", "").split(",") if metadata.get("topic_tags") else []
                        )
                        
                        chunk = KnowledgeChunk(
                            source_ref=metadata["source_ref"],
                            source_type=SourceType(metadata["source_type"]),
                            text_arabic=metadata.get("text_arabic"),
                            text_translated=document,
                            metadata=chunk_metadata,
                            authority_score=metadata.get("authority_score", 0.8)
                        )

                        results.append(ScoredChunk(chunk=chunk, score=similarity_score))

            except Exception as e:
                logger.error(f"Search failed in collection: {e}")
                continue

        # Sort by score (descending) dan limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def health_check(self, max_retries: int = 3, timeout: float = 5.0) -> bool:
        """
        Check apakah ChromaDB service berjalan dengan baik.

        Args:
            max_retries: Jumlah maksimal percobaan
            timeout: Timeout per percobaan (seconds)

        Returns:
            True jika healthy, False jika tidak
        """
        for attempt in range(max_retries):
            try:
                # Try to heartbeat ChromaDB
                _ = self.client.heartbeat()
                logger.info("ChromaDB health check passed")
                return True
            except Exception as e:
                logger.warning(
                    f"ChromaDB health check failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(timeout / max_retries)

        logger.error("ChromaDB health check failed after all retries")
        return False

    def get_collection_stats(self) -> dict[str, int]:
        """
        Get statistics untuk semua collections.

        Returns:
            Dictionary dengan count per SourceType
        """
        stats = {}
        for source_type, collection in self.collections.items():
            try:
                count = collection.count()
                stats[source_type.value] = count
            except Exception as e:
                logger.error(f"Failed to get stats for {source_type.value}: {e}")
                stats[source_type.value] = -1

        return stats
