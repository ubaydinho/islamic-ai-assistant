"""
Islamic AI Assistant — Knowledge Ingestion Script

Script untuk load knowledge chunks dari JSON/CSV files
dan simpan ke ChromaDB vector store.

Usage:
    python data/scripts/ingest.py --source quran --file data/knowledge/quran_sample.json
    python data/scripts/ingest.py --source hadith --file data/knowledge/hadith_sample.json
    python data/scripts/ingest.py --all  # Ingest all sources
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.schemas import KnowledgeChunk, SourceType, MadhabType, ChunkMetadata
from src.core.vector_store import VectorStore
from src.core.embedding_model import get_embedding_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

class KnowledgeIngester:
    """
    Ingester untuk load knowledge chunks dan simpan ke vector store.
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize ingester.

        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
        self.embedding_model = get_embedding_model()
        logger.info("Knowledge ingester initialized")

    def load_json_file(self, file_path: str) -> List[Dict]:
        """
        Load JSON file dan return list of chunks.

        Args:
            file_path: Path ke JSON file

        Returns:
            List of chunk dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"Loaded {len(data)} chunks from {file_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []

    def ingest_chunks(
        self,
        chunks_data: List[Dict],
        source_type: SourceType
    ) -> tuple[int, int]:
        """
        Ingest chunks ke vector store.

        Args:
            chunks_data: List of chunk dictionaries
            source_type: SourceType untuk chunks ini

        Returns:
            Tuple (success_count, error_count)
        """
        success_count = 0
        error_count = 0

        for i, chunk_data in enumerate(chunks_data):
            try:
                # Parse Arabic and translated text from content
                content = chunk_data["content"]
                lines = content.split('\n', 1)
                text_arabic = lines[0] if len(lines) > 1 else None
                text_translated = lines[1] if len(lines) > 1 else content
                
                # Create metadata
                metadata = ChunkMetadata(
                    book_name=chunk_data.get("book_name", "Unknown"),
                    author=chunk_data.get("author", "Unknown"),
                    chapter=chunk_data.get("chapter", chunk_data.get("context", "")),
                    verse_or_number=chunk_data["source_ref"].split(':')[-1],
                    madhab=MadhabType(chunk_data.get("madhab", "general")),
                    topic_tags=chunk_data.get("topic_tags", [])
                )
                
                # Create KnowledgeChunk dari dict
                chunk = KnowledgeChunk(
                    source_ref=chunk_data["source_ref"],
                    source_type=source_type,
                    text_arabic=text_arabic,
                    text_translated=text_translated,
                    metadata=metadata,
                    authority_score=chunk_data.get("authority_score", 0.8)
                )

                # Generate embedding
                embedding = self.embedding_model.embed(chunk.text_translated)

                # Add to vector store (upsert)
                self.vector_store.add_chunk(chunk, embedding)

                success_count += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(chunks_data)} chunks...")

            except Exception as e:
                logger.error(f"Failed to ingest chunk {i}: {e}")
                error_count += 1
                continue

        return success_count, error_count

    def ingest_file(self, file_path: str, source_type: SourceType) -> tuple[int, int]:
        """
        Ingest single file.

        Args:
            file_path: Path ke file
            source_type: SourceType

        Returns:
            Tuple (success_count, error_count)
        """
        logger.info(f"Ingesting {source_type.value} from {file_path}...")

        # Load chunks
        chunks_data = self.load_json_file(file_path)

        if not chunks_data:
            logger.warning(f"No chunks found in {file_path}")
            return 0, 0

        # Ingest
        success, error = self.ingest_chunks(chunks_data, source_type)

        logger.info(
            f"Ingestion completed for {source_type.value}: "
            f"{success} success, {error} errors"
        )

        return success, error

    def ingest_all(self, data_dir: str = "data/knowledge"):
        """
        Ingest all knowledge files.

        Args:
            data_dir: Directory berisi knowledge files
        """
        data_path = Path(data_dir)

        if not data_path.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return

        # Mapping file → source type
        file_mapping = {
            "quran_sample.json": SourceType.QURAN,
            "hadith_sample.json": SourceType.HADITH,
            "fiqh_sample.json": SourceType.FIQH
        }

        total_success = 0
        total_error = 0

        for filename, source_type in file_mapping.items():
            file_path = data_path / filename

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            success, error = self.ingest_file(str(file_path), source_type)
            total_success += success
            total_error += error

        logger.info(
            f"All ingestion completed: "
            f"total success={total_success}, total errors={total_error}"
        )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Ingest knowledge chunks to vector store")
    parser.add_argument("--source", choices=["quran", "hadith", "fiqh"], help="Source type")
    parser.add_argument("--file", help="Path to JSON file")
    parser.add_argument("--all", action="store_true", help="Ingest all sources")

    args = parser.parse_args()

    # Initialize vector store
    logger.info("Initializing vector store...")
    # Use http mode to connect to ChromaDB service
    vector_store = VectorStore(mode="http", host="chromadb", port=8000)

    # Initialize ingester
    ingester = KnowledgeIngester(vector_store)

    # Ingest
    if args.all:
        ingester.ingest_all()
    elif args.source and args.file:
        source_type = SourceType(args.source.upper())
        ingester.ingest_file(args.file, source_type)
    else:
        parser.print_help()
        sys.exit(1)

    logger.info("Ingestion script completed")

if __name__ == "__main__":
    main()
