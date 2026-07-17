"""
Islamic AI Assistant — Citation Generator

Generate citations dari knowledge chunks untuk transparency.
Setiap citation berisi source reference, text excerpt, dan authority score.
"""

from typing import Optional
from src.models.schemas import KnowledgeChunk, Citation
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CitationGenerator:
    """
    Generator untuk create citations dari knowledge chunks.
    Citations diurutkan berdasarkan authority score (descending).
    """

    def __init__(self):
        """Initialize citation generator"""
        logger.info("Citation generator initialized")

    def _generate_url(self, chunk: KnowledgeChunk) -> Optional[str]:
        """
        Generate URL untuk source reference (jika applicable).

        Args:
            chunk: KnowledgeChunk

        Returns:
            URL string atau None jika tidak ada URL
        """
        # Untuk Al-Qur'an: link ke quran.com
        if chunk.source_type.value == "quran":
            # Parse source_ref format: "surah:ayat" (e.g., "2:255")
            try:
                parts = chunk.source_ref.split(":")
                if len(parts) == 2:
                    surah, ayat = parts
                    return f"https://quran.com/{surah}/{ayat}"
            except:
                pass

        # Untuk Hadith: link ke sunnah.com
        elif chunk.source_type.value == "hadith":
            # Source ref format: "bukhari:1234" atau "muslim:5678"
            try:
                parts = chunk.source_ref.split(":")
                if len(parts) == 2:
                    collection, number = parts
                    if collection.lower() == "bukhari":
                        return f"https://sunnah.com/bukhari:{number}"
                    elif collection.lower() == "muslim":
                        return f"https://sunnah.com/muslim:{number}"
            except:
                pass

        # Default: tidak ada URL
        return None

    def _create_text_excerpt(self, content: str, max_length: int = 150) -> str:
        """
        Create text excerpt dari content (150 karakter pertama).

        Args:
            content: Full content text
            max_length: Maximum length untuk excerpt

        Returns:
            Text excerpt dengan ellipsis jika terpotong
        """
        if len(content) <= max_length:
            return content

        # Potong di max_length dan tambahkan ellipsis
        excerpt = content[:max_length].rsplit(' ', 1)[0]  # Potong di kata terakhir
        return excerpt + "..."

    def generate(self, chunks: list[KnowledgeChunk]) -> list[Citation]:
        """
        Generate citations dari knowledge chunks.

        Pipeline:
        1. Map setiap chunk ke Citation
        2. Extract text excerpt (150 chars pertama)
        3. Generate URL jika applicable
        4. Sort berdasarkan authority_score descending

        Args:
            chunks: List KnowledgeChunk yang digunakan dalam response

        Returns:
            List Citation yang sudah diurutkan berdasarkan authority score
        """
        if not chunks:
            logger.warning("No chunks provided for citation generation")
            return []

        citations = []

        for chunk in chunks:
            # Create text excerpt
            text_excerpt = self._create_text_excerpt(chunk.text_translated, max_length=150)

            # Generate URL
            url = self._generate_url(chunk)

            # Create Citation object
            citation = Citation(
                source_ref=chunk.source_ref,
                text_excerpt=text_excerpt,
                url=url,
                authority_score=chunk.authority_score
            )

            citations.append(citation)
            logger.debug(f"Generated citation: {citation.source_ref}")

        # Sort by authority_score descending
        citations.sort(key=lambda c: c.authority_score, reverse=True)

        logger.info(f"Generated {len(citations)} citations")
        return citations

# Convenience function
def create_citation_generator() -> CitationGenerator:
    """
    Factory function untuk create CitationGenerator instance.

    Returns:
        CitationGenerator instance

    Example:
        >>> from src.core.citation_generator import create_citation_generator
        >>> generator = create_citation_generator()
        >>> citations = generator.generate(chunks)
        >>> for citation in citations:
        ...     print(f"{citation.source_ref}: {citation.text_excerpt}")
    """
    return CitationGenerator()
