"""
Islamic AI Assistant — Prompt Builder

Membangun prompt untuk LLM dengan:
- System prompt Islamic-aware
- Conversation history (max 10 turns terakhir)
- Retrieved knowledge chunks dengan citations
- Token counting dan pruning untuk fit dalam context window
"""

from typing import Optional
from dataclasses import dataclass

from src.models.schemas import (
    KnowledgeChunk,
    ConversationTurn,
    IntentCategory,
    MadhabType
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PromptBuilderError(Exception):
    """Raised when prompt cannot be built (e.g., no chunks after pruning)"""
    pass

# ============================================================
# PROMPT TEMPLATES
# ============================================================

ISLAMIC_SYSTEM_PROMPT = """Anda adalah asisten AI Islam yang berpengetahuan luas dan mengikuti adab Islam dalam setiap respons.

**Prinsip Utama:**
1. Selalu berdasarkan Al-Qur'an, Hadits Sahih, dan pendapat ulama terpercaya
2. Jika ada perbedaan pendapat (ikhtilaf), sebutkan dengan adil dan objektif
3. Jangan memberikan fatwa pribadi - rujuk ke ulama atau lembaga fatwa resmi untuk kasus kompleks
4. Gunakan bahasa yang sopan, rendah hati, dan menghormati
5. Selalu sertakan referensi sumber (ayat, hadits, atau pendapat ulama)

**Mazhab Preferensi:** {MADHAB}

{ADAB_GUIDELINES}

**Penting:**
- Jika pertanyaan di luar kompetensi Anda, arahkan penanya untuk berkonsultasi dengan ulama
- Jika ada istilah Arab, berikan transliterasi dan terjemahan
- Prioritaskan jawaban yang membawa kebaikan dan menjauhkan dari keburukan
"""

ADAB_GUIDELINES_STANDARD = """**Adab Menjawab:**
- Awali dengan salam atau basmalah
- Gunakan nada yang hangat dan mendukung
- Akhiri dengan doa atau ucapan baik
"""

ADAB_GUIDELINES_FIQH = """**Adab Menjawab Fiqh:**
- Sebutkan dengan jelas: wajib, haram, makruh, mubah, atau sunnah
- Jelaskan dalil dan alasan hukum
- Sebutkan jika ada perbedaan pendapat antar mazhab
- Untuk kasus kompleks, sarankan konsultasi dengan ulama
"""

MAX_CONTEXT_TOKENS = 8192  # Context window limit
CHARS_PER_TOKEN = 4  # Estimasi: 4 karakter ≈ 1 token

@dataclass
class Prompt:
    """Struktur prompt yang sudah di-build"""
    system_prompt: str
    user_message: str
    conversation_history: list[ConversationTurn]
    total_tokens: int

class PromptBuilder:
    """
    Builder untuk construct prompt LLM dengan token management.
    """

    def __init__(self):
        """Initialize prompt builder"""
        logger.info("Prompt builder initialized")

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimasi jumlah token dari text.
        Menggunakan heuristic: 4 chars ≈ 1 token.

        Args:
            text: Text yang akan diestimasi

        Returns:
            Estimasi jumlah token
        """
        return max(1, len(text) // CHARS_PER_TOKEN)

    def format_chunk(self, chunk: KnowledgeChunk) -> str:
        """
        Format knowledge chunk dengan citation.

        Format:
        [Sumber: {source_ref}]
        {content}

        Args:
            chunk: KnowledgeChunk yang akan diformat

        Returns:
            Formatted string
        """
        formatted = f"[Sumber: {chunk.source_ref}]\n{chunk.text_translated}\n"

        # Tambahkan context jika ada (dalam metadata)
        if chunk.metadata and chunk.metadata.chapter:
            formatted += f"(Konteks: {chunk.metadata.chapter})\n"

        return formatted

    def _format_history(self, history: list[ConversationTurn]) -> str:
        """
        Format conversation history menjadi string.

        Args:
            history: List conversation turns

        Returns:
            Formatted history string
        """
        if not history:
            return ""

        formatted = "**Riwayat Percakapan:**\n\n"
        for turn in history:
            role_label = "User" if turn.role.value == "user" else "Assistant"
            formatted += f"{role_label}: {turn.message}\n\n"

        return formatted

    def build_prompt(
        self,
        query: str,
        history: list[ConversationTurn],
        chunks: list[KnowledgeChunk],
        intent: IntentCategory,
        madhab: MadhabType = MadhabType.GENERAL
    ) -> Prompt:
        """
        Build prompt lengkap dengan token management.

        Pipeline:
        1. Format system prompt dengan madhab
        2. Format chunks sebagai context
        3. Ambil 10 turns terakhir dari history
        4. Hitung total tokens
        5. Prune jika > MAX_CONTEXT_TOKENS:
           - Hapus history terlama (pertahankan min 2 turns terbaru)
           - Hapus chunks dengan score terendah (pertahankan min 1 chunk)
        6. Raise error jika chunks habis

        Args:
            query: User query
            history: Conversation history (akan di-prune ke 10 terakhir)
            chunks: Retrieved knowledge chunks
            intent: IntentCategory
            madhab: Mazhab preference

        Returns:
            Prompt object dengan total_tokens

        Raises:
            PromptBuilderError: Jika chunks kosong setelah pruning
        """
        if not chunks:
            raise PromptBuilderError("No knowledge chunks provided")

        # Step 1: Build system prompt
        adab_guidelines = (
            ADAB_GUIDELINES_FIQH if intent == IntentCategory.FIQH_QUESTION
            else ADAB_GUIDELINES_STANDARD
        )

        madhab_text = (
            f"Mazhab {madhab.value.title()}" if madhab != MadhabType.GENERAL
            else "Umum (menyebutkan perbedaan antar mazhab jika relevan)"
        )

        system_prompt = ISLAMIC_SYSTEM_PROMPT.format(
            MADHAB=madhab_text,
            ADAB_GUIDELINES=adab_guidelines
        )

        # Tambahkan instruksi konsultasi ulama untuk fiqh kompleks
        if intent == IntentCategory.FIQH_QUESTION:
            system_prompt += (
                "\n\n**Catatan Khusus:** Untuk pertanyaan fiqh yang kompleks atau "
                "sensitif, sarankan penanya untuk berkonsultasi langsung dengan "
                "ulama atau lembaga fatwa yang kredibel."
            )

        # Step 2: Ambil 10 turns terakhir dari history
        recent_history = history[-10:] if len(history) > 10 else history

        # Step 3: Format chunks
        chunks_text = "\n**Pengetahuan Relevan:**\n\n"
        for chunk in chunks:
            chunks_text += self.format_chunk(chunk) + "\n"

        # Step 4: Build user message
        history_text = self._format_history(recent_history)
        user_message = f"{history_text}\n{chunks_text}\n**Pertanyaan Saat Ini:**\n{query}"

        # Step 5: Calculate total tokens
        system_tokens = self._estimate_tokens(system_prompt)
        message_tokens = self._estimate_tokens(user_message)
        total_tokens = system_tokens + message_tokens

        logger.debug(
            f"Initial prompt tokens: system={system_tokens}, "
            f"message={message_tokens}, total={total_tokens}"
        )

        # Step 6: Prune if necessary
        if total_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(
                f"Prompt exceeds token limit ({total_tokens} > {MAX_CONTEXT_TOKENS}), "
                "starting pruning..."
            )

            # 6a. Prune history (pertahankan min 2 turns terbaru)
            while len(recent_history) > 2 and total_tokens > MAX_CONTEXT_TOKENS:
                # Remove oldest turn
                recent_history.pop(0)

                # Recalculate
                history_text = self._format_history(recent_history)
                user_message = f"{history_text}\n{chunks_text}\n**Pertanyaan Saat Ini:**\n{query}"
                message_tokens = self._estimate_tokens(user_message)
                total_tokens = system_tokens + message_tokens

                logger.debug(f"Pruned history, new total: {total_tokens} tokens")

            # 6b. Prune chunks if still over limit (pertahankan min 1 chunk)
            while len(chunks) > 1 and total_tokens > MAX_CONTEXT_TOKENS:
                # Remove chunk dengan authority score terendah
                chunks.sort(key=lambda c: c.authority_score)
                removed_chunk = chunks.pop(0)

                logger.debug(
                    f"Removed chunk with low authority: {removed_chunk.source_ref}"
                )

                # Recalculate
                chunks_text = "\n**Pengetahuan Relevan:**\n\n"
                for chunk in chunks:
                    chunks_text += self.format_chunk(chunk) + "\n"

                user_message = f"{history_text}\n{chunks_text}\n**Pertanyaan Saat Ini:**\n{query}"
                message_tokens = self._estimate_tokens(user_message)
                total_tokens = system_tokens + message_tokens

                logger.debug(f"Pruned chunks, new total: {total_tokens} tokens")

            # Check if we still have chunks
            if not chunks:
                raise PromptBuilderError(
                    "All chunks were pruned but token limit still exceeded"
                )

        logger.info(
            f"Prompt built successfully: {total_tokens} tokens, "
            f"{len(chunks)} chunks, {len(recent_history)} history turns"
        )

        return Prompt(
            system_prompt=system_prompt,
            user_message=user_message,
            conversation_history=recent_history,
            total_tokens=total_tokens
        )

# Convenience function
def create_prompt_builder() -> PromptBuilder:
    """
    Factory function untuk create PromptBuilder instance.

    Returns:
        PromptBuilder instance
    """
    return PromptBuilder()
