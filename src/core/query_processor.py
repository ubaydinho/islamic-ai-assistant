"""
Islamic AI Assistant — Query Processor

Main orchestrator untuk pipeline query processing:
1. Filter query (Islamic content filter)
2. Classify intent
3. Retrieve knowledge chunks (RAG)
4. Build prompt
5. Generate LLM response
6. Validate response
7. Filter response
8. Generate citations
9. Build final response
"""

import time
from typing import Optional
from uuid import UUID

from src.models.schemas import (
    FinalResponse,
    IntentCategory,
    SessionContext,
    ConversationTurn,
    GenerationConfig,
    MadhabType,
    RoleType
)
from src.core.islamic_filter import get_islamic_filter
from src.core.rag_engine import RAGEngine
from src.core.prompt_builder import create_prompt_builder, PromptBuilderError
from src.core.llm_orchestrator import create_llm_orchestrator, LLMUnavailableError
from src.core.response_validator import create_response_validator
from src.core.citation_generator import create_citation_generator
from src.core.session_manager import SessionManager
from src.core.audit_logger import AuditLogger
from src.utils.logger import get_logger

logger = get_logger(__name__)

class QueryProcessor:
    """
    Main orchestrator untuk query processing pipeline.
    Menggabungkan semua core components untuk memproses user query.
    """

    def __init__(
        self,
        rag_engine: RAGEngine,
        session_manager: SessionManager,
        audit_logger: AuditLogger
    ):
        """
        Initialize query processor.

        Args:
            rag_engine: RAGEngine instance
            session_manager: SessionManager instance
            audit_logger: AuditLogger instance
        """
        self.rag_engine = rag_engine
        self.session_manager = session_manager
        self.audit_logger = audit_logger

        # Initialize other components
        self.islamic_filter = get_islamic_filter()
        self.prompt_builder = create_prompt_builder()
        self.llm_orchestrator = create_llm_orchestrator()
        self.response_validator = create_response_validator()
        self.citation_generator = create_citation_generator()

        logger.info("Query processor initialized")

    def build_blocked_response(self, filter_result, session_id: Optional[UUID] = None) -> FinalResponse:
        """
        Build response untuk query yang di-block oleh filter.

        Args:
            filter_result: FilterResult dari islamic_filter
            session_id: Session UUID (diperlukan untuk FinalResponse)

        Returns:
            FinalResponse dengan pesan blocked
        """
        from uuid import uuid4

        # Construct blocked message berdasarkan violations
        blocked_message = (
            "Maaf, pertanyaan Anda tidak dapat dijawab karena mengandung "
            "konten yang tidak sesuai dengan ajaran Islam.\n\n"
        )

        if filter_result.violations:
            blocked_message += "Alasan:\n"
            for violation in filter_result.violations[:3]:  # Max 3 violations
                blocked_message += f"- {violation.reason}\n"

        blocked_message += (
            "\nSilakan ajukan pertanyaan yang sesuai dengan prinsip-prinsip Islam. "
            "Jika Anda memiliki pertanyaan tentang hukum Islam, "
            "kami sarankan berkonsultasi dengan ulama yang kredibel."
        )

        return FinalResponse(
            answer=blocked_message,
            citations=[],
            confidence=0.0,
            intent=IntentCategory.OUT_OF_SCOPE,
            is_blocked=True,
            block_reason="Query mengandung konten yang tidak sesuai ajaran Islam",
            session_id=session_id or uuid4(),
            response_time_ms=0
        )


    def process(
        self,
        query: str,
        session_id: UUID,
        config: Optional[GenerationConfig] = None
    ) -> FinalResponse:
        """
        Main method untuk process query melalui full pipeline.

        Pipeline:
        1. Filter query (Islamic content filter)
        2. Classify intent
        3. Retrieve knowledge chunks (RAG)
        4. Build prompt dengan session history
        5. Generate LLM response
        6. Validate response
        7. Filter response
        8. Generate citations
        9. Update session
        10. Log audit
        11. Return final response

        Args:
            query: User query text
            session_id: Session UUID
            config: GenerationConfig (optional)

        Returns:
            FinalResponse dengan answer, citations, dan metadata
        """
        start_time = time.time()
        logger.info(f"Processing query for session {session_id}: {query[:80]}...")

        if config is None:
            config = GenerationConfig()

        # Load session context
        session = self.session_manager.get_session(session_id)
        if not session:
            logger.error(f"Session not found: {session_id}")
            raise ValueError(f"Session {session_id} not found")

        # Step 1: Filter query
        logger.debug("Step 1: Filtering query...")
        filter_result = self.islamic_filter.filter_query(query)

        if not filter_result.is_halal:
            logger.warning(f"Query blocked by filter: {filter_result.suggested_redirect}")

            # Log flagged content
            if filter_result.violations:
                for violation in filter_result.violations:
                    self.audit_logger.log_flagged_content(
                        session_id=session_id,
                        query=query,
                        violation_category=violation.category.value,
                        confidence=violation.confidence,
                        reason=violation.reason
                    )

            return self.build_blocked_response(filter_result, session_id=session_id)

        # Step 2: Intent already classified in filter_result
        intent = filter_result.intent
        logger.info(f"Step 2: Intent classified: {intent.value}")

        # Step 3: Retrieve knowledge chunks
        logger.debug("Step 3: Retrieving knowledge chunks...")
        try:
            chunks = self.rag_engine.retrieve(
                query_text=query,
                intent=intent,
                top_k=5,
                min_score=0.2
            )

            if not chunks:
                logger.warning("No knowledge chunks retrieved")
                # Return fallback response
                return FinalResponse(
                    answer=(
                        "Maaf, saya tidak menemukan informasi yang relevan untuk "
                        "menjawab pertanyaan Anda. Silakan ajukan pertanyaan dengan "
                        "cara yang berbeda atau berkonsultasi dengan ulama."
                    ),
                    citations=[],
                    confidence=0.0,
                    intent=intent,
                    session_id=session_id,
                    response_time_ms=int((time.time() - start_time) * 1000)
                )
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            # Return error response
            return FinalResponse(
                answer=(
                    "Maaf, terjadi kesalahan saat mencari informasi. "
                    "Silakan coba lagi dalam beberapa saat."
                ),
                citations=[],
                confidence=0.0,
                intent=intent,
                session_id=session_id,
                response_time_ms=int((time.time() - start_time) * 1000)
            )

        logger.info(f"Retrieved {len(chunks)} knowledge chunks")

        # Step 4: Build prompt
        logger.debug("Step 4: Building prompt...")
        try:
            prompt = self.prompt_builder.build_prompt(
                query=query,
                history=session.history,
                chunks=chunks,
                intent=intent,
                madhab=MadhabType(session.madhab_preference)
            )
        except PromptBuilderError as e:
            logger.error(f"Prompt building failed: {e}")
            return FinalResponse(
                answer=(
                    "Maaf, terjadi kesalahan saat menyiapkan jawaban. "
                    "Silakan coba lagi."
                ),
                citations=[],
                confidence=0.0,
                intent=intent,
                session_id=session_id,
                response_time_ms=int((time.time() - start_time) * 1000)
            )

        logger.info(f"Prompt built: {prompt.total_tokens} tokens")

        # Step 5: Generate LLM response
        logger.debug("Step 5: Generating LLM response...")
        try:
            model = self.llm_orchestrator.select_model(intent, complexity="simple")
            response_text = self.llm_orchestrator.generate(
                system_prompt=prompt.system_prompt,
                user_message=prompt.user_message,
                config=config,
                model=model
            )
        except LLMUnavailableError as e:
            logger.error(f"LLM generation failed: {e}")
            return FinalResponse(
                answer=(
                    "Maaf, layanan AI sedang tidak tersedia. "
                    "Silakan coba lagi dalam beberapa saat."
                ),
                citations=[],
                confidence=0.0,
                intent=intent,
                session_id=session_id,
                response_time_ms=int((time.time() - start_time) * 1000)
            )

        logger.info("LLM response generated successfully")

        # Step 6: Validate response
        logger.debug("Step 6: Validating response...")
        validation_result = self.response_validator.validate(
            response=response_text,
            query=query,
            chunks=chunks
        )

        logger.info(
            f"Validation: confidence={validation_result.confidence:.2f}, "
            f"hallucination={validation_result.hallucination_score:.2f}"
        )

        # Step 7: Filter response
        logger.debug("Step 7: Filtering response...")
        response_filter_result = self.islamic_filter.filter_response(response_text)

        if not response_filter_result.is_halal:
            logger.warning(f"Response blocked by filter: {response_filter_result.suggested_redirect}")

            # Log flagged response
            if response_filter_result.violations:
                for violation in response_filter_result.violations:
                    self.audit_logger.log_flagged_content(
                        session_id=session_id,
                        query=f"[RESPONSE] {response_text[:100]}",
                        violation_category=violation.category.value,
                        confidence=violation.confidence,
                        reason=violation.reason
                    )

            # Return safe fallback
            return FinalResponse(
                answer=(
                    "Maaf, jawaban yang dihasilkan tidak dapat ditampilkan karena "
                    "tidak sesuai dengan standar konten Islami kami. "
                    "Silakan ajukan pertanyaan dengan cara yang berbeda atau "
                    "berkonsultasi dengan ulama."
                ),
                citations=[],
                confidence=0.0,
                intent=intent,
                session_id=session_id,
                response_time_ms=int((time.time() - start_time) * 1000)
            )

        # Step 8: Generate citations
        logger.debug("Step 8: Generating citations...")
        citations = self.citation_generator.generate(chunks)

        # Step 9: Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Step 10: Build final response
        final_response = FinalResponse(
            answer=response_text,
            citations=citations,
            confidence=validation_result.confidence,
            intent=intent,
            session_id=session_id,
            response_time_ms=processing_time_ms
        )

        # Step 11: Update session with new conversation turn
        logger.debug("Step 11: Updating session...")
        user_turn = ConversationTurn(role=RoleType.USER, message=query)
        assistant_turn = ConversationTurn(role=RoleType.ASSISTANT, message=response_text)
        self.session_manager.update_session(session_id, user_turn)
        self.session_manager.update_session(session_id, assistant_turn)

        # Step 12: Log audit
        logger.debug("Step 12: Logging audit...")
        self.audit_logger.log_query(
            session_id=session_id,
            query=query,
            intent_category=intent,
            response=response_text,
            sources_used=[c.source_ref for c in citations],
            response_time_ms=processing_time_ms
        )

        logger.info(
            f"Query processing completed: {processing_time_ms}ms, "
            f"{len(citations)} citations"
        )

        return final_response

# Convenience function
def create_query_processor(
    rag_engine: RAGEngine,
    session_manager: SessionManager,
    audit_logger: AuditLogger
) -> QueryProcessor:
    """
    Factory function untuk create QueryProcessor instance.

    Args:
        rag_engine: RAGEngine instance
        session_manager: SessionManager instance
        audit_logger: AuditLogger instance

    Returns:
        QueryProcessor instance
    """
    return QueryProcessor(
        rag_engine=rag_engine,
        session_manager=session_manager,
        audit_logger=audit_logger
    )
