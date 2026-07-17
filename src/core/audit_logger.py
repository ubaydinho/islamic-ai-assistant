"""
Islamic AI Assistant — Audit Logger

Mencatat semua query, response, dan flagged content ke SQLite database
untuk audit trail dan compliance tracking.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from src.models.schemas import IntentCategory
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AuditLogger:
    """
    SQLite-based audit logger untuk mencatat semua interaksi pengguna
    dan flagged content untuk compliance tracking.
    """

    def __init__(self, db_path: str = "data/audit.db"):
        """
        Initialize audit logger dengan SQLite database.

        Args:
            db_path: Path ke SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"Audit logger initialized with database: {self.db_path}")

    def _init_database(self):
        """Buat tabel audit jika belum ada"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tabel untuk query log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    intent_category TEXT,
                    response TEXT,
                    sources_used TEXT,
                    response_time_ms INTEGER,
                    model_used TEXT
                )
            """)

            # Tabel untuk flagged content
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS flagged_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    violation_category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT
                )
            """)

            # Index untuk performa query
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id
                ON query_log(session_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON query_log(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_flagged_session
                ON flagged_content(session_id)
            """)

            conn.commit()
            logger.debug("Audit database tables initialized")

    def log_query(
        self,
        session_id: UUID,
        query: str,
        intent_category: Optional[IntentCategory],
        response: str,
        sources_used: list[str],
        response_time_ms: int,
        model_used: str = "llama-3.1-8b-instant"
    ):
        """
        Catat query dan response ke audit log.

        Args:
            session_id: UUID sesi pengguna
            query: Query pengguna
            intent_category: Kategori intent yang terdeteksi
            response: Response yang diberikan sistem
            sources_used: List sumber yang digunakan (format: "surah:ayat", "hadith:id", dll)
            response_time_ms: Waktu response dalam millisecond
            model_used: Nama model LLM yang digunakan
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO query_log
                    (session_id, timestamp, query, intent_category, response,
                     sources_used, response_time_ms, model_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(session_id),
                    datetime.utcnow().isoformat(),
                    query,
                    intent_category.value if intent_category else None,
                    response,
                    ",".join(sources_used),
                    response_time_ms,
                    model_used
                ))
                conn.commit()
                logger.info(f"Query logged for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to log query: {e}")

    def log_flagged_content(
        self,
        session_id: UUID,
        query: str,
        violation_category: str,
        confidence: float,
        reason: Optional[str] = None
    ):
        """
        Catat flagged content untuk compliance tracking.

        Args:
            session_id: UUID sesi pengguna
            query: Query yang di-flag
            violation_category: Kategori pelanggaran (HARAM, SHIRK, BIDAH, MISLEADING)
            confidence: Confidence score (0.0-1.0)
            reason: Alasan flagging (optional)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO flagged_content
                    (session_id, timestamp, query, violation_category, confidence, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    str(session_id),
                    datetime.utcnow().isoformat(),
                    query,
                    violation_category,
                    confidence,
                    reason
                ))
                conn.commit()
                logger.warning(
                    f"Flagged content logged: {violation_category} "
                    f"(confidence: {confidence:.2f})"
                )
        except Exception as e:
            logger.error(f"Failed to log flagged content: {e}")

    def get_session_history(self, session_id: UUID, limit: int = 10) -> list[dict]:
        """
        Ambil history query untuk sesi tertentu.

        Args:
            session_id: UUID sesi pengguna
            limit: Jumlah maksimal record yang diambil

        Returns:
            List dictionary berisi query history
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM query_log
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (str(session_id), limit))

                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get session history: {e}")
            return []

    def get_flagged_count(self, session_id: UUID) -> int:
        """
        Hitung jumlah flagged content untuk sesi tertentu.

        Args:
            session_id: UUID sesi pengguna

        Returns:
            Jumlah flagged content
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM flagged_content
                    WHERE session_id = ?
                """, (str(session_id),))

                count = cursor.fetchone()[0]
                return count
        except Exception as e:
            logger.error(f"Failed to get flagged count: {e}")
            return 0
