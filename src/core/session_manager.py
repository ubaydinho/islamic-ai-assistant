"""
Islamic AI Assistant — Session Manager

Mengelola session context pengguna dengan SQLite sebagai persistent storage.
Setiap session menyimpan history percakapan dan preferensi pengguna.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from src.models.schemas import SessionContext, ConversationTurn
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SessionManager:
    """
    Mengelola session context pengguna dengan SQLite persistence.
    Thread-safe untuk concurrent access.
    """

    def __init__(self, db_path: str = "data/sessions.db"):
        """
        Initialize session manager dengan SQLite database.

        Args:
            db_path: Path ke SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"Session manager initialized with database: {self.db_path}")

    def _init_database(self):
        """Buat tabel sessions jika belum ada"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    history_json TEXT NOT NULL,
                    language TEXT DEFAULT 'id',
                    madhab_preference TEXT DEFAULT 'shafii',
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL
                )
            """)

            # Index untuk query performa
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id
                ON sessions(user_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_active
                ON sessions(last_active)
            """)

            conn.commit()
            logger.debug("Session database tables initialized")

    def create_session(
        self,
        user_id: Optional[str] = None,
        language: str = "id",
        madhab_preference: str = "shafii"
    ) -> SessionContext:
        """
        Buat session baru dengan UUID v4 dan simpan ke database.

        Args:
            user_id: Optional user identifier
            language: Bahasa preferensi (default: 'id' untuk Indonesian)
            madhab_preference: Preferensi mazhab fiqh (default: 'shafii')

        Returns:
            SessionContext object yang baru dibuat
        """
        session_id = uuid4()
        now = datetime.utcnow().isoformat()

        session_context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            history=[],
            language=language,
            madhab_preference=madhab_preference,
            created_at=now,
            last_active=now
        )

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sessions
                    (session_id, user_id, history_json, language,
                     madhab_preference, created_at, last_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(session_id),
                    user_id,
                    json.dumps([]),
                    language,
                    madhab_preference,
                    now,
                    now
                ))
                conn.commit()
                logger.info(f"Created new session: {session_id}")
                return session_context
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise

    def get_session(self, session_id: UUID) -> Optional[SessionContext]:
        """
        Load session context dari database.

        Args:
            session_id: UUID session yang akan di-load

        Returns:
            SessionContext jika ditemukan, None jika tidak ada
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM sessions WHERE session_id = ?
                """, (str(session_id),))

                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Session not found: {session_id}")
                    return None

                # Parse history JSON dengan toleransi format lama
                history_json = json.loads(row["history_json"])
                history = []
                for turn in history_json:
                    try:
                        history.append(ConversationTurn(**turn))
                    except Exception:
                        # Lewati turn yang formatnya tidak kompatibel (data lama)
                        logger.warning(f"Skipped incompatible history turn: {list(turn.keys())}")
                        continue

                session_context = SessionContext(
                    session_id=UUID(row["session_id"]),
                    user_id=UUID(row["user_id"]) if row["user_id"] else None,
                    history=history,
                    language=row["language"],
                    madhab_preference=row["madhab_preference"],
                    created_at=row["created_at"],
                    last_active=row["last_active"]
                )

                logger.debug(f"Loaded session: {session_id}")
                return session_context
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None

    def update_session(
        self,
        session_id: UUID,
        turn: ConversationTurn
    ) -> bool:
        """
        Update session dengan conversation turn baru secara atomic.
        Append turn ke history dan update last_active timestamp.

        Args:
            session_id: UUID session yang akan di-update
            turn: ConversationTurn baru yang akan ditambahkan

        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Ambil history sekarang
                cursor.execute("""
                    SELECT history_json FROM sessions WHERE session_id = ?
                """, (str(session_id),))

                row = cursor.fetchone()
                if not row:
                    logger.error(f"Session not found for update: {session_id}")
                    return False

                # Parse dan append turn baru
                history = json.loads(row[0])
                history.append(turn.model_dump(mode="json"))

                # Update dengan atomic transaction
                cursor.execute("""
                    UPDATE sessions
                    SET history_json = ?, last_active = ?
                    WHERE session_id = ?
                """, (
                    json.dumps(history),
                    datetime.utcnow().isoformat(),
                    str(session_id)
                ))

                conn.commit()
                logger.debug(f"Updated session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            return False

    def delete_old_sessions(self, days: int = 30) -> int:
        """
        Hapus session yang tidak aktif lebih dari N hari.

        Args:
            days: Jumlah hari inaktif sebelum dihapus

        Returns:
            Jumlah session yang dihapus
        """
        try:
            cutoff_date = datetime.utcnow().replace(
                day=datetime.utcnow().day - days
            ).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM sessions
                    WHERE last_active < ?
                """, (cutoff_date,))

                deleted_count = cursor.rowcount
                conn.commit()
                logger.info(f"Deleted {deleted_count} old sessions")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete old sessions: {e}")
            return 0
