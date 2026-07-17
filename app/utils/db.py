import asyncio
import json as _json
import pymysql
from datetime import datetime
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_connection():
    """Retrieve a new connection to the target database."""
    settings = get_settings()
    return pymysql.connect(
        host=settings.db_host,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.db_name,
        port=settings.db_port,
        autocommit=True
    )


def _init_db_sync() -> None:
    """Initialize database and create tables synchronously."""
    settings = get_settings()
    # Ensure database exists
    conn = pymysql.connect(
        host=settings.db_host,
        user=settings.db_user,
        password=settings.db_password,
        port=settings.db_port,
        autocommit=True
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {settings.db_name};")
    finally:
        conn.close()

    # Create tables with callbot_ prefix
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS callbot_sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    client_id VARCHAR(255) NULL,
                    lead_id VARCHAR(255) NULL,
                    system_prompt TEXT NULL,
                    greeting_message TEXT NULL,
                    created_at TIMESTAMP NULL,
                    updated_at TIMESTAMP NULL
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS callbot_chats (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    tokens_used INT NULL,
                    intent_extraction JSON NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES callbot_sessions(session_id) ON DELETE CASCADE
                );
            """)
            # Add updated_at only if it doesn't already exist (IF NOT EXISTS
            # syntax is not supported on all MySQL/RDS versions).
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME   = 'callbot_sessions'
                  AND COLUMN_NAME  = 'updated_at';
            """)
            (col_exists,) = cursor.fetchone()
            if not col_exists:
                cursor.execute("""
                    ALTER TABLE callbot_sessions
                    ADD COLUMN updated_at TIMESTAMP NULL;
                """)

            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME   = 'callbot_sessions'
                  AND COLUMN_NAME  = 'system_prompt';
            """)
            (col_exists,) = cursor.fetchone()
            if not col_exists:
                cursor.execute("""
                    ALTER TABLE callbot_sessions
                    ADD COLUMN system_prompt TEXT NULL;
                """)

            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME   = 'callbot_sessions'
                  AND COLUMN_NAME  = 'greeting_message';
            """)
            (col_exists,) = cursor.fetchone()
            if not col_exists:
                cursor.execute("""
                    ALTER TABLE callbot_sessions
                    ADD COLUMN greeting_message TEXT NULL;
                """)

            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME   = 'callbot_sessions'
                  AND COLUMN_NAME  = 'outcome_data';
            """)
            (col_exists,) = cursor.fetchone()
            if not col_exists:
                cursor.execute("""
                    ALTER TABLE callbot_sessions
                    ADD COLUMN outcome_data JSON NULL;
                """)

            # Add tokens_used column to callbot_chats if it doesn't exist yet.
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME   = 'callbot_chats'
                  AND COLUMN_NAME  = 'tokens_used';
            """)
            (col_exists,) = cursor.fetchone()
            if not col_exists:
                cursor.execute("""
                    ALTER TABLE callbot_chats
                    ADD COLUMN tokens_used INT NULL;
                """)

            # Add intent_extraction column to callbot_chats if it doesn't exist yet.
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME   = 'callbot_chats'
                  AND COLUMN_NAME  = 'intent_extraction';
            """)
            (col_exists,) = cursor.fetchone()
            if not col_exists:
                cursor.execute("""
                    ALTER TABLE callbot_chats
                    ADD COLUMN intent_extraction JSON NULL;
                """)
        logger.info("Database tables initialized successfully with prefix 'callbot_'.")
    except Exception as e:
        logger.exception("Failed to initialize database tables: %s", e)
        raise e
    finally:
        conn.close()


async def init_db() -> None:
    """Initialize database and create tables asynchronously."""
    await asyncio.to_thread(_init_db_sync)


def parse_timestamp(ts_val) -> datetime | None:
    """Parse an arbitrary timestamp input (string/int/float/datetime) to a timezone-aware datetime."""
    if not ts_val:
        return None
    if isinstance(ts_val, (int, float)):
        return datetime.fromtimestamp(ts_val)
    if isinstance(ts_val, datetime):
        return ts_val
    try:
        # Try ISO format
        return datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        # Try numeric string timestamp
        return datetime.fromtimestamp(float(ts_val))
    except ValueError:
        pass
    return None


def _save_session_sync(
    session_id: str,
    client_id: str | None,
    lead_id: str | None,
    system_prompt: str | None,
    greeting_message: str | None,
    timestamp: datetime | None,
    outcome_data: dict | None = None,
) -> None:
    """Insert or update session metadata synchronously."""
    conn = get_connection()
    try:
        ts = timestamp or datetime.utcnow()
        outcome_json = _json.dumps(outcome_data) if outcome_data is not None else None
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO callbot_sessions (
                    session_id, client_id, lead_id, system_prompt, greeting_message, outcome_data, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    client_id = COALESCE(%s, client_id),
                    lead_id = COALESCE(%s, lead_id),
                    system_prompt = COALESCE(%s, system_prompt),
                    greeting_message = COALESCE(%s, greeting_message),
                    outcome_data = COALESCE(%s, outcome_data),
                    updated_at = %s,
                    created_at = COALESCE(%s, created_at)
            """, (
                session_id,
                client_id,
                lead_id,
                system_prompt,
                greeting_message,
                outcome_json,
                ts,
                ts,
                client_id,
                lead_id,
                system_prompt,
                greeting_message,
                outcome_json,
                ts,
                ts,
            ))
    except Exception as e:
        logger.error("Error saving session %s: %s", session_id, e)
        raise
    finally:
        conn.close()


async def save_session(
    session_id: str,
    client_id: str | None,
    lead_id: str | None,
    system_prompt: str | None = None,
    greeting_message: str | None = None,
    timestamp: datetime | None = None,
    outcome_data: dict | None = None,
) -> None:
    """Insert or update session metadata asynchronously."""
    await asyncio.to_thread(
        _save_session_sync,
        session_id,
        client_id,
        lead_id,
        system_prompt,
        greeting_message,
        timestamp,
        outcome_data,
    )


def _save_message_sync(
    session_id: str,
    role: str,
    content: str,
    timestamp: datetime | None = None,
    tokens_used: int | None = None,
    intent_extraction: dict | None = None,
) -> None:
    """Insert chat turn message synchronously."""
    import json as _json
    conn = get_connection()
    try:
        now = timestamp or datetime.utcnow()
        intent_json = _json.dumps(intent_extraction) if intent_extraction is not None else None
        with conn.cursor() as cursor:
            # Ensure the session exists in the parent table first to avoid FK constraints.
            cursor.execute("""
                INSERT IGNORE INTO callbot_sessions (
                    session_id, created_at, updated_at
                ) VALUES (%s, %s, %s)
            """, (session_id, now, now))

            # Save the message
            cursor.execute("""
                INSERT INTO callbot_chats (session_id, role, message, tokens_used, intent_extraction, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (session_id, role, content, tokens_used, intent_json, now))

            cursor.execute(
                "UPDATE callbot_sessions SET updated_at = %s WHERE session_id = %s",
                (now, session_id),
            )
    except Exception as e:
        logger.error("Error saving message for session %s: %s", session_id, e)
        raise
    finally:
        conn.close()


async def save_message(
    session_id: str,
    role: str,
    content: str,
    timestamp: datetime | None = None,
    tokens_used: int | None = None,
    intent_extraction: dict | None = None,
) -> None:
    """Insert chat turn message asynchronously."""
    await asyncio.to_thread(
        _save_message_sync, session_id, role, content, timestamp, tokens_used, intent_extraction
    )


def _get_session_list_sync(
    client_id: str | None = None,
    lead_id: str | None = None,
) -> list[dict]:
    """Retrieve session metadata filtered by client and/or lead."""
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            query = [
                "SELECT session_id, client_id, lead_id, system_prompt, greeting_message, outcome_data, created_at, updated_at",
                "FROM callbot_sessions",
            ]
            params: list[object] = []
            filters: list[str] = []

            if client_id:
                filters.append("client_id = %s")
                params.append(client_id)
            if lead_id:
                filters.append("lead_id = %s")
                params.append(lead_id)
            if filters:
                query.append("WHERE " + " AND ".join(filters))
            query.append("ORDER BY updated_at DESC, created_at DESC")

            cursor.execute("\n".join(query), tuple(params))
            rows = cursor.fetchall()
            for row in rows:
                if isinstance(row["created_at"], datetime):
                    row["created_at"] = row["created_at"].isoformat()
                if isinstance(row["updated_at"], datetime):
                    row["updated_at"] = row["updated_at"].isoformat()
                if isinstance(row.get("outcome_data"), str):
                    try:
                        row["outcome_data"] = _json.loads(row["outcome_data"])
                    except Exception:
                        row["outcome_data"] = None
            return rows
    except Exception as e:
        logger.error("Error fetching sessions: %s", e)
        return []
    finally:
        conn.close()


def _get_session_sync(session_id: str) -> dict | None:
    """Retrieve a single session record synchronously."""
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT session_id, client_id, lead_id, system_prompt, greeting_message, outcome_data, created_at, updated_at
                FROM callbot_sessions
                WHERE session_id = %s
            """, (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            if isinstance(row["created_at"], datetime):
                row["created_at"] = row["created_at"].isoformat()
            if isinstance(row["updated_at"], datetime):
                row["updated_at"] = row["updated_at"].isoformat()
            if isinstance(row.get("outcome_data"), str):
                try:
                    row["outcome_data"] = _json.loads(row["outcome_data"])
                except Exception:
                    row["outcome_data"] = None
            return row
    except Exception as e:
        logger.error("Error fetching session %s: %s", session_id, e)
        return None
    finally:
        conn.close()


async def get_session(session_id: str) -> dict | None:
    """Retrieve a single session record asynchronously."""
    return await asyncio.to_thread(_get_session_sync, session_id)


def _get_chat_history_sync(session_id: str, page: int = 1, page_size: int = 100) -> list[dict]:
    """Retrieve history for a session synchronously with pagination."""
    import json as _json
    offset = (page - 1) * page_size
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT role, message, tokens_used, intent_extraction, created_at
                FROM callbot_chats
                WHERE session_id = %s
                ORDER BY created_at ASC, id ASC
                LIMIT %s OFFSET %s
            """, (session_id, page_size, offset))
            rows = cursor.fetchall()
            for row in rows:
                if isinstance(row["created_at"], datetime):
                    row["created_at"] = row["created_at"].isoformat()
                # intent_extraction may come back as a string from older MySQL drivers
                if isinstance(row.get("intent_extraction"), str):
                    try:
                        row["intent_extraction"] = _json.loads(row["intent_extraction"])
                    except Exception:
                        row["intent_extraction"] = None
            return rows
    except Exception as e:
        logger.error("Error fetching chat history for session %s: %s", session_id, e)
        return []
    finally:
        conn.close()


async def get_sessions(
    client_id: str | None = None,
    lead_id: str | None = None,
) -> list[dict]:
    """Retrieve session metadata asynchronously."""
    return await asyncio.to_thread(_get_session_list_sync, client_id, lead_id)


async def get_chat_history(session_id: str, page: int = 1, page_size: int = 100) -> list[dict]:
    """Retrieve history for a session asynchronously."""
    return await asyncio.to_thread(_get_chat_history_sync, session_id, page, page_size)


def _update_last_assistant_intent_sync(
    session_id: str,
    tokens_used: int | None,
    intent_extraction: dict | None,
) -> None:
    """Patch the most-recent assistant row for a session with intent data synchronously."""
    import json as _json
    conn = get_connection()
    try:
        intent_json = _json.dumps(intent_extraction) if intent_extraction is not None else None
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE callbot_chats
                SET    tokens_used      = COALESCE(%s, tokens_used),
                       intent_extraction = COALESCE(%s, intent_extraction)
                WHERE  session_id = %s
                  AND  role       = 'assistant'
                  AND  id = (
                        SELECT id FROM (
                            SELECT MAX(id) AS id
                            FROM callbot_chats
                            WHERE session_id = %s AND role = 'assistant'
                        ) AS t
                  )
                """,
                (tokens_used, intent_json, session_id, session_id),
            )
    except Exception as e:
        logger.error(
            "Error updating intent for last assistant row in session %s: %s", session_id, e
        )
        raise
    finally:
        conn.close()


async def update_last_assistant_intent(
    session_id: str,
    tokens_used: int | None = None,
    intent_extraction: dict | None = None,
) -> None:
    """Patch the most-recent assistant row with intent data asynchronously."""
    await asyncio.to_thread(
        _update_last_assistant_intent_sync, session_id, tokens_used, intent_extraction
    )
