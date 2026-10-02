"""Append-only SQLite store for audit records (standard-library `sqlite3`).

Append-only is enforced by the database itself: triggers abort any UPDATE or
DELETE on the events table. One review per request_id (a second submission is
refused, not merged). The store never migrates or deletes data: opening a file
written with a different feedback schema version raises instead.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from conversion_router.feedback.schemas import FEEDBACK_SCHEMA_VERSION, AuditRecord

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "logs" / "feedback" / "feedback.sqlite3"
DB_PATH_ENV_VAR = "CQR_FEEDBACK_DB"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS store_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    request_id TEXT NOT NULL UNIQUE,
    agent_decision TEXT NOT NULL,
    human_response TEXT NOT NULL,
    override_reason TEXT,
    final_decision TEXT NOT NULL,
    record_json TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS feedback_events_no_update
BEFORE UPDATE ON feedback_events
BEGIN SELECT RAISE(ABORT, 'feedback_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS feedback_events_no_delete
BEFORE DELETE ON feedback_events
BEGIN SELECT RAISE(ABORT, 'feedback_events is append-only'); END;
"""


class DuplicateFeedbackError(ValueError):
    """The request already has a recorded review."""


class IncompatibleStoreError(RuntimeError):
    """The file was written with another feedback schema version."""


class CorruptRecordError(RuntimeError):
    """A stored row no longer validates against the audit-record contract."""


class FeedbackStore:
    def __init__(self, path: Path | str = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            row = conn.execute(
                "SELECT value FROM store_meta WHERE key = 'feedback_schema_version'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO store_meta (key, value) VALUES ('feedback_schema_version', ?)",
                    (FEEDBACK_SCHEMA_VERSION,),
                )
            elif row[0] != FEEDBACK_SCHEMA_VERSION:
                raise IncompatibleStoreError(
                    f"{self.path} uses feedback schema {row[0]}, this code expects "
                    f"{FEEDBACK_SCHEMA_VERSION}. Use a new file; existing data is never migrated."
                )

    @classmethod
    def from_env(cls) -> FeedbackStore:
        return cls(os.environ.get(DB_PATH_ENV_VAR, DEFAULT_DB_PATH))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """One short-lived connection per operation: commit on success, always close."""
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def append(self, record: AuditRecord) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO feedback_events (event_id, created_at, request_id, "
                    "agent_decision, human_response, override_reason, final_decision, "
                    "record_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.event_id,
                        record.created_at,
                        record.request_id,
                        record.agent_decision.value,
                        record.human_response.value,
                        record.override_reason.value if record.override_reason else None,
                        record.final_decision.value,
                        record.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "request_id" in str(exc):
                raise DuplicateFeedbackError(
                    f"Request {record.request_id} already has a recorded review."
                ) from exc
            raise

    def get_by_request(self, request_id: str) -> AuditRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT seq, record_json FROM feedback_events WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return self._parse(*row) if row else None

    def records(self, limit: int | None = None) -> list[AuditRecord]:
        """Oldest first; with `limit`, the most recent `limit` records."""
        query = "SELECT seq, record_json FROM feedback_events ORDER BY seq"
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        if limit is not None:
            rows = rows[-limit:] if limit > 0 else []
        return [self._parse(seq, payload) for seq, payload in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]

    @staticmethod
    def _parse(seq: int, payload: str) -> AuditRecord:
        try:
            return AuditRecord.model_validate(json.loads(payload))
        except ValueError as exc:
            raise CorruptRecordError(f"Stored feedback row {seq} is not valid.") from exc
