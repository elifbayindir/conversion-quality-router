import ast
import hashlib
import sqlite3
import subprocess
from pathlib import Path

import pytest

from conversion_router.feedback.schemas import FeedbackSubmission, build_audit_record
from conversion_router.feedback.store import (
    DEFAULT_DB_PATH,
    CorruptRecordError,
    DuplicateFeedbackError,
    FeedbackStore,
    IncompatibleStoreError,
)

from .feedback_factories import SESSION, decision, prediction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_DIR = PROJECT_ROOT / "src" / "conversion_router" / "feedback"


def _record(request_id: str, response: str = "AGREE", **override):
    return build_audit_record(
        SESSION, prediction(request_id), decision(request_id),
        FeedbackSubmission(response=response, **override),
    )


def test_append_and_read_back(tmp_path):
    store = FeedbackStore(tmp_path / "fb.sqlite3")
    first = _record("req_1")
    second = _record("req_2", "OVERRIDE", override_decision="LOG_ONLY",
                     override_reason="SIGNAL_WEAKER_THAN_SCORED")
    store.append(first)
    store.append(second)
    assert store.count() == 2
    assert store.records() == [first, second]
    assert store.records(limit=1) == [second]
    assert store.get_by_request("req_2") == second
    assert store.get_by_request("req_missing") is None


def test_records_survive_a_restart(tmp_path):
    path = tmp_path / "fb.sqlite3"
    written = _record("req_restart")
    FeedbackStore(path).append(written)
    reopened = FeedbackStore(path)  # new instance, new connections
    assert reopened.records() == [written]


def test_database_rejects_update_and_delete(tmp_path):
    path = tmp_path / "fb.sqlite3"
    FeedbackStore(path).append(_record("req_locked"))
    conn = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE feedback_events SET final_decision = 'LOG_ONLY'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM feedback_events")
    finally:
        conn.close()
    assert FeedbackStore(path).count() == 1


def test_store_exposes_no_update_or_delete_api():
    public = {name for name in dir(FeedbackStore) if not name.startswith("_")}
    assert public == {"append", "count", "from_env", "get_by_request", "records"}


def test_second_review_of_the_same_request_is_refused(tmp_path):
    store = FeedbackStore(tmp_path / "fb.sqlite3")
    store.append(_record("req_dup"))
    with pytest.raises(DuplicateFeedbackError):
        store.append(_record("req_dup"))
    assert store.count() == 1


def test_other_schema_version_is_refused_without_migration(tmp_path):
    path = tmp_path / "fb.sqlite3"
    FeedbackStore(path).append(_record("req_v1"))
    conn = sqlite3.connect(path)
    with conn:
        conn.execute("UPDATE store_meta SET value = '0.9.0'")
    conn.close()
    with pytest.raises(IncompatibleStoreError):
        FeedbackStore(path)
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0] == 1
    conn.close()


def test_corrupt_rows_are_reported_not_skipped(tmp_path):
    path = tmp_path / "fb.sqlite3"
    store = FeedbackStore(path)
    conn = sqlite3.connect(path)
    with conn:
        conn.execute(
            "INSERT INTO feedback_events (event_id, created_at, request_id, agent_decision, "
            "human_response, final_decision, record_json) VALUES "
            "('fb_bad', 't', 'req_bad', 'LOG_ONLY', 'AGREE', 'LOG_ONLY', '{\"x\": 1}')"
        )
    conn.close()
    with pytest.raises(CorruptRecordError):
        store.records()


def test_default_location_is_git_ignored():
    assert DEFAULT_DB_PATH.relative_to(PROJECT_ROOT).parts[0] == "logs"
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(DEFAULT_DB_PATH)], cwd=PROJECT_ROOT, check=False
    )
    assert result.returncode == 0


def test_feedback_package_cannot_touch_model_or_policy():
    forbidden = ("torch", "sklearn", "joblib", "conversion_router.modeling",
                 "conversion_router.agent", "conversion_router.api", "conversion_router.ui")
    for path in FEEDBACK_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            assert not [m for m in modules if m.startswith(forbidden)], path.name


def _tree_hash(*roots: Path) -> str:
    digest = hashlib.sha256()
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_recording_feedback_leaves_artifacts_and_prompts_unchanged(tmp_path):
    protected = (PROJECT_ROOT / "artifacts", PROJECT_ROOT / "prompts")
    before = _tree_hash(*protected)
    store = FeedbackStore(tmp_path / "fb.sqlite3")
    for i in range(25):
        store.append(_record(f"req_{i}", "OVERRIDE", override_decision="PRIORITY_REVIEW",
                             override_reason="SIGNAL_STRONGER_THAN_SCORED"))
    assert _tree_hash(*protected) == before
