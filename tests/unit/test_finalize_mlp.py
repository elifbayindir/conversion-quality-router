import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from finalize_mlp import sha256_of  # noqa: E402

CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "models" / "mlp_checkpoint.pt"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "mlp_metadata.json"


def test_sha256_of_matches_hashlib_reference(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"conversion-quality-router" * 1000)

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    actual = sha256_of(path)

    assert actual == expected


def test_sha256_of_is_deterministic_across_calls(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"\x00\x01\x02" * 5000)

    assert sha256_of(path) == sha256_of(path)


def test_finalize_mlp_script_runs_and_freezes_expected_metadata():
    if not CHECKPOINT_PATH.exists():
        pytest.skip(
            "MLP checkpoint not present; run scripts/train_mlp.py + calibrate_mlp.py first."
        )

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "finalize_mlp.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    for key in (
        "model_version",
        "seed",
        "feature_contract_version",
        "training_configuration",
        "calibration",
        "threshold_selection",
        "package_versions",
        "artifact_hashes",
    ):
        assert key in metadata

    assert metadata["frozen_before_test_evaluation"] is True
    assert 0.0 < metadata["threshold_selection"]["threshold"] < 1.0
    assert len(metadata["artifact_hashes"]["mlp_checkpoint_sha256"]) == 64
