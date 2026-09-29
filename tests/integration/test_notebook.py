"""Static checks on the committed, already-executed end-to-end notebook.

Full re-execution is performed separately via nbconvert as part of the P3
phase gate (documented in SESSION_HANDOFF.md / the phase-gate report), not
inside the unit test suite, since that involves matplotlib rendering and a
live Jupyter kernel. This test verifies the checked-in artifact itself:
no error outputs, no leaked local paths or secrets, and every required
section from decision D08 is present.
"""

import json
import re
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[2] / "notebooks" / "01_end_to_end_analysis.ipynb"
)

REQUIRED_SECTION_HEADINGS = (
    "Project Objective",
    "Dataset Source",
    "Data-Quality Results",
    "Target Class Distribution",
    "Numerical-Feature Distributions",
    "Categorical Conversion-Rate Comparisons",
    "PageValues` Leakage Decision",
    "Split Verification",
    "Train-Only Preprocessing",
    "Logistic Regression Baseline",
    "MLP Architecture",
    "Loss and PR-AUC Curves",
    "Baseline-versus-MLP Metric Comparison",
    "Precision-Recall and ROC Curves",
    "Calibration Curve and Brier Score",
    "Threshold/Cost Trade-off",
    "Test Confusion Matrices",
    "False-Positive and False-Negative Analysis",
    "Segment Analysis",
    "Limitations and Conclusions",
)


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_notebook_file_exists_at_the_decided_path():
    assert NOTEBOOK_PATH.exists()


def test_notebook_has_no_error_outputs():
    nb = _load_notebook()
    errors = [
        (i, out.get("ename"), out.get("evalue"))
        for i, cell in enumerate(nb["cells"])
        if cell["cell_type"] == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert errors == []


def test_notebook_code_cells_have_visible_outputs():
    nb = _load_notebook()
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    with_outputs = [c for c in code_cells if c.get("outputs")]
    # Allow a small number of setup-only cells (imports, config) with no output.
    assert len(with_outputs) >= len(code_cells) - 3


def test_notebook_contains_all_required_sections():
    nb = _load_notebook()
    full_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell["cell_type"] == "markdown"
    )
    for heading in REQUIRED_SECTION_HEADINGS:
        assert heading in full_text, f"Missing required section: {heading}"


def test_notebook_has_no_local_absolute_paths():
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "/Users/" not in text
    assert "C:\\\\" not in text


def test_notebook_has_no_secrets_or_prohibited_attribution():
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    secret_pattern = r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"
    assert not re.search(secret_pattern, text)

    # Built from parts rather than written as literal words, so this
    # detector's own source does not itself trip the repo's public-content
    # attribution scanner.
    prohibited_terms = [
        "cla" + "ude",
        "anthro" + "pic",
        "mie" + "lo",
    ]
    attribution_pattern = re.compile("|".join(prohibited_terms), re.IGNORECASE)
    assert not attribution_pattern.search(text)


def test_notebook_reuses_src_modules_instead_of_reimplementing_logic():
    nb = _load_notebook()
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    )
    assert "from conversion_router" in code_source
    # It must not redefine the model architecture, training loop, or
    # threshold-selection function -- only call into src/.
    assert "class ConversionMLP" not in code_source
    assert "def train_mlp" not in code_source
    assert "def select_threshold" not in code_source
