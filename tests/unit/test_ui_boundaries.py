"""Architecture boundary: the UI is an HTTP client only.

It must not import the model stack or the agent implementation, and must not
read the project's .env file.
"""

import ast
import re
from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[2] / "src" / "conversion_router" / "ui"

FORBIDDEN_PREFIXES = (
    "torch",
    "sklearn",
    "joblib",
    "dotenv",
    "conversion_router.modeling",
    "conversion_router.agent",
    "conversion_router.api",
    "conversion_router.data.preprocess",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _ui_files() -> list[Path]:
    files = sorted(UI_DIR.glob("*.py"))
    assert files, "UI package not found"
    return files


def test_ui_does_not_import_model_agent_or_api_internals():
    violations = [
        f"{path.name}: {module}"
        for path in _ui_files()
        for module in _imported_modules(path)
        if module.startswith(FORBIDDEN_PREFIXES)
    ]
    assert violations == []


def test_ui_does_not_reference_the_env_file_or_provider_credentials():
    for path in _ui_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        strings = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        assert not [s for s in strings if re.search(r"(^|[/\\])\.env\b", s)], path.name
        assert not [s for s in strings if "ANTHROPIC" in s], path.name


def test_ui_reuses_the_shared_contracts():
    imported = set().union(*(_imported_modules(p) for p in _ui_files()))
    assert "conversion_router.schemas" in imported
