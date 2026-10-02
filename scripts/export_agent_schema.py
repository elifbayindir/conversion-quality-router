"""Export LLMDecisionCore JSON Schema to prompts/decision_agent_schema_v1.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = PROJECT_ROOT / "prompts" / "decision_agent_schema_v1.json"


def _generate() -> str:
    from conversion_router.schemas import LLMDecisionCore

    schema = LLMDecisionCore.model_json_schema()
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed file matches the runtime schema; exit 1 if stale.",
    )
    args = parser.parse_args()

    expected = _generate()

    if args.check:
        if not SCHEMA_PATH.exists():
            print(f"FAIL: {SCHEMA_PATH.relative_to(PROJECT_ROOT)} does not exist")
            sys.exit(1)
        actual = SCHEMA_PATH.read_text(encoding="utf-8")
        if actual == expected:
            print("PASS: schema artifact matches runtime LLMDecisionCore")
        else:
            print("FAIL: schema artifact is stale — rerun without --check to regenerate")
            sys.exit(1)
    else:
        SCHEMA_PATH.write_text(expected, encoding="utf-8")
        print(f"Written: {SCHEMA_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
