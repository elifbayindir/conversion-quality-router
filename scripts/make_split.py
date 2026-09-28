#!/usr/bin/env python3
"""Generate and persist the reproducible train/validation/test split.

Writes row indices to data/processed/split_indices.json (git-ignored,
regenerated deterministically from the fixed seed) and a git-tracked summary
to artifacts/metadata/split_summary.json for phase-gate evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from conversion_router.data.load import load_raw_dataset
from conversion_router.data.split import (
    save_split_indices,
    split_summary,
    stratified_split,
)
from conversion_router.data.validate import validate_raw_schema

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "split_summary.json"


def main() -> None:
    df = load_raw_dataset()
    validate_raw_schema(df)

    split, fingerprints = stratified_split(df)
    save_split_indices(split)

    summary = split_summary(df, split, fingerprints)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for part in summary["splits"]:
        print(
            f"{part['name']}: {part['n_rows']} rows "
            f"({part['fraction_of_total']:.1%}), "
            f"positive_rate={part['positive_rate']:.4f}"
        )
    dup = summary["duplicate_leakage_check"]
    print(
        f"duplicate fingerprint groups: {dup['n_duplicate_fingerprint_groups']} "
        f"({dup['n_duplicate_extra_rows']} extra rows), "
        f"conflicting-label groups: {dup['n_conflicting_label_groups']}, "
        f"cross-split fingerprint groups: {dup['n_cross_split_fingerprint_groups']}"
    )
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
