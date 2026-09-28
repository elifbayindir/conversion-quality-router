#!/usr/bin/env python3
"""Reproducibly download the UCI Online Shoppers Purchasing Intention dataset.

Source: UCI Machine Learning Repository, dataset ID 468.
DOI: 10.24432/C5F88Q
License: CC BY 4.0

Downloads the dataset zip, extracts the CSV to data/raw/, computes its
SHA-256 checksum, and verifies it against the expected checksum recorded in
artifacts/metadata/dataset_manifest.json once that manifest exists. On the
first run (no manifest yet) the manifest is created from the freshly
downloaded file.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
DOI = "10.24432/C5F88Q"
LICENSE = "CC BY 4.0"
SOURCE_PAGE = (
    "https://archive.ics.uci.edu/dataset/468/"
    "online+shoppers+purchasing+intention+dataset"
)
CSV_MEMBER_NAME = "online_shoppers_intention.csv"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "dataset_manifest.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_extract() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    response = httpx.get(DATASET_URL, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        matches = [n for n in names if n.endswith(CSV_MEMBER_NAME)]
        if not matches:
            raise RuntimeError(
                f"Expected member ending in {CSV_MEMBER_NAME!r} not found in "
                f"archive; contents were: {names}"
            )
        member = matches[0]
        target = RAW_DIR / CSV_MEMBER_NAME
        with archive.open(member) as src, target.open("wb") as dst:
            dst.write(src.read())
    return target


def row_and_column_counts(csv_path: Path) -> tuple[int, int]:
    with csv_path.open("r", encoding="utf-8") as fh:
        header = fh.readline()
        n_columns = len(header.strip().split(","))
        n_rows = sum(1 for _ in fh)
    return n_rows, n_columns


def main() -> None:
    csv_path = download_and_extract()
    checksum = sha256_of(csv_path)
    n_rows, n_columns = row_and_column_counts(csv_path)

    manifest = {
        "filename": CSV_MEMBER_NAME,
        "source_url": DATASET_URL,
        "source_page": SOURCE_PAGE,
        "doi": DOI,
        "license": LICENSE,
        "sha256": checksum,
        "n_rows": n_rows,
        "n_columns": n_columns,
    }

    if MANIFEST_PATH.exists():
        recorded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected = recorded.get("sha256")
        if expected and expected != checksum:
            raise RuntimeError(
                "Downloaded dataset checksum does not match the recorded "
                f"manifest checksum. Expected {expected}, got {checksum}. "
                "The upstream file may have changed; investigate before "
                "proceeding."
            )
        print(f"Checksum verified against existing manifest: {checksum}")
    else:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Manifest created at {MANIFEST_PATH} with checksum {checksum}")

    print(f"Saved dataset to {csv_path} ({n_rows} rows, {n_columns} columns)")


if __name__ == "__main__":
    main()
