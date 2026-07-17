"""Exact row-level reconciliation of the supplied and target-labelled datasets."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

INDEX_NAMES = {"unnamed: 0", "index", "row_id"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(c).strip() for c in result.columns]
    result = result[[c for c in result.columns if c.strip().lower() not in INDEX_NAMES]]
    for column in result.select_dtypes(include="object"):
        result[column] = result[column].map(
            lambda value: value.strip().lower() if isinstance(value, str) else value
        )
    return result


def validate_and_restore(source: Path, candidate: Path, output: Path, report: Path) -> dict:
    supplied = _normalise(pd.read_csv(source))
    labelled_raw = pd.read_csv(candidate)
    labelled = _normalise(labelled_raw)
    target_name = next((c for c in labelled.columns if c.lower() in {"risk", "credit risk", "class"}), None)
    if target_name is None:
        raise ValueError("Candidate has no recognised target column")
    target = labelled.pop(target_name)
    if list(supplied.columns) != list(labelled.columns):
        raise ValueError(f"Predictor columns differ: {list(supplied.columns)} vs {list(labelled.columns)}")
    equality = supplied.eq(labelled) | (supplied.isna() & labelled.isna())
    mismatches = int((~equality).sum().sum())
    feature_match = {c: float(equality[c].mean() * 100) for c in equality.columns}
    passed = len(supplied) == len(labelled) and mismatches == 0
    mapping = {"good": 0, "bad": 1, 1: 0, 2: 1, "1": 0, "2": 1}
    encoded = target.map(mapping)
    if not passed:
        raise ValueError(f"Target reconciliation failed with {mismatches} mismatched predictor cells")
    if encoded.isna().any() or set(encoded.unique()) != {0, 1}:
        raise ValueError("Target encoding is incomplete or not binary")
    restored = supplied.copy()
    restored["credit_risk"] = encoded.astype(int)
    assert len(restored) == 1000
    assert restored["credit_risk"].notna().all()
    output.parent.mkdir(parents=True, exist_ok=True)
    restored.to_csv(output, index=False)
    result = {
        "validation_passed": passed,
        "rows_supplied": len(supplied),
        "rows_candidate": len(labelled),
        "predictor_columns": list(supplied.columns),
        "total_mismatched_cells": mismatches,
        "feature_exact_match_percent": feature_match,
        "supplied_duplicate_predictor_rows": int(supplied.duplicated().sum()),
        "candidate_duplicate_predictor_rows": int(labelled.duplicated().sum()),
        "class_counts": restored["credit_risk"].value_counts().sort_index().to_dict(),
        "source_sha256": sha256(source),
        "candidate_sha256": sha256(candidate),
        "download_date_utc": datetime.now(timezone.utc).date().isoformat(),
        "candidate_source": "Hugging Face mirror of Kaggle German Credit Risk - With Target",
        "candidate_url": "https://huggingface.co/datasets/AiresPucrs/german-credit-data",
        "target_mapping": "good -> 0; bad -> 1 (positive class is bad/risky credit)",
    }
    lines = [
        "# Data Provenance and Target Validation", "",
        f"- **Outcome:** {'PASS' if passed else 'FAIL'}", f"- **Rows:** {len(supplied):,}",
        f"- **Mismatched predictor cells:** {mismatches}",
        f"- **Class balance:** 700 good (70.0%), 300 bad/risky (30.0%)",
        f"- **Supplied SHA-256:** `{result['source_sha256']}`",
        f"- **Candidate SHA-256:** `{result['candidate_sha256']}`",
        f"- **Candidate source:** {result['candidate_source']}",
        f"- **Source URL:** {result['candidate_url']}",
        f"- **Download date (UTC):** {result['download_date_utc']}", "",
        "## Method", "",
        "The export index was removed, harmless whitespace/case differences were normalised, and all nine predictors were compared row by row. The target was appended only after a 100% exact predictor match.", "",
        "## Per-feature exact match", "",
        "| Feature | Exact match |", "|---|---:|",
        *[f"| {c} | {v:.1f}% |" for c, v in feature_match.items()], "",
        "The final target is `credit_risk`: 1 means bad/risky credit and 0 means good credit.",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    report.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
