#!/usr/bin/env python3
"""Read-only inventory for BabyLM 2026 EWoK evaluation material.

The script never writes files.  It prints JSON containing paths, counts, schemas,
and readiness checks, without printing EWoK item text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_FULL_LINES = {
    "agent-properties": 2210,
    "material-dynamics": 770,
    "material-properties": 170,
    "physical-dynamics": 120,
    "physical-interactions": 556,
    "physical-relations": 818,
    "quantitative-properties": 314,
    "social-interactions": 294,
    "social-properties": 328,
    "social-relations": 1548,
    "spatial-relations": 490,
}
EXPECTED_FAST_LINES = {name: 100 for name in EXPECTED_FULL_LINES}
REQUIRED_FIELDS = {
    "Domain",
    "ConceptA",
    "ConceptB",
    "ContextType",
    "ContextDiff",
    "TargetDiff",
    "Context1",
    "Context2",
    "Target1",
    "Target2",
}
UPSTREAM_PARQUET_BYTES = 110_797


def display_path(path: Path) -> str:
    path = path.resolve()
    workspace = Path("/workspace")
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return path.as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def inspect_jsonl_directory(path: Path, expected: dict[str, int]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    if not path.is_dir():
        return {"path": display_path(path), "exists": False, "ready": False, "files": files}

    for item in sorted(path.glob("*.jsonl")):
        record: dict[str, Any] = {
            "lines": count_lines(item),
            "bytes": item.stat().st_size,
            "sha256": sha256(item),
        }
        try:
            with item.open(encoding="utf-8") as stream:
                first = json.loads(next(stream))
            record["fields"] = sorted(first)
            record["schema_ok"] = REQUIRED_FIELDS.issubset(first)
        except (StopIteration, UnicodeDecodeError, json.JSONDecodeError) as error:
            record["schema_ok"] = False
            record["error"] = type(error).__name__
        files[item.stem] = record

    actual_counts = {name: info["lines"] for name, info in files.items()}
    schema_ok = all(info.get("schema_ok", False) for info in files.values())
    return {
        "path": display_path(path),
        "exists": True,
        "file_count": len(files),
        "line_total": sum(actual_counts.values()),
        "expected_line_total": sum(expected.values()),
        "counts_match": actual_counts == expected,
        "schema_ok": schema_ok,
        "ready": actual_counts == expected and schema_ok,
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-root",
        type=Path,
        default=Path("experiments/archive/initial_model_studies"),
        help="Collection root containing the recorded scientific inputs.",
    )
    parser.add_argument(
        "--strict-root",
        type=Path,
        default=Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict"),
        help="BabyLM strict evaluator root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = args.session_root.resolve()
    strict = args.strict_root.resolve()
    full = strict / "evaluation_data/full_eval/ewok_filtered"

    ewok_dirs = sorted(
        path
        for path in session.rglob("*")
        if path.is_dir() and path.name.lower() in {"ewok_fast", "ewok_filtered"}
    )
    fast_dirs = [path for path in ewok_dirs if path.name.lower() == "ewok_fast"]
    exact_size_matches = sorted(
        path
        for path in session.rglob("*")
        if path.is_file() and path.stat().st_size == UPSTREAM_PARQUET_BYTES
    )
    named_files = sorted(
        path
        for path in session.rglob("*")
        if path.is_file()
        and "ewok" in path.name.lower()
        and path.suffix.lower() in {".zip", ".parquet", ".arrow", ".jsonl", ".metadata"}
    )
    prediction_files = sorted(
        path
        for path in session.rglob("predictions.json")
        if "ewok" in path.as_posix().lower()
    )

    report = {
        "session_root": display_path(session),
        "strict_root": display_path(strict),
        "expected": {
            "upstream_test_rows_documented": 4374,
            "upstream_parquet_bytes": UPSTREAM_PARQUET_BYTES,
            "full_filtered_originals": sum(EXPECTED_FULL_LINES.values()) // 2,
            "full_evaluator_rows": sum(EXPECTED_FULL_LINES.values()),
            "full_rows_by_domain": EXPECTED_FULL_LINES,
            "fast_evaluator_rows": sum(EXPECTED_FAST_LINES.values()),
            "fast_rows_by_domain": EXPECTED_FAST_LINES,
        },
        "full": inspect_jsonl_directory(full, EXPECTED_FULL_LINES),
        "fast_candidates": [inspect_jsonl_directory(path, EXPECTED_FAST_LINES) for path in fast_dirs],
        "ewok_directories": [display_path(path) for path in ewok_dirs],
        "named_candidate_files": [
            {"path": display_path(path), "bytes": path.stat().st_size}
            for path in named_files
        ],
        "files_matching_upstream_parquet_size": [display_path(path) for path in exact_size_matches],
        "ewok_prediction_files": [display_path(path) for path in prediction_files],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
