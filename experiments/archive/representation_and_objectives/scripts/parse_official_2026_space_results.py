#!/usr/bin/env python3
"""research: parse BabyLM 2026 public Space result datasets with the Space code.

Uses public HF datasets declared by the downloaded Space:
  BabyLM-community/leaderboard-2026-all-results
  BabyLM-community/leaderboard-2026-all-requests
  BabyLM-community/leaderboard-datasets
If accessible, imports the Space's own get_leaderboard_df and writes Strict-Small rows.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

OUT = Path("experiments/archive/representation_and_objectives/data/official_leaderboard_refresh")
SPACE_DIR = OUT / "space_BabyLM-Leaderboard-2026"
RESULTS_REPO = "BabyLM-community/leaderboard-2026-all-results"
QUEUE_REPO = "BabyLM-community/leaderboard-2026-all-requests"
DATASETS_REPO = "BabyLM-community/leaderboard-datasets"
RESULTS_DIR = OUT / "official_results_dataset"
QUEUE_DIR = OUT / "official_requests_dataset"
DATASETS_DIR = OUT / "official_datasets_dataset"
OUT_JSON = OUT / "official_2026_strict_small_table.json"
OUT_CSV = OUT / "official_2026_strict_small_table.csv"
OUT_MD = OUT / "official_2026_strict_small_table.md"


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except Exception:
        return str(p)


def snapshot(repo_id: str, repo_type: str, local_dir: Path, errors: list[dict[str, Any]]) -> dict[str, Any] | None:
    from huggingface_hub import snapshot_download
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        p = snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(local_dir), tqdm_class=None, etag_timeout=30)
        files = [q for q in Path(p).rglob("*") if q.is_file() and ".git" not in q.parts]
        return {"repo_id": repo_id, "repo_type": repo_type, "local_dir": rel(local_dir), "file_count": len(files), "total_bytes": sum(q.stat().st_size for q in files), "files_sample": [rel(q) for q in files[:50]]}
    except Exception as e:
        errors.append({"repo_id": repo_id, "repo_type": repo_type, "error": repr(e)})
        return None


def import_space_modules():
    if not SPACE_DIR.exists():
        raise RuntimeError(f"missing downloaded Space dir: {SPACE_DIR}")
    # Put Space root first so imports resolve to downloaded source.
    sys.path.insert(0, str(SPACE_DIR.resolve()))
    # Ensure envs uses our downloaded snapshots rather than HF_HOME globals if app code imports it.
    os.environ.setdefault("HF_HOME", str(OUT / "hf_home_dummy"))
    from src.display.utils import COLS, BENCHMARK_COLS, COLS_MULTILINGUAL, BENCHMARK_COLS_MULTILINGUAL  # type: ignore
    from src.populate import get_leaderboard_df  # type: ignore
    return get_leaderboard_df, COLS, BENCHMARK_COLS, COLS_MULTILINGUAL, BENCHMARK_COLS_MULTILINGUAL


def df_to_rows(df) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df is None or getattr(df, "empty", True):
        return rows
    for _, r in df.iterrows():
        row = {}
        for k, v in r.to_dict().items():
            try:
                if hasattr(v, "item"):
                    v = v.item()
            except Exception:
                pass
            if str(v) == "nan":
                v = None
            row[str(k)] = v
        rows.append(row)
    return rows


def parse_with_space_code(errors: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"space_code_status": "not_run"}
    try:
        get_leaderboard_df, COLS, BENCHMARK_COLS, COLS_MULTILINGUAL, BENCHMARK_COLS_MULTILINGUAL = import_space_modules()
        df = get_leaderboard_df(str(RESULTS_DIR), str(QUEUE_DIR), COLS, BENCHMARK_COLS)
        cols = list(df.columns) if df is not None else []
        all_rows = df_to_rows(df)
        strict_small = [r for r in all_rows if str(r.get("Track", "")).lower() == "strict-small"]
        out.update({
            "space_code_status": "ok",
            "all_non_multilingual_rows": len(all_rows),
            "strict_small_rows": len(strict_small),
            "columns": cols,
            "rows": strict_small,
        })
    except Exception as e:
        errors.append({"space_code_parse_error": repr(e)})
        out.update({"space_code_status": "error", "error": repr(e)})
    return out


def fallback_parse_json(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for p in RESULTS_DIR.rglob("*.json"):
        if ".git" in p.parts:
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            config = obj.get("config", {}) if isinstance(obj, dict) else {}
            track = obj.get("track") or config.get("track") or ""
            if str(track).lower() != "strict-small":
                continue
            results = obj.get("results", {}) if isinstance(obj, dict) else {}
            # mimic likely Space scoring: store task means as fractions in results and metadata in config.
            row = {"source_file": rel(p), "Track": track, "Model": config.get("model_name") or config.get("hf_repo") or p.stem, "HF Repo": config.get("hf_repo")}
            # Preserve top-level aggregates if already present.
            for key in ["Overall Average", "NLP Average", "Human-like Average"]:
                if key in obj:
                    row[key] = obj[key]
                elif key.lower().replace(" ", "_") in obj:
                    row[key] = obj[key.lower().replace(" ", "_")]
            row["raw_keys"] = sorted(obj.keys())
            row["result_keys"] = sorted(results.keys()) if isinstance(results, dict) else []
            rows.append(row)
        except Exception as e:
            errors.append({"fallback_parse_error": repr(e), "path": rel(p)})
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = []
    for preferred in ["Model", "HF Repo", "Track", "Overall Average", "NLP Average", "Human-like Average", "BLiMP", "BLiMP Supplement", "EWoK", "Entity Tracking", "COMPS", "GlobalPIQA", "(Super)GLUE", "Reading", "AoA", "source_file"]:
        if any(preferred in r for r in rows):
            keys.append(preferred)
    for r in rows:
        for k in r:
            if k not in keys and k != "raw_keys" and k != "result_keys":
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []
    snapshots = []
    for repo, local in [(RESULTS_REPO, RESULTS_DIR), (QUEUE_REPO, QUEUE_DIR), (DATASETS_REPO, DATASETS_DIR)]:
        s = snapshot(repo, "dataset", local, errors)
        if s:
            snapshots.append(s)
    parsed = parse_with_space_code(errors)
    rows = parsed.get("rows") if isinstance(parsed.get("rows"), list) else []
    fallback_rows = [] if rows else fallback_parse_json(errors)
    final_rows = rows or fallback_rows
    # Sort if possible by Overall Average, descending.
    def sort_key(r: dict[str, Any]):
        v = r.get("Overall Average")
        try:
            return -float(v)
        except Exception:
            return 0.0
    final_rows = sorted(final_rows, key=sort_key)
    highest = final_rows[0] if final_rows else None
    record = {
        "status": "OFFICIAL_2026_STRICT_SMALL_TABLE",
        "elapsed_sec": round(time.time() - t0, 3),
        "repos": {"results": RESULTS_REPO, "requests": QUEUE_REPO, "datasets": DATASETS_REPO},
        "snapshots": snapshots,
        "errors": errors,
        "space_code_parse": parsed,
        "fallback_rows_count": len(fallback_rows),
        "strict_small_rows": final_rows,
        "highest_row": highest,
        "paths": {"json": rel(OUT_JSON), "csv": rel(OUT_CSV), "note": rel(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    if final_rows:
        write_csv(final_rows, OUT_CSV)
    lines = [
        "# research official 2026 Strict-Small table refresh",
        "",
        f"Results repo: `{RESULTS_REPO}`; requests repo: `{QUEUE_REPO}`.",
        f"Space-code parse status: `{parsed.get('space_code_status')}`.",
        f"Strict-Small rows: `{len(final_rows)}`.",
    ]
    if highest:
        lines.append(f"Highest parsed row: `{highest.get('Model')}` Overall `{highest.get('Overall Average')}` NLP `{highest.get('NLP Average')}`.")
    else:
        lines.append("No Strict-Small rows were parsed from public result snapshots; use the top model card and Space source as current evidence, and refresh again if needed.")
    if errors:
        lines.append("")
        lines.append("Errors/sample:")
        for e in errors[:8]:
            lines.append(f"- `{e}`")
    lines += ["", f"JSON: `{rel(OUT_JSON)}`", f"CSV: `{rel(OUT_CSV)}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "rows": len(final_rows), "highest": highest, "out_json": rel(OUT_JSON), "note": rel(OUT_MD)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
