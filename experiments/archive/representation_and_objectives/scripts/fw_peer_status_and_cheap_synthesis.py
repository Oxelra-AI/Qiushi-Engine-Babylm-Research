#!/usr/bin/env python3
"""Summarize FW compact/breadth completion and available cheap eval.

This is CPU-only evidence reading. It does not launch evaluation or training. It guards
against over-reading partial JSON files by recording per-target completeness and by
separating within-family compact-vs-breadth deltas from absolute progress versus the
known legal reference trajectory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
A01_WS = ROOT / "experiments/archive/representation_and_objectives"
A02_WS = ROOT / "experiments/archive/frontier_consolidation"
OUT = A01_WS / "data/fw_peer_status"
NOTE = A01_WS / "notes/fw_peer_status_and_cheap_synthesis.md"

RUNS = {
    "compact_view": {
        "run_dir": A02_WS / "training/runs/fw_compact_view_shared16k_seed43022",
        "stream_sha256": "c8d7f24b5edd2dad21f589c8cde72671d0b76038d9632fcd3d6c79178a24be68",
        "role": "same-source compact rewrite companion",
    },
    "source_breadth_rowblock": {
        "run_dir": A02_WS / "training/runs/fw_source_breadth_shared16k_seed43022",
        "stream_sha256": "1b98269fb210cc9494885ec47ee26d9fd1b6ead60a1c308f5d9fd1b6ead60a1c308f5d9c47d9b385731d".replace("5d9fd1b6ead60a1c308f", "5d9f"),
        "role": "whole-sentence independent-source breadth companion, row-block layout",
    },
}
# Correct literal for row-block stream (kept outside the long string above for readability).
RUNS["source_breadth_rowblock"]["stream_sha256"] = "1b98269fb210cc9494885ec47ee26d9fd1b6ead60a1c308f5d9c47d9b385731d"

EXPECTED_TOKENIZER_SHA = "e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366"
CHEAP_ROOT = A02_WS / "data/fw_comparison_eval/per_target"
# The reference note gives cheap7 scores on a different legal coordinate.
LEGAL_REFERENCE_CHEAP7 = {"70M": 42.6086, "80M": 42.9486, "100M": 43.0057}
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RAW_GPIQA = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                last = line
    if last is None:
        return None
    try:
        return json.loads(last)
    except Exception:
        return {"raw_tail": last[-1000:]}


def summarize_run(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(spec["run_dir"])
    hf = run_dir / "hf_model"
    metrics_path = run_dir / "scientific_metrics.json"
    metrics = read_json(metrics_path)
    checkpoints = sorted([p.name for p in hf.iterdir() if p.is_dir()], key=lambda x: (len(x), x)) if hf.exists() else []
    tok_sha = sha256_file(hf / "tokenizer.json")
    chck100 = hf / "chck_100M"
    expected_ladder = [f"chck_{i}M" for i in range(1, 101)]
    missing = [c for c in expected_ladder if c not in checkpoints]
    msum = {}
    if isinstance(metrics, dict):
        saved = [x.get("name") for x in metrics.get("saved_checkpoints", []) if isinstance(x, dict)]
        msum = {
            "variant": metrics.get("variant"),
            "model_family": metrics.get("model_family"),
            "parameter_count": metrics.get("parameter_count"),
            "vocab_size": metrics.get("vocab_size"),
            "tokenizer_label": metrics.get("tokenizer_label"),
            "word_exposure": metrics.get("word_exposure"),
            "actual_training_steps": metrics.get("actual_training_steps"),
            "loss_first": metrics.get("loss_first"),
            "loss_last": metrics.get("loss_last"),
            "seed": metrics.get("seed"),
            "hidden_size": metrics.get("hidden_size"),
            "n_layer": metrics.get("n_layer"),
            "n_head": metrics.get("n_head"),
            "seq_length": metrics.get("seq_length"),
            "saved_checkpoint_count": len(saved),
            "saved_checkpoint_first_last": [saved[:3], saved[-3:]],
        }
    errors = []
    if not chck100.exists(): errors.append("missing chck_100M")
    if missing: errors.append(f"missing {len(missing)} chck_1M..chck_100M checkpoints")
    if tok_sha != EXPECTED_TOKENIZER_SHA: errors.append(f"tokenizer_sha {tok_sha} != expected {EXPECTED_TOKENIZER_SHA}")
    if msum.get("word_exposure") != 100_000_000: errors.append(f"word_exposure {msum.get('word_exposure')} != 100000000")
    if msum.get("vocab_size") != 16_384: errors.append(f"vocab_size {msum.get('vocab_size')} != 16384")
    return {
        "name": name,
        "role": spec["role"],
        "run_dir": str(run_dir),
        "hf_model": str(hf),
        "chck_100M_exists": chck100.exists(),
        "checkpoint_dir_count": len(checkpoints),
        "missing_1M_ladder": missing,
        "tokenizer_sha256": tok_sha,
        "expected_tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
        "metrics_path_exists": metrics_path.exists(),
        "metrics_summary": msum,
        "last_training_record": last_jsonl(run_dir / "training_log.jsonl"),
        "status": "READY" if not errors else "NOT_READY",
        "errors": errors,
    }


def safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def cheap_json_path(arm: str, checkpoint: str) -> Path:
    tag = "compact_view" if arm == "compact_view" else "source_breadth"
    return CHEAP_ROOT / f"step086_{tag}_chck_{checkpoint}.json"


def parse_scores(rec: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    official = rec.get("official_overall") if isinstance(rec, dict) else None
    if isinstance(official, dict) and isinstance(official.get("scores"), dict):
        for k, v in official["scores"].items():
            fv = safe_float(v)
            if fv is not None:
                scores[k] = fv
    tasks = rec.get("tasks") if isinstance(rec, dict) else None
    if isinstance(tasks, dict):
        for k, t in tasks.items():
            if not isinstance(t, dict):
                continue
            if k == "Reading" and isinstance(t.get("scores"), dict):
                fv = safe_float(t["scores"].get("Reading"))
                if fv is not None: scores["Reading"] = fv
            else:
                fv = safe_float(t.get("score"))
                if fv is not None: scores[k] = fv
    if "GlobalPIQA" not in scores:
        gp = [scores.get(k) for k in RAW_GPIQA if scores.get(k) is not None]
        if len(gp) == 2:
            scores["GlobalPIQA"] = sum(gp) / 2.0
    return scores


def summarize_cheap_eval() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    for ckpt in ["70M", "80M", "100M"]:
        for arm in ["compact_view", "source_breadth_rowblock"]:
            p = cheap_json_path(arm, ckpt)
            rec = read_json(p)
            key = f"{arm}_{ckpt}"
            raw[key] = {"path": str(p), "exists": p.exists(), "json_ok": isinstance(rec, dict), "record": rec if isinstance(rec, dict) else None}
            scores = parse_scores(rec) if isinstance(rec, dict) else {}
            n = len([c for c in CHEAP_COLUMNS if c in scores])
            cheap7 = sum(scores[c] for c in CHEAP_COLUMNS if c in scores) / 7.0 if n == 7 else None
            rows.append({
                "arm": arm,
                "checkpoint": ckpt,
                "path": str(p),
                "exists": p.exists(),
                "scores_available": n,
                "cheap7_complete": n == 7,
                "cheap7": cheap7,
                **{c: scores.get(c) for c in CHEAP_COLUMNS},
                "SuperGLUE": scores.get("SuperGLUE"),
                "AoA": scores.get("AoA"),
                "finished_utc": rec.get("finished_utc") if isinstance(rec, dict) else None,
            })
        # paired deltas if possible
        row_c = next(r for r in rows if r["arm"] == "compact_view" and r["checkpoint"] == ckpt)
        row_b = next(r for r in rows if r["arm"] == "source_breadth_rowblock" and r["checkpoint"] == ckpt)
        common_cols = [c for c in CHEAP_COLUMNS if row_c.get(c) is not None and row_b.get(c) is not None]
        d = {"checkpoint": ckpt, "common_columns": len(common_cols), "common_column_names": common_cols}
        for c in common_cols:
            d[f"compact_minus_breadth_{c}"] = row_c[c] - row_b[c]
        if len(common_cols) == 7:
            d["compact_minus_breadth_cheap7"] = row_c["cheap7"] - row_b["cheap7"]
            ref = LEGAL_REFERENCE_CHEAP7.get(ckpt)
            if ref is not None:
                d["compact_minus_legal_ref_cheap7"] = row_c["cheap7"] - ref
                d["breadth_minus_legal_ref_cheap7"] = row_b["cheap7"] - ref
        deltas.append(d)
    return rows, deltas, raw


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_status = {k: summarize_run(k, v) for k, v in RUNS.items()}
    cheap_rows, cheap_deltas, raw = summarize_cheap_eval()
    status = {
        "status": "FW_PEER_STATUS_SYNTHESIS",
        "created_utc": now(),
        "training_status": run_status,
        "cheap_eval_rows": cheap_rows,
        "cheap_eval_deltas": cheap_deltas,
        "reference_cheap7": LEGAL_REFERENCE_CHEAP7,
        "interpretation": {
            "completed_training": all(r["status"] == "READY" for r in run_status.values()),
            "cheap_eval_complete_checkpoints": [d["checkpoint"] for d in cheap_deltas if d.get("compact_minus_breadth_cheap7") is not None],
            "warning": "partial cheap-eval files are listed but not used for cheap7 mechanism decisions until all seven cheap columns exist for both arms at a checkpoint",
        },
    }
    (OUT / "fw_peer_status_and_cheap_synthesis.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Avoid embedding huge in-progress records in CSV; write lean rows.
    with (OUT / "fw_peer_cheap_scores.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["arm", "checkpoint", "exists", "scores_available", "cheap7_complete", "cheap7", *CHEAP_COLUMNS, "finished_utc", "path"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows([{k: r.get(k) for k in fieldnames} for r in cheap_rows])
    with (OUT / "fw_peer_cheap_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        all_fields = ["checkpoint", "common_columns", "common_column_names", *[f"compact_minus_breadth_{c}" for c in CHEAP_COLUMNS], "compact_minus_breadth_cheap7", "compact_minus_legal_ref_cheap7", "breadth_minus_legal_ref_cheap7"]
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader(); w.writerows([{k: d.get(k) for k in all_fields} for d in cheap_deltas])
    # Human note.
    lines = []
    lines.append("# research — A02 FW peer training and cheap-eval status\n")
    lines.append("## Training artifacts\n")
    for name, rec in run_status.items():
        m = rec.get("metrics_summary") or {}
        lines.append(f"- **{name}** ({rec['role']}): status `{rec['status']}`, chck_100M={rec['chck_100M_exists']}, checkpoint dirs={rec['checkpoint_dir_count']}, tokenizer SHA `{(rec.get('tokenizer_sha256') or '')[:16]}`; word exposure={m.get('word_exposure')}, steps={m.get('actual_training_steps')}, loss {m.get('loss_first')} → {m.get('loss_last')}. Errors: {rec['errors']}")
    lines.append("\n## Available cheap-eval files\n")
    for r in cheap_rows:
        val = "NA" if r["cheap7"] is None else f"{r['cheap7']:.4f}"
        lines.append(f"- {r['checkpoint']} {r['arm']}: {r['scores_available']}/7 cheap columns, cheap7={val}, path `{r['path']}`")
    lines.append("\n## Paired deltas now safe to read\n")
    any_full = False
    for d in cheap_deltas:
        if d.get("compact_minus_breadth_cheap7") is None:
            lines.append(f"- {d['checkpoint']}: only {d['common_columns']} common cheap columns ({', '.join(d['common_column_names'])}); not a cheap7 mechanism decision yet.")
        else:
            any_full = True
            lines.append(f"- {d['checkpoint']}: compact-breadth cheap7 = {d['compact_minus_breadth_cheap7']:+.4f}; compact-ref = {d.get('compact_minus_legal_ref_cheap7'):+.4f}; breadth-ref = {d.get('breadth_minus_legal_ref_cheap7'):+.4f}.")
    if not any_full:
        lines.append("\nNo checkpoint currently has complete paired cheap7 evidence in the visible files. The partial 70M rows can show early column directions, but should not trigger full official evaluation or a new training route by themselves.")
    lines.append("\n## Evidence files\n")
    lines.append(f"- JSON: `{OUT / 'fw_peer_status_and_cheap_synthesis.json'}`")
    lines.append(f"- Score CSV: `{OUT / 'fw_peer_cheap_scores.csv'}`")
    lines.append(f"- Delta CSV: `{OUT / 'fw_peer_cheap_deltas.csv'}`")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status["status"], "completed_training": status["interpretation"]["completed_training"], "cheap_eval_complete_checkpoints": status["interpretation"]["cheap_eval_complete_checkpoints"], "json": str(OUT / "fw_peer_status_and_cheap_synthesis.json"), "note": str(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
