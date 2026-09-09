#!/usr/bin/env python3
"""Trainer-exact token and WWM-group exposure measurement for density arms.

Measures the frozen 10M pools using the same portable tokenizer and word-start
logic as the COMPACT_EXPERIENCE/REPRESENTATION_FRONTIER_STUDIES training recipe.  This is a CPU measurement of the data
seen by the trainer, not a model evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import statistics
import sys
from typing import Any, Iterable

import torch

USER_ROOT = _public_path('.')
COMPACT_EXPERIENCE_SCRIPTS = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))
from masking_curriculum_trainer import make_portable_tokenizer, is_word_start  # noqa: E402

WORKSPACE = pathlib.Path("experiments/archive/frontier_consolidation")
DATA_DIR = WORKSPACE / "data" / "density_cleanqwen_overlay_medium_riskhard"
META = DATA_DIR / "density_cleanqwen_rowholdout_overlay_metadata.json"
TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
OUT_DIR_DEFAULT = WORKSPACE / "data" / "trainer_exact_token_exposure_measurement"
NOTE_DEFAULT = (_PUBLIC_ROOT / 'research/notes/frontier_consolidation/trainer_exact_token_exposure_measurement.md')
SEQ = 256
MASK_PROB = 0.15
PASSES = 10

ARM_TO_FILE = {
    "near_repeat": "cleanqwen_fineweb_repeat_near_core_10M.jsonl",
    "near_view": "cleanqwen_fineweb_near_view_core_10M.jsonl",
    "compact_repeat_core": "cleanqwen_fineweb_repeat_compact_core_neutral_10M.jsonl",
    "compact_view_core": "cleanqwen_fineweb_compact_view_core_neutral_10M.jsonl",
    "compact_view_reinvest": "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl",
    "compact_repeat_reinvest": "cleanqwen_fineweb_repeat_compact_reinvest_10M.jsonl",
    "lengthmatched_compact_core": "cleanqwen_lengthmatched_compact_core_neutral_10M.jsonl",
    "lengthmatched_reinvest": "cleanqwen_lengthmatched_compact_reinvest_10M.jsonl",
}


def iter_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ys[lo])
    return float(ys[lo] * (hi - pos) + ys[hi] * (pos - lo))


def summarize_list(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "min": min(xs),
        "p05": percentile(xs, 0.05),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p95": percentile(xs, 0.95),
        "max": max(xs),
        "sum": sum(xs),
    }


def measure_text(tokenizer, text: str, special_ids: set[int]) -> dict[str, Any]:
    enc_full = tokenizer(text, add_special_tokens=False, truncation=False, return_tensors=None)
    ids_full = enc_full["input_ids"]
    if isinstance(ids_full[0] if ids_full else [], list):
        ids_full = ids_full[0]
    full_len = len(ids_full)
    ids = ids_full[:SEQ]
    visible = len(ids)
    candidate = [tid for tid in ids if int(tid) not in special_ids]
    groups = 0
    prev = False
    for i, tid in enumerate(ids):
        tid = int(tid)
        if tid in special_ids:
            prev = False
            continue
        if (not prev) or is_word_start(str(tokenizer.convert_ids_to_tokens(tid))) or i == 0:
            groups += 1
        prev = True
    return {
        "tokens_untruncated": full_len,
        "tokens_visible": visible,
        "candidate_tokens_visible": len(candidate),
        "wwm_groups_visible": groups,
        "truncated_tokens": max(0, full_len - SEQ),
        "over_seq256": full_len > SEQ,
    }


def measure_arm(tokenizer, arm: str, path: pathlib.Path, sample_rows: int) -> dict[str, Any]:
    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    sums = {
        "rows": 0,
        "word_total": 0,
        "tokens_untruncated": 0,
        "tokens_visible": 0,
        "candidate_tokens_visible": 0,
        "wwm_groups_visible": 0,
        "truncated_tokens": 0,
        "over_seq256_rows": 0,
    }
    per_word_visible: list[float] = []
    per_word_candidates: list[float] = []
    per_word_groups: list[float] = []
    token_word_deltas: list[float] = []
    example_rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(iter_jsonl(path)):
        text = str(rec.get("text", ""))
        words = int(rec.get("words") or len(text.split()))
        m = measure_text(tokenizer, text, special_ids)
        sums["rows"] += 1
        sums["word_total"] += words
        for k in ["tokens_untruncated", "tokens_visible", "candidate_tokens_visible", "wwm_groups_visible", "truncated_tokens"]:
            sums[k] += int(m[k])
        if m["over_seq256"]:
            sums["over_seq256_rows"] += 1
        if words > 0:
            per_word_visible.append(m["tokens_visible"] / words)
            per_word_candidates.append(m["candidate_tokens_visible"] / words)
            per_word_groups.append(m["wwm_groups_visible"] / words)
            token_word_deltas.append(m["tokens_visible"] - words)
        if len(example_rows) < sample_rows:
            ex = {"row_index": idx, "words": words, **m, "text_prefix": text[:220]}
            example_rows.append(ex)
    sums["expected_masked_tokens_per_epoch_token_level_0p15"] = sums["candidate_tokens_visible"] * MASK_PROB
    sums["expected_masked_wwm_groups_per_epoch_0p15"] = sums["wwm_groups_visible"] * MASK_PROB
    sums["ten_pass_candidate_tokens_visible"] = sums["candidate_tokens_visible"] * PASSES
    sums["ten_pass_wwm_groups_visible"] = sums["wwm_groups_visible"] * PASSES
    sums["ten_pass_expected_masked_wwm_groups_0p15"] = sums["expected_masked_wwm_groups_per_epoch_0p15"] * PASSES
    return {
        "arm": arm,
        "path": str(path),
        **sums,
        "tokens_visible_per_word_stats": summarize_list(per_word_visible),
        "candidate_tokens_visible_per_word_stats": summarize_list(per_word_candidates),
        "wwm_groups_visible_per_word_stats": summarize_list(per_word_groups),
        "tokens_visible_minus_words_per_row_stats": summarize_list(token_word_deltas),
        "sample_rows": example_rows,
    }


def contrast(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rows", "word_total", "tokens_untruncated", "tokens_visible", "candidate_tokens_visible",
        "wwm_groups_visible", "truncated_tokens", "over_seq256_rows",
        "expected_masked_tokens_per_epoch_token_level_0p15", "expected_masked_wwm_groups_per_epoch_0p15",
        "ten_pass_candidate_tokens_visible", "ten_pass_wwm_groups_visible", "ten_pass_expected_masked_wwm_groups_0p15",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        av = a.get(k)
        bv = b.get(k)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            out[k] = {
                "b_minus_a": bv - av,
                "relative_to_a": None if av == 0 else (bv - av) / av,
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["near_repeat", "near_view", "compact_repeat_core", "compact_view_core", "compact_view_reinvest", "compact_repeat_reinvest", "lengthmatched_compact_core"])
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--note", default=str(NOTE_DEFAULT))
    ap.add_argument("--sample-rows", type=int, default=3)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = make_portable_tokenizer(TOKENIZER)
    rows: dict[str, dict[str, Any]] = {}
    for arm in args.arms:
        path = DATA_DIR / ARM_TO_FILE[arm]
        rec = measure_arm(tokenizer, arm, path, args.sample_rows)
        rows[arm] = rec
        (out_dir / f"{arm}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"event": "measured", "arm": arm, "rows": rec["rows"], "words": rec["word_total"], "candidate_tokens": rec["candidate_tokens_visible"], "wwm_groups": rec["wwm_groups_visible"], "over_seq256_rows": rec["over_seq256_rows"]}), flush=True)

    contrasts: dict[str, Any] = {}
    pairs = [
        ("near_view_minus_near_repeat", "near_repeat", "near_view"),
        ("compact_view_core_minus_compact_repeat_core", "compact_repeat_core", "compact_view_core"),
        ("compact_view_reinvest_minus_compact_view_core", "compact_view_core", "compact_view_reinvest"),
        ("compact_repeat_reinvest_minus_compact_repeat_core", "compact_repeat_core", "compact_repeat_reinvest"),
        ("compact_view_core_minus_lengthmatched_compact_core", "lengthmatched_compact_core", "compact_view_core"),
    ]
    for name, a, b in pairs:
        if a in rows and b in rows:
            contrasts[name] = contrast(rows[a], rows[b])
    payload = {
        "status": "TRAINER_EXACT_TOKEN_EXPOSURE_MEASUREMENT",
        "scope": "Frozen 10M pools for research medium risk-hard clean-Qwen row-holdout overlay; ten-pass training exposure is an exact 10x multiplier.",
        "tokenizer": str(TOKENIZER),
        "metadata": str(META),
        "seq_length": SEQ,
        "mask_prob": MASK_PROB,
        "pass_count": PASSES,
        "arms": rows,
        "contrasts": contrasts,
    }
    summary_path = out_dir / "trainer_exact_token_exposure_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def pct(x: Any) -> str:
        return "NA" if x is None else f"{100*float(x):+.3f}%"

    lines = [
        "# research trainer-exact token and WWM-group exposure measurement",
        "",
        f"Summary JSON: `{summary_path}`",
        f"Tokenizer: `{TOKENIZER}`; seq length {SEQ}; WWM probability {MASK_PROB}; ten-pass exposure is exact 10x.",
        "",
        "## Pool totals",
        "",
        "| arm | words | rows | visible candidate tokens | visible WWM groups | over-seq256 rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, rec in rows.items():
        lines.append(f"| {arm} | {rec['word_total']} | {rec['rows']} | {rec['candidate_tokens_visible']} | {rec['wwm_groups_visible']} | {rec['over_seq256_rows']} |")
    lines.extend(["", "## Key contrasts", ""])
    for name, rec in contrasts.items():
        cand = rec.get("candidate_tokens_visible", {})
        groups = rec.get("wwm_groups_visible", {})
        over = rec.get("over_seq256_rows", {})
        lines.append(f"- {name}: candidate tokens {cand.get('b_minus_a')} ({pct(cand.get('relative_to_a'))}), WWM groups {groups.get('b_minus_a')} ({pct(groups.get('relative_to_a'))}), over-seq256 rows {over.get('b_minus_a')}.")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Use this measurement to separate semantic/view effects from raw token or WWM-group exposure differences when reading the research compact-core task gains.",
    ])
    note = pathlib.Path(args.note)
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary": str(summary_path), "note": str(note)}, indent=2))


if __name__ == "__main__":
    main()
