#!/usr/bin/env python3
"""research: audit target-credit mass in real-stream objective comparisons.

The research/063 practical tests compare objectives on the same packed legal stream.
The key interpretive issue is that `correspondence_focus` makes source evidence
visible on qwen pairs but also changes relative target mass: qwen rows receive
view-word targets instead of ordinary WWM over source+view tokens. This script
computes update-level target allocations without training so that later learner
results can be interpreted as (i) better use of correspondence, (ii) reduced qwen
pressure / preservation, or (iii) insufficient focused credit.

It audits three tails by default:
  * research structural compact+topup with topups appended (the currently running run),
  * research structural compact+topup with topups interleaved,
  * research unchanged inherited source/current-rewrite segment tail.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import os
import pathlib
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import corrected_bridge_trainer as bridge  # noqa: E402
import real_stream_train as s62  # noqa: E402

TAILS = {
    "structural_appended": _public_path('experiments/archive/functional_learning/data/structural_policy_materialization/compact_structural_reinvest/compact_structural_reinvest_reference_tail_wordpaced_segments.jsonl'),
    "structural_interleaved": _public_path('experiments/archive/functional_learning/data/structural_policy_materialization/compact_structural_reinvest_interleaved/compact_structural_reinvest_interleaved_reference_tail_wordpaced_segments.jsonl'),
    "unchanged_inherited": _public_path('experiments/archive/functional_learning/data/unchanged_pair_segments_tail/reference_tail_unchanged_qwen_segments.jsonl'),
}
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/real_stream_credit_audit')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wc(text: str) -> int:
    return len((text or "").strip().split())


def iter_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_prefix_rows(path: pathlib.Path, max_updates: int, words_per_update: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total_words = 0
    max_words = int(max_updates) * int(words_per_update)
    info = Counter()
    source_rows = Counter()
    for row in iter_jsonl(path):
        if total_words >= max_words:
            break
        r = dict(row)
        w = int(r.get("words", wc(r.get("text", ""))))
        rows.append(r)
        total_words += w
        src = str(r.get("source", ""))
        source_rows[src] += 1
        if src == "qwen_pair_packed":
            info["qwen_rows"] += 1
            info["qwen_pair_segments"] += len(r.get("qwen_pair_segments") or [])
        if r.get("compact_modified_pair_ids"):
            info["compact_modified_rows"] += 1
            info["compact_modified_pair_occurrences"] += len(r.get("compact_modified_pair_ids") or [])
        if r.get("structural_compact_topup"):
            info["topup_rows"] += 1
    return rows, {
        "tail_jsonl": rel(path),
        "requested_max_updates": int(max_updates),
        "words_per_update": int(words_per_update),
        "prefix_rows": len(rows),
        "prefix_words": total_words,
        "target_prefix_words": max_words,
        "qwen_rows": int(info["qwen_rows"]),
        "qwen_pair_segments": int(info["qwen_pair_segments"]),
        "compact_modified_rows": int(info["compact_modified_rows"]),
        "compact_modified_pair_occurrences": int(info["compact_modified_pair_occurrences"]),
        "topup_rows": int(info["topup_rows"]),
        "source_rows": dict(source_rows),
    }


def row_target_counts(row: Dict[str, Any], tokenizer, wgb, seq_length: int, train_seed: int,
                      mask_prob: float, focus_prob: float, max_focus_groups_per_row: int) -> Dict[str, Any]:
    tok = bridge.tokenize_row(row, tokenizer, seq_length, wgb)
    is_qwen = row.get("source") == "qwen_pair_packed" and bool(row.get("qwen_pair_segments"))
    seed_wwm = s62.stable_seed("wwm", "ordinary_stream", int(train_seed), bridge.row_key(row))
    _masked, labels, mstats = bridge.apply_wwm_row(tok["input_ids"], tok["attention_mask"], tok["word_group"], tokenizer, seed_wwm, mask_prob)
    wwm_targets = int(mstats.get("n_target_tokens", int((labels != -100).sum().item())))
    out = {
        "is_qwen": bool(is_qwen),
        "is_compact_modified": bool(row.get("compact_modified_pair_ids")),
        "is_topup": bool(row.get("structural_compact_topup")),
        "source": str(row.get("source", "")),
        "row_words": int(row.get("words", wc(row.get("text", "")))),
        "wwm_targets": wwm_targets,
        "focus_targets": 0,
        "focus_candidate_groups": 0,
        "focus_selected_groups": 0,
        "focus_selected_pairs": 0,
        "focus_kind_counts": {},
    }
    if is_qwen:
        seed_focus = s62.stable_seed("view-focus", int(train_seed), bridge.row_key(row))
        _fm, flabels, fstats = s62.apply_view_focus_row(row, tok, tokenizer, seed_focus, focus_prob, max_focus_groups_per_row)
        out["focus_targets"] = int(fstats.get("n_target_tokens", int((flabels != -100).sum().item())))
        out["focus_candidate_groups"] = int(fstats.get("n_candidate_groups", 0))
        out["focus_selected_groups"] = int(fstats.get("n_selected_groups", 0))
        out["focus_selected_pairs"] = len(fstats.get("selected_pair_ids") or [])
        out["focus_kind_counts"] = dict(fstats.get("selected_candidate_kind_counts") or {})
    return out


def summarize_vals(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def audit_tail(name: str, path: pathlib.Path, args: argparse.Namespace, tokenizer) -> Dict[str, Any]:
    rows, prefix_info = load_prefix_rows(path, args.max_updates, args.words_per_update)
    wgb = bridge.WordGroupBuilder(tokenizer)
    out_dir = pathlib.Path(args.out_dir)
    per_update: List[Dict[str, Any]] = []
    kind_counts_total = Counter()
    row_cache: List[Dict[str, Any]] = []
    t0 = time.time()
    cursor = 0
    for update_i in range(int(args.max_updates)):
        macro_rows: List[Dict[str, Any]] = []
        words = 0
        while cursor < len(rows) and words < int(args.words_per_update):
            r = rows[cursor]
            macro_rows.append(r)
            words += int(r.get("words", wc(r.get("text", ""))))
            cursor += 1
        if not macro_rows:
            break
        st = Counter()
        focus_kind_counts = Counter()
        source_rows = Counter()
        source_targets_wwm = Counter()
        for r in macro_rows:
            c = row_target_counts(r, tokenizer, wgb, int(args.seq_length), int(args.train_seed), float(args.mask_prob), float(args.focus_prob), int(args.max_focus_groups_per_row))
            row_cache.append(c)
            source_rows[c["source"]] += 1
            source_targets_wwm[c["source"]] += int(c["wwm_targets"])
            st["rows"] += 1
            st["words"] += int(c["row_words"])
            st["wwm_targets_total"] += int(c["wwm_targets"])
            if c["is_qwen"]:
                st["qwen_rows"] += 1
                st["qwen_wwm_targets"] += int(c["wwm_targets"])
                st["qwen_focus_targets"] += int(c["focus_targets"])
                st["qwen_focus_candidate_groups"] += int(c["focus_candidate_groups"])
                st["qwen_focus_selected_groups"] += int(c["focus_selected_groups"])
                st["qwen_focus_selected_pair_refs"] += int(c["focus_selected_pairs"])
                focus_kind_counts.update(c.get("focus_kind_counts") or {})
            else:
                st["nonqwen_rows"] += 1
                st["nonqwen_wwm_targets"] += int(c["wwm_targets"])
            if c["is_compact_modified"]:
                st["compact_modified_rows"] += 1
                st["compact_modified_wwm_targets"] += int(c["wwm_targets"])
                st["compact_modified_focus_targets"] += int(c["focus_targets"])
            if c["is_topup"]:
                st["topup_rows"] += 1
                st["topup_wwm_targets"] += int(c["wwm_targets"])
        inherited_total = int(st["wwm_targets_total"])
        focus_total = int(st["nonqwen_wwm_targets"] + st["qwen_focus_targets"])
        rec = {
            "tail_name": name,
            "update": update_i + 1,
            "rows": int(st["rows"]),
            "words": int(st["words"]),
            "qwen_rows": int(st["qwen_rows"]),
            "compact_modified_rows": int(st["compact_modified_rows"]),
            "topup_rows": int(st["topup_rows"]),
            "inherited_wwm_total_targets": inherited_total,
            "inherited_wwm_qwen_targets": int(st["qwen_wwm_targets"]),
            "inherited_wwm_nonqwen_targets": int(st["nonqwen_wwm_targets"]),
            "inherited_wwm_qwen_target_fraction": float(st["qwen_wwm_targets"] / inherited_total) if inherited_total else 0.0,
            "inherited_wwm_compact_modified_targets": int(st["compact_modified_wwm_targets"]),
            "correspondence_focus_total_targets": focus_total,
            "correspondence_focus_qwen_targets": int(st["qwen_focus_targets"]),
            "correspondence_focus_nonqwen_targets": int(st["nonqwen_wwm_targets"]),
            "correspondence_focus_qwen_target_fraction": float(st["qwen_focus_targets"] / focus_total) if focus_total else 0.0,
            "correspondence_focus_candidate_groups": int(st["qwen_focus_candidate_groups"]),
            "correspondence_focus_selected_groups": int(st["qwen_focus_selected_groups"]),
            "correspondence_focus_selected_pair_refs": int(st["qwen_focus_selected_pair_refs"]),
            "correspondence_focus_compact_modified_targets": int(st["compact_modified_focus_targets"]),
            "target_ratio_focus_vs_wwm": float(focus_total / inherited_total) if inherited_total else 0.0,
            "qwen_target_ratio_focus_vs_wwm": float(st["qwen_focus_targets"] / st["qwen_wwm_targets"]) if st["qwen_wwm_targets"] else 0.0,
            "focus_selected_candidate_kind_counts": dict(focus_kind_counts),
            "source_rows": dict(source_rows),
            "source_targets_wwm": dict(source_targets_wwm),
        }
        kind_counts_total.update(focus_kind_counts)
        per_update.append(rec)
        if update_i == 0 or (update_i + 1) % 20 == 0:
            print(json.dumps({"event": "credit_audit_update", "tail": name, "update": update_i + 1, "elapsed_sec": round(time.time()-t0, 1), "wwm_qwen_frac": rec["inherited_wwm_qwen_target_fraction"], "focus_qwen_frac": rec["correspondence_focus_qwen_target_fraction"]}, ensure_ascii=False), flush=True)
    # CSV per tail
    csv_path = out_dir / f"{name}_per_update_credit.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tail_name", "update", "rows", "words", "qwen_rows", "compact_modified_rows", "topup_rows",
        "inherited_wwm_total_targets", "inherited_wwm_qwen_targets", "inherited_wwm_nonqwen_targets", "inherited_wwm_qwen_target_fraction", "inherited_wwm_compact_modified_targets",
        "correspondence_focus_total_targets", "correspondence_focus_qwen_targets", "correspondence_focus_nonqwen_targets", "correspondence_focus_qwen_target_fraction", "correspondence_focus_candidate_groups", "correspondence_focus_selected_groups", "correspondence_focus_selected_pair_refs", "correspondence_focus_compact_modified_targets",
        "target_ratio_focus_vs_wwm", "qwen_target_ratio_focus_vs_wwm",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in per_update:
            wr.writerow({k: r.get(k) for k in fieldnames})

    totals = Counter()
    for r in per_update:
        for k in [
            "rows", "words", "qwen_rows", "compact_modified_rows", "topup_rows",
            "inherited_wwm_total_targets", "inherited_wwm_qwen_targets", "inherited_wwm_nonqwen_targets", "inherited_wwm_compact_modified_targets",
            "correspondence_focus_total_targets", "correspondence_focus_qwen_targets", "correspondence_focus_nonqwen_targets", "correspondence_focus_candidate_groups", "correspondence_focus_selected_groups", "correspondence_focus_selected_pair_refs", "correspondence_focus_compact_modified_targets",
        ]:
            totals[k] += int(r.get(k, 0))
    wwm_total = int(totals["inherited_wwm_total_targets"])
    focus_total = int(totals["correspondence_focus_total_targets"])
    summary = {
        "tail_name": name,
        "tail_jsonl": rel(path),
        "prefix_info": prefix_info,
        "updates_audited": len(per_update),
        "totals": dict(totals),
        "aggregate_inherited_wwm_qwen_target_fraction": float(totals["inherited_wwm_qwen_targets"] / wwm_total) if wwm_total else 0.0,
        "aggregate_correspondence_focus_qwen_target_fraction": float(totals["correspondence_focus_qwen_targets"] / focus_total) if focus_total else 0.0,
        "aggregate_target_ratio_focus_vs_wwm": float(focus_total / wwm_total) if wwm_total else 0.0,
        "aggregate_qwen_target_ratio_focus_vs_wwm": float(totals["correspondence_focus_qwen_targets"] / totals["inherited_wwm_qwen_targets"]) if totals["inherited_wwm_qwen_targets"] else 0.0,
        "per_update_stats": {
            "inherited_wwm_qwen_target_fraction": summarize_vals([float(r["inherited_wwm_qwen_target_fraction"]) for r in per_update]),
            "correspondence_focus_qwen_target_fraction": summarize_vals([float(r["correspondence_focus_qwen_target_fraction"]) for r in per_update]),
            "qwen_target_ratio_focus_vs_wwm": summarize_vals([float(r["qwen_target_ratio_focus_vs_wwm"]) for r in per_update]),
            "target_ratio_focus_vs_wwm": summarize_vals([float(r["target_ratio_focus_vs_wwm"]) for r in per_update]),
        },
        "focus_selected_candidate_kind_counts": dict(kind_counts_total),
        "per_update_csv": rel(csv_path),
        "interpretation": "Credit audit only: it measures objective target allocation on the real-stream prefix, not learner outcomes. Use with common-target and Cheap7 results to distinguish correspondence use from reduced qwen pressure.",
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--max-updates", type=int, default=80)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--train-seed", type=int, default=62062)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--focus-prob", type=float, default=0.35)
    ap.add_argument("--max-focus-groups-per-row", type=int, default=16)
    ap.add_argument("--tails", nargs="*", default=list(TAILS.keys()), choices=list(TAILS.keys()))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    summaries = []
    t0 = time.time()
    for name in args.tails:
        summaries.append(audit_tail(name, TAILS[name], args, tokenizer))
    final = {
        "status": "REAL_STREAM_CREDIT_AUDIT_DONE",
        "created_utc": now(),
        "out_dir": rel(args.out_dir),
        "parameters": {
            "max_updates": int(args.max_updates),
            "words_per_update": int(args.words_per_update),
            "seq_length": int(args.seq_length),
            "train_seed": int(args.train_seed),
            "mask_prob": float(args.mask_prob),
            "focus_prob": float(args.focus_prob),
            "max_focus_groups_per_row": int(args.max_focus_groups_per_row),
        },
        "tail_summaries": summaries,
        "elapsed_sec": round(time.time() - t0, 1),
        "scientific_use": "Interpret pending and future real-stream results: source-visible focus protects source evidence but changes qwen loss mass; unchanged-tail control distinguishes existing-correspondence exploitation from generated compaction.",
    }
    (args.out_dir / "credit_audit_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
