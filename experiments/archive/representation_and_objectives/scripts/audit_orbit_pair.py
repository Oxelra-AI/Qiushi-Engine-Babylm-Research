#!/usr/bin/env python3
"""research audit of research deterministic token-matched orbit pair.

The research language-scale pair was cancelled before interpretation because the
`stable` arm may not implement an exact corpus-global key->alias mapping.  This
audit reads the produced 10M corpora and independently replays the deterministic
constructor to measure:

  * whether the per_row and stable arms transform the same row/key occurrence
    support;
  * how many aliases each original lexical key receives in the stable arm;
  * whether the actual output files match the replayed transform;
  * how much of the changed support would be retained by a strictly stable
    global mapping if one alias per key is required to work in every supported
    row.

It writes a JSON and Markdown summary under data/orbit_pair_audit/.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path("experiments/archive/representation_and_objectives")
DEFAULT_STEP259_SCRIPT = ROOT / "scripts/deterministic_token_matched_entity_orbit.py"
DEFAULT_ORIG10 = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
DEFAULT_PER10 = ROOT / "data/deterministic_token_matched_entity_orbit/det_tokenmatched_orbit_per_row_10M.jsonl"
DEFAULT_STABLE10 = ROOT / "data/deterministic_token_matched_entity_orbit/det_tokenmatched_orbit_stable_10M.jsonl"
DEFAULT_TOKENIZER = Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
DEFAULT_OUT = ROOT / "data/orbit_pair_audit"


def load_mod(path: Path):
    spec = importlib.util.spec_from_file_location("orbit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orbit"] = mod
    spec.loader.exec_module(mod)
    return mod


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def entropy(counter: Counter) -> float:
    n = sum(counter.values())
    if n <= 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=str(DEFAULT_STEP259_SCRIPT))
    ap.add_argument("--orig10", default=str(DEFAULT_ORIG10))
    ap.add_argument("--per10", default=str(DEFAULT_PER10))
    ap.add_argument("--stable10", default=str(DEFAULT_STABLE10))
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit_rows", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=10000)
    ap.add_argument("--max_examples", type=int, default=30)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = load_mod(Path(args.script))
    tok = mod.AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    bank, eligible_keys, bank_meta = mod.mine_alias_bank(Path(args.orig10), tok, min_freq=2, max_per_profile=512, max_candidate_freq=2000)

    counters = Counter()
    support_diff_examples: List[dict] = []
    mismatch_examples: List[dict] = []
    stable_alias_by_key: Dict[str, Counter] = defaultdict(Counter)
    per_alias_by_key: Dict[str, Counter] = defaultdict(Counter)
    stable_rows_by_key: Dict[str, int] = Counter()
    support_rows_by_key: Dict[str, int] = Counter()
    support_occ_by_key: Dict[str, int] = Counter()
    paired_rows = 0
    paired_row_keys = 0

    orig_iter = iter_jsonl(Path(args.orig10))
    per_iter = iter_jsonl(Path(args.per10))
    st_iter = iter_jsonl(Path(args.stable10))
    for idx, (orig, actual_per, actual_st) in enumerate(zip(orig_iter, per_iter, st_iter), start=1):
        if args.limit_rows > 0 and idx > args.limit_rows:
            break
        pred_per, stats_per, recs_per = mod.transform_row(orig, tok, bank, eligible_keys, max_tries=200, assignment_mode="per_row")
        pred_st, stats_st, recs_st = mod.transform_row(orig, tok, bank, eligible_keys, max_tries=200, assignment_mode="stable")
        counters["rows"] += 1
        counters["words"] += int(orig["words"])
        if actual_per.get("text") != pred_per.get("text"):
            counters["per_output_mismatch"] += 1
            if len(mismatch_examples) < args.max_examples:
                mismatch_examples.append({"row": idx, "mode": "per_row", "example_id": orig.get("example_id"), "actual": actual_per.get("text", "")[:300], "pred": pred_per.get("text", "")[:300]})
        if actual_st.get("text") != pred_st.get("text"):
            counters["stable_output_mismatch"] += 1
            if len(mismatch_examples) < args.max_examples:
                mismatch_examples.append({"row": idx, "mode": "stable", "example_id": orig.get("example_id"), "actual": actual_st.get("text", "")[:300], "pred": pred_st.get("text", "")[:300]})
        per_support = {(r["key"], r["occurrences"], tuple(r["profile"])) for r in recs_per}
        st_support = {(r["key"], r["occurrences"], tuple(r["profile"])) for r in recs_st}
        per_keys = {r["key"] for r in recs_per}
        st_keys = {r["key"] for r in recs_st}
        if recs_per or recs_st:
            paired_rows += 1
        if per_support != st_support:
            counters["support_diff_rows"] += 1
            counters["support_per_only_keyevents"] += len(per_support - st_support)
            counters["support_stable_only_keyevents"] += len(st_support - per_support)
            if len(support_diff_examples) < args.max_examples:
                support_diff_examples.append({
                    "row": idx,
                    "example_id": orig.get("example_id"),
                    "per_only": sorted([list(x) for x in per_support - st_support])[:10],
                    "stable_only": sorted([list(x) for x in st_support - per_support])[:10],
                    "original": str(orig.get("text", ""))[:500],
                })
        for r in recs_per:
            per_alias_by_key[r["key"]][r["alias"]] += 1
        for r in recs_st:
            stable_alias_by_key[r["key"]][r["alias"]] += 1
            stable_rows_by_key[r["key"]] += 1
        for r in recs_per:
            support_rows_by_key[r["key"]] += 1
            support_occ_by_key[r["key"]] += int(r["occurrences"])
        paired_row_keys += len(per_keys & st_keys)
        counters["per_assigned_keys"] += len(recs_per)
        counters["stable_assigned_keys"] += len(recs_st)
        counters["per_changed_rows"] += int(bool(recs_per))
        counters["stable_changed_rows"] += int(bool(recs_st))
        if idx % args.progress_every == 0:
            print(json.dumps({"event": "progress", "rows": idx, "support_diff_rows": counters["support_diff_rows"], "stable_keys": len(stable_alias_by_key), "elapsed": round(time.time() - t0, 1)}), flush=True)

    variable_stable_keys = {k: c for k, c in stable_alias_by_key.items() if len(c) > 1}
    variable_records = []
    for k, c in sorted(variable_stable_keys.items(), key=lambda kv: (-sum(kv[1].values()), -len(kv[1]), kv[0])):
        variable_records.append({
            "key": k,
            "rows": sum(c.values()),
            "n_aliases": len(c),
            "entropy_bits": entropy(c),
            "top_aliases": c.most_common(10),
        })
    exact_stable_keys = len(stable_alias_by_key) - len(variable_stable_keys)
    stable_alias_events = sum(sum(c.values()) for c in stable_alias_by_key.values())
    variable_events = sum(sum(c.values()) for c in variable_stable_keys.values())
    nonmodal_events = sum(sum(c.values()) - c.most_common(1)[0][1] for c in variable_stable_keys.values())

    summary = {
        "status": "STEP259_ORBIT_PAIR_AUDIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Audit research per_row/stable corpora before interpreting or continuing expensive language-scale training.",
        "inputs": {"orig10": args.orig10, "per10": args.per10, "stable10": args.stable10, "tokenizer": args.tokenizer, "limit_rows": args.limit_rows},
        "counts": dict(counters),
        "rows_with_any_transformation": paired_rows,
        "paired_row_key_intersection": paired_row_keys,
        "support_diff_examples": support_diff_examples,
        "output_mismatch_examples": mismatch_examples,
        "stable_alias_summary": {
            "keys_total": len(stable_alias_by_key),
            "keys_exact_one_alias": exact_stable_keys,
            "keys_multiple_aliases": len(variable_stable_keys),
            "assigned_key_events": stable_alias_events,
            "events_on_multi_alias_keys": variable_events,
            "nonmodal_alias_events_on_multi_alias_keys": nonmodal_events,
            "fraction_events_on_multi_alias_keys": variable_events / stable_alias_events if stable_alias_events else 0.0,
            "fraction_nonmodal_events": nonmodal_events / stable_alias_events if stable_alias_events else 0.0,
            "mean_entropy_bits_per_changed_key": sum(entropy(c) for c in stable_alias_by_key.values()) / len(stable_alias_by_key) if stable_alias_by_key else 0.0,
        },
        "top_multi_alias_stable_keys": variable_records[:100],
        "alias_bank": bank_meta,
        "interpretation": "If keys_multiple_aliases>0, the research stable arm is not a true corpus-global mapping. Even if support_diff_rows=0, per_row-vs-stable training cannot be interpreted as the decisive identity-persistence contrast until rebuilt on identical support with one compatible global alias per key or with explicit token-familiarity controls.",
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    out_json = out_dir / ("orbit_pair_audit" + (f"_pilot_{args.limit_rows}" if args.limit_rows > 0 else "") + ".json")
    out_md = out_dir / ("orbit_pair_audit" + (f"_pilot_{args.limit_rows}" if args.limit_rows > 0 else "") + ".md")
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ss = summary["stable_alias_summary"]
    md = []
    md.append("# research audit of research orbit pair\n\n")
    md.append(f"Rows/words audited: {counters['rows']} / {counters['words']}\n\n")
    md.append(f"Output replay mismatches: per_row={counters['per_output_mismatch']}, stable={counters['stable_output_mismatch']}\n\n")
    md.append(f"Support-different rows: {counters['support_diff_rows']} (per-only key-events {counters['support_per_only_keyevents']}, stable-only {counters['support_stable_only_keyevents']})\n\n")
    md.append("## Stable key-to-alias entropy\n\n")
    md.append(f"Stable transformed lexical keys: {ss['keys_total']}\n\n")
    md.append(f"Keys with exactly one alias: {ss['keys_exact_one_alias']}\n\n")
    md.append(f"Keys with multiple aliases: {ss['keys_multiple_aliases']}\n\n")
    md.append(f"Assigned key-events on multi-alias keys: {ss['events_on_multi_alias_keys']} / {ss['assigned_key_events']} = {ss['fraction_events_on_multi_alias_keys']:.4f}\n\n")
    md.append(f"Non-modal alias events: {ss['nonmodal_alias_events_on_multi_alias_keys']} / {ss['assigned_key_events']} = {ss['fraction_nonmodal_events']:.4f}\n\n")
    md.append(f"Mean entropy per changed key: {ss['mean_entropy_bits_per_changed_key']:.4f} bits\n\n")
    md.append("## Top multi-alias stable keys\n\n")
    md.append("| key | rows | n_aliases | entropy_bits | top aliases |\n|---|---:|---:|---:|---|\n")
    for r in variable_records[:30]:
        md.append(f"| {r['key']} | {r['rows']} | {r['n_aliases']} | {r['entropy_bits']:.3f} | {r['top_aliases']} |\n")
    md.append("\nJSON summary: `" + str(out_json) + "`\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "rows": counters["rows"],
        "support_diff_rows": counters["support_diff_rows"],
        "per_output_mismatch": counters["per_output_mismatch"],
        "stable_output_mismatch": counters["stable_output_mismatch"],
        "stable_keys_total": ss["keys_total"],
        "stable_keys_multiple_aliases": ss["keys_multiple_aliases"],
        "fraction_events_on_multi_alias_keys": ss["fraction_events_on_multi_alias_keys"],
        "fraction_nonmodal_events": ss["fraction_nonmodal_events"],
        "summary_json": str(out_json),
        "elapsed_seconds": summary["elapsed_seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
