#!/usr/bin/env python3
"""Pair-level summary for research source-attested fluent bridge prototype."""
from __future__ import annotations
import collections, json, math, statistics
from pathlib import Path

ROOT = Path("experiments/archive/frontier_consolidation")
ROWS = ROOT / "data/source_attested_fluent_bridge_prototype/source_attested_fluent_bridge_generation_rows.jsonl"
OUT = ROOT / "data/source_attested_fluent_bridge_prototype/source_attested_fluent_bridge_pair_level_summary.json"
NOTE = (ROOT.parents[2] / 'research/notes/frontier_consolidation/source_attested_fluent_bridge_prototype_readout.md')

def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def stat(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return {"n": 0}
    vals.sort()
    def q(p):
        if len(vals) == 1:
            return vals[0]
        z = p * (len(vals)-1); lo = math.floor(z); hi = math.ceil(z)
        return vals[lo] if lo == hi else vals[lo]*(hi-z)+vals[hi]*(z-lo)
    return {"n": len(vals), "mean": statistics.fmean(vals), "median": statistics.median(vals), "p10": q(0.1), "p25": q(0.25), "p75": q(0.75), "p90": q(0.9), "min": vals[0], "max": vals[-1]}

def main():
    rows = read_jsonl(ROWS)
    by_pair = collections.defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)
    selected = []
    for pid, rs in by_pair.items():
        acc = [r for r in rs if r.get("prototype_accept")]
        if acc:
            best = min(acc, key=lambda r: (abs(float(r.get("generated_to_natural_words_ratio") or 99)-1.0),
                                           int(r.get("unsupported_content_lemma_count") or 0),
                                           -float(r.get("source_content_recall") or 0),
                                           r.get("regime") or ""))
            selected.append(best)
    def summarize(rs):
        return {
            "n": len(rs),
            "pair_ids": len({r["pair_id"] for r in rs}),
            "accept_rows": sum(1 for r in rs if r.get("prototype_accept")),
            "accept_row_rate": sum(1 for r in rs if r.get("prototype_accept")) / max(1, len(rs)),
            "zero_unsupported_rows": sum(1 for r in rs if int(r.get("unsupported_content_lemma_count") or 0) == 0),
            "generated_to_natural_words_ratio": stat([r.get("generated_to_natural_words_ratio") for r in rs]),
            "active_token_ratio_to_natural": stat([r.get("active_token_ratio_to_natural") for r in rs]),
            "generated_content_fraction": stat([r.get("generated_content_fraction") for r in rs]),
            "source_content_recall": stat([r.get("source_content_recall") for r in rs]),
            "natural_content_overlap_recall": stat([r.get("natural_content_overlap_recall") for r in rs]),
            "relation_proxy_retained_rate": sum(1 for r in rs if r.get("relation_proxy_retained")) / max(1, len(rs)),
        }
    bucket_all = collections.defaultdict(list)
    bucket_sel = collections.defaultdict(list)
    for r in rows:
        bucket_all[r.get("prototype_bucket")].append(r)
    for r in selected:
        bucket_sel[r.get("prototype_bucket")].append(r)
    reg_all = collections.defaultdict(list)
    reg_acc = collections.defaultdict(list)
    for r in rows:
        reg_all[r.get("regime")].append(r)
        if r.get("prototype_accept"):
            reg_acc[r.get("regime")].append(r)
    payload = {
        "status": "PAIR_LEVEL_SOURCE_ATTESTED_FLUENT_BRIDGE_SUMMARY",
        "rows_file": str(ROWS),
        "total_prompt_rows": len(rows),
        "total_pairs": len(by_pair),
        "accepted_any_pair_count": len(selected),
        "accepted_any_pair_rate": len(selected) / max(1, len(by_pair)),
        "selected_best_per_accepted_pair": summarize(selected),
        "by_regime_all_rows": {k: summarize(v) for k,v in sorted(reg_all.items())},
        "by_regime_accepted_rows": {k: summarize(v) for k,v in sorted(reg_acc.items())},
        "by_bucket_all_rows": {k: summarize(v) for k,v in sorted(bucket_all.items())},
        "by_bucket_selected_pairs": {k: summarize(v) for k,v in sorted(bucket_sel.items())},
        "best_selected_examples": selected[:12],
        "all_hard_reason_counts": dict(collections.Counter(x for r in rows for x in (r.get("hard_reasons") or []))),
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research source-attested fluent bridge prototype readout\n\n",
        "research bridge trainings were cancelled because they only extended the telegraphic source-only extraction family. This file reads the replacement constructibility prototype: Qwen3.5 generated two source-attested fluent-view attempts for 48 compact-source pairs.\n\n",
        f"Rows: {len(rows)}; pairs: {len(by_pair)}; pairs with at least one automatically accepted output: {len(selected)}/{len(by_pair)} ({len(selected)/max(1,len(by_pair)):.3f}).\n\n",
        f"Best accepted per pair: generated/natural word ratio mean {payload['selected_best_per_accepted_pair']['generated_to_natural_words_ratio']['mean']:.3f}, active-token ratio mean {payload['selected_best_per_accepted_pair']['active_token_ratio_to_natural']['mean']:.3f}, source-content recall mean {payload['selected_best_per_accepted_pair']['source_content_recall']['mean']:.3f}, natural-content overlap recall mean {payload['selected_best_per_accepted_pair']['natural_content_overlap_recall']['mean']:.3f}.\n\n",
        "| bucket | selected pairs | generated/natural words mean | token/natural mean | source recall mean | natural overlap mean |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for k,s in payload["by_bucket_selected_pairs"].items():
        lines.append(f"| {k} | {s['n']} | {s['generated_to_natural_words_ratio'].get('mean', float('nan')):.3f} | {s['active_token_ratio_to_natural'].get('mean', float('nan')):.3f} | {s['source_content_recall'].get('mean', float('nan')):.3f} | {s['natural_content_overlap_recall'].get('mean', float('nan')):.3f} |\n")
    lines.append(f"\nPair-level JSON: `{OUT}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "accepted_any_pairs": payload["accepted_any_pair_count"], "pair_rate": payload["accepted_any_pair_rate"], "out": str(OUT), "note": str(NOTE)}, indent=2))
if __name__ == "__main__":
    main()
