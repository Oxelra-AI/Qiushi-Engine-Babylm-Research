#!/usr/bin/env python3
"""research: audit lexical copy overlap in compact-view pair subsets.

The reciprocal/topology route is scientifically fragile because research found copied
lexical lift dominates reciprocal-view probes. This script quantifies the exact
word-multiset copy overlap of (a) all available compact pairs and (b) the current
research 2x2 selected subset, and reports whether a low-copy dose-matched subset
could be constructed for a future sharper non-copy semantic screen.

No model training/evaluation occurs.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, re, statistics
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
PAIRS_FILE = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
SELECTED_FILE = ROOT / "data/causal_topology_2x2_scaffold/selected_topology2x2_pairs.jsonl"
OUT_DEFAULT = ROOT / "data/pair_copy_overlap_audit"
TARGET_DUAL_WORDS = 423_512


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def wc(t: str) -> int:
    return len(t.split())


def norm(w: str) -> str:
    return re.sub(r"^\W+|\W+$", "", w.lower())


def overlap_fraction(view: str, source: str) -> dict[str, Any]:
    a = [norm(x) for x in view.split()]
    b = [norm(x) for x in source.split()]
    a = [x for x in a if x]
    counts: dict[str, int] = {}
    for x in b:
        if x:
            counts[x] = counts.get(x, 0) + 1
    matched = 0
    for x in a:
        if counts.get(x, 0) > 0:
            matched += 1
            counts[x] -= 1
    return {"view_words_normed": len(a), "matched": matched, "overlap_frac": matched / len(a) if a else None, "noncopy_frac": 1 - matched / len(a) if a else None}


def stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p25": None, "p75": None, "p90": None, "min": None, "max": None}
    ys = sorted(xs)
    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        pos = p * (len(ys) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ys) - 1)
        frac = pos - lo
        return ys[lo] * (1 - frac) + ys[hi] * frac
    return {"n": len(ys), "mean": float(sum(ys)/len(ys)), "median": float(statistics.median(ys)), "p10": float(q(0.10)), "p25": float(q(0.25)), "p75": float(q(0.75)), "p90": float(q(0.90)), "min": float(ys[0]), "max": float(ys[-1])}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    over = [r["overlap_frac"] for r in rows if r.get("overlap_frac") is not None]
    non = [r["noncopy_frac"] for r in rows if r.get("noncopy_frac") is not None]
    dual_words = [r["dual_pair_words"] for r in rows]
    bins = {
        "overlap_lt_0p50": sum(1 for x in over if x < 0.50),
        "overlap_0p50_0p70": sum(1 for x in over if 0.50 <= x < 0.70),
        "overlap_0p70_0p85": sum(1 for x in over if 0.70 <= x < 0.85),
        "overlap_ge_0p85": sum(1 for x in over if x >= 0.85),
    }
    return {
        "n_pairs": len(rows),
        "dual_pair_words_total": sum(dual_words),
        "single_pair_words_total": sum(dual_words)//2,
        "overlap_frac_stats": stats(over),
        "noncopy_frac_stats": stats(non),
        "dual_pair_words_stats": stats([float(x) for x in dual_words]),
        "bins": bins,
    }


def greedy_subset(rows: list[dict[str, Any]], target: int, key) -> list[dict[str, Any]]:
    out = []
    total = 0
    for r in sorted(rows, key=key):
        if total >= target:
            break
        out.append(r)
        total += r["dual_pair_words"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=pathlib.Path, default=PAIRS_FILE)
    ap.add_argument("--selected", type=pathlib.Path, default=SELECTED_FILE)
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DEFAULT)
    ap.add_argument("--target-dual-words", type=int, default=TARGET_DUAL_WORDS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_jsonl(args.pairs)
    selected = read_jsonl(args.selected) if args.selected.exists() else []
    selected_ids = {r["pair_id"] for r in selected}
    enriched = []
    for idx, p in enumerate(pairs):
        source = p["source_text"]
        rewrite = p["rewrite_text"]
        ow = overlap_fraction(rewrite, source)
        sw = int(p.get("source_words") or wc(source))
        rw = int(p.get("rewrite_words") or wc(rewrite))
        row = {
            "idx": idx,
            "pair_id": p.get("pair_id"),
            "source_words": sw,
            "rewrite_words": rw,
            "dual_pair_words": 2*(sw+rw),
            "overlap_frac": ow["overlap_frac"],
            "noncopy_frac": ow["noncopy_frac"],
            "matched_words": ow["matched"],
            "view_words_normed": ow["view_words_normed"],
            "is_step179_selected": p.get("pair_id") in selected_ids,
        }
        enriched.append(row)

    selected_rows = [r for r in enriched if r["is_step179_selected"]]
    lowcopy = greedy_subset(enriched, args.target_dual_words, key=lambda r: (r["overlap_frac"] if r["overlap_frac"] is not None else 1.0, r["dual_pair_words"]))
    high_noncopy = greedy_subset(enriched, args.target_dual_words, key=lambda r: (-(r["noncopy_frac"] if r["noncopy_frac"] is not None else 0.0), r["dual_pair_words"]))
    # One could target enough pairs while avoiding extreme length shifts; provide both simple and length-balanced heuristic.
    length_balanced = greedy_subset(enriched, args.target_dual_words, key=lambda r: (abs(r["dual_pair_words"] - 70), r["overlap_frac"] if r["overlap_frac"] is not None else 1.0))

    result = {
        "status": "PAIR_COPY_OVERLAP_AUDIT",
        "pairs_file": str(args.pairs),
        "pairs_sha256": sha256_file(args.pairs),
        "selected_file": str(args.selected),
        "selected_sha256": sha256_file(args.selected) if args.selected.exists() else None,
        "target_dual_pair_words": args.target_dual_words,
        "all_pairs": summarize(enriched),
        "current_step179_selected": summarize(selected_rows),
        "greedy_low_overlap_dose_subset": summarize(lowcopy),
        "greedy_high_noncopy_dose_subset": summarize(high_noncopy),
        "length_near70_then_low_overlap_subset": summarize(length_balanced),
        "scientific_reading": [
            "High source/rewrite word overlap means copied-token lift is a serious confound for reciprocal-view claims.",
            "A lower-overlap subset is feasible for a sharper non-copy semantic screen if this route is later revived, but changing the pair subset also changes data distribution and must be treated as a different matched screen rather than a direct continuation of research.",
            "No training decision follows from this audit alone."
        ]
    }
    out_json = args.out_dir / "pair_copy_overlap_audit.json"
    out_md = args.out_dir / "pair_copy_overlap_audit.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = ["# research pair copy-overlap audit\n\n"]
    md.append("No training/evaluation was run. This quantifies copied-token confounding for any future topology 2×2 screen.\n\n")
    for name in ["all_pairs", "current_step179_selected", "greedy_low_overlap_dose_subset", "length_near70_then_low_overlap_subset"]:
        s = result[name]
        ov = s["overlap_frac_stats"]
        non = s["noncopy_frac_stats"]
        md.append(f"## {name}\n\n")
        md.append(f"pairs={s['n_pairs']}, dual_pair_words={s['dual_pair_words_total']:,}, mean overlap={ov['mean']:.4f}, median overlap={ov['median']:.4f}, p25={ov['p25']:.4f}, p75={ov['p75']:.4f}, mean noncopy={non['mean']:.4f}.\n\n")
        md.append(f"bins: {s['bins']}\n\n")
    md.append("Scientific reading: the current 2×2 subset has high lexical overlap; if independent evidence revives reciprocal topology, a lower-overlap matched subset can test non-copy semantic interaction more sharply, but it changes the data distribution and therefore should be interpreted separately.\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
