#!/usr/bin/env python3
"""research corrected source-position/content audit for repeat controls.

The source-position atlas names a first-N repeat, but the historical repeat
builder actually uses a hash-rotated cyclic source segment of the compact view
length.  This CPU audit compares compact, prefix repeat, historical hash repeat,
and BPE-matched cyclic repeat on source-content coverage, tail/decile coverage,
content fraction, and BPE load.  It supports the next ordinary-WWM factorial
construction; it does not mutate corpora or launch training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def find_user_root() -> Path:
    return _PUBLIC_ROOT

USER_ROOT = find_user_root()
DEFAULT_PAIRS = USER_ROOT / "experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
DEFAULT_TOKENIZER = USER_ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer"
DEFAULT_BPE_RECORDS = USER_ROOT / "experiments/archive/representation_and_objectives/data/repeat_bpe_match_audit/repeat_bpe_match_records.jsonl"
DEFAULT_OUT = USER_ROOT / "experiments/archive/representation_and_objectives/data/repeat_position_coverage_audit"

STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "his", "i", "if",
    "in", "into", "is", "it", "its", "itself", "may", "more", "most", "must", "no", "not", "of",
    "on", "or", "our", "she", "so", "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "to", "up", "was", "were", "what", "when", "where", "which",
    "while", "who", "will", "with", "within", "would", "you", "your", "we", "also", "all", "any",
    "because", "between", "through", "using", "use", "used", "uses", "like", "one", "two", "many",
}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-'’]*|\d+(?:\.\d+)?")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(w: str) -> str:
    m = WORD_RE.search(w)
    if not m:
        return ""
    return m.group(0).lower().strip("-'’")


def is_content(n: str) -> bool:
    return bool(n) and len(n) > 2 and n not in STOP


def hash_start(source_text: str, n_words: int, salt: str) -> int:
    toks = source_text.split()
    if not toks or n_words <= 0:
        return 0
    h = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16)
    return h % len(toks)


def segment_positions(n_source: int, start: int, k: int) -> list[int]:
    return [(start + t) % n_source for t in range(k)] if n_source and k > 0 else []


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def qstats(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p25": None, "p75": None, "p90": None, "min": None, "max": None}
    xs = sorted(float(x) for x in vals)
    def q(p: float) -> float:
        if len(xs) == 1: return xs[0]
        pos = p * (len(xs)-1); lo = math.floor(pos); hi = math.ceil(pos)
        if lo == hi: return xs[lo]
        return xs[lo]*(hi-pos)+xs[hi]*(pos-lo)
    return {"n": len(xs), "mean": sum(xs)/len(xs), "median": q(0.5), "p10": q(0.1), "p25": q(0.25), "p75": q(0.75), "p90": q(0.9), "min": xs[0], "max": xs[-1]}


def analyze_repeat(source_norms: list[str], k: int, positions: list[int]) -> dict[str, Any]:
    content_positions = [i for i, n in enumerate(source_norms) if is_content(n)]
    tail_positions = [i for i in content_positions if i >= k]
    pos_set = set(positions)
    covered = [i for i in content_positions if i in pos_set]
    tail_cov = [i for i in tail_positions if i in pos_set]
    content_in_segment = [i for i in positions if i < len(source_norms) and is_content(source_norms[i])]
    dec = [0]*10
    for i in covered:
        d = min(9, int(10*i/max(1,len(source_norms))))
        dec[d]+=1
    return {
        "content_coverage": len(covered)/len(content_positions) if content_positions else None,
        "tail_content_coverage": len(tail_cov)/len(tail_positions) if tail_positions else None,
        "content_fraction": len(content_in_segment)/len(positions) if positions else None,
        "covered_count": len(covered),
        "tail_covered_count": len(tail_cov),
        "decile_covered": dec,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--bpe_records", default=str(DEFAULT_BPE_RECORDS))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    pairs_path = Path(args.pairs); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(pairs_path)
    bpe_recs = {str(r["pair_id"]): r for r in read_jsonl(Path(args.bpe_records))}
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    vals: dict[str, dict[str, list[float]]] = {arm: collections.defaultdict(list) for arm in ["compact", "prefix_repeat", "hash_repeat", "best_bpe_repeat"]}
    dec_total = [0]*10
    dec_cov = {arm: [0]*10 for arm in vals}
    token_deltas: dict[str, list[float]] = {"compact_minus_prefix": [], "compact_minus_hash": [], "compact_minus_best": []}
    rows = []
    for p in pairs:
        pid = str(p["pair_id"])
        src_words = str(p["source_text"]).split()
        view_words = str(p.get("view_text") or p.get("original_compact_text")).split()
        k = len(view_words)
        source_norms = [norm(w) for w in src_words]
        compact_norms = [norm(w) for w in view_words]
        compact_set = {n for n in compact_norms if is_content(n)}
        content_positions = [i for i,n in enumerate(source_norms) if is_content(n)]
        tail_positions = [i for i in content_positions if i>=k]
        compact_covered = [i for i in content_positions if source_norms[i] in compact_set]
        compact_tail = [i for i in tail_positions if source_norms[i] in compact_set]
        c_dec = [0]*10
        for i in compact_covered:
            c_dec[min(9, int(10*i/max(1,len(source_norms))))]+=1
        compact = {
            "content_coverage": len(compact_covered)/len(content_positions) if content_positions else None,
            "tail_content_coverage": len(compact_tail)/len(tail_positions) if tail_positions else None,
            "content_fraction": sum(1 for n in compact_norms if is_content(n))/len(view_words) if view_words else None,
            "covered_count": len(compact_covered),
            "tail_covered_count": len(compact_tail),
            "decile_covered": c_dec,
        }
        prefix = analyze_repeat(source_norms, k, list(range(k)))
        hstart = hash_start(str(p["source_text"]), k, pid)
        hashr = analyze_repeat(source_norms, k, segment_positions(len(src_words), hstart, k))
        brec = bpe_recs.get(pid)
        if brec is None:
            raise RuntimeError(f"missing bpe record for {pid}")
        best = analyze_repeat(source_norms, k, segment_positions(len(src_words), int(brec["best_start"]), k))
        arm_recs = {"compact": compact, "prefix_repeat": prefix, "hash_repeat": hashr, "best_bpe_repeat": best}
        for d in range(10):
            dec_total[d] += sum(1 for i in content_positions if min(9, int(10*i/max(1,len(source_norms)))) == d)
        for arm, rec in arm_recs.items():
            for metric in ["content_coverage", "tail_content_coverage", "content_fraction"]:
                if rec[metric] is not None:
                    vals[arm][metric].append(float(rec[metric]))
            for d,x in enumerate(rec["decile_covered"]):
                dec_cov[arm][d]+=x
        compact_bpe = len(tok(" ".join(view_words), add_special_tokens=False)["input_ids"])
        prefix_bpe = len(tok(" ".join(src_words[:k]), add_special_tokens=False)["input_ids"])
        hash_bpe = int(brec["old_repeat_bpe"])
        best_bpe = int(brec["best_repeat_bpe"])
        token_deltas["compact_minus_prefix"].append(compact_bpe-prefix_bpe)
        token_deltas["compact_minus_hash"].append(compact_bpe-hash_bpe)
        token_deltas["compact_minus_best"].append(compact_bpe-best_bpe)
        rows.append({
            "pair_id": pid,
            "source_words": len(src_words),
            "view_words": k,
            "hash_start": hstart,
            "best_start": int(brec["best_start"]),
            "compact": compact,
            "prefix_repeat": prefix,
            "hash_repeat": hashr,
            "best_bpe_repeat": best,
            "bpe": {"compact": compact_bpe, "prefix_repeat": prefix_bpe, "hash_repeat": hash_bpe, "best_bpe_repeat": best_bpe},
        })
    summary = {
        "status": "REPEAT_POSITION_COVERAGE_AUDIT",
        "created_utc": now_utc(),
        "meaning": "Corrected CPU audit of compact, prefix repeat, historical hash-rotated repeat, and BPE-matched cyclic repeat source-position/content coverage. No training.",
        "inputs": {"pairs": str(pairs_path), "pairs_sha256": sha256_file(pairs_path), "bpe_records": args.bpe_records, "n_pairs": len(pairs)},
        "arm_stats": {arm: {m: qstats(v) for m,v in metrics.items()} for arm, metrics in vals.items()},
        "decile_content_total": dec_total,
        "decile_coverage": {arm: [dec_cov[arm][i]/dec_total[i] if dec_total[i] else None for i in range(10)] for arm in dec_cov},
        "bpe_deltas_compact_minus_repeat": {k: qstats(v) for k,v in token_deltas.items()},
        "interpretation": "Historical repeat is hash-rotated, not prefix-first-N. BPE-matched cyclic repeat greatly reduces token-load mismatch but changes source-position coverage; if used, its tail/decile coverage must be treated as part of the low-diversity control rather than as the historical repeat baseline.",
    }
    (out_dir/"repeat_position_coverage_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    with (out_dir/"repeat_position_coverage_records.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    lines = ["# research repeat source-position/content coverage audit", "", f"JSON: `{out_dir/'repeat_position_coverage_audit.json'}`", "", "## Mean content coverage"]
    for arm in ["compact", "prefix_repeat", "hash_repeat", "best_bpe_repeat"]:
        st = summary["arm_stats"][arm]
        lines.append(f"- {arm}: content {st['content_coverage']['mean']:.6f}, tail {st['tail_content_coverage']['mean'] if st['tail_content_coverage']['mean'] is not None else None}, content_fraction {st['content_fraction']['mean']:.6f}")
    lines += ["", "## Mean compact-minus-repeat BPE deltas"]
    for k, st in summary["bpe_deltas_compact_minus_repeat"].items():
        lines.append(f"- {k}: {st['mean']:.6f} (median {st['median']})")
    lines += ["", "## Decile coverage (content positions)"]
    for arm, arr in summary["decile_coverage"].items():
        lines.append(f"- {arm}: " + ", ".join("NA" if x is None else f"{x:.3f}" for x in arr))
    (out_dir/"repeat_position_coverage_audit.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_dir/"repeat_position_coverage_audit.json"),
        "compact_content": summary["arm_stats"]["compact"]["content_coverage"]["mean"],
        "hash_content": summary["arm_stats"]["hash_repeat"]["content_coverage"]["mean"],
        "best_content": summary["arm_stats"]["best_bpe_repeat"]["content_coverage"]["mean"],
        "compact_minus_hash_bpe": summary["bpe_deltas_compact_minus_repeat"]["compact_minus_hash"]["mean"],
        "compact_minus_best_bpe": summary["bpe_deltas_compact_minus_repeat"]["compact_minus_best"]["mean"],
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
