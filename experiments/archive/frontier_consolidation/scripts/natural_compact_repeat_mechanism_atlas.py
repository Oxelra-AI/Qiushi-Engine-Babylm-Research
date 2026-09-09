#!/usr/bin/env python3
"""research: CPU-only atlas of the natural compact-vs-first-N-repeat data marginal.

The pending RoBERTa 100M experiment tests the whole natural compact/repeat marginal.
This script makes that marginal inspectable without touching running GPU jobs: for the
12,155 underlying research compact-pair objects it compares the generated compact view
to the matched first-N source repetition with the same word count, under the legal
research tokenizer.  Token counts here are individual pair-view encodings; use the
research packed-row audit for exact training-stream BPE exposure.

It measures source-position coverage, tail recurrence, source-absent content words,
BPE/candidate-token load, and category composition.  It is text/data analysis only;
no training, model evaluation, upload, or leaderboard submission.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import csv
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so", "because", "as", "than",
    "to", "of", "in", "on", "for", "with", "without", "by", "from", "at", "into", "onto", "over",
    "under", "about", "between", "among", "through", "during", "before", "after", "above", "below",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "doing",
    "have", "has", "had", "having", "can", "could", "may", "might", "must", "shall", "should",
    "will", "would", "this", "that", "these", "those", "there", "here", "it", "its", "they", "them",
    "their", "theirs", "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours", "you",
    "your", "yours", "i", "me", "my", "mine", "who", "whom", "whose", "which", "what", "where",
    "when", "why", "how", "not", "no", "nor", "only", "just", "also", "very", "more", "most", "less",
    "least", "much", "many", "some", "any", "all", "each", "every", "other", "another", "such", "own",
    "same", "too", "again", "still", "already", "yet", "up", "down", "out", "off", "back", "away",
    "into", "within", "across", "per", "via", "using", "used", "use", "uses", "become", "became",
}
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)?")


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_PAIRS = WS / "data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
DEFAULT_TOKENIZER = WS / "data/compliant_tokenizer"
DEFAULT_OUT = WS / "data/natural_compact_repeat_mechanism_atlas"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_words(text: str) -> list[str]:
    return [w for w in text.split() if w]


def norm_word(word: str) -> str:
    parts = WORD_RE.findall(word)
    if not parts:
        return ""
    return "".join(parts).lower()


def is_content(norm: str) -> bool:
    if not norm:
        return False
    if norm in STOPWORDS:
        return False
    if norm.isdigit():
        return True
    return len(norm) >= 4


def mean(xs: list[float | int | None]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def median(xs: list[float | int | None]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return statistics.median(vals) if vals else None


def quantile(xs: list[float | int | None], q: float) -> float | None:
    vals = sorted(float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x)))
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[idx]


def stat(xs: list[float | int | None]) -> dict[str, Any]:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "p10": quantile(vals, 0.10),
        "p25": quantile(vals, 0.25),
        "p75": quantile(vals, 0.75),
        "p90": quantile(vals, 0.90),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def encode_stats(tokenizer, text: str) -> dict[str, int]:
    enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=256)
    ids = enc["input_ids"]
    toks = tokenizer.convert_ids_to_tokens(ids)
    special = set(tokenizer.all_special_ids)
    active = sum(1 for i in ids if i not in special)
    word_starts = 0
    for tok, tid in zip(toks, ids):
        if tid in special:
            continue
        s = str(tok)
        if s.startswith("Ġ") or s.startswith("▁") or word_starts == 0:
            word_starts += 1
    return {"active_tokens": active, "candidate_tokens": active, "word_groups": word_starts, "truncated": int(len(ids) >= 256)}


def jaccard_set(a: set[str], b: set[str]) -> float | None:
    if not a and not b:
        return None
    u = a | b
    return len(a & b) / len(u) if u else None


def read_pairs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "source_text" not in r or "view_text" not in r:
                raise ValueError(f"pair row lacks source_text/view_text: {line[:200]}")
            rows.append(r)
    return rows


def analyze_pair(tokenizer, r: dict[str, Any]) -> dict[str, Any]:
    source_words = split_words(r["source_text"])
    compact_words = split_words(r["view_text"])
    k = len(compact_words)
    repeat_words = source_words[:k]
    repeat_text = " ".join(repeat_words)
    compact_text = r["view_text"]

    source_norms = [norm_word(w) for w in source_words]
    compact_norms = [norm_word(w) for w in compact_words]
    repeat_norms = [norm_word(w) for w in repeat_words]
    source_set = {n for n in source_norms if n}
    prefix_set = {n for n in source_norms[:k] if n}
    tail_set = {n for n in source_norms[k:] if n}
    compact_set = {n for n in compact_norms if n}
    repeat_set = {n for n in repeat_norms if n}

    content_positions = [i for i, n in enumerate(source_norms) if is_content(n)]
    tail_content_positions = [i for i in content_positions if i >= k]
    compact_covered = [i for i in content_positions if source_norms[i] in compact_set]
    repeat_covered = [i for i in content_positions if i < k and source_norms[i] in repeat_set]
    compact_tail_covered = [i for i in tail_content_positions if source_norms[i] in compact_set]

    compact_content = [n for n in compact_norms if is_content(n)]
    repeat_content = [n for n in repeat_norms if is_content(n)]
    compact_source_absent_content = [n for n in compact_content if n not in source_set]
    compact_copied_content = [n for n in compact_content if n in source_set]
    compact_tail_only_content = [n for n in compact_content if n in tail_set and n not in prefix_set]
    compact_tail_any_content = [n for n in compact_content if n in tail_set]

    dec_total = [0] * 10
    dec_compact = [0] * 10
    dec_repeat = [0] * 10
    for i in content_positions:
        d = min(9, int(10 * i / max(1, len(source_words))))
        dec_total[d] += 1
        if source_norms[i] in compact_set:
            dec_compact[d] += 1
        if i < k:
            dec_repeat[d] += 1

    compact_tok = encode_stats(tokenizer, compact_text)
    repeat_tok = encode_stats(tokenizer, repeat_text)
    source_tok = encode_stats(tokenizer, r["source_text"])

    return {
        "pair_id": r.get("pair_id"),
        "key": r.get("key"),
        "source_words": len(source_words),
        "view_words": len(compact_words),
        "ratio_view_to_source": len(compact_words) / len(source_words) if source_words else None,
        "content_source_positions": len(content_positions),
        "tail_content_positions_after_view_len": len(tail_content_positions),
        "compact_source_content_coverage": len(compact_covered) / len(content_positions) if content_positions else None,
        "repeat_source_content_coverage": len(repeat_covered) / len(content_positions) if content_positions else None,
        "compact_minus_repeat_content_coverage": (len(compact_covered) - len(repeat_covered)) / len(content_positions) if content_positions else None,
        "compact_tail_content_coverage": len(compact_tail_covered) / len(tail_content_positions) if tail_content_positions else None,
        "compact_has_tail_content_recovery": bool(compact_tail_covered),
        "compact_content_fraction": len(compact_content) / len(compact_words) if compact_words else None,
        "repeat_content_fraction": len(repeat_content) / len(repeat_words) if repeat_words else None,
        "compact_source_absent_content_words": len(compact_source_absent_content),
        "compact_source_absent_content_fraction_of_view": len(compact_source_absent_content) / len(compact_words) if compact_words else None,
        "compact_source_absent_content_fraction_of_content": len(compact_source_absent_content) / len(compact_content) if compact_content else None,
        "compact_copied_content_words": len(compact_copied_content),
        "compact_tail_only_content_words": len(compact_tail_only_content),
        "compact_tail_any_content_words": len(compact_tail_any_content),
        "compact_tail_only_content_fraction_of_content": len(compact_tail_only_content) / len(compact_content) if compact_content else None,
        "compact_content_jaccard_source": jaccard_set(set(compact_content), {n for n in source_norms if is_content(n)}),
        "repeat_content_jaccard_source": jaccard_set(set(repeat_content), {n for n in source_norms if is_content(n)}),
        "compact_vs_repeat_content_jaccard": jaccard_set(set(compact_content), set(repeat_content)),
        "decile_total": dec_total,
        "decile_compact": dec_compact,
        "decile_repeat": dec_repeat,
        "source_token_stats": source_tok,
        "compact_token_stats": compact_tok,
        "repeat_token_stats": repeat_tok,
        "compact_minus_repeat_active_tokens": compact_tok["active_tokens"] - repeat_tok["active_tokens"],
        "compact_minus_repeat_word_groups": compact_tok["word_groups"] - repeat_tok["word_groups"],
        "compact_minus_repeat_truncated": compact_tok["truncated"] - repeat_tok["truncated"],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dec_total = [0] * 10
    dec_compact = [0] * 10
    dec_repeat = [0] * 10
    for r in rows:
        for i in range(10):
            dec_total[i] += r["decile_total"][i]
            dec_compact[i] += r["decile_compact"][i]
            dec_repeat[i] += r["decile_repeat"][i]
    dec_cov = []
    for i in range(10):
        dec_cov.append({
            "decile": i,
            "source_content_positions": dec_total[i],
            "compact_coverage": dec_compact[i] / dec_total[i] if dec_total[i] else None,
            "repeat_coverage": dec_repeat[i] / dec_total[i] if dec_total[i] else None,
            "delta": (dec_compact[i] - dec_repeat[i]) / dec_total[i] if dec_total[i] else None,
        })

    total_source_words = sum(r["source_words"] for r in rows)
    total_view_words = sum(r["view_words"] for r in rows)
    total_source_abs_content = sum(r["compact_source_absent_content_words"] for r in rows)
    total_copied_content = sum(r["compact_copied_content_words"] for r in rows)
    total_tail_only = sum(r["compact_tail_only_content_words"] for r in rows)
    total_compact_content = sum(int(round(r["compact_content_fraction"] * r["view_words"])) if r["compact_content_fraction"] is not None else 0 for r in rows)
    token_totals: dict[str, int] = collections.Counter()
    for r in rows:
        token_totals["source_active_tokens"] += r["source_token_stats"]["active_tokens"]
        token_totals["compact_active_tokens"] += r["compact_token_stats"]["active_tokens"]
        token_totals["repeat_active_tokens"] += r["repeat_token_stats"]["active_tokens"]
        token_totals["compact_word_groups"] += r["compact_token_stats"]["word_groups"]
        token_totals["repeat_word_groups"] += r["repeat_token_stats"]["word_groups"]
        token_totals["compact_truncated_rows"] += r["compact_token_stats"]["truncated"]
        token_totals["repeat_truncated_rows"] += r["repeat_token_stats"]["truncated"]
    token_totals["compact_minus_repeat_active_tokens"] = token_totals["compact_active_tokens"] - token_totals["repeat_active_tokens"]
    token_totals["compact_minus_repeat_word_groups"] = token_totals["compact_word_groups"] - token_totals["repeat_word_groups"]

    return {
        "pairs": len(rows),
        "total_source_words": total_source_words,
        "total_view_words": total_view_words,
        "view_to_source_ratio": total_view_words / total_source_words if total_source_words else None,
        "stats": {
            "source_words": stat([r["source_words"] for r in rows]),
            "view_words": stat([r["view_words"] for r in rows]),
            "ratio_view_to_source": stat([r["ratio_view_to_source"] for r in rows]),
            "compact_source_content_coverage": stat([r["compact_source_content_coverage"] for r in rows]),
            "repeat_source_content_coverage": stat([r["repeat_source_content_coverage"] for r in rows]),
            "compact_minus_repeat_content_coverage": stat([r["compact_minus_repeat_content_coverage"] for r in rows]),
            "compact_tail_content_coverage": stat([r["compact_tail_content_coverage"] for r in rows]),
            "compact_content_fraction": stat([r["compact_content_fraction"] for r in rows]),
            "repeat_content_fraction": stat([r["repeat_content_fraction"] for r in rows]),
            "compact_source_absent_content_words": stat([r["compact_source_absent_content_words"] for r in rows]),
            "compact_source_absent_content_fraction_of_view": stat([r["compact_source_absent_content_fraction_of_view"] for r in rows]),
            "compact_source_absent_content_fraction_of_content": stat([r["compact_source_absent_content_fraction_of_content"] for r in rows]),
            "compact_tail_only_content_words": stat([r["compact_tail_only_content_words"] for r in rows]),
            "compact_tail_only_content_fraction_of_content": stat([r["compact_tail_only_content_fraction_of_content"] for r in rows]),
            "compact_vs_repeat_content_jaccard": stat([r["compact_vs_repeat_content_jaccard"] for r in rows]),
            "compact_minus_repeat_active_tokens": stat([r["compact_minus_repeat_active_tokens"] for r in rows]),
            "compact_minus_repeat_word_groups": stat([r["compact_minus_repeat_word_groups"] for r in rows]),
        },
        "fractions": {
            "pairs_with_tail_content_recovery": mean([1.0 if r["compact_has_tail_content_recovery"] else 0.0 for r in rows]),
            "pairs_with_positive_content_coverage_delta": mean([1.0 if (r["compact_minus_repeat_content_coverage"] is not None and r["compact_minus_repeat_content_coverage"] > 0) else 0.0 for r in rows]),
            "pairs_with_any_source_absent_content": mean([1.0 if r["compact_source_absent_content_words"] > 0 else 0.0 for r in rows]),
            "pairs_with_any_tail_only_content": mean([1.0 if r["compact_tail_only_content_words"] > 0 else 0.0 for r in rows]),
        },
        "content_word_totals": {
            "compact_content_words": total_compact_content,
            "compact_copied_content_words": total_copied_content,
            "compact_source_absent_content_words": total_source_abs_content,
            "compact_tail_only_content_words": total_tail_only,
            "source_absent_fraction_of_compact_content": total_source_abs_content / total_compact_content if total_compact_content else None,
            "tail_only_fraction_of_compact_content": total_tail_only / total_compact_content if total_compact_content else None,
        },
        "token_totals": dict(token_totals),
        "source_position_decile_coverage": dec_cov,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "pair_id", "key", "source_words", "view_words", "ratio_view_to_source",
        "compact_source_content_coverage", "repeat_source_content_coverage", "compact_minus_repeat_content_coverage",
        "compact_tail_content_coverage", "compact_has_tail_content_recovery", "compact_content_fraction", "repeat_content_fraction",
        "compact_source_absent_content_words", "compact_source_absent_content_fraction_of_view", "compact_source_absent_content_fraction_of_content",
        "compact_copied_content_words", "compact_tail_only_content_words", "compact_tail_any_content_words",
        "compact_tail_only_content_fraction_of_content", "compact_content_jaccard_source", "repeat_content_jaccard_source",
        "compact_vs_repeat_content_jaccard", "compact_minus_repeat_active_tokens", "compact_minus_repeat_word_groups",
        "compact_minus_repeat_truncated",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def fmt_pct(x: Any) -> str:
    return "NA" if x is None else f"{100.0 * float(x):.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(args.pairs)
    if args.limit and args.limit > 0:
        pairs = pairs[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True, local_files_only=True)
    rows = [analyze_pair(tokenizer, r) for r in pairs]
    summary = aggregate(rows)
    payload = {
        "status": "NATURAL_COMPACT_REPEAT_MECHANISM_ATLAS",
        "created_utc": now_utc(),
        "meaning": "CPU/file/tokenizer atlas of the natural compact view versus matched first-N repeat view in the same 12,155 underlying compact-pair objects; supports interpretation of pending RoBERTa compact-vs-repeat transfer without touching running jobs. Exact packed-training-stream token counts remain in the research audit.",
        "inputs": {
            "pairs": str(args.pairs),
            "pairs_sha256": sha256_file(args.pairs),
            "tokenizer": str(args.tokenizer),
            "tokenizer_json_sha256": sha256_file(args.tokenizer / "tokenizer.json"),
            "limit": args.limit,
        },
        "summary": summary,
        "interpretation": {
            "natural_marginal_components": [
                "compact is shorter than the source and word-count matched to first-N repeat for each changed row",
                "compact reuses source content from later positions that prefix repeat cannot expose",
                "compact has a different content/function mix and some source-absent content words",
                "legal-tokenizer BPE/candidate-token load differs slightly and is part of the natural treatment",
                "saved words are reinvested in additional legal source diversity in the full 10M pool",
            ],
            "what_this_does_not_establish": "This atlas is not a model result and does not by itself prove downstream transfer, SOTA improvement, or a single causal factor; the pending RoBERTa and existing DeBERTa/GPT2 results remain the official-compatible evidence.",
        },
    }
    write_csv(args.out_dir / "pair_atlas.csv", rows)
    (args.out_dir / "mechanism_atlas.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    s = summary
    lines = [
        "# research natural compact-vs-repeat mechanism atlas",
        "",
        f"Created UTC: {payload['created_utc']}",
        "",
        "This is CPU/file/tokenizer analysis only. It did not touch running RoBERTa jobs. The rows here are underlying compact-pair objects, not packed changed JSONL rows.",
        "",
        "## Inputs",
        f"- pairs: `{args.pairs}` SHA `{payload['inputs']['pairs_sha256']}` rows {s['pairs']}",
        f"- legal research tokenizer: `{args.tokenizer}` tokenizer.json SHA `{payload['inputs']['tokenizer_json_sha256']}`",
        "",
        "## Main geometry",
        f"- Total source/view words in changed rows: {s['total_source_words']} / {s['total_view_words']} (ratio {s['view_to_source_ratio']:.4f})",
        f"- Mean compact source-content coverage: {fmt_pct(s['stats']['compact_source_content_coverage']['mean'])}",
        f"- Mean first-N repeat source-content coverage: {fmt_pct(s['stats']['repeat_source_content_coverage']['mean'])}",
        f"- Mean compact-repeat coverage delta: {fmt_pct(s['stats']['compact_minus_repeat_content_coverage']['mean'])}; positive-delta pairs {fmt_pct(s['fractions']['pairs_with_positive_content_coverage_delta'])}",
        f"- Mean compact tail-content coverage after the repeat length: {fmt_pct(s['stats']['compact_tail_content_coverage']['mean'])}; pairs with any tail recovery {fmt_pct(s['fractions']['pairs_with_tail_content_recovery'])}",
        f"- Compact content fraction {fmt_pct(s['stats']['compact_content_fraction']['mean'])} vs repeat {fmt_pct(s['stats']['repeat_content_fraction']['mean'])}",
        f"- Pairs with any source-absent compact content word: {fmt_pct(s['fractions']['pairs_with_any_source_absent_content'])}; source-absent fraction of compact content totals {fmt_pct(s['content_word_totals']['source_absent_fraction_of_compact_content'])}",
        f"- Pairs with any tail-only compact content word: {fmt_pct(s['fractions']['pairs_with_any_tail_only_content'])}; tail-only fraction of compact content totals {fmt_pct(s['content_word_totals']['tail_only_fraction_of_compact_content'])}",
        "",
        "## Legal-tokenizer load in changed rows",
        f"- individual pair-view encoding: compact active tokens {s['token_totals']['compact_active_tokens']}; repeat active tokens {s['token_totals']['repeat_active_tokens']}; compact-minus-repeat {s['token_totals']['compact_minus_repeat_active_tokens']}. Use research's packed-row audit as the exact training-stream number (+24000 active/candidate tokens per 10M pass; +240000 over 100M).",
        f"- compact word groups: {s['token_totals']['compact_word_groups']}; repeat word groups: {s['token_totals']['repeat_word_groups']}; compact-minus-repeat {s['token_totals']['compact_minus_repeat_word_groups']}",
        f"- compact/repeat truncated rows: {s['token_totals']['compact_truncated_rows']} / {s['token_totals']['repeat_truncated_rows']}",
        "",
        "## Source-position deciles",
        "| decile | content positions | compact coverage | repeat coverage | delta |",
        "|---:|---:|---:|---:|---:|",
    ]
    for d in s["source_position_decile_coverage"]:
        lines.append(f"| {d['decile']} | {d['source_content_positions']} | {fmt_pct(d['compact_coverage'])} | {fmt_pct(d['repeat_coverage'])} | {fmt_pct(d['delta'])} |")
    lines += [
        "",
        "## Scientific use",
        "This file names the bundled data marginal: compact changes source-position spread, content density, source-absent content, lexical overlap with repeat, and BPE/candidate-token exposure while the full pool reinvests saved words in extra source diversity. For exact training-stream BPE exposure use the research packed-row audit; this atlas decomposes the underlying pair-level source/view relation. It should prevent over-reading the pending RoBERTa result as a single-factor verdict.",
        "",
        "Artifacts: `mechanism_atlas.json` and `pair_atlas.csv` in this directory.",
    ]
    (args.out_dir / "mechanism_atlas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_dir": str(args.out_dir),
        "pairs": s["pairs"],
        "mean_compact_coverage": s["stats"]["compact_source_content_coverage"]["mean"],
        "mean_repeat_coverage": s["stats"]["repeat_source_content_coverage"]["mean"],
        "mean_tail_coverage": s["stats"]["compact_tail_content_coverage"]["mean"],
        "source_absent_fraction_of_compact_content": s["content_word_totals"]["source_absent_fraction_of_compact_content"],
        "compact_minus_repeat_active_tokens_changed_rows": s["token_totals"]["compact_minus_repeat_active_tokens"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
