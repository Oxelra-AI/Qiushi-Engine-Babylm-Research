#!/usr/bin/env python3
"""research: measure compact-view source-wide skeleton recurrence.

The goal is to determine whether the compact second view is mostly a noisy
semantic paraphrase, or whether it is a structured re-exposure of a source-wide
content skeleton.  The matched repeat arm repeats only the first N source words,
where N is the compact-view word count.  If the compact view copies many source
content words from positions beyond that prefix, then research's large copied-token
lift should be read as source-wide semantic-skeleton recurrence rather than just
trivial local duplication.

No training or model evaluation is performed here.  The measurement is purely on
legal paired text already used in the compact-view experiments.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import re
import statistics
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_PAIRS = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
DEFAULT_SELECTED = ROOT / "data/reciprocal_causal_transfer_scaffold_dosematched_shuffle/selected_reciprocal_pairs.jsonl"
DEFAULT_OUT = ROOT / "data/compact_skeleton_recurrence"
DEFAULT_FIG = ROOT / "figures/compact_skeleton_source_position_coverage.png"

# A deliberately simple closed stopword list: enough to separate content-bearing
# words from articles, auxiliaries, pronouns, and common connectives without using
# any external model or corpus statistics.
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
}

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)?")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_words(text: str) -> list[str]:
    # Use whitespace splitting for exact compatibility with the word-budget files.
    return [w for w in text.split() if w]


def norm_word(word: str) -> str:
    parts = WORD_RE.findall(word)
    if not parts:
        return ""
    return "".join(parts).lower()


def is_content_norm(norm: str) -> bool:
    if not norm:
        return False
    if norm in STOPWORDS:
        return False
    if norm.isdigit():
        return True
    return len(norm) >= 4


def multiset_counter(words: list[str]) -> collections.Counter[str]:
    c: collections.Counter[str] = collections.Counter()
    for w in words:
        n = norm_word(w)
        if n:
            c[n] += 1
    return c


def jaccard_multiset(a: collections.Counter[str], b: collections.Counter[str]) -> float | None:
    if not a and not b:
        return None
    keys = set(a) | set(b)
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return float(inter / union) if union else None


def read_pairs(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("source_text") and r.get("rewrite_text"):
                    rows.append(r)
    return rows


def read_selected_ids(path: pathlib.Path | None) -> set[str]:
    ids: set[str] = set()
    if path is None or not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r.get("pair_id")
            if pid:
                ids.add(str(pid))
    return ids


def safe_mean(xs: list[float]) -> float | None:
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(sum(ys) / len(ys)) if ys else None


def safe_median(xs: list[float]) -> float | None:
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(statistics.median(ys)) if ys else None


def pct(x: float | None) -> str:
    if x is None:
        return ""
    return f"{100.0 * float(x):.2f}%"


def f6(x: float | None) -> str:
    if x is None:
        return ""
    return f"{float(x):.6f}"


def measure_pair(r: dict[str, Any], selected_ids: set[str]) -> dict[str, Any]:
    source_words = split_words(r["source_text"])
    rewrite_words = split_words(r["rewrite_text"])
    repeat_len = int(r.get("rewrite_words") or len(rewrite_words))
    repeat_words = source_words[: min(repeat_len, len(source_words))]
    source_norms = [norm_word(w) for w in source_words]
    rewrite_norms = [norm_word(w) for w in rewrite_words]
    repeat_norms = [norm_word(w) for w in repeat_words]
    prefix_norms = set(n for n in source_norms[:repeat_len] if n)
    tail_norms = set(n for n in source_norms[repeat_len:] if n)
    rewrite_norm_set = set(n for n in rewrite_norms if n)
    repeat_norm_set = set(n for n in repeat_norms if n)

    source_len = len(source_words)
    content_positions = [i for i, n in enumerate(source_norms) if is_content_norm(n)]
    prefix_content_positions = [i for i in content_positions if i < repeat_len]
    tail_content_positions = [i for i in content_positions if i >= repeat_len]
    half_content_positions = [i for i in content_positions if i >= 0.5 * max(1, source_len)]

    compact_covered_content = [i for i in content_positions if source_norms[i] in rewrite_norm_set]
    repeat_covered_content = [i for i in content_positions if i < repeat_len and source_norms[i] in repeat_norm_set]
    compact_tail_covered = [i for i in tail_content_positions if source_norms[i] in rewrite_norm_set]
    compact_half_covered = [i for i in half_content_positions if source_norms[i] in rewrite_norm_set]

    compact_content_words = [n for n in rewrite_norms if is_content_norm(n)]
    repeat_content_words = [n for n in repeat_norms if is_content_norm(n)]
    compact_copied_content_words = [n for n in compact_content_words if n in set(source_norms)]
    compact_tail_only_words = [n for n in compact_content_words if n in tail_norms and n not in prefix_norms]
    compact_any_tail_words = [n for n in compact_content_words if n in tail_norms]

    # Source-position distribution of compact coverage: counts per decile among source content positions.
    decile_total = [0] * 10
    decile_compact = [0] * 10
    decile_repeat = [0] * 10
    for i in content_positions:
        d = min(9, int(10 * i / max(1, source_len)))
        decile_total[d] += 1
        if source_norms[i] in rewrite_norm_set:
            decile_compact[d] += 1
        if i < repeat_len:
            decile_repeat[d] += 1

    content_total = len(content_positions)
    tail_content_total = len(tail_content_positions)
    half_content_total = len(half_content_positions)
    compact_coverage = len(compact_covered_content) / content_total if content_total else None
    repeat_coverage = len(repeat_covered_content) / content_total if content_total else None
    tail_cov = len(compact_tail_covered) / tail_content_total if tail_content_total else None
    half_cov = len(compact_half_covered) / half_content_total if half_content_total else None
    rewrite_content_frac = len(compact_content_words) / len(rewrite_words) if rewrite_words else None
    repeat_content_frac = len(repeat_content_words) / len(repeat_words) if repeat_words else None
    rewrite_copy_jacc = jaccard_multiset(multiset_counter(source_words), multiset_counter(rewrite_words))
    repeat_copy_jacc = jaccard_multiset(multiset_counter(source_words), multiset_counter(repeat_words))

    max_compact_pos_frac = None
    mean_compact_pos_frac = None
    if compact_covered_content and source_len:
        max_compact_pos_frac = max(compact_covered_content) / source_len
        mean_compact_pos_frac = safe_mean([i / source_len for i in compact_covered_content])

    return {
        "pair_id": r.get("pair_id"),
        "selected_step178_179_subset": bool(r.get("pair_id") in selected_ids),
        "source_words": source_len,
        "rewrite_words": len(rewrite_words),
        "repeat_prefix_words": len(repeat_words),
        "content_source_positions": content_total,
        "prefix_content_positions": len(prefix_content_positions),
        "tail_content_positions": tail_content_total,
        "compact_content_source_coverage": compact_coverage,
        "repeat_prefix_content_source_coverage": repeat_coverage,
        "compact_minus_repeat_content_coverage": (compact_coverage - repeat_coverage) if compact_coverage is not None and repeat_coverage is not None else None,
        "compact_tail_content_coverage": tail_cov,
        "compact_late_half_content_coverage": half_cov,
        "compact_content_fraction": rewrite_content_frac,
        "repeat_content_fraction": repeat_content_frac,
        "compact_minus_repeat_content_fraction": (rewrite_content_frac - repeat_content_frac) if rewrite_content_frac is not None and repeat_content_frac is not None else None,
        "compact_copied_content_word_count": len(compact_copied_content_words),
        "compact_tail_only_content_word_count": len(compact_tail_only_words),
        "compact_any_tail_content_word_count": len(compact_any_tail_words),
        "compact_tail_only_content_word_fraction": len(compact_tail_only_words) / len(compact_content_words) if compact_content_words else None,
        "compact_any_tail_content_word_fraction": len(compact_any_tail_words) / len(compact_content_words) if compact_content_words else None,
        "compact_source_multiset_jaccard": rewrite_copy_jacc,
        "repeat_source_multiset_jaccard": repeat_copy_jacc,
        "compact_max_covered_content_pos_frac": max_compact_pos_frac,
        "compact_mean_covered_content_pos_frac": mean_compact_pos_frac,
        "decile_total": decile_total,
        "decile_compact": decile_compact,
        "decile_repeat": decile_repeat,
        "source_text": r.get("source_text"),
        "rewrite_text": r.get("rewrite_text"),
        "repeat_text": " ".join(repeat_words),
        "domain_hits": r.get("domain_hits"),
        "content_recall_field": r.get("content_recall"),
        "content_overlap_field": r.get("content_overlap"),
    }


def summarize_records(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    dec_total = [0] * 10
    dec_compact = [0] * 10
    dec_repeat = [0] * 10
    for r in records:
        for i in range(10):
            dec_total[i] += int(r["decile_total"][i])
            dec_compact[i] += int(r["decile_compact"][i])
            dec_repeat[i] += int(r["decile_repeat"][i])
    decile_rates = []
    for i in range(10):
        den = dec_total[i]
        decile_rates.append({
            "decile": i,
            "total_content_positions": den,
            "compact_covered_rate": dec_compact[i] / den if den else None,
            "repeat_prefix_covered_rate": dec_repeat[i] / den if den else None,
            "compact_minus_repeat": (dec_compact[i] - dec_repeat[i]) / den if den else None,
        })

    def vals(key: str) -> list[float]:
        out = []
        for r in records:
            v = r.get(key)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out.append(float(v))
        return out

    deltas = vals("compact_minus_repeat_content_coverage")
    tail_covs = vals("compact_tail_content_coverage")
    tail_only_counts = vals("compact_tail_only_content_word_count")
    any_tail_fracs = vals("compact_any_tail_content_word_fraction")
    content_frac_delta = vals("compact_minus_repeat_content_fraction")
    return {
        "label": label,
        "n_pairs": len(records),
        "mean_source_words": safe_mean(vals("source_words")),
        "mean_rewrite_words": safe_mean(vals("rewrite_words")),
        "mean_compact_content_source_coverage": safe_mean(vals("compact_content_source_coverage")),
        "mean_repeat_prefix_content_source_coverage": safe_mean(vals("repeat_prefix_content_source_coverage")),
        "mean_compact_minus_repeat_content_coverage": safe_mean(deltas),
        "median_compact_minus_repeat_content_coverage": safe_median(deltas),
        "frac_positive_content_coverage_delta": sum(1 for x in deltas if x > 0) / len(deltas) if deltas else None,
        "mean_compact_tail_content_coverage": safe_mean(tail_covs),
        "median_compact_tail_content_coverage": safe_median(tail_covs),
        "frac_pairs_with_tail_content_recovered": sum(1 for x in tail_covs if x > 0) / len(tail_covs) if tail_covs else None,
        "mean_tail_only_content_words_in_compact": safe_mean(tail_only_counts),
        "frac_pairs_with_strict_tail_only_compact_content": sum(1 for x in tail_only_counts if x > 0) / len(tail_only_counts) if tail_only_counts else None,
        "mean_any_tail_content_word_fraction_in_compact": safe_mean(any_tail_fracs),
        "mean_compact_content_fraction": safe_mean(vals("compact_content_fraction")),
        "mean_repeat_content_fraction": safe_mean(vals("repeat_content_fraction")),
        "mean_compact_minus_repeat_content_fraction": safe_mean(content_frac_delta),
        "frac_pairs_compact_more_content_dense": sum(1 for x in content_frac_delta if x > 0) / len(content_frac_delta) if content_frac_delta else None,
        "mean_compact_source_multiset_jaccard": safe_mean(vals("compact_source_multiset_jaccard")),
        "mean_repeat_source_multiset_jaccard": safe_mean(vals("repeat_source_multiset_jaccard")),
        "mean_compact_max_covered_content_pos_frac": safe_mean(vals("compact_max_covered_content_pos_frac")),
        "mean_compact_mean_covered_content_pos_frac": safe_mean(vals("compact_mean_covered_content_pos_frac")),
        "decile_content_coverage": decile_rates,
    }


def write_outputs(result: dict[str, Any], records: list[dict[str, Any]], out_dir: pathlib.Path, fig_path: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "compact_skeleton_recurrence.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = out_dir / "per_pair_skeleton_metrics.csv"
    fieldnames = [
        "pair_id", "selected_step178_179_subset", "source_words", "rewrite_words", "repeat_prefix_words",
        "content_source_positions", "prefix_content_positions", "tail_content_positions",
        "compact_content_source_coverage", "repeat_prefix_content_source_coverage",
        "compact_minus_repeat_content_coverage", "compact_tail_content_coverage", "compact_late_half_content_coverage",
        "compact_content_fraction", "repeat_content_fraction", "compact_minus_repeat_content_fraction",
        "compact_tail_only_content_word_count", "compact_any_tail_content_word_count",
        "compact_tail_only_content_word_fraction", "compact_any_tail_content_word_fraction",
        "compact_source_multiset_jaccard", "repeat_source_multiset_jaccard",
        "compact_max_covered_content_pos_frac", "compact_mean_covered_content_pos_frac",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in fieldnames})

    # Examples with strong late-source skeleton recurrence, avoiding huge text in JSON.
    examples = sorted(
        records,
        key=lambda r: (
            float(r.get("compact_tail_only_content_word_count") or 0),
            float(r.get("compact_tail_content_coverage") or 0),
            float(r.get("compact_minus_repeat_content_coverage") or -9),
        ),
        reverse=True,
    )[:30]
    ex_path = out_dir / "high_tail_skeleton_examples.jsonl"
    with ex_path.open("w", encoding="utf-8") as f:
        for r in examples:
            keep = {k: r.get(k) for k in [
                "pair_id", "source_words", "rewrite_words", "compact_tail_only_content_word_count",
                "compact_tail_content_coverage", "compact_minus_repeat_content_coverage",
                "compact_content_fraction", "repeat_content_fraction", "source_text", "rewrite_text", "repeat_text",
            ]}
            f.write(json.dumps(keep, ensure_ascii=False) + "\n")

    # Figure: aggregate source-position content coverage rates.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        all_summary = result["summaries"]["all_pairs"]
        subset_summary = result["summaries"].get("reference_179_selected_subset")
        xs = list(range(1, 11))
        fig, axes_obj = plt.subplots(1, 2 if subset_summary else 1, figsize=(12 if subset_summary else 6, 4), sharey=True)
        if hasattr(axes_obj, "flat"):
            axes = list(axes_obj.flat)
        elif isinstance(axes_obj, (list, tuple)):
            axes = list(axes_obj)
        else:
            axes = [axes_obj]
        for ax, summary, title in [(axes[0], all_summary, "all compact pairs")] + ([(axes[1], subset_summary, "2×2 selected subset")] if subset_summary else []):
            comp = [d["compact_covered_rate"] for d in summary["decile_content_coverage"]]
            rep = [d["repeat_prefix_covered_rate"] for d in summary["decile_content_coverage"]]
            ax.plot(xs, comp, marker="o", label="compact view covers source content")
            ax.plot(xs, rep, marker="s", label="prefix repeat covers source content")
            ax.set_title(title)
            ax.set_xlabel("source position decile")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("covered content-token fraction")
        fig.suptitle("Compact view re-exposes source-wide content; repeat arm only re-exposes prefix")
        fig.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=180)
        result["figure_path"] = str(fig_path)
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as e:
        result["figure_error"] = repr(e)
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_path = out_dir / "compact_skeleton_recurrence.md"
    lines = ["# research compact-view skeleton recurrence measurement", ""]
    lines.append("This is a CPU text measurement only: no training, no model scoring, and no leaderboard action.")
    lines.append("")
    lines.append(f"Pairs: `{result['pairs_path']}` SHA `{result['pairs_sha256']}`")
    if result.get("selected_subset_path"):
        lines.append(f"Selected subset: `{result['selected_subset_path']}` SHA `{result.get('selected_subset_sha256')}`")
    lines.append("")
    lines.append("## Aggregate measurements")
    lines.append("| subset | n | source words | view words | compact source-content coverage | repeat-prefix source-content coverage | compact-repeat coverage | pairs compact covers more | compact tail-content coverage | pairs with tail recovery | compact content fraction | repeat content fraction | compact-repeat content density | compact source Jaccard | repeat source Jaccard |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ["all_pairs", "reference_179_selected_subset"]:
        s = result["summaries"].get(key)
        if not s:
            continue
        lines.append(
            f"| {key} | {s['n_pairs']} | {f6(s.get('mean_source_words'))} | {f6(s.get('mean_rewrite_words'))} | "
            f"{pct(s.get('mean_compact_content_source_coverage'))} | {pct(s.get('mean_repeat_prefix_content_source_coverage'))} | "
            f"{pct(s.get('mean_compact_minus_repeat_content_coverage'))} | {pct(s.get('frac_positive_content_coverage_delta'))} | "
            f"{pct(s.get('mean_compact_tail_content_coverage'))} | {pct(s.get('frac_pairs_with_tail_content_recovered'))} | "
            f"{pct(s.get('mean_compact_content_fraction'))} | {pct(s.get('mean_repeat_content_fraction'))} | "
            f"{pct(s.get('mean_compact_minus_repeat_content_fraction'))} | {f6(s.get('mean_compact_source_multiset_jaccard'))} | {f6(s.get('mean_repeat_source_multiset_jaccard'))} |"
        )
    lines.append("")
    lines.append("## Source-position deciles")
    for key in ["all_pairs", "reference_179_selected_subset"]:
        s = result["summaries"].get(key)
        if not s:
            continue
        lines.append("")
        lines.append(f"### {key}")
        lines.append("| decile | content positions | compact covered | repeat-prefix covered | compact-repeat |")
        lines.append("|---:|---:|---:|---:|---:|")
        for d in s["decile_content_coverage"]:
            lines.append(f"| {d['decile']} | {d['total_content_positions']} | {pct(d.get('compact_covered_rate'))} | {pct(d.get('repeat_prefix_covered_rate'))} | {pct(d.get('compact_minus_repeat'))} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("The compact view should not be treated as mainly a source-free non-copy paraphrase if it re-exposes source content from across the full source while the repeat arm only re-exposes the prefix. In that case, research's copied-token lift is not merely a nuisance: it is part of a data mechanism in which a short view repeats selected source-wide content keys in a denser context. The still-open question for real training is whether this source-wide skeleton recurrence, rather than generated paraphrase style or causal/MLM topology alone, carries the DeBERTa compact-view gain.")
    lines.append("")
    if result.get("figure_path"):
        lines.append(f"Figure: `{result['figure_path']}`")
    if result.get("figure_error"):
        lines.append(f"Figure generation error: `{result['figure_error']}`")
    lines.append(f"Per-pair CSV: `{csv_path}`")
    lines.append(f"High-tail examples: `{ex_path}`")
    lines.append(f"JSON: `{json_path}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=pathlib.Path, default=DEFAULT_PAIRS)
    ap.add_argument("--selected-subset", type=pathlib.Path, default=DEFAULT_SELECTED)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--figure", type=pathlib.Path, default=DEFAULT_FIG)
    args = ap.parse_args()

    pairs = read_pairs(args.pairs)
    selected_ids = read_selected_ids(args.selected_subset)
    records = [measure_pair(r, selected_ids) for r in pairs]
    selected_records = [r for r in records if r["selected_step178_179_subset"]]
    result: dict[str, Any] = {
        "status": "COMPACT_SKELETON_RECURRENCE_MEASUREMENT",
        "pairs_path": str(args.pairs),
        "pairs_sha256": sha256_file(args.pairs),
        "available_pairs": len(pairs),
        "selected_subset_path": str(args.selected_subset) if args.selected_subset and args.selected_subset.exists() else None,
        "selected_subset_sha256": sha256_file(args.selected_subset) if args.selected_subset and args.selected_subset.exists() else None,
        "selected_pair_ids": len(selected_ids),
        "summaries": {
            "all_pairs": summarize_records(records, "all_pairs"),
            "reference_179_selected_subset": summarize_records(selected_records, "reference_179_selected_subset") if selected_records else None,
        },
        "measurement_notes": [
            "repeat arm is first N source words, with N equal to compact rewrite word count",
            "content word = normalized non-stopword with length at least 4, or a number",
            "tail means source positions after the repeated prefix length",
            "coverage counts source positions whose normalized token appears in the second view",
            "this is text geometry only; model competence still requires official-compatible training/evaluation evidence",
        ],
    }
    write_outputs(result, records, args.out_dir, args.figure)
    print(json.dumps({
        "status": result["status"],
        "out_dir": str(args.out_dir),
        "figure": result.get("figure_path"),
        "n_pairs": len(pairs),
        "selected_records": len(selected_records),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
