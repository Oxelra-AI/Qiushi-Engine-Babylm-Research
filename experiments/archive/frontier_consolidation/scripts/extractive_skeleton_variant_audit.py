#!/usr/bin/env python3
"""research: audit deterministic extractive skeleton variants.

This script asks whether the compact-view mechanism can be dissected into a
source-only, algorithmic skeleton: a same-length extractive view that re-exposes
content from across the source rather than only the first-N prefix.

It builds several exact-word-count views for each compact pair:
  - compact: existing generated compact rewrite (reference, not source-only)
  - prefix_repeat: first N source words (existing repeat control)
  - oracle_compact_projection: source words whose normalized forms appear in the
    compact view, filled deterministically to length N (upper-bound attribution;
    uses compact text, not source-only)
  - spread_even: N source words spread by source position (source-only)
  - content_spread: source-only content-heavy skeleton spread over source content
  - scored_source_skeleton: source-only scored selector emphasizing content,
    numbers/capitalized terms, uniqueness, and tail positions, then source order.

No model is loaded, trained, or evaluated.  Outputs are pair-level text assets and
geometry metrics for future route selection.
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
DEFAULT_OUT = ROOT / "data/extractive_skeleton_variant_audit"
DEFAULT_FIG = ROOT / "figures/extractive_skeleton_variant_deciles.png"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)?")
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so", "because", "as", "than",
    "to", "of", "in", "on", "for", "with", "without", "by", "from", "at", "into", "onto", "over",
    "under", "about", "between", "among", "through", "during", "before", "after", "above", "below",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "doing",
    "have", "has", "had", "having", "can", "could", "may", "might", "must", "shall", "should", "will",
    "would", "this", "that", "these", "those", "there", "here", "it", "its", "they", "them", "their",
    "theirs", "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours", "you", "your",
    "yours", "i", "me", "my", "mine", "who", "whom", "whose", "which", "what", "where", "when",
    "why", "how", "not", "no", "nor", "only", "just", "also", "very", "more", "most", "less", "least",
    "much", "many", "some", "any", "all", "each", "every", "other", "another", "such", "own", "same",
    "too", "again", "still", "already", "yet", "up", "down", "out", "off", "back", "away",
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_word(word: str) -> str:
    parts = WORD_RE.findall(word)
    if not parts:
        return ""
    return "".join(parts).lower()


def is_content(n: str) -> bool:
    return bool(n) and (n.isdigit() or (len(n) >= 4 and n not in STOPWORDS))


def words(text: str) -> list[str]:
    return [w for w in text.split() if w]


def pick_even(indices: list[int], k: int) -> list[int]:
    if k <= 0 or not indices:
        return []
    if k >= len(indices):
        return list(indices)
    chosen = []
    used = set()
    for j in range(k):
        pos = round(j * (len(indices) - 1) / max(1, k - 1))
        # Move outward if quantile collision occurs.
        best = None
        for d in range(len(indices)):
            for p in (pos - d, pos + d):
                if 0 <= p < len(indices) and p not in used:
                    best = p
                    break
            if best is not None:
                break
        used.add(best)
        chosen.append(indices[best])
    return sorted(chosen)


def fill_to_k(chosen: set[int], n_source: int, k: int, preference: list[int] | None = None) -> list[int]:
    if n_source <= 0:
        return []
    pref = preference or []
    for i in pref:
        if len(chosen) >= k:
            break
        if 0 <= i < n_source:
            chosen.add(i)
    # Deterministically fill by nearest unchosen positions to preserve some glue.
    if len(chosen) < k:
        # Prefer positions adjacent to current chosen set, then source order.
        candidates = []
        for i in range(n_source):
            if i in chosen:
                continue
            dist = min((abs(i - j) for j in chosen), default=0)
            candidates.append((dist, i))
        for _, i in sorted(candidates):
            if len(chosen) >= k:
                break
            chosen.add(i)
    if len(chosen) > k:
        # Drop least preferred by a stable centrality/source-order rule.
        ranked = sorted(chosen, key=lambda i: (0 if i in pref else 1, i))[:k]
        chosen = set(ranked)
    return sorted(chosen)


def prefix_repeat(src: list[str], k: int) -> list[int]:
    return list(range(min(k, len(src))))


def oracle_compact_projection(src: list[str], compact: list[str], k: int) -> list[int]:
    compact_norms = {norm_word(w) for w in compact if norm_word(w)}
    content_hits = [i for i, w in enumerate(src) if norm_word(w) in compact_norms and is_content(norm_word(w))]
    all_hits = [i for i, w in enumerate(src) if norm_word(w) in compact_norms]
    chosen = set(content_hits[:k])
    chosen.update(all_hits[: max(k, len(chosen))])
    return fill_to_k(chosen, len(src), k, preference=all_hits + content_hits)


def spread_even(src: list[str], k: int) -> list[int]:
    return pick_even(list(range(len(src))), min(k, len(src)))


def content_spread(src: list[str], k: int) -> list[int]:
    norms = [norm_word(w) for w in src]
    content = [i for i, n in enumerate(norms) if is_content(n)]
    if not content:
        return spread_even(src, k)
    # Use about 80% content positions, then add glue nearby to improve local readability.
    target_content = min(len(content), max(1, round(0.80 * min(k, len(src)))))
    chosen = set(pick_even(content, target_content))
    pref: list[int] = []
    for i in sorted(chosen):
        pref.extend([i - 1, i + 1, i - 2, i + 2])
    pref.extend(pick_even(list(range(len(src))), min(k, len(src))))
    return fill_to_k(chosen, len(src), k, preference=pref)


def scored_source_skeleton(src: list[str], k: int) -> list[int]:
    norms = [norm_word(w) for w in src]
    counts = collections.Counter(n for n in norms if n)
    n_source = len(src)
    scored = []
    for i, (w, n) in enumerate(zip(src, norms)):
        frac = i / max(1, n_source - 1)
        score = 0.0
        if is_content(n):
            score += 4.0
        if n.isdigit() if n else False:
            score += 1.5
        if w[:1].isupper() and i != 0:
            score += 1.0
        if counts.get(n, 0) == 1 and is_content(n):
            score += 0.8
        # Tail boost: force source-wide rather than prefix-only, but not enough to
        # ignore all early content.
        score += 1.25 * frac
        # Retain a little grammatical glue around important words.
        if n in {"because", "after", "before", "during", "without", "between", "among", "according", "while"}:
            score += 1.0
        if not n:
            score -= 0.5
        if n in STOPWORDS:
            score -= 0.7
        scored.append((-score, i))
    chosen = {i for _, i in sorted(scored)[: min(k, n_source)]}
    # Add adjacent glue only if we still have room; usually no room.
    pref: list[int] = []
    for i in sorted(chosen):
        pref.extend([i - 1, i + 1])
    return fill_to_k(chosen, n_source, k, preference=pref)


def make_text(src: list[str], idxs: list[int]) -> str:
    return " ".join(src[i] for i in idxs if 0 <= i < len(src))


def norm_counter(ws: list[str]) -> collections.Counter[str]:
    c = collections.Counter()
    for w in ws:
        n = norm_word(w)
        if n:
            c[n] += 1
    return c


def jaccard(a: collections.Counter[str], b: collections.Counter[str]) -> float | None:
    keys = set(a) | set(b)
    if not keys:
        return None
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else None


def longest_gap(idxs: list[int]) -> int | None:
    if len(idxs) < 2:
        return None
    return max(b - a for a, b in zip(idxs, idxs[1:]))


def safe_mean(xs: list[float]) -> float | None:
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(ys) / len(ys) if ys else None


def safe_median(xs: list[float]) -> float | None:
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return statistics.median(ys) if ys else None


def evaluate_variant(pair: dict[str, Any], variant: str, idxs: list[int], text_words: list[str], compact_words: list[str], prefix_idxs: list[int]) -> dict[str, Any]:
    src = words(pair["source_text"])
    source_norms = [norm_word(w) for w in src]
    text_norms = [norm_word(w) for w in text_words]
    text_norm_set = {n for n in text_norms if n}
    content_positions = [i for i, n in enumerate(source_norms) if is_content(n)]
    repeat_len = len(compact_words)
    tail_content = [i for i in content_positions if i >= repeat_len]
    covered = [i for i in content_positions if source_norms[i] in text_norm_set]
    tail_cov = [i for i in tail_content if source_norms[i] in text_norm_set]
    text_content = [n for n in text_norms if is_content(n)]
    compact_counter = norm_counter(compact_words)
    text_counter = norm_counter(text_words)
    source_counter = norm_counter(src)
    dec_total = [0] * 10
    dec_cov = [0] * 10
    for i in content_positions:
        d = min(9, int(10 * i / max(1, len(src))))
        dec_total[d] += 1
        if source_norms[i] in text_norm_set:
            dec_cov[d] += 1
    return {
        "pair_id": pair.get("pair_id"),
        "variant": variant,
        "source_words": len(src),
        "view_words": len(text_words),
        "target_words": repeat_len,
        "source_content_positions": len(content_positions),
        "tail_content_positions": len(tail_content),
        "source_content_coverage": len(covered) / len(content_positions) if content_positions else None,
        "tail_content_coverage": len(tail_cov) / len(tail_content) if tail_content else None,
        "content_fraction": len(text_content) / len(text_words) if text_words else None,
        "jaccard_with_compact": jaccard(text_counter, compact_counter),
        "jaccard_with_source": jaccard(text_counter, source_counter),
        "mean_selected_source_pos_frac": safe_mean([i / max(1, len(src)) for i in idxs]) if idxs else None,
        "max_selected_source_pos_frac": max([i / max(1, len(src)) for i in idxs]) if idxs else None,
        "longest_source_position_gap": longest_gap(sorted(idxs)),
        "selected_prefix_fraction": sum(1 for i in idxs if i < repeat_len) / len(idxs) if idxs else None,
        "selected_tail_fraction": sum(1 for i in idxs if i >= repeat_len) / len(idxs) if idxs else None,
        "decile_total": dec_total,
        "decile_covered": dec_cov,
        "text": " ".join(text_words),
    }


def read_pairs(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("source_text") and r.get("rewrite_text"):
                    rows.append(r)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_var: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_var[r["variant"]].append(r)
    out = {}
    for v, rs in sorted(by_var.items()):
        def vals(k: str) -> list[float]:
            return [float(r[k]) for r in rs if isinstance(r.get(k), (int, float)) and math.isfinite(float(r[k]))]
        dec_total = [0] * 10
        dec_cov = [0] * 10
        for r in rs:
            for i in range(10):
                dec_total[i] += int(r["decile_total"][i])
                dec_cov[i] += int(r["decile_covered"][i])
        out[v] = {
            "n": len(rs),
            "mean_source_content_coverage": safe_mean(vals("source_content_coverage")),
            "median_source_content_coverage": safe_median(vals("source_content_coverage")),
            "mean_tail_content_coverage": safe_mean(vals("tail_content_coverage")),
            "median_tail_content_coverage": safe_median(vals("tail_content_coverage")),
            "frac_pairs_with_tail_recovery": sum(1 for x in vals("tail_content_coverage") if x > 0) / len(vals("tail_content_coverage")) if vals("tail_content_coverage") else None,
            "mean_content_fraction": safe_mean(vals("content_fraction")),
            "mean_jaccard_with_compact": safe_mean(vals("jaccard_with_compact")),
            "mean_jaccard_with_source": safe_mean(vals("jaccard_with_source")),
            "mean_selected_prefix_fraction": safe_mean(vals("selected_prefix_fraction")),
            "mean_selected_tail_fraction": safe_mean(vals("selected_tail_fraction")),
            "mean_longest_source_position_gap": safe_mean(vals("longest_source_position_gap")),
            "decile_coverage": [
                {"decile": i, "total": dec_total[i], "covered_rate": (dec_cov[i] / dec_total[i] if dec_total[i] else None)}
                for i in range(10)
            ],
        }
    # Deltas vs prefix repeat and compact reference.
    if "prefix_repeat" in out:
        base = out["prefix_repeat"]
        for v, s in out.items():
            s["delta_source_content_coverage_vs_prefix"] = (s["mean_source_content_coverage"] - base["mean_source_content_coverage"]) if s.get("mean_source_content_coverage") is not None else None
            s["delta_tail_content_coverage_vs_prefix"] = (s["mean_tail_content_coverage"] - base["mean_tail_content_coverage"]) if s.get("mean_tail_content_coverage") is not None else None
            s["delta_content_fraction_vs_prefix"] = (s["mean_content_fraction"] - base["mean_content_fraction"]) if s.get("mean_content_fraction") is not None else None
    if "compact" in out:
        comp = out["compact"]
        for v, s in out.items():
            s["delta_source_content_coverage_vs_compact"] = (s["mean_source_content_coverage"] - comp["mean_source_content_coverage"]) if s.get("mean_source_content_coverage") is not None else None
            s["delta_tail_content_coverage_vs_compact"] = (s["mean_tail_content_coverage"] - comp["mean_tail_content_coverage"]) if s.get("mean_tail_content_coverage") is not None else None
            s["delta_content_fraction_vs_compact"] = (s["mean_content_fraction"] - comp["mean_content_fraction"]) if s.get("mean_content_fraction") is not None else None
    return out


def fmt(x: Any, pct: bool = False) -> str:
    if x is None:
        return ""
    return f"{100*float(x):.2f}%" if pct else f"{float(x):.6f}"


def write_figure(summary: dict[str, Any], fig_path: pathlib.Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = ["compact", "prefix_repeat", "spread_even", "content_spread", "scored_source_skeleton", "oracle_compact_projection"]
        xs = list(range(1, 11))
        fig, ax = plt.subplots(figsize=(9, 5))
        for v in order:
            if v not in summary:
                continue
            ys = [d["covered_rate"] for d in summary[v]["decile_coverage"]]
            ax.plot(xs, ys, marker="o", label=v)
        ax.set_xlabel("source position decile")
        ax.set_ylabel("source content coverage")
        ax.set_title("Exact-length source-derived skeletons: content coverage by source position")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=180)
        return str(fig_path)
    except Exception as e:
        return f"ERROR:{e!r}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=pathlib.Path, default=DEFAULT_PAIRS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--figure", type=pathlib.Path, default=DEFAULT_FIG)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(args.pairs)
    metrics: list[dict[str, Any]] = []
    variant_jsonl: dict[str, Any] = {}
    variant_names = ["compact", "prefix_repeat", "oracle_compact_projection", "spread_even", "content_spread", "scored_source_skeleton"]
    writers = {}
    for v in variant_names:
        path = args.out_dir / f"{v}_pairs.jsonl"
        f = path.open("w", encoding="utf-8")
        writers[v] = (f, path)
    try:
        for r in pairs:
            src = words(r["source_text"])
            comp = words(r["rewrite_text"])
            k = len(comp)
            idx_variants = {
                "prefix_repeat": prefix_repeat(src, k),
                "oracle_compact_projection": oracle_compact_projection(src, comp, k),
                "spread_even": spread_even(src, k),
                "content_spread": content_spread(src, k),
                "scored_source_skeleton": scored_source_skeleton(src, k),
            }
            text_variants = {"compact": comp}
            text_variants.update({v: [src[i] for i in idxs] for v, idxs in idx_variants.items()})
            idx_for_compact = []
            comp_norms = [norm_word(w) for w in comp]
            for i, n in enumerate([norm_word(w) for w in src]):
                if n and n in set(comp_norms):
                    idx_for_compact.append(i)
            for v, tw in text_variants.items():
                idxs = idx_for_compact if v == "compact" else idx_variants[v]
                m = evaluate_variant(r, v, idxs, tw, comp, idx_variants["prefix_repeat"])
                metrics.append(m)
                out_row = {
                    "pair_id": r.get("pair_id"),
                    "key": r.get("key"),
                    "source_text": r.get("source_text"),
                    "view_text": m["text"],
                    "variant": v,
                    "source_words": len(src),
                    "view_words": len(tw),
                    "pair_words": len(src) + len(tw),
                    "original_rewrite_text": r.get("rewrite_text"),
                    "metrics": {k2: m.get(k2) for k2 in ["source_content_coverage", "tail_content_coverage", "content_fraction", "jaccard_with_compact", "selected_tail_fraction"]},
                }
                f, _ = writers[v]
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    finally:
        for f, _ in writers.values():
            f.close()
    summary = summarize(metrics)
    fig_result = write_figure(summary, args.figure)
    result = {
        "status": "EXTRACTIVE_SKELETON_VARIANT_AUDIT",
        "pairs_path": str(args.pairs),
        "pairs_sha256": sha256_file(args.pairs),
        "n_pairs": len(pairs),
        "variant_pair_files": {v: str(path) for v, (_, path) in writers.items()},
        "variant_pair_file_sha256": {v: sha256_file(path) for v, (_, path) in writers.items()},
        "summary": summary,
        "figure_path": fig_result if fig_result and not fig_result.startswith("ERROR:") else None,
        "figure_error": fig_result if fig_result and fig_result.startswith("ERROR:") else None,
        "notes": [
            "all variants use the compact rewrite word count as target length",
            "oracle_compact_projection uses the compact rewrite and is not a legal source-only algorithm; it is an attribution upper bound",
            "spread_even, content_spread, and scored_source_skeleton are deterministic source-only exact-length variants",
            "pair files are not full 10M training pools and have not been trained or evaluated",
        ],
    }
    out_json = args.out_dir / "extractive_skeleton_variant_audit.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # CSV without huge text.
    out_csv = args.out_dir / "extractive_skeleton_variant_metrics.csv"
    fieldnames = [k for k in metrics[0].keys() if k not in {"text", "decile_total", "decile_covered"}]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in metrics:
            w.writerow({k: r.get(k) for k in fieldnames})
    # Examples where non-oracle source-only skeletons recover substantial tail material.
    ex_path = args.out_dir / "source_only_skeleton_examples.jsonl"
    source_only = [m for m in metrics if m["variant"] in {"content_spread", "scored_source_skeleton"}]
    examples = sorted(source_only, key=lambda m: ((m.get("tail_content_coverage") or 0), (m.get("source_content_coverage") or 0), (m.get("jaccard_with_compact") or 0)), reverse=True)[:50]
    pair_lookup = {r.get("pair_id"): r for r in pairs}
    with ex_path.open("w", encoding="utf-8") as f:
        for m in examples:
            p = pair_lookup.get(m["pair_id"], {})
            f.write(json.dumps({
                "pair_id": m["pair_id"],
                "variant": m["variant"],
                "source_text": p.get("source_text"),
                "compact_text": p.get("rewrite_text"),
                "skeleton_text": m["text"],
                "source_content_coverage": m.get("source_content_coverage"),
                "tail_content_coverage": m.get("tail_content_coverage"),
                "content_fraction": m.get("content_fraction"),
                "jaccard_with_compact": m.get("jaccard_with_compact"),
            }, ensure_ascii=False) + "\n")
    md_path = args.out_dir / "extractive_skeleton_variant_audit.md"
    lines = ["# research extractive skeleton variant audit", ""]
    lines.append("No training/evaluation. This audits exact-length source-derived variants as possible dissection of the compact-view data mechanism.")
    lines.append("")
    lines.append("## Aggregate geometry")
    lines.append("| variant | n | source-content coverage | tail-content coverage | pairs with tail recovery | content fraction | Jaccard with compact | Jaccard with source | selected tail fraction | Δ source coverage vs prefix | Δ tail coverage vs compact |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    order = ["compact", "prefix_repeat", "spread_even", "content_spread", "scored_source_skeleton", "oracle_compact_projection"]
    for v in order:
        s = summary.get(v)
        if not s:
            continue
        lines.append(
            f"| {v} | {s['n']} | {fmt(s.get('mean_source_content_coverage'), True)} | {fmt(s.get('mean_tail_content_coverage'), True)} | "
            f"{fmt(s.get('frac_pairs_with_tail_recovery'), True)} | {fmt(s.get('mean_content_fraction'), True)} | "
            f"{fmt(s.get('mean_jaccard_with_compact'))} | {fmt(s.get('mean_jaccard_with_source'))} | "
            f"{fmt(s.get('mean_selected_tail_fraction'), True)} | {fmt(s.get('delta_source_content_coverage_vs_prefix'), True)} | "
            f"{fmt(s.get('delta_tail_content_coverage_vs_compact'), True)} |"
        )
    lines.append("")
    lines.append("## Source-position decile coverage")
    for v in order:
        s = summary.get(v)
        if not s:
            continue
        lines.append("")
        lines.append(f"### {v}")
        lines.append("| decile | content positions | covered |")
        lines.append("|---:|---:|---:|")
        for d in s["decile_coverage"]:
            lines.append(f"| {d['decile']} | {d['total']} | {fmt(d.get('covered_rate'), True)} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("If a source-only variant approaches compact tail coverage and content density while remaining exact-length, then a future MLM training dissection can separate source-wide skeleton recurrence from generated paraphrase style. If only the oracle projection approaches compact, then the compact generator is doing nontrivial content selection. If source-only variants recover tail content but look distributionally unnatural, the lowest-cost future test should be a short MLM screen against prefix repeat and compact before any mature run.")
    lines.append("")
    if result.get("figure_path"):
        lines.append(f"Figure: `{result['figure_path']}`")
    if result.get("figure_error"):
        lines.append(f"Figure error: `{result['figure_error']}`")
    lines.append(f"Metrics CSV: `{out_csv}`")
    lines.append(f"Example skeletons: `{ex_path}`")
    lines.append("Variant pair JSONL files:")
    for v in order:
        if v in result["variant_pair_files"]:
            lines.append(f"- {v}: `{result['variant_pair_files'][v]}` SHA `{result['variant_pair_file_sha256'][v]}`")
    lines.append(f"JSON: `{out_json}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_dir": str(args.out_dir), "n_pairs": len(pairs), "figure": result.get("figure_path")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
