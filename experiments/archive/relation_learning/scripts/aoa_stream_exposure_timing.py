#!/usr/bin/env python3
"""research: offline AoA exposure-timing analysis for the v4/coherent86 stream.

This script does not train or evaluate a model. It asks whether the official AoA
failure is predictable from the temporal order of lexical exposure in the actual
100M-word stream used by the inherited strict-small trunk, and whether legal
reorderings of the same multiset could in principle align acquisition timing
with child AoA better than the current order.

The lexical exposure instrument is deliberately simple and transparent: exact
lower-cased alphabetic word-boundary counts of CDI words in the JSONL `text`
field, accumulated along nominal word positions from the stream's `words` field.
It is a route-selection instrument, not an official scoring implementation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import bisect
import csv
import json
import math
import pathlib
import re
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = _public_path('.')
DEFAULT_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
DEFAULT_ORDER_MANIFEST = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/example_order_manifest.json')
DEFAULT_SCI_METRICS = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/scientific_metrics.json')
DEFAULT_WORD_RECORDS = _public_path('experiments/archive/relation_learning/data/aoa_curve_fit_analysis/word_curve_fit_records.csv')
DEFAULT_CDI = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')
DEFAULT_CHANG = _public_path('data/external/object.md')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/aoa_stream_exposure_timing')

LADDER_CHECKPOINTS = [x * 1_000_000 for x in range(1, 11)] + [x * 1_000_000 for x in range(20, 101, 10)]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_float(x: Any) -> float | None:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def corr(xs: list[float], ys: list[float]) -> dict[str, Any]:
    vals = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(vals) < 3 or len(set(x for x, _ in vals)) < 2 or len(set(y for _, y in vals)) < 2:
        return {"n": len(vals), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    x = [a for a, _ in vals]
    y = [b for _, b in vals]
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    return {"n": len(vals), "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue), "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue)}


def load_cdi_words(cdi_path: pathlib.Path) -> list[str]:
    df = pd.read_csv(cdi_path)
    return [str(w).lower() for w in df["word"].tolist() if isinstance(w, str) and w]


def load_word_records(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            word = str(row.get("word", "")).lower()
            if word:
                rows[word] = dict(row)
    return rows


def extract_saved_checkpoints(metrics_path: pathlib.Path) -> list[int]:
    try:
        metrics = read_json(metrics_path)
        vals = []
        for r in metrics.get("saved_checkpoints", []):
            name = str(r.get("name", ""))
            m = re.search(r"chck_(\d+)M", name)
            if m and int(m.group(1)) in set([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]):
                vals.append(int(r.get("actual_cumulative_word_exposure", int(m.group(1))*1_000_000)))
        if vals:
            return sorted(vals)
    except Exception:
        pass
    return LADDER_CHECKPOINTS


def stream_word_counts(stream_path: pathlib.Path, target_words: set[str], checkpoint_bounds: list[int], max_rows: int = 0) -> dict[str, Any]:
    word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
    total_by_word: Counter[str] = Counter()
    source_by_word: dict[str, Counter[str]] = {w: Counter() for w in target_words}
    checkpoint_bins: dict[str, list[int]] = {w: [0 for _ in checkpoint_bounds] for w in target_words}
    source_word_exposure: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()
    source_first_pos: dict[str, int] = {}
    source_last_pos: dict[str, int] = {}
    rows = 0
    consumed_words = 0
    t0 = time.time()
    with stream_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            src = str(obj.get("source", "unknown"))
            try:
                row_words = int(obj.get("words", len(text.split())))
            except Exception:
                row_words = len(text.split())
            start_pos = consumed_words
            consumed_words += row_words
            end_pos = consumed_words
            source_word_exposure[src] += row_words
            source_rows[src] += 1
            source_first_pos.setdefault(src, start_pos)
            source_last_pos[src] = end_pos
            bin_idx = bisect.bisect_left(checkpoint_bounds, end_pos)
            toks = [m.group(0).lower() for m in word_re.finditer(text)]
            counts = Counter(t for t in toks if t in target_words)
            for w, c in counts.items():
                total_by_word[w] += c
                source_by_word[w][src] += c
                if 0 <= bin_idx < len(checkpoint_bounds):
                    checkpoint_bins[w][bin_idx] += c
            if max_rows and rows >= max_rows:
                break
            if rows % 100000 == 0:
                print(json.dumps({"event": "stream_progress", "rows": rows, "consumed_words": consumed_words, "elapsed_sec": round(time.time() - t0, 1)}), flush=True)
    return {
        "rows": rows,
        "consumed_words": consumed_words,
        "total_by_word": total_by_word,
        "source_by_word": source_by_word,
        "checkpoint_bins": checkpoint_bins,
        "source_word_exposure": source_word_exposure,
        "source_rows": source_rows,
        "source_first_pos": source_first_pos,
        "source_last_pos": source_last_pos,
        "elapsed_sec": time.time() - t0,
    }


def crossing_from_bins(bins: list[int], bounds: list[int], total_count: int) -> dict[str, Any]:
    if total_count <= 0:
        return {"status": "absent", "cross_word_pos": None, "cross_log10_pos": None, "cross_checkpoint_index": None}
    half = total_count / 2.0
    cum = 0
    prev_cum = 0
    prev_bound = 0
    for i, c in enumerate(bins):
        prev_cum = cum
        cum += int(c)
        if cum >= half:
            bound = bounds[i]
            if c > 0:
                frac = (half - prev_cum) / c
                frac = max(0.0, min(1.0, frac))
            else:
                frac = 0.0
            pos = prev_bound + frac * (bound - prev_bound)
            return {"status": "ok", "cross_word_pos": float(pos), "cross_log10_pos": float(math.log10(max(pos, 1.0))), "cross_checkpoint_index": i, "cross_checkpoint_bound": bound}
        prev_bound = bounds[i]
    return {"status": "after_last_or_bin_loss", "cross_word_pos": None, "cross_log10_pos": None, "cross_checkpoint_index": None, "final_cum_in_bins": cum}


def crossing_for_source_order(word_source_counts: Counter[str], source_words: Counter[str], source_order: list[str], n_passes: int, total_count: int) -> dict[str, Any]:
    if total_count <= 0:
        return {"status": "absent", "cross_word_pos": None, "cross_log10_pos": None}
    all_sources = [s for s in source_order if s in source_words] + [s for s in sorted(source_words) if s not in set(source_order)]
    pass_word_len = sum(source_words.values()) / max(1, n_passes)
    half = total_count / 2.0
    cum_count = 0.0
    pos = 0.0
    for _pass in range(n_passes):
        for src in all_sources:
            sw = source_words.get(src, 0) / max(1, n_passes)
            wc = word_source_counts.get(src, 0) / max(1, n_passes)
            if wc <= 0:
                pos += sw
                continue
            if cum_count + wc >= half:
                frac = (half - cum_count) / wc
                cross = pos + max(0.0, min(1.0, frac)) * sw
                return {"status": "ok", "cross_word_pos": float(cross), "cross_log10_pos": float(math.log10(max(cross, 1.0))), "source_at_cross": src}
            cum_count += wc
            pos += sw
    return {"status": "after_last", "cross_word_pos": None, "cross_log10_pos": None, "final_cum": cum_count, "pass_word_len": pass_word_len}


def make_candidate_orders(source_words: Counter[str]) -> dict[str, list[str]]:
    sources = list(source_words.keys())
    # Explicit names are those present in the v4 manifest. Missing names are appended later.
    childes_first = ["childes", "bnc_spoken", "switchboard", "open_subtitles", "simple_wiki", "qwen_pair_packed", "cleanqwen_fineweb_compact_view_reinvest", "neutral_cleanqwen_topup_compact_reinvest::open_subtitles", "gutenberg"]
    cdi_spoken_then_simple = ["childes", "bnc_spoken", "switchboard", "simple_wiki", "open_subtitles", "qwen_pair_packed", "cleanqwen_fineweb_compact_view_reinvest", "neutral_cleanqwen_topup_compact_reinvest::open_subtitles", "gutenberg"]
    written_first = ["gutenberg", "simple_wiki", "cleanqwen_fineweb_compact_view_reinvest", "qwen_pair_packed", "open_subtitles", "bnc_spoken", "switchboard", "childes", "neutral_cleanqwen_topup_compact_reinvest::open_subtitles"]
    qwen_first = ["qwen_pair_packed", "cleanqwen_fineweb_compact_view_reinvest", "childes", "open_subtitles", "gutenberg", "simple_wiki", "bnc_spoken", "switchboard", "neutral_cleanqwen_topup_compact_reinvest::open_subtitles"]
    # A pure source-frequency curriculum: source blocks sorted by average early-CDI enrichment will be filled after scan if needed.
    size_desc = [s for s, _ in sorted(source_words.items(), key=lambda kv: (-kv[1], kv[0]))]
    size_asc = [s for s, _ in sorted(source_words.items(), key=lambda kv: (kv[1], kv[0]))]
    return {
        "childes_spoken_first": childes_first,
        "childes_bnc_simple_first": cdi_spoken_then_simple,
        "written_first_control": written_first,
        "qwen_clean_first_control": qwen_first,
        "largest_sources_first": size_desc,
        "smallest_sources_first": size_asc,
        "alphabetical_sources": sorted(sources),
    }


def source_enrichment_order(word_records: dict[str, dict[str, Any]], source_by_word: dict[str, Counter[str]], source_words: Counter[str], words: list[str]) -> list[str]:
    # Rank sources by how strongly log source frequency predicts earlier child AoA among words available in that source.
    scores = []
    for src, sw in source_words.items():
        xs = []
        ys = []
        for w in words:
            child = safe_float(word_records.get(w, {}).get("child_aoa"))
            if child is None:
                continue
            c = source_by_word.get(w, Counter()).get(src, 0)
            xs.append(math.log((c + 0.5) / max(sw, 1) * 1_000_000))
            ys.append(child)
        cr = corr(xs, ys)
        r = cr.get("pearson_r")
        # Negative frequency-childAoA r means words frequent in this source are acquired earlier by children; sort first.
        scores.append((r if r is not None else 1.0, src, cr))
    return [src for _, src, _ in sorted(scores, key=lambda x: (x[0], x[1]))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stream", default=str(DEFAULT_STREAM))
    ap.add_argument("--order-manifest", default=str(DEFAULT_ORDER_MANIFEST))
    ap.add_argument("--scientific-metrics", default=str(DEFAULT_SCI_METRICS))
    ap.add_argument("--word-records", default=str(DEFAULT_WORD_RECORDS))
    ap.add_argument("--cdi-human", default=str(DEFAULT_CDI))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_path = pathlib.Path(args.stream)
    cdi_words = load_cdi_words(pathlib.Path(args.cdi_human))
    word_records = load_word_records(pathlib.Path(args.word_records))
    target_words = sorted(set(cdi_words) | set(word_records))
    checkpoint_bounds = extract_saved_checkpoints(pathlib.Path(args.scientific_metrics))
    print(json.dumps({"event": "start_scan", "stream": rel(stream_path), "target_words": len(target_words), "checkpoints": checkpoint_bounds[:5] + checkpoint_bounds[-3:], "utc": now()}), flush=True)
    scan = stream_word_counts(stream_path, set(target_words), checkpoint_bounds, max_rows=args.max_rows)
    print(json.dumps({"event": "scan_done", "rows": scan["rows"], "consumed_words": scan["consumed_words"], "elapsed_sec": round(scan["elapsed_sec"], 1)}), flush=True)

    source_words: Counter[str] = scan["source_word_exposure"]
    total_by_word: Counter[str] = scan["total_by_word"]
    source_by_word: dict[str, Counter[str]] = scan["source_by_word"]
    checkpoint_bins: dict[str, list[int]] = scan["checkpoint_bins"]
    n_passes = max(1, round(scan["consumed_words"] / 10_000_000))

    candidate_orders = make_candidate_orders(source_words)
    candidate_orders["source_child_aoa_enrichment_first"] = source_enrichment_order(word_records, source_by_word, source_words, target_words)

    word_rows: list[dict[str, Any]] = []
    for w in sorted(target_words):
        rec = word_records.get(w, {})
        child = safe_float(rec.get("child_aoa"))
        model = safe_float(rec.get("model_aoa"))
        total_count = int(total_by_word.get(w, 0))
        current = crossing_from_bins(checkpoint_bins.get(w, [0 for _ in checkpoint_bounds]), checkpoint_bounds, total_count)
        row: dict[str, Any] = {
            "word": w,
            "stream_count": total_count,
            "stream_log_freq_per_million": math.log((total_count + 0.5) / max(scan["consumed_words"], 1) * 1_000_000),
            "child_status": rec.get("child_status"),
            "child_aoa_month": child,
            "model_status": rec.get("model_status"),
            "model_aoa_log10_step": model,
            "model_curve_first_minus_last": safe_float(rec.get("curve_first_minus_last")),
            "current_exposure_status": current.get("status"),
            "current_cross_log10_word_pos": current.get("cross_log10_pos"),
            "current_cross_word_pos": current.get("cross_word_pos"),
        }
        for src in sorted(source_words):
            c = int(source_by_word.get(w, Counter()).get(src, 0))
            row[f"count_src__{src}"] = c
            row[f"logfreq_src__{src}"] = math.log((c + 0.5) / max(source_words[src], 1) * 1_000_000)
        for name, order in candidate_orders.items():
            cr = crossing_for_source_order(source_by_word.get(w, Counter()), source_words, order, n_passes, total_count)
            row[f"{name}_cross_log10_word_pos"] = cr.get("cross_log10_pos")
            row[f"{name}_cross_source"] = cr.get("source_at_cross")
        word_rows.append(row)

    # Correlations relevant to route choice.
    def rows_with_child(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if safe_float(r.get("child_aoa_month")) is not None]

    def rows_with_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if safe_float(r.get("model_aoa_log10_step")) is not None]

    child_rows = rows_with_child(word_rows)
    model_rows = rows_with_model(word_rows)
    both_rows = [r for r in word_rows if safe_float(r.get("child_aoa_month")) is not None and safe_float(r.get("model_aoa_log10_step")) is not None]

    correlations: dict[str, Any] = {
        "current_exposure_vs_model_aoa": corr([r["current_cross_log10_word_pos"] for r in model_rows], [r["model_aoa_log10_step"] for r in model_rows]),
        "current_exposure_vs_child_aoa": corr([r["current_cross_log10_word_pos"] for r in child_rows], [r["child_aoa_month"] for r in child_rows]),
        "model_aoa_vs_child_aoa": corr([r["model_aoa_log10_step"] for r in both_rows], [r["child_aoa_month"] for r in both_rows]),
        "whole_stream_logfreq_vs_child_aoa": corr([r["stream_log_freq_per_million"] for r in child_rows], [r["child_aoa_month"] for r in child_rows]),
        "whole_stream_logfreq_vs_model_aoa": corr([r["stream_log_freq_per_million"] for r in model_rows], [r["model_aoa_log10_step"] for r in model_rows]),
    }
    for src in sorted(source_words):
        correlations[f"source_logfreq_vs_child_aoa__{src}"] = corr([r[f"logfreq_src__{src}"] for r in child_rows], [r["child_aoa_month"] for r in child_rows])
        correlations[f"source_logfreq_vs_model_aoa__{src}"] = corr([r[f"logfreq_src__{src}"] for r in model_rows], [r["model_aoa_log10_step"] for r in model_rows])
    for name in candidate_orders:
        vals = [r[f"{name}_cross_log10_word_pos"] for r in child_rows]
        correlations[f"candidate_cross_vs_child_aoa__{name}"] = corr(vals, [r["child_aoa_month"] for r in child_rows])
        vals2 = [r[f"{name}_cross_log10_word_pos"] for r in model_rows]
        correlations[f"candidate_cross_vs_model_aoa__{name}"] = corr(vals2, [r["model_aoa_log10_step"] for r in model_rows])

    # Rank candidate orders by predicted child alignment on the exact same word set as official model AoA fits.
    candidate_summary: list[dict[str, Any]] = []
    for name, order in candidate_orders.items():
        c_all = correlations[f"candidate_cross_vs_child_aoa__{name}"]
        c_model = correlations[f"candidate_cross_vs_model_aoa__{name}"]
        c_both_child = corr([r[f"{name}_cross_log10_word_pos"] for r in both_rows], [r["child_aoa_month"] for r in both_rows])
        candidate_summary.append({
            "candidate_order": name,
            "source_order": order,
            "child_alignment_all_child_words_pearson_r": c_all.get("pearson_r"),
            "child_alignment_all_child_words_p": c_all.get("pearson_p"),
            "child_alignment_official_model_fit_words_pearson_r": c_both_child.get("pearson_r"),
            "child_alignment_official_model_fit_words_p": c_both_child.get("pearson_p"),
            "model_alignment_pearson_r": c_model.get("pearson_r"),
            "n_child_words": c_all.get("n"),
            "n_model_words": c_model.get("n"),
        })
    candidate_summary.sort(key=lambda r: (-(r["child_alignment_official_model_fit_words_pearson_r"] if r["child_alignment_official_model_fit_words_pearson_r"] is not None else -999), r["candidate_order"]))

    source_summary: list[dict[str, Any]] = []
    for src in sorted(source_words):
        cr_child = correlations[f"source_logfreq_vs_child_aoa__{src}"]
        cr_model = correlations[f"source_logfreq_vs_model_aoa__{src}"]
        source_summary.append({
            "source": src,
            "source_words": int(source_words[src]),
            "source_rows": int(scan["source_rows"][src]),
            "source_fraction_words": float(source_words[src] / max(scan["consumed_words"], 1)),
            "logfreq_child_aoa_pearson_r": cr_child.get("pearson_r"),
            "logfreq_child_aoa_p": cr_child.get("pearson_p"),
            "logfreq_model_aoa_pearson_r": cr_model.get("pearson_r"),
            "logfreq_model_aoa_p": cr_model.get("pearson_p"),
        })
    source_summary.sort(key=lambda r: (r["logfreq_child_aoa_pearson_r"] if r["logfreq_child_aoa_pearson_r"] is not None else 999, r["source"]))

    write_csv(out_dir / "word_exposure_timing.csv", word_rows)
    write_csv(out_dir / "candidate_order_summary.csv", candidate_summary)
    write_csv(out_dir / "source_frequency_summary.csv", source_summary)
    with (out_dir / "word_exposure_timing.jsonl").open("w", encoding="utf-8") as f:
        for r in word_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    top = candidate_summary[:5]
    out = {
        "status": "AOA_STREAM_EXPOSURE_TIMING_DONE",
        "created_utc": now(),
        "inputs": {
            "stream": rel(stream_path),
            "order_manifest": rel(args.order_manifest),
            "scientific_metrics": rel(args.scientific_metrics),
            "word_records": rel(args.word_records),
            "cdi_human": rel(args.cdi_human),
            "chang_bergen_source": rel(DEFAULT_CHANG),
        },
        "scan": {
            "rows": scan["rows"],
            "consumed_words": scan["consumed_words"],
            "elapsed_sec": round(scan["elapsed_sec"], 3),
            "n_passes_inferred": n_passes,
            "source_word_exposure": {k: int(v) for k, v in sorted(source_words.items())},
            "source_rows": {k: int(v) for k, v in sorted(scan["source_rows"].items())},
        },
        "word_sets": {
            "target_words": len(target_words),
            "child_aoa_words": len(child_rows),
            "model_aoa_words": len(model_rows),
            "both_model_child_words": len(both_rows),
        },
        "core_correlations": correlations,
        "candidate_order_summary_top": top,
        "source_frequency_summary": source_summary,
        "outputs": {
            "summary_json": rel(out_dir / "summary.json"),
            "summary_md": rel(out_dir / "summary.md"),
            "word_csv": rel(out_dir / "word_exposure_timing.csv"),
            "candidate_csv": rel(out_dir / "candidate_order_summary.csv"),
            "source_csv": rel(out_dir / "source_frequency_summary.csv"),
        },
        "interpretation_scope": "Exact lower-cased word-boundary stream counts approximate lexical exposure timing. This is a route-selection/prediction instrument for legal curriculum design, not an official AoA score and not evidence that endpoint competence changes.",
    }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x: Any) -> str:
        return "NA" if x is None else f"{float(x):.4f}"

    lines = [
        "# research AoA stream exposure timing",
        "",
        out["interpretation_scope"],
        "",
        "## Scan",
        "",
        f"- Stream rows: {scan['rows']}; consumed words from row metadata: {scan['consumed_words']}; inferred passes over a 10M pool: {n_passes}.",
        f"- Child-AoA words in records: {len(child_rows)}; model-AoA fitted words: {len(model_rows)}; both: {len(both_rows)}.",
        "",
        "## Core correlations",
        "",
        f"- Current exposure crossing vs measured model AoA: r={fmt(correlations['current_exposure_vs_model_aoa'].get('pearson_r'))}, p={fmt(correlations['current_exposure_vs_model_aoa'].get('pearson_p'))}, n={correlations['current_exposure_vs_model_aoa'].get('n')}.",
        f"- Current exposure crossing vs child AoA: r={fmt(correlations['current_exposure_vs_child_aoa'].get('pearson_r'))}, p={fmt(correlations['current_exposure_vs_child_aoa'].get('pearson_p'))}, n={correlations['current_exposure_vs_child_aoa'].get('n')}.",
        f"- Measured model AoA vs child AoA (recomputed on joined words): r={fmt(correlations['model_aoa_vs_child_aoa'].get('pearson_r'))}, p={fmt(correlations['model_aoa_vs_child_aoa'].get('pearson_p'))}, n={correlations['model_aoa_vs_child_aoa'].get('n')}.",
        f"- Whole-stream log frequency vs child AoA: r={fmt(correlations['whole_stream_logfreq_vs_child_aoa'].get('pearson_r'))}, p={fmt(correlations['whole_stream_logfreq_vs_child_aoa'].get('pearson_p'))}.",
        f"- Whole-stream log frequency vs measured model AoA: r={fmt(correlations['whole_stream_logfreq_vs_model_aoa'].get('pearson_r'))}, p={fmt(correlations['whole_stream_logfreq_vs_model_aoa'].get('pearson_p'))}.",
        "",
        "## Source log-frequency relation to child AoA",
        "",
        "| source | words | frac | r(logfreq, child AoA) | p |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in source_summary:
        lines.append(f"| {r['source']} | {r['source_words']} | {r['source_fraction_words']:.3f} | {fmt(r['logfreq_child_aoa_pearson_r'])} | {fmt(r['logfreq_child_aoa_p'])} |")
    lines += ["", "## Candidate source-order exposure crossings", "", "| order | r(cross, child AoA) on official fitted words | p | r(cross, model AoA) |", "|---|---:|---:|---:|"]
    for r in candidate_summary:
        lines.append(f"| {r['candidate_order']} | {fmt(r['child_alignment_official_model_fit_words_pearson_r'])} | {fmt(r['child_alignment_official_model_fit_words_p'])} | {fmt(r['model_alignment_pearson_r'])} |")
    lines += ["", f"Full JSON: `{rel(out_dir / 'summary.json')}`"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": out["status"],
        "summary_json": rel(out_dir / "summary.json"),
        "summary_md": rel(out_dir / "summary.md"),
        "current_exposure_vs_model_aoa": correlations["current_exposure_vs_model_aoa"],
        "current_exposure_vs_child_aoa": correlations["current_exposure_vs_child_aoa"],
        "model_aoa_vs_child_aoa": correlations["model_aoa_vs_child_aoa"],
        "best_candidate_orders": top,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
