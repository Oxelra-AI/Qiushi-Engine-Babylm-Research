#!/usr/bin/env python3
"""research: row-level offline predictions for AoA-oriented legal reordering.

The previous source-block analysis showed that moving entire sources is too coarse.
This script scores row-level reorderings of the same 100M-word stream.  The
primary legal proxy is CHILDES-derived CDI-word frequency computed from the
training stream itself, not the official child AoA labels.  Child AoA labels are
used only after the fact to evaluate predicted order alignment.  An oracle row
ordering using child AoA is included only as an upper-bound diagnostic and must
not be used as a training recipe.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
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
STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
WORD_RECORDS = _public_path('experiments/archive/relation_learning/data/aoa_curve_fit_analysis/word_curve_fit_records.csv')
CDI = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')
OUT = _public_path('experiments/archive/relation_learning/data/aoa_row_curriculum_prediction')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def corr(xs: list[Any], ys: list[Any]) -> dict[str, Any]:
    vals = []
    for x, y in zip(xs, ys):
        fx = safe_float(x)
        fy = safe_float(y)
        if fx is not None and fy is not None:
            vals.append((fx, fy))
    if len(vals) < 3 or len(set(x for x, _ in vals)) < 2 or len(set(y for _, y in vals)) < 2:
        return {"n": len(vals), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    xs2 = [x for x, _ in vals]
    ys2 = [y for _, y in vals]
    pr = pearsonr(xs2, ys2)
    sr = spearmanr(xs2, ys2)
    return {"n": len(vals), "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue), "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue)}


def load_cdi_words() -> list[str]:
    df = pd.read_csv(CDI)
    return [str(w).lower() for w in df["word"].tolist() if isinstance(w, str) and w]


def load_word_records() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with WORD_RECORDS.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            w = str(row.get("word", "")).lower()
            if w:
                out[w] = dict(row)
    return out


def scan_rows(stream: pathlib.Path, target_words: set[str], max_rows: int = 0) -> dict[str, Any]:
    word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
    rows: list[dict[str, Any]] = []
    total_by_word: Counter[str] = Counter()
    source_by_word: dict[str, Counter[str]] = {w: Counter() for w in target_words}
    source_words: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()
    consumed = 0
    t0 = time.time()
    with stream.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            src = str(obj.get("source", "unknown"))
            words_meta = int(obj.get("words", len(text.split())))
            pass_index = int(consumed // 10_000_000)
            toks = [m.group(0).lower() for m in word_re.finditer(text)]
            counts = Counter(t for t in toks if t in target_words)
            rows.append({"row_index": len(rows), "source": src, "words": words_meta, "pass_index": pass_index, "counts": dict(counts), "n_target_tokens": int(sum(counts.values()))})
            consumed += words_meta
            source_words[src] += words_meta
            source_rows[src] += 1
            for w, c in counts.items():
                total_by_word[w] += int(c)
                source_by_word[w][src] += int(c)
            if max_rows and len(rows) >= max_rows:
                break
            if len(rows) % 100000 == 0:
                print(json.dumps({"event": "scan_progress", "rows": len(rows), "consumed_words": consumed, "elapsed_sec": round(time.time() - t0, 1)}), flush=True)
    return {"rows": rows, "total_by_word": total_by_word, "source_by_word": source_by_word, "source_words": source_words, "source_rows": source_rows, "consumed_words": consumed, "elapsed_sec": time.time() - t0}


def build_word_features(word_records: dict[str, dict[str, Any]], total_by_word: Counter[str], source_by_word: dict[str, Counter[str]], source_words: Counter[str], target_words: list[str]) -> dict[str, dict[str, float | None]]:
    total_words = sum(source_words.values())
    feats: dict[str, dict[str, float | None]] = {}
    for w in target_words:
        rec = word_records.get(w, {})
        child = safe_float(rec.get("child_aoa"))
        model = safe_float(rec.get("model_aoa"))
        whole = math.log((total_by_word.get(w, 0) + 0.5) / max(total_words, 1) * 1_000_000)
        childes = math.log((source_by_word.get(w, Counter()).get("childes", 0) + 0.5) / max(source_words.get("childes", 1), 1) * 1_000_000)
        bnc = math.log((source_by_word.get(w, Counter()).get("bnc_spoken", 0) + 0.5) / max(source_words.get("bnc_spoken", 1), 1) * 1_000_000)
        open_sub = math.log((source_by_word.get(w, Counter()).get("open_subtitles", 0) + 0.5) / max(source_words.get("open_subtitles", 1), 1) * 1_000_000)
        # Positive proxy means predicted early child acquisition. It is legal if derived from stream counts only.
        feats[w] = {
            "child_aoa": child,
            "model_aoa": model,
            "whole_logfreq": whole,
            "childes_logfreq": childes,
            "bnc_logfreq": bnc,
            "opensub_logfreq": open_sub,
            "childes_minus_whole": childes - whole,
            "spoken_mix_logfreq": math.log(((source_by_word.get(w, Counter()).get("childes", 0) + source_by_word.get(w, Counter()).get("bnc_spoken", 0) + source_by_word.get(w, Counter()).get("switchboard", 0) + 0.5) / max(source_words.get("childes", 0) + source_words.get("bnc_spoken", 0) + source_words.get("switchboard", 0), 1)) * 1_000_000),
            "oracle_child_early_score": None if child is None else -child,
        }
    return feats


def score_rows(rows: list[dict[str, Any]], feats: dict[str, dict[str, float | None]]) -> None:
    for r in rows:
        counts = r["counts"]
        n = sum(counts.values())
        if n <= 0:
            for k in ["childes_freq", "childes_ratio", "spoken_freq", "whole_freq", "oracle_child", "cdi_density"]:
                r[f"score_{k}"] = -1e9 if k != "cdi_density" else 0.0
            continue
        def wavg(name: str) -> float | None:
            vals = []
            weights = []
            for w, c in counts.items():
                v = feats.get(w, {}).get(name)
                if v is not None and math.isfinite(float(v)):
                    vals.append(float(v))
                    weights.append(int(c))
            if not vals:
                return None
            return float(np.average(vals, weights=weights))
        childes = wavg("childes_logfreq")
        ratio = wavg("childes_minus_whole")
        spoken = wavg("spoken_mix_logfreq")
        whole = wavg("whole_logfreq")
        oracle = wavg("oracle_child_early_score")
        density = n / max(int(r["words"]), 1)
        r["score_childes_freq"] = childes if childes is not None else -1e9
        r["score_childes_ratio"] = ratio if ratio is not None else -1e9
        r["score_spoken_freq"] = spoken if spoken is not None else -1e9
        r["score_whole_freq"] = whole if whole is not None else -1e9
        r["score_oracle_child"] = oracle if oracle is not None else -1e9
        r["score_cdi_density"] = density
        # Composite legal proxy: high child-directed frequency, childes-over-whole ratio, and nonzero density.
        r["score_childes_composite"] = (r["score_childes_freq"] if r["score_childes_freq"] > -1e8 else -50.0) + 0.75 * (r["score_childes_ratio"] if r["score_childes_ratio"] > -1e8 else -50.0) + 1.5 * min(0.05, density)


def ordering_indices(rows: list[dict[str, Any]], mode: str) -> list[int]:
    idxs = list(range(len(rows)))
    if mode == "current":
        return idxs
    source_rank_childes_first = {s: i for i, s in enumerate(["childes", "bnc_spoken", "switchboard", "open_subtitles", "simple_wiki", "qwen_pair_packed", "cleanqwen_fineweb_compact_view_reinvest", "gutenberg", "neutral_cleanqwen_topup_compact_reinvest::open_subtitles"])}
    if mode == "source_childes_first_stable":
        return sorted(idxs, key=lambda i: (source_rank_childes_first.get(rows[i]["source"], 99), rows[i]["row_index"]))
    m = re.match(r"(global|within_pass)__(.+)", mode)
    if not m:
        raise ValueError(mode)
    scope, score_name = m.group(1), m.group(2)
    sk = f"score_{score_name}"
    if scope == "global":
        return sorted(idxs, key=lambda i: (-float(rows[i].get(sk, -1e9)), rows[i]["row_index"]))
    by_pass: dict[int, list[int]] = defaultdict(list)
    for i in idxs:
        by_pass[int(rows[i]["pass_index"])].append(i)
    out: list[int] = []
    for p in sorted(by_pass):
        out.extend(sorted(by_pass[p], key=lambda i: (-float(rows[i].get(sk, -1e9)), rows[i]["row_index"])))
    return out


def crossings_for_order(rows: list[dict[str, Any]], order: list[int], total_by_word: Counter[str], target_words: list[str], max_words: int) -> dict[str, dict[str, Any]]:
    half = {w: total_by_word.get(w, 0) / 2.0 for w in target_words}
    cum: Counter[str] = Counter()
    crossed: dict[str, dict[str, Any]] = {}
    pos = 0
    remaining = set(w for w in target_words if total_by_word.get(w, 0) > 0)
    for oi in order:
        r = rows[oi]
        counts = r["counts"]
        prev_pos = pos
        pos += int(r["words"])
        if not counts or not remaining:
            continue
        for w, c in counts.items():
            if w not in remaining:
                cum[w] += int(c)
                continue
            prev = cum[w]
            new = prev + int(c)
            if new >= half[w] and half[w] > 0:
                frac = (half[w] - prev) / max(int(c), 1)
                cross = prev_pos + max(0.0, min(1.0, frac)) * int(r["words"])
                crossed[w] = {"status": "ok", "cross_pos": float(cross), "cross_log10": float(math.log10(max(cross, 1.0))), "source": r["source"], "pass_index": int(r["pass_index"]), "row_index": int(r["row_index"])}
                remaining.remove(w)
            cum[w] = new
    for w in target_words:
        if total_by_word.get(w, 0) <= 0:
            crossed[w] = {"status": "absent", "cross_pos": None, "cross_log10": None}
        elif w not in crossed:
            crossed[w] = {"status": "not_crossed", "cross_pos": None, "cross_log10": None, "final_cum": int(cum[w])}
    return crossed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stream", default=str(STREAM))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cdi_words = load_cdi_words()
    records = load_word_records()
    target_words = sorted(set(cdi_words) | set(records.keys()))
    print(json.dumps({"event": "start", "stream": rel(args.stream), "target_words": len(target_words), "utc": now()}), flush=True)
    scan = scan_rows(pathlib.Path(args.stream), set(target_words), max_rows=args.max_rows)
    feats = build_word_features(records, scan["total_by_word"], scan["source_by_word"], scan["source_words"], target_words)
    score_rows(scan["rows"], feats)
    print(json.dumps({"event": "scored_rows", "rows": len(scan["rows"]), "elapsed_scan_sec": round(scan["elapsed_sec"], 1)}, flush=True))

    modes = [
        "current",
        "source_childes_first_stable",
        "within_pass__childes_freq",
        "within_pass__childes_ratio",
        "within_pass__spoken_freq",
        "within_pass__childes_composite",
        "within_pass__whole_freq",
        "within_pass__cdi_density",
        "within_pass__oracle_child",
        "global__childes_freq",
        "global__childes_ratio",
        "global__spoken_freq",
        "global__childes_composite",
        "global__whole_freq",
        "global__cdi_density",
        "global__oracle_child",
    ]
    all_cross: dict[str, dict[str, dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    word_rows: list[dict[str, Any]] = []
    child_words = [w for w in target_words if feats[w].get("child_aoa") is not None]
    model_words = [w for w in target_words if feats[w].get("model_aoa") is not None]
    both_words = [w for w in target_words if feats[w].get("child_aoa") is not None and feats[w].get("model_aoa") is not None]

    for mode in modes:
        t0 = time.time()
        order = ordering_indices(scan["rows"], mode)
        cross = crossings_for_order(scan["rows"], order, scan["total_by_word"], target_words, scan["consumed_words"])
        all_cross[mode] = cross
        child_corr = corr([cross[w].get("cross_log10") for w in child_words], [feats[w]["child_aoa"] for w in child_words])
        child_corr_both = corr([cross[w].get("cross_log10") for w in both_words], [feats[w]["child_aoa"] for w in both_words])
        model_corr = corr([cross[w].get("cross_log10") for w in model_words], [feats[w]["model_aoa"] for w in model_words])
        summary_rows.append({
            "mode": mode,
            "r_cross_child_all": child_corr.get("pearson_r"),
            "p_cross_child_all": child_corr.get("pearson_p"),
            "rho_cross_child_all": child_corr.get("spearman_r"),
            "r_cross_child_official_fit_words": child_corr_both.get("pearson_r"),
            "p_cross_child_official_fit_words": child_corr_both.get("pearson_p"),
            "rho_cross_child_official_fit_words": child_corr_both.get("spearman_r"),
            "r_cross_model": model_corr.get("pearson_r"),
            "p_cross_model": model_corr.get("pearson_p"),
            "rho_cross_model": model_corr.get("spearman_r"),
            "n_child_all": child_corr.get("n"),
            "n_both": child_corr_both.get("n"),
            "n_model": model_corr.get("n"),
            "elapsed_sec": round(time.time() - t0, 3),
        })
        print(json.dumps({"event": "mode_done", "mode": mode, "r_child_fit_words": summary_rows[-1]["r_cross_child_official_fit_words"], "r_model": summary_rows[-1]["r_cross_model"], "elapsed_sec": summary_rows[-1]["elapsed_sec"]}), flush=True)

    for w in target_words:
        row = {
            "word": w,
            "stream_count": int(scan["total_by_word"].get(w, 0)),
            "child_aoa": feats[w].get("child_aoa"),
            "model_aoa": feats[w].get("model_aoa"),
            "whole_logfreq": feats[w].get("whole_logfreq"),
            "childes_logfreq": feats[w].get("childes_logfreq"),
            "childes_minus_whole": feats[w].get("childes_minus_whole"),
        }
        for mode in modes:
            row[f"{mode}_cross_log10"] = all_cross[mode][w].get("cross_log10")
            row[f"{mode}_cross_source"] = all_cross[mode][w].get("source")
        word_rows.append(row)

    # Feature correlations: which legal word-level proxy has child/model signal?
    feature_rows: list[dict[str, Any]] = []
    for feat in ["whole_logfreq", "childes_logfreq", "bnc_logfreq", "opensub_logfreq", "spoken_mix_logfreq", "childes_minus_whole"]:
        feature_rows.append({
            "feature": feat,
            "r_feature_child_all": corr([feats[w].get(feat) for w in child_words], [feats[w]["child_aoa"] for w in child_words]).get("pearson_r"),
            "p_feature_child_all": corr([feats[w].get(feat) for w in child_words], [feats[w]["child_aoa"] for w in child_words]).get("pearson_p"),
            "r_feature_model": corr([feats[w].get(feat) for w in model_words], [feats[w]["model_aoa"] for w in model_words]).get("pearson_r"),
            "p_feature_model": corr([feats[w].get(feat) for w in model_words], [feats[w]["model_aoa"] for w in model_words]).get("pearson_p"),
        })

    summary_rows_sorted = sorted(summary_rows, key=lambda r: (-(r["r_cross_child_official_fit_words"] if r["r_cross_child_official_fit_words"] is not None else -999), r["mode"]))
    write_csv(out_dir / "candidate_row_order_summary.csv", summary_rows_sorted)
    write_csv(out_dir / "word_candidate_crossings.csv", word_rows)
    write_csv(out_dir / "word_feature_correlations.csv", feature_rows)

    out = {
        "status": "AOA_ROW_CURRICULUM_PREDICTION_DONE",
        "created_utc": now(),
        "inputs": {"stream": rel(args.stream), "word_records": rel(WORD_RECORDS), "cdi": rel(CDI)},
        "scan": {"rows": len(scan["rows"]), "consumed_words": scan["consumed_words"], "elapsed_sec": round(scan["elapsed_sec"], 3), "source_words": {k: int(v) for k, v in sorted(scan["source_words"].items())}},
        "word_sets": {"target_words": len(target_words), "child_words": len(child_words), "model_words": len(model_words), "both_words": len(both_words)},
        "feature_correlations": feature_rows,
        "candidate_order_summary": summary_rows_sorted,
        "interpretation_scope": "Legal proxy orderings use stream-derived frequencies; oracle_child uses official child AoA labels only as an upper-bound diagnostic and is not a training recipe.",
        "outputs": {"summary_json": rel(out_dir / "summary.json"), "summary_md": rel(out_dir / "summary.md"), "candidate_csv": rel(out_dir / "candidate_row_order_summary.csv"), "word_csv": rel(out_dir / "word_candidate_crossings.csv"), "feature_csv": rel(out_dir / "word_feature_correlations.csv")},
    }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x: Any) -> str:
        return "NA" if x is None else f"{float(x):.4f}"

    lines = [
        "# research AoA row-level curriculum prediction",
        "",
        out["interpretation_scope"],
        "",
        f"Scanned {len(scan['rows'])} rows / {scan['consumed_words']} words.",
        "",
        "## Word-level legal proxy signals",
        "",
        "| feature | r(feature, child AoA) | p | r(feature, measured model AoA) | p |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in feature_rows:
        lines.append(f"| {r['feature']} | {fmt(r['r_feature_child_all'])} | {fmt(r['p_feature_child_all'])} | {fmt(r['r_feature_model'])} | {fmt(r['p_feature_model'])} |")
    lines += ["", "## Candidate row orderings", "", "| mode | r(cross, child AoA) official fit words | rho | p | r(cross, model AoA) |", "|---|---:|---:|---:|---:|"]
    for r in summary_rows_sorted:
        lines.append(f"| {r['mode']} | {fmt(r['r_cross_child_official_fit_words'])} | {fmt(r['rho_cross_child_official_fit_words'])} | {fmt(r['p_cross_child_official_fit_words'])} | {fmt(r['r_cross_model'])} |")
    lines += ["", f"Full JSON: `{rel(out_dir / 'summary.json')}`"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "summary_json": rel(out_dir / "summary.json"), "summary_md": rel(out_dir / "summary.md"), "top_modes": summary_rows_sorted[:6], "feature_correlations": feature_rows}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
