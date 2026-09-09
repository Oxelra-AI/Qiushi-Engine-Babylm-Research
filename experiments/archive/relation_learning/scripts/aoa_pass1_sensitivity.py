#!/usr/bin/env python3
"""research: sensitivity of pass-1 AoA predictions to exposure-response slope.

Uses actual coherent86 mean-surprisal curves as the baseline and applies
candidate pass-1 cumulative-exposure changes only at the official 1M..10M
checkpoints:

    S_candidate(w,t) = S_actual(w,t) + beta * [log1p(E_candidate(w,t)) - log1p(E_actual(w,t))]

Checkpoints after 10M are unchanged.  The resulting curves are scored by the
same official AoA sigmoid/Pearson/p-value clipping code.  This is a stress test
of whether the negative research fixed-effect prediction depends on an overly
weak exposure-response estimate.
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
import os
import pathlib
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = _public_path('.')
STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
SURPRISAL = _public_path('experiments/archive/representation_and_objectives/data/alpha075_aoa_minctx0/collate_fast/results/hf_model/main/zero_shot/mlm/AoA_word/surprisal.json')
WORDS = _public_path('experiments/archive/relation_learning/data/aoa_curve_fit_analysis/word_curve_fit_records.csv')
CDI = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')
TOKENIZER_PATH = _public_path('experiments/archive/representation_and_objectives/training/runs/alpha075_exact_replay_from82M_seed43022/hf_model_alpha0p75/chck_100M')
UTILS_PARENT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
OUT = _public_path('experiments/archive/relation_learning/data/aoa_pass1_sensitivity')
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
PASS_WORDS = 10_000_000


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
        w.writeheader(); w.writerows(rows)


def safe_float(x: Any) -> float | None:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def corr(xs: list[Any], ys: list[Any]) -> dict[str, Any]:
    vals = []
    for x, y in zip(xs, ys):
        fx, fy = safe_float(x), safe_float(y)
        if fx is not None and fy is not None:
            vals.append((fx, fy))
    if len(vals) < 3 or len({x for x, _ in vals}) < 2 or len({y for _, y in vals}) < 2:
        return {"n": len(vals), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr = pearsonr([x for x, _ in vals], [y for _, y in vals])
    sr = spearmanr([x for x, _ in vals], [y for _, y in vals])
    return {"n": len(vals), "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue), "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue)}


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        v = float(x)
        return f"{v:.4f}" if math.isfinite(v) else "NA"
    except Exception:
        return str(x)


def load_surprisal_means() -> tuple[list[str], list[int], dict[tuple[str, int], float]]:
    obj = json.loads(SURPRISAL.read_text(encoding="utf-8"))
    acc: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in obj.get("results", []):
        w = str(r.get("target_word", "")).lower()
        wc = int(r.get("word_count"))
        s = safe_float(r.get("surprisal"))
        if w and s is not None:
            acc[(w, wc)].append(float(s))
    words = sorted({w for w, _ in acc})
    checkpoints = sorted({wc for _, wc in acc})
    means = {k: float(np.mean(v)) for k, v in acc.items()}
    return words, checkpoints, means


def load_step106() -> dict[str, dict[str, Any]]:
    out = {}
    with WORDS.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = str(row.get("word", "")).lower()
            if w:
                out[w] = row
    return out


def load_cdi_words() -> list[str]:
    df = pd.read_csv(CDI)
    return [str(w).lower() for w in df["word"].tolist() if isinstance(w, str) and w]


def scan_stream(words: list[str], checkpoints: list[int]) -> dict[str, Any]:
    targets = set(words)
    cp = sorted(checkpoints)
    cp_i = 0
    cum: Counter[str] = Counter()
    exposures: dict[int, Counter[str]] = {c: Counter() for c in cp}
    whole_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_words: Counter[str] = Counter()
    pass1_rows: list[dict[str, Any]] = []
    consumed = 0
    n_rows = 0
    t0 = time.time()
    with STREAM.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            src = str(obj.get("source", "unknown"))
            row_words = int(obj.get("words", len(text.split())))
            counts: Counter[str] = Counter()
            for m in WORD_RE.finditer(text):
                w = m.group(0).lower()
                if w in targets:
                    counts[w] += 1
            before = consumed
            after = consumed + row_words
            while cp_i < len(cp) and cp[cp_i] <= after:
                boundary = cp[cp_i]
                frac = 0.0 if row_words <= 0 else max(0.0, min(1.0, (boundary - before) / row_words))
                ccopy = Counter(cum)
                for w, c in counts.items():
                    ccopy[w] += frac * int(c)
                exposures[boundary] = ccopy
                cp_i += 1
            if before < PASS_WORDS:
                pass1_rows.append({"row_index": len(pass1_rows), "source": src, "words": row_words, "counts": dict(counts)})
            consumed = after
            n_rows += 1
            source_words[src] += row_words
            for w, c in counts.items():
                cum[w] += int(c)
                whole_counts[w] += int(c)
                source_counts[w][src] += int(c)
            if n_rows % 100000 == 0:
                print(json.dumps({"event": "scan_progress", "rows": n_rows, "words": consumed, "elapsed_sec": round(time.time()-t0,1)}), flush=True)
    return {"rows": n_rows, "consumed_words": consumed, "exposures": exposures, "whole_counts": whole_counts, "source_counts": source_counts, "source_words": source_words, "pass1_rows": pass1_rows, "elapsed_sec": time.time()-t0}


def features(words: list[str], research: dict[str, dict[str, Any]], scan: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    total_words = max(sum(scan["source_words"].values()), 1)
    spoken_den = max(scan["source_words"].get("childes", 0) + scan["source_words"].get("bnc_spoken", 0) + scan["source_words"].get("switchboard", 0), 1)
    out = {}
    for w in words:
        rec = research.get(w, {})
        whole = math.log((scan["whole_counts"].get(w, 0) + 0.5) / total_words * 1_000_000)
        childes = math.log((scan["source_counts"].get(w, Counter()).get("childes", 0) + 0.5) / max(scan["source_words"].get("childes", 1), 1) * 1_000_000)
        spoken = math.log((scan["source_counts"].get(w, Counter()).get("childes", 0) + scan["source_counts"].get(w, Counter()).get("bnc_spoken", 0) + scan["source_counts"].get(w, Counter()).get("switchboard", 0) + 0.5) / spoken_den * 1_000_000)
        child = safe_float(rec.get("child_aoa"))
        out[w] = {"whole_logfreq": whole, "childes_logfreq": childes, "spoken_mix_logfreq": spoken, "childes_minus_whole": childes - whole, "child_aoa": child, "model_aoa": safe_float(rec.get("model_aoa")), "oracle_child_early_score": None if child is None else -child}
    return out


def score_rows(pass1_rows: list[dict[str, Any]], feats: dict[str, dict[str, Any]]) -> None:
    for r in pass1_rows:
        counts = r["counts"]
        n = sum(counts.values())
        density = n / max(int(r["words"]), 1)
        if n <= 0:
            for s in ["childes_freq", "childes_ratio", "spoken_freq", "whole_freq", "cdi_density", "child_early"]:
                r[f"score_{s}"] = 0.0 if s == "cdi_density" else -1e9
            continue
        def wavg(name: str) -> float | None:
            vals = []; weights = []
            for w, c in counts.items():
                v = feats.get(w, {}).get(name)
                if v is not None and math.isfinite(float(v)):
                    vals.append(float(v)); weights.append(int(c))
            return float(np.average(vals, weights=weights)) if vals else None
        mapping = {"childes_freq": "childes_logfreq", "childes_ratio": "childes_minus_whole", "spoken_freq": "spoken_mix_logfreq", "whole_freq": "whole_logfreq", "child_early": "oracle_child_early_score"}
        for sk, fk in mapping.items():
            v = wavg(fk); r[f"score_{sk}"] = v if v is not None else -1e9
        r["score_cdi_density"] = density
        r["score_childes_composite"] = (r["score_childes_freq"] if r["score_childes_freq"] > -1e8 else -50) + 0.75 * (r["score_childes_ratio"] if r["score_childes_ratio"] > -1e8 else -50) + 1.5 * min(0.05, density)


def order_rows(rows: list[dict[str, Any]], mode: str) -> list[int]:
    idxs = list(range(len(rows)))
    if mode == "current":
        return idxs
    if mode == "source_childes_first":
        rank = {s: i for i, s in enumerate(["childes", "bnc_spoken", "switchboard", "open_subtitles", "simple_wiki", "qwen_pair_packed", "cleanqwen_fineweb_compact_view_reinvest", "gutenberg", "neutral_cleanqwen_topup_compact_reinvest::open_subtitles"])}
        return sorted(idxs, key=lambda i: (rank.get(rows[i]["source"], 99), rows[i]["row_index"]))
    if mode == "source_childes_only_then_current":
        return sorted(idxs, key=lambda i: (0 if rows[i]["source"] == "childes" else 1, rows[i]["row_index"]))
    if mode == "source_written_first_control":
        rank = {s: i for i, s in enumerate(["gutenberg", "qwen_pair_packed", "cleanqwen_fineweb_compact_view_reinvest", "simple_wiki", "open_subtitles", "bnc_spoken", "switchboard", "childes", "neutral_cleanqwen_topup_compact_reinvest::open_subtitles"])}
        return sorted(idxs, key=lambda i: (rank.get(rows[i]["source"], 99), rows[i]["row_index"]))
    if not mode.startswith("sort__"):
        raise ValueError(mode)
    sk = "score_" + mode.split("__", 1)[1]
    return sorted(idxs, key=lambda i: (-float(rows[i].get(sk, -1e9)), rows[i]["row_index"]))


def candidate_exposures(rows: list[dict[str, Any]], order: list[int], checkpoints: list[int], actual: dict[int, Counter[str]]) -> dict[int, Counter[str]]:
    out: dict[int, Counter[str]] = {}
    early = [c for c in checkpoints if c <= PASS_WORDS]
    cp_i = 0
    pos = 0
    cum: Counter[str] = Counter()
    for oi in order:
        r = rows[oi]
        rw = int(r["words"]); before = pos; after = pos + rw
        counts = r["counts"]
        while cp_i < len(early) and early[cp_i] <= after:
            boundary = early[cp_i]
            frac = 0.0 if rw <= 0 else max(0.0, min(1.0, (boundary - before)/rw))
            ccopy = Counter(cum)
            for w, c in counts.items():
                ccopy[w] += frac * int(c)
            out[boundary] = ccopy; cp_i += 1
        pos = after
        for w, c in counts.items():
            cum[w] += int(c)
    for c in early[cp_i:]:
        out[c] = Counter(cum)
    for c in checkpoints:
        if c > PASS_WORDS:
            out[c] = actual[c]
    return out


def load_eval(out_dir: pathlib.Path):
    cache = out_dir / "hf_cache"; cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache / "hf_home"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "transformers_cache"))
    os.environ.setdefault("HF_MODULES_CACHE", str(cache / "modules"))
    sys.path.insert(0, str(UTILS_PARENT))
    from transformers import AutoTokenizer  # noqa: WPS433
    from evaluation_pipeline.utils import AoAEvaluator  # noqa: WPS433
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), trust_remote_code=True, local_files_only=True)
    ev = AoAEvaluator(CDI)
    return tok, ev


def official_score(curves: dict[tuple[str, int], float], words: list[str], checkpoints: list[int], tok: Any, ev: Any) -> dict[str, Any]:
    results = []
    for w in words:
        for ck in checkpoints:
            results.append({"step": f"chck_{ck}", "word_count": int(ck), "target_word": w, "context_id": 0, "context": "", "surprisal": float(curves[(w, ck)])})
    return ev.compute_curve_fitness({"results": results}, tokenizer=tok)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--betas", default="-0.10,-0.20,-0.36,-0.50,-0.75,-1.00")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    betas = [float(x) for x in args.betas.split(",") if x.strip()]
    words_obs, checkpoints, means = load_surprisal_means()
    research = load_step106()
    target_words = sorted(set(words_obs) | set(load_cdi_words()))
    print(json.dumps({"event": "start", "words_obs": len(words_obs), "target_words": len(target_words), "betas": betas, "utc": now()}), flush=True)
    scan = scan_stream(target_words, checkpoints)
    feats = features(target_words, research, scan)
    score_rows(scan["pass1_rows"], feats)
    tok, ev = load_eval(out_dir)
    modes = ["current", "source_childes_only_then_current", "source_childes_first", "sort__childes_freq", "sort__childes_ratio", "sort__spoken_freq", "sort__childes_composite", "sort__cdi_density", "sort__whole_freq", "source_written_first_control", "sort__child_early"]
    child_by_word = {w: safe_float(research.get(w, {}).get("child_aoa")) for w in words_obs}
    rows = []
    # Current score uses actual curves and should reproduce the official mean-curve score closely.
    cur_score = official_score(means, words_obs, checkpoints, tok, ev)
    rows.append({"mode": "current_actual", "beta": 0.0, "oracle_only": False, "curve_fitness_clipped": cur_score.get("curve_fitness"), "p_value": cur_score.get("p_value"), "n_words": cur_score.get("n_words"), "mean_monthly_score": cur_score.get("mean_monthly_score")})
    for mode in modes:
        if mode == "current":
            continue
        order = order_rows(scan["pass1_rows"], mode)
        exp = candidate_exposures(scan["pass1_rows"], order, checkpoints, scan["exposures"])
        # Precompute log exposure deltas for early checkpoints.
        dlog = {(w, ck): math.log1p(float(exp[ck].get(w, 0.0))) - math.log1p(float(scan["exposures"][ck].get(w, 0.0))) for w in words_obs for ck in checkpoints}
        for beta in betas:
            curves = {}
            for w in words_obs:
                for ck in checkpoints:
                    curves[(w, ck)] = float(means[(w, ck)] + beta * dlog[(w, ck)])
            score = official_score(curves, words_obs, checkpoints, tok, ev)
            # Recover unclipped correlation by reading p only when significant is impossible from score; compute from returned only if no p absent? Simpler: clipped score is decisive for this instrument.
            rows.append({
                "mode": mode,
                "beta": beta,
                "oracle_only": mode == "sort__child_early",
                "curve_fitness_clipped": score.get("curve_fitness"),
                "p_value": score.get("p_value"),
                "n_words": score.get("n_words"),
                "mean_monthly_score": score.get("mean_monthly_score"),
                "mean_abs_delta_surprisal_1M": float(np.mean([abs(beta * dlog[(w, 1_000_000)]) for w in words_obs])),
                "mean_delta_surprisal_1M": float(np.mean([beta * dlog[(w, 1_000_000)] for w in words_obs])),
                "max_abs_delta_after_10M": float(max(abs(beta * dlog[(w, ck)]) for w in words_obs for ck in checkpoints if ck > PASS_WORDS)),
            })
        print(json.dumps({"event": "mode_done", "mode": mode}), flush=True)
    rows_sorted = sorted(rows, key=lambda r: (bool(r.get("oracle_only")), -(float(r.get("curve_fitness_clipped") or 0.0)), str(r.get("mode")), float(r.get("beta", 0.0))))
    write_csv(out_dir / "sensitivity_scores.csv", rows_sorted)
    best_legal = [r for r in rows_sorted if not r.get("oracle_only") and r["mode"] != "current_actual"][:10]
    best_oracle = [r for r in rows_sorted if r.get("oracle_only")][:10]
    out = {
        "status": "AOA_PASS1_SENSITIVITY_DONE",
        "created_utc": now(),
        "scope": "Actual coherent86 mean-surprisal baseline plus beta times candidate-vs-current log cumulative exposure delta at official checkpoints; after 10M deltas are zero.",
        "inputs": {"stream": rel(STREAM), "surprisal": rel(SURPRISAL), "words": rel(WORDS), "tokenizer": rel(TOKENIZER_PATH)},
        "scan": {"rows": scan["rows"], "consumed_words": scan["consumed_words"], "pass1_rows": len(scan["pass1_rows"]), "elapsed_sec": round(scan["elapsed_sec"], 3)},
        "betas": betas,
        "current_actual_score_from_mean_curves": rows[0],
        "best_legal": best_legal,
        "best_oracle": best_oracle,
        "outputs": {"summary_json": rel(out_dir / "summary.json"), "summary_md": rel(out_dir / "summary.md"), "scores_csv": rel(out_dir / "sensitivity_scores.csv")},
    }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research AoA pass-1 sensitivity",
        "",
        out["scope"],
        "",
        f"Current actual mean-curve score: clipped {fmt(rows[0].get('curve_fitness_clipped'))}, p {fmt(rows[0].get('p_value'))}, n {rows[0].get('n_words')}, mean-monthly {fmt(rows[0].get('mean_monthly_score'))}.",
        "",
        "## Best legal rows",
        "",
        "| mode | beta | clipped score | p | n | mean |ΔS_1M| | max |ΔS| after 10M |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in best_legal:
        lines.append(f"| {r['mode']} | {r['beta']} | {fmt(r.get('curve_fitness_clipped'))} | {fmt(r.get('p_value'))} | {r.get('n_words')} | {fmt(r.get('mean_abs_delta_surprisal_1M'))} | {fmt(r.get('max_abs_delta_after_10M'))} |")
    lines += ["", "## Oracle-only rows", "", "| mode | beta | clipped score | p | n | mean |ΔS_1M| |", "|---|---:|---:|---:|---:|---:|"]
    for r in best_oracle:
        lines.append(f"| {r['mode']} | {r['beta']} | {fmt(r.get('curve_fitness_clipped'))} | {fmt(r.get('p_value'))} | {r.get('n_words')} | {fmt(r.get('mean_abs_delta_surprisal_1M'))} |")
    lines += ["", f"Full JSON: `{rel(out_dir / 'summary.json')}`"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "summary_json": rel(out_dir / "summary.json"), "summary_md": rel(out_dir / "summary.md"), "current": rows[0], "best_legal": best_legal[:5], "best_oracle": best_oracle[:5]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
