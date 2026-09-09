#!/usr/bin/env python3
"""research: across-pass AoA schedule predictor.

The research pass-1 instruments changed early checkpoints only.  This script
extends the same route-selection measurement to schedules that keep the 100M-row
multiset byte-identical but redistribute source/row families across the ten
10M-word phases.  The goal is to ask whether child-directed enrichment can reach
words whose fitted crossings occur after the first pass.

It also measures CDI-word frequency in the official 10M corpus versus child AoA,
so the current v4 stream can be compared with the base strict-small lexical
signal.

This is not an official BabyLM score and not a training run.  It predicts the
sign of an acquisition-order intervention before any trunk H100 spending.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import importlib.util
import json
import math
import pathlib
import re
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

ROOT = _public_path('.')
PREDICTOR = _public_path('experiments/archive/relation_learning/scripts/aoa_pass1_order_predictor.py')
STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
OFFICIAL10M = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_10M.jsonl')
OUT = _public_path('experiments/archive/relation_learning/data/aoa_across_pass_schedule_predictor')
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
PASS_WORDS = 10_000_000
TOTAL_WORDS = 100_000_000


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_mod():
    spec = importlib.util.spec_from_file_location("pred", PREDICTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PREDICTOR}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pred"] = mod
    spec.loader.exec_module(mod)
    return mod


def safe_float(x: Any) -> float | None:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


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
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def fmt(x: Any, nd: int = 4) -> str:
    v = safe_float(x)
    return "NA" if v is None else f"{v:.{nd}f}"


def corr(xs: list[Any], ys: list[Any]) -> dict[str, Any]:
    vals: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        fx, fy = safe_float(x), safe_float(y)
        if fx is not None and fy is not None:
            vals.append((fx, fy))
    if len(vals) < 3 or len({x for x, _ in vals}) < 2 or len({y for _, y in vals}) < 2:
        return {"n": len(vals), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr = pearsonr([x for x, _ in vals], [y for _, y in vals])
    sr = spearmanr([x for x, _ in vals], [y for _, y in vals])
    return {"n": len(vals), "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue), "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue)}


def row_counts(text: str, targets: set[str]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for m in WORD_RE.finditer(text):
        w = m.group(0).lower()
        if w in targets:
            c[w] += 1
    return dict(c)


def load_rows_and_actual_exposures(target_words: list[str], checkpoints: list[int], max_rows: int = 0) -> dict[str, Any]:
    targets = set(target_words)
    cp = sorted(checkpoints)
    exposures: dict[int, Counter[str]] = {c: Counter() for c in cp}
    cp_i = 0
    cum: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    source_words: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()
    source_word_counts: dict[str, Counter[str]] = defaultdict(Counter)
    whole_counts: Counter[str] = Counter()
    consumed = 0
    t0 = time.time()
    with STREAM.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            src = str(obj.get("source", "unknown"))
            words = int(obj.get("words", len(text.split())))
            counts = row_counts(text, targets)
            before = consumed
            after = before + words
            while cp_i < len(cp) and cp[cp_i] <= after:
                b = cp[cp_i]
                frac = 0.0 if words <= 0 else max(0.0, min(1.0, (b - before) / words))
                ccopy = Counter(cum)
                if frac > 0 and counts:
                    for w, n in counts.items():
                        ccopy[w] += frac * int(n)
                exposures[b] = ccopy
                cp_i += 1
            row = {"row_index": len(rows), "source": src, "words": words, "counts": counts, "text_head": text[:80]}
            rows.append(row)
            consumed = after
            source_words[src] += words
            source_rows[src] += 1
            if counts:
                for w, n in counts.items():
                    cum[w] += int(n)
                    whole_counts[w] += int(n)
                    source_word_counts[w][src] += int(n)
            if max_rows and len(rows) >= max_rows:
                break
            if len(rows) % 100000 == 0:
                print(json.dumps({"event": "scan_progress", "rows": len(rows), "words": consumed, "elapsed_sec": round(time.time() - t0, 1)}), flush=True)
    for j in range(cp_i, len(cp)):
        exposures[cp[j]] = Counter(cum)
    return {"rows": rows, "rows_n": len(rows), "consumed_words": consumed, "elapsed_sec": time.time() - t0,
            "exposures": exposures, "whole_counts": whole_counts, "source_word_counts": source_word_counts,
            "source_words": source_words, "source_rows": source_rows}


def count_official10m_frequency(target_words: list[str], child_aoa: dict[str, float]) -> dict[str, Any]:
    targets = set(target_words)
    counts: Counter[str] = Counter()
    source_words: Counter[str] = Counter()
    rows = 0; words_total = 0; t0 = time.time()
    with OFFICIAL10M.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            words = int(obj.get("words", len(text.split())))
            src = str(obj.get("source", "unknown"))
            c = row_counts(text, targets)
            counts.update(c)
            rows += 1; words_total += words; source_words[src] += words
    values = []
    ys = []
    for w, ch in child_aoa.items():
        values.append(math.log((counts.get(w, 0) + 0.5) / max(words_total, 1) * 1_000_000))
        ys.append(ch)
    return {"path": rel(OFFICIAL10M), "rows": rows, "words": words_total, "elapsed_sec": round(time.time() - t0, 3),
            "logfreq_vs_child_aoa": corr(values, ys), "source_words": dict(source_words)}


def build_features(words: list[str], research: dict[str, dict[str, Any]], scan: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    source_words: Counter[str] = scan["source_words"]
    total_words = max(sum(source_words.values()), 1)
    source_word_counts: dict[str, Counter[str]] = scan["source_word_counts"]
    whole_counts: Counter[str] = scan["whole_counts"]
    spoken_den = max(source_words.get("childes", 0) + source_words.get("bnc_spoken", 0) + source_words.get("switchboard", 0), 1)
    feats: dict[str, dict[str, float | None]] = {}
    for w in words:
        rec = research.get(w, {})
        child = safe_float(rec.get("child_aoa"))
        model = safe_float(rec.get("model_aoa"))
        wc = whole_counts.get(w, 0)
        whole = math.log((wc + 0.5) / total_words * 1_000_000)
        childes = math.log((source_word_counts.get(w, Counter()).get("childes", 0) + 0.5) / max(source_words.get("childes", 1), 1) * 1_000_000)
        bnc = math.log((source_word_counts.get(w, Counter()).get("bnc_spoken", 0) + 0.5) / max(source_words.get("bnc_spoken", 1), 1) * 1_000_000)
        spoken = math.log((source_word_counts.get(w, Counter()).get("childes", 0) + source_word_counts.get(w, Counter()).get("bnc_spoken", 0) + source_word_counts.get(w, Counter()).get("switchboard", 0) + 0.5) / spoken_den * 1_000_000)
        feats[w] = {"child_aoa": child, "model_aoa": model, "whole_logfreq": whole, "childes_logfreq": childes,
                    "bnc_logfreq": bnc, "spoken_mix_logfreq": spoken, "childes_minus_whole": childes - whole,
                    "oracle_child_early_score": None if child is None else -child}
    return feats


def score_rows(rows: list[dict[str, Any]], features: dict[str, dict[str, float | None]]) -> None:
    for r in rows:
        counts: dict[str, int] = r["counts"]
        n = int(sum(counts.values()))
        words = max(int(r["words"]), 1)
        density = n / words
        if n <= 0:
            for key in ["childes_freq", "childes_ratio", "spoken_freq", "whole_freq", "child_early", "cdi_density", "childes_composite"]:
                r[f"score_{key}"] = -1e9 if key != "cdi_density" else 0.0
            continue
        def wavg(name: str) -> float | None:
            vals=[]; weights=[]
            for w, c in counts.items():
                v = features.get(w, {}).get(name)
                if v is not None and math.isfinite(float(v)):
                    vals.append(float(v)); weights.append(int(c))
            if not vals:
                return None
            return float(np.average(vals, weights=weights))
        for score_name, feat_name in [("childes_freq", "childes_logfreq"), ("childes_ratio", "childes_minus_whole"),
                                      ("spoken_freq", "spoken_mix_logfreq"), ("whole_freq", "whole_logfreq"),
                                      ("child_early", "oracle_child_early_score")]:
            v = wavg(feat_name)
            r[f"score_{score_name}"] = float(v) if v is not None else -1e9
        r["score_cdi_density"] = float(density)
        cf = r["score_childes_freq"] if r["score_childes_freq"] > -1e8 else -50.0
        cr = r["score_childes_ratio"] if r["score_childes_ratio"] > -1e8 else -50.0
        r["score_childes_composite"] = float(cf + 0.75 * cr + 1.5 * min(0.05, density))


def taper_childes_quotas(total_childes: int, floor_frac: float, decay: float) -> list[float]:
    avg = total_childes / 10.0
    floor = floor_frac * avg
    remaining = max(total_childes - 10 * floor, 0.0)
    weights = [math.exp(-decay * p) for p in range(10)]
    sw = sum(weights)
    quotas = [floor + remaining * w / sw for w in weights]
    # Numerical adjustment.
    quotas[-1] += total_childes - sum(quotas)
    return quotas


def source_taper_order(rows: list[dict[str, Any]], source_words: Counter[str], floor_frac: float, decay: float, within: str) -> list[int]:
    sources = sorted(source_words)
    total_by_source = {s: float(source_words[s]) for s in sources}
    child_total = int(source_words.get("childes", 0))
    child_q = taper_childes_quotas(child_total, floor_frac, decay)
    nonchild_sources = [s for s in sources if s != "childes"]
    nonchild_total = max(sum(source_words[s] for s in nonchild_sources), 1)
    quotas: list[dict[str, float]] = []
    for p in range(10):
        pass_q: dict[str, float] = {}
        cq = min(child_q[p], PASS_WORDS * 0.985)
        pass_q["childes"] = cq
        rem = PASS_WORDS - cq
        for s in nonchild_sources:
            pass_q[s] = rem * float(source_words[s]) / nonchild_total
        quotas.append(pass_q)
    # For row-score variants, the pass quota decides how much CHILDES-like material
    # each phase receives, while each source queue is ordered by the chosen lexical
    # enrichment score.  This is the across-pass version of the hypothesized lever:
    # high child-directed/enrichment rows are not merely moved to the front of each
    # phase; they are allocated to earlier phases under a source-share floor.
    by_source: dict[str, list[int]] = {s: [] for s in sources}
    for i, r in enumerate(rows):
        by_source[str(r["source"])].append(i)
    if within == "childes_ratio":
        for s in sources:
            by_source[s].sort(key=lambda i: (-float(rows[i].get("score_childes_ratio", -1e9)), rows[i]["row_index"]))
    elif within == "childes_composite":
        for s in sources:
            by_source[s].sort(key=lambda i: (-float(rows[i].get("score_childes_composite", -1e9)), rows[i]["row_index"]))
    qs: dict[str, deque[int]] = {s: deque(v) for s, v in by_source.items()}
    order: list[int] = []
    pass_quota_records: list[dict[str, Any]] = []
    for p in range(10):
        pass_idxs: list[int] = []
        for s in ["childes"] + [x for x in sources if x != "childes"]:
            target = quotas[p].get(s, 0.0)
            used = 0
            while qs.get(s) and qs[s] and (used < target or (p == 9 and qs[s])):
                idx = qs[s].popleft()
                pass_idxs.append(idx)
                used += int(rows[idx]["words"])
                if p < 9 and used >= target:
                    break
            pass_quota_records.append({"pass": p + 1, "source": s, "target_words": target, "actual_words_taken": used})
        if within == "childes_ratio":
            pass_idxs.sort(key=lambda i: (-float(rows[i].get("score_childes_ratio", -1e9)), str(rows[i]["source"]), rows[i]["row_index"]))
        elif within == "childes_composite":
            pass_idxs.sort(key=lambda i: (-float(rows[i].get("score_childes_composite", -1e9)), str(rows[i]["source"]), rows[i]["row_index"]))
        # Default keeps source-blocked order inside each pass; this intentionally makes source timing visible.
        order.extend(pass_idxs)
    # If row granularity left any source queue nonempty, append in original source order.
    leftovers = []
    for s in sources:
        leftovers.extend(list(qs[s]))
    if leftovers:
        leftovers.sort(key=lambda i: rows[i]["row_index"])
        order.extend(leftovers)
    if len(order) != len(rows) or len(set(order)) != len(rows):
        raise RuntimeError(f"bad source_taper order len={len(order)} unique={len(set(order))} rows={len(rows)}")
    return order


def full_sort_order(rows: list[dict[str, Any]], score_name: str, reverse: bool = True) -> list[int]:
    key = f"score_{score_name}"
    if reverse:
        return sorted(range(len(rows)), key=lambda i: (-float(rows[i].get(key, -1e9)), rows[i]["row_index"]))
    return sorted(range(len(rows)), key=lambda i: (float(rows[i].get(key, -1e9)), rows[i]["row_index"]))


def current_order(rows: list[dict[str, Any]]) -> list[int]:
    return list(range(len(rows)))


def exposures_for_order(rows: list[dict[str, Any]], order: list[int], checkpoints: list[int]) -> dict[int, Counter[str]]:
    cp = sorted(checkpoints)
    cp_i = 0
    cum: Counter[str] = Counter()
    out: dict[int, Counter[str]] = {c: Counter() for c in cp}
    consumed = 0
    for oi in order:
        r = rows[oi]
        words = int(r["words"])
        counts: dict[str, int] = r["counts"]
        before = consumed
        after = before + words
        while cp_i < len(cp) and cp[cp_i] <= after:
            b = cp[cp_i]
            frac = 0.0 if words <= 0 else max(0.0, min(1.0, (b - before) / words))
            ccopy = Counter(cum)
            if frac > 0 and counts:
                for w, n in counts.items():
                    ccopy[w] += frac * int(n)
            out[b] = ccopy
            cp_i += 1
        consumed = after
        if counts:
            for w, n in counts.items():
                cum[w] += int(n)
    for j in range(cp_i, len(cp)):
        out[cp[j]] = Counter(cum)
    if consumed != TOTAL_WORDS:
        # Keep this explicit because legal accounting matters; row-order scoring can still proceed.
        print(json.dumps({"event": "word_total_warning", "consumed_words": consumed}), flush=True)
    return out


def pass_source_profile(rows: list[dict[str, Any]], order: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    source_words: Counter[str] = Counter()
    pass_i = 1
    pass_start = 0
    consumed = 0
    for oi in order:
        r = rows[oi]
        before = consumed
        after = consumed + int(r["words"])
        # Charge the row to the pass where it begins; row lengths are tiny relative to 10M.
        while before >= pass_i * PASS_WORDS and pass_i <= 10:
            rec = {"pass": pass_i, "start_word": pass_start, "end_word": pass_i * PASS_WORDS, "total_words_approx": sum(source_words.values())}
            rec.update({f"source_words__{s}": int(v) for s, v in source_words.items()})
            out.append(rec)
            pass_i += 1; pass_start = (pass_i - 1) * PASS_WORDS; source_words = Counter()
        source_words[str(r["source"])] += int(r["words"])
        consumed = after
    while pass_i <= 10:
        rec = {"pass": pass_i, "start_word": pass_start, "end_word": pass_i * PASS_WORDS, "total_words_approx": sum(source_words.values())}
        rec.update({f"source_words__{s}": int(v) for s, v in source_words.items()})
        out.append(rec)
        pass_i += 1; pass_start = (pass_i - 1) * PASS_WORDS; source_words = Counter()
    return out


def sigmoid_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a / (1 + np.exp(-b * (x - c))) + d


def model_aoa_from_curve(vals_in: list[float], steps_in: list[int], vocab_size: int, n_subword_tokens: int, maxfev: int = 20000) -> tuple[float | None, str]:
    vals = np.asarray(vals_in, dtype=float); st = np.asarray(steps_in, dtype=float)
    m = np.isfinite(vals)
    if int(m.sum()) < 3:
        return None, "lt3"
    vals = vals[m]; st = st[m]
    random_chance = float(n_subword_tokens * math.log(vocab_size))
    min_surprisal = float(np.min(vals))
    threshold = float(random_chance - 0.5 * (random_chance - min_surprisal))
    neg = -vals; log_steps = np.log10(st + 1)
    rng = float(np.max(neg) - np.min(neg))
    if not math.isfinite(rng):
        return None, "nonfinite"
    p0 = [rng, 1.0, float(np.mean(log_steps)), float(np.min(neg))]
    lower = [0.0, 0.0, float(np.min(log_steps) - 1), float(np.min(neg) - 2 * rng - 1)]
    upper = [10 * rng + 1, 100.0, float(np.max(log_steps) + 1), float(np.max(neg) + 1)]
    try:
        popt, _ = curve_fit(sigmoid_function, log_steps, neg, p0=p0, bounds=(lower, upper), maxfev=maxfev)
        a, b, c, d = [float(z) for z in popt]
    except Exception:
        return None, "fit_exception"
    neg_threshold = -threshold
    if b <= 1e-6 or a <= 1e-6:
        return None, "flat_or_zero_amplitude"
    if neg_threshold <= d:
        return None, "threshold_below_lower_asymptote"
    if neg_threshold >= a + d:
        return None, "threshold_above_upper_asymptote"
    try:
        log_aoa = float(c - math.log((a / (neg_threshold - d)) - 1) / b)
        aoa_step = float(10 ** log_aoa - 1)
    except Exception:
        return None, "solve_exception"
    if aoa_step < float(st[0]):
        return None, "before_first"
    if aoa_step > float(st[-1]):
        return None, "after_last"
    return log_aoa, "ok"


def score_job(job: dict[str, Any]) -> dict[str, Any]:
    mode = job["mode"]; beta = float(job["beta"]); checkpoints = job["checkpoints"]; words = job["words"]
    means = job["means"]; dlog = job.get("dlog"); swl = job["swl"]; child = job["child_aoa"]; vocab_size = int(job["vocab_size"])
    model_aoas=[]; child_vals=[]; statuses=Counter()
    maxfev = int(job.get("maxfev", 20000))
    for w in words:
        if w not in child:
            statuses["no_child_aoa"] += 1; continue
        if dlog is None:
            vals = [means[(w, ck)] for ck in checkpoints]
        else:
            vals = [means[(w, ck)] + beta * dlog[(w, ck)] for ck in checkpoints]
        aoa, status = model_aoa_from_curve(vals, checkpoints, vocab_size, swl[w], maxfev=maxfev)
        statuses[status] += 1
        if aoa is not None:
            model_aoas.append(float(aoa)); child_vals.append(float(child[w]))
    out = {"mode": mode, "beta": beta, "oracle_only": bool(job.get("oracle_only", False)), "n_words": len(model_aoas), "status_counts": dict(statuses)}
    if len(model_aoas) >= 3 and len(set(model_aoas)) > 1 and len(set(child_vals)) > 1:
        pr = pearsonr(model_aoas, child_vals); sr = spearmanr(model_aoas, child_vals)
        out.update({"unclipped_r": float(pr.statistic), "p_value": float(pr.pvalue), "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue), "curve_fitness_clipped": 0.0 if pr.pvalue > 0.1 else float(pr.statistic)})
    else:
        out.update({"unclipped_r": None, "p_value": None, "spearman_r": None, "spearman_p": None, "curve_fitness_clipped": 0.0})
    return out


def schedule_modes(rows: list[dict[str, Any]], source_words: Counter[str]) -> dict[str, list[int]]:
    modes: dict[str, list[int]] = {"current": current_order(rows)}
    for floor in [0.25, 0.40, 0.55, 0.70]:
        for decay in [0.15, 0.25, 0.40, 0.60]:
            tag = f"source_childes_taper_f{int(floor*100):02d}_d{int(decay*100):02d}"
            modes[tag] = source_taper_order(rows, source_words, floor, decay, within="source_block")
    for floor in [0.40, 0.55]:
        for decay in [0.25, 0.40]:
            modes[f"source_childes_taper_rowsort_ratio_f{int(floor*100):02d}_d{int(decay*100):02d}"] = source_taper_order(rows, source_words, floor, decay, within="childes_ratio")
            modes[f"source_childes_taper_rowsort_comp_f{int(floor*100):02d}_d{int(decay*100):02d}"] = source_taper_order(rows, source_words, floor, decay, within="childes_composite")
    for score in ["childes_ratio", "childes_composite", "childes_freq", "spoken_freq", "whole_freq", "cdi_density", "child_early"]:
        modes[f"global_sort__{score}"] = full_sort_order(rows, score, reverse=True)
    modes["global_sort__written_control"] = full_sort_order(rows, "childes_ratio", reverse=False)
    return modes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--betas", default="-0.10,-0.20,-0.36,-0.50,-0.75")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--curve-maxfev", type=int, default=6000)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mod = load_mod()
    words_obs, checkpoints, mean_records = mod.load_aggregate_surprisals()
    means = {(w, ck): float(v["mean_surprisal"]) for (w, ck), v in mean_records.items()}
    research = mod.load_step106_records()
    cdi_words = mod.load_cdi_words()
    target_words = sorted(set(words_obs) | set(cdi_words))
    child_aoa = {w: float(v) for w, rec in research.items() if (v := safe_float(rec.get("child_aoa"))) is not None}
    measured_model = {w: float(v) for w, rec in research.items() if (v := safe_float(rec.get("model_aoa"))) is not None}
    print(json.dumps({"event": "loaded_inputs", "words_obs": len(words_obs), "target_words": len(target_words), "checkpoints": checkpoints, "utc": now()}), flush=True)

    official10m = count_official10m_frequency(target_words, child_aoa)
    print(json.dumps({"event": "official10m_done", "corr": official10m["logfreq_vs_child_aoa"]}), flush=True)

    scan = load_rows_and_actual_exposures(target_words, checkpoints, max_rows=args.max_rows)
    if args.max_rows:
        print(json.dumps({"event": "truncated_scan_warning", "max_rows": args.max_rows}), flush=True)
    feats = build_features(target_words, research, scan)
    score_rows(scan["rows"], feats)
    tok, _evaluator = mod.load_tokenizer_and_evaluator(out_dir)
    swl = mod.subword_lengths(tok, words_obs)
    vocab_size = int(getattr(tok, "vocab_size", len(tok)))

    modes = schedule_modes(scan["rows"], scan["source_words"])
    betas = [float(b) for b in args.betas.split(",") if b.strip()]
    jobs = [{"mode": "current_actual", "beta": 0.0, "checkpoints": checkpoints, "words": words_obs, "means": means, "dlog": None, "swl": swl, "child_aoa": child_aoa, "vocab_size": vocab_size, "oracle_only": False, "maxfev": args.curve_maxfev}]
    delta_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    mode_order_paths: list[dict[str, Any]] = []
    t0 = time.time()
    for mi, (mode, order) in enumerate(modes.items(), 1):
        exp = exposures_for_order(scan["rows"], order, checkpoints)
        dlog = {(w, ck): math.log1p(float(exp[ck].get(w, 0.0))) - math.log1p(float(scan["exposures"][ck].get(w, 0.0))) for w in words_obs for ck in checkpoints}
        max_final = max(abs(dlog[(w, checkpoints[-1])]) for w in words_obs)
        delta_rows.append({"mode": mode,
                           "mean_abs_dlog_1M": float(np.mean([abs(dlog[(w, 1_000_000)]) for w in words_obs if 1_000_000 in checkpoints])),
                           "mean_dlog_1M": float(np.mean([dlog[(w, 1_000_000)] for w in words_obs if 1_000_000 in checkpoints])),
                           "mean_abs_dlog_10M": float(np.mean([abs(dlog[(w, 10_000_000)]) for w in words_obs if 10_000_000 in checkpoints])),
                           "mean_abs_dlog_20M": float(np.mean([abs(dlog[(w, 20_000_000)]) for w in words_obs if 20_000_000 in checkpoints])),
                           "mean_abs_dlog_50M": float(np.mean([abs(dlog[(w, 50_000_000)]) for w in words_obs if 50_000_000 in checkpoints])),
                           "max_abs_dlog_final": float(max_final)})
        for rec in pass_source_profile(scan["rows"], order):
            rec2 = {"mode": mode}; rec2.update(rec); profile_rows.append(rec2)
        if mode != "current":
            for beta in betas:
                jobs.append({"mode": mode, "beta": beta, "checkpoints": checkpoints, "words": words_obs, "means": means, "dlog": dlog, "swl": swl, "child_aoa": child_aoa, "vocab_size": vocab_size, "oracle_only": mode.endswith("child_early"), "maxfev": args.curve_maxfev})
        # Save only lightweight order maps for best-looking schedule later; full maps are written after scoring if needed.
        if mi % 10 == 0:
            print(json.dumps({"event": "mode_prepared", "mode_index": mi, "modes_total": len(modes), "elapsed_sec": round(time.time() - t0, 1)}), flush=True)
    write_csv(out_dir / "schedule_exposure_delta_summary.csv", delta_rows)
    write_csv(out_dir / "schedule_pass_source_profiles.csv", profile_rows)

    results: list[dict[str, Any]] = []
    d_by_mode = {r["mode"]: r for r in delta_rows}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(score_job, j) for j in jobs]
        for fut in as_completed(futs):
            res = fut.result()
            d = d_by_mode.get(res["mode"])
            if d:
                beta = abs(float(res["beta"]))
                for key in ["mean_abs_dlog_1M", "mean_dlog_1M", "mean_abs_dlog_10M", "mean_abs_dlog_20M", "mean_abs_dlog_50M", "max_abs_dlog_final"]:
                    res[key] = d.get(key)
                res["mean_abs_delta_surprisal_1M"] = beta * float(d.get("mean_abs_dlog_1M", 0.0))
                res["mean_abs_delta_surprisal_20M"] = beta * float(d.get("mean_abs_dlog_20M", 0.0))
                res["mean_abs_delta_surprisal_50M"] = beta * float(d.get("mean_abs_dlog_50M", 0.0))
            results.append(res)
            print(json.dumps({"event": "score_done", "mode": res["mode"], "beta": res["beta"], "clip": res.get("curve_fitness_clipped"), "r": res.get("unclipped_r"), "p": res.get("p_value"), "n": res.get("n_words")}), flush=True)
    rows_sorted = sorted(results, key=lambda r: (bool(r.get("oracle_only")), -(float(r.get("curve_fitness_clipped") or 0.0)), -(float(r.get("unclipped_r") or -999)), str(r.get("mode")), float(r.get("beta", 0.0))))
    write_csv(out_dir / "schedule_prediction_scores.csv", rows_sorted)

    current = next(r for r in results if r["mode"] == "current_actual")
    legal = [r for r in rows_sorted if not r.get("oracle_only") and r["mode"] != "current_actual"][:20]
    oracle = [r for r in rows_sorted if r.get("oracle_only")][:10]
    feature_rows = [
        {"feature": "whole_logfreq", **corr([feats[w].get("whole_logfreq") for w in child_aoa], [child_aoa[w] for w in child_aoa])},
        {"feature": "childes_logfreq", **corr([feats[w].get("childes_logfreq") for w in child_aoa], [child_aoa[w] for w in child_aoa])},
        {"feature": "childes_minus_whole", **corr([feats[w].get("childes_minus_whole") for w in child_aoa], [child_aoa[w] for w in child_aoa])},
        {"feature": "spoken_mix_logfreq", **corr([feats[w].get("spoken_mix_logfreq") for w in child_aoa], [child_aoa[w] for w in child_aoa])},
        {"feature": "whole_logfreq_vs_measured_model", **corr([feats[w].get("whole_logfreq") for w in measured_model], [measured_model[w] for w in measured_model])},
        {"feature": "childes_minus_whole_vs_measured_model", **corr([feats[w].get("childes_minus_whole") for w in measured_model], [measured_model[w] for w in measured_model])},
    ]
    write_csv(out_dir / "lexical_feature_correlations.csv", feature_rows)

    # Save order indices for the top non-oracle mode(s), so a later materializer can build exactly the tested schedule.
    saved_modes: set[str] = set()
    for r in legal[:5]:
        mode = str(r["mode"])
        if mode in saved_modes or mode not in modes:
            continue
        saved_modes.add(mode)
        path = out_dir / f"order_indices__{mode}.txt"
        path.write_text("\n".join(str(i) for i in modes[mode]) + "\n", encoding="utf-8")
        mode_order_paths.append({"mode": mode, "order_indices": rel(path)})

    summary = {"status": "AOA_ACROSS_PASS_SCHEDULE_PREDICTOR_DONE", "created_utc": now(),
               "scope": "Predicted official-AoA movement for byte-identical 100M stream reorderings that redistribute CHILDES-enriched rows across ten 10M-word phases; rows are not added or removed.",
               "inputs": {"stream": rel(STREAM), "official10m": rel(OFFICIAL10M), "predictor_base": rel(PREDICTOR)},
               "scan": {"rows": scan["rows_n"], "words": scan["consumed_words"], "elapsed_sec": round(scan["elapsed_sec"], 3),
                         "source_words": dict(scan["source_words"]), "source_rows": dict(scan["source_rows"])},
               "official10m_frequency": official10m,
               "validation": {"current_actual": current,
                              "whole_stream_frequency_vs_measured_model_aoa": corr([feats[w].get("whole_logfreq") for w in measured_model], [measured_model[w] for w in measured_model]),
                              "whole_stream_frequency_vs_child_aoa": corr([feats[w].get("whole_logfreq") for w in child_aoa], [child_aoa[w] for w in child_aoa]),
                              "childes_frequency_vs_child_aoa": corr([feats[w].get("childes_logfreq") for w in child_aoa], [child_aoa[w] for w in child_aoa]),
                              "childes_minus_whole_vs_child_aoa": corr([feats[w].get("childes_minus_whole") for w in child_aoa], [child_aoa[w] for w in child_aoa])},
               "betas": betas, "n_modes": len(modes), "best_legal": legal, "best_oracle": oracle,
               "saved_order_indices": mode_order_paths,
               "important_reading": "Positive predictions would justify a small trunk timing screen, not final admission. If predictions stay near zero, the across-pass schedule should not be launched as an AoA route.",
               "outputs": {"summary_json": rel(out_dir / "summary.json"), "summary_md": rel(out_dir / "summary.md"),
                           "scores_csv": rel(out_dir / "schedule_prediction_scores.csv"),
                           "delta_csv": rel(out_dir / "schedule_exposure_delta_summary.csv"),
                           "pass_profiles_csv": rel(out_dir / "schedule_pass_source_profiles.csv"),
                           "feature_csv": rel(out_dir / "lexical_feature_correlations.csv")}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research AoA across-pass schedule predictor", "", summary["scope"], "",
             f"Current actual reproduction: clipped {fmt(current.get('curve_fitness_clipped'))}, r {fmt(current.get('unclipped_r'))}, p {fmt(current.get('p_value'))}, n {current.get('n_words')}.",
             f"Current whole-stream log frequency vs child AoA r {fmt(summary['validation']['whole_stream_frequency_vs_child_aoa'].get('pearson_r'))}; vs measured model AoA r {fmt(summary['validation']['whole_stream_frequency_vs_measured_model_aoa'].get('pearson_r'))}.",
             f"CHILDES log frequency vs child AoA r {fmt(summary['validation']['childes_frequency_vs_child_aoa'].get('pearson_r'))}; CHILDES-minus-whole vs child AoA r {fmt(summary['validation']['childes_minus_whole_vs_child_aoa'].get('pearson_r'))}.",
             f"Official-only 10M log frequency vs child AoA r {fmt(official10m['logfreq_vs_child_aoa'].get('pearson_r'))}, p {fmt(official10m['logfreq_vs_child_aoa'].get('pearson_p'))}, n {official10m['logfreq_vs_child_aoa'].get('n')}.",
             "", "## Best legal predictions", "", "| mode | beta | clipped score | r | p | n | mean |dlog| 20M | mean |dlog| 50M |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in legal:
        lines.append(f"| {r['mode']} | {r['beta']} | {fmt(r.get('curve_fitness_clipped'))} | {fmt(r.get('unclipped_r'))} | {fmt(r.get('p_value'))} | {r.get('n_words')} | {fmt(r.get('mean_abs_dlog_20M'))} | {fmt(r.get('mean_abs_dlog_50M'))} |")
    lines += ["", "## Oracle-only predictions", "", "| mode | beta | clipped score | r | p | n |", "|---|---:|---:|---:|---:|---:|"]
    for r in oracle:
        lines.append(f"| {r['mode']} | {r['beta']} | {fmt(r.get('curve_fitness_clipped'))} | {fmt(r.get('unclipped_r'))} | {fmt(r.get('p_value'))} | {r.get('n_words')} |")
    lines += ["", "A schedule with a positive stored score would still require a trunk timing screen and endpoint-column checks.  If the best legal rows remain nonsignificant, the measured source-enrichment ceiling is not by itself enough to authorize H100 trunk spending.", "", f"Full JSON: `{rel(out_dir / 'summary.json')}`"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary_json": summary["outputs"]["summary_json"], "summary_md": summary["outputs"]["summary_md"], "current": current, "best_legal": legal[:8], "official10m": official10m["logfreq_vs_child_aoa"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
