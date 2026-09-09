#!/usr/bin/env python3
"""research: AoA pass-1 order predictor at official ladder resolution.

This replaces the research half-exposure crossing instrument, which cannot see
within-pass order under ten repeated passes.  The model here uses the existing
coherent86 AoA ladder surprisals and the actual training stream to estimate an
exposure-to-surprisal relation:

    mean_surprisal(word, checkpoint) = word_effect + checkpoint_effect
                                      + beta * log1p(cumulative lexical exposure)

Then it changes only the order of rows in the first 10M-word pass, recomputes
cumulative exposure at the official 1M..10M,20M..100M checkpoints, predicts
surprisal curves, and scores them with the same sigmoid/Pearson/p-value clipping
semantics as the official AoA evaluator.  Candidate orderings that use child AoA
labels are marked oracle_only and are not legal training recipes.

This is a route-selection instrument, not an official submitted score.
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
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

ROOT = _public_path('.')
STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
SURPRISAL = _public_path('experiments/archive/representation_and_objectives/data/alpha075_aoa_minctx0/collate_fast/results/hf_model/main/zero_shot/mlm/AoA_word/surprisal.json')
WORDS = _public_path('experiments/archive/relation_learning/data/aoa_curve_fit_analysis/word_curve_fit_records.csv')
SUMMARY = _public_path('experiments/archive/relation_learning/data/aoa_curve_fit_analysis/summary.json')
CDI = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')
TOKENIZER_PATH = _public_path('experiments/archive/representation_and_objectives/training/runs/alpha075_exact_replay_from82M_seed43022/hf_model_alpha0p75/chck_100M')
UTILS_PARENT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/aoa_pass1_order_predictor')

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
PASS_WORDS = 10_000_000


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
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def corr(xs: list[Any], ys: list[Any]) -> dict[str, Any]:
    vals: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        fx, fy = safe_float(x), safe_float(y)
        if fx is not None and fy is not None:
            vals.append((fx, fy))
    if len(vals) < 3 or len({x for x, _ in vals}) < 2 or len({y for _, y in vals}) < 2:
        return {"n": len(vals), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    a = [x for x, _ in vals]
    b = [y for _, y in vals]
    pr = pearsonr(a, b)
    sr = spearmanr(a, b)
    return {"n": len(vals), "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue), "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue)}


def format_float(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        v = float(x)
        if not math.isfinite(v):
            return "NA"
        return f"{v:.{nd}f}"
    except Exception:
        return str(x)


def load_step106_records() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with WORDS.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = str(row.get("word", "")).lower()
            if w:
                out[w] = dict(row)
    return out


def load_cdi_words() -> list[str]:
    df = pd.read_csv(CDI)
    return [str(w).lower() for w in df["word"].tolist() if isinstance(w, str) and w]


def load_aggregate_surprisals() -> tuple[list[str], list[int], dict[tuple[str, int], dict[str, Any]]]:
    obj = read_json(SURPRISAL)
    accum: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in obj.get("results", []):
        w = str(r.get("target_word", "")).lower()
        if not w:
            continue
        wc = int(r.get("word_count"))
        s = safe_float(r.get("surprisal"))
        if s is not None:
            accum[(w, wc)].append(float(s))
    words = sorted({w for (w, _wc) in accum})
    checkpoints = sorted({wc for (_w, wc) in accum})
    agg: dict[tuple[str, int], dict[str, Any]] = {}
    for k, vals in accum.items():
        agg[k] = {"mean_surprisal": float(np.mean(vals)), "n_contexts": int(len(vals))}
    return words, checkpoints, agg


def row_counts(text: str, targets: set[str]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for m in WORD_RE.finditer(text):
        w = m.group(0).lower()
        if w in targets:
            c[w] += 1
    return dict(c)


def score_pass1_rows(pass1_rows: list[dict[str, Any]], features: dict[str, dict[str, float | None]]) -> None:
    for r in pass1_rows:
        counts: dict[str, int] = r["counts"]
        n = int(sum(counts.values()))
        words = max(int(r["words"]), 1)
        density = n / words
        if n <= 0:
            for key in ["childes_freq", "childes_ratio", "spoken_freq", "whole_freq", "child_early", "cdi_density", "childes_composite"]:
                r[f"score_{key}"] = -1e9 if key != "cdi_density" else 0.0
            continue
        def wavg(name: str) -> float | None:
            vals = []
            weights = []
            for w, c in counts.items():
                v = features.get(w, {}).get(name)
                if v is not None and math.isfinite(float(v)):
                    vals.append(float(v))
                    weights.append(int(c))
            if not vals:
                return None
            return float(np.average(vals, weights=weights))
        for score_name, feat_name in [
            ("childes_freq", "childes_logfreq"),
            ("childes_ratio", "childes_minus_whole"),
            ("spoken_freq", "spoken_mix_logfreq"),
            ("whole_freq", "whole_logfreq"),
            ("child_early", "oracle_child_early_score"),
        ]:
            v = wavg(feat_name)
            r[f"score_{score_name}"] = float(v) if v is not None else -1e9
        r["score_cdi_density"] = density
        # Legal composite: child-directed lexical concentration, enrichment over total mix,
        # and a small preference for rows containing many AoA/CDI target tokens.
        cf = r["score_childes_freq"] if r["score_childes_freq"] > -1e8 else -50.0
        cr = r["score_childes_ratio"] if r["score_childes_ratio"] > -1e8 else -50.0
        r["score_childes_composite"] = float(cf + 0.75 * cr + 1.5 * min(0.05, density))


def scan_stream_for_exposures(stream: pathlib.Path, target_words: list[str], checkpoints: list[int], max_rows: int = 0) -> dict[str, Any]:
    targets = set(target_words)
    cp = sorted(checkpoints)
    cp_index = 0
    cum: Counter[str] = Counter()
    exposures: dict[int, Counter[str]] = {x: Counter() for x in cp}
    source_words: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()
    source_word_counts: dict[str, Counter[str]] = defaultdict(Counter)
    whole_counts: Counter[str] = Counter()
    pass1_rows: list[dict[str, Any]] = []
    pass1_total: Counter[str] = Counter()
    pass1_words = 0
    pass2_probe: list[tuple[str, int, str]] = []
    pass1_probe: list[tuple[str, int, str]] = []
    consumed = 0
    n_rows = 0
    t0 = time.time()
    with stream.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            src = str(obj.get("source", "unknown"))
            words_meta = int(obj.get("words", len(text.split())))
            counts = row_counts(text, targets)
            before = consumed
            after = consumed + words_meta
            # Fill any checkpoints crossed inside this row by a uniform-within-row approximation.
            while cp_index < len(cp) and cp[cp_index] <= after:
                boundary = cp[cp_index]
                if boundary < before:
                    cp_index += 1
                    continue
                frac = 0.0 if words_meta <= 0 else max(0.0, min(1.0, (boundary - before) / words_meta))
                ccopy = Counter(cum)
                if frac > 0 and counts:
                    for w, c in counts.items():
                        ccopy[w] += frac * int(c)
                exposures[boundary] = ccopy
                cp_index += 1
            consumed = after
            n_rows += 1
            source_words[src] += words_meta
            source_rows[src] += 1
            for w, c in counts.items():
                cum[w] += int(c)
                whole_counts[w] += int(c)
                source_word_counts[w][src] += int(c)
            if before < PASS_WORDS:
                # Keep only complete first-pass rows. The stream is expected to hit 10M on a row boundary;
                # if a row crossed the boundary this still records it and the candidate exposure uses fractions.
                row = {"row_index": len(pass1_rows), "global_row_index": n_rows - 1, "source": src, "words": words_meta, "counts": counts, "text_head": text[:80]}
                pass1_rows.append(row)
                pass1_words += words_meta
                for w, c in counts.items():
                    pass1_total[w] += int(c)
                if len(pass1_probe) < 12:
                    pass1_probe.append((src, words_meta, text[:80]))
            elif PASS_WORDS <= before < 2 * PASS_WORDS and len(pass2_probe) < 12:
                pass2_probe.append((src, words_meta, text[:80]))
            if max_rows and n_rows >= max_rows:
                break
            if n_rows % 100000 == 0:
                print(json.dumps({"event": "scan_progress", "rows": n_rows, "words": consumed, "elapsed_sec": round(time.time() - t0, 1)}), flush=True)
    if cp_index < len(cp):
        # For truncated max_rows runs, fill remaining checkpoints with current cumulative counts.
        for j in range(cp_index, len(cp)):
            exposures[cp[j]] = Counter(cum)
    pass_repeat_probe_equal = pass1_probe == pass2_probe[: len(pass1_probe)] if pass2_probe else None
    return {
        "rows": n_rows,
        "consumed_words": consumed,
        "elapsed_sec": time.time() - t0,
        "exposures": exposures,
        "whole_counts": whole_counts,
        "source_word_counts": source_word_counts,
        "source_words": source_words,
        "source_rows": source_rows,
        "pass1_rows": pass1_rows,
        "pass1_words": pass1_words,
        "pass1_total": pass1_total,
        "pass_repeat_probe_equal": pass_repeat_probe_equal,
        "pass1_probe": pass1_probe,
        "pass2_probe": pass2_probe,
    }


def build_word_features(words: list[str], research: dict[str, dict[str, Any]], scan: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    whole_counts: Counter[str] = scan["whole_counts"]
    source_word_counts: dict[str, Counter[str]] = scan["source_word_counts"]
    source_words: Counter[str] = scan["source_words"]
    total_words = max(sum(source_words.values()), 1)
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
        opensub = math.log((source_word_counts.get(w, Counter()).get("open_subtitles", 0) + 0.5) / max(source_words.get("open_subtitles", 1), 1) * 1_000_000)
        spoken = math.log((source_word_counts.get(w, Counter()).get("childes", 0) + source_word_counts.get(w, Counter()).get("bnc_spoken", 0) + source_word_counts.get(w, Counter()).get("switchboard", 0) + 0.5) / spoken_den * 1_000_000)
        feats[w] = {
            "child_aoa": child,
            "model_aoa": model,
            "whole_logfreq": whole,
            "childes_logfreq": childes,
            "bnc_logfreq": bnc,
            "opensub_logfreq": opensub,
            "spoken_mix_logfreq": spoken,
            "childes_minus_whole": childes - whole,
            "oracle_child_early_score": None if child is None else -child,
        }
    return feats


def candidate_order(pass1_rows: list[dict[str, Any]], mode: str) -> list[int]:
    idxs = list(range(len(pass1_rows)))
    if mode == "current":
        return idxs
    if mode == "source_childes_first":
        rank = {s: i for i, s in enumerate(["childes", "bnc_spoken", "switchboard", "open_subtitles", "simple_wiki", "qwen_pair_packed", "cleanqwen_fineweb_compact_view_reinvest", "gutenberg", "neutral_cleanqwen_topup_compact_reinvest::open_subtitles"])}
        return sorted(idxs, key=lambda i: (rank.get(pass1_rows[i]["source"], 99), pass1_rows[i]["row_index"]))
    if mode == "source_childes_only_then_current":
        return sorted(idxs, key=lambda i: (0 if pass1_rows[i]["source"] == "childes" else 1, pass1_rows[i]["row_index"]))
    if mode == "source_written_first_control":
        rank = {s: i for i, s in enumerate(["gutenberg", "qwen_pair_packed", "cleanqwen_fineweb_compact_view_reinvest", "simple_wiki", "open_subtitles", "bnc_spoken", "switchboard", "childes", "neutral_cleanqwen_topup_compact_reinvest::open_subtitles"])}
        return sorted(idxs, key=lambda i: (rank.get(pass1_rows[i]["source"], 99), pass1_rows[i]["row_index"]))
    m = re.match(r"sort__(.+)", mode)
    if not m:
        raise ValueError(mode)
    score = f"score_{m.group(1)}"
    return sorted(idxs, key=lambda i: (-float(pass1_rows[i].get(score, -1e9)), pass1_rows[i]["row_index"]))


def pass1_candidate_exposures(pass1_rows: list[dict[str, Any]], order: list[int], checkpoints: list[int], actual_exposures: dict[int, Counter[str]]) -> dict[int, Counter[str]]:
    out: dict[int, Counter[str]] = {}
    early_cps = [c for c in checkpoints if c <= PASS_WORDS]
    cp_i = 0
    cum: Counter[str] = Counter()
    pos = 0
    for oi in order:
        r = pass1_rows[oi]
        before = pos
        after = pos + int(r["words"])
        counts: dict[str, int] = r["counts"]
        while cp_i < len(early_cps) and early_cps[cp_i] <= after:
            boundary = early_cps[cp_i]
            frac = 0.0 if int(r["words"]) <= 0 else max(0.0, min(1.0, (boundary - before) / int(r["words"])))
            ccopy = Counter(cum)
            if counts and frac > 0:
                for w, c in counts.items():
                    ccopy[w] += frac * int(c)
            out[boundary] = ccopy
            cp_i += 1
        pos = after
        if counts:
            for w, c in counts.items():
                cum[w] += int(c)
    for j in range(cp_i, len(early_cps)):
        out[early_cps[j]] = Counter(cum)
    for c in checkpoints:
        if c > PASS_WORDS:
            out[c] = actual_exposures[c]
    return out


def fit_fixed_effect_exposure_model(words: list[str], checkpoints: list[int], agg: dict[tuple[str, int], dict[str, Any]], exposures: dict[int, Counter[str]]) -> dict[str, Any]:
    obs_rows: list[dict[str, Any]] = []
    for wi, w in enumerate(words):
        for ti, ck in enumerate(checkpoints):
            a = agg.get((w, ck))
            if not a:
                continue
            exp = float(exposures[ck].get(w, 0.0))
            obs_rows.append({"word": w, "checkpoint": ck, "word_i": wi, "time_i": ti, "y": float(a["mean_surprisal"]), "weight": math.sqrt(max(int(a.get("n_contexts", 1)), 1)), "log_exp": math.log1p(exp), "cum_exp": exp})
    n = len(obs_rows)
    nw = len(words)
    nt = len(checkpoints)
    # No intercept; all word and all checkpoint effects plus one shared exposure slope.
    X = np.zeros((n, nw + nt + 1), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    for i, r in enumerate(obs_rows):
        wt = float(r["weight"])
        X[i, int(r["word_i"])] = wt
        X[i, nw + int(r["time_i"])] = wt
        X[i, nw + nt] = float(r["log_exp"]) * wt
        y[i] = float(r["y"]) * wt
    coef, residuals, rank, singular = np.linalg.lstsq(X, y, rcond=None)
    pred = (X @ coef) / np.array([float(r["weight"]) for r in obs_rows])
    y_raw = np.array([float(r["y"]) for r in obs_rows])
    resid = y_raw - pred
    return {
        "obs_rows": obs_rows,
        "coef": coef,
        "rank": int(rank),
        "residual_sum_squares": float(np.sum(resid ** 2)),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "beta_log_cum_exposure": float(coef[nw + nt]),
        "n_obs": n,
        "n_words": nw,
        "n_checkpoints": nt,
    }


def predict_curve(model: dict[str, Any], words: list[str], checkpoints: list[int], exposures: dict[int, Counter[str]]) -> dict[tuple[str, int], float]:
    coef: np.ndarray = model["coef"]
    nw = len(words)
    nt = len(checkpoints)
    word_index = {w: i for i, w in enumerate(words)}
    time_index = {ck: i for i, ck in enumerate(checkpoints)}
    beta = float(coef[nw + nt])
    out: dict[tuple[str, int], float] = {}
    for w in words:
        wi = word_index[w]
        for ck in checkpoints:
            ti = time_index[ck]
            x = math.log1p(float(exposures[ck].get(w, 0.0)))
            out[(w, ck)] = float(coef[wi] + coef[nw + ti] + beta * x)
    return out


def sigmoid_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a / (1 + np.exp(-b * (x - c))) + d


def model_aoa_from_curve(surprisals: list[float], steps: list[int], vocab_size: int, n_subword_tokens: int) -> tuple[float | None, str]:
    vals = np.asarray(surprisals, dtype=float)
    st = np.asarray(steps, dtype=float)
    m = np.isfinite(vals)
    if m.sum() < 3:
        return None, "lt3"
    vals = vals[m]
    st = st[m]
    random_chance = float(n_subword_tokens * math.log(vocab_size))
    min_surprisal = float(np.min(vals))
    threshold = float(random_chance - 0.5 * (random_chance - min_surprisal))
    neg = -vals
    log_steps = np.log10(st + 1)
    rng = float(np.max(neg) - np.min(neg))
    if not math.isfinite(rng):
        return None, "nonfinite"
    initial_guess = [rng, 1.0, float(np.mean(log_steps)), float(np.min(neg))]
    lower = [0.0, 0.0, float(np.min(log_steps) - 1), float(np.min(neg) - 2 * rng - 1)]
    upper = [10 * rng + 1, 100.0, float(np.max(log_steps) + 1), float(np.max(neg) + 1)]
    try:
        popt, _ = curve_fit(sigmoid_function, log_steps, neg, p0=initial_guess, bounds=(lower, upper), maxfev=20000)
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
        log_aoa_step = float(c - math.log((a / (neg_threshold - d)) - 1) / b)
        aoa_step = float(10 ** log_aoa_step - 1)
    except Exception:
        return None, "solve_exception"
    if aoa_step < float(st[0]):
        return None, "before_first"
    if aoa_step > float(st[-1]):
        return None, "after_last"
    return log_aoa_step, "ok"


def load_tokenizer_and_evaluator(out_dir: pathlib.Path):
    cache = out_dir / "hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache / "hf_home"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "transformers_cache"))
    os.environ.setdefault("HF_MODULES_CACHE", str(cache / "modules"))
    sys.path.insert(0, str(UTILS_PARENT))
    from evaluation_pipeline.utils import AoAEvaluator  # noqa: WPS433
    from transformers import AutoTokenizer  # noqa: WPS433

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), trust_remote_code=True, local_files_only=True)
    return tok, AoAEvaluator(CDI)


def subword_lengths(tokenizer: Any, words: list[str]) -> dict[str, int]:
    prefix = tokenizer("The", add_special_tokens=False)["input_ids"]
    out: dict[str, int] = {}
    for w in words:
        ids = tokenizer("The " + w, add_special_tokens=False)["input_ids"]
        out[w] = max(1, len(ids) - len(prefix))
    return out


def official_like_score(pred: dict[tuple[str, int], float], words: list[str], checkpoints: list[int], tokenizer: Any, evaluator: Any) -> dict[str, Any]:
    results = []
    for w in words:
        for ck in checkpoints:
            results.append({"step": f"chck_{ck}", "word_count": int(ck), "target_word": w, "context_id": 0, "context": "", "surprisal": float(pred[(w, ck)])})
    # This uses the official per-word sigmoid fitting, child AoA fitting, Pearson, and p>0.1 clipping.
    return evaluator.compute_curve_fitness({"results": results}, tokenizer=tokenizer)


def candidate_summary(
    mode: str,
    pred: dict[tuple[str, int], float],
    words: list[str],
    checkpoints: list[int],
    tokenizer: Any,
    evaluator: Any,
    measured_model: dict[str, float],
    child_aoa: dict[str, float],
    swl: dict[str, int],
    vocab_size: int,
    oracle_only: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    score = official_like_score(pred, words, checkpoints, tokenizer, evaluator)
    word_rows = []
    model_words = []
    measured_vals = []
    child_vals = []
    pred_vals_for_child = []
    status_counts: Counter[str] = Counter()
    for w in words:
        curve = [pred[(w, ck)] for ck in checkpoints]
        aoa, status = model_aoa_from_curve(curve, checkpoints, vocab_size, swl[w])
        status_counts[status] += 1
        row = {"mode": mode, "word": w, "pred_model_aoa": aoa, "pred_model_status": status, "measured_model_aoa": measured_model.get(w), "child_aoa": child_aoa.get(w), "oracle_only": int(oracle_only)}
        word_rows.append(row)
        if aoa is not None and w in measured_model:
            model_words.append(aoa)
            measured_vals.append(measured_model[w])
        if aoa is not None and w in child_aoa:
            pred_vals_for_child.append(aoa)
            child_vals.append(child_aoa[w])
    c_model = corr(model_words, measured_vals)
    c_child = corr(pred_vals_for_child, child_vals)
    out = {
        "mode": mode,
        "oracle_only": bool(oracle_only),
        "official_curve_fitness_clipped": score.get("curve_fitness"),
        "official_p_value": score.get("p_value"),
        "official_n_words": score.get("n_words"),
        "official_mean_monthly_score": score.get("mean_monthly_score"),
        "unclipped_pred_child_pearson_r": c_child.get("pearson_r"),
        "unclipped_pred_child_pearson_p": c_child.get("pearson_p"),
        "pred_vs_measured_model_aoa_r": c_model.get("pearson_r"),
        "pred_vs_measured_model_aoa_p": c_model.get("pearson_p"),
        "pred_vs_measured_model_aoa_spearman": c_model.get("spearman_r"),
        "pred_model_status_counts": dict(status_counts),
    }
    return out, word_rows


def first_million_exposure_delta_rows(mode: str, exposure: dict[int, Counter[str]], actual: dict[int, Counter[str]], features: dict[str, dict[str, Any]], words: list[str]) -> list[dict[str, Any]]:
    # Return representative words whose exposure changes most at 1M/2M under a candidate.
    rows = []
    for w in words:
        d1 = float(exposure[1_000_000].get(w, 0.0) - actual[1_000_000].get(w, 0.0))
        d2 = float(exposure[2_000_000].get(w, 0.0) - actual[2_000_000].get(w, 0.0))
        if abs(d1) > 0 or abs(d2) > 0:
            rows.append({"mode": mode, "word": w, "delta_exp_1M": d1, "delta_exp_2M": d2, "child_aoa": features.get(w, {}).get("child_aoa"), "childes_logfreq": features.get(w, {}).get("childes_logfreq"), "whole_logfreq": features.get(w, {}).get("whole_logfreq")})
    rows.sort(key=lambda r: -abs(float(r["delta_exp_1M"])) - 0.5 * abs(float(r["delta_exp_2M"])))
    return rows[:40]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stream", default=str(STREAM))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--candidate", action="append", default=None, help="Candidate mode; default uses the built-in set.")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    words_obs, checkpoints, agg = load_aggregate_surprisals()
    research = load_step106_records()
    cdi_words = load_cdi_words()
    target_words = sorted(set(words_obs) | set(cdi_words))
    print(json.dumps({"event": "loaded_surprisals", "words_obs": len(words_obs), "checkpoints": checkpoints, "target_words": len(target_words), "utc": now()}), flush=True)

    scan = scan_stream_for_exposures(pathlib.Path(args.stream), target_words, checkpoints, max_rows=args.max_rows)
    features = build_word_features(target_words, research, scan)
    score_pass1_rows(scan["pass1_rows"], features)
    print(json.dumps({"event": "stream_scanned", "rows": scan["rows"], "words": scan["consumed_words"], "pass1_rows": len(scan["pass1_rows"]), "pass1_words": scan["pass1_words"], "elapsed_sec": round(scan["elapsed_sec"], 2)}), flush=True)

    # Fit only to observed words; features/row scores may use CDI-only words too.
    model = fit_fixed_effect_exposure_model(words_obs, checkpoints, agg, scan["exposures"])
    actual_pred = predict_curve(model, words_obs, checkpoints, scan["exposures"])
    tokenizer, evaluator = load_tokenizer_and_evaluator(out_dir)
    swl = subword_lengths(tokenizer, words_obs)
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    measured_model = {w: float(v) for w, rec in research.items() if (v := safe_float(rec.get("model_aoa"))) is not None}
    child_aoa = {w: float(v) for w, rec in research.items() if (v := safe_float(rec.get("child_aoa"))) is not None}

    candidate_modes = args.candidate or [
        "current",
        "source_childes_only_then_current",
        "source_childes_first",
        "sort__childes_freq",
        "sort__childes_ratio",
        "sort__spoken_freq",
        "sort__childes_composite",
        "sort__cdi_density",
        "sort__whole_freq",
        "source_written_first_control",
        "sort__child_early",
    ]
    oracle_modes = {"sort__child_early"}
    summaries: list[dict[str, Any]] = []
    word_pred_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    exposure_snapshot_rows: list[dict[str, Any]] = []

    # Current actual candidate: use actual exposures, not recomputed sorted pass1.
    cur_summary, cur_word_rows = candidate_summary("current", actual_pred, words_obs, checkpoints, tokenizer, evaluator, measured_model, child_aoa, swl, vocab_size, oracle_only=False)
    summaries.append(cur_summary)
    word_pred_rows.extend(cur_word_rows)

    for mode in candidate_modes:
        if mode == "current":
            continue
        order = candidate_order(scan["pass1_rows"], mode)
        exp = pass1_candidate_exposures(scan["pass1_rows"], order, checkpoints, scan["exposures"])
        pred = predict_curve(model, words_obs, checkpoints, exp)
        summ, wr = candidate_summary(mode, pred, words_obs, checkpoints, tokenizer, evaluator, measured_model, child_aoa, swl, vocab_size, oracle_only=(mode in oracle_modes))
        summ["mean_delta_pred_surprisal_1M_vs_current"] = float(np.mean([pred[(w, 1_000_000)] - actual_pred[(w, 1_000_000)] for w in words_obs if (w, 1_000_000) in pred]))
        summ["mean_abs_delta_pred_surprisal_1M_vs_current"] = float(np.mean([abs(pred[(w, 1_000_000)] - actual_pred[(w, 1_000_000)]) for w in words_obs if (w, 1_000_000) in pred]))
        # Checkpoints after 10M should be unchanged under pass-1-only reordering.
        post_delta = [abs(pred[(w, ck)] - actual_pred[(w, ck)]) for w in words_obs for ck in checkpoints if ck > PASS_WORDS]
        summ["max_abs_pred_delta_after_10M"] = float(max(post_delta)) if post_delta else 0.0
        summaries.append(summ)
        word_pred_rows.extend(wr)
        delta_rows.extend(first_million_exposure_delta_rows(mode, exp, scan["exposures"], features, words_obs))
        # A few interpretable exposure snapshots, including airplane if present.
        for w in ["airplane", "dog", "mommy", "car", "book", "zoo", "bath", "spaghetti"]:
            if w in words_obs:
                exposure_snapshot_rows.append({
                    "mode": mode,
                    "word": w,
                    "actual_1M": float(scan["exposures"][1_000_000].get(w, 0.0)),
                    "candidate_1M": float(exp[1_000_000].get(w, 0.0)),
                    "actual_2M": float(scan["exposures"][2_000_000].get(w, 0.0)),
                    "candidate_2M": float(exp[2_000_000].get(w, 0.0)),
                    "actual_10M": float(scan["exposures"][10_000_000].get(w, 0.0)),
                    "candidate_10M": float(exp[10_000_000].get(w, 0.0)),
                })
        print(json.dumps({"event": "candidate_done", "mode": mode, "curve_fitness": summ.get("official_curve_fitness_clipped"), "p": summ.get("official_p_value"), "unclipped_r": summ.get("unclipped_pred_child_pearson_r"), "n": summ.get("official_n_words")}), flush=True)

    # Add correlations of legal word features for context.
    child_words = [w for w in target_words if features.get(w, {}).get("child_aoa") is not None]
    model_words = [w for w in target_words if features.get(w, {}).get("model_aoa") is not None]
    feature_rows: list[dict[str, Any]] = []
    for feat in ["whole_logfreq", "childes_logfreq", "bnc_logfreq", "opensub_logfreq", "spoken_mix_logfreq", "childes_minus_whole"]:
        child_corr = corr([features[w].get(feat) for w in child_words], [features[w].get("child_aoa") for w in child_words])
        model_corr = corr([features[w].get(feat) for w in model_words], [features[w].get("model_aoa") for w in model_words])
        feature_rows.append({"feature": feat, "r_feature_child_aoa": child_corr.get("pearson_r"), "p_feature_child_aoa": child_corr.get("pearson_p"), "r_feature_measured_model_aoa": model_corr.get("pearson_r"), "p_feature_measured_model_aoa": model_corr.get("pearson_p"), "n_child": child_corr.get("n"), "n_model": model_corr.get("n")})

    # Sort summaries with legal modes first by clipped fitness/unclipped child relation.
    summaries_sorted = sorted(summaries, key=lambda r: (r.get("oracle_only", False), -(float(r.get("official_curve_fitness_clipped") or 0.0)), -(float(r.get("unclipped_pred_child_pearson_r") or -999.0))))
    write_csv(out_dir / "candidate_pass1_order_predictions.csv", summaries_sorted)
    write_csv(out_dir / "word_predicted_model_aoa_by_candidate.csv", word_pred_rows)
    write_csv(out_dir / "word_feature_correlations.csv", feature_rows)
    write_csv(out_dir / "early_exposure_delta_examples.csv", delta_rows)
    write_csv(out_dir / "selected_word_exposure_snapshots.csv", exposure_snapshot_rows)

    validation = {
        "actual_predicted_curve_official_score": summaries[0],
        "fit_rmse": model["rmse"],
        "fit_mae": model["mae"],
        "beta_log_cum_exposure": model["beta_log_cum_exposure"],
        "rank": model["rank"],
        "n_obs": model["n_obs"],
        "n_words": model["n_words"],
        "n_checkpoints": model["n_checkpoints"],
        "frequency_baseline_whole_stream_vs_measured_model_r": corr([features[w].get("whole_logfreq") for w in model_words], [features[w].get("model_aoa") for w in model_words]),
        "childes_frequency_bound_r_child": corr([features[w].get("childes_logfreq") for w in child_words], [features[w].get("child_aoa") for w in child_words]),
    }
    out = {
        "status": "AOA_PASS1_ORDER_PREDICTOR_DONE",
        "created_utc": now(),
        "inputs": {"stream": rel(args.stream), "surprisal": rel(SURPRISAL), "words": rel(WORDS), "cdi": rel(CDI), "tokenizer": rel(TOKENIZER_PATH)},
        "scope": "Pass-1-only reordering predictor. Checkpoints after 10M retain original exposure totals, so endpoint multiset exposure is conserved by construction in this instrument.",
        "scan": {"rows": scan["rows"], "consumed_words": scan["consumed_words"], "pass1_rows": len(scan["pass1_rows"]), "pass1_words": scan["pass1_words"], "pass_repeat_probe_equal": scan["pass_repeat_probe_equal"], "source_words": {k: int(v) for k, v in sorted(scan["source_words"].items())}, "elapsed_sec": round(scan["elapsed_sec"], 3)},
        "model": validation,
        "candidate_predictions": summaries_sorted,
        "feature_correlations": feature_rows,
        "important_caveat": "The linear fixed-effect law is an offline predictor. A positive predicted AoA score would justify only a 10M pass-1 screen, not a full 100M two-seed run; a weak validation or near-zero predicted movement means the instrument is not strong enough to authorize trunk spending.",
        "outputs": {"summary_json": rel(out_dir / "summary.json"), "summary_md": rel(out_dir / "summary.md"), "candidate_csv": rel(out_dir / "candidate_pass1_order_predictions.csv"), "word_predictions_csv": rel(out_dir / "word_predicted_model_aoa_by_candidate.csv"), "feature_csv": rel(out_dir / "word_feature_correlations.csv"), "delta_examples_csv": rel(out_dir / "early_exposure_delta_examples.csv"), "snapshots_csv": rel(out_dir / "selected_word_exposure_snapshots.csv")},
    }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research AoA pass-1 order predictor",
        "",
        out["scope"],
        "",
        "## Validation on observed coherent86 ladder",
        "",
        f"- Fixed-effect exposure model: {model['n_obs']} word-checkpoint observations, {model['n_words']} words, {model['n_checkpoints']} checkpoints, RMSE {model['rmse']:.4f}, MAE {model['mae']:.4f}.",
        f"- Shared beta on log1p cumulative exposure: {model['beta_log_cum_exposure']:.6f} nats.",
        f"- Current predicted official-like curve fitness: {format_float(summaries[0].get('official_curve_fitness_clipped'))}, p {format_float(summaries[0].get('official_p_value'))}, n {summaries[0].get('official_n_words')}; unclipped r {format_float(summaries[0].get('unclipped_pred_child_pearson_r'))}.",
        f"- Predicted model AoA vs measured model AoA on available words: r {format_float(summaries[0].get('pred_vs_measured_model_aoa_r'))}, Spearman {format_float(summaries[0].get('pred_vs_measured_model_aoa_spearman'))}.",
        f"- Whole-stream frequency baseline vs measured model AoA: r {format_float(validation['frequency_baseline_whole_stream_vs_measured_model_r'].get('pearson_r'))}.",
        f"- CHILDES frequency versus child AoA bound: r {format_float(validation['childes_frequency_bound_r_child'].get('pearson_r'))}.",
        "",
        "## Candidate pass-1 order predictions",
        "",
        "| mode | oracle only | clipped score | p | n | unclipped r(child) | r(pred model AoA, measured model AoA) | mean |ΔS_1M| | max |ΔS| after 10M |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries_sorted:
        lines.append("| {mode} | {oracle} | {score} | {p} | {n} | {ur} | {vr} | {mad} | {post} |".format(
            mode=r["mode"],
            oracle=int(bool(r.get("oracle_only"))),
            score=format_float(r.get("official_curve_fitness_clipped")),
            p=format_float(r.get("official_p_value")),
            n=r.get("official_n_words"),
            ur=format_float(r.get("unclipped_pred_child_pearson_r")),
            vr=format_float(r.get("pred_vs_measured_model_aoa_r")),
            mad=format_float(r.get("mean_abs_delta_pred_surprisal_1M_vs_current")),
            post=format_float(r.get("max_abs_pred_delta_after_10M")),
        ))
    lines += [
        "",
        "## Word-level proxy signals",
        "",
        "| feature | r(feature, child AoA) | p | r(feature, measured model AoA) | p |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in feature_rows:
        lines.append(f"| {r['feature']} | {format_float(r.get('r_feature_child_aoa'))} | {format_float(r.get('p_feature_child_aoa'))} | {format_float(r.get('r_feature_measured_model_aoa'))} | {format_float(r.get('p_feature_measured_model_aoa'))} |")
    lines += ["", f"Full JSON: `{rel(out_dir / 'summary.json')}`"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "summary_json": rel(out_dir / "summary.json"), "summary_md": rel(out_dir / "summary.md"), "validation": validation, "top_candidates": summaries_sorted[:6]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
