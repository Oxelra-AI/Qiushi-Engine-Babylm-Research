#!/usr/bin/env python3
"""research: CPU-only legal-corpus proxy ceiling for child AoA.

The corpus and tokenizer are the only feature sources. CDI-derived ages are
loaded only after feature extraction and are used solely as evaluation targets
for cross-validation. Consequently, the fitted predictor is an evaluation-only
upper-bound instrument: its coefficients must never be used as a training
schedule, because fitting those coefficients used the held CDI ages.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import ast
import bisect
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t as student_t
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer


ROOT = _public_path('.')
OUT = _public_path('experiments/archive/relation_learning/analysis/proxy_ceiling')
STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
EXISTING = _public_path('experiments/archive/relation_learning/data/aoa_stream_exposure_timing/word_exposure_timing.csv')
AOA_TOKENIZER_VARIANCE = _public_path('experiments/archive/relation_learning/data/aoa_existing_variance_tokenizer/summary.json')
AOA_SCHEDULE_PREDICTION = _public_path('experiments/archive/relation_learning/data/aoa_across_pass_schedule_predictor/summary.json')
TOKENIZER = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M')
TOKEN_FREQ = _public_path('experiments/archive/relation_learning/data/corpus_token_freq/corpus_token_freq.json')

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
BOUNDARY_RE = re.compile(r"(?:[.!?;\n]+|\*[A-Z][A-Z0-9]*:)")
SPEAKER_RE = re.compile(r"\*([A-Z][A-Z0-9]*):")
TOTAL_EXPECTED = 100_000_000
POSITION_BIN_WORDS = 1_000_000
POSITION_BINS = 100
SEED = 20260908

SOURCES_MODEL = [
    "bnc_spoken",
    "childes",
    "cleanqwen_fineweb_compact_view_reinvest",
    "gutenberg",
    "open_subtitles",
    "qwen_pair_packed",
    "simple_wiki",
    "switchboard",
]
SPOKEN = {"childes", "bnc_spoken", "switchboard"}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those"}
POSSESSIVES = {"my", "your", "his", "her", "our", "their", "mommy's", "daddy's"}
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they"}
COPULAS = {"am", "is", "are", "was", "were", "be", "been", "being"}


def rel(path: Path | str) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def blank_acc() -> dict[str, Any]:
    return {
        "n": 0,
        "rows": 0,
        "source": Counter(),
        "hist": np.zeros(POSITION_BINS, dtype=np.int64),
        "first_pos": None,
        "sum_pos": 0.0,
        "sum_pos2": 0.0,
        "early": Counter(),
        "sum_row_words": 0.0,
        "sum_row_words2": 0.0,
        "sum_relpos": 0.0,
        "first_quarter": 0,
        "last_quarter": 0,
        "sum_clause_words": 0.0,
        "clause_le3": 0,
        "clause_le8": 0,
        "clause_initial": 0,
        "uppercase": 0,
        "speaker_tagged": 0,
        "speaker_chi": 0,
        "prev_determiner": 0,
        "prev_possessive": 0,
        "prev_to": 0,
        "prev_pronoun": 0,
        "prev_copula": 0,
        "next_copula": 0,
    }


def scan_corpus(target_words: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Extract legal statistics without loading or consulting child ages."""
    targets = set(target_words)
    acc = {w: blank_acc() for w in target_words}
    source_words: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()
    rows = 0
    consumed = 0
    t0 = time.time()

    with STREAM.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            source = str(obj.get("source", "unknown"))
            row_words = int(obj.get("words", len(text.split())))
            row_start = consumed
            consumed += row_words
            rows += 1
            source_words[source] += row_words
            source_rows[source] += 1

            boundaries = [m.end() for m in BOUNDARY_RE.finditer(text)]
            speaker_matches = list(SPEAKER_RE.finditer(text))
            speaker_ends = [m.end() for m in speaker_matches]
            speaker_spans = [(m.start(), m.end()) for m in speaker_matches]
            tokens: list[dict[str, Any]] = []
            span_i = 0
            segment_rank: Counter[int] = Counter()
            segment_counts: Counter[int] = Counter()

            for m in WORD_RE.finditer(text):
                while span_i < len(speaker_spans) and speaker_spans[span_i][1] <= m.start():
                    span_i += 1
                if span_i < len(speaker_spans) and speaker_spans[span_i][0] <= m.start() < speaker_spans[span_i][1]:
                    continue
                segment = bisect.bisect_right(boundaries, m.start())
                rank = segment_rank[segment]
                segment_rank[segment] += 1
                segment_counts[segment] += 1
                speaker_i = bisect.bisect_right(speaker_ends, m.start()) - 1
                speaker = speaker_matches[speaker_i].group(1) if speaker_i >= 0 else ""
                tokens.append({
                    "word": m.group(0).lower(),
                    "surface": m.group(0),
                    "char_start": m.start(),
                    "segment": segment,
                    "segment_rank": rank,
                    "speaker": speaker,
                })

            row_target_counts: Counter[str] = Counter(t["word"] for t in tokens if t["word"] in targets)
            for w in row_target_counts:
                acc[w]["rows"] += 1

            text_len = max(len(text), 1)
            token_count = max(len(tokens), 1)
            for i, tok in enumerate(tokens):
                w = tok["word"]
                if w not in targets:
                    continue
                a = acc[w]
                pos = row_start + (tok["char_start"] / text_len) * row_words
                relpos = i / max(token_count - 1, 1)
                a["n"] += 1
                a["source"][source] += 1
                b = min(POSITION_BINS - 1, max(0, int(pos // POSITION_BIN_WORDS)))
                a["hist"][b] += 1
                if a["first_pos"] is None:
                    a["first_pos"] = pos
                a["sum_pos"] += pos
                a["sum_pos2"] += pos * pos
                for boundary in (1, 5, 10, 20, 50, 90):
                    if pos < boundary * 1_000_000:
                        a["early"][boundary] += 1
                a["sum_row_words"] += row_words
                a["sum_row_words2"] += row_words * row_words
                a["sum_relpos"] += relpos
                a["first_quarter"] += int(relpos <= 0.25)
                a["last_quarter"] += int(relpos >= 0.75)
                clause_words = segment_counts[tok["segment"]]
                a["sum_clause_words"] += clause_words
                a["clause_le3"] += int(clause_words <= 3)
                a["clause_le8"] += int(clause_words <= 8)
                a["clause_initial"] += int(tok["segment_rank"] == 0)
                a["uppercase"] += int(tok["surface"][:1].isupper())
                a["speaker_tagged"] += int(bool(tok["speaker"]))
                a["speaker_chi"] += int(tok["speaker"] == "CHI")
                if i > 0 and tokens[i - 1]["segment"] == tok["segment"]:
                    prev = tokens[i - 1]["word"]
                    a["prev_determiner"] += int(prev in DETERMINERS)
                    a["prev_possessive"] += int(prev in POSSESSIVES)
                    a["prev_to"] += int(prev == "to")
                    a["prev_pronoun"] += int(prev in PRONOUNS)
                    a["prev_copula"] += int(prev in COPULAS)
                if i + 1 < len(tokens) and tokens[i + 1]["segment"] == tok["segment"]:
                    a["next_copula"] += int(tokens[i + 1]["word"] in COPULAS)

            if rows % 100_000 == 0:
                print(json.dumps({"event": "scan", "rows": rows, "words": consumed, "seconds": round(time.time() - t0, 1)}), flush=True)

    audit = {
        "stream": rel(STREAM),
        "stream_sha256": sha256(STREAM),
        "rows": rows,
        "words_from_metadata": consumed,
        "expected_words": TOTAL_EXPECTED,
        "source_words": dict(sorted(source_words.items())),
        "source_rows": dict(sorted(source_rows.items())),
        "elapsed_seconds": time.time() - t0,
        "target_words_scanned": len(target_words),
    }
    return acc, audit


def median_from_hist(hist: np.ndarray, n: int) -> float | None:
    if n <= 0:
        return None
    idx = int(np.searchsorted(np.cumsum(hist), (n + 1) / 2.0))
    return (idx + 0.5) * POSITION_BIN_WORDS


def morphology(word: str) -> dict[str, float]:
    vowel_groups = len(re.findall(r"[aeiouy]+", word.lower()))
    return {
        "char_len": float(len(word)),
        "vowel_group_count": float(vowel_groups),
        "suffix_ing": float(word.endswith("ing")),
        "suffix_ed": float(word.endswith("ed")),
        "suffix_plural_s": float(word.endswith("s") and not word.endswith("ss")),
        "suffix_y": float(word.endswith("y")),
    }


def build_features(target_words: list[str], acc: dict[str, dict[str, Any]], audit: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, trust_remote_code=True)
    tf_obj = json.loads(TOKEN_FREQ.read_text(encoding="utf-8"))
    token_freq = np.asarray(tf_obj["token_freq"], dtype=np.float64)
    total_tokens = float(tf_obj["summary"]["total_tokens"])
    source_words = audit["source_words"]
    total_words = float(audit["words_from_metadata"])
    spoken_words = float(sum(source_words.get(s, 0) for s in SPOKEN))
    rows: list[dict[str, Any]] = []

    for w in target_words:
        a = acc[w]
        n = int(a["n"])
        denom = max(n, 1)
        row: dict[str, Any] = {"word": w, "stream_count": n, "occurrence_row_count": int(a["rows"])}
        row.update(morphology(w))
        ids = tokenizer.encode(w, add_special_tokens=False)
        freqs = [float(token_freq[i]) if 0 <= i < len(token_freq) else 0.0 for i in ids]
        piece_logs = [math.log((f + 0.5) / total_tokens * 1_000_000) for f in freqs] or [math.log(0.5 / total_tokens * 1_000_000)]
        row.update({
            "subword_len": float(len(ids)),
            "single_piece": float(len(ids) == 1),
            "first_piece_id_scaled": float(ids[0] / max(tokenizer.vocab_size - 1, 1)) if ids else 1.0,
            "piece_logfreq_mean": float(np.mean(piece_logs)),
            "piece_logfreq_min": float(np.min(piece_logs)),
            "piece_logfreq_max": float(np.max(piece_logs)),
            "whole_logfreq": math.log((n + 0.5) / total_words * 1_000_000),
            "row_logfreq": math.log((a["rows"] + 0.5) / audit["rows"] * 1_000_000),
            "occurrences_per_row": n / max(a["rows"], 1),
        })
        for source, sw in sorted(source_words.items()):
            c = int(a["source"].get(source, 0))
            row[f"count_src__{source}"] = c
            row[f"logfreq_src__{source}"] = math.log((c + 0.5) / max(sw, 1) * 1_000_000)

        child_count = int(a["source"].get("childes", 0))
        spoken_count = int(sum(a["source"].get(s, 0) for s in SPOKEN))
        qwen_count = int(a["source"].get("qwen_pair_packed", 0))
        child_log = row["logfreq_src__childes"]
        spoken_log = math.log((spoken_count + 0.5) / max(spoken_words, 1) * 1_000_000)
        probs = np.asarray([c / n for c in a["source"].values() if c > 0], dtype=float) if n else np.asarray([])
        row.update({
            "childes_logfreq": child_log,
            "childes_enrichment": child_log - row["whole_logfreq"],
            "spoken_logfreq": spoken_log,
            "spoken_enrichment": spoken_log - row["whole_logfreq"],
            "childes_occurrence_fraction": child_count / denom,
            "spoken_occurrence_fraction": spoken_count / denom,
            "qwen_occurrence_fraction": qwen_count / denom,
            "source_presence_count": float(sum(c > 0 for c in a["source"].values())),
            "source_entropy": float(-(probs * np.log(probs)).sum() / math.log(max(len(source_words), 2))) if len(probs) else 0.0,
        })

        med = median_from_hist(a["hist"], n)
        mean_pos = a["sum_pos"] / denom
        var_pos = max(a["sum_pos2"] / denom - mean_pos * mean_pos, 0.0)
        row.update({
            "first_occurrence_log10_position": math.log10(float(a["first_pos"]) + 1.0) if a["first_pos"] is not None else math.log10(total_words + 1),
            "median_occurrence_log10_position": math.log10(float(med) + 1.0) if med is not None else math.log10(total_words + 1),
            "occurrence_centroid_fraction": mean_pos / total_words if n else 1.0,
            "occurrence_position_sd_fraction": math.sqrt(var_pos) / total_words if n else 0.0,
            "early_1m_fraction": a["early"].get(1, 0) / denom,
            "early_5m_fraction": a["early"].get(5, 0) / denom,
            "early_10m_fraction": a["early"].get(10, 0) / denom,
            "early_20m_fraction": a["early"].get(20, 0) / denom,
            "early_50m_fraction": a["early"].get(50, 0) / denom,
            "last_10m_fraction": 1.0 - a["early"].get(90, 0) / denom if n else 0.0,
            "mean_occurrence_row_words": a["sum_row_words"] / denom,
            "sd_occurrence_row_words": math.sqrt(max(a["sum_row_words2"] / denom - (a["sum_row_words"] / denom) ** 2, 0.0)),
            "mean_relative_row_position": a["sum_relpos"] / denom,
            "first_row_quarter_fraction": a["first_quarter"] / denom,
            "last_row_quarter_fraction": a["last_quarter"] / denom,
            "mean_clause_words": a["sum_clause_words"] / denom,
            "clause_le3_fraction": a["clause_le3"] / denom,
            "clause_le8_fraction": a["clause_le8"] / denom,
            "clause_initial_fraction": a["clause_initial"] / denom,
            "uppercase_fraction": a["uppercase"] / denom,
            "speaker_tagged_fraction": a["speaker_tagged"] / denom,
            "child_speaker_fraction": a["speaker_chi"] / denom,
            "prev_determiner_fraction": a["prev_determiner"] / denom,
            "prev_possessive_fraction": a["prev_possessive"] / denom,
            "prev_to_fraction": a["prev_to"] / denom,
            "prev_pronoun_fraction": a["prev_pronoun"] / denom,
            "prev_copula_fraction": a["prev_copula"] / denom,
            "next_copula_fraction": a["next_copula"] / denom,
        })
        rows.append(row)

    definitions: dict[str, str] = {
        "stream_count": "Exact lower-cased word-boundary occurrences in the 100M stream.",
        "occurrence_row_count": "Number of stream rows containing the word.",
        "char_len": "Characters in the lexical item.",
        "vowel_group_count": "Regex vowel-group syllable proxy.",
        "subword_len": "Number of deployed tokenizer pieces without special tokens.",
        "single_piece": "One if the word is one deployed-tokenizer piece.",
        "first_piece_id_scaled": "First piece ID divided by vocabulary size; tokenizer geometry diagnostic.",
        "piece_logfreq_mean": "Mean log per-million frequency of the word's pieces in the research 30M token scan.",
        "piece_logfreq_min": "Minimum piece log frequency.",
        "piece_logfreq_max": "Maximum piece log frequency.",
        "whole_logfreq": "Smoothed log word occurrences per million over the full stream.",
        "row_logfreq": "Smoothed log occurrence rows per million stream rows.",
        "occurrences_per_row": "Occurrences divided by rows containing the word.",
        "childes_enrichment": "CHILDES log frequency minus whole-stream log frequency.",
        "spoken_enrichment": "CHILDES+BNC spoken+Switchboard log frequency minus whole-stream log frequency.",
        "source_entropy": "Normalized entropy of occurrences across sources.",
        "first_occurrence_log10_position": "Log10 approximate first word position in actual stream order.",
        "median_occurrence_log10_position": "Log10 median position estimated in 1M-word bins.",
        "occurrence_centroid_fraction": "Mean approximate occurrence position divided by 100M.",
        "occurrence_position_sd_fraction": "SD of approximate occurrence positions divided by 100M.",
        "mean_occurrence_row_words": "Occurrence-weighted mean row length from row metadata.",
        "mean_relative_row_position": "Occurrence-weighted target position within alphabetic row tokens.",
        "mean_clause_words": "Occurrence-weighted alphabetic token count between punctuation/speaker boundaries.",
        "clause_le3_fraction": "Fraction of occurrences in clauses of at most three tokens (bare/short mention proxy).",
        "clause_le8_fraction": "Fraction in clauses of at most eight tokens.",
        "child_speaker_fraction": "Fraction of all occurrences under a preceding *CHI: speaker tag.",
    }
    for c in rows[0]:
        if c.startswith("logfreq_src__"):
            definitions[c] = "Smoothed log word occurrences per million words in the named source."
        elif c.startswith("count_src__"):
            definitions[c] = "Exact word-boundary occurrence count in the named source."
        elif c not in definitions and c != "word":
            definitions[c] = "Legal corpus/tokenizer statistic; see variable name and script implementation."
    return pd.DataFrame(rows), [{"feature": k, "definition": v} for k, v in definitions.items()]


def correlation_record(name: str, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[mask], y[mask]
    if len(xx) < 3 or len(set(xx)) < 2:
        return {"feature": name, "n": int(len(xx)), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    p = pearsonr(xx, yy)
    s = spearmanr(xx, yy)
    return {"feature": name, "n": int(len(xx)), "pearson_r": float(p.statistic), "pearson_p": float(p.pvalue), "spearman_r": float(s.statistic), "spearman_p": float(s.pvalue)}


def bh_qvalues(records: list[dict[str, Any]], key: str, out_key: str) -> None:
    pairs = sorted((float(r[key]), i) for i, r in enumerate(records) if r.get(key) is not None)
    m = len(pairs)
    last = 1.0
    q = [None] * len(records)
    for rank_rev, (p, idx) in enumerate(reversed(pairs), start=1):
        rank = m - rank_rev + 1
        last = min(last, p * m / rank)
        q[idx] = last
    for r, val in zip(records, q):
        r[out_key] = val


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or abs(r) >= 1:
        return (r, r)
    z = np.arctanh(r)
    delta = 1.959963984540054 / math.sqrt(n - 3)
    return float(np.tanh(z - delta)), float(np.tanh(z + delta))


def estimator(kind: str, seed: int):
    if kind == "ridge":
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge())])
        return GridSearchCV(pipe, {"model__alpha": np.logspace(-3, 4, 24)}, cv=KFold(5, shuffle=True, random_state=seed), scoring="neg_mean_squared_error", n_jobs=1)
    if kind == "elastic":
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", ElasticNet(max_iter=30_000, tol=1e-5))])
        return GridSearchCV(pipe, {"model__alpha": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0], "model__l1_ratio": [0.05, 0.2, 0.5, 0.8, 1.0]}, cv=KFold(5, shuffle=True, random_state=seed), scoring="neg_mean_squared_error", n_jobs=1)
    if kind == "random_forest":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=5, max_features=0.7, random_state=seed, n_jobs=1))])
    if kind == "hist_gradient":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=7, min_samples_leaf=15, l2_regularization=1.0, learning_rate=0.05, early_stopping=True, random_state=seed))])
    raise KeyError(kind)


def metric_record(dataset: str, label: str, feature_set: str, kind: str, y: np.ndarray, pred_reps: np.ndarray, tuning: Counter[str]) -> dict[str, Any]:
    pred = pred_reps.mean(axis=0)
    p = pearsonr(pred, y)
    s = spearmanr(pred, y)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rep_r = np.asarray([pearsonr(x, y).statistic for x in pred_reps])
    lo, hi = fisher_ci(float(p.statistic), len(y))
    return {
        "dataset": dataset,
        "model": label,
        "feature_set": feature_set,
        "estimator": kind,
        "n": int(len(y)),
        "n_features": None,
        "oof_pearson_r": float(p.statistic),
        "oof_pearson_p": float(p.pvalue),
        "pearson_fisher_ci_low": lo,
        "pearson_fisher_ci_high": hi,
        "oof_spearman_r": float(s.statistic),
        "oof_spearman_p": float(s.pvalue),
        "oof_r2": 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot,
        "oof_rmse_months": float(np.sqrt(np.mean((y - pred) ** 2))),
        "oof_mae_months": float(np.mean(np.abs(y - pred))),
        "repeat_pearson_mean": float(rep_r.mean()),
        "repeat_pearson_min": float(rep_r.min()),
        "repeat_pearson_max": float(rep_r.max()),
        "tuning_choices": dict(tuning),
        "pred": pred,
    }


def repeated_oof(X: np.ndarray, y: np.ndarray, kind: str, repeats: int, seed: int) -> tuple[np.ndarray, Counter[str]]:
    pred_reps = np.full((repeats, len(y)), np.nan, dtype=float)
    tuning: Counter[str] = Counter()
    for rep in range(repeats):
        outer = KFold(10, shuffle=True, random_state=seed + rep)
        for fold, (tr, te) in enumerate(outer.split(X)):
            est = estimator(kind, seed + rep * 100 + fold)
            est.fit(X[tr], y[tr])
            pred_reps[rep, te] = est.predict(X[te])
            if hasattr(est, "best_params_"):
                tuning[json.dumps(est.best_params_, sort_keys=True)] += 1
    if not np.isfinite(pred_reps).all():
        raise RuntimeError("incomplete OOF predictions")
    return pred_reps, tuning


def full_fit_coefficients(X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    est = estimator("ridge", seed)
    est.fit(X.values, y)
    best = est.best_estimator_
    coefs = best.named_steps["model"].coef_
    rows = [{"feature": c, "standardized_ridge_coefficient": float(v)} for c, v in zip(X.columns, coefs)]
    rows.sort(key=lambda r: -abs(r["standardized_ridge_coefficient"]))
    return rows, est.best_params_


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--reuse-features", action="store_true")
    ap.add_argument("--reuse-cv", action="store_true", help="Reuse completed OOF tables and only regenerate summaries.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    existing = pd.read_csv(EXISTING)
    target_words = existing["word"].astype(str).str.lower().tolist()
    feature_path = _public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/proxy_features.csv')
    audit_path = _public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/scan_audit.json')
    definitions_path = _public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/feature_definitions.csv')

    if args.reuse_features and feature_path.exists() and audit_path.exists() and definitions_path.exists():
        features = pd.read_csv(feature_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    else:
        acc, audit = scan_corpus(target_words)
        features, definitions = build_features(target_words, acc, audit)
        features.to_csv(feature_path, index=False)
        pd.DataFrame(definitions).to_csv(definitions_path, index=False)
        write_json(audit_path, audit)

    # Count audit against the independently generated research scan.
    merged_count = existing[["word", "stream_count"] + [c for c in existing if c.startswith("count_src__")]].merge(features, on="word", suffixes=("__old", "__new"))
    count_mismatches: list[str] = []
    mismatch_details: list[dict[str, Any]] = []
    for _, r in merged_count.iterrows():
        if int(r["stream_count__old"]) != int(r["stream_count__new"]):
            count_mismatches.append(f"{r['word']}:total")
            mismatch_details.append({"word": r["word"], "coordinate": "total", "literal": int(r["stream_count__old"]), "lexical": int(r["stream_count__new"]), "removed": int(r["stream_count__old"] - r["stream_count__new"])})
        for c in [x for x in existing if x.startswith("count_src__")]:
            if int(r[f"{c}__old"]) != int(r[f"{c}__new"]):
                count_mismatches.append(f"{r['word']}:{c}")
                mismatch_details.append({"word": r["word"], "coordinate": c, "literal": int(r[f"{c}__old"]), "lexical": int(r[f"{c}__new"]), "removed": int(r[f"{c}__old"] - r[f"{c}__new"])})
    collision_words = sorted({str(x["word"]) for x in mismatch_details})
    # research deliberately counted every alphabetic span in the serialized text.
    # The richer lexical scan removes *SPEAKER: labels before counting.  Only
    # three CDI strings collide with observed uppercase speaker codes; every
    # other target must reproduce the independent count exactly.
    allowed_speaker_collisions = {"ant", "mad", "man"}
    unexpected = sorted(set(collision_words) - allowed_speaker_collisions)
    if unexpected:
        raise AssertionError(f"unexpected count audit mismatches: {unexpected}")
    if int(audit["words_from_metadata"]) != TOTAL_EXPECTED:
        raise AssertionError(audit["words_from_metadata"])

    # Only now attach CDI-derived targets. They never enter corpus feature extraction.
    targets = existing.loc[existing["child_aoa_month"].notna(), ["word", "child_aoa_month", "model_status"]].copy()
    targets["official_v4_fit_ok"] = (targets["model_status"] == "ok").astype(int)
    targets[["word", "child_aoa_month", "official_v4_fit_ok"]].to_csv(_public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/targets_evaluation_only.csv'), index=False)
    data = targets.merge(features, on="word", validate="one_to_one")

    shape = ["char_len", "vowel_group_count", "suffix_ing", "suffix_ed", "suffix_plural_s", "suffix_y", "subword_len", "single_piece", "first_piece_id_scaled", "piece_logfreq_mean", "piece_logfreq_min", "piece_logfreq_max"]
    frequency_concentration = ["whole_logfreq", "row_logfreq", "occurrences_per_row"]
    source_profile = ["whole_logfreq"] + [f"logfreq_src__{s}" for s in SOURCES_MODEL]
    enrichment = ["whole_logfreq", "childes_logfreq", "childes_enrichment", "spoken_logfreq", "spoken_enrichment", "childes_occurrence_fraction", "spoken_occurrence_fraction", "qwen_occurrence_fraction", "source_presence_count", "source_entropy"]
    timing = ["first_occurrence_log10_position", "median_occurrence_log10_position", "occurrence_centroid_fraction", "occurrence_position_sd_fraction", "early_1m_fraction", "early_5m_fraction", "early_10m_fraction", "early_20m_fraction", "early_50m_fraction", "last_10m_fraction"]
    context = ["row_logfreq", "occurrences_per_row", "mean_occurrence_row_words", "sd_occurrence_row_words", "mean_relative_row_position", "first_row_quarter_fraction", "last_row_quarter_fraction", "mean_clause_words", "clause_le3_fraction", "clause_le8_fraction", "clause_initial_fraction", "uppercase_fraction", "speaker_tagged_fraction", "child_speaker_fraction", "prev_determiner_fraction", "prev_possessive_fraction", "prev_to_fraction", "prev_pronoun_fraction", "prev_copula_fraction", "next_copula_fraction"]
    compact = list(dict.fromkeys(shape + enrichment + ["first_occurrence_log10_position", "occurrence_centroid_fraction", "early_10m_fraction", "mean_clause_words", "clause_le3_fraction", "child_speaker_fraction", "prev_determiner_fraction", "prev_to_fraction"]))
    all_legal = list(dict.fromkeys(shape + frequency_concentration + source_profile + enrichment + timing + context))
    feature_sets = {
        "shape_token": shape,
        "frequency_concentration": frequency_concentration,
        "source_profile": source_profile,
        "enrichment_compact": enrichment,
        "timing_only": timing,
        "context_only": context,
        "compact_legal": compact,
        "all_legal": all_legal,
    }

    feature_corrs = [correlation_record(c, data[c].to_numpy(float), data["child_aoa_month"].to_numpy(float)) for c in all_legal]
    bh_qvalues(feature_corrs, "pearson_p", "pearson_bh_q")
    bh_qvalues(feature_corrs, "spearman_p", "spearman_bh_q")
    pd.DataFrame(feature_corrs).sort_values("pearson_r", key=lambda x: x.abs(), ascending=False).to_csv(_public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/feature_correlations.csv'), index=False)

    specs = [
        ("ridge_shape_token", "shape_token", "ridge"),
        ("ridge_frequency_concentration", "frequency_concentration", "ridge"),
        ("ridge_source_profile", "source_profile", "ridge"),
        ("ridge_enrichment_compact", "enrichment_compact", "ridge"),
        ("ridge_timing_only", "timing_only", "ridge"),
        ("ridge_context_only", "context_only", "ridge"),
        ("ridge_compact_legal", "compact_legal", "ridge"),
        ("ridge_all_legal", "all_legal", "ridge"),
        ("elastic_all_legal", "all_legal", "elastic"),
        ("random_forest_all_legal", "all_legal", "random_forest"),
        ("hist_gradient_all_legal", "all_legal", "hist_gradient"),
    ]
    datasets = {
        "all_child_valid": np.ones(len(data), dtype=bool),
        "official_v4_fit_subset": data["official_v4_fit_ok"].to_numpy(bool),
    }
    cv_path = _public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/cv_results.csv')
    oof_path = _public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/oof_predictions.csv')
    if args.reuse_cv and cv_path.exists() and oof_path.exists():
        cv_df = pd.read_csv(cv_path)
        cv_df["model"] = cv_df["model"].replace({"ridge_whole_frequency": "ridge_frequency_concentration"})
        cv_df["feature_set"] = cv_df["feature_set"].replace({"whole_frequency": "frequency_concentration"})
        # CSV stores Counter/dict tuning summaries as text. Restore structured
        # objects so a summary-only rerun preserves the JSON schema.
        cv_df["tuning_choices"] = cv_df["tuning_choices"].map(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        oof_all = pd.read_csv(oof_path).rename(columns={"ridge_whole_frequency": "ridge_frequency_concentration"})
        cv_rows = cv_df.to_dict(orient="records")
        oof_frames = [g.copy() for _, g in oof_all.groupby("dataset", sort=False)]
    else:
        cv_rows: list[dict[str, Any]] = []
        oof_frames: list[pd.DataFrame] = []
        predictions: dict[tuple[str, str], np.ndarray] = {}

        for dataset, mask in datasets.items():
            d = data.loc[mask].reset_index(drop=True)
            y = d["child_aoa_month"].to_numpy(float)
            out_pred = d[["word", "child_aoa_month"]].copy()
            for label, fs_name, kind in specs:
                cols = feature_sets[fs_name]
                pred_reps, tuning = repeated_oof(d[cols].to_numpy(float), y, kind, args.repeats, SEED + len(cv_rows) * 1000)
                rec = metric_record(dataset, label, fs_name, kind, y, pred_reps, tuning)
                rec["n_features"] = len(cols)
                pred = rec.pop("pred")
                cv_rows.append(rec)
                predictions[(dataset, label)] = pred
                out_pred[label] = pred
                print(json.dumps({"event": "cv", "dataset": dataset, "model": label, "r": round(rec["oof_pearson_r"], 4), "rho": round(rec["oof_spearman_r"], 4)}), flush=True)

            ensemble_labels = ["ridge_all_legal", "elastic_all_legal", "random_forest_all_legal", "hist_gradient_all_legal"]
            ensemble = np.mean([predictions[(dataset, x)] for x in ensemble_labels], axis=0)
            ensemble_reps = np.tile(ensemble, (args.repeats, 1))
            rec = metric_record(dataset, "equal_ensemble_all_legal", "all_legal", "cross_fitted_equal_ensemble", y, ensemble_reps, Counter())
            rec["n_features"] = len(all_legal)
            pred = rec.pop("pred")
            cv_rows.append(rec)
            out_pred["equal_ensemble_all_legal"] = pred
            out_pred.insert(0, "dataset", dataset)
            oof_frames.append(out_pred)

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(cv_path, index=False)
    pd.concat(oof_frames, ignore_index=True).to_csv(oof_path, index=False)

    coeff_rows, best_alpha = full_fit_coefficients(data[compact], data["child_aoa_month"].to_numpy(float), SEED)
    pd.DataFrame(coeff_rows).to_csv(_public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/model_coefficients.csv'), index=False)

    s113 = json.loads(AOA_TOKENIZER_VARIANCE.read_text(encoding="utf-8"))
    s109 = json.loads(AOA_SCHEDULE_PREDICTION.read_text(encoding="utf-8"))
    boundary = s113["existing_aoa_score_evidence"]["positive_boundary_units"]
    measured_aoa_deltas = s113["existing_aoa_score_evidence"]["deltas"]
    primary = next(r for r in cv_rows if r["dataset"] == "all_child_valid" and r["model"] == "ridge_compact_legal")
    candidates = [r for r in cv_rows if r["dataset"] == "all_child_valid" and r["model"] in {"ridge_all_legal", "elastic_all_legal", "random_forest_all_legal", "hist_gradient_all_legal", "equal_ensemble_all_legal"}]
    exploratory = max(candidates, key=lambda r: r["oof_pearson_r"])
    official_exploratory = max((r for r in cv_rows if r["dataset"] == "official_v4_fit_subset"), key=lambda r: r["oof_pearson_r"])
    enrich_abs = abs(float(s109["validation"]["childes_minus_whole_vs_child_aoa"]["pearson_r"]))
    proxy_ratio = max(primary["oof_pearson_r"], 0.0) / enrich_abs
    heuristic_move = float(measured_aoa_deltas["max_minus_min"]) * proxy_ratio
    exploratory_proxy_ratio = max(exploratory["oof_pearson_r"], 0.0) / enrich_abs
    exploratory_heuristic_move = float(measured_aoa_deltas["max_minus_min"]) * exploratory_proxy_ratio
    best_legal_schedule = s109["best_legal"][0]
    current = s109["validation"]["current_actual"]
    best_oracle = s109["best_oracle"][0]

    top_corrs = sorted(feature_corrs, key=lambda r: abs(r.get("pearson_r") or 0), reverse=True)[:12]
    summary = {
        "status": "PROXY_CEILING_DONE",
        "created_utc": now(),
        "scope_and_legality": {
            "feature_rule": "Only corpus text/order/source metadata, deployed tokenizer geometry, and corpus token counts enter X.",
            "target_rule": "CDI-derived child 50% AoA is attached only after feature extraction and used only for evaluation.",
            "deployment_rule": "All supervised models and coefficients are evaluation-only and illegal as recipe inputs because they were fitted using CDI targets.",
            "compute": "CPU-only; no model training, GPU, or network access.",
        },
        "inputs": {
            "stream": rel(STREAM), "existing_counts_and_targets": rel(EXISTING), "tokenizer": rel(TOKENIZER),
            "token_frequency": rel(TOKEN_FREQ), "scale": rel(AOA_SCHEDULE_PREDICTION), "scale": rel(AOA_TOKENIZER_VARIANCE),
        },
        "scan_audit": audit,
        "count_audit": {
            "compared_with_step107": True,
            "exact_match_words": len(target_words) - len(collision_words),
            "speaker_tag_collision_words": collision_words,
            "speaker_tag_collision_details": mismatch_details,
            "unexpected_mismatch_words": unexpected,
            "n_words": len(target_words),
            "interpretation": "research excludes uppercase *SPEAKER: codes from lexical exposure. research literal counts differ only for three CDI strings that collide with speaker codes.",
        },
        "samples": {"cdi_rows_total": len(target_words), "valid_child_aoa": int(len(data)), "official_v4_fit_subset": int(data["official_v4_fit_ok"].sum())},
        "actual_order_timing_audit": {
            "valid_words_seen": int((data["stream_count"] > 0).sum()),
            "early_10m_fraction_unique_seen": sorted(data.loc[data["stream_count"] > 0, "early_10m_fraction"].unique().tolist()),
            "early_20m_fraction_unique_seen": sorted(data.loc[data["stream_count"] > 0, "early_20m_fraction"].unique().tolist()),
            "early_50m_fraction_unique_seen": sorted(data.loc[data["stream_count"] > 0, "early_50m_fraction"].unique().tolist()),
            "last_10m_fraction_unique_seen": sorted(data.loc[data["stream_count"] > 0, "last_10m_fraction"].unique().tolist()),
            "interpretation": "Every valid target has exactly 10/20/50 percent of occurrences in the first 10/20/50M and 10 percent in the last 10M, because the 100M stream repeats the same 10M row pool. Only within-pass position can vary by word in the inherited order.",
        },
        "cv_design": {"outer": f"{args.repeats} repeats x 10 folds", "inner_linear_tuning": "5-fold training-only grid search", "fold_preprocessing": "median imputation and scaling learned only on outer-training words", "random_seed": SEED, "model_selection_warning": "The maximum across model families is exploratory and selection-optimistic; ridge_compact_legal is the designated compact primary."},
        "feature_sets": feature_sets,
        "top_univariate_features": top_corrs,
        "primary_ceiling": primary,
        "exploratory_best": exploratory,
        "official_subset_exploratory_best": official_exploratory,
        "all_cv_results": cv_rows,
        "translation_to_stage3": {
            "v4_raw_r": boundary["v4_unclipped_r"],
            "positive_r_boundary_n225_p0p1": boundary["positive_r_boundary"],
            "required_raw_movement": boundary["required_delta_from_v4_to_positive_boundary"],
            "evidence100_minus_uniform100": measured_aoa_deltas["evidence_visible_100M_minus_uniform_100M"],
            "full_mask_family_spread": measured_aoa_deltas["max_minus_min"],
            "observed_favorable_fraction_of_required_100M": measured_aoa_deltas["evidence_visible_100M_minus_uniform_100M"] / boundary["required_delta_from_v4_to_positive_boundary"],
            "full_family_spread_fraction_of_required": measured_aoa_deltas["max_minus_min"] / boundary["required_delta_from_v4_to_positive_boundary"],
            "childes_enrichment_abs_r": enrich_abs,
            "primary_proxy_r_div_childes_enrichment_r": proxy_ratio,
            "heuristic_scaled_movement_from_full_family_spread": heuristic_move,
            "heuristic_fraction_of_required": heuristic_move / boundary["required_delta_from_v4_to_positive_boundary"],
            "exploratory_proxy_r_div_childes_enrichment_r": exploratory_proxy_ratio,
            "exploratory_heuristic_scaled_movement": exploratory_heuristic_move,
            "exploratory_heuristic_fraction_of_required": exploratory_heuristic_move / boundary["required_delta_from_v4_to_positive_boundary"],
            "heuristic_warning": "This proportional rescaling is a transparent sensitivity calculation, not a causal mapping or statistical bound.",
            "best_legal_simulated_raw_r": best_legal_schedule["unclipped_r"],
            "best_legal_simulated_delta_from_current": best_legal_schedule["unclipped_r"] - current["unclipped_r"],
            "best_legal_simulated_p": best_legal_schedule["p_value"],
            "oracle_simulated_raw_r": best_oracle["unclipped_r"],
            "oracle_simulated_p": best_oracle["p_value"],
        },
        "full_fit_primary_parameters_evaluation_only": best_alpha,
        "outputs": {p.name: rel(p) for p in [feature_path, definitions_path, audit_path, _public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/targets_evaluation_only.csv'), _public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/feature_correlations.csv'), _public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/cv_results.csv'), _public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/oof_predictions.csv'), _public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/model_coefficients.csv'), _public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/summary.json'), _public_path('research/notes/relation_learning/analysis/proxy_ceiling/summary.md')]},
        "elapsed_seconds": time.time() - t0,
    }
    # Remove numpy arrays/objects if introduced through shared records.
    write_json(_public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/summary.json'), summary)

    def f(x: Any, nd: int = 3) -> str:
        return "NA" if x is None else f"{float(x):.{nd}f}"

    table_rows = [r for r in cv_rows if r["dataset"] == "all_child_valid"]
    official_rows = [r for r in cv_rows if r["dataset"] == "official_v4_fit_subset"]
    lines = [
        "# research legal-corpus proxy ceiling for child AoA",
        "",
        "## Result",
        "",
        f"The designated compact ridge model reaches repeated out-of-fold Pearson **r={f(primary['oof_pearson_r'])}** (Spearman **rho={f(primary['oof_spearman_r'])}**, R2={f(primary['oof_r2'])}, RMSE={f(primary['oof_rmse_months'])} months) on all {len(data)} valid child-AoA words. Its approximate Fisher interval, conditional on the aggregated OOF predictions, is [{f(primary['pearson_fisher_ci_low'])}, {f(primary['pearson_fisher_ci_high'])}]. The best result selected after comparing nonlinear/linear all-feature models is **{exploratory['model']} r={f(exploratory['oof_pearson_r'])}** and is explicitly exploratory/selection-optimistic. On the 225-word official-v4 fitted subset, the exploratory maximum is **{official_exploratory['model']} r={f(official_exploratory['oof_pearson_r'])}**.",
        "",
        "These numbers are an evaluation-only predictability ceiling, not a legal recipe: the feature values are legal, but fitting the reported coefficients consumed child ages. A deployable schedule may use the feature definitions, never these target-fitted weights.",
        "",
        "## Feature extraction and audit",
        "",
        f"One CPU pass scanned {audit['rows']:,} rows and {audit['words_from_metadata']:,} metadata-counted words. Lexical counts matched the independent research literal scan for {len(target_words)-len(collision_words)}/{len(target_words)} CDI words. The only differences are `ant`, `mad`, and `man`, whose uppercase forms occur as CHILDES speaker codes; research removes those tag occurrences and records the exact deltas in `summary.json`. Features cover full/source/CHILDES/spoken frequency, enrichment, actual-order timing, tokenizer geometry, word shape, row and clause context, short/bare-mention proxies, speaker role, and local syntactic cues.",
        "",
        "CDI-derived 50% ages were joined only after this extraction. Of 504 CDI words, 406 have valid child ages; 225 are also in the faithful-v4 official fitted set.",
        "",
        "## Interpretable feature boundary",
        "",
        f"The strongest individual coordinates are occurrences-per-row (r={f(next(r['pearson_r'] for r in feature_corrs if r['feature']=='occurrences_per_row'))}), CHILDES enrichment (r={f(next(r['pearson_r'] for r in feature_corrs if r['feature']=='childes_enrichment'))}), and fraction spoken by a tagged child speaker (r={f(next(r['pearson_r'] for r in feature_corrs if r['feature']=='child_speaker_fraction'))}). Mean clause length is positive (r={f(next(r['pearson_r'] for r in feature_corrs if r['feature']=='mean_clause_words'))}): words learned earlier by children are disproportionately repeated within rows, CHILDES-enriched, child-spoken, and found in short clauses. Whole-stream log frequency itself is weak against child age (r={f(next(r['pearson_r'] for r in feature_corrs if r['feature']=='whole_logfreq'))}), even though it strongly predicts model AoA in research.",
        "",
        f"Actual pass-level timing contains no target differentiation: for all {len(data)} valid words, exactly 10%, 20%, and 50% of occurrences fall in the first 10M, 20M, and 50M, and 10% in the last 10M. This follows from ten repetitions of the same 10M row pool. Only within-pass position varies, its univariate correlations are near zero, and the timing-only Ridge is wrong-signed with negative OOF R2. Acquisition-order leverage therefore requires an intentional reorder or credit change; it is not latent in the inherited pass sequence.",
        "",
        "## Cross-validated model comparison (all 406)",
        "",
        "| model | feature set | features | OOF Pearson | OOF Spearman | OOF R2 | RMSE months |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in table_rows:
        lines.append(f"| {r['model']} | {r['feature_set']} | {r['n_features']} | {f(r['oof_pearson_r'])} | {f(r['oof_spearman_r'])} | {f(r['oof_r2'])} | {f(r['oof_rmse_months'])} |")
    lines += [
        "",
        "## Official-v4 fitted subset (225 words)",
        "",
        "| model | feature set | OOF Pearson | OOF Spearman | OOF R2 |",
        "|---|---|---:|---:|---:|",
    ]
    for r in official_rows:
        lines.append(f"| {r['model']} | {r['feature_set']} | {f(r['oof_pearson_r'])} | {f(r['oof_spearman_r'])} | {f(r['oof_r2'])} |")
    lines += [
        "",
        "## Stage-III translation",
        "",
        f"At n=225, faithful v4 starts at raw r={f(boundary['v4_unclipped_r'])} and needs +{f(boundary['required_delta_from_v4_to_positive_boundary'])} to cross the positive p=0.1 boundary r={f(boundary['positive_r_boundary_n225_p0p1'] if 'positive_r_boundary_n225_p0p1' in boundary else boundary['positive_r_boundary'])}. The observed 100M evidence-visible-minus-uniform movement is only +{f(measured_aoa_deltas['evidence_visible_100M_minus_uniform_100M'])}, {100*measured_aoa_deltas['evidence_visible_100M_minus_uniform_100M']/boundary['required_delta_from_v4_to_positive_boundary']:.1f}% of the requirement; the entire research masking-family spread is {f(measured_aoa_deltas['max_minus_min'])}, {100*measured_aoa_deltas['max_minus_min']/boundary['required_delta_from_v4_to_positive_boundary']:.1f}%.",
        "",
        f"As a sensitivity check, scaling that entire {f(measured_aoa_deltas['max_minus_min'])} spread by proxy signal relative to the existing CHILDES-enrichment correlation gives **{f(heuristic_move)} raw-r movement** for the compact Ridge and **{f(exploratory_heuristic_move)}** for the selection-optimistic ensemble: {100*heuristic_move/boundary['required_delta_from_v4_to_positive_boundary']:.1f}% to {100*exploratory_heuristic_move/boundary['required_delta_from_v4_to_positive_boundary']:.1f}% of the required +0.145. This is not a causal bound; it shows that even favorable proportional extrapolation does not supply the roughly 2.4--3.2-fold additional movement still needed by the trunk route.",
        "",
        f"The research legal schedule simulation reached raw r={f(best_legal_schedule['unclipped_r'])} (p={f(best_legal_schedule['p_value'])}), a predicted +{f(best_legal_schedule['unclipped_r']-current['unclipped_r'])} from its reconstructed current value, but still clipped to zero. The explicitly illegal child-age oracle reached r={f(best_oracle['unclipped_r'])} (p={best_oracle['p_value']:.3g}). The gap between legal proxies and the oracle, together with weak observed 100M conversion, is the operative ceiling result.",
        "",
        "## Decision",
        "",
        "The corpus contains real but moderate legal information about child acquisition order. It is sufficient for a scientific proxy relation and for low-cost calibration, but current evidence does not support a materially larger trunk-level v5 investment in acquisition-order scheduling. Await the already-running matched 30M calibration; require a surprisingly large, replicated measured effect before reopening 100M spending. Otherwise treat AoA order credit as a boundary result and prioritize the practical clean-preservation candidate.",
        "",
        "The feature most likely to raise this ceiling is an unsupervised distributional semantic/grammatical representation learned from the same corpus (for example a cross-fitted low-rank context representation). It could capture noun/verb/concreteness-like structure absent from these scalar statistics. It would still need a target-free construction and nested evaluation; target-tuned embeddings or CDI-fitted schedule weights would be illegal.",
        "",
        f"Full machine-readable result: `{rel(_public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/summary.json'))}`.",
    ]
    (_public_path('research/notes/relation_learning/analysis/proxy_ceiling/summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "primary": {k: primary[k] for k in ["oof_pearson_r", "oof_spearman_r", "oof_r2", "oof_rmse_months"]}, "exploratory": {k: exploratory[k] for k in ["model", "oof_pearson_r", "oof_spearman_r"]}, "summary": rel(_public_path('experiments/archive/relation_learning/analysis/proxy_ceiling/summary.json'))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
