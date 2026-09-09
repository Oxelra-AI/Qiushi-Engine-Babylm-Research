#!/usr/bin/env python3
"""Analyze whether the materialized C/S arms separate the intended mechanisms.

This script reads the already-written 4M JSONL arms, computes row-level surface
and tokenizer features for the union of selected rows, and measures whether the
C-high and S-high arms differ beyond source, position, genre/style, text quality,
and tokenizer-compression proxies. It does not launch training.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ARM_FILES = {
    "R-a": "r_a.jsonl",
    "R-b": "r_b.jsonl",
    "C-high": "c_high.jsonl",
    "C-control": "c_control.jsonl",
    "S-high": "s_high.jsonl",
}

TRAIN_FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]
SOURCE_COUNTS = {
    "bnc_spoken.train.txt": 4763,
    "childes.train.txt": 17757,
    "gutenberg.train.txt": 15986,
    "open_subtitles.train.txt": 14268,
    "simple_wiki.train.txt": 9572,
    "switchboard.train.txt": 154,
}
SOURCE_OFFSETS: dict[str, int] = {}
_offset = 0
for _src in TRAIN_FILES:
    SOURCE_OFFSETS[_src] = _offset
    _offset += SOURCE_COUNTS[_src]

FILLER = {
    "uh", "um", "erm", "hmm", "mhm", "huh", "yeah", "yep", "yup", "nah",
    "ok", "okay", "oh", "ah", "wow", "hm", "mm",
}
PRONOUNS = {
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs",
}
FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "that", "which", "who", "whom",
    "this", "these", "those", "to", "of", "in", "on", "for", "with", "by", "from", "at", "into", "over",
    "under", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "can", "could", "may", "might", "must", "shall", "should", "will", "would", "not", "n't",
}
DIALOGUE_RE = re.compile(r"^\*[A-Z]+:$|^[A-Z]:$|^\[[^\]]+\]$")
WORD_ALPHA_RE = re.compile(r"[A-Za-z]")
SENT_END_RE = re.compile(r"[.!?]")


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=float), q))


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std()), 6),
        "p10": round(float(np.percentile(arr, 10)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
    }


def safe_corr(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    aa = np.array(a, dtype=float)
    bb = np.array(b, dtype=float)
    if float(aa.std()) == 0.0 or float(bb.std()) == 0.0:
        return 0.0
    return round(float(np.corrcoef(aa, bb)[0, 1]), 6)


def is_noise_word(w: str) -> bool:
    if w.isdigit():
        return True
    if all(not c.isalpha() for c in w):
        return True
    if len(w) == 1 and not w.isalpha():
        return True
    return False


def read_arms(base: Path) -> tuple[dict[int, dict[str, Any]], dict[str, set[int]], dict[str, list[str]]]:
    rows: dict[int, dict[str, Any]] = {}
    arm_ids: dict[str, set[int]] = {}
    consistency: dict[str, list[str]] = defaultdict(list)
    for arm, fn in ARM_FILES.items():
        p = base / fn
        ids: set[int] = set()
        with p.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                eid = int(obj["example_id"])
                ids.add(eid)
                text = str(obj["text"])
                rec = {
                    "example_id": eid,
                    "text": text,
                    "words": int(obj["words"]),
                    "source": obj["source"],
                    "c_score": float(obj["c_score"]),
                    "s_score": float(obj["s_score"]),
                }
                if eid in rows:
                    old = rows[eid]
                    for key in ["text", "words", "source"]:
                        if old[key] != rec[key]:
                            consistency[arm].append(f"line {line_num} id {eid}: inconsistent {key}")
                    # prefer the first text; c/s should be identical to rounded precision
                    if abs(float(old["c_score"]) - rec["c_score"]) > 1e-6 or abs(float(old["s_score"]) - rec["s_score"]) > 1e-6:
                        consistency[arm].append(f"line {line_num} id {eid}: inconsistent scores")
                else:
                    rows[eid] = rec
        arm_ids[arm] = ids
    return rows, arm_ids, consistency


def add_basic_features(rows: dict[int, dict[str, Any]]) -> None:
    for rec in rows.values():
        text = rec["text"]
        words = text.split()
        n = max(len(words), 1)
        low = [w.strip(".,!?;:'\"()[]{}<>`).“”‘’").lower() for w in words]
        chars = sum(len(w) for w in words)
        alpha_words = sum(1 for w in words if WORD_ALPHA_RE.search(w))
        noise = sum(1 for w in words if is_noise_word(w))
        digit = sum(1 for w in words if any(c.isdigit() for c in w))
        punct_only = sum(1 for w in words if all(not c.isalnum() for c in w))
        fillers = sum(1 for w in low if w in FILLER)
        pronouns = sum(1 for w in low if w in PRONOUNS)
        function_words = sum(1 for w in low if w in FUNCTION_WORDS)
        dialogue = sum(1 for w in words if DIALOGUE_RE.match(w))
        upper = sum(1 for w in words if len(w) > 1 and w.isupper() and any(c.isalpha() for c in w))
        sent_end = len(SENT_END_RE.findall(text))
        quotes = text.count('"') + text.count("'") + text.count("“") + text.count("”")
        type_count = len(set(low))
        src = rec["source"]
        offset = SOURCE_OFFSETS.get(src, 0)
        denom = max(SOURCE_COUNTS.get(src, 1) - 1, 1)
        rec.update({
            "source_pos": (rec["example_id"] - offset) / denom,
            "char_per_word": chars / n,
            "alpha_word_rate": alpha_words / n,
            "noise_word_rate": noise / n,
            "digit_word_rate": digit / n,
            "punct_only_rate": punct_only / n,
            "filler_rate": fillers / n,
            "pronoun_rate": pronouns / n,
            "function_word_rate": function_words / n,
            "dialogue_marker_rate_row": dialogue / n,
            "upper_word_rate": upper / n,
            "sent_end_per_word": sent_end / n,
            "quote_char_per_word": quotes / n,
            "word_ttr": type_count / n,
        })


def add_tokenizer_features(rows: dict[int, dict[str, Any]], tokenizer_path: str) -> dict[str, float]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
    freq: Counter[int] = Counter()
    encoded: dict[int, list[int]] = {}
    for eid, rec in rows.items():
        ids = tokenizer.encode(rec["text"], add_special_tokens=False)
        encoded[eid] = ids
        freq.update(ids)
    counts = np.array(list(freq.values()), dtype=float) if freq else np.array([0.0])
    p50 = float(np.percentile(counts, 50))
    p95 = float(np.percentile(counts, 95))
    for eid, rec in rows.items():
        ids = encoded[eid]
        tc = max(len(ids), 1)
        mod = sum(1 for tid in ids if p50 <= freq.get(tid, 0) <= p95)
        bigram_unique = len(set(zip(ids[:-1], ids[1:]))) if len(ids) >= 2 else 0
        bigram_total = max(len(ids) - 1, 1)
        rec.update({
            "token_count": len(ids),
            "tokens_per_word_row": len(ids) / max(rec["words"], 1),
            "token_ttr_row": len(set(ids)) / tc,
            "moderate_token_ratio_union": mod / tc,
            "bigram_diversity_token": bigram_unique / bigram_total,
        })
    return {"union_unique_tokens": float(len(freq)), "union_token_freq_p50": p50, "union_token_freq_p95": p95}


def design_matrix(records: list[dict[str, Any]], feature_names: list[str], include_source: bool = True) -> np.ndarray:
    cols: list[np.ndarray] = []
    if include_source:
        for src in TRAIN_FILES[1:]:  # first source absorbed by intercept
            cols.append(np.array([1.0 if r["source"] == src else 0.0 for r in records], dtype=float))
    for name in feature_names:
        arr = np.array([float(r.get(name, 0.0)) for r in records], dtype=float)
        sd = float(arr.std())
        if sd > 1e-12:
            arr = (arr - float(arr.mean())) / sd
        else:
            arr = arr * 0.0
        cols.append(arr)
    if not cols:
        return np.ones((len(records), 1), dtype=float)
    x = np.column_stack(cols)
    intercept = np.ones((len(records), 1), dtype=float)
    return np.concatenate([intercept, x], axis=1)


def fit_residual(records: list[dict[str, Any]], y_name: str, feature_names: list[str], include_source: bool = True) -> tuple[np.ndarray, float]:
    y = np.array([float(r[y_name]) for r in records], dtype=float)
    x = design_matrix(records, feature_names, include_source=include_source)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    resid = y - pred
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return resid, round(float(r2), 6)


def summarize_by_arm(rows: dict[int, dict[str, Any]], arm_ids: dict[str, set[int]], feature_names: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for arm, ids in arm_ids.items():
        recs = [rows[i] for i in ids]
        out[arm] = {
            "rows": len(recs),
            "words": sum(int(r["words"]) for r in recs),
            "source_rows": dict(Counter(r["source"] for r in recs)),
            "features": {name: stats([float(r.get(name, 0.0)) for r in recs]) for name in feature_names},
        }
    return out


def pair_overlap(arm_ids: dict[str, set[int]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    arms = list(arm_ids)
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            shared = len(arm_ids[a] & arm_ids[b])
            denom_a = max(len(arm_ids[a]), 1)
            denom_b = max(len(arm_ids[b]), 1)
            out[f"{a}__{b}"] = {
                "shared_ids": shared,
                "fraction_of_first": round(shared / denom_a, 6),
                "fraction_of_second": round(shared / denom_b, 6),
            }
    return out


def write_note(note_path: Path, result: dict[str, Any]) -> None:
    arm = result["arm_summary"]
    reg = result["regression_summary"]
    overlap = result["overlap"]
    means = {a: arm[a]["features"] for a in arm}
    ra_rb_c = (means["R-a"]["c_score"]["mean"] + means["R-b"]["c_score"]["mean"]) / 2.0
    ra_rb_s = (means["R-a"]["s_score"]["mean"] + means["R-b"]["s_score"]["mean"]) / 2.0
    lines = []
    lines.append("# research C/S 4M screen mechanism analysis\n")
    lines.append("This note analyzes the materialized JSONL arms in `data/cs_4m_screen/` before any H100 training. The purpose is to determine whether the selected arms separate compositional-constraint density from recoverable-signal density rather than mostly selecting cleaner, easier-to-tokenize, or internally different text.\n")
    lines.append("## Exact file-level results\n")
    lines.append(f"- Union of selected rows: {result['union_rows']} of 62,500 official 160-word rows.")
    lines.append(f"- All five arms contain 25,000 rows and 4,000,000 words: {result['all_arm_word_counts_ok']}.")
    lines.append(f"- C-high/S-high overlap: {overlap['C-high__S-high']['shared_ids']} shared rows = {overlap['C-high__S-high']['fraction_of_first']:.3f} of C-high.")
    lines.append(f"- C-high/C-control overlap: {overlap['C-high__C-control']['shared_ids']} rows.")
    lines.append(f"- R-a/R-b overlap: {overlap['R-a__R-b']['shared_ids']} rows = {overlap['R-a__R-b']['fraction_of_first']:.3f} of R-a, near the expected 0.4 for independent 40% samples.\n")
    lines.append("## Score movement\n")
    lines.append(f"- Random-reference mean C-score: {ra_rb_c:.6f}; C-high mean C-score: {means['C-high']['c_score']['mean']:.6f}; C-control mean C-score: {means['C-control']['c_score']['mean']:.6f}; S-high mean C-score: {means['S-high']['c_score']['mean']:.6f}.")
    lines.append(f"- Random-reference mean S-score: {ra_rb_s:.6f}; S-high mean S-score: {means['S-high']['s_score']['mean']:.6f}; C-high mean S-score: {means['C-high']['s_score']['mean']:.6f}; C-control mean S-score: {means['C-control']['s_score']['mean']:.6f}.")
    lines.append(f"- Row-level C/S correlation on the selected-row union: {result['score_correlations']['union_c_s']:.3f}. By source: {json.dumps(result['score_correlations']['by_source'], ensure_ascii=False)}.\n")
    lines.append("## Surface, style, and tokenizer-compression coupling\n")
    lines.append(f"- A source+surface+tokenizer-compression linear model explains R²={reg['surface_style_compression_predicts_c_score_r2']:.3f} of C-score and R²={reg['surface_style_compression_predicts_s_score_r2']:.3f} of S-score on the selected-row union.")
    lines.append(f"- After removing source, row position, style/quality, and compression features, C-high residual C-score mean is {means['C-high']['c_residual_surface']['mean']:.6f} vs random mean {(means['R-a']['c_residual_surface']['mean'] + means['R-b']['c_residual_surface']['mean']) / 2.0:.6f}; S-high residual S-score mean is {means['S-high']['s_residual_surface']['mean']:.6f} vs random mean {(means['R-a']['s_residual_surface']['mean'] + means['R-b']['s_residual_surface']['mean']) / 2.0:.6f}.")
    lines.append(f"- After additionally removing the other intended score, S-high residual S-score mean is {means['S-high']['s_residual_surface_plus_c']['mean']:.6f}; C-high residual C-score mean after removing S-score is {means['C-high']['c_residual_surface_plus_s']['mean']:.6f}.\n")
    lines.append("## Interpretation for the next run\n")
    lines.append("The JSONL files are structurally usable by the trainer, but this arm set is not yet a clean mechanism experiment. C-high and S-high share more than two thirds of their rows, S-high still has a high C-score, and both scores are substantially predictable from source-internal position, style/quality, and tokenizer-compression proxies. A positive training result from these exact arms would not identify whether the useful factor was compositional structure, recoverable lexical signal, cleaner text, tokenization ease, or an internal genre slice.\n")
    lines.append("Do not send these arms to H100 as the first decisive screen. Treat them as a profile of the official pool. The next construction should use the saved row-level union features here and, preferably, a full-pool feature table to build residualized or nearest-neighbor matched arms: C-high with a confound-matched lower-C reference, S-high selected on non-structural recoverable signal after matching C-score and tokenizer compression, and a C/S-separated arm with low overlap by construction.\n")
    lines.append(f"\nDetailed JSON: `{result['json_path']}`\n")
    lines.append(f"Row-level feature CSV: `{result['csv_path']}`\n")
    note_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base = Path("experiments/archive/compact_experience/data/cs_4m_screen")
    out_json = base / "mechanism_analysis.json"
    out_csv = base / "row_feature_union.csv"
    note = Path("research/notes/compact_experience/cs_4m_screen_mechanism_analysis.md")
    tokenizer_path = (
        "experiments/archive/initial_model_studies/training/runs"
        "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model"
    )

    rows, arm_ids, consistency = read_arms(base)
    add_basic_features(rows)
    tok_summary = add_tokenizer_features(rows, tokenizer_path)

    records = [rows[eid] for eid in sorted(rows)]
    surface_features = [
        "source_pos",
        "char_per_word",
        "alpha_word_rate",
        "noise_word_rate",
        "digit_word_rate",
        "punct_only_rate",
        "filler_rate",
        "pronoun_rate",
        "function_word_rate",
        "dialogue_marker_rate_row",
        "upper_word_rate",
        "sent_end_per_word",
        "quote_char_per_word",
        "word_ttr",
        "tokens_per_word_row",
        "token_ttr_row",
        "moderate_token_ratio_union",
        "bigram_diversity_token",
    ]

    c_res_surface, c_r2 = fit_residual(records, "c_score", surface_features, include_source=True)
    s_res_surface, s_r2 = fit_residual(records, "s_score", surface_features, include_source=True)
    s_res_surface_c, s_r2_plus_c = fit_residual(records, "s_score", surface_features + ["c_score"], include_source=True)
    c_res_surface_s, c_r2_plus_s = fit_residual(records, "c_score", surface_features + ["s_score"], include_source=True)
    for rec, cr, sr, src, crs in zip(records, c_res_surface, s_res_surface, s_res_surface_c, c_res_surface_s):
        rec["c_residual_surface"] = float(cr)
        rec["s_residual_surface"] = float(sr)
        rec["s_residual_surface_plus_c"] = float(src)
        rec["c_residual_surface_plus_s"] = float(crs)

    feature_names_for_summary = [
        "c_score", "s_score",
        "c_residual_surface", "s_residual_surface",
        "s_residual_surface_plus_c", "c_residual_surface_plus_s",
        *surface_features,
    ]
    arm_summary = summarize_by_arm(rows, arm_ids, feature_names_for_summary)

    by_source_corr = {}
    for src in TRAIN_FILES:
        src_rows = [r for r in records if r["source"] == src]
        by_source_corr[src] = safe_corr([r["c_score"] for r in src_rows], [r["s_score"] for r in src_rows])

    all_arm_word_counts_ok = all(sum(rows[i]["words"] for i in ids) == 4_000_000 for ids in arm_ids.values())
    all_arm_rows_ok = all(len(ids) == 25_000 for ids in arm_ids.values())

    overlap = pair_overlap(arm_ids)
    result = {
        "status": "ok",
        "input_dir": str(base),
        "json_path": str(out_json),
        "csv_path": str(out_csv),
        "union_rows": len(rows),
        "all_arm_word_counts_ok": all_arm_word_counts_ok,
        "all_arm_rows_ok": all_arm_rows_ok,
        "consistency_messages": consistency,
        "overlap": overlap,
        "tokenizer_summary_union": tok_summary,
        "score_correlations": {
            "union_c_s": safe_corr([r["c_score"] for r in records], [r["s_score"] for r in records]),
            "by_source": by_source_corr,
        },
        "regression_summary": {
            "surface_features": surface_features,
            "surface_style_compression_predicts_c_score_r2": c_r2,
            "surface_style_compression_predicts_s_score_r2": s_r2,
            "surface_plus_c_predicts_s_score_r2": s_r2_plus_c,
            "surface_plus_s_predicts_c_score_r2": c_r2_plus_s,
        },
        "arm_summary": arm_summary,
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "example_id", "source", "source_pos", "words", "c_score", "s_score",
            "c_residual_surface", "s_residual_surface", "s_residual_surface_plus_c", "c_residual_surface_plus_s",
            *surface_features,
            *[f"in_{arm.replace('-', '_').lower()}" for arm in ARM_FILES],
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for eid in sorted(rows):
            rec = rows[eid]
            row = {k: rec.get(k, "") for k in fields}
            for arm in ARM_FILES:
                row[f"in_{arm.replace('-', '_').lower()}"] = 1 if eid in arm_ids[arm] else 0
            writer.writerow(row)

    write_note(note, result)
    print(json.dumps({
        "status": "ok",
        "json": str(out_json),
        "csv": str(out_csv),
        "note": str(note),
        "union_rows": len(rows),
        "c_s_overlap": overlap["C-high__S-high"],
        "union_c_s_corr": result["score_correlations"]["union_c_s"],
        "surface_r2_c": c_r2,
        "surface_r2_s": s_r2,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
