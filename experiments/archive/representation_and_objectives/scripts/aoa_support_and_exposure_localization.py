#!/usr/bin/env python3
"""research: passive localization of compact-view AoA failure.

Reads existing full/AoA artifacts and already materialized 10M corpora.  This is
post-hoc mechanism analysis only: it must not be used as a training-side selector
or mask over official AoA target words.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import csv
import json
import math
import pathlib
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
STUDY = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
STRICT_ROOT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
sys.path.insert(0, str(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')))
from evaluation_pipeline.utils import AoAEvaluator  # noqa: E402

OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/compact_aoa_localization')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/compact_aoa_localization/compact_aoa_support_and_exposure_localization.json')
OUT_NOTE = _public_path('research/notes/representation_and_objectives/compact_aoa_support_and_exposure_localization.md')

A02_AOA_ANALYSIS = _public_path('experiments/archive/frontier_consolidation/data/aoa_mechanism_analysis/aoa_mechanism_analysis.json')
A02_FULL_CORE = _public_path('experiments/archive/frontier_consolidation/data/density_full_eval/density_full_eval_summary.json')
COMPACT_EXPERIENCE_FULL = _public_path('experiments/archive/compact_experience/data/full_eval/full_eval_summary.json')
META = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json')

CLEAN_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
CORE_POOL = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_core_neutral_10M.jsonl')
REINVEST_POOL = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
HELDOUT_ROWS = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl')
CDI_HUMAN = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')

TOKEN_RE = re.compile(r"[a-z]+")


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(_public_path('.')))
    except Exception:
        return str(path)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def age_bin(month: float) -> str:
    if month <= 20.0:
        return "early_le20"
    if month <= 25.0:
        return "middle_20to25"
    return "late_gt25"


def basic_stats(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in vals if v is not None and math.isfinite(float(v)))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {
        "n": len(xs), "min": xs[0], "p05": q(0.05), "mean": sum(xs) / len(xs),
        "median": q(0.5), "p95": q(0.95), "max": xs[-1],
    }


def corr(xs: List[float], ys: List[float]) -> Dict[str, Any]:
    if len(xs) < 3:
        return {"n": len(xs), "pearson": None, "pearson_p": None, "spearman": None, "spearman_p": None}
    pr, pp = pearsonr(xs, ys)
    sr, sp = spearmanr(xs, ys)
    return {"n": len(xs), "pearson": float(pr), "pearson_p": float(pp), "spearman": float(sr), "spearman_p": float(sp)}


def bin_counts(words: Iterable[str], child_aoa: Dict[str, float]) -> Dict[str, int]:
    out: Dict[str, int] = collections.Counter()
    for w in words:
        if w in child_aoa:
            out[age_bin(child_aoa[w])] += 1
        else:
            out["missing_child_aoa"] += 1
    return dict(out)


def load_child_aoa() -> Dict[str, float]:
    evaluator = AoAEvaluator(CDI_HUMAN)
    out: Dict[str, float] = {}
    for idx, row in evaluator.cdi_data.iterrows():
        word = str(row["word"]).lower()
        val = evaluator.compute_child_aoa(idx)
        if val is not None:
            out[word] = float(val)
    return out


def model_corr_for_words(model: Dict[str, Any], words: Iterable[str]) -> Dict[str, Any]:
    per = model.get("per_word") or {}
    xs: List[float] = []
    ys: List[float] = []
    used: List[str] = []
    for w in sorted(words):
        if w not in per:
            continue
        rec = per[w]
        xs.append(float(rec["model_aoa_log10_words"]))
        ys.append(float(rec["child_aoa_month"]))
        used.append(w)
    out = corr(xs, ys)
    out["words"] = used
    if xs:
        out["model_aoa_stats"] = basic_stats(xs)
        by_bin: Dict[str, List[float]] = collections.defaultdict(list)
        for w, ma, ca in zip(used, xs, ys):
            by_bin[age_bin(ca)].append(ma)
        out["model_aoa_by_child_age_bin"] = {k: basic_stats(v) for k, v in by_bin.items()}
        if by_bin.get("early_le20") and by_bin.get("late_gt25"):
            out["late_minus_early_model_aoa_log10"] = float(np.mean(by_bin["late_gt25"]) - np.mean(by_bin["early_le20"]))
    return out


def support_delta(clean: Dict[str, Any], compact: Dict[str, Any], child_aoa: Dict[str, float]) -> Dict[str, Any]:
    clean_words = set((clean.get("per_word") or {}).keys())
    compact_words = set((compact.get("per_word") or {}).keys())
    common = clean_words & compact_words
    only_clean = clean_words - compact_words
    only_compact = compact_words - clean_words

    def word_rec(model: Dict[str, Any], w: str) -> Dict[str, Any]:
        rec = model["per_word"][w]
        return {
            "word": w,
            "child_aoa_month": float(rec["child_aoa_month"]),
            "child_bin": age_bin(float(rec["child_aoa_month"])),
            "model_aoa_log10_words": float(rec["model_aoa_log10_words"]),
        }

    def extreme_for_set(model: Dict[str, Any], words: Iterable[str], reverse: bool) -> List[Dict[str, Any]]:
        rows = [word_rec(model, w) for w in words]
        rows.sort(key=lambda r: (r["child_aoa_month"], r["word"]), reverse=reverse)
        return rows[:25]

    # compact full support can be decomposed into common + only-compact fitted words.
    comp_common = model_corr_for_words(compact, common)
    comp_only = model_corr_for_words(compact, only_compact)
    clean_common = model_corr_for_words(clean, common)
    clean_only = model_corr_for_words(clean, only_clean)

    # Same words, compact minus clean model-AoA shift.
    deltas: List[Tuple[str, float, float, float, float]] = []
    for w in sorted(common):
        ca = float(compact["per_word"][w]["child_aoa_month"])
        a = float(clean["per_word"][w]["model_aoa_log10_words"])
        b = float(compact["per_word"][w]["model_aoa_log10_words"])
        deltas.append((w, ca, a, b, b - a))
    dvals = [r[4] for r in deltas]
    child_vals = [r[1] for r in deltas]
    by_bin: Dict[str, List[float]] = collections.defaultdict(list)
    for _, ca, _, _, d in deltas:
        by_bin[age_bin(ca)].append(d)

    def pack_delta(row: Tuple[str, float, float, float, float]) -> Dict[str, Any]:
        w, ca, a, b, d = row
        return {"word": w, "child_aoa_month": ca, "child_bin": age_bin(ca), "clean_model_aoa": a, "compact_model_aoa": b, "compact_minus_clean": d}

    return {
        "clean_n": len(clean_words),
        "compact_n": len(compact_words),
        "common_n": len(common),
        "only_clean_n": len(only_clean),
        "only_compact_n": len(only_compact),
        "clean_bins": bin_counts(clean_words, child_aoa),
        "compact_bins": bin_counts(compact_words, child_aoa),
        "common_bins": bin_counts(common, child_aoa),
        "only_clean_bins": bin_counts(only_clean, child_aoa),
        "only_compact_bins": bin_counts(only_compact, child_aoa),
        "clean_official_support": {
            "pearson": clean.get("pearson_model_aoa_vs_child_aoa"),
            "pearson_p": clean.get("pearson_p_value"),
            "official_thresholded_leaderboard_score": clean.get("official_thresholded_leaderboard_score"),
        },
        "compact_official_support": {
            "pearson": compact.get("pearson_model_aoa_vs_child_aoa"),
            "pearson_p": compact.get("pearson_p_value"),
            "official_thresholded_leaderboard_score": compact.get("official_thresholded_leaderboard_score"),
        },
        "clean_on_common_words": clean_common,
        "compact_on_common_words": comp_common,
        "clean_only_words": clean_only,
        "compact_only_words": comp_only,
        "compact_minus_clean_model_aoa_delta_on_common": {
            "delta_stats": basic_stats(dvals),
            "child_aoa_vs_delta": corr(child_vals, dvals),
            "delta_by_child_bin": {k: basic_stats(v) for k, v in by_bin.items()},
            "largest_compact_advances": [pack_delta(r) for r in sorted(deltas, key=lambda x: x[4])[:30]],
            "largest_compact_delays": [pack_delta(r) for r in sorted(deltas, key=lambda x: x[4], reverse=True)[:30]],
        },
        "only_compact_child_latest": extreme_for_set(compact, only_compact, reverse=True),
        "only_compact_child_earliest": extreme_for_set(compact, only_compact, reverse=False),
        "only_clean_child_latest": extreme_for_set(clean, only_clean, reverse=True),
        "only_clean_child_earliest": extreme_for_set(clean, only_clean, reverse=False),
    }


def count_file_target_words(path: pathlib.Path, target_words: set[str], child_aoa: Dict[str, float], max_lines: Optional[int] = None) -> Dict[str, Any]:
    counts: Dict[str, int] = collections.Counter()
    by_bin: Dict[str, int] = collections.Counter()
    by_source: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.Counter())
    total_words = 0
    rows = 0
    parse_errors = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_lines is not None and rows >= max_lines:
                break
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                parse_errors += 1
                continue
            rows += 1
            text = str(rec.get("text", ""))
            src = str(rec.get("source", "unknown"))
            wcount = int(rec.get("words") or len(text.split()))
            total_words += wcount
            toks = TOKEN_RE.findall(text.lower())
            for tok in toks:
                if tok in target_words:
                    counts[tok] += 1
                    b = age_bin(child_aoa[tok])
                    by_bin[b] += 1
                    by_source[src][b] += 1
                    by_source[src]["total"] += 1
    return {
        "path": rel(path),
        "rows": rows,
        "declared_words": total_words,
        "parse_errors": parse_errors,
        "target_occurrences_total": int(sum(counts.values())),
        "target_occurrences_by_bin": dict(by_bin),
        "target_occurrences_per_million_words_by_bin": {k: (v / total_words * 1_000_000.0 if total_words else None) for k, v in by_bin.items()},
        "target_occurrences_per_million_words_total": (sum(counts.values()) / total_words * 1_000_000.0 if total_words else None),
        "by_source_top": dict(sorted(((src, dict(c)) for src, c in by_source.items()), key=lambda kv: kv[1].get("total", 0), reverse=True)[:30]),
        "per_word_counts": dict(counts),
    }


def diff_counts(after: Dict[str, Any], before: Dict[str, Any], child_aoa: Dict[str, float]) -> Dict[str, Any]:
    ca = collections.Counter(after.get("per_word_counts") or {})
    cb = collections.Counter(before.get("per_word_counts") or {})
    words = sorted(set(ca) | set(cb))
    rows = []
    by_bin: Dict[str, int] = collections.Counter()
    for w in words:
        d = int(ca[w] - cb[w])
        if d == 0:
            continue
        b = age_bin(child_aoa[w]) if w in child_aoa else "missing"
        rows.append({"word": w, "child_aoa_month": child_aoa.get(w), "child_bin": b, "before": int(cb[w]), "after": int(ca[w]), "delta": d})
        by_bin[b] += d
    rows_by_delta = sorted(rows, key=lambda r: (r["delta"], r["word"]))
    total_delta = int(sum(r["delta"] for r in rows))
    return {
        "before": before.get("path"),
        "after": after.get("path"),
        "word_delta_total": total_delta,
        "bin_delta": dict(by_bin),
        "total_per_million_delta": (after.get("target_occurrences_per_million_words_total") or 0.0) - (before.get("target_occurrences_per_million_words_total") or 0.0),
        "bin_per_million_delta": {k: (after.get("target_occurrences_per_million_words_by_bin", {}).get(k, 0.0) - before.get("target_occurrences_per_million_words_by_bin", {}).get(k, 0.0)) for k in sorted(set(after.get("target_occurrences_per_million_words_by_bin", {})) | set(before.get("target_occurrences_per_million_words_by_bin", {})))},
        "largest_losses": rows_by_delta[:40],
        "largest_gains": list(reversed(rows_by_delta[-40:])),
    }


def summarize_payload(payload: Dict[str, Any]) -> str:
    full = payload["compact_core_full_result"]
    support = payload["support_localization"]
    exposure = payload["corpus_exposure"]
    lines: List[str] = []
    lines.append("# research compact-view AoA support and corpus-exposure localization")
    lines.append("")
    lines.append("This is passive analysis of existing A02/COMPACT_EXPERIENCE AoA files and already materialized 10M corpora. It uses official AoA words only to interpret an observed failure, not to choose future training examples.")
    lines.append("")
    lines.append("## Compact-core full result now available")
    s = full["scores"]
    lines.append(f"A02 full official-compatible `compact_view_core` scored Overall {full['Overall']:.4f}: BLiMP {s['BLiMP']:.2f}, Supplement {s['Supplement']:.2f}, EWoK {s['EWoK']:.2f}, Entity {s['Entity']:.2f}, COMPS {s['COMPS']:.2f}, GlobalPIQA {s['GlobalPIQA']:.3f}, SuperGLUE {s['SuperGLUE']:.4f}, Reading {s['Reading']:.2f}, AoA {s['AoA']:.3f}.")
    d = full["delta_vs_clean_qwen"]
    lines.append(f"Against COMPACT_EXPERIENCE clean-Qwen, NLP_average is essentially tied ({d['NLP_average']:+.4f}) but Overall is {d['Overall']:+.4f}, dominated by AoA {d['AoA']:+.3f} and SuperGLUE {d['SuperGLUE']:+.3f}/GlobalPIQA {d['GlobalPIQA']:+.3f} losses.")
    lines.append("")
    lines.append("## What changed in the AoA score")
    lines.append(f"Clean support: n={support['clean_n']}, r={support['clean_official_support']['pearson']:.4f}, p={support['clean_official_support']['pearson_p']:.4g}, official leaderboard AoA={support['clean_official_support']['official_thresholded_leaderboard_score']:.3f}.")
    lines.append(f"Compact support: n={support['compact_n']}, r={support['compact_official_support']['pearson']:.4f}, p={support['compact_official_support']['pearson_p']:.4g}, official leaderboard AoA={support['compact_official_support']['official_thresholded_leaderboard_score']:.3f}.")
    lines.append(f"Same fitted words common to clean and compact: n={support['common_n']}. On those common words compact r={support['compact_on_common_words']['pearson']:.4f}, p={support['compact_on_common_words']['pearson_p']:.4g}; clean r={support['clean_on_common_words']['pearson']:.4f}, p={support['clean_on_common_words']['pearson_p']:.4g}.")
    cd = support['compact_minus_clean_model_aoa_delta_on_common']
    lines.append(f"On common words, compact shifts fitted model-AoA earlier by mean {cd['delta_stats']['mean']:.4f} log10 words, but child-AoA vs shift is r={cd['child_aoa_vs_delta']['pearson']:.4f}, p={cd['child_aoa_vs_delta']['pearson_p']:.4g}; this does not support a clean same-word developmental inversion caused by compact views.")
    lines.append(f"The nonzero compact official AoA partly comes from fit-support composition: compact has {support['only_compact_n']} fitted words not fitted in clean, clean has {support['only_clean_n']} words not fitted in compact. Compact-only bins: {support['only_compact_bins']}; clean-only bins: {support['only_clean_bins']}.")
    if support.get('compact_only_words', {}).get('n', 0) >= 3:
        ow = support['compact_only_words']
        lines.append(f"Among compact-only fitted words, r={ow['pearson']:.4f}, p={ow['pearson_p']:.4g}; this small support should be read as a sensitivity clue rather than a robust mechanism.")
    lines.append("")
    lines.append("## Training-corpus AoA-word exposure")
    ch = exposure['changed_block_core_minus_heldout_cleanqwen']
    rh = exposure['changed_block_reinvest_minus_heldout_cleanqwen']
    lines.append(f"Core replacement block vs the held-out clean-Qwen slice: target-word occurrence delta {ch['word_delta_total']:+d}; bin deltas {ch['bin_delta']}; per-million deltas {ch['bin_per_million_delta']}.")
    lines.append(f"Reinvest replacement block vs the same held-out slice: target-word occurrence delta {rh['word_delta_total']:+d}; bin deltas {rh['bin_delta']}; per-million deltas {rh['bin_per_million_delta']}.")
    pc = exposure['pool_core_minus_clean_qwen']
    pr = exposure['pool_reinvest_minus_clean_qwen']
    lines.append(f"At full 10M-pool scale, compact-core vs clean-Qwen changes AoA target occurrences by {pc['word_delta_total']:+d} total, bin deltas {pc['bin_delta']}; reinvest vs clean-Qwen changes {pr['word_delta_total']:+d}, bin deltas {pr['bin_delta']}.")
    lines.append("")
    lines.append("## Scientific reading for the next action")
    lines.append("The compact-density intervention is not simply shifting the same AoA words into an anti-child order. On harmonized word support, compact and clean are both statistically nonzero only after thresholding, and the compact-minus-clean per-word timing shifts are not child-AoA structured. The dangerous part is that compact FineWeb replacement changes which words yield valid fitted acquisition curves and it substitutes a small but vocabulary-visible slice of the clean-Qwen developmental substrate. Because the full `compact_view_core` endpoint loses 1.36 Overall despite tied NLP_average, no new 100M compact-dose or test-shaped patch should be launched before the pending reinvest endpoint is actually seen. If reinvest inherits negative AoA, the compact FineWeb overlay should be treated as a mechanism probe rather than the SOTA route; a better repair would need to preserve the clean-Qwen AoA-neutral substrate while testing compact density inside that substrate or via a schedule/content form justified independently of the AoA word list.")
    lines.append("")
    lines.append(f"Machine-readable JSON: `{rel(OUT_JSON)}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _public_path('research/notes/representation_and_objectives').mkdir(parents=True, exist_ok=True)
    child_aoa = load_child_aoa()
    # Restrict corpus counts to single orthographic tokens so regex token counting is unambiguous.
    target_words = {w for w in child_aoa if TOKEN_RE.fullmatch(w)}

    aoa = read_json(A02_AOA_ANALYSIS)
    full_core = read_json(A02_FULL_CORE)["targets"]["compact_view_core"]["official_overall"]
    clean_full = read_json(COMPACT_EXPERIENCE_FULL)["targets"]["qwen_clean_aligned"]
    meta = read_json(META)
    clean_model = aoa["models"]["clean_qwen_seed43022"]
    compact_model = aoa["models"]["density_compact_view_core"]

    scores = full_core["scores"]
    clean_scores = {k: clean_full[k] for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA", "Overall", "NLP_average", "Human_like_average"]}
    delta_vs_clean: Dict[str, float] = {}
    for k in clean_scores:
        if k in scores:
            delta_vs_clean[k] = float(scores[k]) - float(clean_scores[k])
        elif k in full_core:
            delta_vs_clean[k] = float(full_core[k]) - float(clean_scores[k])
    delta_vs_clean["Overall"] = float(full_core["Overall"]) - float(clean_full["Overall"])
    delta_vs_clean["NLP_average"] = float(full_core["NLP_average"]) - float(clean_full["NLP_average"])
    delta_vs_clean["Human_like_average"] = float(full_core["Human_like_average"]) - float(clean_full["Human_like_average"])

    support = support_delta(clean_model, compact_model, child_aoa)

    core_changed_rows = int(meta["families"]["compact_core_neutral"]["changed_block_rows"])
    reinvest_changed_rows = int(meta["families"]["compact_reinvest"]["changed_block_rows"])
    clean_pool_counts = count_file_target_words(CLEAN_POOL, target_words, child_aoa)
    core_pool_counts = count_file_target_words(CORE_POOL, target_words, child_aoa)
    reinvest_pool_counts = count_file_target_words(REINVEST_POOL, target_words, child_aoa)
    heldout_counts = count_file_target_words(HELDOUT_ROWS, target_words, child_aoa)
    core_changed_counts = count_file_target_words(CORE_POOL, target_words, child_aoa, max_lines=core_changed_rows)
    reinvest_changed_counts = count_file_target_words(REINVEST_POOL, target_words, child_aoa, max_lines=reinvest_changed_rows)

    payload: Dict[str, Any] = {
        "status": "COMPACT_AOA_SUPPORT_AND_EXPOSURE_LOCALIZATION",
        "purpose": "Passive localization of compact-view AoA failure using existing full-eval/AoA files and materialized corpora; not a training selector.",
        "inputs": {
            "a02_full_core_summary": rel(A02_FULL_CORE),
            "a02_aoa_analysis": rel(A02_AOA_ANALYSIS),
            "compact_experience_full_summary": rel(COMPACT_EXPERIENCE_FULL),
            "metadata": rel(META),
            "clean_pool": rel(CLEAN_POOL),
            "core_pool": rel(CORE_POOL),
            "reinvest_pool": rel(REINVEST_POOL),
            "heldout_rows": rel(HELDOUT_ROWS),
            "cdi_human": rel(CDI_HUMAN),
        },
        "target_word_counting_scope": {
            "child_aoa_words_with_valid_child_fit": len(child_aoa),
            "single_orthographic_token_targets_counted": len(target_words),
            "note": "Official AoA targets are used only for post-hoc interpretation of existing results, not for corpus construction.",
        },
        "compact_core_full_result": {
            "scores": scores,
            "Overall": full_core["Overall"],
            "NLP_average": full_core["NLP_average"],
            "Human_like_average": full_core["Human_like_average"],
            "delta_vs_clean_qwen": delta_vs_clean,
        },
        "support_localization": support,
        "corpus_exposure": {
            "clean_qwen_pool": {k: v for k, v in clean_pool_counts.items() if k != "per_word_counts"},
            "compact_core_pool": {k: v for k, v in core_pool_counts.items() if k != "per_word_counts"},
            "compact_reinvest_pool": {k: v for k, v in reinvest_pool_counts.items() if k != "per_word_counts"},
            "heldout_cleanqwen_slice": {k: v for k, v in heldout_counts.items() if k != "per_word_counts"},
            "compact_core_changed_block": {k: v for k, v in core_changed_counts.items() if k != "per_word_counts"},
            "compact_reinvest_changed_block": {k: v for k, v in reinvest_changed_counts.items() if k != "per_word_counts"},
            "changed_block_core_minus_heldout_cleanqwen": diff_counts(core_changed_counts, heldout_counts, child_aoa),
            "changed_block_reinvest_minus_heldout_cleanqwen": diff_counts(reinvest_changed_counts, heldout_counts, child_aoa),
            "pool_core_minus_clean_qwen": diff_counts(core_pool_counts, clean_pool_counts, child_aoa),
            "pool_reinvest_minus_clean_qwen": diff_counts(reinvest_pool_counts, clean_pool_counts, child_aoa),
        },
        "out_json": rel(OUT_JSON),
        "out_note": rel(OUT_NOTE),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_NOTE.write_text(summarize_payload(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": rel(OUT_JSON),
        "out_note": rel(OUT_NOTE),
        "compact_core_overall": full_core["Overall"],
        "compact_core_aoa": scores["AoA"],
        "common_words": support["common_n"],
        "only_compact_words": support["only_compact_n"],
        "pool_core_target_delta": payload["corpus_exposure"]["pool_core_minus_clean_qwen"]["word_delta_total"],
        "pool_reinvest_target_delta": payload["corpus_exposure"]["pool_reinvest_minus_clean_qwen"]["word_delta_total"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
