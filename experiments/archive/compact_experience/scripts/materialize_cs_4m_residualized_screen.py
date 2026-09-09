#!/usr/bin/env python3
"""Materialize residualized C/S 4M arms for a cleaner mechanism screen.

This repairs the first raw C/S materialization by separating intended mechanisms
from confounds before any H100 training:
  * C-unique: compositional structure density after removing source, coarse
    source position, surface/style/quality, tokenizer compression, and S-score.
  * S-unique: recoverable-signal density after removing the same confounds and
    C-score.

Selection is made within coarse confound cells so every high/control arm has the
same source x position x compression x quality-bin quota. The script writes all
features and validation summaries so the resulting arms can be judged before
training.
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
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = _public_path('.')
MAT_PATH = _public_path('experiments/archive/compact_experience/scripts/materialize_cs_4m_screen.py')
ANA_PATH = _public_path('experiments/archive/compact_experience/scripts/analyze_cs_4m_screen_mechanism.py')


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


mat = import_module(MAT_PATH, "compact_experience_materializer_base")
ana = import_module(ANA_PATH, "compact_experience_mechanism_analyzer")

STRICT_CONFOUNDS = [
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
]
OVERCONTROL_FEATURES = STRICT_CONFOUNDS + [
    "token_ttr_row",
    "moderate_token_ratio_union",
    "bigram_diversity_token",
]
SUMMARY_FEATURES = [
    "c_score",
    "s_score",
    "c_unique_resid",
    "s_unique_resid",
    "c_style_resid",
    "s_style_resid",
    *OVERCONTROL_FEATURES,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw_dir",
        default="experiments/archive/initial_model_studies/training/runs"
        "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/raw_dataset",
    )
    p.add_argument(
        "--tokenizer_path",
        default="experiments/archive/initial_model_studies/training/runs"
        "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model",
    )
    p.add_argument("--out_dir", default="experiments/archive/compact_experience/data/cs_4m_residualized_screen")
    p.add_argument("--max_pool_words", type=int, default=10_000_000)
    p.add_argument("--words_per_example", type=int, default=160)
    p.add_argument("--target_words", type=int, default=4_000_000)
    p.add_argument("--selection_fraction", type=float, default=0.4)
    p.add_argument("--spacy_batch", type=int, default=256)
    p.add_argument("--seed_ra", type=int, default=3100)
    p.add_argument("--seed_rb", type=int, default=3200)
    p.add_argument("--pos_bins", type=int, default=5)
    p.add_argument("--tpw_bins", type=int, default=5)
    p.add_argument("--quality_bins", type=int, default=2)
    return p.parse_args()


def fit_pred_resid(records: list[dict[str, Any]], y_name: str, feature_names: list[str], include_source: bool = True):
    y = np.array([float(r[y_name]) for r in records], dtype=float)
    x = ana.design_matrix(records, feature_names, include_source=include_source)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    resid = y - pred
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return pred, resid, round(float(r2), 6)


def add_quantile_bins(records: list[dict[str, Any]], feature: str, bin_name: str, n_bins: int) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)
    for src, recs in by_source.items():
        vals = np.array([float(r[feature]) for r in recs], dtype=float)
        if len(set(vals.tolist())) <= 1:
            for r in recs:
                r[bin_name] = 0
            continue
        qs = [100 * k / n_bins for k in range(1, n_bins)]
        edges = np.percentile(vals, qs)
        for r in recs:
            r[bin_name] = int(np.searchsorted(edges, float(r[feature]), side="right"))


def assign_cells(records: list[dict[str, Any]], pos_bins: int, tpw_bins: int, quality_bins: int) -> None:
    # source_pos is already [0,1] within the original source stream.
    for r in records:
        r["pos_bin"] = min(pos_bins - 1, max(0, int(float(r["source_pos"]) * pos_bins)))
        r["quality_index"] = (
            float(r["noise_word_rate"])
            + float(r["filler_rate"])
            + float(r["dialogue_marker_rate_row"])
            + float(r["upper_word_rate"])
        )
    add_quantile_bins(records, "tokens_per_word_row", "tpw_bin", tpw_bins)
    add_quantile_bins(records, "quality_index", "quality_bin", quality_bins)
    for r in records:
        r["confound_cell"] = f"{r['source']}|p{r['pos_bin']}|t{r['tpw_bin']}|q{r['quality_bin']}"


def allocate_cell_quotas(records: list[dict[str, Any]], fraction: float) -> dict[str, int]:
    by_source_cell: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_source_cell[r["source"]][r["confound_cell"]].append(r)

    qmap: dict[str, int] = {}
    for src in mat.TRAIN_FILES:
        cells = by_source_cell.get(src, {})
        if not cells:
            continue
        n_src = sum(len(v) for v in cells.values())
        q_src = round(n_src * fraction)
        parts = []
        base_sum = 0
        cap_sum = 0
        for cell, recs in cells.items():
            n = len(recs)
            raw = n * fraction
            cap = n // 2  # ensures top/bottom high/control can be disjoint in this cell
            base = min(int(math.floor(raw)), cap)
            parts.append([cell, n, raw - math.floor(raw), cap, base])
            base_sum += base
            cap_sum += cap
        if q_src > cap_sum:
            raise RuntimeError(f"Source {src} target quota {q_src} exceeds disjoint cell capacity {cap_sum}")
        remaining = q_src - base_sum
        # Fill largest fractional parts first, then larger cells for deterministic stability.
        parts.sort(key=lambda x: (x[2], x[1], x[0]), reverse=True)
        idx = 0
        while remaining > 0:
            progressed = False
            for part in parts:
                if remaining <= 0:
                    break
                if part[4] < part[3]:
                    part[4] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                raise RuntimeError(f"Could not allocate remaining quota for {src}")
            idx += 1
            if idx > 10_000:
                raise RuntimeError("quota allocation loop runaway")
        for cell, _n, _frac, _cap, base in parts:
            qmap[cell] = int(base)
    return qmap


def select_by_cell(records_by_id: dict[int, dict[str, Any]], qmap: dict[str, int], score_name: str, high: bool) -> list[int]:
    cells: dict[str, list[int]] = defaultdict(list)
    for eid, r in records_by_id.items():
        cells[r["confound_cell"]].append(eid)
    selected: list[int] = []
    for cell, ids in cells.items():
        q = qmap.get(cell, 0)
        if q <= 0:
            continue
        ids_sorted = sorted(ids, key=lambda i: (float(records_by_id[i][score_name]), -i if high else i), reverse=high)
        chosen = ids_sorted[:q]
        selected.extend(chosen)
    return sorted(selected)


def select_random_by_cell(records_by_id: dict[int, dict[str, Any]], qmap: dict[str, int], seed: int) -> list[int]:
    rng = random.Random(seed)
    cells: dict[str, list[int]] = defaultdict(list)
    for eid, r in records_by_id.items():
        cells[r["confound_cell"]].append(eid)
    selected: list[int] = []
    for cell, ids in sorted(cells.items()):
        q = qmap.get(cell, 0)
        if q <= 0:
            continue
        if q > len(ids):
            raise RuntimeError(f"cell {cell} quota {q} > n {len(ids)}")
        selected.extend(rng.sample(ids, q))
    return sorted(selected)


def mean_feature(rows: list[dict[str, Any]], name: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(r.get(name, 0.0)) for r in rows]))


def std_feature(rows: list[dict[str, Any]], name: str) -> float:
    if not rows:
        return 1.0
    sd = float(np.std([float(r.get(name, 0.0)) for r in rows]))
    return sd if sd > 1e-12 else 1.0


def max_abs_smd(target: list[dict[str, Any]], ref: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    vals = {}
    max_name = None
    max_val = -1.0
    pool = target + ref
    for name in features:
        denom = std_feature(pool, name)
        smd = (mean_feature(target, name) - mean_feature(ref, name)) / denom
        vals[name] = round(float(smd), 6)
        if abs(smd) > max_val:
            max_val = abs(smd)
            max_name = name
    return {"max_abs_smd": round(float(max_val), 6), "max_feature": max_name, "smd_by_feature": vals}


def summarize_arm(records_by_id: dict[int, dict[str, Any]], ids: list[int], poolrows_by_id: dict[int, Any], tokenizer) -> dict[str, Any]:
    recs = [records_by_id[i] for i in ids]
    poolrows = [poolrows_by_id[i] for i in ids]
    base_stats = mat.compute_arm_stats(poolrows, tokenizer)
    return {
        **base_stats,
        "source_rows": dict(Counter(r["source"] for r in recs)),
        "cell_count": len(set(r["confound_cell"] for r in recs)),
        "features": {name: mat.score_stats([float(r.get(name, 0.0)) for r in recs]) for name in SUMMARY_FEATURES},
        "example_ids": ids,
        "text_multiset_hash": mat.text_multiset_hash(poolrows),
    }


def write_arm_jsonl(out_dir: Path, arm_name: str, ids: list[int], records_by_id: dict[int, dict[str, Any]]) -> Path:
    path = out_dir / f"{arm_name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for eid in ids:
            r = records_by_id[eid]
            obj = {
                "text": r["text"],
                "words": int(r["words"]),
                "example_id": int(eid),
                "source": r["source"],
                "kind": "official",
                "c_score": round(float(r["c_score"]), 6),
                "s_score": round(float(r["s_score"]), 6),
                "c_unique_resid": round(float(r["c_unique_resid"]), 6),
                "s_unique_resid": round(float(r["s_unique_resid"]), 6),
                "confound_cell": r["confound_cell"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return path


def pair_overlap(arms: dict[str, list[int]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    names = list(arms)
    sets = {k: set(v) for k, v in arms.items()}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = len(sets[a] & sets[b])
            out[f"{a}__{b}"] = {
                "shared_ids": shared,
                "fraction_of_first": round(shared / max(len(sets[a]), 1), 6),
                "fraction_of_second": round(shared / max(len(sets[b]), 1), 6),
            }
    return out


def main() -> None:
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    note_path = Path("research/notes/compact_experience/cs_4m_residualized_screen.md")

    print(json.dumps({"event": "building_pool", "raw_dir": args.raw_dir}), flush=True)
    pool = mat.build_pool(Path(args.raw_dir), args.max_pool_words, args.words_per_example)
    poolrows_by_id = {r.example_id: r for r in pool}
    print(json.dumps({"event": "pool_built", "rows": len(pool), "words": sum(r.words for r in pool), "elapsed_sec": round(time.time() - t0, 1)}), flush=True)

    print(json.dumps({"event": "loading_spacy"}), flush=True)
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print(json.dumps({"event": "scoring_c", "rows": len(pool)}), flush=True)
    tc = time.time()
    mat.score_c_batch(pool, nlp, batch_size=args.spacy_batch)
    print(json.dumps({"event": "c_scored", "elapsed_sec": round(time.time() - tc, 1), "mean": round(float(np.mean([r.c_score for r in pool])), 6)}), flush=True)

    print(json.dumps({"event": "loading_tokenizer"}), flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True, local_files_only=True)
    print(json.dumps({"event": "scoring_s"}), flush=True)
    ts = time.time()
    gf, p50, p95 = mat.build_global_unigram_freq(pool, tokenizer)
    mat.score_s_batch(pool, tokenizer, gf, p50, p95)
    print(json.dumps({"event": "s_scored", "elapsed_sec": round(time.time() - ts, 1), "mean": round(float(np.mean([r.s_score for r in pool])), 6), "p50": round(p50, 1), "p95": round(p95, 1)}), flush=True)

    records_by_id: dict[int, dict[str, Any]] = {}
    for r in pool:
        records_by_id[r.example_id] = {
            "example_id": r.example_id,
            "text": r.text,
            "words": r.words,
            "source": r.source,
            "c_score": float(r.c_score),
            "s_score": float(r.s_score),
        }
    ana.add_basic_features(records_by_id)
    # This adds full-pool token features with tokenizer p50/p95 over the same pool.
    tok_summary = ana.add_tokenizer_features(records_by_id, args.tokenizer_path)
    records = [records_by_id[i] for i in sorted(records_by_id)]

    print(json.dumps({"event": "fitting_residuals", "records": len(records)}), flush=True)
    c_pred_style, c_res_style, c_r2_style = fit_pred_resid(records, "c_score", STRICT_CONFOUNDS, include_source=True)
    s_pred_style, s_res_style, s_r2_style = fit_pred_resid(records, "s_score", STRICT_CONFOUNDS, include_source=True)
    c_pred_unique, c_res_unique, c_r2_unique = fit_pred_resid(records, "c_score", STRICT_CONFOUNDS + ["s_score"], include_source=True)
    s_pred_unique, s_res_unique, s_r2_unique = fit_pred_resid(records, "s_score", STRICT_CONFOUNDS + ["c_score"], include_source=True)
    c_pred_over, c_res_over, c_r2_over = fit_pred_resid(records, "c_score", OVERCONTROL_FEATURES + ["s_score"], include_source=True)
    s_pred_over, s_res_over, s_r2_over = fit_pred_resid(records, "s_score", OVERCONTROL_FEATURES + ["c_score"], include_source=True)

    for idx, r in enumerate(records):
        r["c_style_pred"] = float(c_pred_style[idx])
        r["s_style_pred"] = float(s_pred_style[idx])
        r["c_style_resid"] = float(c_res_style[idx])
        r["s_style_resid"] = float(s_res_style[idx])
        r["c_unique_pred"] = float(c_pred_unique[idx])
        r["s_unique_pred"] = float(s_pred_unique[idx])
        r["c_unique_resid"] = float(c_res_unique[idx])
        r["s_unique_resid"] = float(s_res_unique[idx])
        r["c_overcontrol_resid"] = float(c_res_over[idx])
        r["s_overcontrol_resid"] = float(s_res_over[idx])

    assign_cells(records, args.pos_bins, args.tpw_bins, args.quality_bins)
    qmap = allocate_cell_quotas(records, args.selection_fraction)
    print(json.dumps({"event": "cells_allocated", "cells": len(qmap), "quota_rows": sum(qmap.values())}), flush=True)
    if sum(qmap.values()) * args.words_per_example != args.target_words:
        raise RuntimeError(f"quota rows {sum(qmap.values())} gives words {sum(qmap.values()) * args.words_per_example}, target {args.target_words}")

    arms: dict[str, list[int]] = {
        "r_strat_a": select_random_by_cell(records_by_id, qmap, args.seed_ra),
        "r_strat_b": select_random_by_cell(records_by_id, qmap, args.seed_rb),
        "c_unique_high": select_by_cell(records_by_id, qmap, "c_unique_resid", high=True),
        "c_unique_control": select_by_cell(records_by_id, qmap, "c_unique_resid", high=False),
        "s_unique_high": select_by_cell(records_by_id, qmap, "s_unique_resid", high=True),
        "s_unique_control": select_by_cell(records_by_id, qmap, "s_unique_resid", high=False),
    }

    print(json.dumps({"event": "writing_full_pool_features"}), flush=True)
    feature_csv = out_dir / "full_pool_features.csv"
    feature_fields = [
        "example_id", "source", "source_pos", "words", "confound_cell", "pos_bin", "tpw_bin", "quality_bin",
        "c_score", "s_score", "c_style_pred", "s_style_pred", "c_style_resid", "s_style_resid",
        "c_unique_pred", "s_unique_pred", "c_unique_resid", "s_unique_resid", "c_overcontrol_resid", "s_overcontrol_resid",
        *OVERCONTROL_FEATURES,
    ]
    with feature_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=feature_fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in feature_fields})

    arm_metas: dict[str, Any] = {}
    errors: list[str] = []
    for arm_name, ids in arms.items():
        print(json.dumps({"event": "processing_arm", "arm": arm_name, "rows": len(ids)}), flush=True)
        if len(ids) != args.target_words // args.words_per_example:
            errors.append(f"{arm_name}: rows {len(ids)} != {args.target_words // args.words_per_example}")
        total_words = sum(records_by_id[i]["words"] for i in ids)
        if total_words != args.target_words:
            errors.append(f"{arm_name}: words {total_words} != {args.target_words}")
        jsonl_path = write_arm_jsonl(out_dir, arm_name, ids, records_by_id)
        jsonl_errs = mat.validate_jsonl_dryrun(jsonl_path, args.target_words)
        errors.extend([f"{arm_name}/JSONL: {e}" for e in jsonl_errs])
        meta = summarize_arm(records_by_id, ids, poolrows_by_id, tokenizer)
        meta["jsonl_path"] = str(jsonl_path)
        meta["selection_method"] = {
            "r_strat_a": f"random within confound cells, seed={args.seed_ra}",
            "r_strat_b": f"random within confound cells, seed={args.seed_rb}",
            "c_unique_high": "top c_unique_resid within each confound cell",
            "c_unique_control": "bottom c_unique_resid within each confound cell",
            "s_unique_high": "top s_unique_resid within each confound cell",
            "s_unique_control": "bottom s_unique_resid within each confound cell",
        }[arm_name]
        (out_dir / f"{arm_name}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        arm_metas[arm_name] = meta

    overlap = pair_overlap(arms)
    r_ref_ids = arms["r_strat_a"]
    ref_recs = [records_by_id[i] for i in r_ref_ids]
    balance = {}
    for arm_name, ids in arms.items():
        if arm_name == "r_strat_a":
            continue
        balance[f"{arm_name}_vs_r_strat_a"] = max_abs_smd([records_by_id[i] for i in ids], ref_recs, STRICT_CONFOUNDS)
    balance["c_unique_high_vs_control"] = max_abs_smd([records_by_id[i] for i in arms["c_unique_high"]], [records_by_id[i] for i in arms["c_unique_control"]], STRICT_CONFOUNDS)
    balance["s_unique_high_vs_control"] = max_abs_smd([records_by_id[i] for i in arms["s_unique_high"]], [records_by_id[i] for i in arms["s_unique_control"]], STRICT_CONFOUNDS)

    validation = {
        "word_counts_ok": all(sum(records_by_id[i]["words"] for i in ids) == args.target_words for ids in arms.values()),
        "jsonl_ok": len([e for e in errors if "/JSONL" in e]) == 0,
        "c_high_control_disjoint": overlap["c_unique_high__c_unique_control"]["shared_ids"] == 0,
        "s_high_control_disjoint": overlap["s_unique_high__s_unique_control"]["shared_ids"] == 0,
        "c_s_high_overlap_fraction": overlap["c_unique_high__s_unique_high"]["fraction_of_first"],
        "c_s_overlap_under_0_55": overlap["c_unique_high__s_unique_high"]["fraction_of_first"] <= 0.55,
        "strict_confound_balance_high_control_max_smd": {
            "c": balance["c_unique_high_vs_control"]["max_abs_smd"],
            "s": balance["s_unique_high_vs_control"]["max_abs_smd"],
        },
    }

    summary = {
        "status": "ok" if not errors else "errors",
        "screen_name": "cs_4m_residualized_mechanism_screen",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 1),
        "pool_rows": len(pool),
        "pool_words": sum(r.words for r in pool),
        "target_words_per_arm": args.target_words,
        "selection_fraction": args.selection_fraction,
        "confound_cell_definition": {
            "source": True,
            "source_position_bins": args.pos_bins,
            "tokens_per_word_bins_per_source": args.tpw_bins,
            "quality_bins_per_source": args.quality_bins,
            "quality_index": "noise_word_rate + filler_rate + dialogue_marker_rate_row + upper_word_rate",
            "quota_rule": "same cell quotas for all arms; q_cell capped at floor(n/2) for high/control disjointness",
        },
        "strict_confound_features": STRICT_CONFOUNDS,
        "overcontrol_features": OVERCONTROL_FEATURES,
        "tokenizer_summary": tok_summary,
        "residual_models": {
            "c_style_r2": c_r2_style,
            "s_style_r2": s_r2_style,
            "c_unique_r2_with_s_score": c_r2_unique,
            "s_unique_r2_with_c_score": s_r2_unique,
            "c_overcontrol_r2": c_r2_over,
            "s_overcontrol_r2": s_r2_over,
        },
        "quota_cells": len(qmap),
        "quota_rows": sum(qmap.values()),
        "arm_metas": arm_metas,
        "overlap": overlap,
        "balance": balance,
        "validation": validation,
        "errors": errors,
        "full_pool_features_csv": str(feature_csv),
    }
    summary_path = out_dir / "screen_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Short scientific note.
    means = {a: arm_metas[a]["features"] for a in arm_metas}
    lines = [
        "# research residualized C/S 4M screen construction",
        "",
        "This construction repairs the raw C/S arms by selecting within matched source × source-position × tokenizer-compression × quality cells and by using residual scores rather than raw C/S scores.",
        "",
        "## Key validation",
        f"- Arms written: {', '.join(arms.keys())}; each has {args.target_words:,} words: {validation['word_counts_ok']}.",
        f"- C-unique high/control disjoint: {validation['c_high_control_disjoint']}; S-unique high/control disjoint: {validation['s_high_control_disjoint']}.",
        f"- C-unique-high/S-unique-high overlap: {overlap['c_unique_high__s_unique_high']['shared_ids']} rows = {overlap['c_unique_high__s_unique_high']['fraction_of_first']:.3f}.",
        f"- Strict-confound max SMD, C high vs control: {balance['c_unique_high_vs_control']['max_abs_smd']:.3f} ({balance['c_unique_high_vs_control']['max_feature']}); S high vs control: {balance['s_unique_high_vs_control']['max_abs_smd']:.3f} ({balance['s_unique_high_vs_control']['max_feature']}).",
        "",
        "## Mean intended-score movements",
        f"- c_unique_high: C={means['c_unique_high']['c_score']['mean']:.6f}, S={means['c_unique_high']['s_score']['mean']:.6f}, c_unique_resid={means['c_unique_high']['c_unique_resid']['mean']:.6f}, s_unique_resid={means['c_unique_high']['s_unique_resid']['mean']:.6f}.",
        f"- c_unique_control: C={means['c_unique_control']['c_score']['mean']:.6f}, S={means['c_unique_control']['s_score']['mean']:.6f}, c_unique_resid={means['c_unique_control']['c_unique_resid']['mean']:.6f}, s_unique_resid={means['c_unique_control']['s_unique_resid']['mean']:.6f}.",
        f"- s_unique_high: C={means['s_unique_high']['c_score']['mean']:.6f}, S={means['s_unique_high']['s_score']['mean']:.6f}, c_unique_resid={means['s_unique_high']['c_unique_resid']['mean']:.6f}, s_unique_resid={means['s_unique_high']['s_unique_resid']['mean']:.6f}.",
        f"- s_unique_control: C={means['s_unique_control']['c_score']['mean']:.6f}, S={means['s_unique_control']['s_score']['mean']:.6f}, c_unique_resid={means['s_unique_control']['c_unique_resid']['mean']:.6f}, s_unique_resid={means['s_unique_control']['s_unique_resid']['mean']:.6f}.",
        "",
        "## Interpretation before H100 training",
        "These arms are much closer to a mechanism-distinguishing screen than the raw C/S materialization because source-internal position, coarse genre/quality, and tokenizer-compression distributions are held by cell quotas, and the high/control contrasts are driven by residual intended scores. They should still be treated as a screening instrument: inspect `screen_summary.json`, the arm metadata, and a few text samples before launching training.",
        "",
        f"Summary JSON: `{summary_path}`",
        f"Full-pool features: `{feature_csv}`",
    ]
    note_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "summary": str(summary_path),
        "note": str(note_path),
        "elapsed_sec": summary["elapsed_sec"],
        "validation": validation,
        "c_s_high_overlap": overlap["c_unique_high__s_unique_high"],
        "c_balance_max_smd": balance["c_unique_high_vs_control"]["max_abs_smd"],
        "s_balance_max_smd": balance["s_unique_high_vs_control"]["max_abs_smd"],
    }, ensure_ascii=False), flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
