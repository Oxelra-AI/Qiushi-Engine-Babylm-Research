#!/usr/bin/env python3
"""research integration: natural restatement, seed43122 Entity, ordinary split shares, source concentration.

This script is CPU-only and uses already-scored files. It writes concise tables and
research notes that make the relation-typed composition argument exact.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import re
import statistics
import time
from collections import defaultdict
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/integrate_delivered_and_concentration.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
WIKI_DIR = _public_path('experiments/archive/relation_learning/data/wikipedia_simplification_score')
ENTITY_ORIG_ROWS = _public_path('experiments/archive/relation_learning/data/entity_relevant_update_analysis/entity_prediction_augmented_rows.csv')
ENTITY_META_ROWS = _public_path('experiments/archive/relation_learning/data/entity_relevant_update_analysis/entity_item_metadata.csv')
ENTITY_SPLIT_431_DIR = _public_path('experiments/archive/relation_learning/data/split_entity_official_seed43122')
ORD_DIR = _public_path('experiments/archive/relation_learning/data/ordinary_heldout_price_probe')
SPEC_DIR = _public_path('experiments/archive/relation_learning/data/source_specificity_misfire')
OUT_DIR = _public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism')
NOTE = _public_path('research/notes/relation_learning/scope_and_mechanism_integration.md')

CKS = ["chck_80M", "chck_90M", "chck_100M"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], preferred: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    if preferred:
        fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fnum(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(vals) if vals else float("nan")


def sd(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.stdev(vals) if len(vals) >= 2 else float("nan")


def norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


# ---------------------------------------------------------------------------
# Wikipedia natural-restatement integration
# ---------------------------------------------------------------------------

def pick_across(rows: list[dict[str, str]], overlap_bin: str, token_class: str, contrast: str, estimand: str) -> dict[str, str] | None:
    for r in rows:
        if (r.get("overlap_bin") == overlap_bin and r.get("token_class") == token_class
            and r.get("contrast") == contrast and r.get("estimand") == estimand):
            return r
    return None


def integrate_wikipedia() -> tuple[list[dict[str, Any]], list[str]]:
    across = read_csv(_public_path('experiments/archive/relation_learning/data/wikipedia_simplification_score/wikipedia_across_seed_contrasts.csv'))
    late = read_csv(_public_path('experiments/archive/relation_learning/data/wikipedia_simplification_score/wikipedia_late_arm_contrasts.csv'))
    key_rows: list[dict[str, Any]] = []
    for overlap_bin in ["ALL", "high", "medium", "low"]:
        for token_class in ["nonoverlap", "overlap", "ALL"]:
            for contrast in ["RminusC", "VminusC", "VminusR"]:
                rec = {"overlap_bin": overlap_bin, "token_class": token_class, "contrast": contrast}
                for estimand in ["gain_T_vs_N", "gain_T_vs_U", "gain_U_vs_N"]:
                    r = pick_across(across, overlap_bin, token_class, contrast, estimand)
                    if r:
                        rec[estimand + "_mean"] = fnum(r["mean_difference_across_seeds"])
                        rec[estimand + "_seed_sd"] = fnum(r["seed_sd"])
                        rec[estimand + "_mean_pairs"] = fnum(r["mean_pairs"])
                if any(k.endswith("_mean") for k in rec):
                    key_rows.append(rec)

    # seed-level nonoverlap all-bin values for the three central contrasts
    seed_rows = []
    for r in late:
        if r.get("overlap_bin") == "ALL" and r.get("token_class") == "nonoverlap" and r.get("estimand") in {"gain_T_vs_N", "gain_T_vs_U", "gain_U_vs_N"}:
            if r.get("contrast") in {"RminusC", "VminusC", "VminusR"}:
                seed_rows.append({
                    "seed": int(r["seed"]),
                    "contrast": r["contrast"],
                    "estimand": r["estimand"],
                    "n_pairs": int(float(r["n_pairs"])),
                    "mean_difference": fnum(r["mean_difference"]),
                    "se_pair_difference": fnum(r["se_pair_difference"]),
                    "fraction_positive": fnum(r["fraction_positive"]),
                })
    write_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/wikipedia_scope_key_across_seed.csv'), key_rows,
              ["overlap_bin", "token_class", "contrast", "gain_T_vs_N_mean", "gain_T_vs_N_seed_sd", "gain_T_vs_U_mean", "gain_T_vs_U_seed_sd", "gain_U_vs_N_mean", "gain_U_vs_N_seed_sd"])
    write_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/wikipedia_nonoverlap_seed_values.csv'), seed_rows,
              ["seed", "contrast", "estimand", "n_pairs", "mean_difference", "se_pair_difference", "fraction_positive"])

    bullets = []
    def fmt(overlap_bin: str, token_class: str, contrast: str, estimand: str) -> str:
        r = pick_across(across, overlap_bin, token_class, contrast, estimand)
        if not r:
            return "NA"
        return f"{fnum(r['mean_difference_across_seeds']):+.4f} ± {fnum(r['seed_sd']):.4f}"

    bullets.append("Wikipedia natural-restatement scorer convention: `gain_T_vs_N = NLL(N) - NLL(T)`, so positive contrast means an arm gains more from the true source relative to the neutral source than the comparison arm.")
    bullets.append(f"Nonoverlap ALL: R−C gain_T_vs_N {fmt('ALL','nonoverlap','RminusC','gain_T_vs_N')}; gain_U_vs_N {fmt('ALL','nonoverlap','RminusC','gain_U_vs_N')}. This is a source-specific loss of true-source benefit for REPEAT on natural restatements.")
    bullets.append(f"Nonoverlap ALL: V−C gain_T_vs_N {fmt('ALL','nonoverlap','VminusC','gain_T_vs_N')}; gain_U_vs_N {fmt('ALL','nonoverlap','VminusC','gain_U_vs_N')}. VIEW's compact positive source-conditioned residual does not appear as a stable natural nonoverlap benefit in this probe.")
    bullets.append(f"Nonoverlap ALL: V−R gain_T_vs_N {fmt('ALL','nonoverlap','VminusR','gain_T_vs_N')}; V is still much better than R because R is actively below C.")
    bullets.append(f"High-overlap bin, nonoverlap targets: R−C gain_T_vs_N {fmt('high','nonoverlap','RminusC','gain_T_vs_N')}; V−C {fmt('high','nonoverlap','VminusC','gain_T_vs_N')}. The R-side cost strengthens when source/rewrite overlap is high, matching a transferred recognition trigger.")
    return key_rows, bullets


# ---------------------------------------------------------------------------
# Entity seed43122 integration
# ---------------------------------------------------------------------------

def load_entity_meta() -> dict[tuple[str, int], dict[str, str]]:
    out: dict[tuple[str, int], dict[str, str]] = {}
    for r in read_csv(ENTITY_META_ROWS):
        out[(r["uid"], int(r["item_index"]))] = r
    return out


def load_entity_original_seed43122() -> list[dict[str, Any]]:
    rows = []
    for r in read_csv(ENTITY_ORIG_ROWS):
        if r.get("seed") != "43122" or r.get("checkpoint") not in CKS or r.get("arm") not in {"C", "R", "V"}:
            continue
        rows.append({
            "seed": 43122,
            "arm": r["arm"],
            "checkpoint": r["checkpoint"],
            "uid": r["uid"],
            "item_index": int(r["item_index"]),
            "correct": int(r["correct"]),
            "reported_numops": int(r["reported_numops"]),
            "relevant_updates": int(r["relevant_updates"]),
            "total_ops": int(r["total_ops"]),
            "prefix_words": int(r["prefix_words"]),
            "stale_available": int(r.get("stale_available", 0)),
            "stale_is_gold": int(r.get("stale_is_gold", 0)),
            "pred_is_stale_initial": int(r.get("pred_is_stale_initial", 0)),
            "source_family": "original",
        })
    return rows


def load_entity_split_seed43122(meta: dict[tuple[str, int], dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for er in read_csv(_public_path('experiments/archive/relation_learning/data/split_entity_official_seed43122/split_entity_eval_rows.csv')):
        if er.get("returncode") != "0" or not er.get("predictions"):
            continue
        arm = er["role"]
        ck = er["checkpoint"]
        pred_path = ROOT / er["predictions"]
        obj = read_json(pred_path)
        for uid, block in obj.items():
            preds = block.get("predictions", [])
            for i, pr in enumerate(preds):
                key = (uid, i)
                if key not in meta:
                    raise RuntimeError(f"missing Entity metadata for {key}")
                m = meta[key]
                pred = str(pr.get("pred", ""))
                gold = str(m.get("gold", ""))
                stale = str(m.get("stale_initial", ""))
                rows.append({
                    "seed": 43122,
                    "arm": arm,
                    "checkpoint": ck,
                    "uid": uid,
                    "item_index": i,
                    "correct": int(norm(pred) == norm(gold)),
                    "reported_numops": int(m["reported_numops"]),
                    "relevant_updates": int(m["relevant_updates"]),
                    "total_ops": int(m["total_ops"]),
                    "prefix_words": int(m["prefix_words"]),
                    "stale_available": int(m.get("stale_available", 0)),
                    "stale_is_gold": int(m.get("stale_is_gold", 0)),
                    "pred_is_stale_initial": int(norm(pred) == norm(stale) and stale != ""),
                    "source_family": "split",
                    "prediction_path": rel(pred_path),
                })
    return rows


def entity_groups(r: dict[str, Any]) -> list[str]:
    relu = int(r["relevant_updates"])
    total = int(r["total_ops"])
    groups = ["ALL", f"rel_updates_{relu}", f"total_ops_{total}"]
    if relu == 0:
        groups += ["rel_eq0", f"rel0_total_ops_{total}"]
    if relu >= 1:
        groups.append("rel_ge1")
    if relu >= 2:
        groups.append("rel_ge2")
    if relu >= 3:
        groups.append("rel_ge3")
    if relu >= 4:
        groups.append("rel_ge4")
    return groups


def summarize_entity(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bins: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for g in entity_groups(r):
            bins[(r["arm"], r["checkpoint"], g)].append(r)
    summary = []
    for (arm, ck, group), vals in sorted(bins.items()):
        wrong = [v for v in vals if not int(v["correct"])]
        stale_wrong = [v for v in wrong if int(v.get("stale_available", 0)) and not int(v.get("stale_is_gold", 0))]
        summary.append({
            "seed": 43122,
            "arm": arm,
            "checkpoint": ck,
            "group": group,
            "n": len(vals),
            "accuracy_pct": 100.0 * sum(int(v["correct"]) for v in vals) / len(vals),
            "mean_relevant_updates": mean([float(v["relevant_updates"]) for v in vals]),
            "mean_total_ops": mean([float(v["total_ops"]) for v in vals]),
            "mean_prefix_words": mean([float(v["prefix_words"]) for v in vals]),
            "stale_pick_pct_among_wrong_stale_available_not_gold": 100.0 * sum(int(v["pred_is_stale_initial"]) for v in stale_wrong) / len(stale_wrong) if stale_wrong else float("nan"),
        })
    late_bins: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in summary:
        if r["checkpoint"] in CKS:
            late_bins[(r["arm"], r["group"])].append(r)
    late = []
    for (arm, group), vals in sorted(late_bins.items()):
        late.append({
            "seed": 43122,
            "arm": arm,
            "group": group,
            "n_checkpoints": len(vals),
            "n": int(vals[0]["n"]),
            "late_mean_accuracy_pct": mean([float(v["accuracy_pct"]) for v in vals]),
            "mean_relevant_updates": mean([float(v["mean_relevant_updates"]) for v in vals]),
            "mean_total_ops": mean([float(v["mean_total_ops"]) for v in vals]),
            "mean_prefix_words": mean([float(v["mean_prefix_words"]) for v in vals]),
            "late_mean_stale_pick_pct_among_wrong_stale_available_not_gold": mean([float(v["stale_pick_pct_among_wrong_stale_available_not_gold"]) for v in vals]),
        })
    idx = {(r["arm"], r["group"]): r for r in late}
    pairs = [("RminusC", "R", "C"), ("RSminusC", "RS", "C"), ("RSminusR", "RS", "R"),
             ("VminusC", "V", "C"), ("VSminusC", "VS", "C"), ("VSminusV", "VS", "V"),
             ("RminusV", "R", "V"), ("RSminusVS", "RS", "VS")]
    con = []
    for group in sorted({r["group"] for r in late}):
        for cname, a, b in pairs:
            if (a, group) not in idx or (b, group) not in idx:
                continue
            ra, rb = idx[(a, group)], idx[(b, group)]
            con.append({
                "seed": 43122,
                "group": group,
                "contrast": cname,
                "n": int(ra["n"]),
                "n_checkpoints": min(int(ra["n_checkpoints"]), int(rb["n_checkpoints"])),
                "late_mean_delta_accuracy_pct": float(ra["late_mean_accuracy_pct"]) - float(rb["late_mean_accuracy_pct"]),
                "late_mean_acc_a": float(ra["late_mean_accuracy_pct"]),
                "late_mean_acc_b": float(rb["late_mean_accuracy_pct"]),
                "mean_relevant_updates": float(ra["mean_relevant_updates"]),
                "mean_total_ops": float(ra["mean_total_ops"]),
                "mean_prefix_words": float(ra["mean_prefix_words"]),
            })
    return summary, late, con


def integrate_entity_seed43122() -> tuple[list[dict[str, Any]], list[str]]:
    meta = load_entity_meta()
    rows = load_entity_original_seed43122() + load_entity_split_seed43122(meta)
    summary, late, con = summarize_entity(rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/entity_seed43122_rows_original_and_split.csv'), rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/entity_seed43122_late_accuracy_by_group.csv'), late)
    write_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/entity_seed43122_late_contrasts_by_group.csv'), con)
    def get(group: str, contrast: str) -> float:
        for r in con:
            if r["group"] == group and r["contrast"] == contrast:
                return float(r["late_mean_delta_accuracy_pct"])
        return float("nan")
    bullets = []
    bullets.append(f"Seed43122 official Entity split integration: RS−C zero-update {get('rel_eq0','RSminusC'):+.2f} versus original R−C {get('rel_eq0','RminusC'):+.2f}; RS−R {get('rel_eq0','RSminusR'):+.2f}.")
    bullets.append(f"Seed43122 deeper updates: rel_ge3 RS−C {get('rel_ge3','RSminusC'):+.2f} versus original R−C {get('rel_ge3','RminusC'):+.2f}; RS−R {get('rel_ge3','RSminusR'):+.2f}.")
    bullets.append(f"Seed43122 VIEW split behavior differs from seed43022: rel_eq0 VS−C {get('rel_eq0','VSminusC'):+.2f}, rel_ge3 VS−C {get('rel_ge3','VSminusC'):+.2f}, VS−V at rel_ge3 {get('rel_ge3','VSminusV'):+.2f}. The seed43022 VIEW_SPLIT positive-update behavior does not replicate.")
    return con, bullets


# ---------------------------------------------------------------------------
# Ordinary held-out decomposition
# ---------------------------------------------------------------------------

def integrate_ordinary() -> tuple[list[dict[str, Any]], list[str]]:
    rows = read_csv(_public_path('experiments/archive/relation_learning/data/ordinary_heldout_price_probe/ordinary_heldout_late_contrasts.csv'))
    idx = {r["contrast"]: fnum(r["delta_loss"]) for r in rows}
    out = []
    r_total = idx["RminusC"]
    r_split = idx["RSminusC"]
    r_local = idx["RminusRS"]
    v_total = idx["VminusC"]
    v_split = idx["VSminusC"]
    v_local = idx["VminusVS"]
    out.append({"arm_family": "REPEAT", "total_vs_clean": r_total, "split_exposure_vs_clean": r_split, "inwindow_share_local_minus_split": r_local, "share_inwindow_of_total": r_local / r_total if r_total else float("nan"), "share_split_of_total": r_split / r_total if r_total else float("nan")})
    out.append({"arm_family": "VIEW", "total_vs_clean": v_total, "split_exposure_vs_clean": v_split, "inwindow_share_local_minus_split": v_local, "share_inwindow_of_total": v_local / v_total if v_total else float("nan"), "share_split_of_total": v_split / v_total if v_total else float("nan")})
    write_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/ordinary_loss_decomposition.csv'), out,
              ["arm_family", "total_vs_clean", "split_exposure_vs_clean", "inwindow_share_local_minus_split", "share_inwindow_of_total", "share_split_of_total"])
    bullets = [
        f"Ordinary held-out decomposition at seed43022: R−C {r_total:+.4f} nats = RS−C {r_split:+.4f} + R−RS {r_local:+.4f}; the in-window share is {100*r_local/r_total:.1f}% of the small ordinary-loss excess.",
        f"For VIEW: V−C {v_total:+.4f} nats = VS−C {v_split:+.4f} + V−VS {v_local:+.4f}; the in-window share is {100*v_local/v_total:.1f}%. Most ordinary held-out loss difference is not the same-window relation component, in contrast to compact T/U/N.",
    ]
    return out, bullets


# ---------------------------------------------------------------------------
# Source-content concentration
# ---------------------------------------------------------------------------

def integrate_concentration() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows = read_csv(_public_path('experiments/archive/relation_learning/data/source_specificity_misfire/specificity_all.csv'))
    # T rows are the scientifically relevant true-source-present condition.
    # source_content_concentration = target_prob / true_src_content_mass. This is an upper-bound proxy for
    # probability on the correct source-overlap token because research saved total content mass, not full vocab vectors.
    # If shifted mass were precise, R−C should raise this ratio on overlap targets; if diffuse, mass rises while ratio falls.
    bins: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    derived_rows = []
    for r in rows:
        cond = r["condition"]
        if cond != "T":
            continue
        seed = int(r["seed"])
        role = r["role"]
        ck = r["checkpoint"]
        tprob = fnum(r["target_prob"])
        mass = fnum(r["true_src_content_mass"])
        n_content = int(float(r["n_true_src_content"])) if str(r.get("n_true_src_content", "")).strip() else 0
        ratio = tprob / mass if mass > 0 else float("nan")
        avg_content_prob = mass / n_content if n_content > 0 else float("nan")
        enrichment = tprob / avg_content_prob if avg_content_prob > 0 else float("nan")
        # research rows are token-nonoverlap rewrite masks by construction; the target token is absent from true source.
        # Concentration still tests whether source-mass movement competes with the target rather than supporting it.
        dr = {
            "seed": seed, "role": role, "checkpoint": ck, "condition": cond,
            "pair_index": int(r["pair_index"]), "target_token_id": int(r["target_token_id"]),
            "true_src_content_mass": mass, "target_prob": tprob, "n_true_src_content": n_content,
            "target_over_source_content_mass": ratio,
            "target_over_average_true_source_content_token_prob": enrichment,
        }
        derived_rows.append(dr)
        bins[(seed, role, ck, "T")].append(dr)
    write_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/source_specificity_T_concentration_rows.csv'), derived_rows,
              ["seed", "role", "checkpoint", "pair_index", "target_token_id", "true_src_content_mass", "target_prob", "n_true_src_content", "target_over_source_content_mass", "target_over_average_true_source_content_token_prob"])

    term_rows = []
    for (seed, role, ck, cond), vals in sorted(bins.items()):
        term_rows.append({
            "seed": seed, "role": role, "checkpoint": ck, "condition": cond,
            "n": len(vals),
            "mean_true_src_content_mass": mean([v["true_src_content_mass"] for v in vals]),
            "mean_target_prob": mean([v["target_prob"] for v in vals]),
            "mean_target_over_source_content_mass": mean([v["target_over_source_content_mass"] for v in vals]),
            "mean_target_over_average_true_source_content_token_prob": mean([v["target_over_average_true_source_content_token_prob"] for v in vals]),
        })
    # late average by role/seed
    late_bins: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in term_rows:
        if r["checkpoint"] in CKS:
            late_bins[(r["seed"], r["role"])].append(r)
    late_terms = []
    for (seed, role), vals in sorted(late_bins.items()):
        late_terms.append({
            "seed": seed, "role": role, "n_checkpoints": len(vals),
            "mean_true_src_content_mass": mean([v["mean_true_src_content_mass"] for v in vals]),
            "mean_target_prob": mean([v["mean_target_prob"] for v in vals]),
            "mean_target_over_source_content_mass": mean([v["mean_target_over_source_content_mass"] for v in vals]),
            "mean_target_over_average_true_source_content_token_prob": mean([v["mean_target_over_average_true_source_content_token_prob"] for v in vals]),
        })
    idx = {(r["seed"], r["role"]): r for r in late_terms}
    con_rows = []
    for seed in sorted({r["seed"] for r in late_terms}):
        for cname, a, b in [("RminusC", "R", "C"), ("VminusC", "V", "C"), ("VminusR", "V", "R")]:
            if (seed, a) not in idx or (seed, b) not in idx:
                continue
            ra, rb = idx[(seed, a)], idx[(seed, b)]
            con_rows.append({
                "seed": seed, "contrast": cname,
                "delta_true_src_content_mass": ra["mean_true_src_content_mass"] - rb["mean_true_src_content_mass"],
                "delta_target_prob": ra["mean_target_prob"] - rb["mean_target_prob"],
                "delta_target_over_source_content_mass": ra["mean_target_over_source_content_mass"] - rb["mean_target_over_source_content_mass"],
                "delta_target_over_average_true_source_content_token_prob": ra["mean_target_over_average_true_source_content_token_prob"] - rb["mean_target_over_average_true_source_content_token_prob"],
            })
    # across-seed contrast summary
    across = []
    for cname in ["RminusC", "VminusC", "VminusR"]:
        vals = [r for r in con_rows if r["contrast"] == cname]
        if not vals:
            continue
        rec = {"contrast": cname, "n_seeds": len(vals)}
        for col in ["delta_true_src_content_mass", "delta_target_prob", "delta_target_over_source_content_mass", "delta_target_over_average_true_source_content_token_prob"]:
            xs = [float(v[col]) for v in vals]
            rec[col + "_mean"] = mean(xs)
            rec[col + "_seed_sd"] = sd(xs)
        across.append(rec)
    write_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/source_specificity_T_concentration_late_terms.csv'), late_terms,
              ["seed", "role", "n_checkpoints", "mean_true_src_content_mass", "mean_target_prob", "mean_target_over_source_content_mass", "mean_target_over_average_true_source_content_token_prob"])
    write_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/source_specificity_T_concentration_late_contrasts.csv'), con_rows,
              ["seed", "contrast", "delta_true_src_content_mass", "delta_target_prob", "delta_target_over_source_content_mass", "delta_target_over_average_true_source_content_token_prob"])
    write_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/source_specificity_T_concentration_across_seed.csv'), across,
              ["contrast", "n_seeds", "delta_true_src_content_mass_mean", "delta_true_src_content_mass_seed_sd", "delta_target_prob_mean", "delta_target_prob_seed_sd", "delta_target_over_source_content_mass_mean", "delta_target_over_source_content_mass_seed_sd", "delta_target_over_average_true_source_content_token_prob_mean", "delta_target_over_average_true_source_content_token_prob_seed_sd"])

    def get_across(cname: str, col: str) -> float:
        for r in across:
            if r["contrast"] == cname:
                return float(r[col])
        return float("nan")
    bullets = [
        "research concentration readout is restricted to tokenizer-nonoverlap rewrite targets, so the target token is absent from the true source. It cannot show exact copying of the answer token; it tests whether the recognized-source mass movement supports or competes with the target.",
        f"R−C under true-source T: source-content mass changes by {get_across('RminusC','delta_true_src_content_mass_mean'):+.4f} ± {get_across('RminusC','delta_true_src_content_mass_seed_sd'):.4f}, while target probability changes by {get_across('RminusC','delta_target_prob_mean'):+.4f} ± {get_across('RminusC','delta_target_prob_seed_sd'):.4f}.",
        f"R−C target/source-content-mass ratio changes by {get_across('RminusC','delta_target_over_source_content_mass_mean'):+.4f} ± {get_across('RminusC','delta_target_over_source_content_mass_seed_sd'):.4f}; the shifted mass is not precision on the correct nonoverlap target, but diffuse source-content pull competing with the target.",
        f"V−C under T also raises source-content mass by {get_across('VminusC','delta_true_src_content_mass_mean'):+.4f} ± {get_across('VminusC','delta_true_src_content_mass_seed_sd'):.4f}, but raises target probability by {get_across('VminusC','delta_target_prob_mean'):+.4f} ± {get_across('VminusC','delta_target_prob_seed_sd'):.4f}; this separates VIEW from REPEAT as content-conditioned support rather than source-content competition.",
    ]
    return across, con_rows, bullets


# ---------------------------------------------------------------------------
# Note
# ---------------------------------------------------------------------------

def write_note(wiki_bullets: list[str], entity_bullets: list[str], ordinary_bullets: list[str], concentration_bullets: list[str]) -> None:
    lines: list[str] = []
    lines.append("# research scope and mechanism integration")
    lines.append("")
    lines.append(f"Created: {now()}")
    lines.append("")
    lines.append("This note integrates the delivered Wikipedia natural-restatement scores and seed43122 split Entity scores, then repairs two wording-sensitive parts of the relation-typed composition argument using already-scored data: ordinary held-out decomposition and source-content concentration.")
    lines.append("")
    lines.append("## 1. Wikipedia natural-restatement T/U/N")
    lines.append("")
    for b in wiki_bullets:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("Interpretation: the adequate natural restatement probe carries the negative REPEAT-side pattern into source-determined Wikipedia/Simple-English sentence pairs. It does not carry the compact VIEW positive residual on nonoverlap targets. This narrows transfer: exact local recurrence has a natural-domain cost for related nonidentical use; the positive VIEW side remains compact-format strong unless later evidence adds a natural setting where the true source gives a stable V−C advantage.")
    lines.append("")
    lines.append("## 2. Seed43122 official Entity split integration")
    lines.append("")
    for b in entity_bullets:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("Interpretation: REPEAT's zero-update Entity benefit is localized at two seeds; splitting removes it at seed43122 as at seed43022. The VIEW_SPLIT positive-update behavior from seed43022 does not reproduce at seed43122, so the behavioral face should remain REPEAT-asymmetric rather than two-sided.")
    lines.append("")
    lines.append("## 3. Ordinary held-out decomposition of local versus split share")
    lines.append("")
    for b in ordinary_bullets:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("This directly corrects the relation to ICLM: the present experiment separates corpus exposure from window adjacency. At this BabyLM dose, the window-adjacency component is small on ordinary held-out text and large only on compact/source-conditioned probes. Therefore the present data do not explain ICLM's broad perplexity loss; they identify a targeted in-window computation that ICLM did not isolate.")
    lines.append("")
    lines.append("## 4. Source-content concentration from research")
    lines.append("")
    for b in concentration_bullets:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("Mechanism wording should change from 'identity pull' to: exact recurrence trains a trigger that transfers to related spans and moves probability mass onto source-content tokens, but its answer-level readout does not transfer to nonidentical targets. VIEW also recognizes source content, but the target probability moves with it rather than against it.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append(f"- `{rel(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/wikipedia_scope_key_across_seed.csv'))}`")
    lines.append(f"- `{rel(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/wikipedia_nonoverlap_seed_values.csv'))}`")
    lines.append(f"- `{rel(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/entity_seed43122_late_contrasts_by_group.csv'))}`")
    lines.append(f"- `{rel(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/ordinary_loss_decomposition.csv'))}`")
    lines.append(f"- `{rel(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/source_specificity_T_concentration_across_seed.csv'))}`")
    _public_path('research/notes/relation_learning').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wiki_rows, wiki_bullets = integrate_wikipedia()
    entity_con, entity_bullets = integrate_entity_seed43122()
    ordinary_rows, ordinary_bullets = integrate_ordinary()
    concentration_across, concentration_con, concentration_bullets = integrate_concentration()
    write_note(wiki_bullets, entity_bullets, ordinary_bullets, concentration_bullets)
    payload = {
        "status": "INTEGRATION_DONE",
        "created_utc": now(),
        "note": rel(NOTE),
        "out_dir": rel(OUT_DIR),
        "wiki_key_rows": len(wiki_rows),
        "entity_contrast_rows": len(entity_con),
        "ordinary_decomposition_rows": len(ordinary_rows),
        "concentration_across_rows": len(concentration_across),
    }
    (_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/integration_summary.json')).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
