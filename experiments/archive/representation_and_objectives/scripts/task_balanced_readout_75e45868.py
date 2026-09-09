#!/usr/bin/env python3
"""research: within-trajectory fast-path task-balanced readout.

This readout consumes saved official-compatible per-target payloads only. It is
with the corrected comparison criterion: frozen chck_80M is not a genuinely
independent anchor, but a same-seed scale-1.75 checkpoint two million words before
chck_82M. Therefore a positive result can show within-trajectory checkpoint
robustness of the private fast-path idea; it cannot establish trajectory-general
stability-plasticity.

The readout tests whether coherent private replay from chck_80M shows
retained-plus-new competence beyond both (i) ordinary same-trajectory continuation
to chck_84M and (ii) a spanbreak/shuffled private replay from the same frozen
anchor. It also implements the promised prediction-change-zone signature: the
items whose top predictions change relative to the frozen anchor are selected
without using correctness labels; within that label-free selected change zone,
the changed predictions must be more often helpful than the anchor's displaced
predictions and more helpful than the corresponding ordinary/shuffled change
zones.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import asdict
from statistics import mean
from typing import Any, Dict, Iterable

ROOT = _public_path('.')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(A02_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A02_SCRIPTS))

import pairwise_item_flip_analysis as s107  # noqa: E402

# Make research prediction/data path resolution independent of the shell cwd.
s107.ROOT = ROOT

READOUT_COLS = list(s107.DISCRETE_COLUMNS)  # BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA
CHEAP_COLS = list(s107.CHEAP_COLS)          # plus Reading
ARM_NAMES = ["anchor80", "coherent80", "shuffled80", "ordinary84"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_cheap7(payload: dict[str, Any]) -> float:
    scores = payload.get("official_overall", {}).get("scores", {})
    vals = []
    missing = []
    for col in CHEAP_COLS:
        v = scores.get(col)
        if v is None:
            missing.append(col)
        else:
            vals.append(float(v))
    if missing:
        raise RuntimeError(f"Missing cheap7 columns: {missing}")
    return mean(vals)


def pred_key(x: Any) -> str:
    if x is None:
        return "<NONE>"
    return s107.norm_text(x)


def safe_div(num: float, den: float, default: float | None = None) -> float | None:
    if den == 0:
        return default
    return num / den


def pct(num: float, den: float) -> float | None:
    v = safe_div(num, den, None)
    return None if v is None else 100.0 * v


def official_score_common(rows: list[s107.ItemRow], column: str) -> float | None:
    try:
        return float(s107.official_score(rows, column))
    except Exception:
        return None


def load_column_maps(loaders: dict[str, s107.PayloadLoader], column: str) -> dict[str, dict[str, s107.ItemRow]]:
    out: dict[str, dict[str, s107.ItemRow]] = {}
    for arm, loader in loaders.items():
        rows, _meta = loader.load_column(column)
        out[arm] = {r.item_id: r for r in rows}
    return out


def pair_counts(anchor: dict[str, s107.ItemRow], cand: dict[str, s107.ItemRow], item_ids: Iterable[str]) -> dict[str, Any]:
    gain = loss = both_correct = both_wrong = 0
    pred_change = 0
    pred_change_cand_correct = 0
    pred_change_anchor_correct = 0
    pred_change_gain = 0
    pred_change_loss = 0
    pred_change_both_correct = 0
    pred_change_both_wrong = 0
    examples = {"gain": [], "loss": [], "change_gain": [], "change_loss": []}
    for item_id in item_ids:
        a = anchor[item_id]
        c = cand[item_id]
        ac = bool(a.correct)
        cc = bool(c.correct)
        if (not ac) and cc:
            typ = "gain"; gain += 1
        elif ac and (not cc):
            typ = "loss"; loss += 1
        elif ac and cc:
            typ = "both_correct"; both_correct += 1
        else:
            typ = "both_wrong"; both_wrong += 1
        changed = pred_key(a.pred) != pred_key(c.pred)
        if changed:
            pred_change += 1
            pred_change_cand_correct += int(cc)
            pred_change_anchor_correct += int(ac)
            if typ == "gain":
                pred_change_gain += 1
                if len(examples["change_gain"]) < 8:
                    examples["change_gain"].append(example_record(item_id, a, c))
            elif typ == "loss":
                pred_change_loss += 1
                if len(examples["change_loss"]) < 8:
                    examples["change_loss"].append(example_record(item_id, a, c))
            elif typ == "both_correct":
                pred_change_both_correct += 1
            else:
                pred_change_both_wrong += 1
        if typ in ("gain", "loss") and len(examples[typ]) < 8:
            examples[typ].append(example_record(item_id, a, c))

    anchor_correct = both_correct + loss
    anchor_wrong = gain + both_wrong
    total = anchor_correct + anchor_wrong
    change_benefit = safe_div(pred_change_cand_correct - pred_change_anchor_correct, pred_change, 0.0)
    return {
        "total_items": total,
        "anchor_correct": anchor_correct,
        "anchor_wrong": anchor_wrong,
        "gain": gain,
        "loss": loss,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_gain_minus_loss": gain - loss,
        "retention": safe_div(both_correct, anchor_correct, 1.0),
        "discovery": safe_div(gain, anchor_wrong, 0.0),
        "correct_pct": pct(gain + both_correct, total),
        "anchor_correct_pct": pct(anchor_correct, total),
        "gain_pct": pct(gain, total),
        "loss_pct": pct(loss, total),
        "prediction_change_zone": {
            "size": pred_change,
            "candidate_correct": pred_change_cand_correct,
            "anchor_correct": pred_change_anchor_correct,
            "candidate_accuracy": safe_div(pred_change_cand_correct, pred_change, None),
            "anchor_accuracy": safe_div(pred_change_anchor_correct, pred_change, None),
            "benefit": change_benefit,
            "gain": pred_change_gain,
            "loss": pred_change_loss,
            "both_correct": pred_change_both_correct,
            "both_wrong": pred_change_both_wrong,
        },
        "examples": examples,
    }


def example_record(item_id: str, anchor: s107.ItemRow, cand: s107.ItemRow) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "uid": anchor.uid,
        "sub": anchor.sub,
        "anchor_pred": anchor.pred,
        "candidate_pred": cand.pred,
        "gold": anchor.gold,
        "anchor_correct": bool(anchor.correct),
        "candidate_correct": bool(cand.correct),
        "meta": anchor.meta,
    }


def multiarm_counts(rows: dict[str, dict[str, s107.ItemRow]], item_ids: list[str]) -> dict[str, Any]:
    counts: dict[str, int] = {
        "coherent_gain_total": 0,
        "coherent_gain_also_ordinary": 0,
        "coherent_gain_also_shuffled": 0,
        "coherent_gain_shared_by_both_comparators": 0,
        "coherent_gain_unique_vs_both": 0,
        "coherent_loss_total": 0,
        "coherent_loss_also_ordinary_wrong": 0,
        "coherent_loss_also_shuffled_wrong": 0,
        "coherent_loss_shared_by_both_comparators": 0,
        "coherent_loss_unique_vs_both": 0,
        "retained_anchor_correct_both_comparators_wrong": 0,
        "coherent_correct_both_comparators_wrong": 0,
        "ordinary_correct_coherent_wrong": 0,
        "shuffled_correct_coherent_wrong": 0,
        "coherent_correct_ordinary_wrong": 0,
        "coherent_correct_shuffled_wrong": 0,
        "coherent_pred_change": 0,
        "ordinary_pred_change": 0,
        "shuffled_pred_change": 0,
        "coherent_pred_change_shared_ordinary": 0,
        "coherent_pred_change_shared_shuffled": 0,
        "coherent_pred_change_unique_vs_both": 0,
    }
    examples = {
        "coherent_gain_unique_vs_both": [],
        "retained_anchor_correct_both_comparators_wrong": [],
        "coherent_loss_unique_vs_both": [],
    }
    for item_id in item_ids:
        a = rows["anchor80"][item_id]
        c = rows["coherent80"][item_id]
        s = rows["shuffled80"][item_id]
        o = rows["ordinary84"][item_id]
        ac, cc, sc, oc = bool(a.correct), bool(c.correct), bool(s.correct), bool(o.correct)
        cchg = pred_key(c.pred) != pred_key(a.pred)
        ochg = pred_key(o.pred) != pred_key(a.pred)
        schg = pred_key(s.pred) != pred_key(a.pred)
        counts["coherent_pred_change"] += int(cchg)
        counts["ordinary_pred_change"] += int(ochg)
        counts["shuffled_pred_change"] += int(schg)
        counts["coherent_pred_change_shared_ordinary"] += int(cchg and ochg)
        counts["coherent_pred_change_shared_shuffled"] += int(cchg and schg)
        counts["coherent_pred_change_unique_vs_both"] += int(cchg and not ochg and not schg)

        if (not ac) and cc:
            counts["coherent_gain_total"] += 1
            counts["coherent_gain_also_ordinary"] += int(oc)
            counts["coherent_gain_also_shuffled"] += int(sc)
            counts["coherent_gain_shared_by_both_comparators"] += int(oc and sc)
            unique = (not oc) and (not sc)
            counts["coherent_gain_unique_vs_both"] += int(unique)
            if unique and len(examples["coherent_gain_unique_vs_both"]) < 10:
                examples["coherent_gain_unique_vs_both"].append(example_record_multi(item_id, a, c, s, o))
        if ac and (not cc):
            counts["coherent_loss_total"] += 1
            counts["coherent_loss_also_ordinary_wrong"] += int(not oc)
            counts["coherent_loss_also_shuffled_wrong"] += int(not sc)
            counts["coherent_loss_shared_by_both_comparators"] += int((not oc) and (not sc))
            unique_loss = oc and sc
            counts["coherent_loss_unique_vs_both"] += int(unique_loss)
            if unique_loss and len(examples["coherent_loss_unique_vs_both"]) < 10:
                examples["coherent_loss_unique_vs_both"].append(example_record_multi(item_id, a, c, s, o))
        if ac and cc and (not oc) and (not sc):
            counts["retained_anchor_correct_both_comparators_wrong"] += 1
            if len(examples["retained_anchor_correct_both_comparators_wrong"]) < 10:
                examples["retained_anchor_correct_both_comparators_wrong"].append(example_record_multi(item_id, a, c, s, o))
        counts["coherent_correct_both_comparators_wrong"] += int(cc and (not oc) and (not sc))
        counts["ordinary_correct_coherent_wrong"] += int(oc and (not cc))
        counts["shuffled_correct_coherent_wrong"] += int(sc and (not cc))
        counts["coherent_correct_ordinary_wrong"] += int(cc and (not oc))
        counts["coherent_correct_shuffled_wrong"] += int(cc and (not sc))

    out: dict[str, Any] = dict(counts)
    total = len(item_ids)
    out["n_common_all_arms"] = total
    out["coherent_gain_unique_fraction_of_coherent_gains"] = safe_div(
        counts["coherent_gain_unique_vs_both"], counts["coherent_gain_total"], None
    )
    out["coherent_gain_shared_shuffled_fraction"] = safe_div(
        counts["coherent_gain_also_shuffled"], counts["coherent_gain_total"], None
    )
    out["coherent_gain_shared_ordinary_fraction"] = safe_div(
        counts["coherent_gain_also_ordinary"], counts["coherent_gain_total"], None
    )
    out["coherent_loss_unique_fraction_of_coherent_losses"] = safe_div(
        counts["coherent_loss_unique_vs_both"], counts["coherent_loss_total"], None
    )
    out["coherent_pred_change_overlap_shuffled_fraction"] = safe_div(
        counts["coherent_pred_change_shared_shuffled"], counts["coherent_pred_change"], None
    )
    out["coherent_pred_change_overlap_ordinary_fraction"] = safe_div(
        counts["coherent_pred_change_shared_ordinary"], counts["coherent_pred_change"], None
    )
    out["unique_gain_minus_unique_loss"] = counts["coherent_gain_unique_vs_both"] - counts["coherent_loss_unique_vs_both"]
    out["examples"] = examples
    return out


def example_record_multi(item_id: str, anchor: s107.ItemRow, coherent: s107.ItemRow, shuffled: s107.ItemRow, ordinary: s107.ItemRow) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "uid": anchor.uid,
        "sub": anchor.sub,
        "gold": anchor.gold,
        "anchor": {"pred": anchor.pred, "correct": bool(anchor.correct)},
        "coherent": {"pred": coherent.pred, "correct": bool(coherent.correct)},
        "shuffled": {"pred": shuffled.pred, "correct": bool(shuffled.correct)},
        "ordinary": {"pred": ordinary.pred, "correct": bool(ordinary.correct)},
        "meta": anchor.meta,
    }


def fmt_float(v: Any, digits: int = 4) -> str:
    if v is None:
        return "NA"
    try:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "NA"
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def aggregate_pair(pair_by_col: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "gain": 0,
        "loss": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "total_items": 0,
        "anchor_correct": 0,
        "anchor_wrong": 0,
        "prediction_change_size": 0,
        "prediction_change_candidate_correct": 0,
        "prediction_change_anchor_correct": 0,
    }
    for rec in pair_by_col.values():
        out["gain"] += int(rec["gain"])
        out["loss"] += int(rec["loss"])
        out["both_correct"] += int(rec["both_correct"])
        out["both_wrong"] += int(rec["both_wrong"])
        out["total_items"] += int(rec["total_items"])
        out["anchor_correct"] += int(rec["anchor_correct"])
        out["anchor_wrong"] += int(rec["anchor_wrong"])
        cz = rec["prediction_change_zone"]
        out["prediction_change_size"] += int(cz["size"])
        out["prediction_change_candidate_correct"] += int(cz["candidate_correct"])
        out["prediction_change_anchor_correct"] += int(cz["anchor_correct"])
    out["net_gain_minus_loss"] = out["gain"] - out["loss"]
    out["retention"] = safe_div(out["both_correct"], out["anchor_correct"], 1.0)
    out["discovery"] = safe_div(out["gain"], out["anchor_wrong"], 0.0)
    out["change_zone_benefit"] = safe_div(
        out["prediction_change_candidate_correct"] - out["prediction_change_anchor_correct"],
        out["prediction_change_size"],
        0.0,
    )
    out["change_zone_candidate_accuracy"] = safe_div(out["prediction_change_candidate_correct"], out["prediction_change_size"], None)
    out["change_zone_anchor_accuracy"] = safe_div(out["prediction_change_anchor_correct"], out["prediction_change_size"], None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-payload", required=True)
    ap.add_argument("--coherent-payload", required=True)
    ap.add_argument("--shuffled-payload", required=True)
    ap.add_argument("--ordinary-payload", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_paths = {
        "anchor80": pathlib.Path(args.anchor_payload),
        "coherent80": pathlib.Path(args.coherent_payload),
        "shuffled80": pathlib.Path(args.shuffled_payload),
        "ordinary84": pathlib.Path(args.ordinary_payload),
    }
    for name, path in payload_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    payload_json = {name: load_json(path) for name, path in payload_paths.items()}
    cheap7 = {name: get_cheap7(payload_json[name]) for name in ARM_NAMES}
    score_columns = {
        name: {col: payload_json[name].get("official_overall", {}).get("scores", {}).get(col) for col in CHEAP_COLS}
        for name in ARM_NAMES
    }
    for name in ARM_NAMES:
        print(json.dumps({"event": "loaded", "arm": name, "cheap7": cheap7[name], "path": rel(payload_paths[name])}), flush=True)

    loaders = {name: s107.PayloadLoader(path) for name, path in payload_paths.items()}

    per_column: dict[str, Any] = {}
    pair_by_arm: dict[str, dict[str, dict[str, Any]]] = {"coherent80": {}, "shuffled80": {}, "ordinary84": {}}
    pooled_multi_counts: dict[str, int] = {}
    total_common_all = 0

    for col in READOUT_COLS:
        maps = load_column_maps(loaders, col)
        common = sorted(set.intersection(*(set(m.keys()) for m in maps.values())))
        total_common_all += len(common)
        pair_records = {}
        for arm in ["coherent80", "shuffled80", "ordinary84"]:
            rec = pair_counts(maps["anchor80"], maps[arm], common)
            pair_records[arm] = rec
            pair_by_arm[arm][col] = rec
        multi = multiarm_counts(maps, common)
        for k, v in multi.items():
            if isinstance(v, int):
                pooled_multi_counts[k] = pooled_multi_counts.get(k, 0) + v
        common_scores = {
            arm: official_score_common([maps[arm][item_id] for item_id in common], col)
            for arm in ARM_NAMES
        }
        per_column[col] = {
            "n_common_all_arms": len(common),
            "scores_reconstructed_common": common_scores,
            "score_deltas_vs_anchor_common": {
                arm: None if common_scores[arm] is None or common_scores["anchor80"] is None else common_scores[arm] - common_scores["anchor80"]
                for arm in ["coherent80", "shuffled80", "ordinary84"]
            },
            "pair_vs_anchor": pair_records,
            "multiarm": multi,
        }

    aggregate = {arm: aggregate_pair(pair_by_arm[arm]) for arm in ["coherent80", "shuffled80", "ordinary84"]}

    # R1: cheap7 must beat all matched alternatives, not just the frozen anchor.
    cheap7_pass = (
        cheap7["coherent80"] > cheap7["anchor80"]
        and cheap7["coherent80"] > cheap7["ordinary84"]
        and cheap7["coherent80"] > cheap7["shuffled80"]
    )

    # R3: task-balanced retained-plus-new competence against ordinary and shuffled controls.
    ret_threshold = max(1, len(READOUT_COLS) - 2)  # 4/6
    disc_threshold = max(1, len(READOUT_COLS) // 2)  # 3/6
    r3_cols: dict[str, Any] = {}
    ret_wins_ord = disc_wins_ord = ret_wins_shuf = disc_wins_shuf = 0
    ret_wins_both = disc_wins_both = 0
    for col in READOUT_COLS:
        coh = per_column[col]["pair_vs_anchor"]["coherent80"]
        ord_ = per_column[col]["pair_vs_anchor"]["ordinary84"]
        shf = per_column[col]["pair_vs_anchor"]["shuffled80"]
        coh_ret, ord_ret, shf_ret = coh["retention"], ord_["retention"], shf["retention"]
        coh_disc, ord_disc, shf_disc = coh["discovery"], ord_["discovery"], shf["discovery"]
        ret_ord = coh_ret is not None and ord_ret is not None and coh_ret >= ord_ret
        disc_ord = coh_disc is not None and ord_disc is not None and coh_disc >= ord_disc
        ret_shf = coh_ret is not None and shf_ret is not None and coh_ret >= shf_ret
        disc_shf = coh_disc is not None and shf_disc is not None and coh_disc >= shf_disc
        ret_both = ret_ord and ret_shf
        disc_both = disc_ord and disc_shf
        ret_wins_ord += int(ret_ord); disc_wins_ord += int(disc_ord)
        ret_wins_shuf += int(ret_shf); disc_wins_shuf += int(disc_shf)
        ret_wins_both += int(ret_both); disc_wins_both += int(disc_both)
        r3_cols[col] = {
            "coherent_retention": coh_ret,
            "ordinary_retention": ord_ret,
            "shuffled_retention": shf_ret,
            "retention_win_vs_ordinary": ret_ord,
            "retention_win_vs_shuffled": ret_shf,
            "retention_win_vs_both": ret_both,
            "coherent_discovery": coh_disc,
            "ordinary_discovery": ord_disc,
            "shuffled_discovery": shf_disc,
            "discovery_win_vs_ordinary": disc_ord,
            "discovery_win_vs_shuffled": disc_shf,
            "discovery_win_vs_both": disc_both,
            "coherent_net_items": coh["net_gain_minus_loss"],
            "ordinary_net_items": ord_["net_gain_minus_loss"],
            "shuffled_net_items": shf["net_gain_minus_loss"],
            "coherent_change_benefit": coh["prediction_change_zone"]["benefit"],
            "ordinary_change_benefit": ord_["prediction_change_zone"]["benefit"],
            "shuffled_change_benefit": shf["prediction_change_zone"]["benefit"],
        }
    r3_ordinary_pass = ret_wins_ord >= ret_threshold and disc_wins_ord >= disc_threshold
    r3_shuffled_pass = ret_wins_shuf >= ret_threshold and disc_wins_shuf >= disc_threshold
    r3_both_pass = ret_wins_both >= ret_threshold and disc_wins_both >= disc_threshold

    # R4: prediction-change-zone private benefit must be positive and exceed both controls.
    coh_benefit = aggregate["coherent80"]["change_zone_benefit"]
    ord_benefit = aggregate["ordinary84"]["change_zone_benefit"]
    shf_benefit = aggregate["shuffled80"]["change_zone_benefit"]
    r4_positive = coh_benefit > 0
    r4_better_ordinary = coh_benefit > ord_benefit
    r4_better_shuffled = coh_benefit > shf_benefit
    r4_pass = r4_positive and r4_better_ordinary and r4_better_shuffled

    decision = "CONTINUE_WITHIN_TRAJECTORY_LEAD" if (cheap7_pass and r3_both_pass and r4_pass) else "STOP_FASTPATH_INSTANTIATION"
    interpretation: list[str] = []
    if decision.startswith("CONTINUE"):
        interpretation.append(
            "Coherent80 passes the within-trajectory readout: it beats anchor80, ordinary84, and shuffled80 in cheap7, has task-balanced retention/discovery against both controls, and its prediction-change zone is more helpful than both controls. This is still not trajectory-general; the next test would need a truly independent seed or trajectory."
        )
    else:
        reasons = []
        if not cheap7_pass:
            reasons.append("cheap7 does not beat all matched alternatives")
        if not r3_both_pass:
            reasons.append("task-balanced retention/discovery does not beat both ordinary and shuffled controls")
        if not r4_pass:
            reasons.append("prediction-change-zone benefit is not positive and above both controls")
        interpretation.append(
            "Fast-path instantiation should stop under the research readout because " + "; ".join(reasons) + "."
        )
    interpretation.append(
        "The anchor is chck_80M from the same seed43022 scale-1.75 trajectory, so any positive result would indicate within-trajectory checkpoint robustness rather than an independent-anchor or trajectory-general learning principle."
    )
    interpretation.append(
        f"Pooled discrete item nets vs anchor: coherent {aggregate['coherent80']['net_gain_minus_loss']:+d}, ordinary {aggregate['ordinary84']['net_gain_minus_loss']:+d}, shuffled {aggregate['shuffled80']['net_gain_minus_loss']:+d}."
    )
    interpretation.append(
        f"Pooled prediction-change benefits: coherent {coh_benefit:+.4f}, ordinary {ord_benefit:+.4f}, shuffled {shf_benefit:+.4f}."
    )
    if pooled_multi_counts:
        ug = pooled_multi_counts.get("coherent_gain_unique_vs_both", 0)
        ul = pooled_multi_counts.get("coherent_loss_unique_vs_both", 0)
        sg = pooled_multi_counts.get("coherent_gain_also_shuffled", 0)
        tg = pooled_multi_counts.get("coherent_gain_total", 0)
        interpretation.append(
            f"Coherent gains unique against both controls: {ug}/{tg}; coherent losses unique against both controls: {ul}; gain-overlap with shuffled: {sg}/{tg}."
        )

    summary = {
        "status": "WITHIN_TRAJECTORY_FASTPATH_TASK_BALANCED_READOUT",
        "created_utc": now(),
        "decision": decision,
        "anchor_interpretation": "chck_80M is a same-seed same-trajectory checkpoint, not a genuinely independent anchor; positive results imply within-trajectory robustness only.",
        "payload_paths": {name: rel(path) for name, path in payload_paths.items()},
        "score_columns": score_columns,
        "cheap7": cheap7,
        "criteria": {
            "cheap7_pass_beats_anchor_ordinary_shuffled": cheap7_pass,
            "r3_retention_threshold_cols": ret_threshold,
            "r3_discovery_threshold_cols": disc_threshold,
            "r3_retention_wins_vs_ordinary": ret_wins_ord,
            "r3_discovery_wins_vs_ordinary": disc_wins_ord,
            "r3_retention_wins_vs_shuffled": ret_wins_shuf,
            "r3_discovery_wins_vs_shuffled": disc_wins_shuf,
            "r3_retention_wins_vs_both": ret_wins_both,
            "r3_discovery_wins_vs_both": disc_wins_both,
            "r3_ordinary_pass": r3_ordinary_pass,
            "r3_shuffled_pass": r3_shuffled_pass,
            "r3_both_pass": r3_both_pass,
            "r4_prediction_change_benefit_coherent": coh_benefit,
            "r4_prediction_change_benefit_ordinary": ord_benefit,
            "r4_prediction_change_benefit_shuffled": shf_benefit,
            "r4_positive": r4_positive,
            "r4_better_than_ordinary": r4_better_ordinary,
            "r4_better_than_shuffled": r4_better_shuffled,
            "r4_pass": r4_pass,
        },
        "aggregate_vs_anchor": aggregate,
        "pooled_multiarm_counts": pooled_multi_counts,
        "per_column_decisions": r3_cols,
        "per_column_detail": per_column,
        "interpretation": interpretation,
    }

    out_json = out_dir / "task_balanced_readout.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md: list[str] = []
    md.append("# research within-trajectory fast-path readout")
    md.append("")
    md.append(f"**Decision: {decision}**")
    md.append("")
    md.append("This is a same-seed scale-1.75 trajectory test: chck_80M is only two million words before the original chck_82M anchor. A positive result would not prove trajectory-general stability-plasticity; it would justify a truly independent seed or trajectory test.")
    md.append("")
    md.append("## Cheap7")
    md.append("")
    md.append("| Arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | Cheap7 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARM_NAMES:
        sc = score_columns[arm]
        md.append(
            f"| {arm} | {fmt_float(sc.get('BLiMP'), 3)} | {fmt_float(sc.get('Supplement'), 3)} | {fmt_float(sc.get('EWoK'), 3)} | {fmt_float(sc.get('Entity'), 3)} | {fmt_float(sc.get('COMPS'), 3)} | {fmt_float(sc.get('GlobalPIQA'), 3)} | {fmt_float(sc.get('Reading'), 3)} | {cheap7[arm]:.4f} |"
        )
    md.append("")
    md.append(f"Cheap7 beats anchor, ordinary, and shuffled: **{'PASS' if cheap7_pass else 'FAIL'}**")
    md.append("")
    md.append("## R3: task-balanced retention and discovery")
    md.append("")
    md.append("| Column | Coh ret | Ord ret | Shuf ret | Ret ≥ both | Coh disc | Ord disc | Shuf disc | Disc ≥ both | Coh net | Ord net | Shuf net |")
    md.append("|---|---:|---:|---:|:---:|---:|---:|---:|:---:|---:|---:|---:|")
    for col in READOUT_COLS:
        d = r3_cols[col]
        md.append(
            f"| {col} | {fmt_float(d['coherent_retention'])} | {fmt_float(d['ordinary_retention'])} | {fmt_float(d['shuffled_retention'])} | {'✓' if d['retention_win_vs_both'] else '✗'} | {fmt_float(d['coherent_discovery'])} | {fmt_float(d['ordinary_discovery'])} | {fmt_float(d['shuffled_discovery'])} | {'✓' if d['discovery_win_vs_both'] else '✗'} | {d['coherent_net_items']:+d} | {d['ordinary_net_items']:+d} | {d['shuffled_net_items']:+d} |"
        )
    md.append("")
    md.append(f"Retention wins vs both controls: {ret_wins_both}/{len(READOUT_COLS)} (need ≥{ret_threshold})")
    md.append(f"Discovery wins vs both controls: {disc_wins_both}/{len(READOUT_COLS)} (need ≥{disc_threshold})")
    md.append(f"R3 vs both controls: **{'PASS' if r3_both_pass else 'FAIL'}**")
    md.append("")
    md.append("## R4: prediction-change-zone signature")
    md.append("")
    md.append("The change zone is selected by prediction changes relative to the frozen anchor, before looking at correctness. Helpful residuals should have positive benefit and exceed ordinary and shuffled controls.")
    md.append("")
    md.append("| Arm | change-zone size | cand correct | anchor correct | cand acc | anchor acc | benefit | total net vs anchor |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ["coherent80", "ordinary84", "shuffled80"]:
        ag = aggregate[arm]
        md.append(
            f"| {arm} | {ag['prediction_change_size']} | {ag['prediction_change_candidate_correct']} | {ag['prediction_change_anchor_correct']} | {fmt_float(ag['change_zone_candidate_accuracy'])} | {fmt_float(ag['change_zone_anchor_accuracy'])} | {ag['change_zone_benefit']:+.4f} | {ag['net_gain_minus_loss']:+d} |"
        )
    md.append("")
    md.append(f"R4 positive and above ordinary/shuffled: **{'PASS' if r4_pass else 'FAIL'}**")
    md.append("")
    md.append("## Multi-arm sharing")
    md.append("")
    if pooled_multi_counts:
        md.append(f"- coherent gains unique against both controls: {pooled_multi_counts.get('coherent_gain_unique_vs_both', 0)}/{pooled_multi_counts.get('coherent_gain_total', 0)}")
        md.append(f"- coherent gains also correct under shuffled: {pooled_multi_counts.get('coherent_gain_also_shuffled', 0)}/{pooled_multi_counts.get('coherent_gain_total', 0)}")
        md.append(f"- coherent gains also correct under ordinary: {pooled_multi_counts.get('coherent_gain_also_ordinary', 0)}/{pooled_multi_counts.get('coherent_gain_total', 0)}")
        md.append(f"- coherent losses unique against both controls: {pooled_multi_counts.get('coherent_loss_unique_vs_both', 0)}/{pooled_multi_counts.get('coherent_loss_total', 0)}")
        md.append(f"- unique gain minus unique loss: {pooled_multi_counts.get('unique_gain_minus_unique_loss', 0):+d}")
    md.append("")
    md.append("## Scientific interpretation")
    md.append("")
    for line in interpretation:
        md.append(f"- {line}")
    md.append("")
    md.append(f"JSON: `{rel(out_json)}`")
    out_md = out_dir / "task_balanced_readout.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "decision": decision,
        "cheap7": cheap7,
        "cheap7_pass": cheap7_pass,
        "r3_both_pass": r3_both_pass,
        "r4_pass": r4_pass,
        "coherent_net_items": aggregate["coherent80"]["net_gain_minus_loss"],
        "ordinary_net_items": aggregate["ordinary84"]["net_gain_minus_loss"],
        "shuffled_net_items": aggregate["shuffled80"]["net_gain_minus_loss"],
        "coherent_change_benefit": coh_benefit,
        "ordinary_change_benefit": ord_benefit,
        "shuffled_change_benefit": shf_benefit,
        "out_json": rel(out_json),
        "out_md": rel(out_md),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
