#!/usr/bin/env python3
"""research: synthesize ACS paired-screen evidence.

Inputs are the repaired validated readouts and mechanism diagnostics.  The output
answers the route question: did ACS produce interaction-specific movement beyond
generic hard-negative/CE reshaping, relative to the matched standard-control tail?
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
TREAT_SUMMARY = _public_path('experiments/archive/representation_and_objectives/data/acs_validated_readout/treatment/acs_validated_readout_summary.json')
CTRL_SUMMARY = _public_path('experiments/archive/representation_and_objectives/data/acs_validated_readout/control/acs_validated_readout_summary.json')
DIAG_SUMMARY = _public_path('experiments/archive/representation_and_objectives/data/acs_mechanism_diagnostics/acs_mechanism_diagnostics_summary.json')
ANATOMY = _public_path('experiments/archive/representation_and_objectives/data/globalpiqa_parallel_anatomy/globalpiqa_parallel_anatomy.json')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/acs_synthesis')
NOTE = _public_path('research/notes/representation_and_objectives/acs_screen_synthesis.md')

TREAT_GP_PAR = _public_path('experiments/archive/representation_and_objectives/data/acs_validated_readout/treatment/globalpiqa/acs_treatment_100M_parallel_rows.csv')
CTRL_GP_PAR = _public_path('experiments/archive/representation_and_objectives/data/acs_validated_readout/control/globalpiqa/acs_control_100M_parallel_rows.csv')
TREAT_GP_NONPAR = _public_path('experiments/archive/representation_and_objectives/data/acs_validated_readout/treatment/globalpiqa/acs_treatment_100M_nonparallel_rows.csv')
CTRL_GP_NONPAR = _public_path('experiments/archive/representation_and_objectives/data/acs_validated_readout/control/globalpiqa/acs_control_100M_nonparallel_rows.csv')
TREAT_EWOK = _public_path('experiments/archive/representation_and_objectives/data/acs_validated_readout/treatment/ewok/acs_treatment_100M/ewok_interaction_records.csv')
CTRL_EWOK = _public_path('experiments/archive/representation_and_objectives/data/acs_validated_readout/control/ewok/acs_control_100M/ewok_interaction_records.csv')


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes"}


def parse_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def parse_int(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def qstats(vals) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v)))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {
        "n": len(xs),
        "min": xs[0],
        "p10": q(0.10),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p90": q(0.90),
        "max": xs[-1],
    }


def load_hard_ids_and_categories() -> tuple[set[str], dict[str, list[str]]]:
    d = load_json(ANATOMY)
    rows = d.get("agreement", {}).get("parallel", {}).get("rows", [])
    hard = {r["example_id"] for r in rows if r.get("n_ok") == 0}
    cats = {r["example_id"]: (r.get("categories") or []) for r in rows}
    return hard, cats


def summarize_gp_pair(ctrl_path: Path, treat_path: Path, mode: str, hard_ids: set[str], cats: dict[str, list[str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    crows = {r["example_id"]: r for r in read_csv(ctrl_path)}
    trows = {r["example_id"]: r for r in read_csv(treat_path)}
    ids = sorted(set(crows) & set(trows))
    trans = []
    for eid in ids:
        c = crows[eid]; t = trows[eid]
        cc = parse_bool(c.get("correct")); tc = parse_bool(t.get("correct"))
        cm = parse_float(c.get("top_minus_correct")); tm = parse_float(t.get("top_minus_correct"))
        cr = parse_int(c.get("correct_rank")); tr = parse_int(t.get("correct_rank"))
        rec = {
            "example_id": eid,
            "mode": mode,
            "is_fixed_hard52": eid in hard_ids,
            "control_correct": cc,
            "treatment_correct": tc,
            "transition": ("C->C" if cc and tc else "C->W" if cc and not tc else "W->C" if (not cc and tc) else "W->W"),
            "control_rank": cr,
            "treatment_rank": tr,
            "rank_delta_treat_minus_control": tr - cr,
            "control_top_minus_correct": cm,
            "treatment_top_minus_correct": tm,
            "margin_delta_treat_minus_control": tm - cm if math.isfinite(tm) and math.isfinite(cm) else float("nan"),
            "categories": cats.get(eid, []),
            "control_choice": c.get("choice"),
            "treatment_choice": t.get("choice"),
        }
        trans.append(rec)
    def subset(pred):
        return [r for r in trans if pred(r)]
    all_rows = subset(lambda r: True)
    hard_rows = subset(lambda r: r["is_fixed_hard52"])
    def pack(rows: list[dict[str, Any]]) -> dict[str, Any]:
        cat_counter = Counter()
        for r in rows:
            for cat in r.get("categories") or ["all"]:
                cat_counter[cat] += 1
        return {
            "n": len(rows),
            "control_correct": sum(r["control_correct"] for r in rows),
            "treatment_correct": sum(r["treatment_correct"] for r in rows),
            "net_correct_treat_minus_control": sum(r["treatment_correct"] for r in rows) - sum(r["control_correct"] for r in rows),
            "transitions": dict(Counter(r["transition"] for r in rows)),
            "rank_delta_stats": qstats([r["rank_delta_treat_minus_control"] for r in rows]),
            "margin_delta_stats_treat_minus_control": qstats([r["margin_delta_treat_minus_control"] for r in rows]),
            "changed_choice": sum(str(r["control_choice"]) != str(r["treatment_choice"]) for r in rows),
            "changed_choice_frac": sum(str(r["control_choice"]) != str(r["treatment_choice"]) for r in rows) / len(rows) if rows else None,
            "categories": dict(cat_counter.most_common()),
        }
    out = {
        "mode": mode,
        "n_common": len(ids),
        "all_rows": pack(all_rows),
        "fixed_hard52": pack(hard_rows) if mode == "parallel" else None,
    }
    return out, trans


def summarize_ewok_pair(ctrl_path: Path, treat_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    crows = {r["global_index"]: r for r in read_csv(ctrl_path)}
    trows = {r["global_index"]: r for r in read_csv(treat_path)}
    ids = sorted(set(crows) & set(trows), key=lambda x: int(x))
    trans = []
    for gid in ids:
        c = crows[gid]; t = trows[gid]
        cc = parse_bool(c.get("saved_model_correct_flag")); tc = parse_bool(t.get("saved_model_correct_flag"))
        cs = parse_bool(c.get("conditional_reversal_failure_stable")); ts = parse_bool(t.get("conditional_reversal_failure_stable"))
        ci = parse_float(c.get("interaction_sum")); ti = parse_float(t.get("interaction_sum"))
        rec = {
            "global_index": parse_int(gid),
            "domain": c.get("domain"),
            "ContextDiff": c.get("ContextDiff"),
            "ContextType": c.get("ContextType"),
            "TargetDiff": c.get("TargetDiff"),
            "control_correct": cc,
            "treatment_correct": tc,
            "correct_transition": ("C->C" if cc and tc else "C->W" if cc and not tc else "W->C" if (not cc and tc) else "W->W"),
            "control_stable_failure": cs,
            "treatment_stable_failure": ts,
            "stable_transition": ("S->S" if cs and ts else "S->N" if cs and not ts else "N->S" if (not cs and ts) else "N->N"),
            "control_interaction_sum": ci,
            "treatment_interaction_sum": ti,
            "interaction_delta_treat_minus_control": ti - ci if math.isfinite(ti) and math.isfinite(ci) else float("nan"),
            "control_within_both_positive_wrong": parse_bool(c.get("both_within_context_sum_positive")),
            "treatment_within_both_positive_wrong": parse_bool(t.get("both_within_context_sum_positive")),
            "target1": c.get("target1"),
            "target2": c.get("target2"),
            "context_diff_c1_texts_joined": c.get("context_diff_c1_texts_joined"),
            "context_diff_c2_texts_joined": c.get("context_diff_c2_texts_joined"),
        }
        trans.append(rec)
    def subset(pred):
        return [r for r in trans if pred(r)]
    def domain_counts(rows):
        return dict(Counter(str(r.get("domain")) for r in rows).most_common(15))
    removed = subset(lambda r: r["stable_transition"] == "S->N")
    added = subset(lambda r: r["stable_transition"] == "N->S")
    w2c = subset(lambda r: r["correct_transition"] == "W->C")
    c2w = subset(lambda r: r["correct_transition"] == "C->W")
    out = {
        "n_common": len(trans),
        "accuracy_transition_counts": dict(Counter(r["correct_transition"] for r in trans)),
        "net_correct_treat_minus_control": len(w2c) - len(c2w),
        "stable_transition_counts": dict(Counter(r["stable_transition"] for r in trans)),
        "net_stable_failures_treat_minus_control": len(added) - len(removed),
        "stable_removed_domains": domain_counts(removed),
        "stable_added_domains": domain_counts(added),
        "correct_gained_domains": domain_counts(w2c),
        "correct_lost_domains": domain_counts(c2w),
        "interaction_delta_all_stats": qstats([r["interaction_delta_treat_minus_control"] for r in trans]),
        "interaction_delta_control_wrong_stats": qstats([r["interaction_delta_treat_minus_control"] for r in trans if not r["control_correct"]]),
        "interaction_delta_control_stable_failure_stats": qstats([r["interaction_delta_treat_minus_control"] for r in trans if r["control_stable_failure"]]),
        "within_both_positive_wrong_delta_count": sum(r["treatment_within_both_positive_wrong"] for r in trans if not r["treatment_correct"]) - sum(r["control_within_both_positive_wrong"] for r in trans if not r["control_correct"]),
    }
    return out, trans


def get_metric(summary: dict[str, Any], path: list[str]) -> Any:
    cur: Any = summary
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    treat = load_json(TREAT_SUMMARY)
    ctrl = load_json(CTRL_SUMMARY)
    diag = load_json(DIAG_SUMMARY)
    hard_ids, cats = load_hard_ids_and_categories()

    gp_parallel, gp_parallel_rows = summarize_gp_pair(CTRL_GP_PAR, TREAT_GP_PAR, "parallel", hard_ids, cats)
    gp_nonparallel, gp_nonparallel_rows = summarize_gp_pair(CTRL_GP_NONPAR, TREAT_GP_NONPAR, "nonparallel", hard_ids, {})
    ewok_pair, ewok_rows = summarize_ewok_pair(CTRL_EWOK, TREAT_EWOK)

    def tgp(path): return get_metric(treat, ["globalpiqa", "acs_treatment_100M", *path])
    def cgp(path): return get_metric(ctrl, ["globalpiqa", "acs_control_100M", *path])
    def tew(path): return get_metric(treat, ["ewok", "acs_treatment_100M", "summary", *path])
    def cew(path): return get_metric(ctrl, ["ewok", "acs_control_100M", "summary", *path])

    scalar_deltas = {
        "globalpiqa_parallel_accuracy_pp": tgp(["modes", "parallel", "accuracy"]) - cgp(["modes", "parallel", "accuracy"]),
        "globalpiqa_nonparallel_accuracy_pp": tgp(["modes", "nonparallel", "accuracy"]) - cgp(["modes", "nonparallel", "accuracy"]),
        "globalpiqa_fixed_hard52_accuracy_pp": tgp(["modes", "parallel", "always_wrong_subset", "accuracy"]) - cgp(["modes", "parallel", "always_wrong_subset", "accuracy"]),
        "globalpiqa_fixed_hard52_mean_margin_nats": tgp(["modes", "parallel", "always_wrong_subset", "mean_top_minus_correct"]) - cgp(["modes", "parallel", "always_wrong_subset", "mean_top_minus_correct"]),
        "globalpiqa_all_rows_mean_margin_nats": tgp(["modes", "parallel", "all_rows_margin_summary", "mean_top_minus_correct"]) - cgp(["modes", "parallel", "all_rows_margin_summary", "mean_top_minus_correct"]),
        "ewok_accuracy": tew(["accuracy"]) - cew(["accuracy"]),
        "ewok_saved_wrong": tew(["saved_wrong"]) - cew(["saved_wrong"]),
        "ewok_stable_failure_count": tew(["stable_failure"]) - cew(["stable_failure"]),
        "ewok_stable_failure_frac_all": tew(["stable_failure_frac_all"]) - cew(["stable_failure_frac_all"]),
        "ewok_wrong_interaction_mean_nats": tew(["interaction_sum_wrong", "mean"]) - cew(["interaction_sum_wrong", "mean"]),
        "training_ce_loss_mean": get_metric(diag, ["runs", "treatment", "training_log_summary", "ce_loss_stats", "mean"]) - get_metric(diag, ["runs", "control", "training_log_summary", "ce_loss_stats", "mean"]),
        "training_ce_loss_last": get_metric(diag, ["runs", "treatment", "training_log_summary", "ce_loss_last"]) - get_metric(diag, ["runs", "control", "training_log_summary", "ce_loss_last"]),
        "grad_clip_frac": get_metric(diag, ["runs", "treatment", "training_log_summary", "clip_frac_preclip_grad_norm_gt_1"]) - get_metric(diag, ["runs", "control", "training_log_summary", "clip_frac_preclip_grad_norm_gt_1"]),
        "matched_batch_selected_mass_mean": get_metric(diag, ["runs", "treatment", "matched_batch_diagnostic", "acs_selection", "selected_mass_mean"]) - get_metric(diag, ["runs", "control", "matched_batch_diagnostic", "acs_selection", "selected_mass_mean"]),
        "matched_batch_correct_prob_mean": get_metric(diag, ["runs", "treatment", "matched_batch_diagnostic", "acs_selection", "correct_prob_mean"]) - get_metric(diag, ["runs", "control", "matched_batch_diagnostic", "acs_selection", "correct_prob_mean"]),
        "matched_batch_acs_ce_grad_cosine": get_metric(diag, ["runs", "treatment", "matched_batch_diagnostic", "gradient_alignment", "all_trainable", "cosine"]) - get_metric(diag, ["runs", "control", "matched_batch_diagnostic", "gradient_alignment", "all_trainable", "cosine"]),
    }

    verdict = {
        "acs_single_context_screen_interpretation": (
            "ACS alpha0.5/topk8 completed, but fixed-coordinate readout does not show the required "
            "interaction-specific natural transfer: GlobalPIQA_parallel and fixed hard52 correctness worsen, "
            "EWoK stable failures improve only 20 rows while EWoK accuracy slightly drops, and diagnostics show "
            "ACS increases CE loss/gradient clipping and lowers selected-set probability mass on a matched batch."
        ),
        "route_action": "Do not start alpha/K tuning or from-beginning ACS training from this result; treat ACS-alpha0.5-topk8 as a negative or at best non-decisive single-context hard-negative CE screen unless a separate paired-context coupling mechanism is invented and screened.",
    }

    out = {
        "status": "ACS_SCREEN_SYNTHESIS_DONE",
        "created_utc": now_utc(),
        "inputs": {
            "treatment_summary": rel(TREAT_SUMMARY),
            "control_summary": rel(CTRL_SUMMARY),
            "mechanism_diagnostics": rel(DIAG_SUMMARY),
            "fixed_hard52_source": rel(ANATOMY),
        },
        "fixed_hard52_n": len(hard_ids),
        "scalar_deltas_treatment_minus_control": scalar_deltas,
        "globalpiqa_pair_transitions": {
            "parallel": gp_parallel,
            "nonparallel": gp_nonparallel,
        },
        "ewok_pair_transitions": ewok_pair,
        "diagnostic_highlights": {
            "treatment_training": get_metric(diag, ["runs", "treatment", "training_log_summary"]),
            "control_training": get_metric(diag, ["runs", "control", "training_log_summary"]),
            "treatment_matched_batch": get_metric(diag, ["runs", "treatment", "matched_batch_diagnostic"]),
            "control_matched_batch": get_metric(diag, ["runs", "control", "matched_batch_diagnostic"]),
            "deltas": get_metric(diag, ["deltas_treatment_minus_control"]),
        },
        "verdict": verdict,
    }

    (_public_path('experiments/archive/representation_and_objectives/data/acs_synthesis/globalpiqa_parallel_transitions.jsonl')).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in gp_parallel_rows) + "\n", encoding="utf-8")
    (_public_path('experiments/archive/representation_and_objectives/data/acs_synthesis/globalpiqa_nonparallel_transitions.jsonl')).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in gp_nonparallel_rows) + "\n", encoding="utf-8")
    (_public_path('experiments/archive/representation_and_objectives/data/acs_synthesis/ewok_transitions.jsonl')).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ewok_rows) + "\n", encoding="utf-8")
    out_path = _public_path('experiments/archive/representation_and_objectives/data/acs_synthesis/acs_screen_synthesis.json')
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research ACS paired-screen synthesis")
    lines.append("")
    lines.append("The research ad hoc readout was not used for scientific interpretation. research used the validated length-normalized all-option GlobalPIQA reader and the fixed cross-endpoint hard52 set from research, plus the research EWoK four-cell reader.")
    lines.append("")
    lines.append("## Main treatment-minus-control deltas")
    lines.append("")
    for k, v in scalar_deltas.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("GlobalPIQA row transitions:")
    lines.append(f"- parallel all rows: {gp_parallel['all_rows']}")
    lines.append(f"- fixed hard52: {gp_parallel['fixed_hard52']}")
    lines.append(f"- nonparallel all rows: {gp_nonparallel['all_rows']}")
    lines.append("")
    lines.append("EWoK row transitions:")
    lines.append(f"- {ewok_pair}")
    lines.append("")
    lines.append("## Mechanism diagnostics")
    lines.append("")
    lines.append(f"- Treatment mean CE loss {get_metric(diag, ['runs','treatment','training_log_summary','ce_loss_stats','mean'])}; control {get_metric(diag, ['runs','control','training_log_summary','ce_loss_stats','mean'])}; delta {scalar_deltas['training_ce_loss_mean']}.")
    lines.append(f"- Treatment clipping fraction {get_metric(diag, ['runs','treatment','training_log_summary','clip_frac_preclip_grad_norm_gt_1'])}; control {get_metric(diag, ['runs','control','training_log_summary','clip_frac_preclip_grad_norm_gt_1'])}.")
    lines.append(f"- Matched-batch selected-set mass: treatment {get_metric(diag, ['runs','treatment','matched_batch_diagnostic','acs_selection','selected_mass_mean'])}; control {get_metric(diag, ['runs','control','matched_batch_diagnostic','acs_selection','selected_mass_mean'])}.")
    lines.append(f"- Matched-batch ACS--CE gradient cosine: treatment {get_metric(diag, ['runs','treatment','matched_batch_diagnostic','gradient_alignment','all_trainable','cosine'])}; control {get_metric(diag, ['runs','control','matched_batch_diagnostic','gradient_alignment','all_trainable','cosine'])}.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(verdict["acs_single_context_screen_interpretation"])
    lines.append(verdict["route_action"])
    lines.append("")
    lines.append("Files:")
    lines.append(f"- synthesis JSON: `{rel(out_path)}`")
    lines.append(f"- transitions: `{rel(_public_path('experiments/archive/representation_and_objectives/data/acs_synthesis/globalpiqa_parallel_transitions.jsonl'))}`, `{rel(_public_path('experiments/archive/representation_and_objectives/data/acs_synthesis/ewok_transitions.jsonl'))}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "summary": rel(out_path), "note": rel(NOTE)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
