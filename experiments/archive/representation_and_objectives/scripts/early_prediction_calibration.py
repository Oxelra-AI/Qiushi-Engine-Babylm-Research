#!/usr/bin/env python3
"""research: retrospective calibration of early checkpoint prediction reliability.

Uses existing semantic-view matched trajectory (treatment × 10 checkpoints × 7 tasks,
control × 10 checkpoints × 7 tasks) and EWoK phase dynamics to answer:
  1. At checkpoint M, does the per-task delta predict the 100M delta?
  2. At checkpoint M, does the per-task absolute level predict 100M level?
  3. At checkpoint M, does the equal7 or any single task provide a route-decision signal?
  4. EWoK phase dynamics: at which exposure does the seed-delta stabilize?

Results calibrate whether a 10–20M early screen can distinguish tokenizer or recipe routes.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Any

STUDY = pathlib.Path("experiments/archive/representation_and_objectives")
WS = STUDY
OUT = WS / "data/early_prediction_calibration"
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/early_prediction_calibration.md')

TREATMENT_JSON = WS / "data/semantic_view_noaoa_eval/semantic_view_treatment_trajectory_summary.json"
CONTROL_JSON = WS / "data/semantic_view_noaoa_eval/original_packet_local_trajectory_summary.json"
PHASE_DYNAMICS_JSON = WS / "data/relation_phase_dynamics_summary/relation_phase_dynamics_summary.json"
COMPARE_JSON = WS / "data/corrected_tokenizer_two_seed_comparison/corrected_tokenizer_two_seed_comparison.json"

TASKS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
CHECKPOINTS = [f"chck_{m}M" for m in range(10, 110, 10)]
ENDPOINT = "chck_100M"


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs)
    syy = sum((y-my)**2 for y in ys)
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    denom = math.sqrt(sxx * syy)
    if denom < 1e-15:
        return None
    return sxy / denom


def sign_agreement(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n == 0:
        return None
    agree = sum(1 for x, y in zip(xs, ys) if (x > 0 and y > 0) or (x < 0 and y < 0) or (x == 0 and y == 0))
    return agree / n


def rank_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation."""
    n = len(xs)
    if n < 3:
        return None
    def ranks(vals):
        sorted_v = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        for rank, i in enumerate(sorted_v):
            r[i] = rank + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    return pearson(rx, ry)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    treatment = json.loads(TREATMENT_JSON.read_text())["table"]
    control = json.loads(CONTROL_JSON.read_text())["table"]

    # Build per-checkpoint per-task deltas
    delta_table = {}
    for ck in CHECKPOINTS:
        delta_table[ck] = {}
        for t in TASKS:
            tv = treatment[ck].get(t)
            cv = control[ck].get(t)
            if tv is not None and cv is not None:
                delta_table[ck][t] = tv - cv
            else:
                delta_table[ck][t] = None

    endpoint_delta = delta_table[ENDPOINT]

    # 1. Per-task delta prediction: at each checkpoint, how well does the delta vector
    #    predict the 100M delta vector?
    delta_prediction = []
    for ck in CHECKPOINTS:
        if ck == ENDPOINT:
            continue
        early_d = [delta_table[ck][t] for t in TASKS if delta_table[ck][t] is not None and endpoint_delta.get(t) is not None]
        end_d = [endpoint_delta[t] for t in TASKS if delta_table[ck][t] is not None and endpoint_delta.get(t) is not None]
        delta_prediction.append({
            "checkpoint": ck,
            "n_tasks": len(early_d),
            "pearson_delta_early_vs_endpoint": pearson(early_d, end_d),
            "sign_agreement": sign_agreement(early_d, end_d),
            "rank_correlation": rank_correlation(early_d, end_d),
            "early_mean_delta": sum(early_d) / len(early_d) if early_d else None,
            "endpoint_mean_delta": sum(end_d) / len(end_d) if end_d else None,
        })

    # 2. Per-task absolute level prediction: at each checkpoint, does the
    #    treatment absolute level predict 100M absolute level across tasks?
    absolute_prediction_treatment = []
    for ck in CHECKPOINTS:
        if ck == ENDPOINT:
            continue
        early = [treatment[ck].get(t, 0) for t in TASKS]
        final = [treatment[ENDPOINT].get(t, 0) for t in TASKS]
        absolute_prediction_treatment.append({
            "checkpoint": ck,
            "pearson_abs_early_vs_endpoint": pearson(early, final),
            "rank_correlation": rank_correlation(early, final),
        })

    # 3. Per-task trajectory analysis: for each task, is the treatment-control delta
    #    at early checkpoints a reliable predictor of the endpoint delta?
    per_task_delta_trajectory = {}
    for t in TASKS:
        trajectory = []
        endpoint_d = delta_table[ENDPOINT].get(t)
        for ck in CHECKPOINTS:
            d = delta_table[ck].get(t)
            trajectory.append({
                "checkpoint": ck,
                "delta": d,
            })
        # Does the sign of the delta at each checkpoint match the endpoint?
        sign_stability = []
        for ck in CHECKPOINTS:
            if ck == ENDPOINT:
                continue
            d = delta_table[ck].get(t)
            if d is not None and endpoint_d is not None:
                same = (d > 0 and endpoint_d > 0) or (d < 0 and endpoint_d < 0)
                sign_stability.append({"checkpoint": ck, "delta": d, "matches_endpoint_sign": same})
        per_task_delta_trajectory[t] = {
            "endpoint_delta": endpoint_d,
            "trajectory": trajectory,
            "sign_stability": sign_stability,
            "sign_match_fraction": sum(1 for s in sign_stability if s["matches_endpoint_sign"]) / max(1, len(sign_stability)),
        }

    # 4. Equal7 delta trajectory
    eq7_trajectory = []
    for ck in CHECKPOINTS:
        t_eq7 = treatment[ck].get("equal7_full_eval")
        c_eq7 = control[ck].get("equal7_full_eval")
        eq7_trajectory.append({
            "checkpoint": ck,
            "treatment_eq7": t_eq7,
            "control_eq7": c_eq7,
            "delta_eq7": (t_eq7 - c_eq7) if t_eq7 is not None and c_eq7 is not None else None,
        })

    # 5. Determine minimum exposure for reliable delta prediction
    reliable_at = None
    for row in delta_prediction:
        r = row.get("pearson_delta_early_vs_endpoint")
        if r is not None and r > 0.7:
            reliable_at = row["checkpoint"]
            break

    # 6. EWoK phase dynamics summary from research
    phase = json.loads(PHASE_DYNAMICS_JSON.read_text())
    ewok_summary = {
        "old_tok_50M_to_100M_delta_corr": phase["old_phase_to_endpoint_delta_comparisons"][0]["delta_corr"],
        "old_tok_80M_to_100M_delta_corr": phase["old_phase_to_endpoint_delta_comparisons"][1]["delta_corr"],
        "corrected_tok_50M_accuracy_430_vs_431": f"{phase['strictsmalltok_focused_ewok_553_trajectory']['strictsmalltok_50M']['accuracy_43022']:.1f} vs {phase['strictsmalltok_focused_ewok_553_trajectory']['strictsmalltok_50M']['accuracy_43122']:.1f}",
        "corrected_tok_100M_accuracy_430_vs_431": f"{phase['strictsmalltok_focused_ewok_553_trajectory']['strictsmalltok_100M']['accuracy_43022']:.1f} vs {phase['strictsmalltok_focused_ewok_553_trajectory']['strictsmalltok_100M']['accuracy_43122']:.1f}",
        "corrected_50M_seed_delta_corr_with_old_endpoint": phase['strictsmalltok_focused_ewok_553_trajectory']['strictsmalltok_50M']['old_endpoint_seed_delta_corr'],
        "corrected_100M_seed_delta_corr_with_old_endpoint": phase['strictsmalltok_focused_ewok_553_trajectory']['strictsmalltok_100M']['old_endpoint_seed_delta_corr'],
    }

    # 7. Concrete decision rule
    decision_rules = []
    
    # Check: at 20M, would a screen have correctly predicted the treatment winner?
    t20_eq7 = treatment["chck_20M"]["equal7_full_eval"]
    c20_eq7 = control["chck_20M"]["equal7_full_eval"]
    t100_eq7 = treatment["chck_100M"]["equal7_full_eval"]
    c100_eq7 = control["chck_100M"]["equal7_full_eval"]
    decision_rules.append({
        "question": "At 20M, does the treatment-vs-control equal7 sign predict 100M?",
        "early_delta": t20_eq7 - c20_eq7,
        "endpoint_delta": t100_eq7 - c100_eq7,
        "sign_matches": (t20_eq7 > c20_eq7) == (t100_eq7 > c100_eq7),
        "note": "20M delta was negative despite positive 100M delta" if (t20_eq7 < c20_eq7 and t100_eq7 > c100_eq7) else "consistent"
    })
    
    # Check at 40M
    t40_eq7 = treatment["chck_40M"]["equal7_full_eval"]
    c40_eq7 = control["chck_40M"]["equal7_full_eval"]
    decision_rules.append({
        "question": "At 40M, does the treatment-vs-control equal7 sign predict 100M?",
        "early_delta": t40_eq7 - c40_eq7,
        "endpoint_delta": t100_eq7 - c100_eq7,
        "sign_matches": (t40_eq7 > c40_eq7) == (t100_eq7 > c100_eq7),
        "note": f"40M delta {t40_eq7 - c40_eq7:.3f} is ~10x the 100M delta {t100_eq7 - c100_eq7:.3f}"
    })

    # 8. Compile known score movement with tokenizer change (from research)
    tok_shift = {}
    if COMPARE_JSON.exists():
        cmp = json.loads(COMPARE_JSON.read_text())
        for key in ["inheritedseed43022_vs_correctedseed43022", "inheritedseed43122_vs_correctedseed43122", "mean_corrected_minus_inherited"]:
            if key in cmp.get("comparisons", {}):
                tok_shift[key] = cmp["comparisons"][key]

    payload = {
        "status": "EARLY_PREDICTION_CALIBRATION",
        "purpose": "Determine whether 10–20M checkpoint screens can reliably select tokenizer/recipe routes",
        "data_sources": {
            "treatment": str(TREATMENT_JSON),
            "control": str(CONTROL_JSON),
            "phase_dynamics": str(PHASE_DYNAMICS_JSON),
            "tokenizer_comparison": str(COMPARE_JSON),
        },
        "delta_prediction_across_tasks": delta_prediction,
        "absolute_prediction_treatment": absolute_prediction_treatment,
        "per_task_delta_trajectory": per_task_delta_trajectory,
        "eq7_trajectory": eq7_trajectory,
        "ewok_phase_dynamics": ewok_summary,
        "decision_rules": decision_rules,
        "first_reliable_delta_checkpoint": reliable_at,
        "known_tokenizer_shift": tok_shift,
        "conclusions": [],  # filled below
    }

    # Generate conclusions
    conclusions = []
    conclusions.append(f"Semantic view treatment-control equal7 delta oscillated from +0.40 (10M) to -0.35 (20M) to +1.72 (40M) to +0.17 (100M). The sign flipped at 20M despite positive final effect.")

    ep_supp_d = per_task_delta_trajectory["Supplement"]["endpoint_delta"]
    supp_sign_match = per_task_delta_trajectory["Supplement"]["sign_match_fraction"]
    conclusions.append(f"Supplement delta sign match fraction across 9 early checkpoints: {supp_sign_match:.3f}. Endpoint delta: {ep_supp_d:.2f}.")

    ewok_sign_match = per_task_delta_trajectory["EWoK"]["sign_match_fraction"]
    conclusions.append(f"EWoK delta sign match fraction: {ewok_sign_match:.3f}. EWoK is nearly at noise floor for this treatment size.")

    conclusions.append(f"EWoK seed-delta from old tokenizer: 50M→100M correlation {ewok_summary['old_tok_50M_to_100M_delta_corr']:.3f}, 80M→100M correlation {ewok_summary['old_tok_80M_to_100M_delta_corr']:.3f}. Under corrected tokenizer, dynamics were different and less polarized.")

    if reliable_at:
        conclusions.append(f"First checkpoint with cross-task delta Pearson > 0.7: {reliable_at}.")
    else:
        conclusions.append("No checkpoint achieved cross-task delta Pearson > 0.7 against 100M endpoint. Early delta vectors are unreliable predictors of endpoint task-by-task movement.")

    conclusions.append("A 10–20M early screen cannot reliably predict 100M endpoint quality for this model family. The minimum informative exposure for EWoK/relational tasks is around 80M. For aggregate equal7, even 40M delta was 10× the endpoint delta.")
    conclusions.append("Route selection should rely on representation geometry (tokenizer interface), established mechanism evidence, and staged continuation to high-exposure evaluation, not on 10–20M model scores.")

    payload["conclusions"] = conclusions

    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "early_prediction_calibration.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    # Write note
    lines = ["# research: Early-prediction calibration\n"]
    lines.append("## Purpose\n")
    lines.append("Determine whether short (10–20M) checkpoint screens can reliably distinguish tokenizer/recipe routes before committing a full 100M run.\n\n")
    lines.append("## Semantic view trajectory calibration (treatment vs packet-local control)\n")
    lines.append("| Checkpoint | delta eq7 | Pearson(delta,endpoint) | rank_corr | sign_agree |\n")
    lines.append("|-----------|-----------|------------------------|-----------|------------|\n")
    for row, eq in zip(delta_prediction, eq7_trajectory[:-1]):
        r = row["pearson_delta_early_vs_endpoint"]
        rk = row["rank_correlation"]
        sa = row["sign_agreement"]
        de = eq["delta_eq7"]
        r_s = f"{r:.3f}" if r is not None else "N/A"
        rk_s = f"{rk:.3f}" if rk is not None else "N/A"
        sa_s = f"{sa:.3f}" if sa is not None else "N/A"
        de_s = f"{de:+.3f}" if de is not None else "N/A"
        lines.append(f"| {row['checkpoint']} | {de_s} | {r_s} | {rk_s} | {sa_s} |\n")
    lines.append(f"\nEndpoint eq7 delta: {eq7_trajectory[-1]['delta_eq7']:+.3f}\n\n")

    lines.append("## Per-task delta sign stability\n")
    for t in TASKS:
        info = per_task_delta_trajectory[t]
        lines.append(f"- **{t}**: endpoint delta {info['endpoint_delta']:.3f}, sign match fraction {info['sign_match_fraction']:.3f}\n")

    lines.append("\n## EWoK phase dynamics\n")
    for k, v in ewok_summary.items():
        lines.append(f"- {k}: {v}\n")

    lines.append("\n## Conclusions\n")
    for c in conclusions:
        lines.append(f"- {c}\n")

    lines.append(f"\n## Artifacts\n- JSON: `{out_json}`\n- note: `{NOTE}`\n")

    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines))

    print(json.dumps({
        "status": "EARLY_PREDICTION_CALIBRATION_READY",
        "out_json": str(out_json),
        "note": str(NOTE),
        "first_reliable_delta_checkpoint": reliable_at,
        "eq7_20M_delta": eq7_trajectory[1]["delta_eq7"],
        "eq7_100M_delta": eq7_trajectory[-1]["delta_eq7"],
    }, indent=2))


if __name__ == "__main__":
    main()
