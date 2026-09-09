#!/usr/bin/env python3
"""research: beta/alpha readout for recipient-sensitive selection.

This script corrects the research interpretation.  For a recipient pair let

  U = log P(new | target-updated context) - log P(source | target-updated context)
  R = log P(source | distractor-updated context) - log P(new | distractor-updated context)

and define

  beta  = (U + R) / 2        # recipient-dependent component
  alpha = (U - R) / 2        # context-shared new-vs-source preference
  gamma = beta - abs(alpha)  # pair is jointly correct iff gamma > 0

Balanced ordinary answer CE already rewards U>0 and R>0; it is not indifferent to
recipient selection.  The paired loss can add direct pressure on beta, but by
itself it does not control alpha, so beta must be read together with joint
correctness and gamma.
"""

from __future__ import annotations

import json
import math
import pathlib
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path("experiments/archive/functional_learning")
DATA = ROOT / "data"
FIGS = ROOT / "figures"
OUT = DATA / "beta_alpha_readout"


def beta_alpha_gamma(U: float, R: float) -> Tuple[float, float, float]:
    beta = 0.5 * (U + R)
    alpha = 0.5 * (U - R)
    gamma = beta - abs(alpha)
    return beta, alpha, gamma


def binary_ce_from_margins(U: float, R: float) -> float:
    # Binary-candidate correct-answer CE up to constants for margins U and R.
    # softplus(-U) + softplus(-R)
    def softplus(x: float) -> float:
        if x > 40:
            return x
        if x < -40:
            return math.exp(x)
        return math.log1p(math.exp(x))
    return softplus(-U) + softplus(-R)


def stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    s = sorted(values)
    def q(p: float) -> float:
        if len(s) == 1:
            return s[0]
        i = p * (len(s) - 1)
        lo = int(math.floor(i)); hi = int(math.ceil(i))
        if lo == hi:
            return s[lo]
        return s[lo] * (hi - i) + s[hi] * (i - lo)
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "q25": q(0.25),
        "median": q(0.50),
        "q75": q(0.75),
        "max": max(values),
        "n_pos": sum(1 for v in values if v > 0),
    }


def read_json(path: pathlib.Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_record(name: str, U: float, R: float, n: Optional[int] = None,
                     joint: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    beta, alpha, gamma = beta_alpha_gamma(float(U), float(R))
    return {
        "name": name,
        "U": float(U),
        "R": float(R),
        "beta": beta,
        "alpha": alpha,
        "gamma_from_means": gamma,
        "binary_candidate_ce_from_mean_margins": binary_ce_from_margins(float(U), float(R)),
        "n": n,
        "joint": joint,
        "joint_frac": (joint / n) if (n and joint is not None) else None,
        **(extra or {}),
    }


def per_group_records_from_template_eval(ev: Dict[str, Any], label: str) -> Dict[str, Any]:
    recs = []
    for g in ev.get("per_group", []):
        U = float(g["mean_update_new_minus_source"] if "mean_update_new_minus_source" in g else g["update_margin_new_over_source"])
        R = float(g["mean_retain_source_minus_new"] if "mean_retain_source_minus_new" in g else g["retain_margin_source_over_new"])
        beta, alpha, gamma = beta_alpha_gamma(U, R)
        recs.append({
            "id": g.get("group_id"),
            "U": U,
            "R": R,
            "beta": beta,
            "alpha": alpha,
            "gamma": gamma,
            "joint_correct": bool((U > 0) and (R > 0)),
            "update_correct": bool(U > 0),
            "retain_correct": bool(R > 0),
            "template": g.get("template_name"),
            "query_side": g.get("query_side"),
            "source_order": g.get("source_order"),
            "new_state": g.get("new_state"),
            "source_state": g.get("source_state"),
        })
    return summarise_records(label, recs)


def summarise_records(label: str, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "label": label,
        "n": len(recs),
        "n_joint": sum(1 for r in recs if r.get("joint_correct")),
        "n_update": sum(1 for r in recs if r.get("update_correct")),
        "n_retain": sum(1 for r in recs if r.get("retain_correct")),
        "beta": stats([float(r["beta"]) for r in recs]),
        "alpha": stats([float(r["alpha"]) for r in recs]),
        "abs_alpha": stats([abs(float(r["alpha"])) for r in recs]),
        "gamma": stats([float(r["gamma"]) for r in recs]),
        "mean_U": (sum(float(r["U"]) for r in recs) / len(recs)) if recs else None,
        "mean_R": (sum(float(r["R"]) for r in recs) / len(recs)) if recs else None,
        "records": recs,
    }
    return out


def parse_step033() -> Dict[str, Any]:
    paths = {
        "answer_only_file": DATA / "recipient_only_balanced_answer_only_offset7_e500" / "recipient_only_binding_repair_summary.json",
        "bg15_file": DATA / "recipient_only_balanced_offset7_bg15_e500" / "recipient_only_binding_repair_summary.json",
    }
    out = {}
    for tag, path in paths.items():
        if not path.exists():
            continue
        data = read_json(path)
        case = {
            "baseline_train": per_group_records_from_template_eval(data["baseline_train_eval"], f"{tag}/baseline_train"),
            "baseline_held": per_group_records_from_template_eval(data["baseline_held_eval"], f"{tag}/baseline_held"),
            "arms": {},
        }
        for arm in data.get("arms", []):
            name = arm.get("arm_name", "arm")
            case["arms"][name] = {
                "final_train": per_group_records_from_template_eval(arm["final_train_eval"], f"{tag}/{name}/final_train"),
                "final_held": per_group_records_from_template_eval(arm["final_held_eval"], f"{tag}/{name}/final_held"),
                "mode": arm.get("mode"),
                "mask_prob": arm.get("mask_prob"),
                "target_stats_totals": arm.get("target_stats_totals", {}),
            }
        out[tag] = case
    return out


def parse_step036_template() -> Dict[str, Any]:
    path = DATA / "paired_context_template" / "paired_context_full_results.json"
    data = read_json(path)
    rows = []
    base_h = data["baseline_held"]
    rows.append(aggregate_record(
        "template/baseline_held",
        base_h["mean_update_new_minus_source"], base_h["mean_retain_source_minus_new"],
        n=base_h.get("n_groups"), joint=base_h.get("n_recipient_flip_correct"),
        extra={"split": "held", "arm": "baseline", "epoch": 0},
    ))
    trajectories = {}
    for arm_name, arm in data["arms"].items():
        traj_out = []
        for p in arm.get("trajectory", []):
            rec = aggregate_record(
                f"template/{arm_name}/held/e{p['epoch']}",
                p["held_U"], p["held_R"], n=p.get("held_n"), joint=p.get("held_flips"),
                extra={"split": "held", "arm": arm_name, "epoch": p["epoch"],
                       "train_beta": p["train_recip_dep"], "train_alpha": p["train_shared_pref"],
                       "train_gamma_from_means": p["train_recip_dep"] - abs(p["train_shared_pref"]),
                       "train_joint": p.get("train_flips"), "train_n": p.get("train_n")},
            )
            rows.append(rec); traj_out.append(rec)
        trajectories[arm_name] = traj_out
    return {"aggregate_rows": rows, "trajectories": trajectories,
            "note": "research did not save per-group final scores; gamma distributions require future reruns that preserve per-group records."}


def parse_step036_natural_partial() -> Dict[str, Any]:
    path = DATA / "natural_ce_only_extracted.json"
    if not path.exists():
        return {}
    data = read_json(path)
    rows = []
    trajectories = {}
    for key, name in [("ce_only_trajectory", "ce_only"), ("combined_partial_trajectory", "combined_partial")]:
        traj = []
        for p in data.get(key, []):
            rec = aggregate_record(
                f"natural_partial/{name}/held/e{p['epoch']}",
                p["he_U"], p["he_R"], n=data.get("n_held"), joint=p.get("he_j"),
                extra={"split": "held", "arm": name, "epoch": p["epoch"],
                       "train_beta": p["tr_dep"], "train_alpha": p["tr_sh"],
                       "train_gamma_from_means": p["tr_dep"] - abs(p["tr_sh"]),
                       "train_joint": p.get("tr_j"), "train_n": data.get("n_train")},
            )
            rows.append(rec); traj.append(rec)
        trajectories[name] = traj
    return {"aggregate_rows": rows, "trajectories": trajectories,
            "note": "extracted from timed-out stdout; combined_partial stops at epoch 60."}


def parse_natural_baseline_pair_scores() -> Dict[str, Any]:
    path = DATA / "multitoken_scorer_pilot512" / "pair_scores.jsonl"
    rows = read_jsonl(path)
    recs = []
    for r in rows:
        U = float(r["U_mean_new_minus_source"])
        R = float(r["R_mean_source_minus_new"])
        beta, alpha, gamma = beta_alpha_gamma(U, R)
        recs.append({
            "id": r.get("pair_id"),
            "U": U, "R": R, "beta": beta, "alpha": alpha, "gamma": gamma,
            "joint_correct": bool(r.get("joint_correct_mean")),
            "update_correct": bool(r.get("update_correct_mean")),
            "retain_correct": bool(r.get("retain_correct_mean")),
            "same_token_length_both_rows": bool(r.get("same_token_length_both_rows")),
            "role_position_relation": r.get("role_position_relation"),
            "update_answer_n_tokens": r.get("update_answer_n_tokens"),
            "update_foil_n_tokens": r.get("update_foil_n_tokens"),
            "retain_answer_n_tokens": r.get("retain_answer_n_tokens"),
            "retain_foil_n_tokens": r.get("retain_foil_n_tokens"),
            "answer_also_elsewhere_update": bool(r.get("answer_also_elsewhere_update")),
            "answer_also_elsewhere_retain": bool(r.get("answer_also_elsewhere_retain")),
        })
    overall = summarise_records("a02_natural_baseline/all55", recs)
    subsets = {}
    subset_defs = {
        "same_token_length": lambda r: r["same_token_length_both_rows"],
        "unequal_token_length": lambda r: not r["same_token_length_both_rows"],
        "target_earlier": lambda r: r["role_position_relation"] == "target_earlier",
        "target_later": lambda r: r["role_position_relation"] == "target_later",
        "answer_not_elsewhere_both": lambda r: (not r["answer_also_elsewhere_update"]) and (not r["answer_also_elsewhere_retain"]),
    }
    for name, pred in subset_defs.items():
        subsets[name] = summarise_records(f"a02_natural_baseline/{name}", [r for r in recs if pred(r)])
    return {"overall": overall, "subsets": subsets}


def plot_loss_geometry(fig_path: pathlib.Path):
    xs = [(-6 + i * 0.12) for i in range(101)]
    ys = [(-1 + i * 0.08) for i in range(126)]  # beta
    Z_ce, Z_pair = [], []
    for beta in ys:
        row_ce, row_pair = [], []
        for alpha in xs:
            U = alpha + beta
            R = beta - alpha
            row_ce.append(binary_ce_from_margins(U, R))
            row_pair.append(math.log1p(math.exp(-2 * beta)))
        Z_ce.append(row_ce); Z_pair.append(row_pair)
    import numpy as np
    X, Y = np.meshgrid(xs, ys)
    Z_ce = np.array(Z_ce); Z_pair = np.array(Z_pair)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, Z, title in [(axes[0], Z_ce, "balanced answer CE"), (axes[1], Z_pair, "paired beta loss")]:
        levels = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
        cs = ax.contour(X, Y, Z, levels=levels, linewidths=0.9)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")
        ax.fill_between(xs, [abs(x) for x in xs], [max(ys)] * len(xs), color="#d5f5d5", alpha=0.35,
                        label="joint-correct region: beta>|alpha|")
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("alpha=(U-R)/2 shared preference")
        ax.set_ylabel("beta=(U+R)/2 recipient component")
        ax.set_title(title)
        ax.set_ylim(min(ys), max(ys)); ax.set_xlim(min(xs), max(xs))
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("CE rewards beta>|alpha|; paired loss raises beta but does not reduce |alpha|")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def plot_trajectory(trajectories: Dict[str, List[Dict[str, Any]]], fig_path: pathlib.Path, title: str):
    if not trajectories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    colors = {"ce_only": "#1f77b4", "combined": "#d62728", "paired_only": "#2ca02c", "combined_partial": "#ff7f0e"}
    for arm, traj in trajectories.items():
        if not traj:
            continue
        epochs = [r["epoch"] for r in traj]
        beta = [r["beta"] for r in traj]
        absalpha = [abs(r["alpha"]) for r in traj]
        gamma = [r["gamma_from_means"] for r in traj]
        joint = [r["joint_frac"] if r["joint_frac"] is not None else float("nan") for r in traj]
        c = colors.get(arm, None)
        axes[0].plot(epochs, beta, marker="o", color=c, label=f"{arm} beta")
        axes[0].plot(epochs, absalpha, marker="x", linestyle="--", color=c, label=f"{arm} |alpha|")
        axes[1].plot(epochs, gamma, marker="o", color=c, label=f"{arm} gamma")
        axes[1].plot(epochs, joint, marker="s", linestyle="--", color=c, alpha=0.65, label=f"{arm} joint frac")
    axes[0].set_title("held beta vs shared-preference magnitude")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("margin component")
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].legend(fontsize=7)
    axes[1].set_title("held gamma=beta-|alpha| and joint correctness")
    axes[1].set_xlabel("epoch"); axes[1].axhline(0, color="black", lw=0.6)
    axes[1].legend(fontsize=7)
    fig.suptitle(title)
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def plot_distribution(summary: Dict[str, Any], fig_path: pathlib.Path, title: str):
    recs = summary.get("records", [])
    if not recs:
        return
    beta = [r["beta"] for r in recs]
    alpha = [r["alpha"] for r in recs]
    gamma = [r["gamma"] for r in recs]
    joint = [r["joint_correct"] for r in recs]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for ok in [False, True]:
        xs = [alpha[i] for i, b in enumerate(joint) if b == ok]
        ys = [beta[i] for i, b in enumerate(joint) if b == ok]
        axes[0].scatter(xs, ys, s=35, alpha=0.75, label="joint" if ok else "not joint")
    xlim = max(1.0, max(abs(x) for x in alpha) * 1.05)
    xs = [(-xlim + 2*xlim*i/200) for i in range(201)]
    axes[0].plot(xs, [abs(x) for x in xs], color="black", lw=1, label="beta=|alpha|")
    axes[0].axhline(0, color="black", lw=0.5); axes[0].axvline(0, color="black", lw=0.5)
    axes[0].set_xlabel("alpha"); axes[0].set_ylabel("beta")
    axes[0].set_title("pair-level components")
    axes[0].legend(fontsize=8)
    axes[1].hist(gamma, bins=20, color="#4477aa", alpha=0.8)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel("gamma=beta-|alpha|"); axes[1].set_ylabel("pair count")
    axes[1].set_title("positive gamma iff both answers are correct")
    fig.suptitle(title)
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {"status": "BETA_ALPHA_READOUT"}
    result["definitions"] = {
        "U": "new-source margin in target-updated context",
        "R": "source-new margin in distractor-updated context",
        "beta": "(U+R)/2 recipient-dependent component",
        "alpha": "(U-R)/2 context-shared new-vs-source preference",
        "gamma": "beta-|alpha|; pair-level joint correctness iff gamma>0",
        "ce_correction": "Balanced ordinary answer CE already favors U>0 and R>0; the paired beta loss may reshape optimization but does not add a label distinction absent from CE.",
    }

    # Existing studies.
    result["template_per_group"] = parse_step033()
    result["template_aggregate"] = parse_step036_template()
    result["natural_partial_aggregate"] = parse_step036_natural_partial()
    result["natural_step058_baseline_per_pair"] = parse_natural_baseline_pair_scores()

    # Figures.
    plot_loss_geometry(FIGS / "loss_geometry_beta_alpha.png")
    plot_trajectory(result["template_aggregate"].get("trajectories", {}),
                    FIGS / "template_held_beta_alpha_trajectories.png",
                    "research template held trajectory: aggregate means")
    plot_trajectory(result["natural_partial_aggregate"].get("trajectories", {}),
                    FIGS / "natural_held_beta_alpha_partial.png",
                    "research natural-row held trajectory: partial combined run")
    plot_distribution(result["natural_step058_baseline_per_pair"]["overall"],
                      FIGS / "natural_baseline_pair_distribution.png",
                      "research natural-row baseline: pair-level beta/alpha")

    # Concise derived table for the most relevant existing endpoint rows.
    table = []
    s36 = result["template_aggregate"]["aggregate_rows"]
    for r in s36:
        if r.get("epoch") in (0, 500):
            table.append({k: r.get(k) for k in ["name", "U", "R", "beta", "alpha", "gamma_from_means", "joint", "n", "joint_frac"]})
    nat = result["natural_partial_aggregate"].get("aggregate_rows", [])
    for r in nat:
        if r.get("epoch") in (0, 60, 200):
            table.append({k: r.get(k) for k in ["name", "U", "R", "beta", "alpha", "gamma_from_means", "joint", "n", "joint_frac"]})
    result["endpoint_mean_table"] = table

    # Natural subset table.
    subset_table = []
    for name, summ in result["natural_step058_baseline_per_pair"]["subsets"].items():
        subset_table.append({
            "subset": name,
            "n": summ["n"],
            "joint": summ["n_joint"],
            "mean_U": summ["mean_U"],
            "mean_R": summ["mean_R"],
            "mean_beta": summ["beta"].get("mean"),
            "mean_abs_alpha": summ["abs_alpha"].get("mean"),
            "mean_gamma": summ["gamma"].get("mean"),
            "n_gamma_pos": summ["gamma"].get("n_pos"),
        })
    result["natural_baseline_subset_table"] = subset_table

    out_json = OUT / "beta_alpha_readout.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    # Markdown summary.
    lines = []
    lines.append("# research beta/alpha readout for recipient-sensitive selection")
    lines.append("")
    lines.append("## Corrected geometry")
    lines.append("")
    lines.append("For every recipient pair, define `U` as the new-over-source margin in the target-updated context and `R` as the source-over-new margin in the distractor-updated context. Let `beta=(U+R)/2`, `alpha=(U-R)/2`, and `gamma=beta-|alpha|`. Both answers are correct exactly when `gamma>0`.")
    lines.append("")
    lines.append("Balanced answer CE already rewards the same contextual distinction because its binary-candidate form is `softplus(-U)+softplus(-R)`. It is therefore wrong to say that CE is indifferent to shared preference and recipient-sensitive selection. The paired loss `softplus(-2 beta)` can increase direct pressure on `beta`, but it does not reduce `|alpha|`; a run with higher mean beta can still fail RETAIN if shared preference remains large.")
    lines.append("")
    lines.append("Figures saved:")
    lines.append("- `figures/loss_geometry_beta_alpha.png`")
    lines.append("- `figures/template_held_beta_alpha_trajectories.png`")
    lines.append("- `figures/natural_held_beta_alpha_partial.png`")
    lines.append("- `figures/natural_baseline_pair_distribution.png`")
    lines.append("")
    lines.append("## Existing endpoint readout from available aggregate outputs")
    lines.append("")
    lines.append("| Source | joint | U | R | beta | alpha | gamma from means |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in table:
        joint = "" if r.get("joint") is None else f"{r['joint']}/{r['n']}"
        lines.append(f"| {r['name']} | {joint} | {r['U']:+.3f} | {r['R']:+.3f} | {r['beta']:+.3f} | {r['alpha']:+.3f} | {r['gamma_from_means']:+.3f} |")
    lines.append("")
    lines.append("The completed research template run illustrates the corrected reading: CE and combined both reached 3/12 held joint answers. Combined raised held beta only slightly relative to CE (+2.099 vs +1.977), while |alpha| stayed much larger than beta (6.614 vs 6.720 for CE); the negative gamma from means explains the persistent RETAIN failure.")
    lines.append("")
    lines.append("## Natural research baseline pair distribution")
    overall = result["natural_step058_baseline_per_pair"]["overall"]
    lines.append(f"All 55 pilot pairs: joint {overall['n_joint']}/{overall['n']}, mean beta={overall['beta']['mean']:+.3f}, mean |alpha|={overall['abs_alpha']['mean']:+.3f}, mean gamma={overall['gamma']['mean']:+.3f}, gamma-positive pairs={overall['gamma']['n_pos']}/{overall['n']}.")
    lines.append("")
    lines.append("| Subset | n | joint | mean beta | mean |alpha| | mean gamma | gamma>0 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in subset_table:
        lines.append(f"| {s['subset']} | {s['n']} | {s['joint']}/{s['n']} | {s['mean_beta']:+.3f} | {s['mean_abs_alpha']:+.3f} | {s['mean_gamma']:+.3f} | {s['n_gamma_pos']}/{s['n']} |")
    lines.append("")
    lines.append("The research pilot rows start with a strong shared candidate preference and weak pair-level gamma. Equal token length alone does not solve this, and role-position subsets remain weak. These rows are useful for pipeline testing but are not yet strong evidence about scalable ALN-preserving training.")
    lines.append("")
    lines.append("## Implication for pending and future runs")
    lines.append("")
    lines.append("The running multi-seed and natural combined experiments should be read by joint correctness and pair-level gamma whenever those records are available. A higher average beta is useful only if beta overtakes |alpha| on held pairs and both UPDATE and RETAIN are correct. If the additional paired term does not accomplish that, it means this objective did not solve transferable selection on the tested substrate; it does not decide among experience support, row validity, optimization, or representation as the deeper cause.")

    out_md = (OUT.parents[4] / 'research/documents/functional_learning/data/beta_alpha_readout/beta_alpha_readout.md')
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "figures": [
            str(FIGS / "loss_geometry_beta_alpha.png"),
            str(FIGS / "template_held_beta_alpha_trajectories.png"),
            str(FIGS / "natural_held_beta_alpha_partial.png"),
            str(FIGS / "natural_baseline_pair_distribution.png"),
        ],
        "natural_all55_joint": f"{overall['n_joint']}/{overall['n']}",
        "natural_all55_mean_beta": overall["beta"].get("mean"),
        "natural_all55_mean_abs_alpha": overall["abs_alpha"].get("mean"),
        "natural_all55_mean_gamma": overall["gamma"].get("mean"),
    }, indent=2))


if __name__ == "__main__":
    main()
