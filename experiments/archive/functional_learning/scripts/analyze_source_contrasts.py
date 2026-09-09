#!/usr/bin/env python3
"""Analyze research source-specific contrasts and write a synthesis note."""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STUDY = _public_path('experiments/archive/functional_learning')
DATA_DIRS = {
    "overlap_same_attr": _public_path('experiments/archive/functional_learning/data/matched_support_s42_43_100_e300'),
    "nonoverlap_rewrite": _public_path('experiments/archive/functional_learning/data/revision_008b_nonoverlap_s42_43_100_e300'),
}
OUT_DIR = _public_path('experiments/archive/functional_learning/data/source_contrast_synthesis')
FIG_DIR = _public_path('experiments/archive/functional_learning/figures')
NOTE_PATH = _public_path('research/notes/functional_learning/matched_support_source_contrast_result.md')


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def final_row(summary: Dict, cond: str) -> Dict:
    rows = summary["aggregate"]["conditions"]
    return {k: v["mean"] for k, v in rows[cond].items() if isinstance(v, dict) and "mean" in v}


def agg_contrast(summary: Dict, A: str, B: str, relation: str, u_kind: str = "nomatch") -> Dict:
    label = f"{A}|{B}|{relation}|{u_kind}"
    d = summary["aggregate"]["contrasts"][label]
    return {k: {"mean": v["mean"], "sd": v["sd"], "values": v["values"]} for k, v in d.items()}


def fmt_ms(d: Dict, key: str) -> str:
    return f"{d[key]['mean']:+.3f}±{d[key]['sd']:.3f}"


def load_history(root: Path, seed: int, cond: str) -> List[Dict]:
    p = root / f"seed{seed}" / cond / "result.json"
    return load_json(p)["history"]


def scan_curve(root: Path, seeds: List[int], A: str, B: str, relation: str = "content") -> Dict:
    """Return per-seed and aggregate excess-true-cost curve summary."""
    per_seed = []
    for seed in seeds:
        ha = load_history(root, seed, A)
        hb = load_history(root, seed, B)
        byb = {r["epoch"]: r for r in hb}
        vals = []
        for ra in ha:
            ep = ra["epoch"]
            if ep not in byb:
                continue
            rb = byb[ep]
            T = ra[f"{relation}_T_nll"] - rb[f"{relation}_T_nll"]
            U = ra[f"{relation}_U_nomatch_nll"] - rb[f"{relation}_U_nomatch_nll"]
            excess = T - U
            vals.append({"epoch": ep, "T_delta": T, "U_delta": U, "excess_true_cost": excess, "gain_delta": -excess})
        per_seed.append({
            "seed": seed,
            "final": vals[-1],
            "max_excess": max(vals, key=lambda x: x["excess_true_cost"]),
            "min_excess": min(vals, key=lambda x: x["excess_true_cost"]),
            "curve": vals,
        })
    final_vals = [x["final"]["excess_true_cost"] for x in per_seed]
    max_vals = [x["max_excess"]["excess_true_cost"] for x in per_seed]
    return {
        "A": A,
        "B": B,
        "relation": relation,
        "per_seed": per_seed,
        "final_mean": sum(final_vals) / len(final_vals),
        "final_sd": 0.0 if len(final_vals) == 1 else math.sqrt(sum((v - sum(final_vals)/len(final_vals))**2 for v in final_vals)/(len(final_vals)-1)),
        "max_excess_any_seed": max(max_vals),
    }


def make_figure(summaries: Dict[str, Dict], out_path: Path):
    cases = [
        ("overlap\nR-full − support", "overlap_same_attr", "repeat_full", "support_control", "content"),
        ("nonoverlap\nR-full − support", "nonoverlap_rewrite", "repeat_full", "support_control", "content"),
        ("nonoverlap\nV-full − support", "nonoverlap_rewrite", "varied_full", "support_control", "content"),
        ("nonoverlap\nR-full − V-full", "nonoverlap_rewrite", "repeat_full", "varied_full", "content"),
        ("nonoverlap\nR-full − R-masked", "nonoverlap_rewrite", "repeat_full", "repeat_masked", "content"),
    ]
    x = list(range(len(cases)))
    T, U, E, Terr, Uerr, Eerr = [], [], [], [], [], []
    for _, key, A, B, rel in cases:
        c = agg_contrast(summaries[key], A, B, rel)
        T.append(c["T_delta_A_minus_B"]["mean"])
        U.append(c["U_delta_A_minus_B"]["mean"])
        E.append(c["excess_true_cost"]["mean"])
        Terr.append(c["T_delta_A_minus_B"]["sd"])
        Uerr.append(c["U_delta_A_minus_B"]["sd"])
        Eerr.append(c["excess_true_cost"]["sd"])
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axhline(0, color="black", lw=0.8)
    ax.bar([i - width for i in x], T, width, yerr=Terr, label="T delta", color="#4C78A8", alpha=0.85, capsize=3)
    ax.bar(x, U, width, yerr=Uerr, label="U delta", color="#F58518", alpha=0.85, capsize=3)
    ax.bar([i + width for i in x], E, width, yerr=Eerr, label="excess T cost", color="#54A24B", alpha=0.85, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cases], rotation=0)
    ax.set_ylabel("NLL contrast (nats); mean ± sd over 3 seeds")
    ax.set_title("research source-specific decomposition: absolute harm versus true-source-specific cost")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {k: load_json(v / "summary.json") for k, v in DATA_DIRS.items()}
    seeds = summaries["nonoverlap_rewrite"]["args"]["seeds"]

    scans = {}
    for name, root in DATA_DIRS.items():
        scans[name] = {}
        for A, B in (("repeat_full", "support_control"), ("repeat_full", "unique"), ("repeat_full", "repeat_masked"), ("repeat_full", "varied_full"), ("varied_full", "support_control"), ("varied_full", "varied_masked")):
            scans[name][f"{A}_minus_{B}"] = scan_curve(root, seeds, A, B, relation="content")

    fig_path = _public_path('experiments/archive/functional_learning/figures/source_decomposition.png')
    make_figure(summaries, fig_path)

    out_summary = {
        "status": "SOURCE_CONTRAST_SYNTHESIS_DONE",
        "inputs": {k: str(v.relative_to(STUDY)) for k, v in DATA_DIRS.items()},
        "figure": str(fig_path.relative_to(STUDY)),
        "key_aggregate_contrasts": {},
        "curve_scans": scans,
    }
    for key, summary in summaries.items():
        out_summary["key_aggregate_contrasts"][key] = {
            "repeat_full_minus_support_content": agg_contrast(summary, "repeat_full", "support_control", "content"),
            "repeat_full_minus_masked_content": agg_contrast(summary, "repeat_full", "repeat_masked", "content"),
            "varied_full_minus_support_content": agg_contrast(summary, "varied_full", "support_control", "content"),
            "repeat_full_minus_varied_full_content": agg_contrast(summary, "repeat_full", "varied_full", "content"),
        }
    (_public_path('experiments/archive/functional_learning/data/source_contrast_synthesis/source_contrast_synthesis.json')).write_text(json.dumps(out_summary, indent=2), encoding="utf-8")

    ov = summaries["overlap_same_attr"]
    no = summaries["nonoverlap_rewrite"]
    ov_r_s = agg_contrast(ov, "repeat_full", "support_control", "content")
    no_r_s = agg_contrast(no, "repeat_full", "support_control", "content")
    no_r_m = agg_contrast(no, "repeat_full", "repeat_masked", "content")
    no_v_s = agg_contrast(no, "varied_full", "support_control", "content")
    no_r_v = agg_contrast(no, "repeat_full", "varied_full", "content")
    no_r_u = agg_contrast(no, "repeat_full", "unique", "content")
    no_w = agg_contrast(no, "wrong_full", "support_control", "content") if "wrong_full|support_control|content|nomatch" in no["aggregate"]["contrasts"] else None

    lines = []
    lines.append("# research result: source-specific decomposition of the synthetic bridge\n")
    lines.append("## Why this step was needed\n")
    lines.append("research v2 showed that exact repetition could create large absolute damage on the unpracticed content/restatement probe, but the subsequent audit showed that the source-specific decomposition did not match the BabyLM relation-learning term: in research the unrelated-source/no-match control was damaged even more than the true-source probe, so the model still used the true source better rather than showing an excess true-source cost. research therefore repaired train/probe support before any multi-operation extension.\n")
    lines.append("## Experiments executed\n")
    lines.append("1. `matched_support_source_contrast.py`: same attribute token for copy and content templates, but train and probes share source-slot distribution, target-slot distribution, and one matched plus one unmatched final event per relation-arm sequence. `repeat_full`/`repeat_masked` and `varied_full`/`varied_masked` are paired-token-identical; the mask blocks only target-event attention to its source event during training.\n")
    lines.append("2. `revision_008b_nonoverlap_source_contrast.py`: same support and attention intervention, but the content/rewrite target uses a disjoint token `r_i` while the source contains `s_i`. Exact repetition trains `s_i→s_i`; varied restatement trains `s_i→r_i`. This is the closer analogue of relation_learning's tokenizer-nonoverlap rewrite probe.\n")
    lines.append("## Main numerical results\n")
    lines.append("Positive excess true-source cost is `(T_A-T_B)-(U_A-U_B)`, so it means A is worse on true-source use beyond its unrelated-source/no-match change.\n")
    lines.append("| experiment | contrast | T delta | U delta | excess true-source cost | gain delta | reading |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    lines.append(f"| same-token content | R_full − support | {fmt_ms(ov_r_s, 'T_delta_A_minus_B')} | {fmt_ms(ov_r_s, 'U_delta_A_minus_B')} | {fmt_ms(ov_r_s, 'excess_true_cost')} | {fmt_ms(ov_r_s, 'gain_delta_A_minus_B')} | true source helps more than unrelated control; not BabyLM-like recurrence cost |")
    lines.append(f"| nonoverlap rewrite | R_full − support | {fmt_ms(no_r_s, 'T_delta_A_minus_B')} | {fmt_ms(no_r_s, 'U_delta_A_minus_B')} | {fmt_ms(no_r_s, 'excess_true_cost')} | {fmt_ms(no_r_s, 'gain_delta_A_minus_B')} | large absolute rewrite harm, but T and U move together; not source-specific |")
    lines.append(f"| nonoverlap rewrite | R_full − R_masked | {fmt_ms(no_r_m, 'T_delta_A_minus_B')} | {fmt_ms(no_r_m, 'U_delta_A_minus_B')} | {fmt_ms(no_r_m, 'excess_true_cost')} | {fmt_ms(no_r_m, 'gain_delta_A_minus_B')} | attention access changes absolute NLL but not selective true-source cost |")
    lines.append(f"| nonoverlap rewrite | V_full − support | {fmt_ms(no_v_s, 'T_delta_A_minus_B')} | {fmt_ms(no_v_s, 'U_delta_A_minus_B')} | {fmt_ms(no_v_s, 'excess_true_cost')} | {fmt_ms(no_v_s, 'gain_delta_A_minus_B')} | varied restatement gives strong true-source rewrite use |")
    lines.append(f"| nonoverlap rewrite | R_full − V_full | {fmt_ms(no_r_v, 'T_delta_A_minus_B')} | {fmt_ms(no_r_v, 'U_delta_A_minus_B')} | {fmt_ms(no_r_v, 'excess_true_cost')} | {fmt_ms(no_r_v, 'gain_delta_A_minus_B')} | R is much worse than V on true-source rewrite; this matches V−R ordering but not R−C |")
    lines.append("\n## What changed scientifically\n")
    lines.append("The repaired support tests do not support treating research's absolute content damage as the same source-specific recurrence cost measured by relation_learning. In the same-token version, exact repetition actually improves source-conditioned content relative to the unrelated control (R_full − support excess ≈ −0.34). This exposed that the earlier content target could be solved by copying the source attribute token.\n")
    lines.append("The nonoverlap version removes that shortcut. Exact repetition then causes large absolute rewrite-token damage relative to clean/support controls: R_full − support has T≈+1.83 nats and U≈+1.88 nats. But the excess true-source cost remains near zero and slightly negative (≈−0.054±0.045). Thus the source is not being used selectively worse than the unrelated control; both true-source and no-match rewrite targets are broadly disfavored.\n")
    lines.append("The paired attention intervention remains important but its meaning is narrower. In nonoverlap rewrite, R_full − R_masked shifts content T and U by similarly large amounts (T≈+1.35, U≈+1.38, excess≈−0.026), so within-context attention access can install a broad final-slot/target-space expectation that damages rewrite tokens, but it does not by itself produce the BabyLM-like positive excess true-source cost. For varied restatement, attention access is necessary for strong rewrite use: V_full − V_masked has excess≈−2.03 and content gain≈+2.03.\n")
    lines.append("The most robust synthetic result after repair is therefore relational support specificity rather than direct BabyLM recurrence-cost identity: finite experience installs the relation it demonstrates. Repetition installs `same entity → same source token` and varied restatement installs `same entity + source token → disjoint rewrite token`. When evaluation asks for the other relation, damage can be large, but the T/U decomposition distinguishes broad target-prior damage from source-specific content competition.\n")
    lines.append("## Learning-curve scan\n")
    lines.append("Across stored eval epochs, the nonoverlap R_full − support content excess did not show a large hidden positive phase. Final per-seed excesses are roughly −0.011, −0.101, and −0.051. R_full − R_masked final excesses are roughly −0.007, −0.066, and −0.006. Seed-level curves occasionally fluctuate, but the repeated pattern is not a +0.4 to +1.0 nat positive excess comparable to the BabyLM DeBERTa term. The exact curve values are saved in `data/source_contrast_synthesis/source_contrast_synthesis.json`.\n")
    lines.append("## Implication for the next research step\n")
    lines.append("Do not extend this exact substrate to Entity-depth state tracking as though the BabyLM source-specific mechanism has already been reproduced. A depth extension could still test how relation-specific training affects multi-update tracking, but the bridge to the BabyLM nonoverlap R−C cost needs a sharper substrate first: the clean/control arm must share broad final-slot and target-token distribution with repetition while lacking only the repeated identity relation, and the probe must preserve disjoint target tokens while preventing broad rewrite-token suppression from masquerading as source-use impairment. A useful next construction is a counterbalanced dual-relation corpus where each arm sees the same marginal rates of copy tokens and rewrite tokens at matched and unmatched final slots, while only the joint source→target pairing differs (identity-paired vs rewrite-paired vs independent-paired).\n")
    lines.append("## Files\n")
    lines.append("- overlap script: `scripts/matched_support_source_contrast.py`\n")
    lines.append("- overlap data: `data/matched_support_s42_43_100_e300/`\n")
    lines.append("- nonoverlap script: `scripts/revision_008b_nonoverlap_source_contrast.py`\n")
    lines.append("- nonoverlap data: `data/revision_008b_nonoverlap_s42_43_100_e300/`\n")
    lines.append(f"- synthesis data: `data/source_contrast_synthesis/source_contrast_synthesis.json`\n")
    lines.append(f"- figure: `figures/{fig_path.name}`\n")
    NOTE_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "SOURCE_CONTRAST_SYNTHESIS_DONE",
        "note": str(NOTE_PATH.relative_to(STUDY)),
        "figure": str(fig_path.relative_to(STUDY)),
        "summary": str((_public_path('experiments/archive/functional_learning/data/source_contrast_synthesis/source_contrast_synthesis.json')).relative_to(STUDY)),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
