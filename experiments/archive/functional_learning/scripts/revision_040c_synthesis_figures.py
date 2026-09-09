#!/usr/bin/env python3
"""Step040c: synthesize repaired relation-first acquisition evidence and plot trajectories."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
from typing import Any, Dict, List

import matplotlib.pyplot as plt

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
E80 = _public_path('experiments/archive/functional_learning/data/relation_first_repaired_e80/answer_only_private_e80_seed40040/training_summary.json')
PARENT = _public_path('experiments/archive/functional_learning/data/relation_first_repaired/trusted_parent_summary.json')
BG = _public_path('experiments/archive/functional_learning/data/revision_040b_repaired_bg_comparison/bg_comparison_summary.json')
FIG_DIR = _public_path('experiments/archive/functional_learning/figures')
NOTE = _public_path('research/notes/functional_learning/repaired_relation_first_acquisition_synthesis.md')
OUT_JSON = _public_path('experiments/archive/functional_learning/data/repaired_relation_first_synthesis/synthesis.json')
_public_path('experiments/archive/functional_learning/data/repaired_relation_first_synthesis').mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
_public_path('research/notes/functional_learning').mkdir(parents=True, exist_ok=True)


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load(p: pathlib.Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def traj_xy(traj: List[Dict[str, Any]], split: str, key: str):
    xs, ys = [], []
    for e in traj:
        xs.append(e["epoch"])
        ys.append(e[split][key])
    return xs, ys


def main() -> None:
    e80 = load(E80)
    parent = load(PARENT)
    bg = load(BG)

    # Figure 1: held success trajectories.
    plt.figure(figsize=(8.5, 5.2))
    xs, ys = traj_xy(e80["trajectory"], "held", "n_four_condition_success")
    plt.plot(xs, ys, marker="o", linewidth=2.2, label="answer-only, no background corruption")
    for arm in bg["arms"]:
        xs, ys = traj_xy(arm["trajectory"], "held", "n_four_condition_success")
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=arm["arm"].replace("_", " "))
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.35)
    plt.xlabel("epoch")
    plt.ylabel("held four-condition success (of 30 pairs)")
    plt.title("Repaired relation-first held acquisition under trusted coherent86 private adapters")
    plt.legend(frameon=False)
    plt.tight_layout()
    fig1 = _public_path('experiments/archive/functional_learning/figures/repaired_relation_first_held_success.png')
    plt.savefig(fig1, dpi=180)
    plt.close()

    # Figure 2: min-four signed margin trajectories.
    plt.figure(figsize=(8.5, 5.2))
    xs, ys = traj_xy(e80["trajectory"], "held", "mean_min_four_signed_margin")
    plt.plot(xs, ys, marker="o", linewidth=2.2, label="answer-only, no background corruption")
    for arm in bg["arms"]:
        xs, ys = traj_xy(arm["trajectory"], "held", "mean_min_four_signed_margin")
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=arm["arm"].replace("_", " "))
    plt.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    plt.xlabel("epoch")
    plt.ylabel("held mean min of four signed margins")
    plt.title("Strict success margin separates full relation use from shared new-phrase preference")
    plt.legend(frameon=False)
    plt.tight_layout()
    fig2 = _public_path('experiments/archive/functional_learning/figures/repaired_relation_first_min4_margin.png')
    plt.savefig(fig2, dpi=180)
    plt.close()

    # Figure 3: beta vs alpha trajectory for no-corruption held.
    plt.figure(figsize=(6.4, 5.8))
    xs = [e["held"]["mean_beta_pair_average"] for e in e80["trajectory"]]
    ys = [e["held"]["mean_abs_alpha_pair_average"] for e in e80["trajectory"]]
    epochs = [e["epoch"] for e in e80["trajectory"]]
    plt.plot(xs, ys, marker="o", linewidth=2.0)
    for x, y, ep in zip(xs, ys, epochs):
        plt.text(x, y, str(ep), fontsize=8, ha="left", va="bottom")
    mx = max(max(xs), max(ys)) + 0.5
    plt.plot([0, mx], [0, mx], linestyle="--", color="gray", linewidth=1.0, label="beta = |alpha|")
    plt.xlabel("held mean beta (recipient-dependent component)")
    plt.ylabel("held mean |alpha| (shared preference magnitude)")
    plt.title("Answer-only training moves repaired held pairs across the beta>|alpha| region")
    plt.legend(frameon=False)
    plt.tight_layout()
    fig3 = _public_path('experiments/archive/functional_learning/figures/repaired_relation_first_beta_alpha_traj.png')
    plt.savefig(fig3, dpi=180)
    plt.close()

    no_corrupt_final = e80["final_held"]
    bg_arm = {a["arm"]: a for a in bg["arms"]}
    result = {
        "status": "REPAIRED_RELATION_FIRST_SYNTHESIS",
        "parent_held": parent["held"],
        "answer_only_no_corruption_final_held": no_corrupt_final,
        "corrupted_answer_only_final_held": bg_arm["corrupted_answer_only"]["final_held"],
        "corrupted_answer_plus_bg_final_held": bg_arm["corrupted_answer_plus_bg"]["final_held"],
        "figures": [rel(fig1), rel(fig2), rel(fig3)],
        "key_interpretation": "Trusted coherent86 parent shows shared new-phrase preference but zero recipient-conditioned selection; repaired private-adapter answer-only training without context corruption installs held recipient selection; adding background input corruption greatly slows acquisition, with background labels adding a smaller further deficit at e60.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# research repaired relation-first acquisition synthesis\n\n")
    lines.append("## What was corrected\n\n")
    lines.append("The research relation-first extraction is preserved as a semantic improvement, but its first scorer/trainer is not usable evidence for coherent86 acquisition. A model-identity audit showed that generic `AutoModelForMaskedLM` loading without trusted custom loading instantiates stock `DebertaV2ForMaskedLM` with 0 private-adapter parameters and ignores adapter tensors. The repaired harness uses the established `FrozenSlowPrivateDebertaV2ForMaskedLM` loader, verifies 995,584 private-adapter parameters and executed scale 0.75 in all 8 layers, and freezes every non-private parameter for training.\n\n")
    lines.append("The construction was also repaired: each pair now has one shared replacement value, so for a fixed query the `update A` versus `update B` contrast changes only the update recipient. Scoring uses the explicit final `{STATE}` span with all candidate tokens masked simultaneously and no fallback search. Four-condition success is the minimum over the four individual signed margins, not gamma after averaging.\n\n")
    lines.append("## Trusted parent baseline\n\n")
    ph = parent["held"]
    lines.append(f"On the repaired held set, trusted coherent86 has four-condition success {ph['n_four_condition_success']}/{ph['n_pairs']} and query-orientation success {ph['n_query_orientation_success']}/{ph['n_query_orientations']}. Mean U={ph['mean_U']:+.3f}, mean R={ph['mean_R']:+.3f}, beta={ph['mean_beta_pair_average']:+.3f}, |alpha|={ph['mean_abs_alpha_pair_average']:+.3f}, and mean min-four signed margin={ph['mean_min_four_signed_margin']:+.3f}. Thus the valid failure statement is not the research wording; it is that coherent86 strongly prefers the shared new phrase in update contexts but does not condition that preference on which entity was updated when the new phrase is fixed.\n\n")
    lines.append("## Bounded acquisition result\n\n")
    fh = e80["final_held"]
    ft = e80["final_train"]
    lines.append(f"A bounded answer-only run on the repaired rows, using only coherent86 private adapters, moved train to {ft['n_four_condition_success']}/{ft['n_pairs']} four-condition success and held to {fh['n_four_condition_success']}/{fh['n_pairs']} four-condition success ({fh['n_query_orientation_success']}/{fh['n_query_orientations']} query orientations) at 80 epochs. Held mean U={fh['mean_U']:+.3f}, R={fh['mean_R']:+.3f}, beta={fh['mean_beta_pair_average']:+.3f}, |alpha|={fh['mean_abs_alpha_pair_average']:+.3f}, min-four={fh['mean_min_four_signed_margin']:+.3f}. This establishes learnability and held source/entity transfer under direct relation-aligned answer supervision on the repaired natural relation substrate. It does not by itself attribute the effect to credit allocation relative to ordinary MLM; that requires matched arms.\n\n")
    lines.append("## Background corruption comparison\n\n")
    for arm in bg["arms"]:
        st = arm["cumulative_stats"]
        hfinal = arm["final_held"]
        lines.append(f"- {arm['arm']}: held {hfinal['n_four_condition_success']}/{hfinal['n_pairs']} four-condition success and {hfinal['n_query_orientation_success']}/{hfinal['n_query_orientations']} query orientations at 60 epochs; mean min-four={hfinal['mean_min_four_signed_margin']:+.3f}. Cumulative answer-label positions {st.get('answer_label_positions')}, background-corrupted positions {st.get('bg_corrupted_positions')}, answer/background overlap {st.get('answer_bg_overlap_positions')}.\n")
    lines.append("\nBoth corrupted arms used identical background input corruption counts and no answer/background overlap. Compared with the no-corruption answer-only trajectory, standard-style corruption of the supporting context sharply slows acquisition; adding background labels produces a smaller additional deficit at this horizon. This means the repaired natural substrate separates at least three factors: semantic packet validity, direct answer credit, and preservation of the evidence tokens the answer needs. It weakens any simple story that background loss alone is the bottleneck.\n\n")
    lines.append("## Consequence for the BabyLM bridge\n\n")
    lines.append("The next scientifically meaningful BabyLM-facing comparison should keep the repaired contract and trusted loader, then compare matched relation-packet objectives under legal exposure: answer-only with uncorrupted support, answer-only with standard support corruption, and answer+background with matched support corruption, before mixing with ALN/filler. Held relation transfer is now real in the clean answer-only setting; the unresolved question is how to retain that selection computation while coexisting with ordinary MLM/ALN rather than treating a successful cloze intervention as a submit-ready training principle.\n\n")
    lines.append("## Figures and files\n\n")
    for f in result["figures"]:
        lines.append(f"- `{f}`\n")
    lines.append(f"- model identity audit: `research/documents/functional_learning/data/model_identity_audit/model_identity_audit.md`\n")
    lines.append(f"- repaired parent score: `experiments/archive/functional_learning/data/relation_first_repaired/trusted_parent_summary.json`\n")
    lines.append(f"- answer-only run: `{rel(E80)}`\n")
    lines.append(f"- background comparison: `{rel(BG)}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    for fig in [fig1, fig2, fig3]:
        print(f"{rel(fig)} {fig.stat().st_size} bytes")
    print(json.dumps({
        "status": result["status"],
        "note": rel(NOTE),
        "out_json": rel(OUT_JSON),
        "answer_only_held_four": f"{fh['n_four_condition_success']}/{fh['n_pairs']}",
        "corrupted_answer_only_held_four": f"{bg_arm['corrupted_answer_only']['final_held']['n_four_condition_success']}/{bg_arm['corrupted_answer_only']['final_held']['n_pairs']}",
        "corrupted_plus_bg_held_four": f"{bg_arm['corrupted_answer_plus_bg']['final_held']['n_four_condition_success']}/{bg_arm['corrupted_answer_plus_bg']['final_held']['n_pairs']}",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
