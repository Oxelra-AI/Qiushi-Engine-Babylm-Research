#!/usr/bin/env python3
"""Step040f: final synthesis of repaired relation-first research experiments."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib

import matplotlib.pyplot as plt

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
PARENT = _public_path('experiments/archive/functional_learning/data/relation_first_repaired/trusted_parent_summary.json')
E80 = _public_path('experiments/archive/functional_learning/data/relation_first_repaired_e80/answer_only_private_e80_seed40040/training_summary.json')
BG = _public_path('experiments/archive/functional_learning/data/revision_040b_repaired_bg_comparison/bg_comparison_summary.json')
CLEAN = _public_path('experiments/archive/functional_learning/data/revision_040d_clean_same_codepath/bg_comparison_summary.json')
PROTECT = _public_path('experiments/archive/functional_learning/data/revision_040e_protect_critical_corruption/protect_critical_summary.json')
MODEL_AUDIT = _public_path('research/documents/functional_learning/data/model_identity_audit/model_identity_audit.md')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/final_synthesis')
FIG_DIR = _public_path('experiments/archive/functional_learning/figures')
NOTE = _public_path('research/notes/functional_learning/repaired_relation_first_full_synthesis.md')
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
_public_path('research/notes/functional_learning').mkdir(parents=True, exist_ok=True)


def rel(p):
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def compact(s):
    return {
        "four": f"{s['n_four_condition_success']}/{s['n_pairs']}",
        "orient": f"{s['n_query_orientation_success']}/{s['n_query_orientations']}",
        "U": s["mean_U"],
        "R": s["mean_R"],
        "beta": s["mean_beta_pair_average"],
        "abs_alpha": s["mean_abs_alpha_pair_average"],
        "min4": s["mean_min_four_signed_margin"],
    }


def find_arm(summary, name):
    for a in summary["arms"]:
        if a["arm"] == name:
            return a
    raise KeyError(name)


def main():
    parent = load(PARENT)
    e80 = load(E80)
    bg = load(BG)
    clean = load(CLEAN)
    protect = load(PROTECT)
    clean_arm = find_arm(clean, "corrupted_answer_only")
    corrupt_ao = find_arm(bg, "corrupted_answer_only")
    corrupt_bg = find_arm(bg, "corrupted_answer_plus_bg")

    conditions = [
        ("parent\n(no training)", parent["held"], "#777777"),
        ("clean AO\ne60 same path", clean_arm["final_held"], "#2ca02c"),
        ("protect critical\n15% corrupt e60", protect["final_held"], "#ff7f0e"),
        ("broad corrupt AO\ne60", corrupt_ao["final_held"], "#d62728"),
        ("broad corrupt +bg\ne60", corrupt_bg["final_held"], "#9467bd"),
        ("clean AO\ne80", e80["final_held"], "#1f77b4"),
    ]
    labels = [c[0] for c in conditions]
    four = [c[1]["n_four_condition_success"] for c in conditions]
    orient = [c[1]["n_query_orientation_success"] for c in conditions]
    min4 = [c[1]["mean_min_four_signed_margin"] for c in conditions]
    beta = [c[1]["mean_beta_pair_average"] for c in conditions]
    colors = [c[2] for c in conditions]

    plt.figure(figsize=(10.5, 5.5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(range(len(labels)), four, color=colors, alpha=0.88)
    ax1.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax1.set_ylabel("held four-condition successes / 30")
    ax1.set_ylim(0, 30)
    ax1.set_title("Strict held success")
    for i, y in enumerate(four):
        ax1.text(i, y + 0.6, str(y), ha="center", fontsize=9)
    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(range(len(labels)), min4, color=colors, alpha=0.88)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax2.set_ylabel("held mean min-four signed margin")
    ax2.set_title("Strict margin")
    for i, y in enumerate(min4):
        ax2.text(i, y + (0.25 if y >= 0 else -0.45), f"{y:+.2f}", ha="center", fontsize=9)
    plt.suptitle("research repaired relation-first acquisition: evidence preservation and focused credit")
    plt.tight_layout()
    fig1 = _public_path('experiments/archive/functional_learning/figures/final_condition_comparison.png')
    plt.savefig(fig1, dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 5.8))
    for label, s, color in conditions:
        plt.scatter(s["mean_beta_pair_average"], s["mean_abs_alpha_pair_average"], s=90, color=color, label=label.replace("\n", " "))
        plt.text(s["mean_beta_pair_average"], s["mean_abs_alpha_pair_average"], label.split("\n")[0], fontsize=8, ha="left", va="bottom")
    mx = max(max(beta), max(c[1]["mean_abs_alpha_pair_average"] for c in conditions)) + 0.5
    plt.plot([0, mx], [0, mx], linestyle="--", color="gray", label="beta=|alpha|")
    plt.xlabel("held mean beta")
    plt.ylabel("held mean |alpha|")
    plt.title("Repaired substrate moves from shared update preference to conditional selection")
    plt.legend(frameon=False, fontsize=8, loc="best")
    plt.tight_layout()
    fig2 = _public_path('experiments/archive/functional_learning/figures/final_beta_alpha_conditions.png')
    plt.savefig(fig2, dpi=180)
    plt.close()

    summary = {
        "status": "FINAL_REPAIRED_RELATION_FIRST_SYNTHESIS",
        "conditions": {label: compact(s) for label, s, _ in conditions},
        "key_updates": {
            "invalidated": "generic AutoModelForMaskedLM loaded stock DebertaV2 with 0 private-adapter params; research trainer optimized wrong graph and non-private full model",
            "parent_failure": "trusted coherent86 has strong shared new phrase preference but 0/30 held four-condition success when shared replacement and recipient-only contrast are enforced",
            "clean_acquisition": "private-adapter answer-only training on repaired rows reaches 19/30 held at e60 in same Step040b code path and 28/30 held at e80 in research harness",
            "corruption_localization": "15% broad context corruption yields only 2/30 held at e60; protecting entity/source-value/replacement word groups yields 11/30, showing key evidence-token corruption explains a large but incomplete part of the gap",
            "background_labels": "with broad corruption, adding background labels changes held success little at e60 (2/30 vs 2/30) and worsens continuous margins modestly; background labels alone are not established as the main obstacle",
        },
        "figures": [rel(fig1), rel(fig2)],
        "notes": rel(NOTE),
    }
    (_public_path('experiments/archive/functional_learning/data/final_synthesis/final_synthesis.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# research final repaired relation-first synthesis\n\n")
    lines.append("## Core correction\n\n")
    lines.append("The relation-first extraction from research remains valuable, but research scoring/training is not coherent86 evidence. The audit shows that generic `AutoModelForMaskedLM` without trusted custom loading selects stock `DebertaV2ForMaskedLM` with 0 private-adapter parameters and discards adapter tensors. The repaired scripts use the established custom coherent86 loader and train only 995,584 private-adapter parameters. The contrast now uses one shared replacement value per pair, so for a fixed query only update-recipient identity changes. Scoring masks the explicit final answer span simultaneously and has no token-search fallback.\n\n")
    lines.append("## What the repaired evidence establishes\n\n")
    p = compact(parent["held"])
    lines.append(f"- Trusted parent baseline: held {p['four']} four-condition and {p['orient']} orientations; U={p['U']:+.3f}, R={p['R']:+.3f}, beta={p['beta']:+.3f}, |alpha|={p['abs_alpha']:+.3f}, min4={p['min4']:+.3f}. Coherent86 strongly follows the newly appended value in update contexts but does not condition that preference on the updated entity when the replacement is fixed.\n")
    c60 = compact(clean_arm["final_held"])
    c80 = compact(e80["final_held"])
    lines.append(f"- Clean answer-only private-adapter acquisition: same-codepath e60 held {c60['four']} ({c60['orient']}), min4={c60['min4']:+.3f}; e80 held {c80['four']} ({c80['orient']}), U={c80['U']:+.3f}, R={c80['R']:+.3f}, beta={c80['beta']:+.3f}, |alpha|={c80['abs_alpha']:+.3f}, min4={c80['min4']:+.3f}. This establishes learnability and held source/entity transfer under concentrated relation-aligned answer supervision.\n")
    ca = compact(corrupt_ao["final_held"])
    cb = compact(corrupt_bg["final_held"])
    pr = compact(protect["final_held"])
    lines.append(f"- Broad 15% context corruption with answer-only labels at e60: held {ca['four']} ({ca['orient']}), min4={ca['min4']:+.3f}. With background labels added under identical corruption: held {cb['four']} ({cb['orient']}), min4={cb['min4']:+.3f}.\n")
    lines.append(f"- Protecting critical evidence word groups under 15% corruption (entity names, both source values, and shared replacement occurrences outside final answer): held {pr['four']} ({pr['orient']}), min4={pr['min4']:+.3f}. Protected groups prevented 75,125 would-have-been-selected token positions from being corrupted while 240,711 background positions still were corrupted. This recovers much of the clean e60 success but not all of it, separating destruction of key support tokens from broader noise/optimization effects.\n\n")
    lines.append("## Updated mechanism view\n\n")
    lines.append("The repaired natural relation substrate now supports a stronger and more precise mechanism than the research/research stories. Under finite experience, the parent already has a high-gain operation for adopting a recently appended relation value, but it is entity-agnostic. Concentrated answer supervision can redirect the private adapters toward an entity-conditioned retain/update rule, primarily by changing the RETAIN side from preferring the new value to preserving the queried entity's source value. Standard MLM-style corruption damages this acquisition mainly by hiding sparse support tokens needed for selection; broad background labels are not yet the dominant source of failure in the repaired comparison. The emerging principle candidate is therefore not simply 'background loss interferes', but: data-efficient conditional retrieval requires sparse relational evidence to remain available while credit is concentrated on the output that forces entity-conditioned selection.\n\n")
    lines.append("## What remains before BabyLM integration\n\n")
    lines.append("This is a bounded acquisition result, not a SOTA improvement. Next work should use the same repaired substrate to run matched multi-seed arms, token-region corruption controls, and ordinary pooled MLM/ALN-compatible objectives with legal exposure accounting. The held set currently shares relation templates and replacement pools with training; future splits should separate documents/entities/values/replacement pools/templates and balance relation types beyond death_place. ALN-preserving continuation should not begin until the intervention shows robust held selection while preserving general language competence.\n\n")
    lines.append("## Files\n\n")
    for pth in [MODEL_AUDIT, PARENT, E80, BG, CLEAN, PROTECT, _public_path('experiments/archive/functional_learning/data/final_synthesis/final_synthesis.json'), fig1, fig2]:
        lines.append(f"- `{rel(pth)}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "note": rel(NOTE),
        "figures": summary["figures"],
        "parent_held": p,
        "clean_e60_held": c60,
        "protect_e60_held": pr,
        "broad_corrupt_e60_held": ca,
        "broad_corrupt_plus_bg_e60_held": cb,
        "clean_e80_held": c80,
    }, indent=2, ensure_ascii=False), flush=True)
    print(f"{rel(fig1)} {fig1.stat().st_size} bytes")
    print(f"{rel(fig2)} {fig2.stat().st_size} bytes")


if __name__ == "__main__":
    main()
