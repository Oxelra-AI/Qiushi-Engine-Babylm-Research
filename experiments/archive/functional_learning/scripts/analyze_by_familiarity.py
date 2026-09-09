#!/usr/bin/env python3
"""research analysis: decompose three-way and reassignment results by entity familiarity.

Reads outputs from threeway_and_reassignment_probe.py and produces
decomposed statistics and figures.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')

V1_PAIRS = _public_path('experiments/archive/functional_learning/data/relation_first_packets/relation_first_pairs.jsonl')
THREEWAY_DIR = _public_path('experiments/archive/functional_learning/data/threeway_probe')
FIG_DIR = _public_path('experiments/archive/functional_learning/figures')


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_entity_familiarity(pair, train_entities):
    ea, eb = pair.get("entity_a", ""), pair.get("entity_b", "")
    ea_in = ea in train_entities
    eb_in = eb in train_entities
    if ea_in and eb_in:
        return "both_familiar"
    elif ea_in or eb_in:
        return "one_familiar"
    else:
        return "both_novel"


def main():
    # Load pairs
    pairs = read_jsonl(V1_PAIRS)
    train_entities = set()
    for p in pairs:
        if p.get("split") == "train":
            train_entities.add(p["entity_a"])
            train_entities.add(p["entity_b"])

    # Build pair_id → familiarity map
    pair_familiarity = {}
    for p in pairs:
        pair_familiarity[p["pair_id"]] = classify_entity_familiarity(p, train_entities)

    # Load three-way results
    summary_path = _public_path('experiments/archive/functional_learning/data/threeway_probe/threeway_reassignment_summary.json')
    if not summary_path.exists():
        print(f"Summary not found at {summary_path}. Run threeway_and_reassignment_probe.py first.")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    # Load per-pair records
    trained_held = read_jsonl(_public_path('experiments/archive/functional_learning/data/threeway_probe/trained_held_threeway_records.jsonl'))
    trained_train = read_jsonl(_public_path('experiments/archive/functional_learning/data/threeway_probe/trained_train_threeway_records.jsonl'))
    parent_held = read_jsonl(_public_path('experiments/archive/functional_learning/data/threeway_probe/parent_held_threeway_records.jsonl'))
    reassignment = read_jsonl(_public_path('experiments/archive/functional_learning/data/threeway_probe/reassignment_records.jsonl'))
    parent_reassign = read_jsonl(_public_path('experiments/archive/functional_learning/data/threeway_probe/parent_reassignment_records.jsonl'))

    # Annotate with familiarity
    for rec in trained_held + trained_train + parent_held + reassignment + parent_reassign:
        rec["familiarity"] = pair_familiarity.get(rec["pair_id"], "unknown")

    # === Decompose trained held by familiarity ===
    print("\n" + "=" * 70)
    print("TRAINED MODEL HELD THREE-WAY: DECOMPOSED BY ENTITY FAMILIARITY")
    print("=" * 70)
    for fam in ["both_novel", "one_familiar", "both_familiar"]:
        subset = [r for r in trained_held if r["familiarity"] == fam]
        n = len(subset)
        if n == 0:
            continue
        qa_ret = sum(1 for r in subset if r["retain_qa_entity_retrieval"])
        qb_ret = sum(1 for r in subset if r["retain_qb_entity_retrieval"])
        both_ret = sum(1 for r in subset if r["retain_qa_entity_retrieval"] and r["retain_qb_entity_retrieval"])
        qa_full = sum(1 for r in subset if r["retain_qa_full_retrieval"])
        qb_full = sum(1 for r in subset if r["retain_qb_full_retrieval"])
        both_full = sum(1 for r in subset if r["retain_qa_full_retrieval"] and r["retain_qb_full_retrieval"])
        mean_cross = np.mean([r["retain_qa_cross_source_margin"] for r in subset] +
                             [r["retain_qb_cross_source_margin"] for r in subset])
        mean_correct_new = np.mean([r["retain_qa_correct_over_new"] for r in subset] +
                                    [r["retain_qb_correct_over_new"] for r in subset])
        update_qa = sum(1 for r in subset if r["update_qa_correct"])
        update_qb = sum(1 for r in subset if r["update_qb_correct"])
        print(f"\n{fam} ({n} pairs):")
        print(f"  Cross-source retrieval: qa={qa_ret}/{n}, qb={qb_ret}/{n}, both={both_ret}/{n}")
        print(f"  Full retrieval (cross>0 AND correct>new): qa={qa_full}/{n}, qb={qb_full}/{n}, both={both_full}/{n}")
        print(f"  Mean cross-source margin: {mean_cross:+.3f}")
        print(f"  Mean correct-over-new margin: {mean_correct_new:+.3f}")
        print(f"  UPDATE success: qa={update_qa}/{n}, qb={update_qb}/{n}")

    # === Parent held decomposed ===
    print("\n" + "=" * 70)
    print("PARENT MODEL HELD THREE-WAY: DECOMPOSED BY ENTITY FAMILIARITY")
    print("=" * 70)
    for fam in ["both_novel", "one_familiar", "both_familiar"]:
        subset = [r for r in parent_held if r["familiarity"] == fam]
        n = len(subset)
        if n == 0:
            continue
        both_ret = sum(1 for r in subset if r["retain_qa_entity_retrieval"] and r["retain_qb_entity_retrieval"])
        mean_cross = np.mean([r["retain_qa_cross_source_margin"] for r in subset] +
                             [r["retain_qb_cross_source_margin"] for r in subset])
        print(f"\n{fam} ({n} pairs):")
        print(f"  Cross-source retrieval both: {both_ret}/{n}")
        print(f"  Mean cross-source margin: {mean_cross:+.3f}")

    # === Reassignment decomposed ===
    print("\n" + "=" * 70)
    print("TRAINED MODEL HELD REASSIGNMENT: DECOMPOSED BY ENTITY FAMILIARITY")
    print("=" * 70)
    for fam in ["both_novel", "one_familiar", "both_familiar"]:
        subset = [r for r in reassignment if r["familiarity"] == fam]
        n = len(subset)
        if n == 0:
            continue
        follows_both = sum(1 for r in subset if r["swap_qa_follows_reassignment"] and r["swap_qb_follows_reassignment"])
        follows_qa = sum(1 for r in subset if r["swap_qa_follows_reassignment"])
        follows_qb = sum(1 for r in subset if r["swap_qb_follows_reassignment"])
        mean_qa = np.mean([r["swap_qa_margin"] for r in subset])
        mean_qb = np.mean([r["swap_qb_margin"] for r in subset])
        print(f"\n{fam} ({n} pairs):")
        print(f"  Follows reassignment: qa={follows_qa}/{n}, qb={follows_qb}/{n}, both={follows_both}/{n}")
        print(f"  Mean swap margins: qa={mean_qa:+.3f}, qb={mean_qb:+.3f}")

    # === Parent reassignment decomposed ===
    print("\n" + "=" * 70)
    print("PARENT MODEL HELD REASSIGNMENT: DECOMPOSED BY ENTITY FAMILIARITY")
    print("=" * 70)
    for fam in ["both_novel", "one_familiar", "both_familiar"]:
        subset = [r for r in parent_reassign if r["familiarity"] == fam]
        n = len(subset)
        if n == 0:
            continue
        follows_both = sum(1 for r in subset if r["swap_qa_follows_reassignment"] and r["swap_qb_follows_reassignment"])
        print(f"  {fam} ({n} pairs): follows both={follows_both}/{n}")

    # === Figures ===
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Cross-source margin by familiarity (trained vs parent, held)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fam_labels = ["both_novel", "one_familiar", "both_familiar"]
    fam_names = ["Both Novel", "One Familiar", "Both Familiar"]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    for ax, fam, fname, col in zip(axes, fam_labels, fam_names, colors):
        trained_sub = [r for r in trained_held if r["familiarity"] == fam]
        parent_sub = [r for r in parent_held if r["familiarity"] == fam]

        if trained_sub:
            trained_margins = ([r["retain_qa_cross_source_margin"] for r in trained_sub] +
                              [r["retain_qb_cross_source_margin"] for r in trained_sub])
            parent_margins = ([r["retain_qa_cross_source_margin"] for r in parent_sub] +
                             [r["retain_qb_cross_source_margin"] for r in parent_sub])
            positions = [0, 1]
            bp = ax.boxplot([parent_margins, trained_margins], positions=positions,
                          widths=0.6, patch_artist=True)
            bp["boxes"][0].set_facecolor("#bdc3c7")
            bp["boxes"][1].set_facecolor(col)
            ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            ax.set_xticks(positions)
            ax.set_xticklabels(["Parent", "Trained"])
        ax.set_title(f"{fname}\n(n={len(trained_sub)})", fontsize=11)
        ax.set_ylabel("Cross-source margin\n(correct - wrong source logP)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("RETAIN Cross-Source Discrimination by Entity Familiarity (Held)", fontsize=13)
    plt.tight_layout()
    fig_path = _public_path('experiments/archive/functional_learning/figures/cross_source_by_familiarity.png')
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fig_path} ({fig_path.stat().st_size} bytes)")

    # Figure 2: Reassignment following by familiarity
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (fam, fname, col) in enumerate(zip(fam_labels, fam_names, colors)):
        trained_sub = [r for r in reassignment if r["familiarity"] == fam]
        parent_sub = [r for r in parent_reassign if r["familiarity"] == fam]
        n_t = len(trained_sub)
        n_p = len(parent_sub)
        if n_t > 0:
            t_rate = sum(1 for r in trained_sub if r["swap_qa_follows_reassignment"] and r["swap_qb_follows_reassignment"]) / n_t
            p_rate = sum(1 for r in parent_sub if r["swap_qa_follows_reassignment"] and r["swap_qb_follows_reassignment"]) / n_p if n_p > 0 else 0
            ax.bar(i * 2.5, p_rate, width=0.8, color="#bdc3c7", label="Parent" if i == 0 else "")
            ax.bar(i * 2.5 + 1, t_rate, width=0.8, color=col, label=fname)
            ax.text(i * 2.5, p_rate + 0.02, f"{int(p_rate*n_p)}/{n_p}", ha="center", fontsize=9)
            ax.text(i * 2.5 + 1, t_rate + 0.02, f"{int(t_rate*n_t)}/{n_t}", ha="center", fontsize=9)

    ax.set_xticks([0.5, 3.0, 5.5])
    ax.set_xticklabels(fam_names)
    ax.set_ylabel("Fraction following source reassignment\n(both query orientations)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Source-Reassignment Following by Entity Familiarity (Held)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig_path = _public_path('experiments/archive/functional_learning/figures/reassignment_by_familiarity.png')
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path} ({fig_path.stat().st_size} bytes)")

    # Save JSON summary
    result = {
        "status": "ENTITY_FAMILIARITY_DECOMPOSITION",
        "entity_overlap": {
            "train_entities": len(train_entities),
            "held_both_novel": sum(1 for r in trained_held if r["familiarity"] == "both_novel"),
            "held_one_familiar": sum(1 for r in trained_held if r["familiarity"] == "one_familiar"),
            "held_both_familiar": sum(1 for r in trained_held if r["familiarity"] == "both_familiar"),
        },
    }
    for model_name, records in [("trained", trained_held), ("parent", parent_held)]:
        for fam in fam_labels:
            subset = [r for r in records if r["familiarity"] == fam]
            n = len(subset)
            if n == 0:
                continue
            result[f"{model_name}_{fam}"] = {
                "n": n,
                "cross_source_retrieval_both": sum(1 for r in subset if r["retain_qa_entity_retrieval"] and r["retain_qb_entity_retrieval"]),
                "full_retrieval_both": sum(1 for r in subset if r["retain_qa_full_retrieval"] and r["retain_qb_full_retrieval"]),
                "mean_cross_source": float(np.mean([r["retain_qa_cross_source_margin"] for r in subset] +
                                                    [r["retain_qb_cross_source_margin"] for r in subset])),
            }
    for model_name, records in [("trained", reassignment), ("parent", parent_reassign)]:
        for fam in fam_labels:
            subset = [r for r in records if r["familiarity"] == fam]
            n = len(subset)
            if n == 0:
                continue
            result[f"reassignment_{model_name}_{fam}"] = {
                "n": n,
                "follows_both": sum(1 for r in subset if r["swap_qa_follows_reassignment"] and r["swap_qb_follows_reassignment"]),
                "mean_swap_margin": float(np.mean([r["swap_qa_margin"] for r in subset] + [r["swap_qb_margin"] for r in subset])),
            }

    out_path = _public_path('experiments/archive/functional_learning/data/threeway_probe/entity_familiarity_decomposition.json')
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
