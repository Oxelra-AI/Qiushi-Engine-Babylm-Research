#!/usr/bin/env python3
"""research: analyze binding-factorial outputs on an entity-gating coordinate.

The research/073 natural recombination factorial initially reported raw joint
pair correctness.  Raw joint can rise when a model changes its global
source-vs-update preference, without learning to assign the source and update
states to different entities in the same context.  This script reads the saved
per-pair margin files for the three arms and computes:

  gating_count = joint_correct - (a_correct * b_correct / n_pairs)
  gating_frac  = joint_acc - a_acc * b_acc

where A is the unchanged-entity/source-retention half and B is the
updated-entity/update-selection half.  Positive values indicate more paired
success than expected from independent marginal half success; negative values
indicate anticorrelation or packet-level bias.  It also reports the 119-pair
subset where the updated entity is already present in the source, the cleanest
within-source identity assignment subset from research.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import statistics
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path.cwd()
BASE = ROOT / "experiments/archive/relation_learning/data/binding_factorial"
ROWS = ROOT / "experiments/archive/relation_learning/data/recombination_rows"
STRATA = ROOT / "experiments/archive/relation_learning/data/binding_pair_strata_and_export/heldout_binding_pair_source_presence.csv"
OUT = ROOT / "experiments/archive/relation_learning/data/binding_factorial_gating_analysis"
ARMS = ["answer_clean", "uniform_wwm", "answer_corrupt_update_state"]
EPOCHS = [0, 5, 10, 15, 20]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_bool(x: str) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def load_strata() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with STRATA.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rr: dict[str, Any] = dict(r)
            for k in ["target_entity_in_source", "updated_entity_in_source", "both_entities_in_source", "same_source_update_context"]:
                rr[k] = parse_bool(str(rr.get(k, "")))
            out[str(rr["pair_id"])] = rr
    return out


def count_rows(path: pathlib.Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    by_answer = Counter(r.get("answer_kind", "") for r in rows)
    by_half = Counter(r.get("pair_half", "") for r in rows)
    by_packet = Counter(r.get("packet_type", "") for r in rows)
    by_role = Counter(r.get("role", "") for r in rows)
    by_combo = Counter((r.get("pair_half", ""), r.get("answer_kind", ""), r.get("packet_type", ""), r.get("role", "")) for r in rows)
    return {
        "path": rel(path),
        "n_rows": len(rows),
        "answer_kind": dict(by_answer),
        "pair_half": dict(by_half),
        "packet_type": dict(by_packet),
        "role": dict(by_role),
        "combo": {"|".join(map(str, k)): int(v) for k, v in sorted(by_combo.items())},
    }


def summarize_pair_rows(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(pair_rows)
    if n == 0:
        return {
            "n": 0, "a_correct": 0, "b_correct": 0, "joint_correct": 0,
            "a_acc": float("nan"), "b_acc": float("nan"), "joint_acc": float("nan"),
            "expected_joint_independent_count": float("nan"), "gating_count": float("nan"), "gating_frac": float("nan"),
            "mean_a_margin": float("nan"), "mean_b_margin": float("nan"), "mean_joint_min_margin": float("nan"),
        }
    a = int(sum(int(p.get("a_correct", 0)) for p in pair_rows))
    b = int(sum(int(p.get("b_correct", 0)) for p in pair_rows))
    j = int(sum(int(p.get("joint_correct", 0)) for p in pair_rows))
    expected = (a * b / n) if n else float("nan")
    def mean(key: str) -> float:
        vals = [float(p[key]) for p in pair_rows if key in p and math.isfinite(float(p[key]))]
        return float(statistics.mean(vals)) if vals else float("nan")
    return {
        "n": n,
        "a_correct": a,
        "b_correct": b,
        "joint_correct": j,
        "a_acc": a / n,
        "b_acc": b / n,
        "joint_acc": j / n,
        "expected_joint_independent_count": expected,
        "expected_joint_independent_acc": expected / n,
        "gating_count": j - expected,
        "gating_frac": j / n - (a / n) * (b / n),
        "mean_a_margin": mean("a_correct_margin"),
        "mean_b_margin": mean("b_correct_margin"),
        "mean_joint_min_margin": mean("joint_min_margin"),
    }


def analyze_eval(path: pathlib.Path, strata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ev = read_json(path)
    pair_rows = ev.get("pair_margins", [])
    for p in pair_rows:
        p["_strata"] = strata.get(str(p.get("pair_id")), {})
    subsets = {
        "all_pairs": pair_rows,
        "updated_entity_in_source_119": [p for p in pair_rows if p["_strata"].get("updated_entity_in_source") is True],
        "updated_entity_not_in_source_81": [p for p in pair_rows if p["_strata"].get("updated_entity_in_source") is False],
        "both_entities_in_source_115": [p for p in pair_rows if p["_strata"].get("both_entities_in_source") is True],
        "not_both_entities_in_source_85": [p for p in pair_rows if p["_strata"].get("both_entities_in_source") is False],
        "target_entity_in_source_193": [p for p in pair_rows if p["_strata"].get("target_entity_in_source") is True],
        "target_entity_not_in_source_7": [p for p in pair_rows if p["_strata"].get("target_entity_in_source") is False],
    }
    out = {name: summarize_pair_rows(rows) for name, rows in subsets.items()}
    # Preserve the original by-role summary, but note that it includes UPDATED single rows for the original unbalanced heldout.
    out["raw_eval_header"] = {k: v for k, v in ev.items() if k not in {"row_margins", "pair_margins"}}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    strata = load_strata()
    dataset_counts = {
        "train": count_rows(ROWS / "recombination_train.jsonl"),
        "heldout": count_rows(ROWS / "recombination_heldout.jsonl"),
        "binding_pairs": {"path": rel(ROWS / "binding_pairs_heldout.jsonl"), "n_pairs": len(read_jsonl(ROWS / "binding_pairs_heldout.jsonl"))},
        "strata_counts": dict(Counter(
            "updated_entity_in_source" if s.get("updated_entity_in_source") else "updated_entity_not_in_source"
            for s in strata.values()
        )),
    }
    results: dict[str, Any] = {
        "status": "BINDING_FACTORIAL_GATING_ANALYSIS",
        "inputs": {
            "factorial_root": rel(BASE),
            "strata_csv": rel(STRATA),
            "rows_root": rel(ROWS),
        },
        "dataset_counts": dataset_counts,
        "arms": {},
    }
    flat_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_dir = BASE / arm
        identity = read_json(arm_dir / "model_identity.json")
        freeze = read_json(arm_dir / "freeze_stats.json")
        summary = read_json(arm_dir / "summary.json")
        arm_out = {
            "model_identity": identity,
            "freeze_stats": freeze,
            "n_train_tokenized": summary.get("n_train_tokenized"),
            "n_heldout_tokenized": summary.get("n_heldout_tokenized"),
            "n_binding_pairs": summary.get("n_binding_pairs"),
            "trajectory": {},
        }
        for epoch in EPOCHS:
            p = arm_dir / f"eval_epoch_{epoch:03d}.json"
            if not p.exists():
                continue
            ep = analyze_eval(p, strata)
            arm_out["trajectory"][str(epoch)] = ep
            for subset, vals in ep.items():
                if subset == "raw_eval_header":
                    continue
                flat_rows.append({
                    "arm": arm,
                    "epoch": epoch,
                    "subset": subset,
                    **{k: vals.get(k) for k in [
                        "n", "a_correct", "b_correct", "joint_correct", "a_acc", "b_acc", "joint_acc",
                        "expected_joint_independent_count", "expected_joint_independent_acc", "gating_count", "gating_frac",
                        "mean_a_margin", "mean_b_margin", "mean_joint_min_margin",
                    ]},
                })
        results["arms"][arm] = arm_out
    # Arm contrasts on all pairs and source-present subset at epoch 20.
    contrasts: list[dict[str, Any]] = []
    for subset in ["all_pairs", "updated_entity_in_source_119", "updated_entity_not_in_source_81", "both_entities_in_source_115"]:
        vals = {arm: results["arms"][arm]["trajectory"].get("20", {}).get(subset, {}) for arm in ARMS}
        for a1, a2 in [("answer_clean", "uniform_wwm"), ("answer_clean", "answer_corrupt_update_state"), ("answer_corrupt_update_state", "uniform_wwm")]:
            contrasts.append({
                "epoch": 20,
                "subset": subset,
                "contrast": f"{a1}-minus-{a2}",
                "joint_correct_delta": vals[a1].get("joint_correct", float("nan")) - vals[a2].get("joint_correct", float("nan")),
                "a_correct_delta": vals[a1].get("a_correct", float("nan")) - vals[a2].get("a_correct", float("nan")),
                "b_correct_delta": vals[a1].get("b_correct", float("nan")) - vals[a2].get("b_correct", float("nan")),
                "gating_count_delta": vals[a1].get("gating_count", float("nan")) - vals[a2].get("gating_count", float("nan")),
                "gating_frac_delta": vals[a1].get("gating_frac", float("nan")) - vals[a2].get("gating_frac", float("nan")),
                "mean_a_margin_delta": vals[a1].get("mean_a_margin", float("nan")) - vals[a2].get("mean_a_margin", float("nan")),
                "mean_b_margin_delta": vals[a1].get("mean_b_margin", float("nan")) - vals[a2].get("mean_b_margin", float("nan")),
            })
    results["epoch20_contrasts"] = contrasts

    # Save machine-readable files.
    (OUT / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "trajectory_flat.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(flat_rows[0].keys()) if flat_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(flat_rows)
    with (OUT / "epoch20_contrasts.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(contrasts[0].keys()) if contrasts else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(contrasts)

    # Markdown with the main scientific readings.
    lines: list[str] = []
    lines.append("# research binding factorial gating analysis\n")
    lines.append("The coordinate reported here is `joint_correct - a_correct*b_correct/n`. Positive values mean pairwise entity assignment succeeds more often than expected from the two marginal half-accuracies; negative values mean the two halves are anticorrelated or governed by a packet-level source/update preference.\n")
    lines.append("## Dataset balance\n")
    lines.append(f"- Train rows: {dataset_counts['train']['n_rows']}; answer kinds {dataset_counts['train']['answer_kind']}; pair halves {dataset_counts['train']['pair_half']}.\n")
    lines.append(f"- Heldout rows: {dataset_counts['heldout']['n_rows']}; answer kinds {dataset_counts['heldout']['answer_kind']}; pair halves {dataset_counts['heldout']['pair_half']}.\n")
    lines.append("- The 4,998 training rows therefore contain 1,666 source-state answer rows and 3,332 new-state answer rows; answer-only training admits a 2:1 update-state prior before entity identity is read.\n")
    lines.append("## Epoch 20: all 200 binding pairs\n")
    lines.append("| arm | joint | A source-retain | B update-select | expected joint | gating count | gating frac | mean A margin | mean B margin |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for arm in ARMS:
        v = results["arms"][arm]["trajectory"]["20"]["all_pairs"]
        lines.append(f"| {arm} | {v['joint_correct']}/{v['n']} | {v['a_correct']}/{v['n']} | {v['b_correct']}/{v['n']} | {v['expected_joint_independent_count']:.2f} | {v['gating_count']:.2f} | {v['gating_frac']:.3f} | {v['mean_a_margin']:.3f} | {v['mean_b_margin']:.3f} |\n")
    lines.append("## Epoch 20: updated entity already present in source (119 pairs)\n")
    lines.append("| arm | joint | unchanged-source retention | updated-state selection | expected joint | gating count | gating frac | mean A margin | mean B margin |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for arm in ARMS:
        v = results["arms"][arm]["trajectory"]["20"]["updated_entity_in_source_119"]
        lines.append(f"| {arm} | {v['joint_correct']}/{v['n']} | {v['a_correct']}/{v['n']} | {v['b_correct']}/{v['n']} | {v['expected_joint_independent_count']:.2f} | {v['gating_count']:.2f} | {v['gating_frac']:.3f} | {v['mean_a_margin']:.3f} | {v['mean_b_margin']:.3f} |\n")
    lines.append("## Trajectory, all pairs\n")
    lines.append("| arm | epoch | joint | A | B | expected | gating | mean A margin | mean B margin |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for arm in ARMS:
        for epoch in EPOCHS:
            v = results["arms"][arm]["trajectory"][str(epoch)]["all_pairs"]
            lines.append(f"| {arm} | {epoch} | {v['joint_correct']} | {v['a_correct']} | {v['b_correct']} | {v['expected_joint_independent_count']:.2f} | {v['gating_count']:.2f} | {v['mean_a_margin']:.3f} | {v['mean_b_margin']:.3f} |\n")
    lines.append("## Interpretation\n")
    lines.append("- `answer_clean` raises raw joint over uniform WWM, but its epoch-20 gating coordinate remains negative on all pairs and on the 119-pair within-source subset. The main movement is stronger B/update selection with weak or declining A/source retention, not a clean entity gate.\n")
    lines.append("- `answer_corrupt_update_state` improves A/source retention while suppressing B/update selection, showing the rows are sensitive to source-vs-update support. Its raw joint is still far below the independence product, so it also does not learn pairwise assignment.\n")
    lines.append("- `uniform_wwm` barely moves from the base and remains strongly negative on the gating coordinate.\n")
    lines.append("- The unbalanced row mix is a construction cause: source-state answer rows are outnumbered 2:1 by new-state answer rows because UPDATED single rows were stacked on top of the paired rows. A balanced paired rerun should drop or mirror singles and keep both halves of each pair in the same update.\n")
    (OUT / "summary.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": results["status"], "summary": rel(OUT / "summary.json"), "md": rel(OUT / "summary.md"), "csv": rel(OUT / "trajectory_flat.csv")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
