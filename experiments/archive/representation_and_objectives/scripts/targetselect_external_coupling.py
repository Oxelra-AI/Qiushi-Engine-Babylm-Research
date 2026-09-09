#!/usr/bin/env python3
"""research: summarize early external coupling of research target-selective arms.

This reads the repaired research official-compatible 20M cheap7 outputs for
full/drop_abs/drop_copied_tok and computes column deltas plus EWoK domain/subtask
movement.  It is deliberately an early-coupling measurement, not a route decision.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
DEFAULT_ROOT = ROOT / "data/target_selective_cheap7_eval_repaired"
GOLD_EWOK = ROOT / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
OUT_DIR = ROOT / "data/targetselect_external_coupling"

ARM_JSONS = {
    "full": DEFAULT_ROOT / "full20/per_target/full20.json",
    "drop_abs": DEFAULT_ROOT / "dropabs20/per_target/dropabs20.json",
    "drop_copied_tok": DEFAULT_ROOT / "dropcopied20/per_target/dropcopied20.json",
}
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RELATIONAL = ["social-properties", "physical-dynamics", "spatial-relations", "physical-relations"]
INDEPENDENT = ["material-properties", "social-interactions"]


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def score_table(arms: dict[str, Any]) -> dict[str, Any]:
    scores = {name: {c: arm["official_overall"]["scores"].get(c) for c in COLUMNS} for name, arm in arms.items()}
    for name in scores:
        vals = [scores[name][c] for c in COLUMNS]
        scores[name]["cheap7_equal_mean"] = round(sum(vals) / len(vals), 6)
    deltas: dict[str, Any] = {}
    for lhs, rhs in [("drop_abs", "full"), ("drop_copied_tok", "full"), ("drop_abs", "drop_copied_tok")]:
        key = f"{lhs}_minus_{rhs}"
        deltas[key] = {}
        for c in COLUMNS + ["cheap7_equal_mean"]:
            deltas[key][c] = round(scores[lhs][c] - scores[rhs][c], 6)
    return {"scores": scores, "deltas": deltas}


def iter_gold(gold_dir: pathlib.Path):
    for gf in sorted(gold_dir.glob("*.jsonl")):
        rows = [json.loads(l) for l in gf.read_text(encoding="utf-8").splitlines() if l.strip()]
        yield gf.stem, rows


def find_ewok_pred(arm_json: dict[str, Any]) -> pathlib.Path:
    p = pathlib.Path(arm_json["tasks"]["EWoK"]["predictions"])
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def score_ewok(pred_path: pathlib.Path, gold_dir: pathlib.Path) -> dict[str, Any]:
    pred = load_json(pred_path)
    total = correct = 0
    by_domain: dict[str, dict[str, int]] = {}
    by_subtask: dict[str, dict[str, int]] = {}
    for sub, rows in iter_gold(gold_dir):
        preds = pred[sub]["predictions"]
        if len(preds) != len(rows):
            raise ValueError(f"{sub}: pred {len(preds)} != gold {len(rows)}")
        for pr, g in zip(preds, rows):
            target = " ".join([g["Context1"], g["Target1"]]).strip()
            ok = (pr.get("pred") or "").strip() == target
            dom = g.get("Domain") or sub
            for tab, key in [(by_domain, dom), (by_subtask, sub)]:
                d = tab.setdefault(key, {"n": 0, "correct": 0})
                d["n"] += 1
                d["correct"] += int(ok)
            total += 1
            correct += int(ok)
    def finalize(tab: dict[str, dict[str, int]]) -> dict[str, Any]:
        return {k: {**v, "accuracy": round(100.0 * v["correct"] / max(v["n"], 1), 6)} for k, v in sorted(tab.items())}
    return {"total": total, "correct": correct, "accuracy": round(100.0 * correct / max(total, 1), 6), "domains": finalize(by_domain), "subtasks": finalize(by_subtask), "prediction_path": str(pred_path)}


def group_acc(domain_scores: dict[str, Any], domains: list[str]) -> dict[str, Any]:
    n = c = 0
    missing = []
    for d in domains:
        rec = domain_scores.get(d)
        if rec is None:
            missing.append(d)
        else:
            n += rec["n"]
            c += rec["correct"]
    return {"n": n, "correct": c, "accuracy": round(100.0 * c / max(n, 1), 6), "missing": missing}


def ewok_table(arms: dict[str, Any]) -> dict[str, Any]:
    arm_scores = {name: score_ewok(find_ewok_pred(arm), GOLD_EWOK) for name, arm in arms.items()}
    domains = sorted(set().union(*(set(v["domains"]) for v in arm_scores.values())))
    domain_table: dict[str, Any] = {}
    for d in domains:
        rec = {name: arm_scores[name]["domains"][d]["accuracy"] for name in arm_scores if d in arm_scores[name]["domains"]}
        if len(rec) == len(arm_scores):
            domain_table[d] = {
                "n": arm_scores["full"]["domains"][d]["n"],
                "scores": rec,
                "drop_abs_minus_full": round(rec["drop_abs"] - rec["full"], 6),
                "drop_copied_tok_minus_full": round(rec["drop_copied_tok"] - rec["full"], 6),
                "drop_abs_minus_drop_copied_tok": round(rec["drop_abs"] - rec["drop_copied_tok"], 6),
            }
    subtask_table: dict[str, Any] = {}
    subtasks = sorted(set().union(*(set(v["subtasks"]) for v in arm_scores.values())))
    for s in subtasks:
        rec = {name: arm_scores[name]["subtasks"][s]["accuracy"] for name in arm_scores if s in arm_scores[name]["subtasks"]}
        if len(rec) == len(arm_scores):
            subtask_table[s] = {
                "n": arm_scores["full"]["subtasks"][s]["n"],
                "scores": rec,
                "drop_abs_minus_full": round(rec["drop_abs"] - rec["full"], 6),
                "drop_copied_tok_minus_full": round(rec["drop_copied_tok"] - rec["full"], 6),
                "drop_abs_minus_drop_copied_tok": round(rec["drop_abs"] - rec["drop_copied_tok"], 6),
            }
    group_scores = {}
    for group_name, group_domains in [("relational_domains", RELATIONAL), ("adjacency_independent_domains", INDEPENDENT)]:
        vals = {name: group_acc(arm_scores[name]["domains"], group_domains) for name in arm_scores}
        group_scores[group_name] = {
            "scores": vals,
            "drop_abs_minus_full": round(vals["drop_abs"]["accuracy"] - vals["full"]["accuracy"], 6),
            "drop_copied_tok_minus_full": round(vals["drop_copied_tok"]["accuracy"] - vals["full"]["accuracy"], 6),
            "drop_abs_minus_drop_copied_tok": round(vals["drop_abs"]["accuracy"] - vals["drop_copied_tok"]["accuracy"], 6),
        }
    return {"arm_scores": arm_scores, "domain_table": domain_table, "subtask_table": subtask_table, "group_scores": group_scores}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = {name: load_json(path) for name, path in ARM_JSONS.items()}
    table = score_table(arms)
    ewok = ewok_table(arms)
    payload = {
        "status": "TARGETSELECT_EXTERNAL_COUPLING",
        "meaning": "Early 20M official-style coupling for target-selective arms. This should not decide route survival because compact-view advantages matured late; it only shows which surfaces already move.",
        "inputs": {"arm_jsons": {k: str(v) for k, v in ARM_JSONS.items()}, "gold_ewok": str(GOLD_EWOK)},
        "cheap7": table,
        "ewok": ewok,
        "scientific_interpretation": [
            "drop_abs improves EWoK and GlobalPIQA mean relative to full at 20M despite hurting source-absent fixed-event denoising, so early external score movement is not a direct readout of the internal channel.",
            "drop_abs versus token copied-drop has mixed columns: EWoK and GlobalPIQA higher for drop_abs, Supplement/Entity/Reading lower. Read this as early coupling/noise and wait for the whole-word copied-content control plus historical-geometry linkage."
        ],
    }
    out = out_dir / "targetselect_external_coupling.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# research target-selective early external coupling",
        "",
        "This summarizes the repaired research 20M official-style readout. It is not a route decision; the original compact-view advantage matured late.",
        "",
        "## cheap7-style scores",
        "",
        "| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | equal7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, sc in table["scores"].items():
        md.append("| " + name + " | " + " | ".join(f"{sc[c]:.3f}" for c in COLUMNS + ["cheap7_equal_mean"]) + " |")
    md += ["", "## Deltas", "", "| contrast | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | equal7 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, de in table["deltas"].items():
        md.append("| " + name + " | " + " | ".join(f"{de[c]:+.3f}" for c in COLUMNS + ["cheap7_equal_mean"]) + " |")
    md += ["", "## EWoK domain deltas: drop_abs minus token copied-drop", "", "| domain | n | delta pp | full | drop_abs | drop_copied_tok |", "|---|---:|---:|---:|---:|---:|"]
    for d, rec in sorted(ewok["domain_table"].items(), key=lambda kv: kv[1]["drop_abs_minus_drop_copied_tok"], reverse=True):
        sc = rec["scores"]
        md.append(f"| {d} | {rec['n']} | {rec['drop_abs_minus_drop_copied_tok']:+.3f} | {sc['full']:.2f} | {sc['drop_abs']:.2f} | {sc['drop_copied_tok']:.2f} |")
    md += ["", f"JSON: `{out}`", ""]
    (out_dir / "targetselect_external_coupling.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out), "cheap7_scores": table["scores"], "cheap7_deltas": table["deltas"], "ewok_group_scores": ewok["group_scores"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
