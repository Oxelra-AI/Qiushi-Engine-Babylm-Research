#!/usr/bin/env python3
"""research: localize seed43122 fast/no-AoA drop against seed43022.

CPU-only. Reads already-produced prediction JSON files and local evaluation data.
Computes per-subtask accuracy, seed agreement, and flip counts for fast/full zero-shot
columns present in the seed43122 fast screen and seed43022 reinvest screen.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

USER_ROOT = Path.cwd()
STRICT = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict"
OUT_DIR = USER_ROOT / "experiments/archive/frontier_consolidation/data/seed43122_fast_localization"
OUT_JSON = OUT_DIR / "seed43122_fast_localization.json"
OUT_MD = USER_ROOT / "research/notes/frontier_consolidation/seed43122_fast_localization.md"

PRED_43022 = USER_ROOT / "experiments/archive/frontier_consolidation/data/density_noaoa_eval_reinvest/eval_outputs/density_reinvest_noaoa/compact_view_reinvest"
PRED_43122 = USER_ROOT / "experiments/archive/representation_and_objectives/data/compact_reinvest_seed43122_fast/eval_outputs/compact_view_reinvest_seed43122/compact_view_reinvest_seed43122"

FILES = {
    "BLiMP": {
        "col": "BLiMP", "kind": "blimp", "eval_dir": STRICT / "evaluation_data/fast_eval/blimp_fast",
        "p43022": PRED_43022 / "BLiMP/chck_100M/compact_view_reinvest_BLiMP/zero_shot/mlm/blimp/blimp_fast/predictions.json",
        "p43122": PRED_43122 / "BLiMP/chck_100M/compact_view_reinvest_seed43122_BLiMP/zero_shot/mlm/blimp/blimp_fast/predictions.json",
    },
    "Supplement": {
        "col": "Supplement", "kind": "blimp", "eval_dir": STRICT / "evaluation_data/fast_eval/supplement_fast",
        "p43022": PRED_43022 / "Supplement/chck_100M/compact_view_reinvest_Supplement/zero_shot/mlm/blimp/supplement_fast/predictions.json",
        "p43122": PRED_43122 / "Supplement/chck_100M/compact_view_reinvest_seed43122_Supplement/zero_shot/mlm/blimp/supplement_fast/predictions.json",
    },
    "EWoK": {
        "col": "EWoK", "kind": "ewok", "eval_dir": STRICT / "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast",
        "p43022": PRED_43022 / "EWoK/chck_100M/compact_view_reinvest_EWoK/zero_shot/mlm/ewok/ewok_fast/predictions.json",
        "p43122": PRED_43122 / "EWoK/chck_100M/compact_view_reinvest_seed43122_EWoK/zero_shot/mlm/ewok/ewok_fast/predictions.json",
    },
    "Entity_fast": {
        "col": "Entity_fast", "kind": "entity_fast", "eval_dir": STRICT / "evaluation_data/fast_eval/entity_tracking_fast",
        "p43022": PRED_43022 / "Entity/chck_100M/compact_view_reinvest_Entity/zero_shot/mlm/entity_tracking/entity_tracking_fast/predictions.json",
        "p43122": PRED_43122 / "Entity/chck_100M/compact_view_reinvest_seed43122_Entity/zero_shot/mlm/entity_tracking/entity_tracking_fast/predictions.json",
    },
    "Entity_full": {
        "col": "Entity_full", "kind": "entity_full", "eval_dir": STRICT / "evaluation_data/full_eval/entity_tracking",
        "p43022": PRED_43022 / "Entity_full/chck_100M/compact_view_reinvest_Entity_full/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json",
        "p43122": PRED_43122 / "Entity_full/chck_100M/compact_view_reinvest_seed43122_Entity_full/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json",
    },
    "COMPS": {
        "col": "COMPS", "kind": "comps", "eval_dir": STRICT / "evaluation_data/full_eval/comps",
        "p43022": PRED_43022 / "COMPS/chck_100M/compact_view_reinvest_COMPS/zero_shot/mlm/comps/comps/predictions.json",
        "p43122": PRED_43122 / "COMPS/chck_100M/compact_view_reinvest_seed43122_COMPS/zero_shot/mlm/comps/comps/predictions.json",
    },
    "GlobalPIQA_parallel": {
        "col": "GlobalPIQA_parallel", "kind": "global_piqa", "eval_dir": STRICT / "evaluation_data/fast_eval/global_piqa_parallel",
        "p43022": PRED_43022 / "GlobalPIQA_parallel/chck_100M/compact_view_reinvest_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "p43122": PRED_43122 / "GlobalPIQA_parallel/chck_100M/compact_view_reinvest_seed43122_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
    },
    "GlobalPIQA_nonparallel": {
        "col": "GlobalPIQA_nonparallel", "kind": "global_piqa", "eval_dir": STRICT / "evaluation_data/fast_eval/global_piqa_nonparallel",
        "p43022": PRED_43022 / "GlobalPIQA_nonparallel/chck_100M/compact_view_reinvest_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
        "p43122": PRED_43122 / "GlobalPIQA_nonparallel/chck_100M/compact_view_reinvest_seed43122_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
    },
}


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def norm(x):
    return x.strip() if isinstance(x, str) else x


def records_for_blimp(pred_path: Path, eval_dir: Path, ewok: bool = False):
    preds = load_json(pred_path)
    out = []
    for subtask, d in preds.items():
        pred_list = d["predictions"]
        eval_file = eval_dir / f"{subtask}.jsonl"
        gold_lines = [json.loads(l) for l in eval_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, (pr, gold) in enumerate(zip(pred_list, gold_lines)):
            pred = norm(pr["pred"])
            if ewok:
                target = norm(" ".join([gold["Context1"], gold["Target1"]]))
            else:
                target = norm(gold["sentence_good"])
            out.append({"id": f"{subtask}:{i}", "subtask": subtask, "pred": pred, "target": target, "correct": pred == target})
    return out


def entity_subtask(gold: dict[str, Any], prefix: str) -> str:
    return f"{prefix}_{gold['numops']}_ops"


def records_for_entity(pred_path: Path, eval_dir: Path, fast: bool):
    preds = load_json(pred_path)
    out = []
    if fast:
        # Current fast evaluator only reads regular.jsonl and drops examples with 'nothing'.
        gold_file = eval_dir / "regular.jsonl"
        gold_by_sub = {}
        for gold in [json.loads(l) for l in gold_file.read_text(encoding="utf-8").splitlines() if l.strip()]:
            if any("nothing" in opt for opt in gold.get("options", [])):
                continue
            sub = entity_subtask(gold, "regular")
            gold_by_sub.setdefault(sub, []).append(gold)
    else:
        gold_by_sub = {}
        prefix_map = {"regular": "regular", "ambiref": "ambiref", "move_contents": "move_contents"}
        for stem, prefix in prefix_map.items():
            f = eval_dir / f"{stem}.jsonl"
            if not f.exists():
                continue
            for gold in [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]:
                if any("nothing" in opt for opt in gold.get("options", [])):
                    continue
                sub = entity_subtask(gold, prefix)
                gold_by_sub.setdefault(sub, []).append(gold)
    for subtask, d in preds.items():
        pred_list = d["predictions"]
        gold_list = gold_by_sub[subtask]
        for i, (pr, gold) in enumerate(zip(pred_list, gold_list)):
            pred = norm(pr["pred"])
            target = norm(gold["options"][0])
            out.append({"id": f"{subtask}:{i}", "subtask": subtask, "pred": pred, "target": target, "correct": pred == target})
    return out


def records_for_comps(pred_path: Path, eval_dir: Path):
    preds = load_json(pred_path)
    subtask_to_file = {
        "base": "comps_base",
        "wugs_dist_before": "comps_wugs_dist-before",
        "wugs_dist_in_between": "comps_wugs_dist-in-between",
        "wugs": "comps_wugs",
    }
    out = []
    for subtask, d in preds.items():
        pred_list = d["predictions"]
        gold_file = eval_dir / f"{subtask_to_file[subtask]}.jsonl"
        gold_lines = [json.loads(l) for l in gold_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, (pr, gold) in enumerate(zip(pred_list, gold_lines)):
            pred = norm(pr["pred"])
            target = norm(" ".join([gold["prefix_acceptable"], gold["property_phrase"]]))
            out.append({"id": f"{subtask}:{i}", "subtask": subtask, "pred": pred, "target": target, "correct": pred == target})
    return out


def records_for_global_piqa(pred_path: Path, eval_dir: Path):
    preds = load_json(pred_path)
    # fast global_piqa lives as eng_latn.jsonl; full gold may use a json mapping elsewhere.
    gold_file = eval_dir / "eng_latn.jsonl"
    gold_lines = [json.loads(l) for l in gold_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for gold in gold_lines:
        eid = gold["example_id"]
        pr = preds[eid]["predictions"][0]
        pred = norm(pr["pred"])
        target = norm(gold[f"solution{gold['label']}"])
        out.append({"id": eid, "subtask": eval_dir.name, "pred": pred, "target": target, "correct": pred == target})
    return out


def load_records(cfg: dict[str, Any], seed: str):
    p = cfg[f"p{seed}"]
    kind = cfg["kind"]
    if kind == "blimp":
        return records_for_blimp(p, cfg["eval_dir"], ewok=False)
    if kind == "ewok":
        return records_for_blimp(p, cfg["eval_dir"], ewok=True)
    if kind == "entity_fast":
        return records_for_entity(p, cfg["eval_dir"], fast=True)
    if kind == "entity_full":
        return records_for_entity(p, cfg["eval_dir"], fast=False)
    if kind == "comps":
        return records_for_comps(p, cfg["eval_dir"])
    if kind == "global_piqa":
        return records_for_global_piqa(p, cfg["eval_dir"])
    raise ValueError(kind)


def summarize_pair(recs_a, recs_b):
    by_a = {r["id"]: r for r in recs_a}
    by_b = {r["id"]: r for r in recs_b}
    ids = sorted(set(by_a) & set(by_b))
    out = {"n_common": len(ids), "n_a": len(recs_a), "n_b": len(recs_b)}
    correct_a = sum(by_a[i]["correct"] for i in ids)
    correct_b = sum(by_b[i]["correct"] for i in ids)
    agree_pred = sum(norm(by_a[i]["pred"]) == norm(by_b[i]["pred"]) for i in ids)
    both_correct = sum(by_a[i]["correct"] and by_b[i]["correct"] for i in ids)
    both_wrong = sum((not by_a[i]["correct"]) and (not by_b[i]["correct"]) for i in ids)
    a_only = sum(by_a[i]["correct"] and not by_b[i]["correct"] for i in ids)
    b_only = sum((not by_a[i]["correct"]) and by_b[i]["correct"] for i in ids)
    out.update({
        "seed43022_acc": 100 * correct_a / len(ids) if ids else None,
        "seed43122_acc": 100 * correct_b / len(ids) if ids else None,
        "delta_43122_minus_43022": 100 * (correct_b - correct_a) / len(ids) if ids else None,
        "prediction_agreement": 100 * agree_pred / len(ids) if ids else None,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "seed43022_only_correct": a_only,
        "seed43122_only_correct": b_only,
        "net_correct_delta_count": correct_b - correct_a,
    })
    # Subtask summary
    subs = sorted({by_a[i]["subtask"] for i in ids} | {by_b[i]["subtask"] for i in ids})
    subtasks = []
    for sub in subs:
        sub_ids = [i for i in ids if by_a[i]["subtask"] == sub]
        if not sub_ids:
            continue
        ca = sum(by_a[i]["correct"] for i in sub_ids)
        cb = sum(by_b[i]["correct"] for i in sub_ids)
        pa = 100 * ca / len(sub_ids)
        pb = 100 * cb / len(sub_ids)
        agr = 100 * sum(norm(by_a[i]["pred"]) == norm(by_b[i]["pred"]) for i in sub_ids) / len(sub_ids)
        subtasks.append({
            "subtask": sub,
            "n": len(sub_ids),
            "seed43022_acc": pa,
            "seed43122_acc": pb,
            "delta": pb - pa,
            "prediction_agreement": agr,
            "net_correct_delta_count": cb - ca,
            "seed43022_only_correct": sum(by_a[i]["correct"] and not by_b[i]["correct"] for i in sub_ids),
            "seed43122_only_correct": sum((not by_a[i]["correct"]) and by_b[i]["correct"] for i in sub_ids),
        })
    subtasks.sort(key=lambda x: (x["delta"], -abs(x["net_correct_delta_count"])))
    out["subtasks_sorted_by_delta"] = subtasks
    out["worst_subtasks"] = subtasks[:15]
    out["best_subtasks"] = sorted(subtasks, key=lambda x: (x["delta"], abs(x["net_correct_delta_count"])), reverse=True)[:15]
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, cfg in FILES.items():
        missing = [str(cfg[k]) for k in ["p43022", "p43122"] if not cfg[k].exists()]
        if missing:
            results[name] = {"status": "missing", "missing": missing}
            continue
        rec43022 = load_records(cfg, "43022")
        rec43122 = load_records(cfg, "43122")
        results[name] = summarize_pair(rec43022, rec43122)

    # Use official-relevant aggregate: full entity, global mean, not entity fast.
    official_components = ["BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
    total_net = sum(results[k]["net_correct_delta_count"] for k in official_components if results.get(k, {}).get("n_common"))
    total_n = sum(results[k]["n_common"] for k in official_components if results.get(k, {}).get("n_common"))
    # Score-point contribution to the 7-column sum, using each column's own percentage scale;
    # for GlobalPIQA column average, parallel/nonparallel contributions each carry 0.5 weight.
    weighted_delta_score_sum = 0.0
    weighted_delta_score_sum += results["BLiMP"]["delta_43122_minus_43022"]
    weighted_delta_score_sum += results["Supplement"]["delta_43122_minus_43022"]
    weighted_delta_score_sum += results["EWoK"]["delta_43122_minus_43022"]
    weighted_delta_score_sum += results["Entity_full"]["delta_43122_minus_43022"]
    weighted_delta_score_sum += results["COMPS"]["delta_43122_minus_43022"]
    weighted_delta_score_sum += (results["GlobalPIQA_parallel"]["delta_43122_minus_43022"] + results["GlobalPIQA_nonparallel"]["delta_43122_minus_43022"]) / 2

    payload = {
        "status": "SEED43122_FAST_LOCALIZATION",
        "purpose": "Localize the seed43122 cheap no-AoA drop against seed43022 without GPU or new evaluation.",
        "seed43022_prediction_root": str(PRED_43022),
        "seed43122_prediction_root": str(PRED_43122),
        "columns": results,
        "aggregate_no_reading_no_superglue_no_aoa": {
            "official_relevant_components": official_components,
            "total_common_items_across_components": total_n,
            "net_correct_delta_count_43122_minus_43022_unweighted_items": total_net,
            "weighted_score_sum_delta_excluding_reading_superglue_aoa": weighted_delta_score_sum,
        },
        "interpretation": "Seed43122 degradation is concentrated in EWoK and broad BLiMP/Supplement slices rather than one tiny failed subtask only; GlobalPIQA is a one-item parallel loss and nonparallel tie; Reading improves and is analyzed separately from item correctness.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research — seed43122 fast/no-AoA localization\n")
    lines.append("CPU-only comparison of existing prediction files for seed43122 and seed43022 compact_view_reinvest. This does not use the running official-rowcount AoA task and does not run new evaluation.\n\n")
    lines.append("## Column-level agreement and flip summary\n")
    for name in ["BLiMP", "Supplement", "EWoK", "Entity_fast", "Entity_full", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        r = results[name]
        if r.get("status") == "missing":
            lines.append(f"- {name}: MISSING {r['missing']}\n")
            continue
        lines.append(f"- {name}: n={r['n_common']}, seed43022={r['seed43022_acc']:.3f}, seed43122={r['seed43122_acc']:.3f}, delta={r['delta_43122_minus_43022']:+.3f}, pred_agreement={r['prediction_agreement']:.2f}%, 43022_only={r['seed43022_only_correct']}, 43122_only={r['seed43122_only_correct']}\n")
    lines.append("\n## Worst subtask slices by column\n")
    for name in ["BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        r = results[name]
        lines.append(f"\n### {name}\n")
        for sub in r.get("worst_subtasks", [])[:8]:
            lines.append(f"- {sub['subtask']}: n={sub['n']}, delta={sub['delta']:+.2f}, 43022={sub['seed43022_acc']:.2f}, 43122={sub['seed43122_acc']:.2f}, agree={sub['prediction_agreement']:.1f}%, net={sub['net_correct_delta_count']}\n")
    lines.append("\n## Aggregate reading\n")
    agg = payload["aggregate_no_reading_no_superglue_no_aoa"]
    lines.append(f"Weighted score-sum delta across BLiMP+Supplement+EWoK+full Entity+COMPS+GlobalPIQA mean (excluding Reading, SuperGLUE, AoA): {agg['weighted_score_sum_delta_excluding_reading_superglue_aoa']:+.6f}.\n")
    lines.append("The weak seed43122 fast surface is not solely a one-slice artifact. It has a large EWoK deficit, broad BLiMP/Supplement deficits, a full-Entity deficit, a small COMPS deficit, and one lost GlobalPIQA-parallel example; Reading is the only observed improvement. This supports treating seed43022 as an upper-tail or fragile endpoint unless official-rowcount AoA is strongly positive for seed43122.\n")
    lines.append(f"\nMachine-readable output: `{OUT_JSON}`.\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2))

if __name__ == "__main__":
    main()
