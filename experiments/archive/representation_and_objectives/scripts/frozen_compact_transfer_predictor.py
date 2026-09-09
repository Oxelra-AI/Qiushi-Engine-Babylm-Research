#!/usr/bin/env python3
"""research: frozen-evidence test of compact-view transferability.

Question: after the RoBERTa HS/LS/HD/LD factorial failed at the stable-family item level,
can a concrete DeBERTa-specific compact-view signature predict both the large DeBERTa
compact gain and the RoBERTa/GPT failures from existing frozen artifacts?

This script does not train or evaluate models. It reconstructs item-level DeBERTa triangle
correctness from existing prediction files, joins to frozen RoBERTa compact-vs-repeat and
RoBERTa factorial item movement files, and compares DeBERTa compact-responsive items/subtasks
against the cross-coordinate movements.
"""
from __future__ import annotations

import collections
import csv
import json
import math
import pathlib
import statistics
from typing import Any, Dict, Iterable, List, Tuple

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
OUT_DIR = ROOT / "data" / "frozen_compact_transfer_predictor"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DeBERTa triangle prediction roots used by research.
VIEW_ROOT = pathlib.Path("experiments/archive/frontier_consolidation/data/density_noaoa_eval_reinvest/eval_outputs/density_reinvest_noaoa/compact_view_reinvest")
REPEAT_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/compact_triangle_noaoa_eval/eval_outputs/compact_triangle_noaoa_guarded/compact_repeat_reinvest")
ADJBREAK_ROOT = pathlib.Path("experiments/archive/representation_and_objectives/data/compact_triangle_noaoa_eval/eval_outputs/compact_triangle_noaoa_guarded/adjbreak_reinvest")
EVAL_DATA = pathlib.Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data")
FAST_EVAL = EVAL_DATA / "fast_eval"
FULL_EVAL = EVAL_DATA / "full_eval"

ROBERTA_CR_CSV = pathlib.Path("experiments/archive/frontier_consolidation/data/roberta100_selected_prediction_movement/per_item_movement.csv")
ROBERTA_FACT_CSV = pathlib.Path("experiments/archive/representation_and_objectives/data/factorial_item_interaction_terminal/factorial_item_interaction_rows.csv")
CAUSAL_GPT_JSON = pathlib.Path("experiments/archive/representation_and_objectives/data/compact_directional_cheap7_eval_r2/directional_interaction_readout.json")

TASKS = ["BLiMP", "Supplement", "EWoK", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
STABLE_NONBLIMP = {"Supplement", "EWoK", "COMPS"}
STABLE_WITH_BLIMP = {"BLiMP", "Supplement", "EWoK", "COMPS"}


def pred_path(arm_root: pathlib.Path, task: str) -> pathlib.Path:
    mapping = {
        "BLiMP": "BLiMP/chck_100M/*/zero_shot/mlm/blimp/blimp_fast/predictions.json",
        "Supplement": "Supplement/chck_100M/*/zero_shot/mlm/blimp/supplement_fast/predictions.json",
        "EWoK": "EWoK/chck_100M/*/zero_shot/mlm/ewok/ewok_fast/predictions.json",
        "COMPS": "COMPS/chck_100M/*/zero_shot/mlm/comps/comps/predictions.json",
        "GlobalPIQA_parallel": "GlobalPIQA_parallel/chck_100M/*/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "GlobalPIQA_nonparallel": "GlobalPIQA_nonparallel/chck_100M/*/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
    }
    candidates = list(arm_root.glob(mapping[task]))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected 1 prediction file for {task} in {arm_root}, found {len(candidates)}: {candidates}")
    return candidates[0]


def load_json(path: pathlib.Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_blimp_supplement(pred_dict: dict, gold_dir: pathlib.Path, task: str) -> List[dict]:
    rows = []
    for subtask, data in pred_dict.items():
        gold_file = gold_dir / f"{subtask}.jsonl"
        if not gold_file.exists():
            continue
        with open(gold_file, "r", encoding="utf-8") as gf:
            gold_rows = [json.loads(line) for line in gf]
        for idx, (pred_item, gold_row) in enumerate(zip(data["predictions"], gold_rows)):
            correct = pred_item["pred"].strip() == gold_row["sentence_good"].strip()
            rows.append({"item_id": f"{task}:{subtask}:{idx}", "column": task, "subtask": subtask, "correct": int(correct)})
    return rows


def score_ewok(pred_dict: dict, gold_dir: pathlib.Path) -> List[dict]:
    rows = []
    for subtask, data in pred_dict.items():
        gold_file = gold_dir / f"{subtask}.jsonl"
        if not gold_file.exists():
            continue
        with open(gold_file, "r", encoding="utf-8") as gf:
            gold_rows = [json.loads(line) for line in gf]
        for idx, (pred_item, gold_row) in enumerate(zip(data["predictions"], gold_rows)):
            target = " ".join([gold_row["Context1"], gold_row["Target1"]]).strip()
            correct = pred_item["pred"].strip() == target
            rows.append({"item_id": f"EWoK:{subtask}:{idx}", "column": "EWoK", "subtask": subtask, "correct": int(correct)})
    return rows


def score_comps(pred_dict: dict, gold_dir: pathlib.Path) -> List[dict]:
    subtask_to_file = {
        "base": "comps_base",
        "wugs_dist_before": "comps_wugs_dist-before",
        "wugs_dist_in_between": "comps_wugs_dist-in-between",
        "wugs": "comps_wugs",
    }
    rows = []
    for subtask, data in pred_dict.items():
        stem = subtask_to_file.get(subtask)
        if stem is None:
            continue
        gold_file = gold_dir / f"{stem}.jsonl"
        if not gold_file.exists():
            continue
        with open(gold_file, "r", encoding="utf-8") as gf:
            gold_rows = [json.loads(line) for line in gf]
        for idx, (pred_item, gold_row) in enumerate(zip(data["predictions"], gold_rows)):
            target = " ".join([gold_row["prefix_acceptable"], gold_row["property_phrase"]]).strip()
            correct = pred_item["pred"].strip() == target
            rows.append({"item_id": f"COMPS:{subtask}:{idx}", "column": "COMPS", "subtask": subtask, "correct": int(correct)})
    return rows


def score_global_piqa(pred_dict: dict, gold_file: pathlib.Path, task: str) -> List[dict]:
    rows = []
    with open(gold_file, "r", encoding="utf-8") as gf:
        gold_rows = [json.loads(line) for line in gf]
    gold_by_id = {row.get("example_id", ""): row for row in gold_rows}
    for example_id, data in pred_dict.items():
        gold_row = gold_by_id.get(example_id)
        if gold_row is None:
            continue
        label = gold_row["label"]
        target = gold_row[f"solution{label}"].strip()
        for idx, pred_item in enumerate(data["predictions"]):
            correct = pred_item["pred"].strip() == target
            rows.append({"item_id": f"{task}:{example_id}:{idx}", "column": task, "subtask": example_id, "correct": int(correct)})
    return rows


def score_arm(arm_root: pathlib.Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for task in TASKS:
        pred = load_json(pred_path(arm_root, task))
        if task == "BLiMP":
            rows = score_blimp_supplement(pred, FAST_EVAL / "blimp_fast", task)
        elif task == "Supplement":
            rows = score_blimp_supplement(pred, FAST_EVAL / "supplement_fast", task)
        elif task == "EWoK":
            rows = score_ewok(pred, FAST_EVAL / "evaluation_data" / "fast_eval" / "ewok_fast")
        elif task == "COMPS":
            rows = score_comps(pred, FULL_EVAL / "comps")
        elif task == "GlobalPIQA_parallel":
            rows = score_global_piqa(pred, FAST_EVAL / "global_piqa_parallel" / "eng_latn.jsonl", task)
        elif task == "GlobalPIQA_nonparallel":
            rows = score_global_piqa(pred, FAST_EVAL / "global_piqa_nonparallel" / "eng_latn.jsonl", task)
        else:
            rows = []
        for r in rows:
            out[r["item_id"]] = r
    return out


def load_roberta_compact_repeat() -> Dict[str, dict]:
    out = {}
    with open(ROBERTA_CR_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            left = int(r["left_correct"])
            right = int(r["right_correct"])
            out[r["item_id"]] = {
                "column": r["column"],
                "subtask": r["subtask"],
                "r_compact_minus_repeat": right - left,
                "r_left_correct": left,
                "r_right_correct": right,
            }
    return out


def load_roberta_factorial() -> Dict[str, dict]:
    out = {}
    with open(ROBERTA_FACT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out[r["item_id"]] = {
                "column": r["column"],
                "subtask": r["subtask"],
                "r_factorial_interaction": int(r["interaction"]),
                "r_same_anchor_effect": int(r["same_anchor_effect"]),
                "r_deranged_effect": int(r["deranged_effect"]),
            }
    return out


def safe_mean(xs: List[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def mean_pp(xs: List[float]) -> float | None:
    m = safe_mean(xs)
    return None if m is None else 100.0 * m


def se_pp(xs: List[float]) -> float | None:
    if len(xs) <= 1:
        return None
    m = safe_mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return 100.0 * math.sqrt(var / len(xs))


def summarize_values(xs: List[float]) -> dict:
    return {
        "n": len(xs),
        "mean_pp": mean_pp(xs),
        "se_pp": se_pp(xs),
        "sum": sum(xs),
        "counts": dict(collections.Counter(xs)),
    }


def grouped_means(joined: List[dict], selector_name: str, selector_key: str, outcome_key: str, columns: Iterable[str] | None = None) -> dict:
    cols = set(columns) if columns is not None else None
    groups: Dict[int, List[float]] = collections.defaultdict(list)
    for r in joined:
        if cols is not None and r["column"] not in cols:
            continue
        groups[int(r[selector_key])].append(float(r[outcome_key]))
    return {
        "selector": selector_name,
        "outcome": outcome_key,
        "columns": sorted(cols) if cols is not None else "all_joined",
        "by_selector_value": {str(k): summarize_values(v) for k, v in sorted(groups.items())},
        "gain_minus_loss_mean_pp": None if (1 not in groups or -1 not in groups) else 100.0 * (safe_mean(groups[1]) - safe_mean(groups[-1])),
    }


def aggregate_by_key(rows: List[dict], delta_key: str, key: str) -> Dict[str, dict]:
    buckets: Dict[str, List[dict]] = collections.defaultdict(list)
    for r in rows:
        buckets[str(r[key])].append(r)
    out = {}
    for name, rs in buckets.items():
        deltas = [float(r[delta_key]) for r in rs]
        out[name] = {
            "n": len(rs),
            "column": rs[0]["column"] if key == "subtask" else name,
            "delta_mean_pp": mean_pp(deltas),
            "delta_sum": sum(deltas),
        }
    return out


def pearson_pairs(a: Dict[str, dict], b: Dict[str, dict], field_a: str = "delta_mean_pp", field_b: str = "delta_mean_pp", min_n: int = 1) -> dict:
    keys = sorted(set(a) & set(b))
    xs, ys, weights = [], [], []
    for k in keys:
        if a[k].get("n", 0) < min_n or b[k].get("n", 0) < min_n:
            continue
        va = a[k].get(field_a)
        vb = b[k].get(field_b)
        if va is None or vb is None:
            continue
        xs.append(float(va)); ys.append(float(vb)); weights.append(min(a[k].get("n", 1), b[k].get("n", 1)))
    def corr(x, y):
        n = len(x)
        if n < 2:
            return None
        mx, my = sum(x)/n, sum(y)/n
        vx = sum((u-mx)**2 for u in x)
        vy = sum((v-my)**2 for v in y)
        if vx <= 0 or vy <= 0:
            return None
        return sum((u-mx)*(v-my) for u,v in zip(x,y)) / math.sqrt(vx*vy)
    def wcorr(x, y, w):
        if len(x) < 2 or sum(w) <= 0:
            return None
        sw = sum(w)
        mx = sum(wi*xi for wi,xi in zip(w,x))/sw
        my = sum(wi*yi for wi,yi in zip(w,y))/sw
        vx = sum(wi*(xi-mx)**2 for wi,xi in zip(w,x))
        vy = sum(wi*(yi-my)**2 for wi,yi in zip(w,y))
        if vx <= 0 or vy <= 0:
            return None
        return sum(wi*(xi-mx)*(yi-my) for wi,xi,yi in zip(w,x,y)) / math.sqrt(vx*vy)
    return {
        "n_keys": len(xs),
        "keys": keys,
        "pearson": corr(xs, ys),
        "weighted_pearson": wcorr(xs, ys, weights),
        "mean_x": safe_mean(xs),
        "mean_y": safe_mean(ys),
    }


def main() -> None:
    print("loading DeBERTa triangle predictions", flush=True)
    view = score_arm(VIEW_ROOT)
    rep = score_arm(REPEAT_ROOT)
    adj = score_arm(ADJBREAK_ROOT)
    print({"view": len(view), "repeat": len(rep), "adjbreak": len(adj)}, flush=True)

    roberta_cr = load_roberta_compact_repeat()
    roberta_fact = load_roberta_factorial()

    common = sorted(set(view) & set(rep) & set(adj) & set(roberta_cr) & set(roberta_fact))
    joined = []
    for item_id in common:
        meta = view[item_id]
        # Trust the RoBERTa item files for the common ID; the official item identities match.
        col = meta["column"]
        sub = meta["subtask"]
        row = {
            "item_id": item_id,
            "column": col,
            "subtask": sub,
            "d_view_minus_repeat": int(view[item_id]["correct"]) - int(rep[item_id]["correct"]),
            "d_view_minus_adjbreak": int(view[item_id]["correct"]) - int(adj[item_id]["correct"]),
            "d_adjbreak_minus_repeat": int(adj[item_id]["correct"]) - int(rep[item_id]["correct"]),
            "r_compact_minus_repeat": roberta_cr[item_id]["r_compact_minus_repeat"],
            "r_factorial_interaction": roberta_fact[item_id]["r_factorial_interaction"],
            "r_same_anchor_effect": roberta_fact[item_id]["r_same_anchor_effect"],
            "r_deranged_effect": roberta_fact[item_id]["r_deranged_effect"],
        }
        joined.append(row)

    # Per-column direct item summaries.
    by_col = {}
    for col in sorted(set(r["column"] for r in joined)):
        rs = [r for r in joined if r["column"] == col]
        by_col[col] = {
            "n": len(rs),
            "deberta_view_minus_repeat_pp": mean_pp([r["d_view_minus_repeat"] for r in rs]),
            "deberta_view_minus_adjbreak_pp": mean_pp([r["d_view_minus_adjbreak"] for r in rs]),
            "deberta_adjbreak_minus_repeat_pp": mean_pp([r["d_adjbreak_minus_repeat"] for r in rs]),
            "roberta_compact_minus_repeat_pp": mean_pp([r["r_compact_minus_repeat"] for r in rs]),
            "roberta_factorial_interaction_pp": mean_pp([r["r_factorial_interaction"] for r in rs]),
            "roberta_factorial_same_anchor_pp": mean_pp([r["r_same_anchor_effect"] for r in rs]),
            "roberta_factorial_deranged_pp": mean_pp([r["r_deranged_effect"] for r in rs]),
        }
    # Merge GlobalPIQA split.
    gp_rows = [r for r in joined if r["column"].startswith("GlobalPIQA")]
    if gp_rows:
        by_col["GlobalPIQA_all_items"] = {
            "n": len(gp_rows),
            "deberta_view_minus_repeat_pp": mean_pp([r["d_view_minus_repeat"] for r in gp_rows]),
            "deberta_view_minus_adjbreak_pp": mean_pp([r["d_view_minus_adjbreak"] for r in gp_rows]),
            "deberta_adjbreak_minus_repeat_pp": mean_pp([r["d_adjbreak_minus_repeat"] for r in gp_rows]),
            "roberta_compact_minus_repeat_pp": mean_pp([r["r_compact_minus_repeat"] for r in gp_rows]),
            "roberta_factorial_interaction_pp": mean_pp([r["r_factorial_interaction"] for r in gp_rows]),
            "roberta_factorial_same_anchor_pp": mean_pp([r["r_same_anchor_effect"] for r in gp_rows]),
            "roberta_factorial_deranged_pp": mean_pp([r["r_deranged_effect"] for r in gp_rows]),
        }

    conditional = []
    for selector_key in ["d_view_minus_repeat", "d_view_minus_adjbreak", "d_adjbreak_minus_repeat"]:
        for outcome_key in ["r_compact_minus_repeat", "r_factorial_interaction", "r_same_anchor_effect", "r_deranged_effect"]:
            conditional.append(grouped_means(joined, selector_key, selector_key, outcome_key, STABLE_WITH_BLIMP))
            conditional.append(grouped_means(joined, selector_key, selector_key, outcome_key, STABLE_NONBLIMP))

    # Subtask-level alignments for common subtasks.
    deb_vr_sub = aggregate_by_key(joined, "d_view_minus_repeat", "subtask")
    deb_va_sub = aggregate_by_key(joined, "d_view_minus_adjbreak", "subtask")
    deb_ar_sub = aggregate_by_key(joined, "d_adjbreak_minus_repeat", "subtask")
    rob_cr_sub = aggregate_by_key(joined, "r_compact_minus_repeat", "subtask")
    rob_fact_sub = aggregate_by_key(joined, "r_factorial_interaction", "subtask")

    subtask_alignment = {
        "deberta_view_repeat_vs_roberta_compact_repeat": pearson_pairs(deb_vr_sub, rob_cr_sub, min_n=20),
        "deberta_view_adjbreak_vs_roberta_factorial_interaction": pearson_pairs(deb_va_sub, rob_fact_sub, min_n=20),
        "deberta_adjbreak_repeat_vs_roberta_compact_repeat": pearson_pairs(deb_ar_sub, rob_cr_sub, min_n=20),
        "deberta_view_repeat_vs_roberta_factorial_interaction": pearson_pairs(deb_vr_sub, rob_fact_sub, min_n=20),
    }

    # Family-level causal GPT comparison: few points, included as directional frozen evidence only.
    causal = load_json(CAUSAL_GPT_JSON)["eval"]["interactions"]
    family_gpt = {}
    for fam in ["BLiMP", "Supplement", "EWoK", "COMPS", "GlobalPIQA"]:
        if fam == "GlobalPIQA":
            deb_rows = gp_rows
            deb_delta = mean_pp([r["d_view_minus_repeat"] for r in deb_rows]) if deb_rows else None
        else:
            deb_delta = by_col.get(fam, {}).get("deberta_view_minus_repeat_pp")
        if fam in causal:
            family_gpt[fam] = {
                "n_deberta_items": by_col.get(fam, {}).get("n", len(gp_rows) if fam == "GlobalPIQA" else None),
                "deberta_view_minus_repeat_pp": deb_delta,
                "causal_reciprocal_minus_oneway_pp": causal[fam]["reciprocal_minus_oneway"],
            }

    # Direct predictor verdict metrics.
    stable_nonblimp_rows = [r for r in joined if r["column"] in STABLE_NONBLIMP]
    stable_with_blimp_rows = [r for r in joined if r["column"] in STABLE_WITH_BLIMP]
    def net(rs, key): return mean_pp([r[key] for r in rs])
    key_verdict = {
        "joined_common_items": len(joined),
        "stable_with_blimp_items": len(stable_with_blimp_rows),
        "stable_nonblimp_items": len(stable_nonblimp_rows),
        "deberta_view_repeat_stable_nonblimp_pp": net(stable_nonblimp_rows, "d_view_minus_repeat"),
        "roberta_compact_repeat_stable_nonblimp_pp": net(stable_nonblimp_rows, "r_compact_minus_repeat"),
        "roberta_factorial_interaction_stable_nonblimp_pp": net(stable_nonblimp_rows, "r_factorial_interaction"),
        "deberta_view_repeat_stable_with_blimp_pp": net(stable_with_blimp_rows, "d_view_minus_repeat"),
        "roberta_compact_repeat_stable_with_blimp_pp": net(stable_with_blimp_rows, "r_compact_minus_repeat"),
        "roberta_factorial_interaction_stable_with_blimp_pp": net(stable_with_blimp_rows, "r_factorial_interaction"),
    }

    payload = {
        "status": "FROZEN_COMPACT_TRANSFER_PREDICTOR",
        "meaning": "Frozen-evidence item/subtask alignment test: does the DeBERTa compact-responsive signature predict RoBERTa compact/factorial and causal-GPT failures? No new training/evaluation.",
        "inputs": {
            "deberta_view_root": str(VIEW_ROOT),
            "deberta_repeat_root": str(REPEAT_ROOT),
            "deberta_adjbreak_root": str(ADJBREAK_ROOT),
            "roberta_compact_repeat_csv": str(ROBERTA_CR_CSV),
            "roberta_factorial_csv": str(ROBERTA_FACT_CSV),
            "causal_gpt_json": str(CAUSAL_GPT_JSON),
        },
        "key_verdict_metrics": key_verdict,
        "by_column": by_col,
        "conditional_outcomes_by_deberta_selector": conditional,
        "subtask_alignment": subtask_alignment,
        "family_level_causal_gpt_comparison": family_gpt,
        "interpretive_boundary": "A frozen predictor would need DeBERTa gains or source-own adjacency gains to select the same stable-family items/subtasks that RoBERTa/GPT move coherently. Flat or sign-incoherent association demotes compact views from the transferable route rather than authorizing more compact decomposition.",
    }
    out_json = OUT_DIR / "transfer_predictor_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")

    # Smaller CSV for subtask vectors.
    out_csv = OUT_DIR / "subtask_alignment_table.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subtask", "column", "n", "deberta_vr_pp", "deberta_va_pp", "deberta_ar_pp", "roberta_cr_pp", "roberta_factorial_pp"])
        for sub in sorted(set(deb_vr_sub) & set(rob_cr_sub) & set(rob_fact_sub)):
            w.writerow([
                sub,
                deb_vr_sub[sub].get("column"),
                deb_vr_sub[sub].get("n"),
                deb_vr_sub[sub].get("delta_mean_pp"),
                deb_va_sub[sub].get("delta_mean_pp"),
                deb_ar_sub[sub].get("delta_mean_pp"),
                rob_cr_sub[sub].get("delta_mean_pp"),
                rob_fact_sub[sub].get("delta_mean_pp"),
            ])

    md = (OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/frozen_compact_transfer_predictor/transfer_predictor_summary.md')
    lines = []
    lines.append("# research frozen compact-transfer predictor")
    lines.append("")
    lines.append("No training or model evaluation was run. This joins existing item predictions to ask whether DeBERTa compact-responsive items/subtasks predict RoBERTa and causal-GPT movement.")
    lines.append("")
    lines.append("## Key item-level net movements")
    lines.append("")
    lines.append("| item pool | n | DeBERTa view-repeat | RoBERTa compact-repeat | RoBERTa factorial interaction |")
    lines.append("|---|---:|---:|---:|---:|")
    def netf(rs, key):
        v = net(rs, key)
        return "NA" if v is None else f"{v:.4f}"
    for name, rs in [("stable_nonBLiMP (Supplement+EWoK+COMPS)", stable_nonblimp_rows), ("stable_with_BLiMP", stable_with_blimp_rows)]:
        lines.append(f"| {name} | {len(rs)} | {netf(rs, 'd_view_minus_repeat')} | {netf(rs, 'r_compact_minus_repeat')} | {netf(rs, 'r_factorial_interaction')} |")
    lines.append("")
    lines.append("## Column table")
    lines.append("")
    lines.append("| column | n | D view-repeat | D view-adj | D adj-repeat | R compact-repeat | R factorial |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for col, rec in by_col.items():
        if not isinstance(rec.get("n"), int):
            continue
        def fmt(x):
            return "NA" if x is None else f"{x:.4f}"
        lines.append(f"| {col} | {rec['n']} | {fmt(rec['deberta_view_minus_repeat_pp'])} | {fmt(rec['deberta_view_minus_adjbreak_pp'])} | {fmt(rec['deberta_adjbreak_minus_repeat_pp'])} | {fmt(rec['roberta_compact_minus_repeat_pp'])} | {fmt(rec['roberta_factorial_interaction_pp'])} |")
    lines.append("")
    lines.append("## Subtask-vector correlations")
    lines.append("")
    lines.append("| comparison | n subtasks | pearson | weighted pearson |")
    lines.append("|---|---:|---:|---:|")
    for name, rec in subtask_alignment.items():
        lines.append(f"| {name} | {rec['n_keys']} | {rec['pearson']} | {rec['weighted_pearson']} |")
    lines.append("")
    lines.append("## Frozen-evidence reading")
    lines.append("")
    lines.append("A compact-transfer predictor would require DeBERTa gains or DeBERTa source-own-adjacency gains to select stable-family items/subtasks that move coherently in RoBERTa or in the causal-GPT reciprocity surface. The full JSON contains conditional item means by DeBERTa selector value and the family-level GPT comparison.")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    lines.append(f"Subtask table: `{out_csv}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "out_md": str(md),
        "out_csv": str(out_csv),
        "joined_common_items": len(joined),
        "key_verdict_metrics": key_verdict,
        "subtask_alignment": subtask_alignment,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
