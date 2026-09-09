#!/usr/bin/env python3
"""research: Entity Tracking numops split for changed-state-bias threat.

File-only post-processing of official Entity prediction JSONs.  It recomputes the
same subtask accuracies used by the BabyLM evaluator and compares view-minus-
repeat by numops (0..5) and split (regular/ambiref/move_contents).

Scientific role: a simple shift toward predicting changed state should improve
higher-numops rows while harming or failing to help 0-op rows.  Genuine record
addressability should not be carried solely by a nonzero-op-vs-zero-op bias
profile.

No model inference, no training, no GPU, no official scoring job, no upload.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import pathlib
import re
import time
from collections import defaultdict
from typing import Any

ROOT = _public_path('.')
WS = _public_path('experiments/archive/frontier_consolidation')
OFFICIAL_ENTITY_ROOT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking')
FIRST_EVAL_ROOT = _public_path('experiments/archive/frontier_consolidation/data/dose_ladder_stable_eval/eval')
SECOND_EVAL_ROOT = _public_path('experiments/archive/frontier_consolidation/data/second_basin_entity_ewok_eval')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/entity_numops_bias_readout')
CHECKPOINTS = [f"chck_{i}M" for i in range(10, 101, 10)]
SPLITS = ["ambiref", "regular", "move_contents"]
NUMOPS = list(range(6))

ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "basin1_max_view": {
        "basin": "seed43022_first_basin",
        "data_arm": "view",
        "seed": 43022,
        "eval_root": FIRST_EVAL_ROOT,
        "target_prefix": "dose_max_view",
        "revision_prefix": "full_step256_dose_max_view",
        "layout": "research",
    },
    "basin1_max_repeat": {
        "basin": "seed43022_first_basin",
        "data_arm": "repeat",
        "seed": 43022,
        "eval_root": FIRST_EVAL_ROOT,
        "target_prefix": "dose_max_repeat",
        "revision_prefix": "full_step256_dose_max_repeat",
        "layout": "research",
    },
    "basin2_max_view": {
        "basin": "seed43122_second_basin",
        "data_arm": "view",
        "seed": 43122,
        "eval_root": _public_path('experiments/archive/frontier_consolidation/data/second_basin_entity_ewok_eval/view_eval'),
        "target_prefix": "second_basin_max_view_seed43122",
        "revision_prefix": "full_step268_second_basin_max_view_seed43122",
        "layout": "arm",
    },
    "basin2_max_repeat": {
        "basin": "seed43122_second_basin",
        "data_arm": "repeat",
        "seed": 43122,
        "eval_root": _public_path('experiments/archive/frontier_consolidation/data/second_basin_entity_ewok_eval/repeat_eval'),
        "target_prefix": "second_basin_max_repeat_seed43122",
        "revision_prefix": "full_step268_second_basin_max_repeat_seed43122",
        "layout": "arm",
    },
    "roberta_max_view": {
        "basin": "roberta_seed43022",
        "data_arm": "view",
        "seed": 43022,
        "eval_root": _public_path('experiments/archive/frontier_consolidation/data/roberta_viewclean_total_stable_eval/eval'),
        "target_prefix": "roberta_max_view",
        "revision_prefix": "full_step260_roberta_max_view",
        "layout": "research",
    },
    "roberta_max_repeat": {
        "basin": "roberta_seed43022",
        "data_arm": "repeat",
        "seed": 43022,
        "eval_root": _public_path('experiments/archive/frontier_consolidation/data/roberta_maxdose_repeat_eval/eval'),
        "target_prefix": "roberta_max_repeat",
        "revision_prefix": "full_step269_roberta_max_repeat",
        "layout": "arm",
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_predictions_by_glob(eval_root: pathlib.Path, target: str) -> pathlib.Path | None:
    candidates = sorted((eval_root / "official_outputs" / target / "Entity").glob("**/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json"))
    if candidates:
        return candidates[0]
    # Some wrapper out_roots may be passed as the official_outputs parent or as an arm root.
    candidates = sorted(eval_root.glob(f"**/{target}/Entity/**/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json"))
    if candidates:
        return candidates[0]
    candidates = sorted(eval_root.glob(f"**/{target}*Entity*/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json"))
    if candidates:
        return candidates[0]
    return None


def pred_path(cfg: dict[str, Any], ck: str) -> pathlib.Path | None:
    cknum = ck.replace("chck_", "").replace("M", "")
    target = f"{cfg['target_prefix']}_{ck}"
    # Known research/research layout.
    p = pathlib.Path(cfg["eval_root"]) / "official_outputs" / target / "Entity" / ck / f"{cfg['revision_prefix']}_{ck}_Entity" / "zero_shot" / "mlm" / "entity_tracking" / "entity_tracking" / "predictions.json"
    if p.exists():
        return p
    # Known research arm out_root layout if the CPU-safe worker uses the same target string.
    p2 = pathlib.Path(cfg["eval_root"]) / "official_outputs" / target / "Entity" / ck / f"{cfg['revision_prefix']}_{ck}_Entity" / "zero_shot" / "mlm" / "entity_tracking" / "entity_tracking" / "predictions.json"
    if p2.exists():
        return p2
    found = find_predictions_by_glob(pathlib.Path(cfg["eval_root"]), target)
    if found and found.exists():
        return found
    # Fallback: use any predictions path containing target and Entity.
    return None


def load_gold() -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for split in SPLITS:
        rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        path = OFFICIAL_ENTITY_ROOT / f"{split}.jsonl"
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # Leaderboard scoring skips rows whose gold option is literally nothing.
                if str(obj["options"][0]).strip() == "nothing.":
                    continue
                rows[int(obj["numops"])].append(obj)
        for n in NUMOPS:
            out[(split, n)] = rows.get(n, [])
    return out


def load_predictions(path: pathlib.Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    raw = read_json(path)
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for split in SPLITS:
        for n in NUMOPS:
            key = f"{split}_{n}_ops"
            rec = raw.get(key)
            if isinstance(rec, dict):
                out[(split, n)] = list(rec.get("predictions") or [])
            else:
                out[(split, n)] = []
    return out


def score_predictions(preds: dict[tuple[str, int], list[dict[str, Any]]], gold: dict[tuple[str, int], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_correct = 0
    total_n = 0
    subtask_scores: list[float] = []
    for split in SPLITS:
        for n in NUMOPS:
            g = gold[(split, n)]
            p = preds[(split, n)]
            m = min(len(g), len(p))
            corr = 0
            pred_option_index_counts: dict[int, int] = defaultdict(int)
            for i in range(m):
                pred = str(p[i].get("pred", "")).strip()
                options = [str(x).strip() for x in g[i].get("options", [])]
                if pred == options[0]:
                    corr += 1
                try:
                    oi = options.index(pred)
                except ValueError:
                    oi = -1
                pred_option_index_counts[int(oi)] += 1
            acc = corr / m if m else None
            if acc is not None:
                subtask_scores.append(100.0 * acc)
                total_correct += corr
                total_n += m
            rows.append({
                "split": split,
                "numops": n,
                "n_gold_filtered": len(g),
                "n_pred": len(p),
                "n_scored": m,
                "correct": corr,
                "accuracy": acc,
                "score_pp": None if acc is None else 100.0 * acc,
                "pred_option0_frac": None if m == 0 else pred_option_index_counts.get(0, 0) / m,
                "pred_option1_frac": None if m == 0 else pred_option_index_counts.get(1, 0) / m,
                "pred_other_option_frac": None if m == 0 else sum(v for k, v in pred_option_index_counts.items() if k not in (0, 1)) / m,
                "pred_not_in_options_frac": None if m == 0 else pred_option_index_counts.get(-1, 0) / m,
            })
    summary = {
        "official_style_entity_score": sum(subtask_scores) / len(subtask_scores) if subtask_scores else None,
        "micro_accuracy": total_correct / total_n if total_n else None,
        "micro_n": total_n,
        "subtask_count": len(subtask_scores),
    }
    return rows, summary


def enrich_group(rows: list[dict[str, Any]], fields: dict[str, Any]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("accuracy") is not None]
    if not valid:
        return {**fields, "n_rows": 0, "macro_score_pp": None, "micro_accuracy": None, "n_scored": 0}
    n_scored = sum(int(r["n_scored"]) for r in valid)
    corr = sum(int(r["correct"]) for r in valid)
    return {
        **fields,
        "n_rows": len(valid),
        "macro_score_pp": sum(float(r["score_pp"]) for r in valid) / len(valid),
        "micro_accuracy": corr / n_scored if n_scored else None,
        "n_scored": n_scored,
        "pred_option0_frac_micro": sum(float(r.get("pred_option0_frac") or 0.0) * int(r["n_scored"]) for r in valid) / n_scored if n_scored else None,
        "pred_not_in_options_frac_micro": sum(float(r.get("pred_not_in_options_frac") or 0.0) * int(r["n_scored"]) for r in valid) / n_scored if n_scored else None,
    }


def aggregate_arm(rows: list[dict[str, Any]], summary: dict[str, Any], meta: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.append({**meta, "group": "all_18_subtasks", "numops_group": "all", "split_group": "all", **summary})
    for n in NUMOPS:
        out.append(enrich_group([r for r in rows if r["numops"] == n], {**meta, "group": f"numops_{n}", "numops_group": str(n), "split_group": "all"}))
    out.append(enrich_group([r for r in rows if r["numops"] == 0], {**meta, "group": "zero_ops", "numops_group": "zero", "split_group": "all"}))
    out.append(enrich_group([r for r in rows if r["numops"] > 0], {**meta, "group": "nonzero_ops", "numops_group": "nonzero", "split_group": "all"}))
    for split in SPLITS:
        out.append(enrich_group([r for r in rows if r["split"] == split], {**meta, "group": f"split_{split}", "numops_group": "all", "split_group": split}))
        out.append(enrich_group([r for r in rows if r["split"] == split and r["numops"] == 0], {**meta, "group": f"{split}_zero_ops", "numops_group": "zero", "split_group": split}))
        out.append(enrich_group([r for r in rows if r["split"] == split and r["numops"] > 0], {**meta, "group": f"{split}_nonzero_ops", "numops_group": "nonzero", "split_group": split}))
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def pair_deltas(agg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for r in agg_rows:
        index[(r["basin"], r["checkpoint"], r["data_arm"], r["group"])] = r
    out: list[dict[str, Any]] = []
    for basin in sorted({r["basin"] for r in agg_rows}):
        if not ("deberta" in basin or "seed" in basin or basin.startswith("seed")):
            pass
        for ck in CHECKPOINTS:
            groups = sorted({r["group"] for r in agg_rows if r["basin"] == basin and r["checkpoint"] == ck})
            for group in groups:
                v = index.get((basin, ck, "view", group))
                rp = index.get((basin, ck, "repeat", group))
                if not v or not rp:
                    continue
                def diff(k: str) -> float | None:
                    a = v.get(k); b = rp.get(k)
                    return None if a is None or b is None else float(a) - float(b)
                out.append({
                    "basin": basin,
                    "checkpoint": ck,
                    "words": int(ck.replace("chck_", "").replace("M", "")) * 1_000_000,
                    "group": group,
                    "numops_group": v.get("numops_group"),
                    "split_group": v.get("split_group"),
                    "view_macro_score_pp": v.get("macro_score_pp", v.get("official_style_entity_score")),
                    "repeat_macro_score_pp": rp.get("macro_score_pp", rp.get("official_style_entity_score")),
                    "macro_delta_pp_view_minus_repeat": diff("macro_score_pp") if v.get("macro_score_pp") is not None else diff("official_style_entity_score"),
                    "micro_accuracy_delta_view_minus_repeat": diff("micro_accuracy"),
                    "pred_option0_frac_delta_view_minus_repeat": diff("pred_option0_frac_micro"),
                    "pred_not_in_options_frac_delta_view_minus_repeat": diff("pred_not_in_options_frac_micro"),
                    "n_scored_view": v.get("n_scored", v.get("micro_n")),
                    "n_scored_repeat": rp.get("n_scored", rp.get("micro_n")),
                })
    return out


def write_md(payload: dict[str, Any], path: pathlib.Path) -> None:
    deltas = payload["pair_deltas"]
    lines: list[str] = []
    lines.append("# research Entity numops changed-state-bias readout\n\n")
    lines.append("File-only recomputation from official Entity prediction JSONs. The key split is 0-op versus nonzero-op rows. Bias toward changed state predicts nonzero-op gains with 0-op loss or no help.\n\n")
    lines.append("## MAX view-minus-repeat by basin/checkpoint\n\n")
    lines.append("| basin | checkpoint | all Δ pp | zero-op Δ pp | nonzero-op Δ pp | nonzero minus zero Δ pp |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    by = defaultdict(dict)
    for r in deltas:
        if r["group"] in {"all_18_subtasks", "zero_ops", "nonzero_ops"}:
            by[(r["basin"], r["checkpoint"])][r["group"]] = r
    for (basin, ck), d in sorted(by.items(), key=lambda x: (x[0][0], int(x[0][1].replace('chck_', '').replace('M','')))):
        all_d = d.get("all_18_subtasks", {}).get("macro_delta_pp_view_minus_repeat")
        z = d.get("zero_ops", {}).get("macro_delta_pp_view_minus_repeat")
        nz = d.get("nonzero_ops", {}).get("macro_delta_pp_view_minus_repeat")
        spread = None if z is None or nz is None else nz - z
        def fmt(x: Any) -> str:
            return "NA" if x is None else f"{float(x):+.3f}"
        lines.append(f"| {basin} | {ck} | {fmt(all_d)} | {fmt(z)} | {fmt(nz)} | {fmt(spread)} |\n")
    lines.append("\n## Interpretation\n\n")
    for k, v in payload.get("interpretation", {}).items():
        lines.append(f"- **{k}**: {v}\n")
    lines.append("\nCSV files in this directory contain all split and per-subtask rows.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--checkpoints", nargs="*", default=CHECKPOINTS)
    ap.add_argument("--arms", nargs="*", default=None, choices=sorted(ARM_CONFIGS))
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.checkpoints or CHECKPOINTS
    arms = args.arms or ["basin1_max_view", "basin1_max_repeat", "basin2_max_view", "basin2_max_repeat"]

    path_rows: list[dict[str, Any]] = []
    for arm in arms:
        cfg = ARM_CONFIGS[arm]
        for ck in checkpoints:
            p = pred_path(cfg, ck)
            path_rows.append({
                "arm_key": arm,
                "basin": cfg["basin"],
                "data_arm": cfg["data_arm"],
                "checkpoint": ck,
                "target": f"{cfg['target_prefix']}_{ck}",
                "predictions": None if p is None else rel(p),
                "exists": bool(p and p.exists()),
            })
    missing = [r for r in path_rows if not r["exists"]]
    plan = {
        "status": "ENTITY_NUMOPS_BIAS_PLAN" if args.plan_only else "ENTITY_NUMOPS_BIAS_START",
        "created_utc": now(),
        "official_entity_root": rel(OFFICIAL_ENTITY_ROOT),
        "arms": arms,
        "checkpoints": checkpoints,
        "prediction_paths": path_rows,
        "missing_count": len(missing),
        "scientific_role": "File-only split of Entity Tracking scores by numops to test changed-state-bias alternative before spending H100 on permuted companion.",
        "no_model_inference_training_gpu_upload_or_leaderboard": True,
    }
    (out_dir / "entity_numops_bias_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    if missing and not args.allow_missing:
        raise SystemExit(f"missing {len(missing)} prediction files; rerun with --allow-missing for partial readout")

    gold = load_gold()
    subtask_rows_all: list[dict[str, Any]] = []
    agg_rows_all: list[dict[str, Any]] = []
    for row in path_rows:
        if not row["exists"]:
            continue
        cfg = ARM_CONFIGS[row["arm_key"]]
        p = ROOT / row["predictions"] if row["predictions"] and not pathlib.Path(row["predictions"]).is_absolute() else pathlib.Path(row["predictions"])
        preds = load_predictions(p)
        subtask_rows, summary = score_predictions(preds, gold)
        meta = {
            "arm_key": row["arm_key"],
            "basin": cfg["basin"],
            "data_arm": cfg["data_arm"],
            "seed": cfg["seed"],
            "checkpoint": row["checkpoint"],
            "words": int(row["checkpoint"].replace("chck_", "").replace("M", "")) * 1_000_000,
            "predictions": row["predictions"],
        }
        for sr in subtask_rows:
            subtask_rows_all.append({**meta, **sr})
        agg_rows_all.extend(aggregate_arm(subtask_rows, summary, meta))
    deltas = pair_deltas(agg_rows_all)
    key_rows = [r for r in deltas if r["group"] in {"all_18_subtasks", "zero_ops", "nonzero_ops"}]
    # Late 80/90/100 summary for DeBERTa basins only.
    late = [r for r in deltas if r["checkpoint"] in {"chck_80M", "chck_90M", "chck_100M"}]
    late_by_group = defaultdict(list)
    for r in late:
        late_by_group[(r["basin"], r["group"])].append(r)
    late_summary = {}
    for k, rs in sorted(late_by_group.items()):
        vals = [r["macro_delta_pp_view_minus_repeat"] for r in rs if r.get("macro_delta_pp_view_minus_repeat") is not None]
        if vals:
            late_summary[f"{k[0]}::{k[1]}"] = sum(vals) / len(vals)
    interp = {
        "missing_prediction_files": f"{len(missing)} missing prediction files; partial readouts are explicitly marked in the plan.",
        "late_zero_nonzero_test": "Compare late_summary zero_ops and nonzero_ops. A larger nonzero-minus-zero spread supports the changed-state-bias alternative; positive zero-op together with nonzero gains weakens pure bias.",
        "permuted_card_decision": "If second-basin official Entity reproduces but numops movement is mostly nonzero-op gain with zero-op loss, delay permuted training and reinterpret Entity as a bias-sensitive benchmark carrier. If zero-op is not harmed and binding affected/unaffected does not show the signed bias, the permuted correspondence arm remains decisive.",
    }
    payload = {
        **plan,
        "status": "ENTITY_NUMOPS_BIAS_DONE",
        "finished_utc": now(),
        "missing": missing,
        "subtask_rows": subtask_rows_all,
        "aggregate_rows": agg_rows_all,
        "pair_deltas": deltas,
        "late_summary": late_summary,
        "interpretation": interp,
    }
    write_csv(out_dir / "entity_numops_subtask_scores.csv", subtask_rows_all)
    write_csv(out_dir / "entity_numops_aggregate_scores.csv", agg_rows_all)
    write_csv(out_dir / "entity_numops_view_minus_repeat_deltas.csv", deltas)
    (out_dir / "entity_numops_bias_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload, out_dir / "entity_numops_bias_summary.md")
    print(json.dumps({
        "status": payload["status"],
        "out_dir": rel(out_dir),
        "path_count": len(path_rows),
        "missing_count": len(missing),
        "delta_rows": len(deltas),
        "late_summary": late_summary,
        "no_model_inference_training_gpu_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
