#!/usr/bin/env python3
"""research: balanced Entity-stratum readout for MAX dose mechanism.

This file-only script reads existing official Entity prediction JSONs and recomputes
Entity Tracking accuracy by split and numops. It makes explicit a critical
arithmetic point: the official Entity score weights the six numops levels equally
(three splits per level), so the aggregate view-minus-repeat delta has 1/6 weight
on zero-operation retention rows and 5/6 weight on nonzero-operation changed rows.
A neutral changed-vs-unchanged reading should also inspect a 50/50 zero/nonzero
average and per-numops pattern.

It covers the available decision arms:
  - DeBERTa first-basin MAX view/repeat late 80/90/100M
  - DeBERTa second-basin MAX view/repeat late 80/90/100M
  - RoBERTa MAX view/repeat late 80/90/100M, if predictions exist
  - DeBERTa MAX breadth, if predictions exist

No model inference, no training, no GPU, no official scoring, no upload.
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
import statistics
import time
from collections import defaultdict
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
ENTITY_ROOT = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict" / "evaluation_data" / "full_eval" / "entity_tracking"
SPLITS = ["ambiref", "regular", "move_contents"]
NUMOPS = list(range(6))
LATE_CKS = ["chck_80M", "chck_90M", "chck_100M"]

ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "deberta_basin1_view": {
        "contrast_group": "deberta_basin1", "model_family": "DeBERTa", "seed": 43022, "data_role": "view",
        "eval_root": WS / "data/dose_ladder_stable_eval/eval",
        "target_prefix": "dose_max_view",
    },
    "deberta_basin1_repeat": {
        "contrast_group": "deberta_basin1", "model_family": "DeBERTa", "seed": 43022, "data_role": "repeat",
        "eval_root": WS / "data/dose_ladder_stable_eval/eval",
        "target_prefix": "dose_max_repeat",
    },
    "deberta_basin2_view": {
        "contrast_group": "deberta_basin2", "model_family": "DeBERTa", "seed": 43122, "data_role": "view",
        "eval_root": WS / "data/second_basin_entity_ewok_eval/view_eval",
        "target_prefix": "second_basin_max_view_seed43122",
    },
    "deberta_basin2_repeat": {
        "contrast_group": "deberta_basin2", "model_family": "DeBERTa", "seed": 43122, "data_role": "repeat",
        "eval_root": WS / "data/second_basin_entity_ewok_eval/repeat_eval",
        "target_prefix": "second_basin_max_repeat_seed43122",
    },
    "roberta_max_view": {
        "contrast_group": "roberta_max", "model_family": "RoBERTa", "seed": 43022, "data_role": "view",
        "eval_root": WS / "data/roberta_viewclean_total_stable_eval/eval",
        "target_prefix": "roberta_viewclean_total_view",
    },
    "roberta_max_repeat": {
        "contrast_group": "roberta_max", "model_family": "RoBERTa", "seed": 43022, "data_role": "repeat",
        "eval_root": WS / "data/roberta_maxdose_repeat_eval/eval",
        "target_prefix": "roberta_max_repeat_seed43022",
    },
    "deberta_breadth": {
        "contrast_group": "deberta_breadth_specificity", "model_family": "DeBERTa", "seed": 43022, "data_role": "breadth",
        "eval_root": WS / "data/breadth_entity_ewok_eval/eval",
        "target_prefix": "max_breadth_seed43022",
    },
}
CONTRASTS = [
    ("deberta_basin1", "deberta_basin1_view", "deberta_basin1_repeat", "view_minus_repeat"),
    ("deberta_basin2", "deberta_basin2_view", "deberta_basin2_repeat", "view_minus_repeat"),
    ("roberta_max", "roberta_max_view", "roberta_max_repeat", "view_minus_repeat"),
    ("deberta_view_minus_breadth", "deberta_basin1_view", "deberta_breadth", "view_minus_breadth"),
    ("deberta_breadth_minus_repeat", "deberta_breadth", "deberta_basin1_repeat", "breadth_minus_repeat"),
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pred_path(cfg: dict[str, Any], ck: str) -> pathlib.Path | None:
    target = f"{cfg['target_prefix']}_{ck}"
    root = pathlib.Path(cfg["eval_root"])
    direct = root / "official_outputs" / target / "Entity"
    hits = sorted(direct.glob("**/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json")) if direct.exists() else []
    if hits:
        return hits[0]
    hits = sorted(root.glob(f"**/{target}/Entity/**/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json"))
    if hits:
        return hits[0]
    hits = sorted(root.glob(f"**/{target}*Entity*/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json"))
    return hits[0] if hits else None


def load_gold() -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for split in SPLITS:
        by_num: dict[int, list[dict[str, Any]]] = defaultdict(list)
        with (ENTITY_ROOT / f"{split}.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # Match prior official-style scripts: skip rows whose gold option is literally nothing.
                if str(obj.get("options", [""])[0]).strip() == "nothing.":
                    continue
                by_num[int(obj["numops"])].append(obj)
        for n in NUMOPS:
            out[(split, n)] = by_num.get(n, [])
    return out


def load_preds(path: pathlib.Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    raw = read_json(path)
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for split in SPLITS:
        for n in NUMOPS:
            key = f"{split}_{n}_ops"
            rec = raw.get(key) if isinstance(raw, dict) else None
            out[(split, n)] = list((rec or {}).get("predictions") or []) if isinstance(rec, dict) else []
    return out


def score_one(preds: dict[tuple[str, int], list[dict[str, Any]]], gold: dict[tuple[str, int], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_correct = 0
    total_n = 0
    for split in SPLITS:
        for n in NUMOPS:
            g = gold[(split, n)]
            p = preds.get((split, n), [])
            m = min(len(g), len(p))
            corr = 0
            opt0_count = 0
            opt_index_counts: dict[int, int] = defaultdict(int)
            for i in range(m):
                options = [str(x).strip() for x in g[i].get("options", [])]
                pred = str(p[i].get("pred", "")).strip()
                if pred == options[0]:
                    corr += 1
                try:
                    oi = options.index(pred)
                except ValueError:
                    oi = -1
                opt_index_counts[int(oi)] += 1
                if oi == 0:
                    opt0_count += 1
            score = 100.0 * corr / m if m else None
            rows.append({
                "split": split, "numops": n, "n_gold_filtered": len(g), "n_pred": len(p), "n_scored": m,
                "correct": corr, "score_pp": score,
                "pred_option0_frac": opt0_count / m if m else None,
                "pred_not_in_options_frac": opt_index_counts.get(-1, 0) / m if m else None,
            })
            total_correct += corr
            total_n += m
    valid_scores = [float(r["score_pp"]) for r in rows if r["score_pp"] is not None]
    summary = {
        "official_style_18_subtask_mean_pp": sum(valid_scores) / len(valid_scores) if valid_scores else None,
        "micro_accuracy_pp": 100.0 * total_correct / total_n if total_n else None,
        "micro_n": total_n,
        "subtask_count": len(valid_scores),
        "zero_subtask_weight_in_official": 3 / 18,
        "nonzero_subtask_weight_in_official": 15 / 18,
    }
    return rows, summary


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def aggregate_scored(rows: list[dict[str, Any]], meta: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    out = [{**meta, "group": "all_18_subtasks", "numops_group": "all", "split_group": "all", **summary}]
    for n in NUMOPS:
        group = [r for r in rows if r["numops"] == n and r["score_pp"] is not None]
        out.append(group_summary(group, {**meta, "group": f"numops_{n}", "numops_group": str(n), "split_group": "all"}))
    out.append(group_summary([r for r in rows if r["numops"] == 0], {**meta, "group": "zero_ops", "numops_group": "zero", "split_group": "all"}))
    out.append(group_summary([r for r in rows if r["numops"] > 0], {**meta, "group": "nonzero_ops", "numops_group": "nonzero", "split_group": "all"}))
    for split in SPLITS:
        out.append(group_summary([r for r in rows if r["split"] == split], {**meta, "group": f"split_{split}", "numops_group": "all", "split_group": split}))
        out.append(group_summary([r for r in rows if r["split"] == split and r["numops"] == 0], {**meta, "group": f"{split}_zero_ops", "numops_group": "zero", "split_group": split}))
        out.append(group_summary([r for r in rows if r["split"] == split and r["numops"] > 0], {**meta, "group": f"{split}_nonzero_ops", "numops_group": "nonzero", "split_group": split}))
    return out


def group_summary(group: list[dict[str, Any]], fields: dict[str, Any]) -> dict[str, Any]:
    valid = [r for r in group if r.get("score_pp") is not None]
    if not valid:
        return {**fields, "subtask_count": 0, "macro_score_pp": None, "micro_accuracy_pp": None, "n_scored": 0, "pred_option0_frac_micro": None}
    n = sum(int(r["n_scored"]) for r in valid)
    corr = sum(int(r["correct"]) for r in valid)
    return {
        **fields,
        "subtask_count": len(valid),
        "macro_score_pp": mean([float(r["score_pp"]) for r in valid]),
        "micro_accuracy_pp": 100.0 * corr / n if n else None,
        "n_scored": n,
        "pred_option0_frac_micro": sum(float(r.get("pred_option0_frac") or 0.0) * int(r["n_scored"]) for r in valid) / n if n else None,
        "pred_not_in_options_frac_micro": sum(float(r.get("pred_not_in_options_frac") or 0.0) * int(r["n_scored"]) for r in valid) / n if n else None,
    }


def get_score(row: dict[str, Any], key: str) -> float | None:
    val = row.get(key)
    return None if val is None else float(val)


def contrast_rows(agg: list[dict[str, Any]], arm_keys: set[str]) -> list[dict[str, Any]]:
    idx: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in agg:
        idx[(r["arm_key"], r["checkpoint"], r["group"])] = r
    out: list[dict[str, Any]] = []
    for contrast_name, akey, bkey, label in CONTRASTS:
        if akey not in arm_keys or bkey not in arm_keys:
            continue
        for ck in LATE_CKS:
            groups = sorted({g for (ak, c, g) in idx if ak in {akey, bkey} and c == ck})
            for group in groups:
                a = idx.get((akey, ck, group)); b = idx.get((bkey, ck, group))
                if not a or not b:
                    continue
                metric_key = "official_style_18_subtask_mean_pp" if group == "all_18_subtasks" else "macro_score_pp"
                av = get_score(a, metric_key); bv = get_score(b, metric_key)
                am = get_score(a, "micro_accuracy_pp"); bm = get_score(b, "micro_accuracy_pp")
                ao = get_score(a, "pred_option0_frac_micro"); bo = get_score(b, "pred_option0_frac_micro")
                out.append({
                    "contrast_name": contrast_name,
                    "contrast": label,
                    "checkpoint": ck,
                    "words": int(ck.replace("chck_", "").replace("M", "")) * 1_000_000,
                    "model_family": a.get("model_family"),
                    "seed_A": a.get("seed"), "seed_B": b.get("seed"),
                    "arm_A": akey, "arm_B": bkey,
                    "group": group,
                    "numops_group": a.get("numops_group"),
                    "split_group": a.get("split_group"),
                    "A_macro_pp": av, "B_macro_pp": bv,
                    "delta_macro_pp": None if av is None or bv is None else av - bv,
                    "A_micro_pp": am, "B_micro_pp": bm,
                    "delta_micro_pp": None if am is None or bm is None else am - bm,
                    "A_pred_option0_frac": ao, "B_pred_option0_frac": bo,
                    "delta_pred_option0_frac": None if ao is None or bo is None else ao - bo,
                    "A_n_scored": a.get("n_scored", a.get("micro_n")),
                    "B_n_scored": b.get("n_scored", b.get("micro_n")),
                })
    return out


def balanced_tables(contrasts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_ck: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in contrasts:
        if r["group"] in {"all_18_subtasks", "zero_ops", "nonzero_ops"}:
            by_ck[(r["contrast_name"], r["checkpoint"])][r["group"]] = r
    rows: list[dict[str, Any]] = []
    for (contrast_name, ck), d in sorted(by_ck.items()):
        all_d = d.get("all_18_subtasks", {}).get("delta_macro_pp")
        z = d.get("zero_ops", {}).get("delta_macro_pp")
        nz = d.get("nonzero_ops", {}).get("delta_macro_pp")
        official_from_groups = None if z is None or nz is None else (1/6) * float(z) + (5/6) * float(nz)
        balanced = None if z is None or nz is None else 0.5 * float(z) + 0.5 * float(nz)
        rows.append({
            "contrast_name": contrast_name,
            "checkpoint": ck,
            "official_all18_delta_pp": all_d,
            "zero_ops_delta_pp": z,
            "nonzero_ops_delta_pp": nz,
            "official_reconstructed_from_zero_nonzero_pp": official_from_groups,
            "balanced_zero_nonzero_delta_pp": balanced,
            "nonzero_minus_zero_spread_pp": None if z is None or nz is None else float(nz) - float(z),
            "zero_official_weight": 1/6,
            "nonzero_official_weight": 5/6,
        })
    late_summary: list[dict[str, Any]] = []
    for contrast_name in sorted({r["contrast_name"] for r in rows}):
        rs = [r for r in rows if r["contrast_name"] == contrast_name]
        for key in ["official_all18_delta_pp", "zero_ops_delta_pp", "nonzero_ops_delta_pp", "balanced_zero_nonzero_delta_pp", "nonzero_minus_zero_spread_pp"]:
            vals = [float(r[key]) for r in rs if r.get(key) is not None]
            if vals:
                late_summary.append({"contrast_name": contrast_name, "window": "late_80_90_100M", "quantity": key, "n": len(vals), "mean": mean(vals), "median": statistics.median(vals), "min": min(vals), "max": max(vals)})
    return rows, late_summary


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def fmt(x: Any) -> str:
    return "NA" if x is None else f"{float(x):+.3f}"


def write_md(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines: list[str] = []
    lines.append("# research balanced Entity-stratum readout\n\n")
    lines.append("This readout recomputes official Entity prediction files by split and numops, then contrasts MAX arms under both the official 1/6 zero-op + 5/6 nonzero-op weighting and a neutral 50/50 zero-vs-nonzero weighting.\n\n")
    lines.append("## Late checkpoint balanced contrasts\n\n")
    lines.append("| contrast | checkpoint | official all18 Δ pp | zero-op Δ pp | nonzero Δ pp | balanced zero/nonzero Δ pp | nonzero-zero spread pp |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in payload["balanced_rows"]:
        lines.append(f"| {r['contrast_name']} | {r['checkpoint']} | {fmt(r['official_all18_delta_pp'])} | {fmt(r['zero_ops_delta_pp'])} | {fmt(r['nonzero_ops_delta_pp'])} | {fmt(r['balanced_zero_nonzero_delta_pp'])} | {fmt(r['nonzero_minus_zero_spread_pp'])} |\n")
    lines.append("\n## Late-window means\n\n")
    lines.append("| contrast | quantity | n | mean | min | max |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for r in payload["late_summary_rows"]:
        lines.append(f"| {r['contrast_name']} | {r['quantity']} | {r['n']} | {fmt(r['mean'])} | {fmt(r['min'])} | {fmt(r['max'])} |\n")
    lines.append("\n## Interpretation\n\n")
    for k, v in payload["interpretation"].items():
        lines.append(f"- **{k}**: {v}\n")
    lines.append("\n## Files\n\n")
    for k, v in payload["files"].items():
        lines.append(f"- {k}: `{v}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(WS / "data/entity_balanced_and_transfer_readout"))
    ap.add_argument("--arms", nargs="*", default=list(ARM_CONFIGS.keys()), choices=sorted(ARM_CONFIGS))
    ap.add_argument("--checkpoints", nargs="*", default=LATE_CKS)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_gold()
    path_rows: list[dict[str, Any]] = []
    subtask_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for arm in args.arms:
        cfg = ARM_CONFIGS[arm]
        for ck in args.checkpoints:
            p = pred_path(cfg, ck)
            path_rows.append({"arm_key": arm, "checkpoint": ck, "target": f"{cfg['target_prefix']}_{ck}", "predictions": rel(p) if p else None, "exists": bool(p and p.exists())})
    missing = [r for r in path_rows if not r["exists"]]
    plan = {"status": "ENTITY_BALANCED_READOUT_START", "created_utc": now(), "arms": args.arms, "checkpoints": args.checkpoints, "prediction_paths": path_rows, "missing_count": len(missing), "official_entity_root": rel(ENTITY_ROOT), "no_model_inference_training_gpu_upload_or_leaderboard": True}
    write_json(out_dir / "entity_balanced_plan.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if missing and not args.allow_missing:
        raise SystemExit(f"missing {len(missing)} prediction files; use --allow-missing for partial readout")

    for row in path_rows:
        if not row["exists"]:
            continue
        arm = row["arm_key"]
        cfg = ARM_CONFIGS[arm]
        p = ROOT / row["predictions"] if not pathlib.Path(row["predictions"]).is_absolute() else pathlib.Path(row["predictions"])
        scored, summary = score_one(load_preds(p), gold)
        meta = {"arm_key": arm, "contrast_group": cfg["contrast_group"], "model_family": cfg["model_family"], "seed": cfg["seed"], "data_role": cfg["data_role"], "checkpoint": row["checkpoint"], "predictions": row["predictions"]}
        for sr in scored:
            subtask_rows.append({**meta, **sr})
        aggregate_rows.extend(aggregate_scored(scored, meta, summary))
    arm_keys = {r["arm_key"] for r in aggregate_rows}
    contrasts = contrast_rows(aggregate_rows, arm_keys)
    balanced, late_summary = balanced_tables(contrasts)
    interpretation = {
        "deberta_macro_entity": "In both DeBERTa basins, a positive official all-18-subtask Entity V-R combined with negative 50/50 zero/nonzero V-R means the aggregate carrier is created by the evaluation mixture's 5:1 nonzero-operation weighting, not by a uniformly improved state record skill.",
        "roberta_transfer": "RoBERTa rows are included if late view and repeat predictions exist. Their balanced rows should be read before treating any aggregate Entity difference as architectural transfer.",
        "breadth_specificity": "Breadth rows are included if delivered; view-minus-breadth and breadth-minus-repeat are decomposed the same way so same-population sentence breadth is not judged by aggregate Entity alone.",
        "margin_next": "Official prediction JSONs contain only the selected string, not option likelihoods. Threshold-free margin separation therefore requires a direct model scoring pass or the pending research sampled margin task, not these prediction files alone.",
    }
    files = {
        "subtask_scores_csv": rel(out_dir / "entity_subtask_scores.csv"),
        "aggregate_scores_csv": rel(out_dir / "entity_aggregate_scores.csv"),
        "contrast_rows_csv": rel(out_dir / "entity_contrast_rows.csv"),
        "balanced_rows_csv": rel(out_dir / "entity_balanced_rows.csv"),
        "late_summary_csv": rel(out_dir / "entity_late_summary_rows.csv"),
        "summary_json": rel(out_dir / "entity_balanced_readout_summary.json"),
        "summary_md": rel(out_dir / "entity_balanced_readout_summary.md"),
    }
    payload = {**plan, "status": "ENTITY_BALANCED_READOUT_DONE", "finished_utc": now(), "missing": missing, "subtask_rows": subtask_rows, "aggregate_rows": aggregate_rows, "contrast_rows": contrasts, "balanced_rows": balanced, "late_summary_rows": late_summary, "interpretation": interpretation, "files": files}
    write_csv(out_dir / "entity_subtask_scores.csv", subtask_rows)
    write_csv(out_dir / "entity_aggregate_scores.csv", aggregate_rows)
    write_csv(out_dir / "entity_contrast_rows.csv", contrasts)
    write_csv(out_dir / "entity_balanced_rows.csv", balanced)
    write_csv(out_dir / "entity_late_summary_rows.csv", late_summary)
    write_json(out_dir / "entity_balanced_readout_summary.json", payload)
    write_md(payload, out_dir / "entity_balanced_readout_summary.md")
    print(json.dumps({"status": payload["status"], "missing_count": len(missing), "balanced_rows": balanced, "late_summary_rows": late_summary, "summary_md": files["summary_md"], "no_model_inference_training_gpu_upload_or_leaderboard": True}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
