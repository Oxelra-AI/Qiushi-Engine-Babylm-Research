#!/usr/bin/env python3
"""research: Entity zero/nonzero-operation readout for decisive research scores.

File-only post-processing of official Entity `predictions.json` files produced by
`decisive_gpu_scorer.py`, plus the matched clean reference predictions.
No model inference, training, GPU, upload, or leaderboard action.

Scientific role
---------------
Entity Tracking has repeatedly behaved as a special operation-propensity carrier:
view/repeat gains can be positive on nonzero-operation rows while harming 0-op
retention, and the official Entity aggregate weights nonzero operations much more
heavily than 0-op retention. The final register/in-corpus distribution contrasts
therefore need an Entity split before any broad data-efficient-learning
interpretation.

This script reads available research per-target payloads, extracts their Entity
prediction file paths, recomputes official-style split×numops accuracies from the
saved predictions, joins clean reference predictions for the same checkpoint, and
writes arm-minus-clean, childspeech-minus-adultprose, and in-corpus-minus-full1x
contrasts for all/zero/nonzero groups.
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


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
ENTITY_ROOT = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict" / "evaluation_data" / "full_eval" / "entity_tracking"
DEFAULT_DECISIVE_ROOTS = [
    WS / "data" / "register_decisive_gpu_eval",
    WS / "data" / "incorpus_decisive_gpu_eval",
]
DECISIVE_ROOTS = list(DEFAULT_DECISIVE_ROOTS)
CLEAN_EVAL_ROOT = WS / "data" / "deberta_maxgeom_clean_stable_eval" / "eval"
OUT_DIR = WS / "data" / "decisive_entity_numops_readout"

SPLITS = ["ambiref", "regular", "move_contents"]
NUMOPS = list(range(6))
GROUPS = ["all_18_subtasks", "zero_ops", "nonzero_ops"] + [f"numops_{i}" for i in NUMOPS]
CHECKPOINTS = ["chck_70M", "chck_80M", "chck_100M"]
CLEAN_TARGET_PREFIX = "deberta_maxgeom_clean_seed43022"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | None) -> str | None:
    if p is None:
        return None
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_prediction_path(ck: str) -> pathlib.Path | None:
    target = f"{CLEAN_TARGET_PREFIX}_{ck}"
    direct = CLEAN_EVAL_ROOT / "official_outputs" / target / "Entity" / ck
    hits = sorted(direct.glob("**/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json")) if direct.exists() else []
    if hits:
        return hits[0]
    hits = sorted(CLEAN_EVAL_ROOT.glob(f"**/{target}/Entity/**/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json"))
    return hits[0] if hits else None


def decisive_payload_paths() -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for root in DECISIVE_ROOTS:
        per = root / "per_target"
        if per.exists():
            paths.extend(sorted(per.glob("*.json")))
    return sorted(paths)


def prediction_records(include_clean: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in decisive_payload_paths():
        try:
            payload = read_json(p)
        except Exception as exc:
            records.append({"source_payload": rel(p), "exists": False, "error": repr(exc)})
            continue
        arm = payload.get("arm")
        ck = payload.get("checkpoint")
        ent = (payload.get("tasks") or {}).get("Entity") or {}
        pred = ent.get("predictions")
        pred_path = ROOT / pred if pred and not pathlib.Path(pred).is_absolute() else pathlib.Path(pred) if pred else None
        records.append({
            "arm": arm,
            "checkpoint": ck,
            "source_payload": rel(p),
            "predictions": rel(pred_path),
            "exists": bool(pred_path and pred_path.exists()),
            "score_from_payload": ent.get("score"),
            "returncode": ent.get("returncode"),
            "model_role": payload.get("role"),
            "is_clean_reference": False,
        })
    if include_clean:
        for ck in CHECKPOINTS:
            pred_path = clean_prediction_path(ck)
            records.append({
                "arm": "clean_maxgeom",
                "checkpoint": ck,
                "source_payload": None,
                "predictions": rel(pred_path),
                "exists": bool(pred_path and pred_path.exists()),
                "score_from_payload": None,
                "returncode": 0 if pred_path and pred_path.exists() else None,
                "model_role": "Matched MAX-geometry clean reference seed43022 from research.",
                "is_clean_reference": True,
            })
    # de-duplicate exact arm/checkpoint/predictions records, preserving order
    out: list[dict[str, Any]] = []
    seen = set()
    for r in records:
        key = (r.get("arm"), r.get("checkpoint"), r.get("predictions"))
        if key not in seen:
            out.append(r); seen.add(key)
    return out


def load_gold() -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for split in SPLITS:
        by: dict[int, list[dict[str, Any]]] = defaultdict(list)
        path = ENTITY_ROOT / f"{split}.jsonl"
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # Match the existing evaluation convention: exclude rows whose correct option is literally nothing.
                if str(obj.get("options", [""])[0]).strip() == "nothing.":
                    continue
                by[int(obj["numops"])].append(obj)
        for n in NUMOPS:
            out[(split, n)] = by.get(n, [])
    return out


def load_predictions(path: pathlib.Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    raw = read_json(path)
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    if not isinstance(raw, dict):
        return out
    for split in SPLITS:
        for n in NUMOPS:
            rec = raw.get(f"{split}_{n}_ops")
            out[(split, n)] = list((rec or {}).get("predictions") or []) if isinstance(rec, dict) else []
    return out


def score_predictions(preds: dict[tuple[str, int], list[dict[str, Any]]], gold: dict[tuple[str, int], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_correct = 0
    total_n = 0
    valid_pp: list[float] = []
    for split in SPLITS:
        for n in NUMOPS:
            g = gold[(split, n)]
            p = preds.get((split, n), [])
            m = min(len(g), len(p))
            corr = 0
            option_counts: dict[int, int] = defaultdict(int)
            for i in range(m):
                options = [str(x).strip() for x in g[i].get("options", [])]
                pred = str(p[i].get("pred", "")).strip()
                if pred == options[0]:
                    corr += 1
                try:
                    oi = options.index(pred)
                except ValueError:
                    oi = -1
                option_counts[int(oi)] += 1
            score_pp = 100.0 * corr / m if m else None
            if score_pp is not None:
                valid_pp.append(score_pp)
                total_correct += corr
                total_n += m
            rows.append({
                "split": split,
                "numops": n,
                "n_gold_filtered": len(g),
                "n_pred": len(p),
                "n_scored": m,
                "correct": corr,
                "score_pp": score_pp,
                "pred_correct_option_frac": None if m == 0 else option_counts.get(0, 0) / m,
                "pred_wrong_option_frac": None if m == 0 else option_counts.get(1, 0) / m,
                "pred_not_in_options_frac": None if m == 0 else option_counts.get(-1, 0) / m,
            })
    return rows, {
        "official_style_18_subtask_mean_pp": statistics.mean(valid_pp) if valid_pp else None,
        "micro_accuracy_pp": 100.0 * total_correct / total_n if total_n else None,
        "micro_n": total_n,
        "subtask_count": len(valid_pp),
        "zero_subtask_weight_in_official": 3 / 18,
        "nonzero_subtask_weight_in_official": 15 / 18,
    }


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def group_summary(rows: list[dict[str, Any]], meta: dict[str, Any], group: str, numops_group: str, split_group: str) -> dict[str, Any]:
    valid = [r for r in rows if r.get("score_pp") is not None]
    n = sum(int(r.get("n_scored", 0)) for r in valid)
    corr = sum(int(r.get("correct", 0)) for r in valid)
    return {
        **meta,
        "group": group,
        "numops_group": numops_group,
        "split_group": split_group,
        "subtask_count": len(valid),
        "macro_score_pp": mean([float(r["score_pp"]) for r in valid]) if valid else None,
        "micro_accuracy_pp": 100.0 * corr / n if n else None,
        "n_scored": n,
        "pred_correct_option_frac_micro": (sum(float(r.get("pred_correct_option_frac") or 0.0) * int(r.get("n_scored", 0)) for r in valid) / n) if n else None,
        "pred_not_in_options_frac_micro": (sum(float(r.get("pred_not_in_options_frac") or 0.0) * int(r.get("n_scored", 0)) for r in valid) / n) if n else None,
    }


def aggregate_rows(scored: list[dict[str, Any]], summary: dict[str, Any], meta: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.append({
        **meta,
        "group": "all_18_subtasks",
        "numops_group": "all",
        "split_group": "all",
        "macro_score_pp": summary.get("official_style_18_subtask_mean_pp"),
        "micro_accuracy_pp": summary.get("micro_accuracy_pp"),
        "n_scored": summary.get("micro_n"),
        "subtask_count": summary.get("subtask_count"),
    })
    for n in NUMOPS:
        out.append(group_summary([r for r in scored if r["numops"] == n], meta, f"numops_{n}", str(n), "all"))
    out.append(group_summary([r for r in scored if r["numops"] == 0], meta, "zero_ops", "zero", "all"))
    out.append(group_summary([r for r in scored if r["numops"] > 0], meta, "nonzero_ops", "nonzero", "all"))
    for split in SPLITS:
        out.append(group_summary([r for r in scored if r["split"] == split], meta, f"split_{split}", "all", split))
        out.append(group_summary([r for r in scored if r["split"] == split and r["numops"] == 0], meta, f"{split}_zero_ops", "zero", split))
        out.append(group_summary([r for r in scored if r["split"] == split and r["numops"] > 0], meta, f"{split}_nonzero_ops", "nonzero", split))
    return out


def build_contrasts(agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idx: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in agg:
        idx[(str(r["arm"]), str(r["checkpoint"]), str(r["group"]))] = r
    pairs: list[tuple[str, str, str, str]] = []
    arms = sorted({str(r["arm"]) for r in agg})
    cks = sorted({str(r["checkpoint"]) for r in agg})
    for arm in arms:
        if arm != "clean_maxgeom":
            pairs.append((f"{arm}_minus_clean", arm, "clean_maxgeom", "arm_minus_clean"))
    pairs += [
        ("register_childspeech_minus_adultprose", "regmax_childspeech", "regmax_adultprose", "childspeech_minus_adultprose"),
        ("incorpus_minus_subdose_full", "incorpus_adultprose", "subdose_full", "incorpus_minus_full1x"),
    ]
    out: list[dict[str, Any]] = []
    for cname, a, b, label in pairs:
        for ck in cks:
            for group in sorted({g for (arm, c, g) in idx if c == ck and arm in {a, b}}):
                ra = idx.get((a, ck, group)); rb = idx.get((b, ck, group))
                if not ra or not rb:
                    continue
                av = ra.get("macro_score_pp"); bv = rb.get("macro_score_pp")
                am = ra.get("micro_accuracy_pp"); bm = rb.get("micro_accuracy_pp")
                out.append({
                    "contrast_name": cname,
                    "contrast": label,
                    "checkpoint": ck,
                    "arm_A": a,
                    "arm_B": b,
                    "group": group,
                    "numops_group": ra.get("numops_group"),
                    "split_group": ra.get("split_group"),
                    "A_macro_pp": av,
                    "B_macro_pp": bv,
                    "delta_macro_pp": None if av is None or bv is None else float(av) - float(bv),
                    "A_micro_pp": am,
                    "B_micro_pp": bm,
                    "delta_micro_pp": None if am is None or bm is None else float(am) - float(bm),
                    "A_n_scored": ra.get("n_scored"),
                    "B_n_scored": rb.get("n_scored"),
                })
    return out


def balanced_rows(contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in contrasts:
        if r.get("group") in {"all_18_subtasks", "zero_ops", "nonzero_ops"}:
            by[(str(r["contrast_name"]), str(r["checkpoint"]))][str(r["group"])] = r
    out: list[dict[str, Any]] = []
    for (cname, ck), d in sorted(by.items()):
        allv = d.get("all_18_subtasks", {}).get("delta_macro_pp")
        z = d.get("zero_ops", {}).get("delta_macro_pp")
        nz = d.get("nonzero_ops", {}).get("delta_macro_pp")
        out.append({
            "contrast_name": cname,
            "checkpoint": ck,
            "official_all18_delta_pp": allv,
            "zero_ops_delta_pp": z,
            "nonzero_ops_delta_pp": nz,
            "balanced_zero_nonzero_delta_pp": None if z is None or nz is None else 0.5 * float(z) + 0.5 * float(nz),
            "nonzero_minus_zero_spread_pp": None if z is None or nz is None else float(nz) - float(z),
            "zero_official_weight": 1 / 6,
            "nonzero_official_weight": 5 / 6,
        })
    return out


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
    lines.append("# research decisive Entity zero/nonzero-operation readout\n\n")
    lines.append("File-only recomputation from official Entity prediction files. This separates Entity from the broad ex-Entity interpretation for the register and in-corpus contrasts.\n\n")
    lines.append("## Available Entity prediction records\n\n")
    lines.append("| arm | checkpoint | exists | payload score | predictions |\n")
    lines.append("|---|---:|---:|---:|---|\n")
    for r in payload["prediction_records"]:
        if r.get("arm") is None:
            continue
        lines.append(f"| {r.get('arm')} | {r.get('checkpoint')} | {r.get('exists')} | {r.get('score_from_payload')} | `{r.get('predictions')}` |\n")
    lines.append("\n## Balanced contrast rows\n\n")
    lines.append("| contrast | checkpoint | official all18 Δ pp | zero-op Δ pp | nonzero Δ pp | balanced zero/nonzero Δ pp | nonzero-zero spread pp |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in payload["balanced_rows"]:
        lines.append(f"| {r['contrast_name']} | {r['checkpoint']} | {fmt(r.get('official_all18_delta_pp'))} | {fmt(r.get('zero_ops_delta_pp'))} | {fmt(r.get('nonzero_ops_delta_pp'))} | {fmt(r.get('balanced_zero_nonzero_delta_pp'))} | {fmt(r.get('nonzero_minus_zero_spread_pp'))} |\n")
    lines.append("\n## Scientific reading rule\n\n")
    lines.append("- A positive Entity aggregate with nonzero-operation gains and zero-operation losses should be treated as operation-propensity allocation, not general record formation.\n")
    lines.append("- The distribution principle should be evaluated mainly on ex-Entity families; Entity can be reported as a separate, benchmark-specific movement unless zero/nonzero balance improves together.\n")
    lines.append("- This script does not recover continuous margins; existing prediction files store choices, not candidate log probabilities.\n")
    lines.append("\n## Files\n\n")
    for k, v in payload["files"].items():
        lines.append(f"- {k}: `{v}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    global DECISIVE_ROOTS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="*", default=[str(p) for p in DEFAULT_DECISIVE_ROOTS], help="Corrected scorer roots to read; failed research/293 roots are intentionally excluded by default")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    DECISIVE_ROOTS = [pathlib.Path(x) if pathlib.Path(x).is_absolute() else ROOT / x for x in args.roots]

    records = prediction_records(include_clean=not args.no_clean)
    missing = [r for r in records if not r.get("exists")]
    plan = {
        "status": "DECISIVE_ENTITY_NUMOPS_PLAN" if args.plan_only else "DECISIVE_ENTITY_NUMOPS_START",
        "created_utc": now(),
        "decisive_roots": [rel(r) for r in DECISIVE_ROOTS],
        "official_entity_root": rel(ENTITY_ROOT),
        "prediction_records": records,
        "missing_count": len(missing),
        "no_model_inference_training_gpu_upload_or_leaderboard": True,
    }
    write_json(out_dir / "entity_numops_plan.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    if missing and not args.allow_missing:
        raise SystemExit(f"missing {len(missing)} prediction files; rerun with --allow-missing for partial output")

    gold = load_gold()
    subtask: list[dict[str, Any]] = []
    agg: list[dict[str, Any]] = []
    for r in records:
        if not r.get("exists"):
            continue
        pp = pathlib.Path(str(r["predictions"]))
        p = pp if pp.is_absolute() else ROOT / pp
        scored, summary = score_predictions(load_predictions(p), gold)
        meta = {
            "arm": r.get("arm"),
            "checkpoint": r.get("checkpoint"),
            "predictions": r.get("predictions"),
            "source_payload": r.get("source_payload"),
            "score_from_payload": r.get("score_from_payload"),
            "is_clean_reference": r.get("is_clean_reference"),
        }
        for sr in scored:
            subtask.append({**meta, **sr})
        agg.extend(aggregate_rows(scored, summary, meta))

    contrasts = build_contrasts(agg)
    bal = balanced_rows(contrasts)
    late_summary: list[dict[str, Any]] = []
    for cname in sorted({r["contrast_name"] for r in bal}):
        rows = [r for r in bal if r["contrast_name"] == cname]
        for quantity in ["official_all18_delta_pp", "zero_ops_delta_pp", "nonzero_ops_delta_pp", "balanced_zero_nonzero_delta_pp", "nonzero_minus_zero_spread_pp"]:
            vals = [float(r[quantity]) for r in rows if r.get(quantity) is not None]
            if vals:
                late_summary.append({"contrast_name": cname, "quantity": quantity, "n": len(vals), "mean": mean(vals), "median": statistics.median(vals), "min": min(vals), "max": max(vals)})

    files = {
        "summary_json": rel(out_dir / "entity_numops_readout_summary.json"),
        "summary_md": rel(out_dir / "entity_numops_readout_summary.md"),
        "subtask_csv": rel(out_dir / "entity_numops_subtask_scores.csv"),
        "aggregate_csv": rel(out_dir / "entity_numops_aggregate_scores.csv"),
        "contrast_csv": rel(out_dir / "entity_numops_contrasts.csv"),
        "balanced_csv": rel(out_dir / "entity_numops_balanced_contrasts.csv"),
        "late_summary_csv": rel(out_dir / "entity_numops_late_summary.csv"),
    }
    payload = {
        **plan,
        "status": "DECISIVE_ENTITY_NUMOPS_DONE",
        "finished_utc": now(),
        "missing": missing,
        "subtask_rows": subtask,
        "aggregate_rows": agg,
        "contrast_rows": contrasts,
        "balanced_rows": bal,
        "late_summary_rows": late_summary,
        "files": files,
        "scientific_reading": {
            "entity_not_broad_principle_by_default": "Positive aggregate Entity movement must be separated into zero-operation retention and nonzero-operation movement before using it as evidence for any general data-efficient learning principle.",
            "official_weighting": "The official all-18 Entity mean gives 1/6 weight to zero-operation rows and 5/6 to nonzero-operation rows; a neutral broad state reading also needs the 50/50 zero/nonzero quantity.",
        },
    }
    write_csv(out_dir / "entity_numops_subtask_scores.csv", subtask)
    write_csv(out_dir / "entity_numops_aggregate_scores.csv", agg)
    write_csv(out_dir / "entity_numops_contrasts.csv", contrasts)
    write_csv(out_dir / "entity_numops_balanced_contrasts.csv", bal)
    write_csv(out_dir / "entity_numops_late_summary.csv", late_summary)
    write_json(out_dir / "entity_numops_readout_summary.json", payload)
    write_md(payload, out_dir / "entity_numops_readout_summary.md")
    print(json.dumps({
        "status": payload["status"],
        "records_scored": len({(r.get('arm'), r.get('checkpoint')) for r in agg}),
        "missing_count": len(missing),
        "balanced_rows": bal,
        "summary_md": files["summary_md"],
        "no_model_inference_training_gpu_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
