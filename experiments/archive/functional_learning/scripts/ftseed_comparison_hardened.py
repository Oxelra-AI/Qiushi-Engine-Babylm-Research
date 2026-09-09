#!/usr/bin/env python3
"""research: hardened matched fine-tuning-seed SuperGLUE comparison.

This script supersedes the lighter research comparison wrapper. It parses seed42
SuperGLUE scores from the completed research all-model profile and parses seed44
runs from faithful_superglue_seeded_summary.json files. It validates
that seed44 SuperGLUE evidence has all seven tasks, returncode 0 for each task,
non-null primary metrics, and existing prediction/result/log files. Optional
`overall_if_all_columns_available` values in the seeded summaries are recorded
only as auxiliary metadata because the research coherent86 task used a mismatched
cheap-summary path; they are not used for scientific comparison.
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
import time
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

ROOT = _public_path('experiments/archive/functional_learning/scripts/ftseed_comparison_hardened.py')
while ROOT.name != "Sessions" and ROOT != _public_path('experiments/archive/functional_learning'):
    ROOT = _public_path('experiments/archive/functional_learning')
ROOT = _public_path('experiments/archive/functional_learning')

OUT_DEFAULT = _public_path('experiments/archive/functional_learning/data/ftseed_comparison_hardened')
SEED42_PROFILE = _public_path('experiments/archive/functional_learning/data/superglue_all_completed_profile/superglue_all_completed_profile.json')

PRIMARY = {
    "boolq": "accuracy",
    "multirc": "accuracy",
    "rte": "accuracy",
    "wsc": "accuracy",
    "mrpc": "f1",
    "qqp": "f1",
    "mnli": "accuracy",
}
TASKS = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]
MODELS = ["coherent86", "ms_acquisition", "clean64"]
PROFILE_NAME = {
    "coherent86": "coherent86",
    "ms_acquisition": "ms_acquisition_seed62064_MS",
    "clean64": "clean_pres_seed62064_MSplusKL",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str | None) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def rj(path: pathlib.Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def as_path(path: str | pathlib.Path | None) -> Optional[pathlib.Path]:
    if path is None:
        return None
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def exists_rel(path: str | pathlib.Path | None) -> bool:
    p = as_path(path)
    return bool(p and p.exists())


def parse_seed42_profile(path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = rj(path)
    metrics = data.get("model_metrics", {})
    out: Dict[str, Dict[str, Any]] = {}
    for model in MODELS:
        profile_key = PROFILE_NAME[model]
        block = metrics.get(profile_key)
        if not isinstance(block, dict):
            continue
        task_block = block.get("tasks", {})
        tasks: Dict[str, float] = {}
        for task in TASKS:
            td = task_block.get(task, {})
            metric = PRIMARY[task]
            value = td.get(metric) if isinstance(td, dict) else None
            if value is not None:
                tasks[task] = float(value)
        if len(tasks) == len(TASKS):
            sg = float(block.get("superglue_mean", mean(tasks.values())))
            computed = mean(tasks.values())
            out[model] = {
                "seed": 42,
                "superglue_primary_mean": sg,
                "computed_mean": computed,
                "mean_matches_profile": abs(sg - computed) <= 1e-8,
                "tasks": tasks,
                "valid_for_superglue_comparison": abs(sg - computed) <= 1e-8,
                "path": rel(path),
                "source": "all_completed_profile",
            }
    return out


def parse_seeded_summary(path: pathlib.Path, expected_seed: int = 44) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    data = rj(path)
    records = data.get("task_records", [])
    tasks: Dict[str, float] = {}
    task_rows: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        task = rec.get("task")
        if task not in PRIMARY:
            continue
        metric = rec.get("primary_metric", PRIMARY[task])
        if metric != PRIMARY[task]:
            errors.append(f"metric_mismatch:{task}:{metric}!={PRIMARY[task]}")
        score = rec.get("primary_score")
        if score is None:
            errors.append(f"missing_primary_score:{task}")
        else:
            tasks[task] = float(score)
        if rec.get("returncode") not in (0, None):
            errors.append(f"nonzero_returncode:{task}:{rec.get('returncode')}")
        # If a task was skipped because existing files were reused, the wrapper may omit returncode.
        # That is acceptable only when the primary score and files are present.
        row = {
            "task": task,
            "primary_metric": metric,
            "primary_score": float(score) if score is not None else None,
            "returncode": rec.get("returncode"),
            "skipped_existing": rec.get("skipped_existing", False),
            "results_txt": rec.get("results_txt"),
            "results_txt_exists": exists_rel(rec.get("results_txt")),
            "predictions": rec.get("predictions"),
            "predictions_exists": exists_rel(rec.get("predictions")),
            "log": rec.get("log"),
            "log_exists": exists_rel(rec.get("log")) if rec.get("log") else rec.get("skipped_existing", False),
        }
        if not row["results_txt_exists"]:
            errors.append(f"missing_results_txt:{task}")
        if not row["predictions_exists"]:
            errors.append(f"missing_predictions:{task}")
        if rec.get("log") and not row["log_exists"]:
            errors.append(f"missing_log:{task}")
        task_rows[task] = row
    for task in TASKS:
        if task not in task_rows:
            errors.append(f"missing_task:{task}")
    sg = data.get("superglue_primary_metric_mean")
    computed = mean(tasks[t] for t in TASKS) if all(t in tasks for t in TASKS) else None
    if sg is None and computed is not None:
        sg = computed
    if sg is not None and computed is not None and abs(float(sg) - computed) > 1e-8:
        errors.append(f"mean_mismatch:payload={sg}:computed={computed}")
    if data.get("finetune_seed") != expected_seed:
        errors.append(f"seed_mismatch:{data.get('finetune_seed')}!={expected_seed}")
    valid = len(errors) == 0 and computed is not None
    return {
        "seed": data.get("finetune_seed"),
        "label": data.get("label"),
        "checkpoint": data.get("checkpoint"),
        "model_path": data.get("model_path"),
        "revision_name": data.get("revision_name"),
        "superglue_primary_mean": float(sg) if sg is not None else None,
        "computed_mean": computed,
        "tasks": tasks,
        "task_rows": [task_rows[t] for t in TASKS if t in task_rows],
        "valid_for_superglue_comparison": valid,
        "errors": errors,
        "path": rel(path),
        "cheap_summary": data.get("cheap_summary"),
        "aoa_summary": data.get("aoa_summary"),
        "overall_if_all_columns_available_ignored": data.get("overall_if_all_columns_available"),
        "optional_overall_note": "Ignored for this comparison; only SuperGLUE primary metrics are compared, because cheap_summary/AoA summary may be supplied only for wrapper convenience.",
    }


def newest_summary_under(root: pathlib.Path) -> Optional[pathlib.Path]:
    if not root.exists():
        return None
    hits = list(root.rglob("faithful_superglue_seeded_summary.json"))
    if not hits:
        return None
    hits.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    return hits[0]


def pick_clean64_summary(explicit: str, a02_root: str) -> Tuple[Optional[pathlib.Path], str]:
    if explicit:
        p = as_path(explicit)
        if p and p.exists():
            return p, "explicit"
    root = as_path(a02_root)
    found = newest_summary_under(root) if root else None
    if found:
        return found, "a02_search_newest"
    return None, "missing"


def comparison_block(data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    c86 = data.get("coherent86")
    ms = data.get("ms_acquisition")
    cl = data.get("clean64")
    if c86 and ms:
        out["ms_minus_coherent86_acquisition_only"] = ms["superglue_primary_mean"] - c86["superglue_primary_mean"]
    if cl and c86:
        out["clean64_minus_coherent86"] = cl["superglue_primary_mean"] - c86["superglue_primary_mean"]
    if cl and ms:
        out["clean64_minus_ms_preservation_specific"] = cl["superglue_primary_mean"] - ms["superglue_primary_mean"]
    return out


def format_val(x: Any, nd: int = 6, signed: bool = False) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return str(x)


def write_md(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research hardened matched ftseed SuperGLUE comparison\n\n")
    lines.append(f"Created: `{result['created_utc']}`\n\n")
    lines.append("This artifact compares only SuperGLUE primary-metric means. Optional Overall values from seeded wrappers are explicitly ignored.\n\n")
    lines.append(f"Seed42 complete: `{result['completeness']['seed42_complete']}/3`; seed44 valid complete: `{result['completeness']['seed44_valid_complete']}/3`.\n\n")
    lines.append("## SuperGLUE primary means\n\n")
    lines.append("| model | seed42 | seed44 | seed44 valid | Δ(44-42) |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for model in MODELS:
        r42 = result["seed42_results"].get(model)
        r44 = result["seed44_results"].get(model)
        v42 = r42.get("superglue_primary_mean") if r42 else None
        v44 = r44.get("superglue_primary_mean") if r44 else None
        delta = (v44 - v42) if v42 is not None and v44 is not None else None
        valid = r44.get("valid_for_superglue_comparison") if r44 else False
        lines.append(f"| {model} | {format_val(v42)} | {format_val(v44)} | {valid} | {format_val(delta, signed=True)} |\n")
    lines.append("\n## Key SuperGLUE comparisons\n\n")
    lines.append("| comparison | seed42 | seed44 | sign relation |\n")
    lines.append("|---|---:|---:|---|\n")
    keys = ["clean64_minus_coherent86", "clean64_minus_ms_preservation_specific", "ms_minus_coherent86_acquisition_only"]
    for key in keys:
        v42 = result["cross_seed_comparison"].get("seed42", {}).get(key)
        v44 = result["cross_seed_comparison"].get("seed44", {}).get(key)
        if v42 is None or v44 is None:
            sign = "pending"
        elif v42 == 0 or v44 == 0:
            sign = "zero involved"
        elif v42 * v44 > 0:
            sign = "same sign"
        else:
            sign = "opposite sign"
        lines.append(f"| {key} | {format_val(v42, signed=True)} | {format_val(v44, signed=True)} | {sign} |\n")
    lines.append("\n## Task-level values\n\n")
    lines.append("| task | metric | coherent86_s42 | coherent86_s44 | ms_s42 | ms_s44 | clean64_s42 | clean64_s44 |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
    for row in result["task_level"]:
        lines.append(
            f"| {row['task']} | {row['primary_metric']} | "
            f"{format_val(row.get('coherent86_seed42'))} | {format_val(row.get('coherent86_seed44'))} | "
            f"{format_val(row.get('ms_acquisition_seed42'))} | {format_val(row.get('ms_acquisition_seed44'))} | "
            f"{format_val(row.get('clean64_seed42'))} | {format_val(row.get('clean64_seed44'))} |\n"
        )
    lines.append("\n## Seed44 validity\n\n")
    for model, rec in result["seed44_results"].items():
        if rec is None:
            lines.append(f"- `{model}`: missing\n")
        else:
            lines.append(f"- `{model}`: valid={rec.get('valid_for_superglue_comparison')} path=`{rec.get('path')}` errors={rec.get('errors')}\n")
    lines.append("\n## Scientific reading\n\n")
    if result["completeness"]["seed44_valid_complete"] < 3:
        lines.append("The matched downstream-seed comparison is not yet complete. Seed42 values are reproduced exactly from the completed research profile, and any available seed44 summaries are validated for SuperGLUE only. Rerun this script after coherent86, exact `(M,S)`, and clean64 seed44 summaries are all present and valid.\n")
    else:
        lines.append("All three seed44 SuperGLUE summaries are present and valid. Interpret the seed44 clean64-minus-exact-`(M,S)` difference as a matched downstream-finetuning-seed check of the SuperGLUE portion of the preservation-specific increment, not as a new Overall coordinate. The zero/Reading/AoA components remain those of the fixed official evaluation; only the downstream fine-tuning randomness in SuperGLUE has been perturbed.\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_csv(path: pathlib.Path, result: Dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "primary_metric", "coherent86_seed42", "coherent86_seed44", "ms_acquisition_seed42", "ms_acquisition_seed44", "clean64_seed42", "clean64_seed44"])
        w.writeheader()
        for row in result["task_level"]:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DEFAULT)
    ap.add_argument("--seed42-profile", type=pathlib.Path, default=SEED42_PROFILE)
    ap.add_argument("--coherent86-seed44", type=pathlib.Path, default=_public_path('experiments/archive/functional_learning/data/matched_ftseed44_superglue/coherent86_alpha075_ftseed44/faithful_superglue_seeded_summary.json'))
    ap.add_argument("--ms-seed44", type=pathlib.Path, default=_public_path('experiments/archive/functional_learning/data/ms_ftseed44_superglue/densemask_sparselabel_seed62064_ftseed44/faithful_superglue_seeded_summary.json'))
    ap.add_argument("--clean64-seed44", default="")
    ap.add_argument("--clean64-seed44-a02", default=str(_public_path('experiments/archive/relation_learning/data/clean_pres62064_ftseed44_superglue')))
    args = ap.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    seed42 = parse_seed42_profile(args.seed42_profile if args.seed42_profile.is_absolute() else ROOT / args.seed42_profile)
    clean_path, clean_source = pick_clean64_summary(args.clean64_seed44, args.clean64_seed44_a02)
    seed44 = {
        "coherent86": parse_seeded_summary(args.coherent86_seed44 if args.coherent86_seed44.is_absolute() else ROOT / args.coherent86_seed44),
        "ms_acquisition": parse_seeded_summary(args.ms_seed44 if args.ms_seed44.is_absolute() else ROOT / args.ms_seed44),
        "clean64": parse_seeded_summary(clean_path) if clean_path else None,
    }
    valid44 = {k: v for k, v in seed44.items() if v and v.get("valid_for_superglue_comparison")}
    # For comparison arithmetic, admit only valid seed44 summaries.
    seed44_for_compare = valid44

    task_level: List[Dict[str, Any]] = []
    for task in TASKS:
        row: Dict[str, Any] = {"task": task, "primary_metric": PRIMARY[task]}
        for model in MODELS:
            row[f"{model}_seed42"] = (seed42.get(model) or {}).get("tasks", {}).get(task)
            row[f"{model}_seed44"] = (valid44.get(model) or {}).get("tasks", {}).get(task)
        task_level.append(row)

    result: Dict[str, Any] = {
        "status": "HARDENED_FTSEED_SUPERGLUE_COMPARISON",
        "created_utc": now(),
        "purpose": "Validate and compare matched downstream fine-tuning seed SuperGLUE means for coherent86, exact (M,S), and clean64.",
        "seed42_profile": rel(args.seed42_profile),
        "seed42_results": seed42,
        "seed44_results": seed44,
        "clean64_seed44_path_source": clean_source,
        "cross_seed_comparison": {
            "seed42": comparison_block(seed42),
            "seed44": comparison_block(seed44_for_compare),
        },
        "task_level": task_level,
        "completeness": {
            "seed42_complete": sum(1 for m in MODELS if m in seed42),
            "seed44_present": sum(1 for m in MODELS if seed44.get(m) is not None),
            "seed44_valid_complete": sum(1 for m in MODELS if seed44.get(m) and seed44[m].get("valid_for_superglue_comparison")),
            "all_three_seed44_valid": all(seed44.get(m) and seed44[m].get("valid_for_superglue_comparison") for m in MODELS),
        },
        "important_scope": "This comparison perturbs only SuperGLUE downstream fine-tuning seed. It is not a new full Overall coordinate and it ignores optional wrapper Overall values.",
    }
    for model in MODELS:
        r42 = seed42.get(model)
        r44 = valid44.get(model)
        if r42 and r44:
            result["cross_seed_comparison"][f"{model}_seed44_minus_seed42"] = r44["superglue_primary_mean"] - r42["superglue_primary_mean"]

    out_json = out / "ftseed_comparison_hardened.json"
    out_md = out / "ftseed_comparison_hardened.md"
    out_csv = out / "ftseed_task_comparison_hardened.csv"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out_md, result)
    write_csv(out_csv, result)
    print(json.dumps({
        "status": result["status"],
        "seed42_complete": result["completeness"]["seed42_complete"],
        "seed44_present": result["completeness"]["seed44_present"],
        "seed44_valid_complete": result["completeness"]["seed44_valid_complete"],
        "cross_seed_comparison": result["cross_seed_comparison"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
