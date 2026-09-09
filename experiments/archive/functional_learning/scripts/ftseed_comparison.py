#!/usr/bin/env python3
"""research: matched finetuning-seed SuperGLUE comparison.

Compares SuperGLUE primary-metric means across finetune seeds 42 (historical)
and 44 (new matched comparison) for coherent86, exact (M,S), and clean64.
Reports whether the preservation-specific increment and policy-vs-parent gain
persist across downstream seeds.
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
import sys
import time
from statistics import mean
from typing import Any

ROOT = _public_path('experiments/archive/functional_learning/scripts/ftseed_comparison.py')
while ROOT.name != "Sessions" and ROOT != _public_path('experiments/archive/functional_learning'):
    ROOT = _public_path('experiments/archive/functional_learning')
ROOT = _public_path('experiments/archive/functional_learning')

OUT_DEFAULT = _public_path('experiments/archive/functional_learning/data/ftseed_comparison')

PRIMARY = {
    "boolq": "accuracy", "multirc": "accuracy", "rte": "accuracy",
    "wsc": "accuracy", "mrpc": "f1", "qqp": "f1", "mnli": "accuracy",
}
TASKS = sorted(PRIMARY.keys())

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def rel(p: pathlib.Path) -> str:
    try: return str(p.resolve().relative_to(ROOT))
    except: return str(p)

def rj(p: pathlib.Path) -> Any:
    return json.loads(p.read_text("utf-8"))


def parse_seed42_from_profile(profile_path: pathlib.Path) -> dict[str, dict]:
    """Parse seed42 per-task scores from research all-completed SuperGLUE profile."""
    if not profile_path.exists():
        return {}
    data = rj(profile_path)
    mm = data.get("model_metrics", {})
    result = {}
    # Map profile model names to our comparison names
    name_map = {
        "coherent86": "coherent86",
        "ms_acquisition_seed62064_MS": "ms_acquisition",
        "clean_pres_seed62064_MSplusKL": "clean64",
    }
    for profile_name, our_name in name_map.items():
        model_data = mm.get(profile_name)
        if model_data is None:
            continue
        task_block = model_data.get("tasks", model_data)
        tasks = {}
        for task in TASKS:
            task_data = task_block.get(task, {})
            if isinstance(task_data, dict):
                pm = PRIMARY[task]
                score = task_data.get(pm)
                if score is not None:
                    tasks[task] = float(score)
        if len(tasks) == 7:
            sg = mean(list(tasks.values()))
            result[our_name] = {"tasks": tasks, "superglue_primary_mean": sg, "path": rel(profile_path), "seed": 42}
    return result


def parse_seed44_summary(path: pathlib.Path) -> dict | None:
    """Parse a research faithful_superglue_seeded_summary.json."""
    if not path.exists():
        return None
    data = rj(path)
    tasks = {}
    for rec in data.get("task_records", []):
        if isinstance(rec, dict) and rec.get("task"):
            t = rec["task"]
            tasks[t] = rec.get("primary_score")
    if len(tasks) < 7:
        return None
    sg = data.get("superglue_primary_metric_mean")
    if sg is None:
        sg = mean([tasks[t] for t in TASKS if tasks.get(t) is not None])
    return {"tasks": tasks, "superglue_primary_mean": sg, "path": rel(path), "seed": data.get("finetune_seed", 44)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    # seed42 from research all-completed profile
    ap.add_argument("--seed42-profile", default=str(_public_path('experiments/archive/functional_learning/data/superglue_all_completed_profile/superglue_all_completed_profile.json')))
    # seed44 summaries (research faithful_superglue_seeded_summary.json)
    ap.add_argument("--coherent86-seed44", default=str(_public_path('experiments/archive/functional_learning/data/matched_ftseed44_superglue/coherent86_alpha075_ftseed44/faithful_superglue_seeded_summary.json')))
    ap.add_argument("--ms-seed44", default=str(_public_path('experiments/archive/functional_learning/data/ms_ftseed44_superglue/densemask_sparselabel_seed62064_ftseed44/faithful_superglue_seeded_summary.json')))
    ap.add_argument("--clean64-seed44", default="")
    ap.add_argument("--clean64-seed44-a02", default=str(_public_path('experiments/archive/relation_learning/data/clean_pres62064_ftseed44_superglue')))
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Parse seed42 from research all-completed profile
    s42 = parse_seed42_from_profile(pathlib.Path(args.seed42_profile))

    # Parse seed44
    s44: dict[str, Any] = {
        "coherent86": parse_seed44_summary(pathlib.Path(args.coherent86_seed44)),
        "ms_acquisition": parse_seed44_summary(pathlib.Path(args.ms_seed44)),
    }

    # Try to find clean64 seed44 from either explicit path or the result-directory search
    clean64_seed44_path = pathlib.Path(args.clean64_seed44) if args.clean64_seed44 else None
    if clean64_seed44_path is None or not clean64_seed44_path.exists():
        # Search the result directory.
        a02_root = pathlib.Path(args.clean64_seed44_a02)
        for candidate in sorted(a02_root.rglob("faithful_superglue_seeded_summary.json")):
            s44["clean64"] = parse_seed44_summary(candidate)
            if s44["clean64"] is not None:
                break
    else:
        s44["clean64"] = parse_seed44_summary(clean64_seed44_path)
    if s44.get("clean64") is None:
        s44["clean64"] = None

    # Build comparison
    result: dict[str, Any] = {
        "status": "FTSEED_COMPARISON",
        "created_utc": now(),
        "purpose": "Matched downstream fine-tuning seed comparison for preservation-specific and policy-vs-parent SuperGLUE claims",
        "seed42_results": {},
        "seed44_results": {},
        "cross_seed_comparison": {},
        "task_level_seed_deltas": [],
    }

    models = ["coherent86", "ms_acquisition", "clean64"]
    for model in models:
        r42 = s42.get(model)
        r44 = s44.get(model)
        if r42:
            result["seed42_results"][model] = {"superglue": r42["superglue_primary_mean"], "path": r42["path"]}
        else:
            result["seed42_results"][model] = {"superglue": None, "missing": True}
        if r44:
            result["seed44_results"][model] = {"superglue": r44["superglue_primary_mean"], "path": r44["path"]}
        else:
            result["seed44_results"][model] = {"superglue": None, "missing": True}

    # Key comparisons at each seed
    for seed_label, data in [("seed42", s42), ("seed44", s44)]:
        comparisons = {}
        c86 = data.get("coherent86")
        ms = data.get("ms_acquisition")
        cl = data.get("clean64")
        if cl and c86:
            comparisons["clean64_minus_coherent86"] = cl["superglue_primary_mean"] - c86["superglue_primary_mean"]
        if cl and ms:
            comparisons["clean64_minus_ms_preservation_specific"] = cl["superglue_primary_mean"] - ms["superglue_primary_mean"]
        if ms and c86:
            comparisons["ms_minus_coherent86_acquisition_only"] = ms["superglue_primary_mean"] - c86["superglue_primary_mean"]
        result["cross_seed_comparison"][seed_label] = comparisons

    # Seed delta for each model
    for model in models:
        r42 = s42.get(model)
        r44 = s44.get(model)
        if r42 and r44:
            result["cross_seed_comparison"][f"{model}_seed44_minus_seed42"] = r44["superglue_primary_mean"] - r42["superglue_primary_mean"]

    # Task-level comparison
    for task in TASKS:
        row = {"task": task, "primary_metric": PRIMARY[task]}
        for model in models:
            for seed_label, data in [("seed42", s42), ("seed44", s44)]:
                rec = data.get(model)
                key = f"{model}_{seed_label}"
                row[key] = rec["tasks"].get(task) if rec else None
        result["task_level_seed_deltas"].append(row)

    # Overall interpretation
    n_complete_42 = sum(1 for m in models if s42.get(m) is not None)
    n_complete_44 = sum(1 for m in models if s44.get(m) is not None)
    result["completeness"] = {
        "seed42_complete": n_complete_42,
        "seed44_complete": n_complete_44,
        "all_three_seed44_available": n_complete_44 == 3,
        "interpretation_valid": n_complete_44 >= 2,  # at minimum coherent86 + one candidate
    }

    # Write outputs
    (out / "ftseed_comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", "utf-8")

    # Markdown
    lines = ["# research matched finetuning-seed SuperGLUE comparison\n\n"]
    lines.append(f"Created: `{result['created_utc']}`\n\n")
    lines.append(f"Complete: seed42={n_complete_42}/3, seed44={n_complete_44}/3\n\n")

    lines.append("## SuperGLUE primary-metric means\n\n")
    lines.append("| model | seed42 | seed44 | Δ(44-42) |\n|---|---|---|---|\n")
    for model in models:
        v42 = result["seed42_results"].get(model, {}).get("superglue")
        v44 = result["seed44_results"].get(model, {}).get("superglue")
        delta = (v44 - v42) if (v42 is not None and v44 is not None) else None
        lines.append(f"| {model} | {v42 if v42 else 'missing'} | {v44 if v44 else 'missing'} | {f'{delta:+.4f}' if delta else '-'} |\n")

    lines.append("\n## Key comparisons\n\n")
    lines.append("| comparison | seed42 | seed44 | sign consistent |\n|---|---|---|---|\n")
    for comp_key in ["clean64_minus_coherent86", "clean64_minus_ms_preservation_specific", "ms_minus_coherent86_acquisition_only"]:
        v42 = result["cross_seed_comparison"].get("seed42", {}).get(comp_key)
        v44 = result["cross_seed_comparison"].get("seed44", {}).get(comp_key)
        consistent = "✓" if (v42 is not None and v44 is not None and v42 * v44 > 0) else ("✗" if (v42 is not None and v44 is not None) else "-")
        lines.append(f"| {comp_key} | {f'{v42:+.4f}' if v42 else '-'} | {f'{v44:+.4f}' if v44 else '-'} | {consistent} |\n")

    lines.append("\n## Task-level details\n\n")
    lines.append("| task | metric | coherent86_s42 | coherent86_s44 | ms_s42 | ms_s44 | clean64_s42 | clean64_s44 |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for row in result["task_level_seed_deltas"]:
        vals = []
        for model in models:
            for sl in ["seed42", "seed44"]:
                v = row.get(f"{model}_{sl}")
                vals.append(f"{v:.4f}" if v is not None else "-")
        lines.append(f"| {row['task']} | {row['primary_metric']} | {' | '.join(vals)} |\n")

    if n_complete_44 < 3:
        lines.append(f"\n**Note:** Only {n_complete_44}/3 seed44 results are available. ")
        missing = [m for m in models if s44.get(m) is None]
        lines.append(f"Missing: {', '.join(missing)}. Rerun this script after all jobs complete.\n")

    (out / "ftseed_comparison.md").write_text("".join(lines), "utf-8")

    # CSV
    with open(out / "ftseed_task_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["task", "metric"]
        for model in models:
            for sl in ["seed42", "seed44"]:
                header.append(f"{model}_{sl}")
        w.writerow(header)
        for row in result["task_level_seed_deltas"]:
            vals = [row["task"], row["primary_metric"]]
            for model in models:
                for sl in ["seed42", "seed44"]:
                    v = row.get(f"{model}_{sl}")
                    vals.append(f"{v:.6f}" if v is not None else "")
            w.writerow(vals)

    print(json.dumps({
        "status": result["status"],
        "seed42_complete": n_complete_42,
        "seed44_complete": n_complete_44,
        "cross_seed_comparison": result["cross_seed_comparison"],
        "out_json": rel(out / "ftseed_comparison.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
