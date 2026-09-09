#!/usr/bin/env python3
"""research SuperGLUE old-path spread and fixed lever reading rules.

This CPU-only script collates SuperGLUE values that were produced when AutoModel
silently loaded the same stock DeBERTa encoder for chck82/coherent/dense adapter
checkpoints.  The resulting spread is the observed fixed-protocol noise floor for
interpreting the repaired faithful AutoModel SuperGLUE comparison.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path.cwd()
OUT = ROOT / "experiments/archive/relation_learning/data/superglue_old_path_spread"
OUT_JSON = OUT / "superglue_old_path_spread_and_lever_rules.json"
OUT_MD = OUT / "superglue_old_path_spread_and_lever_rules.md"

RUNS = {
    "chck82_old_stripped": {
        "path": ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json",
        "kind": "per_target",
        "role": "protected chck82 slow-adapter reference through the old AutoModel path",
    },
    "coherent86_old_stripped": {
        "path": ROOT / "experiments/archive/frontier_consolidation/data/private_scale_superglue_summary_alpha0p75/coherent86_private_scale_0p75_sg_retry_superglue_summary.json",
        "kind": "summary",
        "role": "coherent86 alpha0.75 private endpoint through the old AutoModel path",
    },
    "a01_dense62064_old_stripped": {
        "path": ROOT / "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json",
        "kind": "per_target",
        "role": "dense focus seed62064 through the old AutoModel path",
    },
    "a01_dense62065_old_stripped_current_file": {
        "path": ROOT / "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json",
        "kind": "per_target",
        "role": "dense focus seed62065 current file through the old AutoModel path if complete; contemporaneous notes may still treat it as interim",
    },
}

TASKS = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]
PRIMARY = {"boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1", "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"}
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def details_from_per_target(data: dict[str, Any]) -> tuple[float | None, dict[str, float], dict[str, Any]]:
    rec = (data.get("tasks") or {}).get("SuperGLUE") or {}
    sg = rec.get("superglue_mean")
    details: dict[str, float] = {}
    meta: dict[str, Any] = {"record_keys": sorted(rec.keys())}
    for d in rec.get("superglue_primary_metric_details") or []:
        task = d.get("task")
        if task in TASKS and d.get("score") is not None:
            details[task] = float(d["score"])
    # Some old payloads store only task rows with accuracy; use them only if the
    # patched primary-metric details are absent.
    if not details:
        tasks = rec.get("tasks") or []
        if isinstance(tasks, list):
            for row in tasks:
                task = row.get("task")
                metric = PRIMARY.get(task or "")
                if task in TASKS and metric in row:
                    details[task] = float(row[metric])
    if sg is None and len(details) == len(TASKS):
        sg = sum(details.values()) / len(details)
    meta["task_count"] = len(details)
    return (float(sg) if sg is not None else None), details, meta


def details_from_summary(data: dict[str, Any]) -> tuple[float | None, dict[str, float], dict[str, Any]]:
    sg = data.get("superglue") or data.get("superglue_mean")
    details: dict[str, float] = {}
    for d in data.get("superglue_subtask_metrics") or data.get("superglue_primary_metric_details") or []:
        task = d.get("task")
        if task in TASKS and d.get("score") is not None:
            details[task] = float(d["score"])
    if sg is None and len(details) == len(TASKS):
        sg = sum(details.values()) / len(details)
    return (float(sg) if sg is not None else None), details, {"task_count": len(details), "record_keys": sorted(data.keys())}


def cheap_scores_from_any(path: pathlib.Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    data = read_json(path)
    scores = None
    for key in ("score_arithmetic", "official_overall"):
        if isinstance(data.get(key), dict) and isinstance(data[key].get("scores"), dict):
            scores = data[key]["scores"]
            break
    if scores is None and isinstance(data.get("scores"), dict):
        scores = data["scores"]
    if scores is None and isinstance(data.get("cheap_scores"), dict):
        scores = data["cheap_scores"]
    if not isinstance(scores, dict):
        return None
    if all(c in scores and scores[c] is not None for c in CHEAP_COLUMNS):
        return {c: float(scores[c]) for c in CHEAP_COLUMNS}
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    vals: list[float] = []
    per_task_values: dict[str, list[float]] = {t: [] for t in TASKS}
    for label, spec in RUNS.items():
        path = spec["path"]
        rec: dict[str, Any] = {"path": rel(path), "exists": path.exists(), "role": spec["role"]}
        if path.exists():
            data = read_json(path)
            if spec["kind"] == "summary":
                sg, details, meta = details_from_summary(data)
            else:
                sg, details, meta = details_from_per_target(data)
            rec.update(meta)
            rec["superglue"] = sg
            rec["task_scores"] = details
            if sg is not None:
                vals.append(sg)
            for t, v in details.items():
                per_task_values[t].append(v)
        records[label] = rec

    spread = {
        "n": len(vals),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "range": (max(vals) - min(vals)) if vals else None,
        "mean": (sum(vals) / len(vals)) if vals else None,
        "sample_std": (statistics.stdev(vals) if len(vals) >= 2 else 0.0),
        "values": {k: records[k].get("superglue") for k in records},
    }
    per_task_spread = {}
    for t, xs in per_task_values.items():
        if xs:
            per_task_spread[t] = {"n": len(xs), "min": min(xs), "max": max(xs), "range": max(xs) - min(xs), "values": xs}

    coherent_summary = ROOT / "experiments/archive/frontier_consolidation/data/private_scale_superglue_summary_alpha0p75/coherent86_private_scale_0p75_sg_retry_superglue_summary.json"
    coh = read_json(coherent_summary)
    cheap_scores = {c: float(coh["cheap_scores"][c]) for c in CHEAP_COLUMNS}
    old_coherent_sg = float(coh["superglue"])
    old_coherent_aoa = float(coh.get("aoa_assumed_for_projection", 0.0))
    old_coherent_overall = sum(list(cheap_scores.values()) + [old_coherent_sg, old_coherent_aoa]) / 9.0

    result = {
        "status": "SUPERGLUE_OLD_PATH_SPREAD_AND_LEVER_RULES",
        "question": "How should repaired faithful AutoModel SuperGLUE and multiple legal levers be read before their endpoint scores arrive?",
        "old_path_identity_basis": {
            "source": "research/research hidden-state tests showed old AutoModel loading reduced chck82/coherent/dense custom checkpoints to the same stock DeBERTa encoder; these old SuperGLUE values are therefore observed repeats of one stripped encoder under the fixed official protocol.",
            "important_separation": "The spread bounds observed fixed-protocol run variation, not all fine-tuning seed sensitivity; a final endpoint should receive an additional SuperGLUE fine-tuning seed run before submission.",
        },
        "records": records,
        "macro_superglue_spread": spread,
        "per_task_spread": per_task_spread,
        "old_coherent86_reference": {
            "cheap_scores": cheap_scores,
            "old_stripped_superglue": old_coherent_sg,
            "old_measured_aoa_or_projection_value": old_coherent_aoa,
            "old_overall_reproduced": old_coherent_overall,
            "old_reported_overall_source": rel(coherent_summary),
        },
        "fixed_rules_before_new_results": {
            "superglue_reading": "A repaired coherent86 SuperGLUE movement whose magnitude is no larger than the old stripped-path range should be read as ordinary fixed-protocol variation. A movement outside that range is model-file fidelity evidence. Repaired coherent86 versus repaired chck82 also changes the number of trainable encoder parameters during downstream fine-tuning, so it is not evidence that coherent replay content helps SuperGLUE.",
            "v5_bar": "The training contribution must beat faithful v4, where faithful v4 is coherent86 cheap7 plus repaired coherent86 SuperGLUE plus measured coherent86 AoA. The historical 42.1210 number remains the old stripped-path reference, not the bar for a new trained v5.",
            "lever_entry_rule": "A lever can enter the composed candidate only when both private seeds move the same relevant columns in the same direction beyond the two-seed coherent band and no important column falls beyond its band. The composed candidate is then fixed once and trained/evaluated across two private seeds without further post-result mixing.",
            "format_hierarchy": "coherent_unsplit_special is the special-token control; isolated_all adds row isolation; half_coherent_half_isolated tests context-presence conditioning. Read them in that order.",
            "superglue_seed_repeat": "Before any submission, rerun SuperGLUE for the selected final endpoint under one additional fine-tuning seed; CB/COPA/WSC-like small validation tasks can be noisy even after file loading is repaired.",
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research SuperGLUE old-path spread and fixed lever rules\n\n")
    lines.append("This record is fixed before reading the repaired format endpoints or the repaired faithful SuperGLUE outputs.\n\n")
    lines.append("## Old stripped AutoModel SuperGLUE spread\n\n")
    lines.append("research/research showed that the old AutoModel path loaded the same stock encoder for custom checkpoints. The following values are therefore observed fixed-protocol repeats of one stripped encoder, not evidence for differences among the adapter/private endpoints.\n\n")
    lines.append("| label | SuperGLUE | role |\n|---|---:|---|\n")
    for label, rec in records.items():
        sg = rec.get("superglue")
        lines.append(f"| `{label}` | {sg if sg is not None else 'NA'} | {rec['role']} |\n")
    lines.append("\n")
    lines.append(f"Macro spread: min `{spread['min']}`, max `{spread['max']}`, range `{spread['range']}`, sample std `{spread['sample_std']}`.\n\n")
    lines.append("Per-task spread is concentrated in the components whose downstream run is less stable; tasks with zero spread in this record should not be assumed noiseless under a different fine-tuning seed.\n\n")
    lines.append("| task | range | values |\n|---|---:|---|\n")
    for task in TASKS:
        ps = per_task_spread.get(task, {})
        lines.append(f"| `{task}` | {ps.get('range')} | {ps.get('values')} |\n")
    lines.append("\n## Fixed reading rules before new endpoint scores\n\n")
    for key, text in result["fixed_rules_before_new_results"].items():
        lines.append(f"- **{key}**: {text}\n")
    lines.append("\n")
    lines.append(f"Old coherent86 Overall reproduced from old stripped SuperGLUE and AoA0: `{old_coherent_overall}`. The new training target is not this number; it is faithful v4 after repaired SuperGLUE returns.\n\n")
    lines.append(f"JSON: `{rel(OUT_JSON)}`\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD), "macro_range": spread["range"], "old_coherent86_overall": old_coherent_overall}, indent=2), flush=True)


if __name__ == "__main__":
    main()
