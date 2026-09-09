#!/usr/bin/env python3
"""Extract source values needed for research review.

This is read-only over research artifacts and writes a compact source-derived
summary for downstream report checking.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
OUT = _public_path('experiments/archive/relation_learning/data/report_precision_review')
OUT.mkdir(parents=True, exist_ok=True)

def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)

def load(path: str) -> Any:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)

paths = {
    "research": "experiments/archive/functional_learning/data/same_coordinate_with_ordinary_complete/same_coordinate_with_ordinary.json",
    "admission": "experiments/archive/functional_learning/data/strict_split_eval_admission_after_ms_multirc/strict_split_eval_admission.json",
    "entry": "experiments/archive/functional_learning/data/official_export_finetune_entry/official_export_finetune_entry_check.json",
    "ms65_train": "experiments/archive/relation_learning/data/ms_acquisition_seed62065/train_summary.json",
    "ms64_train": "experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/train_summary.json",
    "v5_bridge_metadata": "experiments/archive/functional_learning/data/v5_candidate_export/Qiushi-Engine-Principle-Guided-Frontier-Advancement/bridge_metadata.json",
}

summary: dict[str, Any] = {"sources": paths}
research = load(paths["research"])
summary["keys"] = list(research.keys())
summary["seed64_endpoints"] = {}
for ep in research.get("endpoints", []):
    name = ep.get("endpoint") or ep.get("label") or ep.get("name") or ep.get("endpoint_label")
    comps = None
    if isinstance(ep.get("zero_reading"), dict):
        comps = ep["zero_reading"].get("scores")
    if comps is None:
        comps = ep.get("scores")
    if comps is None and isinstance(ep.get("overall"), dict):
        comps = ep["overall"].get("components")
    ov = None
    if isinstance(ep.get("overall"), dict):
        ov = ep["overall"].get("Overall")
    if ov is None:
        ov = ep.get("Overall")
    summary["seed64_endpoints"][str(name)] = {"Overall": ov, "scores": comps}

admission = load(paths["admission"])
summary["split_admission"] = {}
for name in ["o62065", "ms62065"]:
    ep = admission["endpoints"].get(name, {})
    summary["split_admission"][name] = {
        "scores": ep.get("scores"),
        "superglue_task_scores": ep.get("superglue_task_scores"),
        "Overall": ep.get("Overall"),
        "complete_for_overall": ep.get("complete_for_overall"),
        "missing_scores": ep.get("missing_scores"),
    }

entry = load(paths["entry"])
summary["official_entry_check"] = {"all_valid": entry.get("all_valid"), "records": {}}
for rec in entry.get("records", []):
    ident = rec.get("identity_from_same_export_path", {})
    summary["official_entry_check"]["records"][rec["label"]] = {
        "primary_score": rec.get("primary_score"),
        "metrics_percent": rec.get("metrics_percent"),
        "returncode": rec.get("returncode"),
        "valid_release_path_entry_record": rec.get("valid_release_path_entry_record"),
        "AutoModel": ident.get("AutoModel"),
        "AutoModelForMaskedLM": ident.get("AutoModelForMaskedLM"),
        "package_dir": rec.get("package_dir"),
        "package_model_sha256": rec.get("package_model_sha256"),
    }

for k in ["ms64_train", "ms65_train"]:
    data = load(paths[k])
    summary[k] = {
        "method": data.get("method"),
        "completed_updates": data.get("completed_updates"),
        "total_words_consumed": data.get("total_words_consumed"),
        "total_focus_targets": data.get("total_focus_targets"),
        "total_ordinary_targets": data.get("total_ordinary_targets"),
        "focus_selected_groups_total": data.get("focus_selected_groups_total"),
        "focus_candidate_groups_total": data.get("focus_candidate_groups_total"),
        "final_update": {
            "focus_target_tokens": data.get("final_update", {}).get("focus_target_tokens"),
            "ordinary_target_tokens": data.get("final_update", {}).get("ordinary_target_tokens"),
            "focus_selected_groups": data.get("final_update", {}).get("focus_selected_groups"),
            "focus_candidate_groups": data.get("final_update", {}).get("focus_candidate_groups"),
            "qwen_rows": data.get("final_update", {}).get("qwen_rows"),
            "qwen_focus_rows": data.get("final_update", {}).get("qwen_focus_rows"),
            "ordinary_wwm_rows": data.get("final_update", {}).get("ordinary_wwm_rows"),
        },
        "exposure_accounting": data.get("exposure_accounting"),
    }

try:
    bridge = load(paths["v5_bridge_metadata"])
except FileNotFoundError:
    bridge = {}
summary["v5_bridge_metadata_keys"] = list(bridge.keys()) if isinstance(bridge, dict) else []
summary["v5_bridge_metadata"] = bridge

out_json = _public_path('experiments/archive/relation_learning/data/report_precision_review/authoritative_values.json')
out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

# Human-readable compact view.
lines = ["# research authoritative values for report precision", ""]
lines.append("## (M,S) construction counts")
for k in ["ms64_train", "ms65_train"]:
    v = summary[k]
    fu = v["final_update"]
    lines.append(f"- {k}: total_focus_targets={v['total_focus_targets']}, total_ordinary_targets={v['total_ordinary_targets']}, focus_candidate_groups_total={v.get('focus_candidate_groups_total')}, focus_selected_groups_total={v.get('focus_selected_groups_total')}; final update focus={fu['focus_target_tokens']}, ordinary={fu['ordinary_target_tokens']}, candidate_groups={fu['focus_candidate_groups']}, selected_groups={fu['focus_selected_groups']}, qwen_rows={fu['qwen_rows']}, ordinary_wwm_rows={fu['ordinary_wwm_rows']}.")
lines.append("")
lines.append("## Official exported-directory entry scores")
for label, rec in summary["official_entry_check"]["records"].items():
    lines.append(f"- {label}: all_valid={summary['official_entry_check']['all_valid']}, primary_score={rec['primary_score']}, metrics={rec['metrics_percent']}, AutoModel_params={rec['AutoModel']['total_params']}, private_params={rec['AutoModel']['private_params']}, slow_params_probe={rec['AutoModel'].get('slow_params')}, adapter_scale_values_probe={rec['AutoModel'].get('adapter_scale_values')}, private_scale_values_probe={rec['AutoModel'].get('private_scale_values')}.")
lines.append("")
lines.append("## Seed64 ladder")
for name, rec in summary["seed64_endpoints"].items():
    scores = rec.get("scores") or {}
    lines.append(f"- {name}: Overall={rec.get('Overall')}, Entity={scores.get('Entity')}, GlobalPIQA={scores.get('GlobalPIQA')}, BLiMP={scores.get('BLiMP')}, Supplement={scores.get('Supplement')}.")
lines.append("")
lines.append("## Split admission")
for name, rec in summary["split_admission"].items():
    lines.append(f"- {name}: complete={rec.get('complete_for_overall')}, Overall={rec.get('Overall')}, scores={rec.get('scores')}, SuperGLUE tasks={rec.get('superglue_task_scores')}, missing={rec.get('missing_scores')}.")
lines.append("")
lines.append(f"Detailed JSON: `{rel(out_json)}`")
(_public_path('research/documents/relation_learning/data/report_precision_review/authoritative_values.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"status": "ok", "out_json": rel(out_json), "out_md": rel(_public_path('research/documents/relation_learning/data/report_precision_review/authoritative_values.md'))}, ensure_ascii=False), flush=True)
