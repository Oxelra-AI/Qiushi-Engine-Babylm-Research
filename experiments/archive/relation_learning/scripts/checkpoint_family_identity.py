#!/usr/bin/env python3
"""research: checkpoint-family identity table for relation evidence and adapter lineage.

This script does not score models.  It reads representative checkpoint configs and
safetensor headers to separate stock relation-family checkpoints from the
adapter-scaled compact-view-reinvest lineage affected by the plain-loader error.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import time
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive/relation_learning"
OUT = WS / "data/checkpoint_family_identity"

REPRESENTATION_FRONTIER_STUDIES_RUNS = ROOT / "experiments/archive/frontier_consolidation/training/runs"
FUNCTIONAL_RELATION_STUDIES_RUNS = ROOT / "experiments/archive/relation_learning/training/runs"
COMPACT_EXPERIENCE_RUNS = ROOT / "experiments/archive/compact_experience/training/runs"

REPRESENTATIVES: list[dict[str, Any]] = [
    {
        "family": "REPRESENTATION_FRONTIER_STUDIES_CLEAN_REPEAT_VIEW_stock_DeBERTa",
        "role": "CLEAN",
        "path": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock relation-family model; plain AutoModel loader should be the deployed object",
    },
    {
        "family": "REPRESENTATION_FRONTIER_STUDIES_CLEAN_REPEAT_VIEW_stock_DeBERTa",
        "role": "REPEAT",
        "path": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock relation-family model; plain AutoModel loader should be the deployed object",
    },
    {
        "family": "REPRESENTATION_FRONTIER_STUDIES_CLEAN_REPEAT_VIEW_stock_DeBERTa",
        "role": "VIEW",
        "path": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock relation-family model; plain AutoModel loader should be the deployed object",
    },
    {
        "family": "split_same_multiset_stock_DeBERTa",
        "role": "REPEAT_SPLIT",
        "path": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock split-locality model; plain AutoModel loader should be the deployed object",
    },
    {
        "family": "split_same_multiset_stock_DeBERTa",
        "role": "VIEW_SPLIT",
        "path": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock split-locality model; plain AutoModel loader should be the deployed object",
    },
    {
        "family": "hash_and_half_view_stock_DeBERTa",
        "role": "HASH_MIX",
        "path": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_hash_mixed_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock control-family model; plain AutoModel loader should be the deployed object",
    },
    {
        "family": "hash_and_half_view_stock_DeBERTa",
        "role": "HALF_VIEW",
        "path": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_half_view_noexact_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock control-family model; plain AutoModel loader should be the deployed object",
    },
    {
        "family": "RoBERTa_boundary_stock",
        "role": "VIEW",
        "path": REPRESENTATION_FRONTIER_STUDIES_RUNS / "roberta_view_dose2p64x_matched_rowholdout_100M_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock RoBERTa boundary model; plain AutoModel loader should be the deployed object",
    },
    {
        "family": "causal_GPT_boundary_stock",
        "role": "CLEAN",
        "path": FUNCTIONAL_RELATION_STUDIES_RUNS / "causal_gpt_c_dose2p64x_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock causal-LM boundary model; plain AutoModel loader should be the deployed object for causal scoring",
    },
    {
        "family": "COMPACT_EXPERIENCE_five_arm_stock_DeBERTa",
        "role": "OFF",
        "path": COMPACT_EXPERIENCE_RUNS / "official_lengthmatched_16k_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock COMPACT_EXPERIENCE relation-design model; plain AutoModel loader should be the deployed object",
    },
    {
        "family": "COMPACT_EXPERIENCE_five_arm_stock_DeBERTa",
        "role": "ALN",
        "path": COMPACT_EXPERIENCE_RUNS / "qwen_clean_aligned_16k_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock COMPACT_EXPERIENCE aligned-restatement model; plain AutoModel loader should be the deployed object",
    },
    {
        "family": "COMPACT_EXPERIENCE_five_arm_stock_DeBERTa",
        "role": "SHUF",
        "path": COMPACT_EXPERIENCE_RUNS / "qwen_shuffled_control_16k_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock COMPACT_EXPERIENCE shuffled-correspondence model; plain AutoModel loader should be the deployed object",
    },
    {
        "family": "COMPACT_EXPERIENCE_five_arm_stock_DeBERTa",
        "role": "DUP",
        "path": COMPACT_EXPERIENCE_RUNS / "selected_original_dup_all_16k_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock COMPACT_EXPERIENCE exact-duplication model; plain AutoModel loader should be the deployed object",
    },
    {
        "family": "COMPACT_EXPERIENCE_five_arm_stock_DeBERTa",
        "role": "SEP",
        "path": COMPACT_EXPERIENCE_RUNS / "qwen_separated_pair_16k_seed43022/hf_model/chck_100M",
        "lineage_reading": "stock COMPACT_EXPERIENCE separated-pair model; plain AutoModel loader should be the deployed object",
    },
    {
        "family": "compact_view_reinvest_adapter_scaled",
        "role": "base_seed43022",
        "path": REPRESENTATION_FRONTIER_STUDIES_RUNS / "adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M",
        "lineage_reading": "adapter-scaled compact-view-reinvest model; trusted custom loading is required",
    },
    {
        "family": "compact_view_reinvest_adapter_scaled",
        "role": "base_seed43122",
        "path": REPRESENTATION_FRONTIER_STUDIES_RUNS / "adapter128_scale1p75_seed43122_dense100M/hf_model/chck_100M",
        "lineage_reading": "adapter-scaled compact-view-reinvest model; trusted custom loading is required",
    },
    {
        "family": "state_update_adapter_scaled",
        "role": "state_seed43022",
        "path": FUNCTIONAL_RELATION_STUDIES_RUNS / "state_update_plainuse_seed43022/hf_model/chck_86M",
        "lineage_reading": "adapter-scaled state-update model; trusted custom loading is required",
    },
    {
        "family": "state_update_adapter_scaled",
        "role": "state_seed43122",
        "path": FUNCTIONAL_RELATION_STUDIES_RUNS / "state_update_plainuse_seed43122/hf_model/chck_86M",
        "lineage_reading": "adapter-scaled state-update model; trusted custom loading is required",
    },
    {
        "family": "probe_clean_restatement_dose_adapter_scaled",
        "role": "dose21_seed43022",
        "path": FUNCTIONAL_RELATION_STUDIES_RUNS / "probe_clean_restatement_dose21_seed43022/hf_model/chck_100M",
        "lineage_reading": "adapter-scaled dose model; trusted custom loading is required",
    },
    {
        "family": "probe_clean_restatement_dose_adapter_scaled",
        "role": "dose25_seed43022",
        "path": FUNCTIONAL_RELATION_STUDIES_RUNS / "probe_clean_restatement_dose25_seed43022/hf_model/chck_100M",
        "lineage_reading": "adapter-scaled dose model; trusted custom loading is required",
    },
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def count_safetensor_params(model_dir: pathlib.Path) -> dict[str, Any]:
    model_file = model_dir / "model.safetensors"
    if not model_file.exists():
        bin_file = model_dir / "pytorch_model.bin"
        return {"weight_file": rel(bin_file), "weight_file_exists": bin_file.exists(), "safetensor_param_count": None, "adapter_tensor_param_count": None, "tensor_count": None}
    from safetensors import safe_open

    total = 0
    adapter = 0
    tensor_count = 0
    adapter_keys: list[str] = []
    with safe_open(str(model_file), framework="pt", device="cpu") as f:
        for key in f.keys():
            shape = f.get_tensor(key).shape
            n = int(math.prod(shape))
            total += n
            tensor_count += 1
            if "adapter" in key.lower():
                adapter += n
                if len(adapter_keys) < 8:
                    adapter_keys.append(key)
    return {
        "weight_file": rel(model_file),
        "weight_file_exists": True,
        "safetensor_param_count": total,
        "adapter_tensor_param_count": adapter,
        "tensor_count": tensor_count,
        "adapter_key_examples": ";".join(adapter_keys),
    }


def inspect_one(rec: dict[str, Any]) -> dict[str, Any]:
    path = pathlib.Path(rec["path"])
    cfg = read_json(path / "config.json") or {}
    dyn_files = sorted(p.name for p in path.glob("*modeling*.py"))
    out: dict[str, Any] = {
        "family": rec["family"],
        "role": rec["role"],
        "checkpoint_path": rel(path),
        "exists": path.exists(),
        "config_exists": (path / "config.json").exists(),
        "architectures": ";".join(str(x) for x in cfg.get("architectures", [])) if isinstance(cfg.get("architectures"), list) else str(cfg.get("architectures", "")),
        "model_type": cfg.get("model_type"),
        "auto_map_present": bool(cfg.get("auto_map")),
        "auto_map": json.dumps(cfg.get("auto_map", {}), sort_keys=True),
        "adapter_enabled": cfg.get("adapter_enabled"),
        "adapter_scale": cfg.get("adapter_scale"),
        "adapter_bottleneck": cfg.get("adapter_bottleneck"),
        "dynamic_modeling_files": ";".join(dyn_files),
        "has_dynamic_modeling_file": bool(dyn_files),
        "lineage_reading": rec["lineage_reading"],
    }
    out.update(count_safetensor_params(path))
    run_dir = path.parent.parent
    metrics = read_json(run_dir / "scientific_metrics.json") or {}
    if metrics:
        out["scientific_metrics_parameter_count"] = metrics.get("parameter_count")
        out["scientific_metrics_word_exposure"] = metrics.get("word_exposure")
        out["scientific_metrics_loss_last"] = metrics.get("loss_last")
    else:
        out["scientific_metrics_parameter_count"] = None
        out["scientific_metrics_word_exposure"] = None
        out["scientific_metrics_loss_last"] = None
    out["requires_trusted_remote_code_for_deployed_object"] = bool(out["auto_map_present"] or out["has_dynamic_modeling_file"] or str(out["architectures"]).startswith("Adapter"))
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    pref = [
        "family", "role", "checkpoint_path", "exists", "architectures", "model_type",
        "auto_map_present", "adapter_enabled", "adapter_scale", "dynamic_modeling_files",
        "safetensor_param_count", "adapter_tensor_param_count", "tensor_count",
        "scientific_metrics_parameter_count", "requires_trusted_remote_code_for_deployed_object",
        "lineage_reading",
    ]
    fields = [f for f in pref if f in fields] + [f for f in fields if f not in pref]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [inspect_one(r) for r in REPRESENTATIVES]
    summary = {
        "status": "CHECKPOINT_FAMILY_IDENTITY_DONE",
        "created_utc": now(),
        "representatives": len(rows),
        "stock_like_count": sum(1 for r in rows if not r["requires_trusted_remote_code_for_deployed_object"]),
        "trusted_required_count": sum(1 for r in rows if r["requires_trusted_remote_code_for_deployed_object"]),
        "csv": rel(OUT / "checkpoint_family_identity.csv"),
        "jsonl": rel(OUT / "checkpoint_family_identity.jsonl"),
        "reading": "Representative REPRESENTATION_FRONTIER_STUDIES/COMPACT_EXPERIENCE/RoBERTa/causal-GPT relation checkpoints have stock configs and no dynamic modeling file; adapter-scaled compact-view-reinvest, state-update, and dose checkpoints carry AdapterDebertaV2/auto_map metadata and adapter tensors, so local scoring of those checkpoints must use the trusted custom loader.",
    }
    write_csv(OUT / "checkpoint_family_identity.csv", rows)
    with (OUT / "checkpoint_family_identity.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "summary.json").write_text(json.dumps({**summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
