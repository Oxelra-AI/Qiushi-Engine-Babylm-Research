#!/usr/bin/env python3
"""research: establish the ordinary-continuation endpoint identity.

This script does not train a new model.  It inspects the tested research
`inherited_wwm` run and verifies that it is the ordinary continuation needed for
the Stage-III attribution comparison: same parent, same unchanged-Qwen prefix,
same 80-update word schedule as exact (M,S), the evaluated learning-rate
trajectory, private-adapter-only optimization, and ordinary 15% WWM on Qwen and
non-Qwen rows rather than the dense-mask/sparse-label patch.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, List

from safetensors.torch import load_file

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_stream_train_weighted as s64  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402

ORD_DIR = _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/inherited_wwm')
MS_DIR = _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted')
PARENT_DIR = bridge.PARENT_PATH
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/ordinary_control_identity')

EXPECTED = {
    "train_seed": 62064,
    "max_updates": 80,
    "words_per_update": 39533,
    "schedule_total": 455,
    "schedule_offset": 101,
    "warmup": 10,
    "lr_peak": 5e-5,
    "mask_prob": 0.15,
    "private_scale": 0.75,
}


def rel(path: pathlib.Path | str | None) -> str | None:
    if path is None:
        return None
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def almost(a: Any, b: Any, tol: float = 1e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return a == b


def same_prefix(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    keys = [
        "tail_jsonl", "requested_max_updates", "words_per_update", "prefix_rows",
        "prefix_words", "target_prefix_words", "qwen_rows", "qwen_pair_segments",
        "compact_modified_rows", "compact_modified_pair_occurrences", "topup_rows",
    ]
    return all(a.get(k) == b.get(k) for k in keys) and a.get("source_rows") == b.get("source_rows")


def tensor_delta_summary(parent_path: pathlib.Path, endpoint_path: pathlib.Path) -> Dict[str, Any]:
    parent = load_file(str(parent_path / "model.safetensors"), device="cpu")
    endpoint = load_file(str(endpoint_path / "model.safetensors"), device="cpu")
    pkeys = set(parent)
    ekeys = set(endpoint)
    missing_in_endpoint = sorted(pkeys - ekeys)
    extra_in_endpoint = sorted(ekeys - pkeys)
    changed: List[str] = []
    unchanged: List[str] = []
    max_abs_by_kind = {"private": 0.0, "non_private": 0.0}
    l2_by_kind = {"private": 0.0, "non_private": 0.0}
    n_by_kind = {"private": 0, "non_private": 0}
    for k in sorted(pkeys & ekeys):
        a = parent[k]
        b = endpoint[k]
        if a.shape != b.shape:
            changed.append(k)
            continue
        d = (a - b).float()
        mx = float(d.abs().max()) if d.numel() else 0.0
        kind = "private" if ".private_adapter." in k else "non_private"
        max_abs_by_kind[kind] = max(max_abs_by_kind[kind], mx)
        l2_by_kind[kind] += float((d * d).sum())
        n_by_kind[kind] += int(d.numel())
        if mx == 0.0:
            unchanged.append(k)
        else:
            changed.append(k)
    changed_private = [k for k in changed if ".private_adapter." in k]
    changed_non_private = [k for k in changed if ".private_adapter." not in k]
    return {
        "parent_model_sha256": sha256_file(parent_path / "model.safetensors"),
        "endpoint_model_sha256": sha256_file(endpoint_path / "model.safetensors"),
        "parent_key_count": len(pkeys),
        "endpoint_key_count": len(ekeys),
        "missing_in_endpoint": missing_in_endpoint,
        "extra_in_endpoint": extra_in_endpoint,
        "changed_tensor_count": len(changed),
        "changed_private_tensor_count": len(changed_private),
        "changed_non_private_tensor_count": len(changed_non_private),
        "unchanged_tensor_count": len(unchanged),
        "changed_private_head": changed_private[:8],
        "changed_non_private_head": changed_non_private[:8],
        "max_abs_delta_private": max_abs_by_kind["private"],
        "max_abs_delta_non_private": max_abs_by_kind["non_private"],
        "rms_delta_private": math.sqrt(l2_by_kind["private"] / max(1, n_by_kind["private"])),
        "rms_delta_non_private": math.sqrt(l2_by_kind["non_private"] / max(1, n_by_kind["non_private"])),
        "private_only_changed": len(missing_in_endpoint) == 0 and len(extra_in_endpoint) == 0 and len(changed_non_private) == 0 and len(changed_private) == 48,
    }


def first_macro_preview(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rows, prefix_info = s64.load_prefix(pathlib.Path(cfg["tail_jsonl"]), int(cfg["max_updates"]), int(cfg["words_per_update"]))
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(PARENT_DIR), local_files_only=True, use_fast=True)
    wgb = bridge.WordGroupBuilder(tokenizer)
    macro_rows = []
    words = 0
    for r in rows:
        if words >= int(cfg["words_per_update"]):
            break
        macro_rows.append(r)
        words += int(r.get("words", s64.wc(r.get("text", ""))))
    examples, prep = s64.prepare_macro(
        macro_rows, tokenizer, int(cfg["seq_length"]), wgb,
        "inherited_wwm", int(cfg["train_seed"]), float(cfg["mask_prob"]),
        float(cfg["focus_prob"]), int(cfg["max_focus_groups_per_row"]),
    )
    focus_positions = sum(int((ex["labels"] != -100).sum()) for ex in examples if ex.get("component") == "focus")
    ordinary_positions = sum(int((ex["labels"] != -100).sum()) for ex in examples if ex.get("component") == "ordinary")
    return {
        "prefix_info_matches_train_config": same_prefix(prefix_info, cfg["prefix_info"]),
        "macro_rows": len(macro_rows),
        "macro_words": words,
        "prepared_stats": prep,
        "focus_label_positions_from_examples": focus_positions,
        "ordinary_label_positions_from_examples": ordinary_positions,
        "actual_short_execution": "called research prepare_macro with objective='inherited_wwm' on the first macro-batch",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ord_cfg = load_json(_public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/inherited_wwm/train_config.json'))
    ord_sum = load_json(_public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/inherited_wwm/train_summary.json'))
    ms_cfg = load_json(_public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/train_config.json'))
    ms_sum = load_json(_public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/train_summary.json'))
    ord_logs = load_jsonl(_public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/inherited_wwm/update_log.jsonl'))
    ms_logs = load_jsonl(_public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/update_log.jsonl'))

    endpoint = _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/inherited_wwm/checkpoints/update_0080')
    tensor_deltas = tensor_delta_summary(PARENT_DIR, endpoint)
    macro_preview = first_macro_preview(ord_cfg)

    parameter_checks = {k: ord_cfg.get(k) == v or almost(ord_cfg.get(k), v) for k, v in EXPECTED.items() if k != "private_scale"}
    # research train_config stores the executed private scale inside the model identity
    # rather than as a top-level argument; checkpoint bridge_metadata has the top-level
    # private_scale.  Both are checked so the identity is not lost through a display-key
    # mismatch.
    endpoint_meta = load_json(endpoint / "bridge_metadata.json")
    parameter_checks["private_scale"] = (
        all(almost(x, EXPECTED["private_scale"]) for x in ord_cfg.get("model_identity", {}).get("executed_private_scales", []))
        and almost(endpoint_meta.get("private_scale"), EXPECTED["private_scale"])
    )
    matched_to_ms = {
        "same_tail_and_prefix": same_prefix(ord_cfg.get("prefix_info", {}), ms_cfg.get("prefix_info", {})),
        "same_train_seed": ord_cfg.get("train_seed") == ms_cfg.get("train_seed"),
        "same_updates": ord_cfg.get("max_updates") == ms_cfg.get("max_updates") == ord_sum.get("completed_updates") == ms_sum.get("completed_updates"),
        "same_words_per_update": ord_cfg.get("words_per_update") == ms_cfg.get("words_per_update"),
        "same_schedule_total": ord_cfg.get("schedule_total") == ms_cfg.get("schedule_total"),
        "same_schedule_offset": ord_cfg.get("schedule_offset") == ms_cfg.get("schedule_offset"),
        "same_warmup": ord_cfg.get("warmup") == ms_cfg.get("warmup"),
        "same_lr_peak": almost(ord_cfg.get("lr_peak"), ms_cfg.get("lr_peak")),
        "same_mask_prob": almost(ord_cfg.get("mask_prob"), ms_cfg.get("mask_prob")),
        "same_private_scale": almost(ord_cfg.get("private_scale"), ms_cfg.get("private_scale")),
        "same_final_schedule_idx": ord_sum.get("final_update", {}).get("schedule_idx") == ms_sum.get("final_update", {}).get("schedule_idx"),
        "same_final_lr": almost(ord_sum.get("final_update", {}).get("lr"), ms_sum.get("final_update", {}).get("lr"), 1e-16),
    }

    log_checks = {
        "n_update_logs": len(ord_logs),
        "all_updates_present": len(ord_logs) == 80 and [r.get("update") for r in ord_logs] == list(range(1, 81)),
        "schedule_idx_sequence_101_to_180": [r.get("schedule_idx") for r in ord_logs] == list(range(101, 181)),
        "all_qwen_rows_use_wwm": all(int(r.get("qwen_focus_rows", -1)) == 0 and int(r.get("qwen_wwm_rows", -1)) == int(r.get("qwen_rows", -2)) for r in ord_logs),
        "all_focus_targets_zero": all(int(r.get("focus_target_tokens", -1)) == 0 for r in ord_logs),
        "all_targets_ordinary": all(int(r.get("ordinary_target_tokens", -1)) == int(r.get("targets", -2)) for r in ord_logs),
        "all_loss_pooled_token_mean": all(r.get("loss_mode") == "pooled_token_mean" for r in ord_logs),
        "total_targets_from_logs": sum(int(r.get("targets", 0)) for r in ord_logs),
        "summary_total_targets": ord_sum.get("total_targets"),
        "total_targets_match_summary": sum(int(r.get("targets", 0)) for r in ord_logs) == int(ord_sum.get("total_targets", -1)),
        "all_logged_lr_matches_schedule_function": all(almost(r.get("lr"), s64.lr_at_update(int(r["schedule_idx"]), int(ord_cfg["schedule_total"]), int(ord_cfg["warmup"]), float(ord_cfg["lr_peak"])), 1e-16) for r in ord_logs),
    }

    optimizer_checks = {
        "trainable_tensors": ord_cfg.get("optimizer", {}).get("trainable_tensors"),
        "trainable_params": ord_cfg.get("optimizer", {}).get("trainable_params"),
        "private_adapter_tensors": ord_cfg.get("model_identity", {}).get("private_adapter_tensors"),
        "private_adapter_params": ord_cfg.get("model_identity", {}).get("private_adapter_params"),
        "private_only_optimizer_recorded": ord_cfg.get("optimizer", {}).get("trainable_tensors") == 48 and ord_cfg.get("optimizer", {}).get("trainable_params") == 995584,
        "private_only_weight_delta_vs_parent": tensor_deltas["private_only_changed"],
    }

    complete_identity = bool(
        all(parameter_checks.values())
        and all(matched_to_ms.values())
        and log_checks["all_updates_present"]
        and log_checks["schedule_idx_sequence_101_to_180"]
        and log_checks["all_qwen_rows_use_wwm"]
        and log_checks["all_focus_targets_zero"]
        and log_checks["all_targets_ordinary"]
        and log_checks["all_loss_pooled_token_mean"]
        and log_checks["total_targets_match_summary"]
        and log_checks["all_logged_lr_matches_schedule_function"]
        and optimizer_checks["private_only_optimizer_recorded"]
        and optimizer_checks["private_only_weight_delta_vs_parent"]
        and macro_preview["prefix_info_matches_train_config"]
        and int(macro_preview["prepared_stats"].get("qwen_focus_rows", -1)) == 0
        and int(macro_preview["prepared_stats"].get("qwen_wwm_rows", -1)) == int(macro_preview["prepared_stats"].get("qwen_rows", -2))
    )

    result = {
        "status": "ORDINARY_CONTROL_IDENTITY",
        "created_utc": now(),
        "ordinary_endpoint": rel(endpoint),
        "matched_reference_ms_endpoint": rel(_public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/checkpoints/update_0080')),
        "parent_endpoint": rel(PARENT_DIR),
        "complete_identity": complete_identity,
        "parameter_checks_vs_evaluated_ms_config": parameter_checks,
        "matched_to_exact_ms_training_config": matched_to_ms,
        "log_checks": log_checks,
        "optimizer_and_weight_delta_checks": optimizer_checks,
        "tensor_delta_summary": tensor_deltas,
        "macro_short_execution_preview": macro_preview,
        "ordinary_summary_core": {
            "completed_updates": ord_sum.get("completed_updates"),
            "total_words_consumed": ord_sum.get("total_words_consumed"),
            "endpoint_exposure_conservative": 86_005_295 + int(ord_sum.get("total_words_consumed", 0)),
            "total_targets": ord_sum.get("total_targets"),
            "total_focus_targets": ord_sum.get("total_focus_targets"),
            "total_ordinary_targets": ord_sum.get("total_ordinary_targets"),
            "final_update": ord_sum.get("final_update"),
        },
        "scientific_reading": "The existing research inherited_wwm endpoint is the ordinary-continuation control for seed62064: standard row-keyed WWM on all rows including Qwen rows, same prefix and learning-rate trajectory as evaluated exact (M,S), and private-adapter-only movement. It still needs complete compatible evaluation before entering the same-coordinate model table.",
    }
    out_json = args.out_dir / "ordinary_control_identity.json"
    out_md = args.out_dir / "ordinary_control_identity.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = [
        "# research ordinary-continuation endpoint identity\n\n",
        f"Created: `{result['created_utc']}`\n\n",
        f"Complete identity: `{complete_identity}`\n\n",
        f"Ordinary endpoint: `{rel(endpoint)}`\n\n",
        "## Matching to the evaluated exact (M,S) run\n\n",
        json.dumps(matched_to_ms, indent=2, ensure_ascii=False) + "\n\n",
        "## Ordinary WWM and optimizer evidence\n\n",
        json.dumps({"log_checks": log_checks, "optimizer": optimizer_checks}, indent=2, ensure_ascii=False) + "\n\n",
        "## Weight movement\n\n",
        json.dumps(tensor_deltas, indent=2, ensure_ascii=False) + "\n\n",
        "## First macro short execution\n\n",
        json.dumps(macro_preview, indent=2, ensure_ascii=False, default=str) + "\n\n",
        result["scientific_reading"] + "\n",
    ]
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "complete_identity": complete_identity,
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "ordinary_endpoint": rel(endpoint),
        "ordinary_sha256": tensor_deltas["endpoint_model_sha256"],
    }, indent=2, ensure_ascii=False), flush=True)
    if not complete_identity:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
