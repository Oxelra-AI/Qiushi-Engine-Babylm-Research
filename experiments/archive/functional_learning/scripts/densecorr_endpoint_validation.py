#!/usr/bin/env python3
"""research: validate the completed dense-corruption preservation endpoint.

This executable measurement makes the delivered research endpoint a trustworthy
research object before any mechanism readout or export-path use. It verifies the
run provenance, schedule, support totals, exposure accounting, RNG records, and
private-adapter-only parameter movement against the coherent86 parent.
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

import corrected_bridge_trainer as bridge  # noqa: E402

DEFAULT_RUN = _public_path('experiments/archive/functional_learning/data/densecorruption_preservation_lambda1_full80')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/densecorr_endpoint_validation')
EXPECTED_SUPPORT = {
    "qwen_rows": 3831,
    "dense_masked_tokens": 176607,
    "sparse_label_targets": 28590,
    "dense_nonlabel_targets": 148017,
    "candidate_groups": 132283,
    "dense_masked_groups": 132283,
    "sparse_label_groups": 21479,
    "zero_dense_nonlabel_rows": 0,
    "examples": 3831,
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


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def close(a: Any, b: Any, tol: float = 1e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def all_bool(xs: Iterable[bool]) -> bool:
    return all(bool(x) for x in xs)


def summarize_update_log(rows: List[Dict[str, Any]], summary: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    update_seq = [int(r.get("update", -1)) for r in rows]
    schedule_seq = [int(r.get("schedule_idx", -1)) for r in rows]
    lr_checks = []
    for r in rows:
        exp_lr = bridge.lr_at_update(int(r["schedule_idx"]), int(config.get("schedule_total", 455)), int(config.get("warmup", 10)), float(config.get("lr_peak", config.get("lr", 5e-5))))
        lr_checks.append(close(r.get("lr"), exp_lr, 2e-15))
    sums = {
        "words": sum(int(r.get("words", 0)) for r in rows),
        "pres_words": sum(int(r.get("pres_words", 0)) for r in rows),
        "focus_target_tokens": sum(int(r.get("focus_target_tokens", 0)) for r in rows),
        "ordinary_target_tokens": sum(int(r.get("ordinary_target_tokens", 0)) for r in rows),
        "pres_targets": sum(int(r.get("pres_targets", 0)) for r in rows),
        "qwen_rows": sum(int(r.get("qwen_rows", 0)) for r in rows),
    }
    support_sum: Dict[str, int] = {}
    for r in rows:
        for k, v in (r.get("pres_support_counts") or {}).items():
            support_sum[k] = support_sum.get(k, 0) + int(v)
    checks = {
        "n_update_logs_80": len(rows) == 80,
        "updates_1_to_80": update_seq == list(range(1, 81)),
        "schedule_idx_101_to_180": schedule_seq == list(range(101, 181)),
        "all_lr_match_schedule": all_bool(lr_checks),
        "sum_words_match_summary": sums["words"] == int(summary.get("total_words_consumed", -1)),
        "sum_pres_words_match_summary": sums["pres_words"] == int(summary.get("total_preservation_words_counted", -1)),
        "sum_focus_targets_match_summary": sums["focus_target_tokens"] == int(summary.get("total_focus_targets", -1)),
        "sum_ordinary_targets_match_summary": sums["ordinary_target_tokens"] == int(summary.get("total_ordinary_targets", -1)),
        "sum_pres_targets_match_summary": sums["pres_targets"] == int(summary.get("total_preservation_targets_prepared", -1)),
        "support_counts_match_summary": support_sum == {k: int(v) for k, v in (summary.get("total_preservation_support_counts") or {}).items()},
        "support_counts_match_expected": support_sum == EXPECTED_SUPPORT,
        "final_update_is_80": bool(summary.get("final_update", {}).get("update") == 80),
        "final_lr_match_schedule": close(summary.get("final_update", {}).get("lr"), bridge.lr_at_update(180, 455, 10, 5e-5), 2e-15),
        "final_kl_matches_recorded": close(summary.get("final_update", {}).get("pres_kl_mean"), 0.00803593651754727, 1e-12),
    }
    return {"update_sequence_head": update_seq[:5], "update_sequence_tail": update_seq[-5:], "schedule_sequence_head": schedule_seq[:5], "schedule_sequence_tail": schedule_seq[-5:], "sums_from_log": sums, "support_sums_from_log": support_sum, "checks": checks}


def summarize_rng(rng_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_records": len(rng_records),
        "recorded_updates": [int(r.get("update", -1)) for r in rng_records],
        "all_restored_after_preservation": all_bool(r.get("restored_after_preservation") for r in rng_records),
        "all_forked_preservation_forward_rng": all_bool(r.get("forked_preservation_forward_rng") for r in rng_records),
        "after_restore_equals_after_acquisition_cuda_digest": all_bool((r.get("after_preservation_restore") or {}).get("torch_cuda_all_sha256") == (r.get("after_acquisition") or {}).get("torch_cuda_all_sha256") for r in rng_records),
        "after_restore_equals_before_acquisition_cpu_digest": all_bool((r.get("after_preservation_restore") or {}).get("torch_cpu_sha256") == (r.get("before_acquisition") or {}).get("torch_cpu_sha256") for r in rng_records),
    }


def tensor_delta(parent_path: pathlib.Path, endpoint_path: pathlib.Path) -> Dict[str, Any]:
    parent = load_file(str(parent_path / "model.safetensors"))
    endpoint = load_file(str(endpoint_path / "model.safetensors"))
    pkeys = set(parent.keys())
    ekeys = set(endpoint.keys())
    shared = sorted(pkeys & ekeys)
    changed: List[str] = []
    unchanged: List[str] = []
    max_priv = 0.0
    max_non = 0.0
    priv_sumsq = 0.0
    priv_n = 0
    non_sumsq = 0.0
    non_n = 0
    for k in shared:
        d = (endpoint[k].float() - parent[k].float()).abs()
        mx = float(d.max().item()) if d.numel() else 0.0
        if mx > 0.0:
            changed.append(k)
        else:
            unchanged.append(k)
        raw = (endpoint[k].float() - parent[k].float()).reshape(-1)
        if ".private_adapter." in k:
            max_priv = max(max_priv, mx)
            priv_sumsq += float((raw * raw).sum().item())
            priv_n += int(raw.numel())
        else:
            max_non = max(max_non, mx)
            non_sumsq += float((raw * raw).sum().item())
            non_n += int(raw.numel())
    changed_private = [k for k in changed if ".private_adapter." in k]
    changed_non = [k for k in changed if ".private_adapter." not in k]
    return {
        "parent_model_sha256": sha256_file(parent_path / "model.safetensors"),
        "endpoint_model_sha256": sha256_file(endpoint_path / "model.safetensors"),
        "parent_key_count": len(pkeys),
        "endpoint_key_count": len(ekeys),
        "missing_in_endpoint": sorted(pkeys - ekeys),
        "extra_in_endpoint": sorted(ekeys - pkeys),
        "changed_tensor_count": len(changed),
        "changed_private_tensor_count": len(changed_private),
        "changed_non_private_tensor_count": len(changed_non),
        "unchanged_tensor_count": len(unchanged),
        "changed_private_head": changed_private[:10],
        "changed_non_private_head": changed_non[:10],
        "max_abs_delta_private": max_priv,
        "max_abs_delta_non_private": max_non,
        "rms_delta_private": math.sqrt(priv_sumsq / priv_n) if priv_n else None,
        "rms_delta_non_private": math.sqrt(non_sumsq / non_n) if non_n else None,
        "private_only_changed": len(changed_non) == 0 and len(changed_private) == 48,
    }


def identity_checks(summary: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    model_identity = summary.get("model_identity") or config.get("student_identity") or {}
    teacher_identity = summary.get("teacher_identity") or config.get("teacher_identity") or {}
    exp = summary.get("exposure_accounting") or {}
    support = summary.get("total_preservation_support_counts") or {}
    return {
        "method_matches_densecorr_policy": summary.get("method") == "ms_acquisition_plus_densecorruption_nonlabel_parent_distillation_clean_rng" and config.get("method") == "ms_acquisition_plus_densecorruption_nonlabel_parent_distillation_clean_rng",
        "completed_updates_80": int(summary.get("completed_updates", -1)) == 80,
        "train_seed_62064": int(config.get("train_seed", -1)) == 62064,
        "private_optimizer_48_tensors": int((config.get("optimizer") or {}).get("trainable_tensors", -1)) == 48,
        "private_optimizer_995584_params": int((config.get("optimizer") or {}).get("trainable_params", -1)) == 995584,
        "student_private_scale_075": (model_identity.get("executed_private_scales") == [0.75] * 8),
        "teacher_private_scale_075": (teacher_identity.get("executed_private_scales") == [0.75] * 8),
        "student_teacher_intended_class": model_identity.get("class") == "FrozenSlowPrivateDebertaV2ForMaskedLM" and teacher_identity.get("class") == "FrozenSlowPrivateDebertaV2ForMaskedLM",
        "lambda_pres_1": close(summary.get("lambda_pres"), 1.0),
        "focus_lambda_015": close(summary.get("focus_lambda"), 0.15),
        "kl_direction_teacher_to_student": summary.get("implemented_kl_direction") == "KL(teacher || student)" and config.get("implemented_kl_direction") == "KL(teacher || student)",
        "pres_student_eval_mode": summary.get("pres_student_mode") == "eval",
        "prefix_words_3162742": int(summary.get("total_words_consumed", -1)) == 3162742,
        "pres_words_517332": int(summary.get("total_preservation_words_counted", -1)) == 517332,
        "endpoint_exposure_89685369": int(exp.get("endpoint_exposure_conservative", -1)) == 89685369,
        "focus_targets_28590": int(summary.get("total_focus_targets", -1)) == 28590,
        "ordinary_targets_592858": int(summary.get("total_ordinary_targets", -1)) == 592858,
        "preservation_targets_148017": int(summary.get("total_preservation_targets_prepared", -1)) == 148017,
        "support_counts_expected": {k: int(v) for k, v in support.items()} == EXPECTED_SUPPORT,
        "checkpoint_0080_exists": (pathlib.Path(summary.get("checkpoints", [{}])[-1].get("path", "")) / "model.safetensors").exists(),
    }


def write_md(path: pathlib.Path, result: Dict[str, Any]) -> None:
    checks = result["checks"]
    lines = [
        "# research dense-corruption endpoint validation",
        "",
        f"Created: `{result['created_utc']}`",
        "",
        f"All validation measurements pass: `{result['all_valid']}`.",
        "",
        "## Endpoint",
        "",
        f"- run directory: `{result['run_dir']}`",
        f"- endpoint: `{result['endpoint']}`",
        f"- endpoint model SHA256: `{result['tensor_delta']['endpoint_model_sha256']}`",
        f"- parent model SHA256: `{result['tensor_delta']['parent_model_sha256']}`",
        "",
        "## Run and support measurements",
        "",
        f"- completed updates: `{result['summary_core']['completed_updates']}`",
        f"- schedule indices: `{result['log_summary']['schedule_sequence_head']} ... {result['log_summary']['schedule_sequence_tail']}`",
        f"- acquisition words: `{result['summary_core']['total_words_consumed']}`",
        f"- counted preservation words: `{result['summary_core']['total_preservation_words_counted']}`",
        f"- conservative endpoint exposure: `{result['summary_core']['endpoint_exposure_conservative']}`",
        f"- preservation targets: `{result['summary_core']['total_preservation_targets_prepared']}`",
        f"- final preservation KL: `{result['summary_core']['final_preservation_kl']}`",
        "",
        "## Parameter movement",
        "",
        f"- changed tensors: `{result['tensor_delta']['changed_tensor_count']}`",
        f"- changed private tensors: `{result['tensor_delta']['changed_private_tensor_count']}`",
        f"- changed non-private tensors: `{result['tensor_delta']['changed_non_private_tensor_count']}`",
        f"- max private delta: `{result['tensor_delta']['max_abs_delta_private']}`",
        f"- RMS private delta: `{result['tensor_delta']['rms_delta_private']}`",
        "",
        "## Measurements",
        "",
        "| measurement | pass |",
        "|---|---:|",
    ]
    for group, vals in checks.items():
        for k, v in vals.items():
            lines.append(f"| {group}.{k} | `{bool(v)}` |")
    lines += [
        "",
        "## Scientific use",
        "",
        result["scientific_use"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=pathlib.Path, default=DEFAULT_RUN)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = pathlib.Path(args.run_dir)
    summary = load_json(run_dir / "train_summary.json")
    config = load_json(run_dir / "train_config.json")
    update_log = load_jsonl(run_dir / "update_log.jsonl")
    rng_records = load_json(run_dir / "rng_records.json")
    endpoint = run_dir / "checkpoints/update_0080"

    id_checks = identity_checks(summary, config)
    log_summary = summarize_update_log(update_log, summary, config)
    rng_summary = summarize_rng(rng_records)
    rng_checks = {
        "three_rng_records_present": rng_summary["n_records"] == 3,
        "updates_1_2_3_recorded": rng_summary["recorded_updates"] == [1, 2, 3],
        "all_restored_after_preservation": rng_summary["all_restored_after_preservation"],
        "all_forked_preservation_forward_rng": rng_summary["all_forked_preservation_forward_rng"],
        "after_restore_equals_after_acquisition_cuda_digest": rng_summary["after_restore_equals_after_acquisition_cuda_digest"],
        "after_restore_equals_before_acquisition_cpu_digest": rng_summary["after_restore_equals_before_acquisition_cpu_digest"],
    }
    tdelta = tensor_delta(bridge.PARENT_PATH, endpoint)
    tensor_checks = {
        "parent_and_endpoint_same_keyset": not tdelta["missing_in_endpoint"] and not tdelta["extra_in_endpoint"],
        "exactly_48_changed_private_tensors": tdelta["changed_private_tensor_count"] == 48,
        "no_non_private_tensor_movement": tdelta["changed_non_private_tensor_count"] == 0,
        "private_only_changed": bool(tdelta["private_only_changed"]),
    }
    checks = {
        "identity": id_checks,
        "update_log": log_summary["checks"],
        "rng": rng_checks,
        "tensor_delta": tensor_checks,
    }
    all_valid = all_bool(v for group in checks.values() for v in group.values())
    result = {
        "status": "DENSECORR_ENDPOINT_VALIDATION",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/functional_learning/scripts/densecorr_endpoint_validation.py')),
        "run_dir": rel(run_dir),
        "endpoint": rel(endpoint),
        "all_valid": all_valid,
        "summary_core": {
            "completed_updates": summary.get("completed_updates"),
            "total_words_consumed": summary.get("total_words_consumed"),
            "total_preservation_words_counted": summary.get("total_preservation_words_counted"),
            "endpoint_exposure_conservative": (summary.get("exposure_accounting") or {}).get("endpoint_exposure_conservative"),
            "total_focus_targets": summary.get("total_focus_targets"),
            "total_ordinary_targets": summary.get("total_ordinary_targets"),
            "total_preservation_targets_prepared": summary.get("total_preservation_targets_prepared"),
            "final_preservation_kl": (summary.get("final_update") or {}).get("pres_kl_mean"),
            "method": summary.get("method"),
        },
        "log_summary": {k: v for k, v in log_summary.items() if k != "checks"},
        "rng_summary": rng_summary,
        "tensor_delta": tdelta,
        "checks": checks,
        "scientific_use": "This endpoint is validated as a private-adapter-only dense-corruption preservation policy with matched schedule and recorded support totals. Use it for bounded acquisition-retention readouts; it is not promoted as a release candidate by this validation alone.",
    }
    out_json = args.out_dir / "densecorr_endpoint_validation.json"
    out_md = args.out_dir / "densecorr_endpoint_validation.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_md(out_md, result)
    print(json.dumps({"status": result["status"], "all_valid": all_valid, "endpoint_sha256": tdelta.get("endpoint_model_sha256"), "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2, ensure_ascii=False), flush=True)
    if not all_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
