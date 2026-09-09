#!/usr/bin/env python3
"""research: fail-closed official collation for truthful alpha0.75 Fast + AoA.

Inputs are produced by two bounded evaluation jobs:
  - alpha075_fast_missing_eval.py: stages full main predictions, links 119
    pre-82M fast files, and generates the 14 chck_90M/chck_100M fast files.
  - alpha075_aoa_minctx0_local.py: computes official min_context=0 AoA
    for the truthful 19-checkpoint alpha0.75 tree.

This wrapper trains nothing. It links the AoA score/surprisal into the fast result
layout, runs the unmodified official collator with --fast --track strict-small,
and then verifies that the collated file contains full predictions, finite AoA,
and all 7 fast arrays populated at all 19 Strict-Small revisions.
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
import shutil
import subprocess
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/representation_and_objectives"
STRICT = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict"
DEFAULT_FAST_OUT = WORKSPACE / "data/alpha075_fast_missing_eval"
DEFAULT_AOA_OUT = WORKSPACE / "data/alpha075_aoa_minctx0"
DEFAULT_BASE_OUT = WORKSPACE / "data/alpha075_collated_fast_aoa"

FAST_REVISIONS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]
FAST_KEYS = ["blimp", "blimp_supplement", "ewok", "entity_tracking_filtered", "global_piqa_parallel", "global_piqa_nonparallel", "reading"]
FULL_KEYS = ["blimp", "blimp_supplement", "ewok", "entity_tracking_filtered", "comps", "global_piqa_parallel", "global_piqa_nonparallel", "reading", "aoa", "aoa_surprisals", "glue"]
GLUE_TASKS = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]
FAST_TASK_RELS = {
    "BLiMP": "zero_shot/mlm/blimp/blimp_fast/predictions.json",
    "Supplement": "zero_shot/mlm/blimp/supplement_fast/predictions.json",
    "EWoK": "zero_shot/mlm/ewok/ewok_fast/predictions.json",
    "Entity": "zero_shot/mlm/entity_tracking/entity_tracking_fast/predictions.json",
    "GlobalPIQA_parallel": "zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
    "GlobalPIQA_nonparallel": "zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
    "Reading": "zero_shot/mlm/reading/predictions.json",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        try:
            return str(p.relative_to(USER_ROOT))
        except Exception:
            return str(p)


def sha256_file(p: pathlib.Path) -> str | None:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_link(src: pathlib.Path, dst: pathlib.Path) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
    rec: dict[str, Any] = {"mode": "symlink", "src": rel(src), "dst": rel(dst), "is_dir": src.is_dir()}
    if src.is_file():
        rec.update({"size_bytes": src.stat().st_size, "sha256": sha256_file(src)})
    return rec


def copy_or_link_tree(src: pathlib.Path, dst: pathlib.Path, force: bool) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        if force:
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        else:
            return {"mode": "reuse_existing", "src": rel(src), "dst": rel(dst)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Copy the directory structure while preserving file symlinks. Do not symlink
    # the root itself: AoA links are inserted after staging, and root-symlinking
    # would write through into the source fast-output tree under a different
    # writable target.
    shutil.copytree(src, dst, symlinks=True)
    return {"mode": "copytree_symlinks_preserved", "src": rel(src), "dst": rel(dst), "is_dir": True}


def inspect_prediction(path: pathlib.Path) -> dict[str, Any]:
    rec: dict[str, Any] = {"path": rel(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None, "sha256": sha256_file(path) if path.exists() else None}
    if path.exists():
        try:
            d = read_json(path)
            rec["top_type"] = type(d).__name__
            if isinstance(d, dict):
                rec["top_keys_count"] = len(d)
                rec["top_keys_sample"] = list(d.keys())[:8]
                rec["prediction_total"] = sum(len(v.get("predictions", [])) for v in d.values() if isinstance(v, dict))
        except Exception as exc:
            rec["read_error"] = repr(exc)
    return rec


def validate_fast_summary(fast_out: pathlib.Path) -> dict[str, Any]:
    summary_path = fast_out / "fast_missing_eval_summary.json"
    if not summary_path.exists():
        return {"ok": False, "error": "missing_fast_summary", "path": rel(summary_path)}
    summary = read_json(summary_path)
    missing = summary.get("missing_prediction_files", [])
    status = summary.get("status")
    expected = summary.get("expected_prediction_files", [])
    return {
        "ok": status == "ALPHA075_FAST_MISSING_EVAL_DONE" and len(missing) == 0 and len(expected) == 133,
        "status": status,
        "summary_path": rel(summary_path),
        "pending_at_start": summary.get("pending_at_start"),
        "task_records_count": summary.get("task_records_count"),
        "missing_prediction_files": len(missing),
        "expected_prediction_files": len(expected),
    }


def validate_aoa_summary(aoa_out: pathlib.Path) -> dict[str, Any]:
    summary_path = aoa_out / "alpha075_aoa_minctx0_summary.json"
    if not summary_path.exists():
        return {"ok": False, "error": "missing_aoa_summary", "path": rel(summary_path)}
    summary = read_json(summary_path)
    return {
        "ok": summary.get("status") == "ALPHA075_AOA_MINCTX0_DONE" and summary.get("num_rows") == 19 * 8005 and summary.get("row_count_values") == [8005] and summary.get("finite_surprisals") is True,
        "status": summary.get("status"),
        "summary_path": rel(summary_path),
        "aoa": summary.get("aoa"),
        "num_rows": summary.get("num_rows"),
        "num_steps": summary.get("num_steps"),
        "row_count_values": summary.get("row_count_values"),
        "finite_surprisals": summary.get("finite_surprisals"),
        "score_path": summary.get("score_path"),
        "surprisal_path": summary.get("surprisal_path"),
    }


def stage_results_tree(base_out: pathlib.Path, fast_out: pathlib.Path, aoa_out: pathlib.Path, force: bool) -> dict[str, Any]:
    src_results = fast_out / "collate_fast/results/hf_model"
    dst_results = base_out / "collate_fast/results/hf_model"
    tree_rec = copy_or_link_tree(src_results, dst_results, force=force)
    aoa_score_src = aoa_out / "collate_fast/results/hf_model/main/zero_shot/mlm/AoA_word/aoa_score.json"
    aoa_surprisal_src = aoa_out / "collate_fast/results/hf_model/main/zero_shot/mlm/AoA_word/surprisal.json"
    aoa_dst_dir = dst_results / "main/zero_shot/mlm/AoA_word"
    aoa_recs = {
        "aoa_score": clean_link(aoa_score_src, aoa_dst_dir / "aoa_score.json"),
        "surprisal": clean_link(aoa_surprisal_src, aoa_dst_dir / "surprisal.json"),
    }
    return {"results_tree": tree_rec, "aoa_links": aoa_recs}


def validate_input_files(base_out: pathlib.Path) -> dict[str, Any]:
    root = base_out / "collate_fast/results/hf_model"
    missing: list[dict[str, Any]] = []
    fast_records: list[dict[str, Any]] = []
    for ckpt in FAST_REVISIONS:
        for task, r in FAST_TASK_RELS.items():
            p = root / ckpt / r
            rec = {"checkpoint": ckpt, "task": task}
            rec.update(inspect_prediction(p))
            fast_records.append(rec)
            if not rec["exists"] or rec.get("read_error"):
                missing.append(rec)
    full_records = []
    full_paths = {
        "BLiMP": "main/zero_shot/mlm/blimp/blimp_filtered/predictions.json",
        "Supplement": "main/zero_shot/mlm/blimp/supplement_filtered/predictions.json",
        "EWoK": "main/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
        "Entity": "main/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json",
        "COMPS": "main/zero_shot/mlm/comps/comps/predictions.json",
        "GlobalPIQA_parallel": "main/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        "GlobalPIQA_nonparallel": "main/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
        "Reading": "main/zero_shot/mlm/reading/predictions.json",
        "AoA_surprisal": "main/zero_shot/mlm/AoA_word/surprisal.json",
        "AoA_score": "main/zero_shot/mlm/AoA_word/aoa_score.json",
    }
    for task in GLUE_TASKS:
        full_paths[f"GLUE_{task}"] = f"main/finetune/{task}/predictions.json"
    for task, r in full_paths.items():
        p = root / r
        rec = {"task": task}
        rec.update(inspect_prediction(p))
        full_records.append(rec)
        if not rec["exists"] or rec.get("read_error"):
            missing.append(rec)
    return {"ok": len(missing) == 0, "missing_or_unreadable": missing, "fast_records_count": len(fast_records), "full_records_count": len(full_records), "fast_records": fast_records, "full_records": full_records}


def make_env(base_out: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    cache = base_out / "runtime_cache"
    for key, path in {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "TMPDIR": cache / "tmp",
        "NLTK_DATA": USER_ROOT / "experiments/archive/initial_model_studies/data/nltk_data",
    }.items():
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        env[key] = str(pathlib.Path(path).resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = str(STRICT.resolve()) + os.pathsep + str((STRICT / "evaluation_pipeline").resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_collate(base_out: pathlib.Path, timeout_sec: int) -> dict[str, Any]:
    results_dir = base_out / "collate_fast/results"
    model_path = base_out / "collate_fast/results/hf_model"
    cmd = [
        sys.executable, "-B", "-m", "evaluation_pipeline.collate_preds",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--results_dir", str(results_dir.resolve()),
        "--revision_name", "main",
        "--fast",
        "--fast_eval_dir", str((base_out / "fast_eval_data").resolve()),
        "--track", "strict-small",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=make_env(base_out), capture_output=True, text=True, timeout=timeout_sec)
    elapsed = time.time() - t0
    log_prefix = base_out / "logs/collate_fast_aoa"
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_log = log_prefix.with_suffix(".stdout.log")
    err_log = log_prefix.with_suffix(".stderr.log")
    out_log.write_text(proc.stdout, encoding="utf-8", errors="replace")
    err_log.write_text(proc.stderr, encoding="utf-8", errors="replace")
    collated = results_dir / "hf_model/all_full_preds_and_fast_scores_mlm.json"
    return {
        "cmd": cmd,
        "cwd": rel(STRICT),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "stdout_log": rel(out_log),
        "stderr_log": rel(err_log),
        "collated_path": rel(collated),
        "collated_exists": collated.exists(),
        "collated_size_bytes": collated.stat().st_size if collated.exists() else None,
        "collated_sha256": sha256_file(collated) if collated.exists() else None,
    }


def inspect_collated(collated: pathlib.Path) -> dict[str, Any]:
    data = read_json(collated)
    errors: list[str] = []
    out: dict[str, Any] = {"top_keys": sorted(data.keys()), "errors": errors}
    for key in FULL_KEYS:
        if key not in data:
            errors.append(f"missing_full_key_{key}")
            continue
        if data[key] is None:
            errors.append(f"null_full_key_{key}")
    aoa = data.get("aoa")
    if not isinstance(aoa, dict) or "aoa" not in aoa:
        errors.append("aoa_score_missing_or_malformed")
    else:
        try:
            if not (isinstance(float(aoa["aoa"]), float)):
                errors.append("aoa_not_floatable")
        except Exception:
            errors.append("aoa_not_floatable")
    surprisal = data.get("aoa_surprisals")
    if not isinstance(surprisal, dict) or not isinstance(surprisal.get("results"), list):
        errors.append("aoa_surprisals_missing_results")
    else:
        counts: dict[str, int] = {}
        finite = True
        for r in surprisal["results"]:
            counts[str(r.get("step"))] = counts.get(str(r.get("step")), 0) + 1
            try:
                if not math.isfinite(float(r.get("surprisal"))):
                    finite = False
            except Exception:
                finite = False
        out["aoa_step_counts"] = counts
        out["aoa_row_count_values"] = sorted(set(counts.values()))
        out["aoa_total_rows"] = len(surprisal["results"])
        out["aoa_finite"] = finite
        if sorted(counts) != sorted(FAST_REVISIONS) or sorted(set(counts.values())) != [8005] or len(surprisal["results"]) != 19 * 8005 or not finite:
            errors.append("aoa_surprisal_shape_or_finite_check_failed")
    fast = data.get("fast_eval_results")
    out["fast_eval_results_present"] = isinstance(fast, dict)
    if not isinstance(fast, dict):
        errors.append("fast_eval_results_missing")
    else:
        for key in FAST_KEYS:
            values = fast.get(key)
            if not isinstance(values, list):
                errors.append(f"fast_key_{key}_missing_or_not_list")
                continue
            non_null = [i for i, v in enumerate(values) if v is not None]
            out[f"fast_{key}"] = {"length": len(values), "non_null_count": len(non_null), "non_null_indices": non_null, "first": values[0] if values else None, "last": values[-1] if values else None}
            if len(values) != len(FAST_REVISIONS):
                errors.append(f"fast_key_{key}_length_{len(values)}")
            if len(non_null) != len(FAST_REVISIONS):
                errors.append(f"fast_key_{key}_non_null_{len(non_null)}")
    out["score_arithmetic_note"] = "Full-score arithmetic is not recomputed from prediction text in this wrapper; the existing alpha0.75 full carrier already records the official-compatible eight measured columns with AoA placeholder 0, and this wrapper records the measured AoA separately for Overall bookkeeping."
    return out



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast-out", default=str(DEFAULT_FAST_OUT))
    ap.add_argument("--aoa-out", default=str(DEFAULT_AOA_OUT))
    ap.add_argument("--base-out", default=str(DEFAULT_BASE_OUT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--collate-timeout-sec", type=int, default=900)
    args = ap.parse_args()

    fast_out = pathlib.Path(args.fast_out)
    aoa_out = pathlib.Path(args.aoa_out)
    base_out = pathlib.Path(args.base_out)
    if not fast_out.is_absolute():
        fast_out = USER_ROOT / fast_out
    if not aoa_out.is_absolute():
        aoa_out = USER_ROOT / aoa_out
    if not base_out.is_absolute():
        base_out = USER_ROOT / base_out
    base_out.mkdir(parents=True, exist_ok=True)

    fast_val = validate_fast_summary(fast_out)
    aoa_val = validate_aoa_summary(aoa_out)
    errors: list[str] = []
    if not fast_val.get("ok"):
        errors.append("fast_summary_not_ready")
    if not aoa_val.get("ok"):
        errors.append("aoa_summary_not_ready")
    if errors:
        out = {"status": "ALPHA075_COLLATE_PREFLIGHT_NEEDS_INPUTS", "created_utc": now(), "fast_validation": fast_val, "aoa_validation": aoa_val, "errors": errors}
        write_json(base_out / "alpha075_collate_fast_aoa_summary.json", out)
        print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
        raise SystemExit(1)

    staging = stage_results_tree(base_out, fast_out, aoa_out, force=args.force)
    # The fast evaluator writes/links fast_eval_data. The official collator needs that data root.
    clean_link(fast_out / "fast_eval_data", base_out / "fast_eval_data")
    input_validation = validate_input_files(base_out)
    if not input_validation.get("ok"):
        out = {"status": "ALPHA075_COLLATE_INPUT_FILES_NEED_REPAIR", "created_utc": now(), "fast_validation": fast_val, "aoa_validation": aoa_val, "staging": staging, "input_validation": input_validation, "errors": ["input_files_missing_or_unreadable"]}
        write_json(base_out / "alpha075_collate_fast_aoa_summary.json", out)
        print(json.dumps({k: out[k] for k in ["status", "errors"]}, indent=2), flush=True)
        raise SystemExit(1)

    collate = run_collate(base_out, timeout_sec=args.collate_timeout_sec)
    collated = base_out / "collate_fast/results/hf_model/all_full_preds_and_fast_scores_mlm.json"
    inspection: dict[str, Any] = {}
    if collate.get("returncode") == 0 and collated.exists():
        inspection = inspect_collated(collated)
    else:
        errors.append("official_collate_command_failed_or_missing_output")
    errors.extend(inspection.get("errors", []))
    # The full metric arithmetic for alpha0.75 is already present in the source carrier
    # manifest with AoA placeholder 0; replacing AoA affects Overall by aoa/9.
    aoa_score = float(aoa_val.get("aoa", 0.0))
    carrier_manifest_path = fast_out / "fast_missing_eval_summary.json"
    fast_summary = read_json(carrier_manifest_path)
    overall_with_aoa0 = fast_summary.get("preparation", {}).get("full_main", {}).get("overall_with_aoa0")
    overall_with_measured_aoa = None
    if overall_with_aoa0 is not None:
        overall_with_measured_aoa = float(overall_with_aoa0) + aoa_score / 9.0

    status = "ALPHA075_COLLATE_FAST_AOA_DONE" if collate.get("returncode") == 0 and collated.exists() and not errors else "ALPHA075_COLLATE_FAST_AOA_NEEDS_REPAIR"
    out = {
        "status": status,
        "created_utc": now(),
        "purpose": {
            "question": "Can coherent86 alpha0.75 be represented as a complete official-compatible Strict-Small prediction artifact with real AoA and all Fast revisions?",
            "minimum_cost_action": "Link existing full predictions and 119 reusable fast files, add the 14 newly generated alpha-specific fast files and one measured AoA run, then collate once with the unmodified official script.",
            "scientific_boundary": "This completes a practical endpoint artifact only; it does not establish a new learning principle and should not displace the mechanism line.",
        },
        "fast_out": rel(fast_out),
        "aoa_out": rel(aoa_out),
        "base_out": rel(base_out),
        "fast_validation": fast_val,
        "aoa_validation": aoa_val,
        "staging": staging,
        "input_validation_summary": {"ok": input_validation.get("ok"), "missing_or_unreadable_count": len(input_validation.get("missing_or_unreadable", [])), "fast_records_count": input_validation.get("fast_records_count"), "full_records_count": input_validation.get("full_records_count")},
        "collate_record": collate,
        "collated_inspection": inspection,
        "final_collated": {"path": rel(collated), "exists": collated.exists(), "size_bytes": collated.stat().st_size if collated.exists() else None, "sha256": sha256_file(collated) if collated.exists() else None},
        "score_arithmetic_from_existing_full_carrier": {"overall_with_aoa0": overall_with_aoa0, "measured_aoa": aoa_score, "overall_with_measured_aoa": overall_with_measured_aoa},
        "errors": errors,
    }
    write_json(base_out / "alpha075_collate_fast_aoa_summary.json", out)
    md = [
        "# research alpha0.75 Fast + AoA collation",
        "",
        f"Status: `{status}`",
        f"Measured AoA: `{aoa_score}`",
        f"Overall from existing full carrier plus measured AoA/9: `{overall_with_measured_aoa}`",
        f"Final collated: `{rel(collated)}`",
        f"Errors: `{errors}`",
        "",
        "This is a practical endpoint artifact, not a mechanism result.",
        f"JSON: `{rel(base_out / 'alpha075_collate_fast_aoa_summary.json')}`",
    ]
    (base_out / "alpha075_collate_fast_aoa_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "final_collated": rel(collated),
        "final_collated_sha256": sha256_file(collated) if collated.exists() else None,
        "measured_aoa": aoa_score,
        "overall_with_measured_aoa": overall_with_measured_aoa,
        "errors": errors,
        "summary": rel(base_out / "alpha075_collate_fast_aoa_summary.json"),
    }, indent=2, ensure_ascii=False), flush=True)
    if status.endswith("NEEDS_REPAIR"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
