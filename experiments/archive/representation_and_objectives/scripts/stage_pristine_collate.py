#!/usr/bin/env python3
"""Generic pristine official-coordinate collation for research+ full evaluations.

This script stages prediction artifacts from an inherited full evaluation wrapper
(zero-shot non-EWoK columns, Reading, SuperGLUE) plus separately repaired
pristine-current EWoK predictions and official min_context=0 AoA outputs, then
runs the unmodified current BabyLM strict collator reconstructed in research.

research hardening: the script is fail-closed on artifact discovery and scoring.
It no longer recursively picks the newest matching prediction when the exact
expected target/endpoint path is absent, it scores GlobalPIQA against the research
current-official generated files directly, and it refuses to score missing AoA as
0.0. These changes do not modify any model predictions or score definitions;
they prevent stale or ambiguous artifacts from becoming a false endpoint.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
STUDY = ROOT / "experiments/archive/representation_and_objectives"
WORKSPACE = STUDY
PRISTINE_STRICT = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict"
GLOBALPIQA_GENERATED_FULL = WORKSPACE / "data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval"
SUPERGLUE = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]
FINETUNE_METRIC = {"boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1", "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"}
OVERALL_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
NLP_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE"]
HUMAN_KEYS = ["Reading", "AoA"]
REQUIRED_AOA_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{10*i}M" for i in range(1, 11)]
EXPECTED_GLOBALPIQA = {
    "global_piqa_parallel": {"sha256": "cb9513ed5096ad8115becbe42fdfc97711980f519ada87808e3ddc5f8d5e4453", "num_lines": 103},
    "global_piqa_nonparallel": {"sha256": "aeec831d3adb09bf268ce1b847a854d105b3922c440a164a2c5893b96b228a8f", "num_lines": 100},
}
ZERO_REL_BY_COLUMN = {
    "BLiMP": "blimp/blimp_filtered/predictions.json",
    "Supplement": "blimp/supplement_filtered/predictions.json",
    "EWoK": "ewok/ewok_filtered/predictions.json",
    "Entity": "entity_tracking/entity_tracking/predictions.json",
    "COMPS": "comps/comps/predictions.json",
    "GlobalPIQA_parallel": "global_piqa_parallel/global_piqa_parallel/predictions.json",
    "GlobalPIQA_nonparallel": "global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
    "Reading": "reading/predictions.json",
}
COLLATED_TOP_KEYS = [
    "aoa", "aoa_surprisals", "blimp", "blimp_supplement", "comps",
    "entity_tracking_filtered", "ewok", "global_piqa_nonparallel",
    "global_piqa_parallel", "glue", "reading",
]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sanitize_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_") or "model"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def jsonl_counts(root: Path) -> dict[str, int]:
    return {p.stem: count_lines(p) for p in sorted(root.glob("*.jsonl"))}


def comps_counts(root: Path) -> dict[str, int]:
    """Return COMPS counts using the key names emitted by the official collator."""
    out = {}
    for p in sorted(root.glob("*.jsonl")):
        key = p.stem
        if key.startswith("comps_"):
            key = key[len("comps_"):]
        key = key.replace("-", "_")
        out[key] = count_lines(p)
    return out


def find_zero_prediction(full_root: Path, target: str, column: str, endpoint: str, pristine_ewok_path: Path) -> Path:
    if column == "EWoK":
        return pristine_ewok_path
    revision = f"full_{target}_{column}"
    p = full_root / "official_outputs" / target / column / endpoint / revision / "zero_shot/mlm" / ZERO_REL_BY_COLUMN[column]
    if not p.exists():
        raise FileNotFoundError({"missing_exact_zero_shot_prediction": str(p), "column": column, "target": target, "endpoint": endpoint})
    return p


def find_aoa(aoa_dir: Path, model_root: Path, which: str) -> Path:
    p = aoa_dir / f"{model_root.stem}_local_ckpts" / "main/zero_shot/mlm/AoA_word" / which
    if not p.exists():
        raise FileNotFoundError({"missing_exact_aoa_file": str(p), "aoa_dir": str(aoa_dir), "model_root_stem": model_root.stem})
    return p


def superglue_paths(full_root: Path, target: str, task: str, endpoint: str) -> tuple[Path, Path]:
    root = full_root / "superglue_results" / target / task / endpoint / "main/finetune" / task
    pred = root / "predictions.json"
    res = root / "results.txt"
    if not pred.exists() or not res.exists():
        raise FileNotFoundError({"task": task, "pred": str(pred), "results": str(res), "target": target, "endpoint": endpoint})
    return pred, res


def copy_or_link(src: Path, dst: Path, symlink: bool) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        os.symlink(src.resolve(), dst)
        mode = "symlink"
    else:
        shutil.copy2(src, dst)
        mode = "copy"
    return {"src": str(src), "dst": str(dst), "mode": mode, "size_bytes": src.stat().st_size, "sha256": sha256_file(src)}


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {"cmd": cmd, "cwd": str(cwd), "returncode": p.returncode, "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-8000:], "elapsed_sec": round(time.time() - t0, 3)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_lengths(block: Any) -> dict[str, int]:
    if not isinstance(block, dict):
        return {}
    out = {}
    for k, v in block.items():
        if isinstance(v, dict):
            preds = v.get("predictions")
            if isinstance(preds, list):
                out[str(k)] = len(preds)
    return out


def summarize_collated(path: Path) -> dict[str, Any]:
    c = load_json(path)
    out: dict[str, Any] = {"top_keys": sorted(c.keys()), "null_keys": sorted(k for k, v in c.items() if v is None)}
    for key in ["blimp", "blimp_supplement", "ewok", "entity_tracking_filtered", "comps", "reading", "glue"]:
        v = c.get(key)
        if isinstance(v, dict):
            out[f"{key}_lengths"] = prediction_lengths(v)
    for key in ["global_piqa_parallel", "global_piqa_nonparallel"]:
        v = c.get(key)
        if isinstance(v, dict):
            out[f"{key}_example_count"] = len(v)
            out[f"{key}_prediction_total"] = sum(len(x.get("predictions", [])) for x in v.values() if isinstance(x, dict))
    aoa_s = c.get("aoa_surprisals")
    if isinstance(aoa_s, dict):
        rows = aoa_s.get("results", [])
        cnt = collections.Counter(r.get("step") for r in rows if isinstance(r, dict))
        out["aoa_surprisal_num_rows"] = len(rows)
        out["aoa_surprisal_num_steps"] = len(cnt)
        out["aoa_surprisal_row_count_values"] = sorted(set(cnt.values()))
        out["aoa_surprisal_step_counts"] = dict(sorted(cnt.items()))
    if isinstance(c.get("aoa"), dict):
        out["aoa_score_keys"] = sorted(c["aoa"].keys())
        out["aoa_value"] = c["aoa"].get("aoa")
    return out


def parse_results_txt(path: Path, metric: str) -> float:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        k, _, v = line.partition(":")
        if k.strip() == metric:
            val = float(v.strip()) * 100.0
            if not math.isfinite(val):
                raise ValueError(f"metric {metric} is not finite in {path}: {v}")
            return val
    raise ValueError(f"metric {metric} not found in {path}")


def expected_entity_counts(full: Path) -> dict[str, int]:
    subtask_to_targets: dict[str, int] = collections.defaultdict(int)
    for data_path in sorted((full / "entity_tracking").glob("*.jsonl")):
        with data_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                if any("nothing" in option for option in ex["options"]):
                    continue
                subtask_to_targets[f'{data_path.stem}_{ex["numops"]}_ops'] += 1
    return dict(sorted(subtask_to_targets.items()))


def globalpiqa_file(task_name: str) -> Path:
    p = GLOBALPIQA_GENERATED_FULL / task_name / "eng_latn.jsonl"
    exp = EXPECTED_GLOBALPIQA[task_name]
    if not p.exists():
        raise FileNotFoundError(p)
    got_sha = sha256_file(p)
    got_lines = count_lines(p)
    if got_sha != exp["sha256"] or got_lines != exp["num_lines"]:
        raise RuntimeError({"globalpiqa_file_mismatch": str(p), "sha256": got_sha, "lines": got_lines, "expected": exp})
    return p


def score_collated(collated_path: Path, staged_superglue_results: dict[str, Path]) -> dict[str, Any]:
    sys.path.insert(0, str((PRISTINE_STRICT / "evaluation_pipeline").resolve()))
    import calculate_results_from_pred as calc  # type: ignore
    import collate_preds as coll  # type: ignore

    c = load_json(collated_path)
    full = PRISTINE_STRICT / "evaluation_data/full_eval"
    scores: dict[str, float] = {}
    details: dict[str, Any] = {}
    scores["BLiMP"] = float(calc._calculate_blimp_results(c["blimp"], full / "blimp_filtered"))
    scores["Supplement"] = float(calc._calculate_blimp_results(c["blimp_supplement"], full / "supplement_filtered"))
    scores["EWoK"] = float(calc._calculate_ewok_results(c["ewok"], full / "ewok_filtered"))
    subtask_to_targets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for data_path in sorted((full / "entity_tracking").glob("*.jsonl")):
        with data_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                if any("nothing" in option for option in ex["options"]):
                    continue
                subtask_to_targets[f'{data_path.stem}_{ex["numops"]}_ops'].append(ex)
    et_scores = []
    et_counts = {}
    for subtask, block in c["entity_tracking_filtered"].items():
        preds = block["predictions"]
        targets = subtask_to_targets[subtask]
        if len(preds) != len(targets):
            raise RuntimeError(f"Entity length mismatch {subtask}: {len(preds)} vs {len(targets)}")
        correct = sum(1 for pred, ex in zip(preds, targets) if pred["pred"].strip() == ex["options"][0].strip())
        et_scores.append(correct / len(targets))
        et_counts[subtask] = {"correct": correct, "total": len(targets)}
    scores["Entity"] = 100.0 * sum(et_scores) / len(et_scores)
    details["Entity"] = et_counts
    scores["COMPS"] = float(calc._calculate_comps_results(c["comps"], full / "comps"))
    gp = []
    gp_detail = {}
    for key, task_name in [("global_piqa_parallel", "global_piqa_parallel"), ("global_piqa_nonparallel", "global_piqa_nonparallel")]:
        f = globalpiqa_file(task_name)
        frac = coll._calculate_global_piqa_results(c[key], f, task_name)[task_name]
        scores[task_name] = 100.0 * float(frac)
        gp.append(scores[task_name])
        gp_detail[task_name] = {"score": scores[task_name], "target_file": str(f), "sha256": sha256_file(f), "num_lines": count_lines(f)}
    scores["GlobalPIQA"] = sum(gp) / 2.0
    details["GlobalPIQA"] = gp_detail
    reading_frac = coll._calculate_reading_results(c["reading"], full / "reading/reading_data.csv")
    scores["Reading"] = 100.0 * (float(reading_frac["spr"]) + float(reading_frac["rt"])) / 2.0
    sg_scores = []
    sg_detail = {}
    for task in SUPERGLUE:
        metric = FINETUNE_METRIC[task]
        p = staged_superglue_results[task]
        val = parse_results_txt(p, metric)
        sg_scores.append(val)
        sg_detail[task] = {"metric": metric, "score": val, "results_txt": str(p)}
    scores["SuperGLUE"] = sum(sg_scores) / len(sg_scores)
    details["SuperGLUE"] = sg_detail
    aoa_block = c.get("aoa")
    if not isinstance(aoa_block, dict) or "aoa" not in aoa_block:
        raise RuntimeError("AoA score field missing from collated file; refusing to substitute 0.0")
    aoa_raw = float(aoa_block["aoa"])
    if not math.isfinite(aoa_raw):
        raise RuntimeError(f"AoA score is not finite: {aoa_raw}")
    scores["AoA"] = 100.0 * aoa_raw
    for k, v in list(scores.items()):
        if not math.isfinite(float(v)):
            raise RuntimeError(f"nonfinite score {k}={v}")
    overall = sum(scores[k] for k in OVERALL_KEYS) / len(OVERALL_KEYS)
    return {
        "scores": {k: scores[k] for k in OVERALL_KEYS},
        "GlobalPIQA_parallel": scores["global_piqa_parallel"],
        "GlobalPIQA_nonparallel": scores["global_piqa_nonparallel"],
        "Overall": overall,
        "NLP_average": sum(scores[k] for k in NLP_KEYS) / len(NLP_KEYS),
        "Human_like_average": sum(scores[k] for k in HUMAN_KEYS) / len(HUMAN_KEYS),
        "margin_over_visible_leader_41p8": overall - 41.8,
        "details": details,
    }


def compare_counts(name: str, got: dict[str, int], expected: dict[str, int]) -> list[str]:
    errors = []
    if set(got) != set(expected):
        errors.append(f"{name} keys mismatch: missing={sorted(set(expected)-set(got))[:10]} extra={sorted(set(got)-set(expected))[:10]}")
    for k in sorted(set(got) & set(expected)):
        if got[k] != expected[k]:
            errors.append(f"{name} count mismatch {k}: {got[k]} != {expected[k]}")
            if len(errors) > 20:
                break
    return errors


def validate_collated(collated_path: Path, summary: dict[str, Any], score: dict[str, Any], collate_started_at: float) -> dict[str, Any]:
    errors: list[str] = []
    evidence: dict[str, Any] = {}
    if not collated_path.exists():
        return {"errors": ["collated file missing"], "evidence": evidence}
    if collated_path.stat().st_mtime + 1e-6 < collate_started_at:
        errors.append("collated file mtime predates this invocation")
    c = load_json(collated_path)
    if sorted(c.keys()) != COLLATED_TOP_KEYS:
        errors.append(f"top-level keys mismatch: {sorted(c.keys())}")
    if summary.get("null_keys"):
        errors.append(f"null collated keys: {summary.get('null_keys')}")
    full = PRISTINE_STRICT / "evaluation_data/full_eval"
    # Dynamically verify prediction counts against the official target data where the mapping is direct.
    expected_blimp = jsonl_counts(full / "blimp_filtered")
    expected_supp = jsonl_counts(full / "supplement_filtered")
    expected_ewok = jsonl_counts(full / "ewok_filtered")
    expected_comps = comps_counts(full / "comps")
    expected_entity = expected_entity_counts(full)
    expected_glue = {task: count_lines(full / "glue_filtered" / f"{task}.valid.jsonl") for task in SUPERGLUE}
    evidence["expected_counts"] = {
        "blimp_total": sum(expected_blimp.values()),
        "supplement_total": sum(expected_supp.values()),
        "ewok_total": sum(expected_ewok.values()),
        "comps_total": sum(expected_comps.values()),
        "entity_total": sum(expected_entity.values()),
        "glue": expected_glue,
    }
    errors += compare_counts("BLiMP", summary.get("blimp_lengths", {}), expected_blimp)
    errors += compare_counts("Supplement", summary.get("blimp_supplement_lengths", {}), expected_supp)
    errors += compare_counts("EWoK", summary.get("ewok_lengths", {}), expected_ewok)
    errors += compare_counts("COMPS", summary.get("comps_lengths", {}), expected_comps)
    errors += compare_counts("Entity", summary.get("entity_tracking_filtered_lengths", {}), expected_entity)
    errors += compare_counts("SuperGLUE", summary.get("glue_lengths", {}), expected_glue)
    # Reading target has a CSV interface; the current official collated output should carry the known 1726 rows.
    if summary.get("reading_lengths", {}).get("reading") != 1726:
        errors.append(f"Reading prediction count {summary.get('reading_lengths', {}).get('reading')} != 1726")
    # Current-official GlobalPIQA generated files: exact count and hash from research.
    for key, task_name in [("global_piqa_parallel", "global_piqa_parallel"), ("global_piqa_nonparallel", "global_piqa_nonparallel")]:
        f = globalpiqa_file(task_name)
        expected_n = EXPECTED_GLOBALPIQA[task_name]["num_lines"]
        ex_count = summary.get(f"{key}_example_count")
        pred_count = summary.get(f"{key}_prediction_total")
        if ex_count != expected_n or pred_count != expected_n:
            errors.append(f"{key} count mismatch: examples={ex_count}, predictions={pred_count}, expected={expected_n}")
        evidence[f"{key}_target"] = {"path": str(f), "sha256": sha256_file(f), "num_lines": count_lines(f)}
    # AoA: exactly 19 official Strict-Small steps and 8005 contexts per step, and a finite explicit score.
    if summary.get("aoa_surprisal_num_rows") != 152095:
        errors.append(f"AoA surprisal rows {summary.get('aoa_surprisal_num_rows')} != 152095")
    if summary.get("aoa_surprisal_num_steps") != 19:
        errors.append(f"AoA step count {summary.get('aoa_surprisal_num_steps')} != 19")
    if summary.get("aoa_surprisal_row_count_values") != [8005]:
        errors.append(f"AoA row count values {summary.get('aoa_surprisal_row_count_values')} != [8005]")
    step_counts = summary.get("aoa_surprisal_step_counts", {})
    if set(step_counts) != set(REQUIRED_AOA_STEPS):
        errors.append(f"AoA step labels mismatch: {sorted(step_counts)}")
    aoa_value = summary.get("aoa_value")
    try:
        if not math.isfinite(float(aoa_value)):
            errors.append(f"AoA value nonfinite: {aoa_value}")
    except Exception:
        errors.append(f"AoA value missing/unparseable: {aoa_value}")
    # Score vector coherence.
    if not isinstance(score, dict) or not isinstance(score.get("scores"), dict):
        errors.append("score summary missing")
    else:
        missing = [k for k in OVERALL_KEYS if k not in score["scores"]]
        if missing:
            errors.append(f"score vector missing keys {missing}")
        else:
            recomputed = sum(float(score["scores"][k]) for k in OVERALL_KEYS) / len(OVERALL_KEYS)
            if abs(recomputed - float(score.get("Overall"))) > 1e-9:
                errors.append(f"Overall mismatch {score.get('Overall')} vs recomputed {recomputed}")
            for k in OVERALL_KEYS:
                if not math.isfinite(float(score["scores"][k])):
                    errors.append(f"score {k} not finite: {score['scores'][k]}")
    return {"errors": errors, "evidence": evidence}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-root", type=Path, required=True, help="Full eval root, e.g. data/strictsmalltok_seed43022_full_eval")
    ap.add_argument("--target", required=True, help="Target key used in the full eval wrapper")
    ap.add_argument("--model-root", type=Path, required=True, help="Local model root containing hf_model checkpoints")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pristine-ewok-predictions", type=Path, required=True)
    ap.add_argument("--aoa-dir", type=Path, required=True)
    ap.add_argument("--endpoint", default="chck_100M")
    ap.add_argument("--tag", default="strictsmalltok")
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tag = sanitize_tag(args.tag)
    full_root = args.full_root.resolve()
    model_root = args.model_root.resolve()
    out_dir = args.out_dir.resolve()
    pristine_ewok = args.pristine_ewok_predictions.resolve()
    aoa_dir = args.aoa_dir.resolve()

    required = {
        "PRISTINE_STRICT": PRISTINE_STRICT,
        "GLOBALPIQA_GENERATED_FULL": GLOBALPIQA_GENERATED_FULL,
        "full_root": full_root,
        "model_root": model_root,
        "pristine_ewok_predictions": pristine_ewok,
        "aoa_dir": aoa_dir,
    }
    missing = {k: str(p) for k, p in required.items() if not p.exists()}
    if missing:
        raise FileNotFoundError(missing)

    if args.dry_run:
        checks = {k: str(p) for k, p in required.items()}
        for col in ["BLiMP", "Supplement", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]:
            checks[f"prediction_{col}"] = str(find_zero_prediction(full_root, args.target, col, args.endpoint, pristine_ewok))
        checks["prediction_EWoK"] = str(pristine_ewok)
        checks["aoa_surprisal"] = str(find_aoa(aoa_dir, model_root, "surprisal.json"))
        checks["aoa_score"] = str(find_aoa(aoa_dir, model_root, "aoa_score.json"))
        for task in SUPERGLUE:
            pred, res = superglue_paths(full_root, args.target, task, args.endpoint)
            checks[f"superglue_{task}_pred"] = str(pred)
            checks[f"superglue_{task}_results"] = str(res)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"pristine_collate_{tag}_dryrun.json"
        payload = {"status": "PRISTINE_COLLATE_DRYRUN", "created_utc": now_utc(), "tag": tag, "checks": checks}
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"], "out_json": str(out_json), "checked_files": len(checks)}, indent=2), flush=True)
        return

    # Clean output directory contents, not the mount root itself. This makes the
    # collated file newly created by this invocation rather than reused.
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in list(out_dir.iterdir()):
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    results_dir = out_dir / "results"
    model_stem = model_root.stem
    zero_root = results_dir / model_stem / "main/zero_shot/mlm"
    fine_root = results_dir / model_stem / "main/finetune"
    staged = []
    columns = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
    for col in columns:
        src = find_zero_prediction(full_root, args.target, col, args.endpoint, pristine_ewok)
        staged.append(copy_or_link(src, zero_root / ZERO_REL_BY_COLUMN[col], symlink=not args.copy))
    staged.append(copy_or_link(find_aoa(aoa_dir, model_root, "surprisal.json"), zero_root / "AoA_word/surprisal.json", symlink=not args.copy))
    staged.append(copy_or_link(find_aoa(aoa_dir, model_root, "aoa_score.json"), zero_root / "AoA_word/aoa_score.json", symlink=not args.copy))

    sg_results: dict[str, Path] = {}
    for task in SUPERGLUE:
        pred, res = superglue_paths(full_root, args.target, task, args.endpoint)
        sg_results[task] = res
        staged.append(copy_or_link(pred, fine_root / task / "predictions.json", symlink=not args.copy))

    env = os.environ.copy()
    for rel, key in [("hf_home", "HF_HOME"), ("hf_home/hub", "HF_HUB_CACHE"), ("datasets", "HF_DATASETS_CACHE"), ("transformers", "TRANSFORMERS_CACHE"), ("modules", "HF_MODULES_CACHE"), ("tmp", "TMPDIR")]:
        p = out_dir / rel
        p.mkdir(parents=True, exist_ok=True)
        env[key] = str(p.resolve())
    env["PYTHONPATH"] = str(PRISTINE_STRICT.resolve()) + os.pathsep + str((PRISTINE_STRICT / "evaluation_pipeline").resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"

    collate_cmd = [
        sys.executable, "evaluation_pipeline/collate_preds.py",
        "--model_path_or_name", str(model_root.resolve()),
        "--backend", "mlm",
        "--results_dir", str(results_dir.resolve()),
        "--revision_name", "main",
        "--track", "strict-small",
    ]
    collate_started_at = time.time()
    collate_run = run(collate_cmd, PRISTINE_STRICT, env)
    collated = results_dir / model_stem / "all_full_preds_and_fast_scores_mlm.json"
    summary = summarize_collated(collated) if collated.exists() else {}
    score: dict[str, Any] = {}
    scoring_error = None
    if collated.exists() and not summary.get("null_keys"):
        try:
            score = score_collated(collated, sg_results)
        except Exception as exc:
            scoring_error = repr(exc)
    validation = validate_collated(collated, summary, score, collate_started_at) if collated.exists() else {"errors": ["collated file missing"], "evidence": {}}
    validation_errors = list(validation.get("errors", []))
    if collate_run.get("returncode") != 0:
        validation_errors.insert(0, f"collate_returncode={collate_run.get('returncode')}")
    if scoring_error:
        validation_errors.append(f"scoring_error={scoring_error}")
    status = "PRISTINE_COLLATE" if not validation_errors else "PRISTINE_COLLATE_FAILED"
    out = {
        "status": status,
        "created_utc": now_utc(),
        "tag": tag,
        "target": args.target,
        "endpoint": args.endpoint,
        "purpose": "Apply the same official-coordinate collation/scoring convention as Steps 37 and 44, with research fail-closed artifact checks, to a research Strict-Small-tokenizer full evaluation.",
        "paths": {
            "out_dir": str(out_dir),
            "results_dir": str(results_dir),
            "collated_path": str(collated),
            "full_eval_root": str(full_root),
            "model_root": str(model_root),
            "pristine_ewok_predictions": str(pristine_ewok),
            "aoa_dir": str(aoa_dir),
            "globalpiqa_generated_full": str(GLOBALPIQA_GENERATED_FULL),
        },
        "staged_files": staged,
        "superglue_result_files": {task: str(path) for task, path in sg_results.items()},
        "collate_run": collate_run,
        "collated_summary": summary,
        "score_summary": score,
        "validation": validation,
        "validation_errors": validation_errors,
        "coordinate_notes": {
            "pristine_strict": str(PRISTINE_STRICT),
            "globalpiqa_lineage": "Scoring reads the research files generated by current official dl.py directly; expected SHA/count assertions are enforced at runtime.",
            "superglue_primary_metric": "f1 for MRPC/QQP, accuracy for boolq/mnli/multirc/rte/wsc, matching research seed43022 coordinate.",
            "aoa_expected": "official min_context=0, 8005 rows/checkpoint over 19 checkpoints; missing AoA is no longer replaced by 0.0.",
            "ewok_expected": "current pristine official 7618-row ewok_filtered coordinate",
            "artifact_discovery": "Exact target/endpoint paths only; recursive latest-file fallback disabled in research.",
        },
    }
    if collated.exists():
        out["collated_sha256"] = sha256_file(collated)
    out_json = out_dir / f"pristine_collate_{tag}_summary.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "collate_returncode": collate_run.get("returncode"),
        "null_keys": summary.get("null_keys"),
        "ewok_total": sum(summary.get("ewok_lengths", {}).values()) if isinstance(summary.get("ewok_lengths"), dict) else None,
        "aoa_row_count_values": summary.get("aoa_surprisal_row_count_values"),
        "overall": score.get("Overall"),
        "margin_over_visible_leader_41p8": score.get("margin_over_visible_leader_41p8"),
        "validation_error_count": len(validation_errors),
        "collated_sha256": out.get("collated_sha256"),
        "out_json": str(out_json),
    }, indent=2), flush=True)
    if validation_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
