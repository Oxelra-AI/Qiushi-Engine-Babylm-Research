#!/usr/bin/env python3
"""Build and audit a pristine BabyLM 2026 Strict official evaluation coordinate.

The research endpoint-protection work showed two official-path mismatches for the
compact_view_reinvest endpoint: AoA rows were produced with min_context=20 rather
than the official min_context=0, and the official collator's hard-coded EWoK
sizes did not match the locally vendored ewok_filtered files.  This script makes
that second issue independent of old local repository state by using a fresh
upstream code checkout plus freshly downloaded/regenerated evaluation data.

It does not patch official code or data.  It records hashes, counts, and a
collation probe using the existing research predictions with the old AoA output.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/representation_and_objectives')
OUT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate')
NOTE = _public_path('research/notes/representation_and_objectives/pristine_official_coordinate_audit.md')
REPO_DIR = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval')
STRICT_DIR = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict')
HF_HOME = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/hf_home')
HF_DATASETS_CACHE = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/hf_datasets_cache')
NLTK_DATA = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/nltk_data')
EXPECTED_CODE_COMMIT = "6f825c291e2c4c78ad33b1935fd64d45f52642dc"
STRICT_EVAL_REPO = "BabyLM-community/BabyLM-2026-Strict-Evals"
EWOK_REPO = "ewok-core/ewok-core-1.0"
MODEL_DIR = _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model')
research = _public_path('experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval')
LOCAL_STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')

ZERO_SHOT_COPY_MAP = [
    (
        _public_path('experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/official_outputs/compact_view_reinvest/BLiMP/chck_100M/full_compact_view_reinvest_BLiMP/zero_shot/mlm/blimp/blimp_filtered/predictions.json'),
        Path("blimp/blimp_filtered/predictions.json"),
    ),
    (
        _public_path('experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/official_outputs/compact_view_reinvest/Supplement/chck_100M/full_compact_view_reinvest_Supplement/zero_shot/mlm/blimp/supplement_filtered/predictions.json'),
        Path("blimp/supplement_filtered/predictions.json"),
    ),
    (
        _public_path('experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/official_outputs/compact_view_reinvest/EWoK/chck_100M/full_compact_view_reinvest_EWoK/zero_shot/mlm/ewok/ewok_filtered/predictions.json'),
        Path("ewok/ewok_filtered/predictions.json"),
    ),
    (
        _public_path('experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/official_outputs/compact_view_reinvest/Entity/chck_100M/full_compact_view_reinvest_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json'),
        Path("entity_tracking/entity_tracking/predictions.json"),
    ),
    (
        _public_path('experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/official_outputs/compact_view_reinvest/COMPS/chck_100M/full_compact_view_reinvest_COMPS/zero_shot/mlm/comps/comps/predictions.json'),
        Path("comps/comps/predictions.json"),
    ),
    (
        _public_path('experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/official_outputs/compact_view_reinvest/GlobalPIQA_parallel/chck_100M/full_compact_view_reinvest_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        Path("global_piqa_parallel/global_piqa_parallel/predictions.json"),
    ),
    (
        _public_path('experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/official_outputs/compact_view_reinvest/GlobalPIQA_nonparallel/chck_100M/full_compact_view_reinvest_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
        Path("global_piqa_nonparallel/global_piqa_nonparallel/predictions.json"),
    ),
    (
        _public_path('experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/official_outputs/compact_view_reinvest/Reading/chck_100M/full_compact_view_reinvest_Reading/zero_shot/mlm/reading/predictions.json'),
        Path("reading/predictions.json"),
    ),
]

FINE_TUNE_TASKS = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def safe_clear_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in list(path.iterdir()):
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 900) -> dict[str, Any]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "cwd": rel(cwd) if cwd is not None else None,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
            "elapsed_sec": time.time() - t0,
        }
    except Exception as exc:  # preserve failures as evidence
        return {
            "cmd": cmd,
            "cwd": rel(cwd) if cwd is not None else None,
            "returncode": None,
            "exception": type(exc).__name__,
            "error": str(exc),
            "elapsed_sec": time.time() - t0,
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def hash_manifest(base: Path, out_jsonl: Path) -> dict[str, Any]:
    rows = []
    if base.exists():
        for p in sorted(x for x in base.rglob("*") if x.is_file() and ".cache" not in x.parts):
            rows.append({
                "path": rel(p),
                "relative_path": str(p.relative_to(base)),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    aggregate = hashlib.sha256("\n".join(f"{r['relative_path']}\t{r['size_bytes']}\t{r['sha256']}" for r in rows).encode()).hexdigest()
    return {"base": rel(base), "n_files": len(rows), "aggregate_sha256": aggregate, "manifest_jsonl": rel(out_jsonl)}


def parse_assignment_dict(py_path: Path, name: str) -> dict[str, Any]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"assignment {name} not found in {py_path}")


def count_domain_jsonl_files(dir_path: Path) -> dict[str, int]:
    if not dir_path.exists():
        return {}
    return {p.stem: count_jsonl(p) for p in sorted(dir_path.glob("*.jsonl"))}


def compare_dicts(a: dict[str, int], b: dict[str, int]) -> dict[str, Any]:
    keys = sorted(set(a) | set(b))
    return {
        "all_equal": all(a.get(k) == b.get(k) for k in keys),
        "diffs": [
            {"key": k, "left": a.get(k), "right": b.get(k), "delta_left_minus_right": (a.get(k) or 0) - (b.get(k) or 0)}
            for k in keys if a.get(k) != b.get(k)
        ],
    }


def symlink_or_copy(src: Path, dst: Path) -> dict[str, Any]:
    if not src.exists():
        return {"src": rel(src), "dst": rel(dst), "exists": False}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
        mode = "symlink"
    except OSError:
        shutil.copy2(src, dst)
        mode = "copy"
    return {"src": rel(src), "dst": rel(dst), "exists": True, "mode": mode, "size_bytes": src.stat().st_size}


def stage_and_collate_min20(env: dict[str, str]) -> dict[str, Any]:
    results_dir = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/collation_probe_min20_pristine')
    safe_clear_dir(results_dir)
    model_stem = MODEL_DIR.stem
    zero_root = results_dir / model_stem / "main" / "zero_shot" / "mlm"
    fine_root = results_dir / model_stem / "main" / "finetune"
    copied = []
    for src, rel_dst in ZERO_SHOT_COPY_MAP:
        copied.append(symlink_or_copy(src, zero_root / rel_dst))
    for task in FINE_TUNE_TASKS:
        src = _public_path('experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/superglue_results/compact_view_reinvest') / task / "chck_100M" / "main" / "finetune" / task / "predictions.json"
        copied.append(symlink_or_copy(src, fine_root / task / "predictions.json"))
    old_aoa_root = _public_path('experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/aoa_outputs/compact_view_reinvest/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word')
    copied.append(symlink_or_copy(old_aoa_root / "surprisal.json", zero_root / "AoA_word" / "surprisal.json"))
    copied.append(symlink_or_copy(old_aoa_root / "aoa_score.json", zero_root / "AoA_word" / "aoa_score.json"))

    cmd = [
        sys.executable,
        "evaluation_pipeline/collate_preds.py",
        "--model_path_or_name", str(MODEL_DIR),
        "--backend", "mlm",
        "--results_dir", str(results_dir),
        "--revision_name", "main",
        "--track", "strict-small",
    ]
    collate_run = run(cmd, cwd=STRICT_DIR, env=env, timeout=300)
    collated = results_dir / model_stem / "all_full_preds_and_fast_scores_mlm.json"
    collated_meta: dict[str, Any] = {
        "collated_path": rel(collated),
        "collated_exists": collated.exists(),
        "collated_size_bytes": collated.stat().st_size if collated.exists() else None,
        "aoa_surprisals_is_null": None,
        "aoa_score_is_null": None,
        "ewok_is_null": None,
        "ewok_domain_lengths": None,
    }
    if collated.exists():
        with collated.open("r", encoding="utf-8") as f:
            data = json.load(f)
        collated_meta.update({
            "top_keys": sorted(data.keys()),
            "aoa_surprisals_is_null": data.get("aoa_surprisals") is None,
            "aoa_score_is_null": data.get("aoa") is None,
            "ewok_is_null": data.get("ewok") is None,
            "ewok_domain_lengths": None if data.get("ewok") is None else {k: len(v.get("predictions", [])) if isinstance(v, dict) else None for k, v in data.get("ewok", {}).items()},
        })
    return {"results_dir": rel(results_dir), "copied": copied, "collate_run": collate_run, "collated": collated_meta}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    _public_path('research/notes/representation_and_objectives').mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "HF_HOME": str(HF_HOME),
        "HF_DATASETS_CACHE": str(HF_DATASETS_CACHE),
        "NLTK_DATA": str(NLTK_DATA),
        "TOKENIZERS_PARALLELISM": "false",
        "GIT_TERMINAL_PROMPT": "0",
    })
    for p in [HF_HOME, HF_DATASETS_CACHE, NLTK_DATA]:
        p.mkdir(parents=True, exist_ok=True)

    # Clean only child paths controlled by this task; do not delete OUT itself.
    for child_name in ["babylm-eval", "collation_probe_min20_pristine"]:
        child = OUT / child_name
        if child.exists():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    record: dict[str, Any] = {
        "status": "PRISTINE_OFFICIAL_COORDINATE_AUDIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Reconstruct a single unmodified upstream BabyLM strict evaluation coordinate and determine whether the EWoK mismatch belongs to local data or official code/data release inconsistency.",
        "out_dir": rel(OUT),
    }

    # Fresh code checkout.
    clone = run(["git", "clone", "https://github.com/babylm-org/babylm-eval.git", str(REPO_DIR)], env=env, timeout=900)
    checkout = run(["git", "checkout", EXPECTED_CODE_COMMIT], cwd=REPO_DIR, env=env, timeout=300) if REPO_DIR.exists() else {"skipped": True}
    head = run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, env=env, timeout=60) if REPO_DIR.exists() else {"skipped": True}
    code_status_predata = run(["git", "status", "--short"], cwd=REPO_DIR, env=env, timeout=60) if REPO_DIR.exists() else {"skipped": True}
    record["code_checkout"] = {"expected_commit": EXPECTED_CODE_COMMIT, "clone": clone, "checkout": checkout, "head": head, "status_predata": code_status_predata}

    # Pin and download the official Strict eval dataset snapshot into the fresh strict checkout.
    dataset_meta: dict[str, Any] = {}
    download_meta: dict[str, Any] = {}
    try:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi()
        strict_info = api.repo_info(repo_id=STRICT_EVAL_REPO, repo_type="dataset")
        ewok_info = api.repo_info(repo_id=EWOK_REPO, repo_type="dataset")
        dataset_meta = {
            "strict_eval_repo": STRICT_EVAL_REPO,
            "strict_eval_sha": strict_info.sha,
            "strict_eval_last_modified": str(getattr(strict_info, "lastModified", None)),
            "ewok_repo": EWOK_REPO,
            "ewok_sha": ewok_info.sha,
            "ewok_last_modified": str(getattr(ewok_info, "lastModified", None)),
        }
        downloaded_path = snapshot_download(
            repo_id=STRICT_EVAL_REPO,
            repo_type="dataset",
            revision=strict_info.sha,
            local_dir=str(STRICT_DIR),
        )
        download_meta = {"ok": True, "downloaded_path": rel(Path(downloaded_path))}
    except Exception as exc:
        download_meta = {"ok": False, "exception": type(exc).__name__, "error": str(exc)}
    record["hf_dataset_revisions"] = dataset_meta
    record["strict_eval_snapshot_download"] = download_meta

    # NLTK resources needed by official ewok/dl_and_filter.py.
    nltk_meta: list[dict[str, Any]] = []
    try:
        import nltk
        for resource in ["punkt", "punkt_tab"]:
            ok = bool(nltk.download(resource, download_dir=str(NLTK_DATA), quiet=True))
            nltk_meta.append({"resource": resource, "ok": ok})
    except Exception as exc:
        nltk_meta.append({"ok": False, "exception": type(exc).__name__, "error": str(exc)})
    record["nltk_resources"] = nltk_meta

    # Generate official EWoK filtered files from untouched official script.
    ewok_run = run([sys.executable, "-m", "evaluation_pipeline.ewok.dl_and_filter"], cwd=STRICT_DIR, env=env, timeout=900) if STRICT_DIR.exists() else {"skipped": True}
    record["official_ewok_dl_and_filter_run"] = ewok_run

    # Code identities after data download/generation.  Untracked data is expected; code diffs are not.
    code_diff = run(["git", "diff", "--", "strict/evaluation_pipeline", "strict/scripts", "strict/README.md"], cwd=REPO_DIR, env=env, timeout=60) if REPO_DIR.exists() else {"skipped": True}
    code_status_postdata = run(["git", "status", "--short", "--", "strict/evaluation_pipeline", "strict/scripts", "strict/README.md"], cwd=REPO_DIR, env=env, timeout=60) if REPO_DIR.exists() else {"skipped": True}
    record["code_cleanliness_after_data"] = {"status_scoped": code_status_postdata, "diff_scoped": code_diff}

    # Hash the evaluation data tree and central official files.
    record["file_identities"] = {}
    central_files = {
        "collate_preds": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/collate_preds.py'),
        "ewok_dl_and_filter": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/ewok/dl_and_filter.py'),
        "ewok_vocab": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/ewok/vocab.txt'),
        "aoa_run": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/AoA_word/run.py'),
        "aoa_eval_util": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/AoA_word/eval_util.py'),
        "readme": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/README.md'),
    }
    for name, path in central_files.items():
        record["file_identities"][name] = {"path": rel(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None, "sha256": sha256_file(path) if path.exists() else None}
    record["manifests"] = {
        "strict_evaluation_data": hash_manifest(_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data'), _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/strict_evaluation_data_manifest.jsonl')),
        "official_generated_ewok_filtered": hash_manifest(_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered'), _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/official_generated_ewok_filtered_manifest.jsonl')),
        "local_existing_ewok_filtered": hash_manifest(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered'), _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/local_existing_ewok_filtered_manifest.jsonl')),
    }

    # Count EWoK from generated data, local data, constants, and raw source dataset.
    pristine_ewok_dir = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered')
    local_ewok_dir = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered')
    pristine_counts = count_domain_jsonl_files(pristine_ewok_dir)
    local_counts = count_domain_jsonl_files(local_ewok_dir)
    constants: dict[str, int] = {}
    try:
        constants = {k: int(v) for k, v in parse_assignment_dict(_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/collate_preds.py'), "EWOK_SIZES").items()}
    except Exception as exc:
        record["ewok_constant_parse_error"] = {"exception": type(exc).__name__, "error": str(exc)}

    raw_counts: dict[str, int] = {}
    filtered_counts_from_source: dict[str, int] = {}
    source_count_error: dict[str, Any] | None = None
    try:
        from datasets import load_dataset
        from nltk.tokenize import word_tokenize
        vocab_path = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/ewok/vocab.txt')
        vocab = {line.strip() for line in vocab_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        ds = load_dataset(EWOK_REPO, split="test", cache_dir=str(HF_DATASETS_CACHE))
        raw = Counter()
        filt = Counter()
        for ex in ds:
            domain = ex["Domain"]
            raw[domain] += 2
            skip = False
            for key in ("Context1", "Context2", "Target1", "Target2"):
                for word in word_tokenize(str(ex[key]).lower()):
                    if word not in vocab:
                        skip = True
                        break
                if skip:
                    break
            if not skip:
                filt[domain] += 2
        raw_counts = dict(sorted(raw.items()))
        filtered_counts_from_source = dict(sorted(filt.items()))
    except Exception as exc:
        source_count_error = {"exception": type(exc).__name__, "error": str(exc)}

    record["ewok_counts_and_release_test"] = {
        "collator_EWOK_SIZES": constants,
        "pristine_generated_ewok_filtered_line_counts": pristine_counts,
        "local_existing_ewok_filtered_line_counts": local_counts,
        "source_dataset_raw_domain_counts_times2": raw_counts,
        "source_dataset_filtered_with_official_vocab_counts_times2": filtered_counts_from_source,
        "source_count_error": source_count_error,
        "collator_vs_pristine_filtered": compare_dicts(constants, pristine_counts) if constants and pristine_counts else None,
        "collator_vs_local_filtered": compare_dicts(constants, local_counts) if constants and local_counts else None,
        "collator_vs_source_raw": compare_dicts(constants, raw_counts) if constants and raw_counts else None,
        "collator_vs_source_filtered_recomputed": compare_dicts(constants, filtered_counts_from_source) if constants and filtered_counts_from_source else None,
        "pristine_filtered_vs_local_filtered_counts": compare_dicts(pristine_counts, local_counts) if pristine_counts and local_counts else None,
    }

    # Compare EWoK file hashes directly where names match.
    file_hash_compare = []
    for name in sorted(set(pristine_counts) | set(local_counts)):
        p = pristine_ewok_dir / f"{name}.jsonl"
        q = local_ewok_dir / f"{name}.jsonl"
        file_hash_compare.append({
            "domain": name,
            "pristine_exists": p.exists(),
            "local_exists": q.exists(),
            "pristine_count": count_jsonl(p),
            "local_count": count_jsonl(q),
            "pristine_sha256": sha256_file(p) if p.exists() else None,
            "local_sha256": sha256_file(q) if q.exists() else None,
            "identical_file": p.exists() and q.exists() and sha256_file(p) == sha256_file(q),
        })
    record["ewok_filtered_pristine_vs_local_file_hashes"] = file_hash_compare

    # Run the unmodified pristine collator on existing final predictions with the known-old AoA, as a coordinate sanity probe.
    record["pristine_collation_probe_with_old_min20_aoa"] = stage_and_collate_min20(env) if STRICT_DIR.exists() else {"skipped": True}

    # Interpret only from recorded facts.
    ev = record.get("ewok_counts_and_release_test", {})
    filtered_matches_local = bool(ev.get("pristine_filtered_vs_local_filtered_counts", {}).get("all_equal")) if isinstance(ev.get("pristine_filtered_vs_local_filtered_counts"), dict) else False
    constants_match_raw = bool(ev.get("collator_vs_source_raw", {}).get("all_equal")) if isinstance(ev.get("collator_vs_source_raw"), dict) else False
    constants_match_filtered = bool(ev.get("collator_vs_pristine_filtered", {}).get("all_equal")) if isinstance(ev.get("collator_vs_pristine_filtered"), dict) else False
    record["interpretation"] = {
        "pristine_generated_filtered_counts_equal_local_counts": filtered_matches_local,
        "collator_constants_equal_raw_ewok_source_counts_times2": constants_match_raw,
        "collator_constants_equal_pristine_filtered_counts": constants_match_filtered,
        "scientific_meaning": (
            "If the pristine official script regenerates the same ewok_filtered counts/files as the local vendored data while EWOK_SIZES matches the raw source-domain counts rather than the filtered files, then the mismatch is an upstream code/data coordinate inconsistency, not an A01 local corruption. The endpoint must still be reproduced through one accepted official coordinate before using the scalar against the public leader."
        ),
    }
    record["elapsed_sec"] = time.time() - t0

    out_json = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/pristine_official_coordinate_audit.json')
    out_json.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    note_lines = [
        "# research pristine official coordinate audit",
        "",
        "Purpose: build a fresh upstream BabyLM evaluation coordinate, hash the evaluation data, and decide whether the EWoK size mismatch comes from old local files or from a code/data release inconsistency.",
        "",
        f"- Audit JSON: `{rel(out_json)}`",
        f"- Fresh code directory: `{rel(REPO_DIR)}`",
        f"- Expected code commit: `{EXPECTED_CODE_COMMIT}`; observed head: `{str(head.get('stdout','')).strip() if isinstance(head, dict) else None}`",
        f"- Strict eval HF dataset revision: `{dataset_meta.get('strict_eval_sha')}`",
        f"- EWoK HF dataset revision: `{dataset_meta.get('ewok_sha')}`",
        f"- Generated EWoK counts equal local counts: {filtered_matches_local}",
        f"- Collator EWOK_SIZES equal source raw domain counts×2: {constants_match_raw}",
        f"- Collator EWOK_SIZES equal generated ewok_filtered counts: {constants_match_filtered}",
        "",
        "## EWoK domain counts",
        "",
        "| domain | collator constant | pristine ewok_filtered | local ewok_filtered | raw source×2 | filtered source×2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    keys = sorted(set(constants) | set(pristine_counts) | set(local_counts) | set(raw_counts) | set(filtered_counts_from_source))
    for k in keys:
        note_lines.append(f"| {k} | {constants.get(k)} | {pristine_counts.get(k)} | {local_counts.get(k)} | {raw_counts.get(k)} | {filtered_counts_from_source.get(k)} |")
    coll_probe = record["pristine_collation_probe_with_old_min20_aoa"]
    if isinstance(coll_probe, dict):
        cr = coll_probe.get("collate_run", {})
        cm = coll_probe.get("collated", {})
        note_lines += [
            "",
            "## Pristine collation probe with old min_context=20 AoA",
            "",
            f"- Return code: `{cr.get('returncode')}`",
            f"- AoA surprisal null: `{cm.get('aoa_surprisals_is_null')}`",
            f"- EWoK null: `{cm.get('ewok_is_null')}`",
            "- Collator stdout excerpt:",
            "",
            "```",
            str(cr.get("stdout", ""))[-4000:],
            "```",
        ]
    note_lines += [
        "",
        "## Interpretation",
        "",
        record["interpretation"]["scientific_meaning"],
        "",
        "No official code or data were patched. This audit does not replace the pending official-min_context=0 AoA rerun; it defines the coordinate that the corrected endpoint must pass through end to end.",
    ]
    NOTE.write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": record["status"],
        "out_json": rel(out_json),
        "note": rel(NOTE),
        "head": str(head.get("stdout", "")).strip() if isinstance(head, dict) else None,
        "strict_eval_sha": dataset_meta.get("strict_eval_sha"),
        "ewok_sha": dataset_meta.get("ewok_sha"),
        "filtered_matches_local": filtered_matches_local,
        "constants_match_raw": constants_match_raw,
        "constants_match_filtered": constants_match_filtered,
        "collation_probe_returncode": record["pristine_collation_probe_with_old_min20_aoa"].get("collate_run", {}).get("returncode") if isinstance(record.get("pristine_collation_probe_with_old_min20_aoa"), dict) else None,
        "elapsed_sec": record["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
