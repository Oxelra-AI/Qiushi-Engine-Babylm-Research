#!/usr/bin/env python3
"""research: Early-stop AoA measurement with reusable shared ancestry.

The BabyLM strict-small README allows early stopping and says code assuming the
maximum training budget should be edited. Coherent86 and both dense candidates
share the same true ancestral checkpoints through 80M; only the final endpoints
differ. This script therefore computes the 17 common ancestral checkpoint
surprisals once, computes candidate endpoints separately, and assembles a
legitimate trajectory per endpoint for AoA scoring.

It writes evidence that distinguishes a measured AoA score of exactly 0.0 from
missing-checkpoint placeholder zero: a candidate is measured only when the
surprisal evidence exists for all required steps, the official AoAEvaluator has
run, and an aoa_score.json/manifest records the scoring status.

Examples:
  # Complete path smoke test on a few words and two ancestor points + endpoint
  python experiments/archive/functional_learning/scripts/shared_aoa.py smoke --target coherent86 --max-words 3 --ancestral-limit 2 --gpu 0

  # Full shared ancestry once
  python experiments/archive/functional_learning/scripts/shared_aoa.py shared --gpu 0

  # Endpoint only, then assemble/score with existing shared ancestry
  python experiments/archive/functional_learning/scripts/shared_aoa.py endpoint --target dense_seed62064 --gpu 1
  python experiments/archive/functional_learning/scripts/shared_aoa.py assemble --target dense_seed62064
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Set a writable HF/cache root before importing transformers.
A01 = Path("experiments/archive/functional_learning")
OUT_ROOT = A01 / "data/shared_aoa"
HF_CACHE = Path(os.environ.get("HF_CACHE", str(OUT_ROOT / "hf_cache")))
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE.resolve())
os.environ["TRANSFORMERS_CACHE"] = str((HF_CACHE / "hub").resolve())
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForMaskedLM, AutoProcessor, AutoTokenizer

BABYLM_STRICT = Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict")
if str(BABYLM_STRICT) not in sys.path:
    sys.path.insert(0, str(BABYLM_STRICT))

from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, load_eval
from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor
from evaluation_pipeline.utils import AoAEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("shared_aoa")

A02 = Path("experiments/archive/frontier_consolidation")
LADDER = A02 / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model"
CDI_WORDS_PATH = BABYLM_STRICT / "evaluation_data/full_eval/aoa/cdi_childes.json"
CDI_HUMAN_PATH = BABYLM_STRICT / "evaluation_data/full_eval/aoa/cdi_human.csv"
PLATFORM_AOA_ESTIMATOR_ID = "current_official_main_6f825c2_AoAEvaluator"
PLATFORM_AOA_VARIANT = "current_official_main_6f825c2"
PLATFORM_AOA_PROVENANCE = A01 / "data/aoa_estimator_provenance/aoa_estimator_platform_provenance.json"
PLATFORM_LEADERBOARD_SNAPSHOT = A01 / "data/leaderboard_current_snapshot/leaderboard_current_snapshot.json"

EARLY_STOP_NAMES = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i*10}M" for i in range(1, 9)]
EARLY_STOP_WORDS = [i * 1_000_000 for i in range(1, 10)] + [10 * i * 1_000_000 for i in range(1, 9)]

ENDPOINTS: dict[str, dict[str, Any]] = {
    "coherent86": {
        "path": A01 / "data/automodel_repair/repaired_coherent86_alpha075",
        "words": 86_005_295,
        "zero_reading_payload": A02 / "data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json",
        "repaired_superglue_payload": A01 / "data/repaired_coherent86_eval/per_target/repaired_coherent86_alpha075.json",
    },
    "dense_seed62064": {
        "path": A01 / "data/automodel_repair/repaired_dense_seed62064_u0080",
        "words": 89_168_037,
        "zero_reading_payload": A01 / "data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json",
        "repaired_superglue_payload": A01 / "data/repaired_dense_eval_seed62064/per_target/repaired_dense_seed62064_u0080.json",
    },
    "dense_seed62065": {
        "path": A01 / "data/automodel_repair/repaired_dense_seed62065_u0080",
        "words": 89_168_037,
        "zero_reading_payload": A01 / "data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json",
        "repaired_superglue_payload": A01 / "data/repaired_dense_eval_seed62065/per_target/repaired_dense_seed62065_u0080.json",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return str(path)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ckpt_weight_path(path: Path) -> Path | None:
    for name in ["model.safetensors", "pytorch_model.bin"]:
        p = path / name
        if p.is_file():
            return p
    return None


def checkpoint_info(label: str, path: Path) -> dict[str, Any]:
    cfg_path = path / "config.json"
    cfg: dict[str, Any] = {}
    if cfg_path.is_file():
        with cfg_path.open() as f:
            cfg = json.load(f)
    w = ckpt_weight_path(path)
    return {
        "label": label,
        "path": rel(path),
        "exists": path.is_dir(),
        "config_sha256": sha256_file(cfg_path),
        "weight_file": w.name if w else None,
        "weight_sha256": sha256_file(w) if w else None,
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "auto_map": cfg.get("auto_map"),
        "adapter_scale": cfg.get("adapter_scale"),
        "private_adapter_scale": cfg.get("private_adapter_scale"),
    }


class PathStepExtractor(StepSurprisalExtractor):
    """Official StepSurprisalExtractor but checkpoints come from explicit paths."""

    def __init__(self, checkpoint_map: dict[str, Path], steps: list[str], word_counts: list[int], backend: str, device: str, model_name: str):
        self.model_name = model_name
        self.model_cache_dir = None
        self.checkpoint_map = checkpoint_map
        self.backend = backend
        self.config = type("Config", (), {"steps": steps, "word_counts": word_counts})()
        self.device = device
        self.current_step = None
        logger.info("PathStepExtractor: %d checkpoints on %s", len(steps), device)

    def load_model_for_step(self, step: str):
        ckpt_path = self.checkpoint_map[step]
        logger.info("Loading %s from %s", step, ckpt_path)
        if self.backend in ["mlm", "mntp"]:
            model = AutoModelForMaskedLM.from_pretrained(str(ckpt_path), trust_remote_code=True)
        else:
            raise ValueError(f"Unsupported backend for AoA evaluation: {self.backend}")
        model = model.to(self.device)
        model.eval()
        return model

    def load_tokenizer_for_step(self, step: str):
        ckpt_path = self.checkpoint_map[step]
        processor = AutoProcessor.from_pretrained(str(ckpt_path), trust_remote_code=True, padding_side="right")
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        return processor, tokenizer


def select_eval_subset(max_words: int = 0) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    target_words, contexts = load_eval(CDI_WORDS_PATH, min_context=0, debug=False)
    total_words = len(target_words)
    total_contexts = sum(len(c) for c in contexts)
    if max_words and max_words > 0:
        target_words = target_words[:max_words]
        contexts = contexts[:max_words]
    return target_words, contexts, {
        "cdi_words_path": rel(CDI_WORDS_PATH),
        "cdi_words_sha256": sha256_file(CDI_WORDS_PATH),
        "cdi_human_path": rel(CDI_HUMAN_PATH),
        "cdi_human_sha256": sha256_file(CDI_HUMAN_PATH),
        "total_words_available": total_words,
        "total_contexts_available": total_contexts,
        "words_evaluated": len(target_words),
        "contexts_evaluated": sum(len(c) for c in contexts),
        "max_words_limit": max_words,
        "target_words_preview": target_words[:10],
    }


def expected_result_count(n_steps: int, contexts: list[list[str]]) -> int:
    return n_steps * sum(len(c) for c in contexts)


def result_summary(results_data: dict[str, Any], expected_steps: list[str] | None = None, expected_contexts: int | None = None) -> dict[str, Any]:
    results = results_data.get("results", [])
    by_step = Counter(r.get("step") for r in results)
    nan_count = 0
    finite_count = 0
    for r in results:
        try:
            v = float(r.get("surprisal"))
            if math.isfinite(v):
                finite_count += 1
            else:
                nan_count += 1
        except Exception:
            nan_count += 1
    missing_steps: list[str] = []
    wrong_counts: dict[str, int] = {}
    if expected_steps is not None:
        missing_steps = [s for s in expected_steps if by_step.get(s, 0) == 0]
        if expected_contexts is not None:
            wrong_counts = {s: by_step.get(s, 0) for s in expected_steps if by_step.get(s, 0) != expected_contexts}
    return {
        "n_results": len(results),
        "n_finite": finite_count,
        "n_nan_or_nonfinite": nan_count,
        "n_steps_observed": len(by_step),
        "counts_by_step": dict(by_step),
        "missing_steps": missing_steps,
        "steps_with_wrong_counts": wrong_counts,
        "first_result": results[0] if results else None,
        "last_result": results[-1] if results else None,
    }


def run_extraction(checkpoint_map: dict[str, Path], steps: list[str], word_counts: list[int], out_dir: Path, gpu: int, max_words: int = 0, use_bos_only: bool = False, model_name: str = "path_step_model") -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    target_words, contexts, eval_info = select_eval_subset(max_words=max_words)
    extractor = PathStepExtractor(checkpoint_map, steps, word_counts, backend="mlm", device=device, model_name=model_name)
    t0 = time.time()
    results_data = extractor.analyze_steps(contexts=contexts, target_words=target_words, use_bos_only=use_bos_only)
    elapsed = time.time() - t0
    surp_path = out_dir / "surprisal.json"
    JsonProcessor.save_json(results_data, surp_path)
    summary = result_summary(results_data, steps, sum(len(c) for c in contexts))
    manifest = {
        "status": "EXTRACTION_COMPLETE",
        "created_utc": utc_now(),
        "elapsed_sec": elapsed,
        "device": device,
        "use_bos_only": use_bos_only,
        "steps": steps,
        "word_counts": word_counts,
        "n_steps": len(steps),
        "expected_results": expected_result_count(len(steps), contexts),
        "surprisal_path": rel(surp_path),
        "eval_info": eval_info,
        "summary": summary,
        "checkpoint_provenance": {s: checkpoint_info(s, checkpoint_map[s]) for s in steps},
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    # The base class sometimes leaves CUDA memory until process exit; clear aggressively.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Extraction complete: %s (%d results, %.1fs)", surp_path, summary["n_results"], elapsed)
    return manifest


def score_trajectory(results_data: dict[str, Any], tokenizer_path: Path, out_dir: Path, target_words: list[str] | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    evaluator = AoAEvaluator(CDI_HUMAN_PATH)
    try:
        fitness = evaluator.compute_curve_fitness(results_data, tokenizer, target_words=target_words)
        raw = float(fitness.get("curve_fitness", 0.0))
        leaderboard = raw * 100.0
        scoring_status = "measured"  # includes legitimate zero caused by significance rule or few valid words
        error = None
    except Exception as e:
        fitness = {}
        raw = None
        leaderboard = None
        scoring_status = "scoring_failed"
        error = repr(e)
    elapsed = time.time() - t0
    # Official run.py writes raw correlation under key 'aoa'. We also write a richer manifest.
    aoa_score_path = out_dir / "aoa_score.json"
    if raw is not None:
        JsonProcessor.save_json({"aoa": raw}, aoa_score_path)
    score_manifest = {
        "status": "AOA_SCORING_COMPLETE" if scoring_status == "measured" else "AOA_SCORING_FAILED",
        "created_utc": utc_now(),
        "aoa_estimator": platform_aoa_evidence(),
        "scoring_status": scoring_status,
        "measured": scoring_status == "measured",
        "aoa_raw_correlation": raw,
        "aoa_leaderboard_score": leaderboard,
        "aoa_score_json": rel(aoa_score_path) if raw is not None else None,
        "legitimate_zero": bool(scoring_status == "measured" and leaderboard == 0.0),
        "zero_interpretation": (
            "A measured zero is the official evaluator output after completed surprisal extraction and curve scoring; it is not a missing-checkpoint placeholder."
            if scoring_status == "measured" and leaderboard == 0.0 else None
        ),
        "elapsed_sec": elapsed,
        "error": error,
        "fitness_summary": {k: v for k, v in fitness.items() if k not in {"model_aoas", "child_aoas", "valid_words", "monthly_scores"}},
        "n_valid_words": len(fitness.get("valid_words", [])) if isinstance(fitness, dict) else None,
        "valid_words_preview": fitness.get("valid_words", [])[:20] if isinstance(fitness, dict) else None,
    }
    # Add full arrays to JSON evidence, but keep md summaries compact.
    if isinstance(fitness, dict):
        for k in ["valid_words", "model_aoas", "child_aoas", "monthly_scores"]:
            if k in fitness:
                score_manifest[k] = fitness[k]
    with (out_dir / "aoa_score_manifest.json").open("w") as f:
        json.dump(score_manifest, f, indent=2, default=str)
    return score_manifest


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def platform_aoa_evidence() -> dict[str, Any]:
    """Attributable scorer provenance for assembled AoA manifests."""
    prov: dict[str, Any] | None = None
    snap: dict[str, Any] | None = None
    if PLATFORM_AOA_PROVENANCE.is_file():
        try:
            prov = load_json(PLATFORM_AOA_PROVENANCE)
        except Exception as e:
            prov = {"load_error": repr(e)}
    if PLATFORM_LEADERBOARD_SNAPSHOT.is_file():
        try:
            snap = load_json(PLATFORM_LEADERBOARD_SNAPSHOT)
        except Exception as e:
            snap = {"load_error": repr(e)}
    return {
        "estimator_id": PLATFORM_AOA_ESTIMATOR_ID,
        "variant_name": PLATFORM_AOA_VARIANT,
        "participant_eval_commit": "6f825c291e2c4c78ad33b1935fd64d45f52642dc",
        "participant_utils_sha256": (prov or {}).get("file_identities", {}).get("utils_py", {}).get("sha256"),
        "participant_archive_equals_local": (prov or {}).get("archive_identity", {}).get("archive_equals_local"),
        "provenance_path": rel(PLATFORM_AOA_PROVENANCE),
        "provenance_sha256": sha256_file(PLATFORM_AOA_PROVENANCE),
        "leaderboard_snapshot_path": rel(PLATFORM_LEADERBOARD_SNAPSHOT),
        "leaderboard_snapshot_sha256": sha256_file(PLATFORM_LEADERBOARD_SNAPSHOT),
        "leaderboard_feature_flags": (snap or {}).get("aoa_feature_flags"),
        "frontier_use": "This is the platform-matching participant-side AoA scorer for the BabyLM frontier coordinate unless a newer platform scorer is explicitly identified.",
    }


def shared_paths(max_words: int = 0, ancestral_limit: int = 0) -> Path:
    suffix = "full" if not max_words and not ancestral_limit else f"test_w{max_words or 'all'}_a{ancestral_limit or 'all'}"
    return OUT_ROOT / "shared_ancestry" / suffix


def endpoint_paths(target: str, max_words: int = 0) -> Path:
    suffix = "full" if not max_words else f"test_w{max_words}"
    return OUT_ROOT / "endpoints" / target / suffix


def assembled_paths(target: str, max_words: int = 0, ancestral_limit: int = 0) -> Path:
    suffix = "full" if not max_words and not ancestral_limit else f"test_w{max_words or 'all'}_a{ancestral_limit or 'all'}"
    return OUT_ROOT / "assembled" / target / suffix


def cmd_shared(args: argparse.Namespace) -> None:
    n = args.ancestral_limit if args.ancestral_limit and args.ancestral_limit > 0 else len(EARLY_STOP_NAMES)
    steps = EARLY_STOP_NAMES[:n]
    words = EARLY_STOP_WORDS[:n]
    ckpt_map = {s: LADDER / s for s in steps}
    out_dir = shared_paths(max_words=args.max_words, ancestral_limit=args.ancestral_limit)
    manifest = run_extraction(ckpt_map, steps, words, out_dir, gpu=args.gpu, max_words=args.max_words, use_bos_only=args.use_bos_only, model_name="shared_ancestry")
    print(json.dumps({
        "status": "shared_complete",
        "out_dir": rel(out_dir),
        "n_results": manifest["summary"]["n_results"],
        "expected_results": manifest["expected_results"],
        "missing_steps": manifest["summary"]["missing_steps"],
        "wrong_counts": manifest["summary"]["steps_with_wrong_counts"],
    }, indent=2))


def endpoint_step_name(target: str) -> str:
    """Parseable final-step label carrying the actual endpoint exposure."""
    words = int(ENDPOINTS[target]["words"])
    return f"endpoint_{words / 1_000_000:.6f}M"


def cmd_endpoint(args: argparse.Namespace) -> None:
    target = args.target
    ep = ENDPOINTS[target]
    ep_step = endpoint_step_name(target)
    ckpt_map = {ep_step: ep["path"]}
    steps = [ep_step]
    words = [int(ep["words"])]
    out_dir = endpoint_paths(target, max_words=args.max_words)
    manifest = run_extraction(ckpt_map, steps, words, out_dir, gpu=args.gpu, max_words=args.max_words, use_bos_only=args.use_bos_only, model_name=f"endpoint_{target}")
    print(json.dumps({
        "status": "endpoint_complete",
        "target": target,
        "endpoint_step": ep_step,
        "out_dir": rel(out_dir),
        "n_results": manifest["summary"]["n_results"],
        "expected_results": manifest["expected_results"],
        "missing_steps": manifest["summary"]["missing_steps"],
        "wrong_counts": manifest["summary"]["steps_with_wrong_counts"],
    }, indent=2))


def assemble_and_score(target: str, max_words: int = 0, ancestral_limit: int = 0) -> dict[str, Any]:
    ep = ENDPOINTS[target]
    n = ancestral_limit if ancestral_limit and ancestral_limit > 0 else len(EARLY_STOP_NAMES)
    shared_dir = shared_paths(max_words=max_words, ancestral_limit=ancestral_limit)
    endpoint_dir = endpoint_paths(target, max_words=max_words)
    shared_surp = shared_dir / "surprisal.json"
    endpoint_surp = endpoint_dir / "surprisal.json"
    if not shared_surp.is_file():
        raise FileNotFoundError(f"Shared ancestry surprisals not found: {shared_surp}")
    if not endpoint_surp.is_file():
        raise FileNotFoundError(f"Endpoint surprisals not found: {endpoint_surp}")

    shared_data = load_json(shared_surp)
    endpoint_data = load_json(endpoint_surp)
    shared_steps = EARLY_STOP_NAMES[:n]
    shared_words = EARLY_STOP_WORDS[:n]
    final_step = endpoint_step_name(target)
    all_steps = shared_steps + [final_step]
    all_words = shared_words + [int(ep["words"])]
    results = list(shared_data.get("results", [])) + list(endpoint_data.get("results", []))
    # Ensure endpoint entries carry the parseable actual-exposure step label and word count.
    # Older test files used step='final'; normalize them so official AoAEvaluator can parse the endpoint.
    for r in results:
        if r.get("step") == "final" or r.get("step") == final_step:
            r["step"] = final_step
            r["word_count"] = int(ep["words"])

    assembled = {
        "metadata": {
            "model_name": target,
            "use_bos_only": bool(load_json(shared_dir / "manifest.json").get("use_bos_only", False)),
            "total_steps": len(all_steps),
            "completed_steps": len(set(r.get("step") for r in results)),
            "early_stop_convention": "BabyLM strict README supports checkpoint ladder only up to amount trained; true milestones through 80M plus exact final endpoint.",
            "shared_ancestry_source": rel(shared_surp),
            "endpoint_source": rel(endpoint_surp),
        },
        "results": results,
    }

    out_dir = assembled_paths(target, max_words=max_words, ancestral_limit=ancestral_limit)
    aoa_word_dir = out_dir / "AoA_word"
    aoa_word_dir.mkdir(parents=True, exist_ok=True)
    assembled_surp = aoa_word_dir / "surprisal.json"
    JsonProcessor.save_json(assembled, assembled_surp)

    # Score with final tokenizer and official AoAEvaluator.
    score = score_trajectory(assembled, ep["path"], aoa_word_dir)
    target_words, contexts, eval_info = select_eval_subset(max_words=max_words)
    summary = result_summary(assembled, all_steps, sum(len(c) for c in contexts))
    complete = (
        summary["missing_steps"] == []
        and summary["steps_with_wrong_counts"] == {}
        and summary["n_results"] == expected_result_count(len(all_steps), contexts)
        and score.get("measured") is True
    )
    manifest = {
        "status": "ASSEMBLED_AOA_MEASURED" if complete else "ASSEMBLED_AOA_INCOMPLETE",
        "created_utc": utc_now(),
        "target": target,
        "max_words": max_words,
        "ancestral_limit": ancestral_limit,
        "early_stop_convention": "17 ancestral checkpoints chck_1M..chck_80M for ~86M/~89M endpoints, plus exact final endpoint; full 19-step max-budget ladder is not borrowed.",
        "shared_ancestry": {
            "steps": shared_steps,
            "word_counts": shared_words,
            "surprisal_path": rel(shared_surp),
            "manifest_path": rel(shared_dir / "manifest.json"),
        },
        "endpoint": {
            "target": target,
            "step": final_step,
            "word_count": int(ep["words"]),
            "checkpoint_path": rel(ep["path"]),
            "surprisal_path": rel(endpoint_surp),
            "manifest_path": rel(endpoint_dir / "manifest.json"),
        },
        "assembled": {
            "steps": all_steps,
            "word_counts": all_words,
            "surprisal_path": rel(assembled_surp),
            "expected_results": expected_result_count(len(all_steps), contexts),
            "summary": summary,
        },
        "eval_info": eval_info,
        "aoa_estimator": platform_aoa_evidence(),
        "score": score,
        "complete_measured_evidence": complete,
        "aoa_payload_for_comparison": {
            "column": "AoA",
            "estimator_id": PLATFORM_AOA_ESTIMATOR_ID,
            "estimator_variant": PLATFORM_AOA_VARIANT,
            "estimator_provenance_path": rel(PLATFORM_AOA_PROVENANCE),
            "leaderboard_snapshot_path": rel(PLATFORM_LEADERBOARD_SNAPSHOT),
            "status": "measured" if score.get("measured") else "not_measured",
            "measured": bool(score.get("measured")),
            "aoa_raw_correlation": score.get("aoa_raw_correlation"),
            "aoa_leaderboard_score": score.get("aoa_leaderboard_score"),
            "legitimate_zero": bool(score.get("legitimate_zero")),
            "surprisal_json": rel(assembled_surp),
            "aoa_score_json": score.get("aoa_score_json"),
            "aoa_score_manifest": rel(aoa_word_dir / "aoa_score_manifest.json"),
            "n_results": summary["n_results"],
            "n_steps": len(all_steps),
            "n_words_evaluated": eval_info["words_evaluated"],
            "n_contexts_evaluated": eval_info["contexts_evaluated"],
            "interpretation": "AoA value is measured because surprisal extraction and official curve scoring completed; value may be exactly zero due to the evaluator significance rule and remains distinct from missing-checkpoint placeholder zero.",
        },
    }
    with (out_dir / "aoa_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    write_md_summary(out_dir / "aoa_manifest.md", manifest)
    return manifest


def write_md_summary(path: Path, manifest: dict[str, Any]) -> None:
    score = manifest.get("score", {})
    assembled = manifest.get("assembled", {})
    summary = assembled.get("summary", {})
    lines = []
    lines.append(f"# research AoA measurement: {manifest.get('target')}\n")
    lines.append(f"Created: {manifest.get('created_utc')}\n\n")
    lines.append(f"Status: `{manifest.get('status')}`\n\n")
    lines.append("## Trajectory\n\n")
    lines.append(f"- Early-stop convention: {manifest.get('early_stop_convention')}\n")
    lines.append(f"- Shared ancestry steps: {', '.join(manifest.get('shared_ancestry', {}).get('steps', []))}\n")
    lines.append(f"- Endpoint word count: `{manifest.get('endpoint', {}).get('word_count')}`\n")
    lines.append(f"- Assembled surprisal: `{assembled.get('surprisal_path')}`\n\n")
    lines.append("## Extraction evidence\n\n")
    lines.append(f"- Results: `{summary.get('n_results')}`\n")
    lines.append(f"- Finite: `{summary.get('n_finite')}`; nonfinite: `{summary.get('n_nan_or_nonfinite')}`\n")
    lines.append(f"- Missing steps: `{summary.get('missing_steps')}`\n")
    lines.append(f"- Steps with wrong counts: `{summary.get('steps_with_wrong_counts')}`\n\n")
    lines.append("## AoA scoring\n\n")
    lines.append(f"- Estimator: `{manifest.get('aoa_estimator', {}).get('estimator_id')}`\n")
    lines.append(f"- Estimator provenance: `{manifest.get('aoa_estimator', {}).get('provenance_path')}`\n")
    lines.append(f"- Measured: `{score.get('measured')}`\n")
    lines.append(f"- Raw correlation: `{score.get('aoa_raw_correlation')}`\n")
    lines.append(f"- Leaderboard score: `{score.get('aoa_leaderboard_score')}`\n")
    lines.append(f"- Legitimate measured zero: `{score.get('legitimate_zero')}`\n")
    lines.append(f"- Valid words: `{score.get('n_valid_words')}`\n\n")
    if score.get("legitimate_zero"):
        lines.append("The zero is produced by completed extraction plus official curve scoring; it is not a missing-checkpoint placeholder.\n")
    path.write_text("".join(lines))


def cmd_assemble(args: argparse.Namespace) -> None:
    manifest = assemble_and_score(args.target, max_words=args.max_words, ancestral_limit=args.ancestral_limit)
    print(json.dumps({
        "status": manifest["status"],
        "target": args.target,
        "complete_measured_evidence": manifest["complete_measured_evidence"],
        "aoa_raw_correlation": manifest["score"].get("aoa_raw_correlation"),
        "aoa_leaderboard_score": manifest["score"].get("aoa_leaderboard_score"),
        "legitimate_zero": manifest["score"].get("legitimate_zero"),
        "manifest": rel(assembled_paths(args.target, args.max_words, args.ancestral_limit) / "aoa_manifest.json"),
    }, indent=2, default=str))


def cmd_smoke(args: argparse.Namespace) -> None:
    # Run shared subset + endpoint subset + assembly through the complete path.
    if not args.max_words or args.max_words <= 0:
        args.max_words = 3
    if not args.ancestral_limit or args.ancestral_limit <= 0:
        args.ancestral_limit = 2
    logger.info("Smoke test: target=%s max_words=%s ancestral_limit=%s", args.target, args.max_words, args.ancestral_limit)
    cmd_shared(args)
    cmd_endpoint(args)
    manifest = assemble_and_score(args.target, max_words=args.max_words, ancestral_limit=args.ancestral_limit)
    print(json.dumps({
        "status": "smoke_complete",
        "target": args.target,
        "assembled_status": manifest["status"],
        "complete_measured_evidence": manifest["complete_measured_evidence"],
        "n_results": manifest["assembled"]["summary"]["n_results"],
        "expected_results": manifest["assembled"]["expected_results"],
        "aoa_raw_correlation": manifest["score"].get("aoa_raw_correlation"),
        "aoa_leaderboard_score": manifest["score"].get("aoa_leaderboard_score"),
        "legitimate_zero": manifest["score"].get("legitimate_zero"),
        "manifest": rel(assembled_paths(args.target, args.max_words, args.ancestral_limit) / "aoa_manifest.json"),
    }, indent=2, default=str))


def cmd_status(args: argparse.Namespace) -> None:
    rows = []
    full_shared = shared_paths()
    rows.append({"kind": "shared_full", "path": rel(full_shared), "surprisal": (full_shared / "surprisal.json").is_file(), "manifest": (full_shared / "manifest.json").is_file()})
    for target in ENDPOINTS:
        epd = endpoint_paths(target)
        asd = assembled_paths(target)
        row = {
            "kind": "target",
            "target": target,
            "endpoint_surprisal": (epd / "surprisal.json").is_file(),
            "endpoint_manifest": (epd / "manifest.json").is_file(),
            "assembled_surprisal": (asd / "AoA_word/surprisal.json").is_file(),
            "assembled_manifest": (asd / "aoa_manifest.json").is_file(),
            "aoa_score": (asd / "AoA_word/aoa_score.json").is_file(),
        }
        if (asd / "aoa_manifest.json").is_file():
            man = load_json(asd / "aoa_manifest.json")
            row["aoa_leaderboard_score"] = man.get("score", {}).get("aoa_leaderboard_score")
            row["complete_measured_evidence"] = man.get("complete_measured_evidence")
        rows.append(row)
    print(json.dumps({"status": "aoa_status", "rows": rows}, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="research shared-ancestry AoA runner")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["shared", "endpoint", "assemble", "smoke", "status"]:
        p = sub.add_parser(name)
        p.add_argument("--target", choices=list(ENDPOINTS.keys()), default="coherent86")
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--max-words", type=int, default=0)
        p.add_argument("--ancestral-limit", type=int, default=0)
        p.add_argument("--use-bos-only", action="store_true")
    args = parser.parse_args()
    if args.cmd == "shared":
        cmd_shared(args)
    elif args.cmd == "endpoint":
        cmd_endpoint(args)
    elif args.cmd == "assemble":
        cmd_assemble(args)
    elif args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "status":
        cmd_status(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
