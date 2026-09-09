#!/usr/bin/env python3
"""Truncated true-prefix AoA sensitivity for a local MLM checkpoint ladder.

This is not the standard strict-small 19-point local score.  It evaluates AoA
using only checkpoints actually observed up to a selected endpoint, optionally
including nonstandard endpoints such as chck_85M or chck_95M at their own labels.
It is used to test whether an endpoint-frozen/plateaued AoA score is driven by
repeated future-coordinate copies rather than the true acquisition prefix.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass

WORKSPACE = _public_path('experiments/archive/compact_experience')
STUDY = _public_path('experiments/archive/compact_experience')
USER_ROOT = _public_path('.')
ROOT_INITIAL_MODEL_STUDIES = _public_path('experiments/archive/initial_model_studies')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
sys.path.insert(0, str(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')))

import torch  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402
from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, load_eval  # noqa: E402
from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor  # noqa: E402
from evaluation_pipeline.utils import AoAEvaluator  # noqa: E402

STANDARD_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{10*i}M" for i in range(1, 11)]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def step_m(step: str) -> int:
    m = re.fullmatch(r"chck_(\d+)M", step)
    if not m:
        raise ValueError(step)
    return int(m.group(1))


def steps_for_endpoint(endpoint: str) -> list[str]:
    em = step_m(endpoint)
    steps = [s for s in STANDARD_STEPS if step_m(s) <= em]
    if endpoint not in steps:
        steps.append(endpoint)
    steps = sorted(set(steps), key=step_m)
    return steps


@dataclass
class TruncatedStepConfig:
    steps: list[str]
    word_counts: list[int]
    resume: bool = False
    debug: bool = False


def setup_env(cache_root: pathlib.Path, gpu: int) -> None:
    cache_root = cache_root.resolve()
    hf_home = cache_root / "hf_home"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    os.environ["HF_MODULES_CACHE"] = str(cache_root / "hf_modules_cache")
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "hf_datasets")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TMPDIR"] = str(cache_root / "tmp")
    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "HF_DATASETS_CACHE", "TMPDIR"]:
        pathlib.Path(os.environ[k]).mkdir(parents=True, exist_ok=True)


class LocalCheckpointSurprisalExtractor(StepSurprisalExtractor):
    def _step_path(self, step: str) -> pathlib.Path:
        p = pathlib.Path(self.model_name) / str(step)
        if not p.exists():
            raise FileNotFoundError(f"Missing local checkpoint path: {p}")
        return p

    def load_model_for_step(self, step: str):
        p = self._step_path(step)
        model = AutoModelForMaskedLM.from_pretrained(p, trust_remote_code=True)
        model = model.to(self.device)
        model.eval()
        return model

    def load_tokenizer_for_step(self, step: str):
        p = self._step_path(step)
        try:
            processor = AutoProcessor.from_pretrained(p, trust_remote_code=True, padding_side="right")
        except (ValueError, KeyError, OSError):
            processor = PreTrainedTokenizerFast.from_pretrained(p, padding_side="right")
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        return processor, tokenizer


def load_score_tokenizer(model_root: pathlib.Path, endpoint: str):
    candidates = [model_root / endpoint, model_root / "chck_100M", model_root / "chck_90M", model_root / "chck_80M", model_root]
    for p in candidates:
        if p.exists():
            try:
                return AutoTokenizer.from_pretrained(p, trust_remote_code=True), str(p.resolve())
            except Exception:
                try:
                    return PreTrainedTokenizerFast.from_pretrained(p, padding_side="right"), str(p.resolve())
                except Exception:
                    pass
    raise RuntimeError(f"Could not load tokenizer from {model_root} or endpoint {endpoint}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", required=True, help="hf_model root containing true observed checkpoints")
    ap.add_argument("--endpoint", required=True, help="Selected endpoint, e.g. chck_90M")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_note", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    model_root = pathlib.Path(args.model_root).resolve()
    endpoint = args.endpoint
    steps = steps_for_endpoint(endpoint)
    missing = [s for s in steps if not (model_root / s).exists()]
    if missing:
        raise FileNotFoundError({"model_root": str(model_root), "missing_true_prefix_steps": missing})
    cfg = TruncatedStepConfig(steps=steps, word_counts=[step_m(s) * 1_000_000 for s in steps])

    out_dir = pathlib.Path(args.out_dir).resolve()
    out_json = pathlib.Path(args.out_json).resolve()
    out_note = pathlib.Path(args.out_note).resolve()
    log = pathlib.Path(args.log).resolve()
    cache_root = out_dir / "runtime_cache"
    setup_env(cache_root, args.gpu)

    word_path = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json')
    cdi_human = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_note.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_words, contexts = load_eval(word_path, 20, False)
    start = {"event": "start", "device": device, "model_root": str(model_root), "endpoint": endpoint, "steps": steps, "word_counts": cfg.word_counts, "target_words": len(target_words), "contexts": len(contexts), "gpu": args.gpu, "cache_root": str(cache_root), "strict_root": str(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')), "created_utc": now()}
    log.write_text(json.dumps(start, indent=2) + "\n", encoding="utf-8")

    extractor = LocalCheckpointSurprisalExtractor(config=cfg, model_name=str(model_root), backend="mlm", device=device)
    results_data = extractor.analyze_steps(contexts=contexts, target_words=target_words, resume_path=None)
    result_dir = out_dir / "hf_model_true_prefix" / "main" / "zero_shot" / "mlm" / "AoA_word"
    result_dir.mkdir(parents=True, exist_ok=True)
    surprisal_path = result_dir / "surprisal.json"
    JsonProcessor.save_json(results_data, surprisal_path)
    tokenizer, score_tokenizer_path = load_score_tokenizer(model_root, endpoint)
    curve_fitness_record = AoAEvaluator(cdi_human).compute_curve_fitness(results_data, tokenizer)
    score = float(curve_fitness_record.get("curve_fitness", 0.0))
    score_path = result_dir / "aoa_score.json"
    JsonProcessor.save_json({"aoa": score, "curve_fitness_record": curve_fitness_record}, score_path)

    rows = results_data.get("results", [])
    step_counts: dict[str, int] = {}
    step_mean: dict[str, float] = {}
    finite_surprisals = True
    for r in rows:
        st = str(r["step"])
        step_counts[st] = step_counts.get(st, 0) + 1
        sv = float(r["surprisal"])
        finite_surprisals = finite_surprisals and math.isfinite(sv)
        step_mean[st] = step_mean.get(st, 0.0) + sv
    step_mean = {k: step_mean[k] / step_counts[k] for k in step_counts}
    missing_results = [s for s in steps if s not in step_counts]
    unexpected_results = sorted(set(step_counts) - set(steps), key=lambda s: (step_m(s) if re.fullmatch(r"chck_\d+M", s) else 10**9, s))
    row_count_values = sorted(set(step_counts.values()))
    if missing_results or unexpected_results or not finite_surprisals or len(row_count_values) != 1:
        raise RuntimeError({"error": "truncated_aoa_step_coverage_or_surprisal_validation_failed", "missing_results": missing_results, "unexpected_results": unexpected_results, "row_count_values": row_count_values, "finite_surprisals": finite_surprisals})

    payload = {
        "status": "TRUNCATED_TRUE_PREFIX_AOA_DONE",
        "interpretation": "Sensitivity metric using only true observed checkpoints through the selected endpoint. It is not the default 19-point strict-small local score and should not be mixed with endpoint-frozen Overall arithmetic without explicit labeling.",
        "model_root": str(model_root),
        "endpoint": endpoint,
        "steps": steps,
        "word_counts": cfg.word_counts,
        "track_name": "strict-small-truncated-true-prefix",
        "score_path": str(score_path),
        "surprisal_path": str(surprisal_path),
        "score_tokenizer_path": score_tokenizer_path,
        "aoa_raw_correlation": score,
        "aoa_leaderboard_units_if_scaled": score * 100.0,
        "curve_fitness_record": curve_fitness_record,
        "num_rows": len(rows),
        "num_steps": len(step_counts),
        "step_counts": step_counts,
        "missing_results": missing_results,
        "unexpected_results": unexpected_results,
        "row_count_values": row_count_values,
        "finite_surprisals": finite_surprisals,
        "step_mean_surprisal": step_mean,
        "elapsed_sec": time.time() - t0,
        "non_leakage_statement": "Post-training AoA sensitivity only; no AoA/CDI item words, child curves, or downstream scores were used for training or endpoint construction.",
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_note.write_text("\n".join([
        "# research truncated true-prefix AoA sensitivity",
        "",
        f"Model root: `{model_root}`",
        f"Endpoint: `{endpoint}`",
        f"Steps: {steps}",
        f"Raw AoA correlation: **{score:.6f}** (scaled {score*100.0:.4f})",
        f"Rows: {len(rows)} across {len(step_counts)} checkpoints; row_count_values={row_count_values}",
        f"Evidence JSON: `{out_json}`",
        "",
        "This is a sensitivity metric using only the true observed training prefix, not the plateaued endpoint-frozen 19-point local score.",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "endpoint": endpoint, "steps": len(steps), "aoa_raw_correlation": score, "out": str(out_json), "rows": len(rows), "elapsed_sec": time.time() - t0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
