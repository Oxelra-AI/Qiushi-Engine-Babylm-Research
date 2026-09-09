#!/usr/bin/env python3
"""research: official min_context=0 AoA for compact_view_reinvest seed43122.

The inherited full-evaluation runner used by research imports the COMPACT_EXPERIENCE runner. That
runner's AoA helper loads `load_eval(word_path, 20, False)`, giving the old 6,560
context rows/checkpoint subset. research/036 showed the current strict-small official
coordinate uses min_context=0: 504 CDI words and 8,005 contexts per checkpoint.

This wrapper reuses the already-trained seed43122 checkpoint ladder and computes only
that official AoA coordinate. It does not retrain and does not change corpus data.
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
from pathlib import Path
import sys
import time
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/representation_and_objectives"
PRISTINE_STRICT = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict"
FALLBACK_STRICT = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict"
STRICT = PRISTINE_STRICT if PRISTINE_STRICT.exists() else FALLBACK_STRICT
sys.path.insert(0, str(STRICT.resolve()))

import torch  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402
from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, StepConfig, load_eval  # noqa: E402
from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor  # noqa: E402
from evaluation_pipeline.utils import AoAEvaluator  # noqa: E402

DEFAULT_MODEL_ROOT = WORKSPACE / "training/runs/repl_compact_view_reinvest_seed43122/hf_model"
DEFAULT_OUT_DIR = WORKSPACE / "data/official_aoa_min0_seed43122"
REQUIRED_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{10*i}M" for i in range(1, 11)]


def setup_env(cache_root: Path, gpu: int) -> None:
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
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "HF_DATASETS_CACHE", "TMPDIR"]:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


class LocalCheckpointSurprisalExtractor(StepSurprisalExtractor):
    """Official AoA extractor with local directory step resolution."""

    def _step_path(self, step: str) -> Path:
        p = Path(self.model_name) / str(step)
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


def load_score_tokenizer(model_root: Path):
    for p in [model_root, model_root / "chck_100M", model_root / "chck_80M", model_root / "chck_10M"]:
        if not p.exists():
            continue
        try:
            return AutoTokenizer.from_pretrained(p, trust_remote_code=True), str(p.resolve())
        except Exception:
            try:
                return PreTrainedTokenizerFast.from_pretrained(p, padding_side="right"), str(p.resolve())
            except Exception:
                pass
    raise RuntimeError(f"Could not load tokenizer from {model_root}")


def count_loaded_contexts(word_path: Path, min_context: int) -> dict[str, Any]:
    target_words, contexts = load_eval(word_path, min_context, False)
    return {
        "min_context": min_context,
        "target_words_loaded": len(target_words),
        "total_contexts_loaded": sum(len(c) for c in contexts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", type=Path, default=DEFAULT_MODEL_ROOT)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--min_context", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="Only verify paths and loaded AoA row counts; do not load model checkpoints.")
    args = ap.parse_args()

    t0 = time.time()
    model_root = args.model_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "runtime_cache"
    setup_env(cache_root, args.gpu)

    word_path = (STRICT / "evaluation_data/full_eval/aoa/cdi_childes.json").resolve()
    cdi_human = (STRICT / "evaluation_data/full_eval/aoa/cdi_human.csv").resolve()
    if not STRICT.exists():
        raise FileNotFoundError(STRICT)
    if not model_root.exists():
        raise FileNotFoundError(model_root)
    missing_steps = [s for s in REQUIRED_STEPS if not (model_root / s).exists()]
    loaded = count_loaded_contexts(word_path, args.min_context)
    loaded20 = count_loaded_contexts(word_path, 20)

    if args.dry_run:
        payload = {
            "status": "OFFICIAL_AOA_MIN0_SEED43122_DRYRUN",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "strict_repo": str(STRICT),
            "model_root": str(model_root),
            "word_path": str(word_path),
            "cdi_human": str(cdi_human),
            "required_steps": REQUIRED_STEPS,
            "missing_required_steps": missing_steps,
            "min_context0_loaded": loaded,
            "min_context20_loaded": loaded20,
            "elapsed_sec": time.time() - t0,
            "interpretation": "Dry run only. Official strict-small AoA coordinate requires min_context=0 with 8005 contexts per checkpoint; inherited helper's min_context=20 yields 6560.",
        }
        out_json = out_dir / "official_aoa_min0_seed43122_dryrun.json"
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "missing_required_steps": missing_steps,
            "min_context0": loaded,
            "min_context20": loaded20,
            "out_json": str(out_json),
        }, indent=2), flush=True)
        return

    if missing_steps:
        raise RuntimeError({"error": "missing_required_aoa_steps", "missing_required_steps": missing_steps, "model_root": str(model_root)})
    if loaded["total_contexts_loaded"] != 8005:
        raise RuntimeError({"error": "unexpected_official_aoa_context_count", "loaded": loaded, "word_path": str(word_path)})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = StepConfig(resume=False, track="strict-small", file_path=None, debug=False)
    print(json.dumps({
        "event": "start",
        "device": device,
        "strict_repo": str(STRICT),
        "min_context": args.min_context,
        "target_words_loaded": loaded["target_words_loaded"],
        "total_contexts_loaded": loaded["total_contexts_loaded"],
        "steps": len(cfg.steps),
        "model_root": str(model_root),
    }), flush=True)

    target_words, contexts = load_eval(word_path, args.min_context, False)
    extractor = LocalCheckpointSurprisalExtractor(config=cfg, model_name=str(model_root), backend="mlm", device=device)
    results_data = extractor.analyze_steps(contexts=contexts, target_words=target_words, resume_path=None)

    result_dir = out_dir / "hf_model_local_ckpts" / "main" / "zero_shot" / "mlm" / "AoA_word"
    result_dir.mkdir(parents=True, exist_ok=True)
    surprisal_path = result_dir / "surprisal.json"
    JsonProcessor.save_json(results_data, surprisal_path)

    tokenizer, score_tokenizer_path = load_score_tokenizer(model_root)
    curve_fitness_record = AoAEvaluator(cdi_human).compute_curve_fitness(results_data, tokenizer)
    score = float(curve_fitness_record.get("curve_fitness", 0.0))
    saved_record = {k: v for k, v in curve_fitness_record.items() if k not in ("valid_words", "model_aoas", "child_aoas", "monthly_scores")}
    score_path = result_dir / "aoa_score.json"
    JsonProcessor.save_json({"aoa": score, "curve_fitness_record": saved_record}, score_path)

    rows = results_data.get("results", [])
    step_counts: dict[str, int] = {}
    finite = True
    for r in rows:
        st = str(r["step"])
        step_counts[st] = step_counts.get(st, 0) + 1
        try:
            if not math.isfinite(float(r["surprisal"])):
                finite = False
        except Exception:
            finite = False
    row_count_values = sorted(set(step_counts.values()))
    validation_errors = []
    if len(step_counts) != len(REQUIRED_STEPS):
        validation_errors.append(f"num_steps {len(step_counts)} != {len(REQUIRED_STEPS)}")
    if sorted(step_counts.keys()) != sorted(REQUIRED_STEPS):
        validation_errors.append("step names differ from strict-small required ladder")
    if row_count_values != [8005]:
        validation_errors.append(f"row_count_values {row_count_values} != [8005]")
    if not finite:
        validation_errors.append("nonfinite surprisal encountered")
    if validation_errors:
        raise RuntimeError({"error": "aoa_output_validation_failed", "validation_errors": validation_errors, "surprisal_path": str(surprisal_path)})

    payload = {
        "status": "OFFICIAL_AOA_MIN0_SEED43122_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strict_repo": str(STRICT),
        "model_root": str(model_root),
        "min_context": args.min_context,
        "word_path": str(word_path),
        "cdi_human": str(cdi_human),
        "target_words_loaded": len(target_words),
        "total_contexts_loaded": sum(len(c) for c in contexts),
        "num_rows": len(rows),
        "num_steps": len(step_counts),
        "row_count_values": row_count_values,
        "finite_surprisals": finite,
        "aoa_raw_correlation": score,
        "aoa_leaderboard_score": score * 100.0,
        "curve_fitness_record": saved_record,
        "score_tokenizer_path": score_tokenizer_path,
        "surprisal_path": str(surprisal_path),
        "score_path": str(score_path),
        "elapsed_sec": time.time() - t0,
        "interpretation": "Official-path AoA for seed43122 compact_view_reinvest using min_context=0 (504 CDI words, 8005 contexts/checkpoint) and the already-trained 19-checkpoint strict-small ladder.",
    }
    out_json = out_dir / "official_aoa_min0_seed43122.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "aoa_raw_correlation": score,
        "aoa_leaderboard_score": score * 100.0,
        "n_words": saved_record.get("n_words"),
        "p_value": saved_record.get("p_value"),
        "row_count_values": row_count_values,
        "num_rows": len(rows),
        "out_json": str(out_json),
        "surprisal_path": str(surprisal_path),
        "score_path": str(score_path),
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
