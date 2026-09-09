#!/usr/bin/env python3
"""research: OFFICIAL-PATH AoA for compact_view_reinvest seed43022.

Purpose
-------
The research full-eval AoA was produced by the inherited helper
`aoa_local_ckpts_for_model.py`, which called
`load_eval(word_path, 20, False)` -> min_context=20. That filter keeps only the
328 CDI words with >=20 contexts, giving 6560 (word,context) surprisal rows per
checkpoint. The OFFICIAL BabyLM strict-small pipeline (evaluation_pipeline/
AoA_word/run.py default `--min_context 0`, and collate_preds.AOA_SIZE=8005)
uses ALL 504 words with 8005 contexts.

Confirmed by counting the official cdi_childes.json:
  min_context=0  -> 504 words, 8005 contexts   (== official AOA_SIZE)
  min_context=20 -> 328 words, 6560 contexts   (== our recorded per-ckpt count)

So the research AoA=0.0 was computed on a NON-OFFICIAL word subset and gated to
0.0 by the p>0.1 rule over only 203 valid words. This script reruns the AoA
surprisal + curve fitness through the UNMODIFIED official extractor and scoring,
changing ONLY min_context to 0 (official), reusing the already-trained
seed43022 checkpoint ladder. It does NOT retrain and does NOT modify the corpus.

This is the decisive test of whether Overall 42.0868 survives the official AoA
path: with 504 words the correlation may pass the p-gate and yield a positive
(raising Overall) or negative (lowering Overall) AoA, or remain 0.0.
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
import sys
import time

WORKSPACE = _public_path('experiments/archive/representation_and_objectives')  # .../representation_and_objectives/workspace
STUDY = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
ROOT_INITIAL_MODEL_STUDIES = _public_path('experiments/archive/initial_model_studies')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
sys.path.insert(0, str(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')))

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForMaskedLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, StepConfig, load_eval  # noqa: E402
from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor  # noqa: E402
from evaluation_pipeline.utils import AoAEvaluator  # noqa: E402


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
    """Official AoA extractor with local directory step resolution (identical to research)."""

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


def load_score_tokenizer(model_root: pathlib.Path):
    candidates = [model_root, model_root / "chck_100M", model_root / "chck_80M", model_root / "chck_10M"]
    for p in candidates:
        if p.exists():
            try:
                return AutoTokenizer.from_pretrained(p, trust_remote_code=True), str(p.resolve())
            except Exception:
                try:
                    return PreTrainedTokenizerFast.from_pretrained(p, padding_side="right"), str(p.resolve())
                except Exception:
                    pass
    raise RuntimeError(f"Could not load tokenizer from {model_root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--min_context", type=int, default=0, help="OFFICIAL default is 0")
    args = ap.parse_args()

    t0 = time.time()
    model_root = pathlib.Path(args.model_root).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "runtime_cache"
    setup_env(cache_root, args.gpu)

    word_path = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json')
    cdi_human = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_words, contexts = load_eval(word_path, args.min_context, False)
    total_contexts = sum(len(c) for c in contexts)
    cfg = StepConfig(resume=False, track="strict-small", file_path=None, debug=False)

    print(json.dumps({
        "event": "start",
        "device": device,
        "min_context": args.min_context,
        "target_words": len(target_words),
        "total_contexts": total_contexts,
        "steps": len(cfg.steps),
        "model_root": str(model_root),
    }), flush=True)

    extractor = LocalCheckpointSurprisalExtractor(
        config=cfg, model_name=str(model_root), backend="mlm", device=device,
    )
    results_data = extractor.analyze_steps(contexts=contexts, target_words=target_words, resume_path=None)

    result_dir = out_dir / "hf_model_local_ckpts" / "main" / "zero_shot" / "mlm" / "AoA_word"
    result_dir.mkdir(parents=True, exist_ok=True)
    surprisal_path = result_dir / "surprisal.json"
    JsonProcessor.save_json(results_data, surprisal_path)

    tokenizer, score_tokenizer_path = load_score_tokenizer(model_root)
    curve_fitness_record = AoAEvaluator(cdi_human).compute_curve_fitness(results_data, tokenizer)
    score = float(curve_fitness_record.get("curve_fitness", 0.0))
    score_path = result_dir / "aoa_score.json"
    # Trim large arrays out of the saved score for readability but keep key stats.
    saved_record = {k: v for k, v in curve_fitness_record.items()
                    if k not in ("valid_words", "model_aoas", "child_aoas", "monthly_scores")}
    JsonProcessor.save_json({"aoa": score, "curve_fitness_record": saved_record}, score_path)

    rows = results_data.get("results", [])
    step_counts: dict[str, int] = {}
    finite = True
    for r in rows:
        st = str(r["step"])
        step_counts[st] = step_counts.get(st, 0) + 1
        if not math.isfinite(float(r["surprisal"])):
            finite = False
    row_count_values = sorted(set(step_counts.values()))

    payload = {
        "status": "OFFICIAL_AOA_MIN0_DONE",
        "model_root": str(model_root),
        "min_context": args.min_context,
        "word_path": str(word_path),
        "cdi_human": str(cdi_human),
        "target_words_loaded": len(target_words),
        "total_contexts_loaded": total_contexts,
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
        "interpretation": (
            "Official-path AoA (min_context=0, all 504 CDI words, 8005 contexts) reusing the "
            "already-trained seed43022 checkpoint ladder. Compare aoa_leaderboard_score against the "
            "research recorded AoA=0.0 to determine whether the official AoA raises, lowers, or preserves "
            "the Overall 42.0868 endpoint."
        ),
    }
    out_json = out_dir / "official_aoa_min0_reinvest.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "OFFICIAL_AOA_MIN0_DONE",
        "aoa_raw_correlation": score,
        "aoa_leaderboard_score": score * 100.0,
        "n_words": saved_record.get("n_words"),
        "p_value": saved_record.get("p_value"),
        "row_count_values": row_count_values,
        "num_rows": len(rows),
        "out": str(out_json),
        "elapsed_sec": time.time() - t0,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
