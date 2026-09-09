#!/usr/bin/env python3
"""Official-row-count local AoA runner for BabyLM MLM checkpoint ladders.

This is a research repair of the historical local helper. The current local
`collate_preds.py` expects AOA_SIZE=8005 predictions per checkpoint, and
`cdi_childes.json` yields 8005 contexts only with `min_context=0`. The older
helper hardcoded `min_context=20` and produced 6560 predictions per checkpoint,
which is useful for some past local comparisons but cannot settle current
official compatibility.

This script trains nothing and uses no AoA information as a pretraining signal. It
only reads an existing `hf_model/chck_*M` ladder and computes the official AoA
surprisal/curve-fitness record with an explicit min_context.
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
from typing import Any

WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
STUDY = _public_path('experiments/archive/frontier_consolidation')
USER_ROOT = _public_path('.')
ROOT_INITIAL_MODEL_STUDIES = _public_path('experiments/archive/initial_model_studies')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
sys.path.insert(0, str(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')))


def _ensure_preimport_writable_hf_cache() -> None:
    """Protect trust_remote_code imports from a read-only default HF cache.

    Transformers reads its dynamic-module cache location at import time.  research
    found that setting HF_MODULES_CACHE only inside main() was too late for custom
    local checkpoints: AutoModelForMaskedLM tried to write under the runtime's
    read-only shared model cache and AoA produced zero usable surprisal rows.  If
    an outer wrapper already supplied writable cache locations we preserve them;
    otherwise we install a local writable default before importing
    Transformers.
    """
    cache_root = _public_path('data/external/hf_preimport_cache_revision_023')
    mapping = {
        "HF_HOME": cache_root / "hf_home",
        "HF_HUB_CACHE": cache_root / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "HF_MODULES_CACHE": cache_root / "modules",
        "HF_DATASETS_CACHE": cache_root / "datasets",
        "TMPDIR": cache_root / "tmp",
    }

    def usable(key: str) -> bool:
        val = os.environ.get(key)
        if not val or "$QIUSHI_MODELS_ROOT" in val:
            return False
        try:
            p = pathlib.Path(val)
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".qiushi_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    for key, path in mapping.items():
        if not usable(key):
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_ensure_preimport_writable_hf_cache()

import torch  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402
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
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "HF_DATASETS_CACHE", "TMPDIR"]:
        pathlib.Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


class LocalCheckpointSurprisalExtractor(StepSurprisalExtractor):
    """Official AoA extractor with local checkpoint-directory step resolution."""

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
    raise RuntimeError(f"Could not load tokenizer from {model_root} or checkpoint children")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", required=True, help="hf_model directory containing chck_*M children")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_note", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--min_context", type=int, default=0)
    ap.add_argument("--expected_rows_per_step", type=int, default=8005)
    args = ap.parse_args()

    t0 = time.time()
    model_root = pathlib.Path(args.model_root).resolve()
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
    target_words, contexts = load_eval(word_path, args.min_context, False)
    cfg = StepConfig(resume=False, track="strict-small", file_path=None, debug=False)
    expected_steps = list(cfg.steps)
    start_record = {
        "event": "start",
        "device": device,
        "model_root": str(model_root),
        "steps": expected_steps,
        "word_counts": cfg.word_counts,
        "word_path": str(word_path),
        "min_context": args.min_context,
        "target_words": len(target_words),
        "contexts": sum(len(x) for x in contexts),
        "expected_rows_per_step": args.expected_rows_per_step,
        "gpu": args.gpu,
        "cache_root": str(cache_root),
        "strict_root": str(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')),
    }
    log.write_text(json.dumps(start_record, indent=2) + "\n", encoding="utf-8")

    extractor = LocalCheckpointSurprisalExtractor(config=cfg, model_name=str(model_root), backend="mlm", device=device)
    results_data = extractor.analyze_steps(contexts=contexts, target_words=target_words, resume_path=None)

    result_dir = out_dir / "hf_model_local_ckpts" / "main" / "zero_shot" / "mlm" / "AoA_word"
    result_dir.mkdir(parents=True, exist_ok=True)
    surprisal_path = result_dir / "surprisal.json"
    JsonProcessor.save_json(results_data, surprisal_path)

    tokenizer, score_tokenizer_path = load_score_tokenizer(model_root)
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
        if not math.isfinite(sv):
            finite_surprisals = False
        step_mean[st] = step_mean.get(st, 0.0) + sv
    step_mean = {k: step_mean[k] / step_counts[k] for k in step_counts}
    missing_steps = [s for s in expected_steps if s not in step_counts]
    unexpected_steps = sorted(set(step_counts) - set(expected_steps))
    row_count_values = sorted(set(step_counts.values()))
    expected_total_rows = len(expected_steps) * args.expected_rows_per_step
    validation_ok = (
        not missing_steps
        and not unexpected_steps
        and finite_surprisals
        and row_count_values == [args.expected_rows_per_step]
        and len(rows) == expected_total_rows
    )
    if not validation_ok:
        payload_error = {
            "error": "aoa_step_or_row_count_validation_failed",
            "missing_steps": missing_steps,
            "unexpected_steps": unexpected_steps,
            "row_count_values": row_count_values,
            "expected_rows_per_step": args.expected_rows_per_step,
            "num_rows": len(rows),
            "expected_total_rows": expected_total_rows,
            "finite_surprisals": finite_surprisals,
        }
        (out_dir / "validation_failed.json").write_text(json.dumps(payload_error, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise RuntimeError(payload_error)

    payload: dict[str, Any] = {
        "status": "AOA_LOCAL_CKPTS_MINCTX_DONE",
        "model_root": str(model_root),
        "backend": "mlm",
        "track_name": "strict-small",
        "word_path": str(word_path),
        "cdi_human": str(cdi_human),
        "output_dir": str(out_dir),
        "score_path": str(score_path),
        "surprisal_path": str(surprisal_path),
        "score_tokenizer_path": score_tokenizer_path,
        "min_context": args.min_context,
        "expected_rows_per_step": args.expected_rows_per_step,
        "aoa": float(score),
        "curve_fitness_record": curve_fitness_record,
        "num_rows": len(rows),
        "num_steps": len(step_counts),
        "step_counts": step_counts,
        "expected_steps": expected_steps,
        "missing_steps": missing_steps,
        "unexpected_steps": unexpected_steps,
        "row_count_values": row_count_values,
        "finite_surprisals": finite_surprisals,
        "step_mean_surprisal": step_mean,
        "elapsed_sec": time.time() - t0,
        "interpretation": "Local direct-checkpoint AoA runner with explicit min_context. For current official collation compatibility, min_context=0 gives 8005 predictions per checkpoint.",
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_note.write_text("\n".join([
        "# research AoA local checkpoints with explicit min_context",
        "",
        f"Evidence JSON: `{out_json}`",
        f"Surprisal JSON: `{surprisal_path}`",
        f"Score JSON: `{score_path}`",
        "",
        f"min_context: **{args.min_context}**",
        f"AoA curve_fitness: **{float(score):.6f}**",
        f"Rows: {len(rows)} across {len(step_counts)} checkpoints; row_count_values={row_count_values}",
        "",
        "This run loads local `hf_model/chck_*M` directories directly and uses explicit AoA row-count settings. For current `collate_preds.py`, official-row-count compatibility is 8005 predictions/checkpoint.",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "AOA_LOCAL_CKPTS_MINCTX_DONE",
        "aoa": float(score),
        "out": str(out_json),
        "rows": len(rows),
        "steps": len(step_counts),
        "row_count_values": row_count_values,
        "min_context": args.min_context,
        "elapsed_sec": time.time() - t0,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
