#!/usr/bin/env python3
"""research: official-row-count AoA for the truthful alpha0.75 Strict-Small ladder.

This is bounded evaluation work. It trains nothing and does not alter model weights.
It loads the research/208 truthful 19-revision alpha0.75 model tree, computes AoA
surprisal using the official BabyLM Strict AoA pipeline at min_context=0, validates
19 checkpoints x 8005 rows, and stages `surprisal.json` and `aoa_score.json` under
the same `collate_fast/results/hf_model/main/zero_shot/mlm/AoA_word/` layout used
by the unmodified official collator.
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


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/representation_and_objectives"
STUDY = USER_ROOT / "experiments/archive/representation_and_objectives"
STRICT = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict"
DEFAULT_MODEL_ROOT = WORKSPACE / "data/alpha075_fast_preflight_v2/hf_model_truthful_alpha075"
DEFAULT_BASE_OUT = WORKSPACE / "data/alpha075_aoa_minctx0"


def setup_preimport_cache(cache_root: pathlib.Path, gpu: int) -> None:
    cache_root = cache_root.resolve()
    mapping = {
        "HF_HOME": cache_root / "hf_home",
        "HF_HUB_CACHE": cache_root / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "HF_MODULES_CACHE": cache_root / "hf_modules_cache",
        "HF_DATASETS_CACHE": cache_root / "hf_datasets",
        "TMPDIR": cache_root / "tmp",
        "NLTK_DATA": USER_ROOT / "experiments/archive/initial_model_studies/data/nltk_data",
    }
    for key, path in mapping.items():
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(pathlib.Path(path).resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["PYTHONPATH"] = str(STRICT.resolve()) + os.pathsep + str((STRICT / "evaluation_pipeline").resolve()) + os.pathsep + os.environ.get("PYTHONPATH", "")


# Install writable caches before importing transformers; custom remote-code modules
# are cached at import/load time and must not target a read-only shared cache.
_PREIMPORT_GPU = int(os.environ.get("AOA_GPU", "0"))
setup_preimport_cache(STUDY / "staging/alpha075_aoa_preimport_cache", _PREIMPORT_GPU)
sys.path.insert(0, str(STRICT.resolve()))

import torch  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402
from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, StepConfig, load_eval  # noqa: E402
from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor  # noqa: E402
from evaluation_pipeline.utils import AoAEvaluator  # noqa: E402


FAST_REVISIONS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]


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
    import hashlib
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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
    candidates = [model_root, model_root / "chck_100M", model_root / "chck_90M", model_root / "chck_80M", model_root / "chck_10M"]
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


def write_json(p: pathlib.Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def model_identity(model_root: pathlib.Path) -> dict[str, Any]:
    out = {}
    for ckpt in FAST_REVISIONS:
        p = model_root / ckpt
        cfg = p / "config.json"
        weights = p / "model.safetensors"
        rec: dict[str, Any] = {
            "path": rel(p),
            "exists": p.exists(),
            "is_symlink": p.is_symlink(),
            "model_sha256": sha256_file(weights),
            "config_sha256": sha256_file(cfg),
        }
        if cfg.exists():
            try:
                c = json.loads(cfg.read_text(encoding="utf-8"))
                rec.update({
                    "architectures": c.get("architectures"),
                    "auto_map": c.get("auto_map"),
                    "adapter_scale": c.get("adapter_scale"),
                    "private_adapter_scale": c.get("private_adapter_scale"),
                    "private_adapter_enabled": c.get("private_adapter_enabled"),
                    "vocab_size": c.get("vocab_size"),
                })
            except Exception as exc:
                rec["config_read_error"] = repr(exc)
        out[ckpt] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT), help="hf_model directory containing chck_*M children")
    ap.add_argument("--base-out", default=str(DEFAULT_BASE_OUT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--min-context", type=int, default=0)
    ap.add_argument("--expected-rows-per-step", type=int, default=8005)
    args = ap.parse_args()

    os.environ["AOA_GPU"] = str(args.gpu)
    base_out = pathlib.Path(args.base_out)
    model_root = pathlib.Path(args.model_root)
    if not model_root.is_absolute():
        model_root = USER_ROOT / model_root
    if not base_out.is_absolute():
        base_out = USER_ROOT / base_out
    base_out.mkdir(parents=True, exist_ok=True)
    setup_preimport_cache(base_out / "runtime_cache", args.gpu)

    t0 = time.time()
    word_path = (STRICT / "evaluation_data/full_eval/aoa/cdi_childes.json").resolve()
    cdi_human = (STRICT / "evaluation_data/full_eval/aoa/cdi_human.csv").resolve()
    target_words, contexts = load_eval(word_path, args.min_context, False)
    cfg = StepConfig(resume=False, track="strict-small", file_path=None, debug=False)
    expected_steps = list(cfg.steps)

    missing_models = [s for s in expected_steps if not (model_root / s / "model.safetensors").exists()]
    if missing_models:
        out = {
            "status": "ALPHA075_AOA_PREFLIGHT_NEEDS_REPAIR",
            "created_utc": now(),
            "model_root": rel(model_root),
            "missing_models": missing_models,
            "expected_steps": expected_steps,
        }
        write_json(base_out / "alpha075_aoa_minctx0_summary.json", out)
        raise SystemExit(1)

    start_record = {
        "event": "start",
        "created_utc": now(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": args.gpu,
        "model_root": rel(model_root),
        "strict_root": rel(STRICT),
        "word_path": rel(word_path),
        "cdi_human": rel(cdi_human),
        "min_context": args.min_context,
        "target_words": len(target_words),
        "contexts": sum(len(x) for x in contexts),
        "expected_rows_per_step": args.expected_rows_per_step,
        "steps": expected_steps,
        "word_counts": cfg.word_counts,
    }
    write_json(base_out / "alpha075_aoa_start_record.json", start_record)

    extractor = LocalCheckpointSurprisalExtractor(config=cfg, model_name=str(model_root.resolve()), backend="mlm", device=start_record["device"])
    results_data = extractor.analyze_steps(contexts=contexts, target_words=target_words, resume_path=None)

    result_dir = base_out / "collate_fast" / "results" / "hf_model" / "main" / "zero_shot" / "mlm" / "AoA_word"
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

    payload: dict[str, Any] = {
        "status": "ALPHA075_AOA_MINCTX0_DONE" if validation_ok else "ALPHA075_AOA_MINCTX0_NEEDS_REPAIR",
        "created_utc": now(),
        "purpose": {
            "question": "Measure the truthful official-compatible AoA column for coherent86 alpha0.75 without substituting model states or scalar AoA zero.",
            "minimum_cost_action": "Run the official AoA surprisal/curve-fit computation once over the truthful 19-checkpoint alpha0.75 tree at min_context=0.",
            "decision_use": "If 19x8005 finite rows pass, this AoA score can replace the placeholder AoA=0 in final official collation; otherwise alpha0.75 remains incomplete.",
        },
        "model_root": rel(model_root),
        "backend": "mlm",
        "track_name": "strict-small",
        "strict_root": rel(STRICT),
        "word_path": rel(word_path),
        "cdi_human": rel(cdi_human),
        "output_dir": rel(base_out),
        "result_dir": rel(result_dir),
        "score_path": rel(score_path),
        "surprisal_path": rel(surprisal_path),
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
        "model_identities": model_identity(model_root),
        "surprisal_sha256": sha256_file(surprisal_path),
        "score_sha256": sha256_file(score_path),
        "elapsed_sec": round(time.time() - t0, 3),
        "interpretation": "Official-row-count local AoA runner with explicit min_context=0. It loads local chck_*M directories directly and stages AoA under collate_fast/results/hf_model/main/zero_shot/mlm/AoA_word.",
    }
    write_json(base_out / "alpha075_aoa_minctx0_summary.json", payload)
    note = [
        "# research alpha0.75 AoA min_context=0",
        "",
        f"Status: `{payload['status']}`",
        f"AoA curve_fitness: **{score:.9f}**",
        f"Rows: {len(rows)} across {len(step_counts)} checkpoints; row_count_values={row_count_values}",
        f"Score file: `{rel(score_path)}`",
        f"Surprisal file: `{rel(surprisal_path)}`",
        "",
        "This is evaluation-only. It replaces neither model state nor checkpoint history; it measures the staged truthful alpha0.75 ladder.",
        f"JSON: `{rel(base_out / 'alpha075_aoa_minctx0_summary.json')}`",
    ]
    (base_out / "alpha075_aoa_minctx0_summary.md").write_text("\n".join(note) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "aoa": float(score),
        "rows": len(rows),
        "steps": len(step_counts),
        "row_count_values": row_count_values,
        "score_path": rel(score_path),
        "surprisal_path": rel(surprisal_path),
        "summary": rel(base_out / "alpha075_aoa_minctx0_summary.json"),
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2), flush=True)
    if not validation_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
