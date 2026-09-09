#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import torch
from transformers import AutoModelForMaskedLM, AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
sys.path.insert(0, str(STRICT.resolve()))

from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, StepConfig, load_eval  # noqa: E402
from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor  # noqa: E402
from evaluation_pipeline.utils import AoAEvaluator  # noqa: E402

MODEL_ROOT = (ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model").resolve()
OUTDIR = (ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/eval_results_aoa_local_ckpts").resolve()
WORD_PATH = (STRICT / "evaluation_data/full_eval/aoa/cdi_childes.json").resolve()
CDI_HUMAN = (STRICT / "evaluation_data/full_eval/aoa/cdi_human.csv").resolve()
OUT_JSON = ROOT / "data/debertav2_b256_aoa_local_ckpts_result.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_b256_aoa_local_ckpts_result.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_b256_aoa_local_ckpts.log')


def setup_env() -> None:
    hf_home = ROOT / "training/hf_home"
    os.environ["HF_HOME"] = str(hf_home.resolve())
    os.environ["HF_HUB_CACHE"] = str((hf_home / "hub").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((hf_home / "transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((ROOT / "training/hf_modules_cache").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(os.environ[k]).mkdir(parents=True, exist_ok=True)


class LocalCheckpointSurprisalExtractor(StepSurprisalExtractor):
    """Official AoA extractor with local directory step resolution.

    The upstream code calls from_pretrained(model_root, revision='chck_*M'), which is
    ignored for plain local directories. This subclass loads model_root/chck_*M
    directly, while preserving the official compute_surprisal and AoA scoring logic.
    """

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
        except (ValueError, KeyError):
            processor = PreTrainedTokenizerFast.from_pretrained(p, padding_side="right")
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        return processor, tokenizer


def main() -> None:
    setup_env()
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_words, contexts = load_eval(WORD_PATH, 20, False)
    cfg = StepConfig(resume=False, track="strict-small", file_path=None, debug=False)
    extractor = LocalCheckpointSurprisalExtractor(config=cfg, model_name=str(MODEL_ROOT), backend="mlm", device=device)
    LOG.write_text(json.dumps({"event":"start", "device":device, "model_root":str(MODEL_ROOT), "steps":cfg.steps, "word_counts":cfg.word_counts, "target_words":len(target_words), "contexts":len(contexts)}, indent=2) + "\n", encoding="utf-8")
    results_data = extractor.analyze_steps(contexts=contexts, target_words=target_words, resume_path=None)
    result_dir = OUTDIR / "hf_model_local_ckpts" / "main" / "zero_shot" / "mlm" / "AoA_word"
    result_dir.mkdir(parents=True, exist_ok=True)
    surprisal_path = result_dir / "surprisal.json"
    JsonProcessor.save_json(results_data, surprisal_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT, trust_remote_code=True)
    score = AoAEvaluator(CDI_HUMAN).compute_curve_fitness(results_data, tokenizer)["curve_fitness"]
    score_path = result_dir / "aoa_score.json"
    JsonProcessor.save_json({"aoa": score}, score_path)
    # Compact trajectory sanity checks: mean surprisal by step for first few words and all rows.
    rows = results_data.get("results", [])
    step_counts = {}
    step_mean = {}
    for r in rows:
        st = r["step"]
        step_counts[st] = step_counts.get(st, 0) + 1
        step_mean[st] = step_mean.get(st, 0.0) + float(r["surprisal"])
    step_mean = {k: step_mean[k] / step_counts[k] for k in step_counts}
    payload = {
        "status": "AOA_LOCAL_CKPTS_DONE",
        "model_root": str(MODEL_ROOT),
        "backend": "mlm",
        "track_name": "strict-small",
        "word_path": str(WORD_PATH),
        "cdi_human": str(CDI_HUMAN),
        "output_dir": str(OUTDIR),
        "score_path": str(score_path),
        "surprisal_path": str(surprisal_path),
        "aoa": float(score),
        "num_rows": len(rows),
        "num_steps": len(step_counts),
        "step_counts": step_counts,
        "step_mean_surprisal": step_mean,
        "elapsed_sec": time.time() - t0,
        "interpretation": "AoA run loads local model_root/chck_*M directories directly, avoiding ignored revision=step on local directories.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE.write_text("\n".join([
        "# research — DeBERTa-v2 b256 AoA with direct local checkpoint loading",
        "",
        f"Evidence JSON: `{OUT_JSON}`",
        f"Surprisal JSON: `{surprisal_path}`",
        f"Score JSON: `{score_path}`",
        "",
        f"AoA: **{float(score):.4f}**",
        f"Rows: {len(rows)} across {len(step_counts)} checkpoints",
        "",
        "This run overrides only checkpoint resolution: it loads `hf_model/chck_*M` directories directly and preserves official AoA target words, surprisal computation, and curve-fitness scoring.",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"status":"AOA_LOCAL_CKPTS_DONE", "aoa":float(score), "out":str(OUT_JSON), "rows":len(rows), "steps":len(step_counts), "elapsed_sec":time.time()-t0}, indent=2))


if __name__ == "__main__":
    main()
