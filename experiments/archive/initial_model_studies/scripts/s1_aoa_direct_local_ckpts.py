#!/usr/bin/env python3
"""research: direct-local-checkpoint AoA for S1 12x384 100M.

Uses the official BabyLM AoA target words, contexts, surprisal extraction, and
curve-fitness scoring, but overrides local checkpoint resolution so that
`hf_model/chck_*M` directories are loaded directly. This avoids the known
Transformers local `revision=` artifact.
"""
from __future__ import annotations
import json, os, pathlib, sys, time
import torch
from transformers import AutoModelForMaskedLM, AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
sys.path.insert(0, str(STRICT.resolve()))
from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, StepConfig, load_eval  # noqa: E402
from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor  # noqa: E402
from evaluation_pipeline.utils import AoAEvaluator  # noqa: E402

MODEL_ROOT = (ROOT / 'training/runs/babylm_leadershape_s1_100M_aligned_micro128/hf_model').resolve()
OUTDIR = (ROOT / 'training/runs/babylm_leadershape_s1_100M_aligned_micro128/eval_results_step328_aoa_local_ckpts').resolve()
WORD_PATH = (STRICT / 'evaluation_data/full_eval/aoa/cdi_childes.json').resolve()
CDI_HUMAN = (STRICT / 'evaluation_data/full_eval/aoa/cdi_human.csv').resolve()
OUT_JSON = ROOT / 'data/s1_100m_aoa_direct_local_ckpts.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/s1_100m_aoa_direct_local_ckpts.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/s1_100m_aoa_direct_local_ckpts.log')


def setup_env() -> None:
    hf_home = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf_home.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf_home / 'hub').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf_home / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    for k in ['HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_MODULES_CACHE']:
        pathlib.Path(os.environ[k]).mkdir(parents=True, exist_ok=True)


class LocalCheckpointSurprisalExtractor(StepSurprisalExtractor):
    def _step_path(self, step: str) -> pathlib.Path:
        p = pathlib.Path(self.model_name) / str(step)
        if not p.exists():
            raise FileNotFoundError(f'Missing local checkpoint path: {p}')
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
            processor = AutoProcessor.from_pretrained(p, trust_remote_code=True, padding_side='right')
        except (ValueError, KeyError):
            processor = PreTrainedTokenizerFast.from_pretrained(p, padding_side='right')
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        return processor, tokenizer


def summarize_steps(rows):
    counts, sums = {}, {}
    for r in rows:
        st = r['step']
        counts[st] = counts.get(st, 0) + 1
        sums[st] = sums.get(st, 0.0) + float(r['surprisal'])
    return counts, {k: sums[k] / counts[k] for k in counts}


def main() -> None:
    setup_env()
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_words, contexts = load_eval(WORD_PATH, 20, False)
    cfg = StepConfig(resume=False, track='strict-small', file_path=None, debug=False)
    missing = [s for s in cfg.steps if not (MODEL_ROOT / s).exists()]
    if missing:
        raise RuntimeError(f'S1 missing checkpoints: {missing}')
    LOG.write_text(json.dumps({
        'event': 'start', 'device': device, 'model_root': str(MODEL_ROOT),
        'steps': cfg.steps, 'word_counts': cfg.word_counts,
        'target_words': len(target_words), 'contexts': len(contexts),
        'word_path': str(WORD_PATH), 'cdi_human': str(CDI_HUMAN)
    }, indent=2) + '\n', encoding='utf-8')

    result_dir = OUTDIR / 'hf_model_local_ckpts' / 'main' / 'zero_shot' / 'mlm' / 'AoA_word'
    score_path = result_dir / 'aoa_score.json'
    surprisal_path = result_dir / 'surprisal.json'
    if score_path.exists() and surprisal_path.exists():
        score_data = json.loads(score_path.read_text())
        results_data = json.loads(surprisal_path.read_text())
        score = float(score_data.get('aoa', score_data.get('curve_fitness')))
    else:
        extractor = LocalCheckpointSurprisalExtractor(config=cfg, model_name=str(MODEL_ROOT), backend='mlm', device=device)
        results_data = extractor.analyze_steps(contexts=contexts, target_words=target_words, resume_path=None)
        result_dir.mkdir(parents=True, exist_ok=True)
        JsonProcessor.save_json(results_data, surprisal_path)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT, trust_remote_code=True)
        score = AoAEvaluator(CDI_HUMAN).compute_curve_fitness(results_data, tokenizer)['curve_fitness']
        JsonProcessor.save_json({'aoa': score}, score_path)

    rows = results_data.get('results', [])
    step_counts, step_mean = summarize_steps(rows)
    payload = {
        'status': 'S1_AOA_DIRECT_LOCAL_CKPTS_DONE',
        'model_root': str(MODEL_ROOT),
        'backend': 'mlm',
        'track_name': 'strict-small',
        'word_path': str(WORD_PATH),
        'cdi_human': str(CDI_HUMAN),
        'output_dir': str(OUTDIR),
        'score_path': str(score_path),
        'surprisal_path': str(surprisal_path),
        'aoa': float(score),
        'num_rows': len(rows),
        'num_steps': len(step_counts),
        'step_counts': step_counts,
        'step_mean_surprisal': step_mean,
        'elapsed_sec': time.time() - t0,
        'interpretation': 'Loads local model_root/chck_*M directories directly, avoiding ignored local revision=step behavior while preserving official AoA scoring logic.'
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    NOTE.write_text('\n'.join([
        '# research — S1 100M AoA with direct local checkpoint loading', '',
        f'Evidence JSON: `{OUT_JSON}`',
        f'Surprisal JSON: `{surprisal_path}`',
        f'Score JSON: `{score_path}`', '',
        f'AoA: **{float(score):.4f}**',
        f'Rows: {len(rows)} across {len(step_counts)} checkpoints', '',
        'This run loads `hf_model/chck_*M` directories directly and preserves the official AoA target words, surprisal extraction, and curve-fitness scoring.'
    ]) + '\n', encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'aoa': float(score), 'out': str(OUT_JSON), 'rows': len(rows), 'steps': len(step_counts), 'elapsed_sec': payload['elapsed_sec']}, indent=2))


if __name__ == '__main__':
    main()
