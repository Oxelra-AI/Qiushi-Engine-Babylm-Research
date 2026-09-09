#!/usr/bin/env python3
"""Parameterized local-checkpoint AoA runner for BabyLM MLM models.

Loads model_root/chck_*M directly for every official strict-small AoA step and
preserves the official AoA target words, surprisal extraction, and curve-fitness
scoring. This generalizes the research protected-DeBERTa runner to new checkpoint
ladders such as wwm_seed43.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, time

import torch
from transformers import AutoModelForMaskedLM, AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
sys.path.insert(0, str(STRICT.resolve()))

from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, StepConfig, load_eval  # noqa: E402
from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor  # noqa: E402
from evaluation_pipeline.utils import AoAEvaluator  # noqa: E402


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
    """Official AoA extractor with local directory step resolution."""
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


def load_score_tokenizer(model_root: pathlib.Path):
    candidates = [model_root, model_root / 'chck_100M', model_root / 'chck_80M', model_root / 'chck_10M']
    for p in candidates:
        if p.exists():
            try:
                return AutoTokenizer.from_pretrained(p, trust_remote_code=True)
            except Exception:
                try:
                    return PreTrainedTokenizerFast.from_pretrained(p, padding_side='right')
                except Exception:
                    pass
    raise RuntimeError(f'Could not load tokenizer from {model_root} or known checkpoint children')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_root', required=True, help='hf_model directory containing chck_*M children')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--out_note', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    setup_env()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    t0 = time.time()
    model_root = pathlib.Path(args.model_root).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_json = pathlib.Path(args.out_json)
    out_note = pathlib.Path(args.out_note)
    log = pathlib.Path(args.log)
    word_path = (STRICT / 'evaluation_data/full_eval/aoa/cdi_childes.json').resolve()
    cdi_human = (STRICT / 'evaluation_data/full_eval/aoa/cdi_human.csv').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_note.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_words, contexts = load_eval(word_path, 20, False)
    cfg = StepConfig(resume=False, track='strict-small', file_path=None, debug=False)
    start_record = {
        'event': 'start', 'device': device, 'model_root': str(model_root),
        'steps': cfg.steps, 'word_counts': cfg.word_counts,
        'target_words': len(target_words), 'contexts': len(contexts), 'gpu': args.gpu,
    }
    log.write_text(json.dumps(start_record, indent=2) + '\n', encoding='utf-8')
    extractor = LocalCheckpointSurprisalExtractor(config=cfg, model_name=str(model_root), backend='mlm', device=device)
    results_data = extractor.analyze_steps(contexts=contexts, target_words=target_words, resume_path=None)

    result_dir = out_dir / 'hf_model_local_ckpts' / 'main' / 'zero_shot' / 'mlm' / 'AoA_word'
    result_dir.mkdir(parents=True, exist_ok=True)
    surprisal_path = result_dir / 'surprisal.json'
    JsonProcessor.save_json(results_data, surprisal_path)
    tokenizer = load_score_tokenizer(model_root)
    score = AoAEvaluator(cdi_human).compute_curve_fitness(results_data, tokenizer)['curve_fitness']
    score_path = result_dir / 'aoa_score.json'
    JsonProcessor.save_json({'aoa': score}, score_path)

    rows = results_data.get('results', [])
    step_counts = {}
    step_mean = {}
    for r in rows:
        st = r['step']
        step_counts[st] = step_counts.get(st, 0) + 1
        step_mean[st] = step_mean.get(st, 0.0) + float(r['surprisal'])
    step_mean = {k: step_mean[k] / step_counts[k] for k in step_counts}
    payload = {
        'status': 'AOA_LOCAL_CKPTS_DONE',
        'model_root': str(model_root), 'backend': 'mlm', 'track_name': 'strict-small',
        'word_path': str(word_path), 'cdi_human': str(cdi_human), 'output_dir': str(out_dir),
        'score_path': str(score_path), 'surprisal_path': str(surprisal_path),
        'aoa': float(score), 'num_rows': len(rows), 'num_steps': len(step_counts),
        'step_counts': step_counts, 'step_mean_surprisal': step_mean,
        'elapsed_sec': time.time() - t0,
        'interpretation': 'Loads local model_root/chck_*M directories directly while preserving official AoA scoring.',
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    out_note.write_text('\n'.join([
        '# research — AoA with direct local checkpoint loading', '',
        f'Evidence JSON: `{out_json}`',
        f'Surprisal JSON: `{surprisal_path}`',
        f'Score JSON: `{score_path}`', '',
        f'AoA: **{float(score):.4f}**',
        f'Rows: {len(rows)} across {len(step_counts)} checkpoints', '',
        'This run loads local `hf_model/chck_*M` directories directly and preserves official AoA target words, surprisal computation, and curve-fitness scoring.',
    ]) + '\n', encoding='utf-8')
    print(json.dumps({'status':'AOA_LOCAL_CKPTS_DONE','aoa':float(score),'out':str(out_json),'rows':len(rows),'steps':len(step_counts),'elapsed_sec':time.time()-t0}, indent=2))


if __name__ == '__main__':
    main()
