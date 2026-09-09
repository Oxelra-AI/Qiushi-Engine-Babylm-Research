#!/usr/bin/env python3
"""research: evaluate research legal BSM 1M three-arm screen.

Completes evaluation of existing arms only. Important attribution warning:
BSM arms have ~3x more optimizer updates than official_control because generated
binding rows are shorter and use targeted masks; coherent-official deltas are not
pure binding effects. The cleaner mechanism contrast is coherent vs swapped,
which better matches row structure, update count, and targeted masking.
"""
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time, random
from typing import Dict, Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
OUT_DIR = ROOT / 'data/legal_bsm_1m_eval'
OUT_JSON = ROOT / 'data/legal_bsm_1m_eval.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/legal_bsm_1m_eval.md')

sys.path.insert(0, str((ROOT/'scripts').resolve()))
from binding_switch_margin_pretest import (  # noqa: E402
    TRAIN_TEMPLATES, HELDOUT_TEMPLATES, build_inventories, generate_pairs, measure_binding_pairs
)
from bsm_density_persistence import NATURAL_TEMPLATES  # noqa: E402

MODELS = {
    'official_control': {
        'path': ROOT / 'training/runs/legal_bsm_1m_official_control_seed42/hf_model',
        'trained_words': 1_000_000, 'binding_words': 0, 'steps': 98,
        'note': 'pure official, standard WWM, exact 1M words'
    },
    'bsm_coherent_20pct': {
        'path': ROOT / 'training/runs/legal_bsm_1m_bsm_coherent_20pct_seed42/hf_model',
        'trained_words': 999_874, 'binding_words': 199_234, 'steps': 292,
        'note': '20% coherent binding replacement, targeted answer masks, final root slightly below 1M due no partial JSONL rows'
    },
    'bsm_swapped_20pct': {
        'path': ROOT / 'training/runs/legal_bsm_1m_bsm_swapped_20pct_seed42/hf_model',
        'trained_words': 999_989, 'binding_words': 198_869, 'steps': 291,
        'note': '20% swapped/random binding replacement, targeted answer masks, final root slightly below 1M due no partial JSONL rows'
    },
}

ZERO_SHOT_TASKS = [
    ('blimp_fast', 'blimp', 'evaluation_data/fast_eval/blimp_fast'),
    ('supplement_fast', 'blimp', 'evaluation_data/fast_eval/supplement_fast'),
    ('entity_tracking_fast', 'entity_tracking', 'evaluation_data/fast_eval/entity_tracking_fast'),
    ('ewok_fast', 'ewok', 'evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast'),
    ('global_piqa_parallel', 'global_piqa_parallel', 'evaluation_data/fast_eval/global_piqa_parallel'),
    ('global_piqa_nonparallel', 'global_piqa_nonparallel', 'evaluation_data/fast_eval/global_piqa_nonparallel'),
]


def setup_env():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def parse_sentence_score(stdout: str):
    # First line convention: temperature<TAB>score, e.g. "1.0\t22.40"
    for line in stdout.splitlines():
        s = line.strip()
        m = re.match(r'^(?:[0-9.]+)\s+([+-]?[0-9]+(?:\.[0-9]+)?)$', s)
        if m:
            return float(m.group(1))
    m = re.search(r'AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)', stdout)
    return float(m.group(1)) if m else None


def parse_reading(stdout: str):
    out = {}
    for label, key in [('EYE TRACKING SCORE', 'reading_eye_tracking'), ('SELF-PACED READING SCORE', 'reading_self_paced')]:
        m = re.search(re.escape(label) + r':\s*([+-]?[0-9]+(?:\.[0-9]+)?)', stdout)
        if m:
            out[key] = float(m.group(1))
    if len(out) == 2:
        out['reading_mean'] = (out['reading_eye_tracking'] + out['reading_self_paced']) / 2.0
    return out


def run_sentence_eval(model_name: str, model_path: pathlib.Path, task_name: str, task_type: str, data_path: str):
    log_path = OUT_DIR / 'logs' / f'{model_name}_{task_name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'evaluation_pipeline.sentence_zero_shot.run',
        '--model_path_or_name', str(model_path.resolve()),
        '--backend', 'mlm', '--task', task_type, '--data_path', data_path,
        '--revision_name', f'step338_{model_name}_{task_name}', '--save_predictions'
    ]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=os.environ.copy(), capture_output=True, text=True, timeout=900)
    log_path.write_text(r.stdout + '\n--- STDERR ---\n' + r.stderr, encoding='utf-8')
    return {'score': parse_sentence_score(r.stdout), 'returncode': r.returncode, 'elapsed_sec': time.time()-t0,
            'log': str(log_path), 'stdout_head': r.stdout[:500], 'stderr_tail': r.stderr[-1000:], 'cmd': cmd}


def run_reading(model_name: str, model_path: pathlib.Path):
    log_path = OUT_DIR / 'logs' / f'{model_name}_reading.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'evaluation_pipeline.reading.run',
        '--model_path_or_name', str(model_path.resolve()), '--backend', 'mlm',
        '--data_path', 'evaluation_data/fast_eval/reading/reading_data.csv',
        '--revision_name', f'step338_{model_name}_reading'
    ]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=os.environ.copy(), capture_output=True, text=True, timeout=900)
    log_path.write_text(r.stdout + '\n--- STDERR ---\n' + r.stderr, encoding='utf-8')
    return {'scores': parse_reading(r.stdout), 'returncode': r.returncode, 'elapsed_sec': time.time()-t0,
            'log': str(log_path), 'stdout_head': r.stdout[:500], 'stderr_tail': r.stderr[-1000:], 'cmd': cmd}


def make_probe_sets(tokenizer, seed=42, n=80):
    inv = build_inventories(tokenizer)
    rng = random.Random(seed)
    return {
        'train': generate_pairs({'entities': inv['train_entities'], 'values': inv['train_values']}, TRAIN_TEMPLATES, 'train', rng, n_pairs=n),
        'heldout_entities': generate_pairs({'entities': inv['heldout_entities'], 'values': inv['train_values']}, TRAIN_TEMPLATES, 'heldout_entities', rng, n_pairs=n),
        'heldout_values': generate_pairs({'entities': inv['train_entities'], 'values': inv['heldout_values']}, TRAIN_TEMPLATES, 'heldout_values', rng, n_pairs=n),
        'heldout_templates': generate_pairs({'entities': inv['train_entities'], 'values': inv['train_values']}, HELDOUT_TEMPLATES, 'heldout_templates', rng, n_pairs=n),
        'heldout_order_flip': generate_pairs({'entities': inv['train_entities'], 'values': inv['train_values']}, TRAIN_TEMPLATES, 'heldout_order_flip', rng, n_pairs=n, flip_order=True),
        'natural_templates': generate_pairs({'entities': inv['train_entities'], 'values': inv['train_values']}, NATURAL_TEMPLATES, 'natural_templates', rng, n_pairs=n),
    }


def run_binding_probes(model_name: str, model_path: pathlib.Path, device):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path.resolve()), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_path.resolve()), trust_remote_code=True).to(device)
    probes = make_probe_sets(tokenizer)
    out = {}
    for split, pairs in probes.items():
        agg, _rows = measure_binding_pairs(model, tokenizer, pairs, device)
        out[split] = agg
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def deltas(scores: Dict[str, Any], a: str, b: str):
    # returns a-b for numeric official columns and binding probes
    out = {'official_fast': {}, 'binding_probes': {}}
    for task, rec in scores[a]['official_fast'].items():
        if task == 'reading':
            continue
        va = rec.get('score'); vb = scores[b]['official_fast'].get(task, {}).get('score')
        out['official_fast'][task] = None if va is None or vb is None else va - vb
    ra = scores[a]['official_fast'].get('reading', {}).get('scores', {})
    rb = scores[b]['official_fast'].get('reading', {}).get('scores', {})
    for k in ['reading_eye_tracking', 'reading_self_paced', 'reading_mean']:
        out['official_fast'][k] = None if k not in ra or k not in rb else ra[k] - rb[k]
    for split, agg in scores[a]['binding_probes'].items():
        va = agg.get('both_correct_frac'); vb = scores[b]['binding_probes'].get(split, {}).get('both_correct_frac')
        out['binding_probes'][split + '_both_correct'] = None if va is None or vb is None else va - vb
    return out


def main():
    setup_env(); OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    results = {}
    for name, meta in MODELS.items():
        model_path = meta['path']
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        print(json.dumps({'event': 'binding_probe_start', 'model': name}), flush=True)
        binding = run_binding_probes(name, model_path, device)
        print(json.dumps({'event': 'binding_probe_done', 'model': name, 'train': binding['train']['both_correct_frac'], 'natural': binding['natural_templates']['both_correct_frac']}), flush=True)
        official = {}
        for task_name, task_type, data_path in ZERO_SHOT_TASKS:
            print(json.dumps({'event': 'official_eval_start', 'model': name, 'task': task_name}), flush=True)
            official[task_name] = run_sentence_eval(name, model_path, task_name, task_type, data_path)
            print(json.dumps({'event': 'official_eval_done', 'model': name, 'task': task_name, 'score': official[task_name]['score'], 'returncode': official[task_name]['returncode']}), flush=True)
        print(json.dumps({'event': 'reading_eval_start', 'model': name}), flush=True)
        official['reading'] = run_reading(name, model_path)
        print(json.dumps({'event': 'reading_eval_done', 'model': name, 'scores': official['reading']['scores'], 'returncode': official['reading']['returncode']}), flush=True)
        meta_serializable = {**meta, 'path': str(meta['path'])}
        results[name] = {'metadata': meta_serializable, 'binding_probes': binding, 'official_fast': official}
    payload = {
        'status': 'LEGAL_BSM_1M_EVAL_DONE',
        'attribution_warning': 'coherent-official deltas are confounded by ~3x update count, shorter binding rows/packing, and targeted-mask supervision; coherent-swapped is the cleaner current mechanism contrast. If coherent-swapped separates, add a matched-update/input-structure official-experience reference before scaling.',
        'models': {k: {**v, 'path': str(v['path'])} for k, v in MODELS.items()},
        'results': results,
        'deltas': {
            'coherent_minus_swapped': deltas(results, 'bsm_coherent_20pct', 'bsm_swapped_20pct'),
            'coherent_minus_official': deltas(results, 'bsm_coherent_20pct', 'official_control'),
            'swapped_minus_official': deltas(results, 'bsm_swapped_20pct', 'official_control'),
        },
        'elapsed_sec': time.time() - t0,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    # concise note
    lines = ['# research — legal BSM 1M three-arm evaluation', '', f'Evidence JSON: `{OUT_JSON}`', '',
             '**Attribution warning:** coherent−official is not a pure binding effect because BSM arms have about 3× optimizer updates and different row packing/targeted masking. Coherent−swapped is the cleaner current comparison; if it separates, a matched-update official-experience reference is needed before scaling.', '',
             '## Official fast scores', '', '| model | BLiMP | Supplement | Entity | EWoK | GPIQA-par | GPIQA-non | GPIQA-mean | Reading |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in MODELS:
        off = results[name]['official_fast']
        gp = None
        if off['global_piqa_parallel']['score'] is not None and off['global_piqa_nonparallel']['score'] is not None:
            gp = (off['global_piqa_parallel']['score'] + off['global_piqa_nonparallel']['score'])/2
        reading = off['reading']['scores'].get('reading_mean')
        lines.append(f"| {name} | {off['blimp_fast']['score']} | {off['supplement_fast']['score']} | {off['entity_tracking_fast']['score']} | {off['ewok_fast']['score']} | {off['global_piqa_parallel']['score']} | {off['global_piqa_nonparallel']['score']} | {gp} | {reading} |")
    lines += ['', '## Binding probes: both-correct fraction', '', '| model | train | heldout_entities | heldout_values | heldout_templates | order_flip | natural |', '|---|---:|---:|---:|---:|---:|---:|']
    for name in MODELS:
        bp = results[name]['binding_probes']
        lines.append(f"| {name} | {bp['train']['both_correct_frac']:.3f} | {bp['heldout_entities']['both_correct_frac']:.3f} | {bp['heldout_values']['both_correct_frac']:.3f} | {bp['heldout_templates']['both_correct_frac']:.3f} | {bp['heldout_order_flip']['both_correct_frac']:.3f} | {bp['natural_templates']['both_correct_frac']:.3f} |")
    lines += ['', '## Coherent minus swapped deltas', '', '```json', json.dumps(payload['deltas']['coherent_minus_swapped'], indent=2), '```']
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'done', 'out': str(OUT_JSON), 'note': str(OUT_NOTE)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
