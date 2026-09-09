#!/usr/bin/env python3
"""research: evaluate replicated same-content curriculum-ordering experiment.

Arms trained in research:
  - official_random_order_a
  - official_random_order_b
  - official_curriculum

All are exact same 4M content, read in file order, seed42. This evaluator runs
official-compatible fast columns at chck_4000k and computes curriculum-minus-
random_a / random_b / random_mean deltas plus an observed weighted proxy:

  proxy = (3/28) * sum(BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA_mean deltas)
          + (1/8) * Reading_delta

SuperGLUE and AoA are not run in this fast screen and are treated as missing/0
delta for route screening; this is not a final Overall estimate.
"""
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
OUT_DIR = ROOT / 'data/curriculum_eval'
OUT_JSON = ROOT / 'data/curriculum_eval.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/curriculum_eval.md')

ARMS = {
    'random_a': ROOT / 'training/runs/curriculum_random_a_seed42/hf_model/chck_4000k',
    'random_b': ROOT / 'training/runs/curriculum_random_b_seed42/hf_model/chck_4000k',
    'curriculum': ROOT / 'training/runs/curriculum_ordered_seed42/hf_model/chck_4000k',
}

ZERO_SHOT_TASKS = [
    ('BLiMP', 'blimp', 'evaluation_data/fast_eval/blimp_fast'),
    ('Supplement', 'blimp', 'evaluation_data/fast_eval/supplement_fast'),
    ('EWoK', 'ewok', 'evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast'),
    ('Entity', 'entity_tracking', 'evaluation_data/fast_eval/entity_tracking_fast'),
    ('COMPS', 'comps', 'evaluation_data/full_eval/comps'),
    ('GlobalPIQA_parallel', 'global_piqa_parallel', 'evaluation_data/fast_eval/global_piqa_parallel'),
    ('GlobalPIQA_nonparallel', 'global_piqa_nonparallel', 'evaluation_data/fast_eval/global_piqa_nonparallel'),
]

NLP_FOR_PROXY = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA_mean']


def setup_env():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def parse_sentence_score(stdout: str):
    # Most BabyLM sentence evaluators print a final line like: "... 54.32" or an AVERAGE block.
    for line in stdout.splitlines()[::-1]:
        s = line.strip()
        m = re.match(r'^(?:[0-9.]+\s+)?([+-]?[0-9]+(?:\.[0-9]+)?)$', s)
        if m:
            val = float(m.group(1))
            if -1000 < val < 1000:
                return val
    m = re.search(r'AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)', stdout)
    if m:
        return float(m.group(1))
    # COMPS variants sometimes print JSON-ish score lines.
    m = re.search(r'(?:accuracy|score|acc)[^0-9+-]*([+-]?[0-9]+(?:\.[0-9]+)?)', stdout, flags=re.I)
    return float(m.group(1)) if m else None


def parse_reading(stdout: str):
    out = {}
    for label, key in [('EYE TRACKING SCORE', 'reading_eye_tracking'), ('SELF-PACED READING SCORE', 'reading_self_paced')]:
        m = re.search(re.escape(label)+r':\s*([+-]?[0-9]+(?:\.[0-9]+)?)', stdout)
        if m:
            out[key] = float(m.group(1))
    if len(out) == 2:
        out['Reading'] = (out['reading_eye_tracking'] + out['reading_self_paced']) / 2.0
    return out


def run_sentence_eval(arm: str, model_path: pathlib.Path, col: str, task: str, data_path: str):
    log_path = OUT_DIR / 'logs' / f'{arm}_{col}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'evaluation_pipeline.sentence_zero_shot.run',
        '--model_path_or_name', str(model_path.resolve()),
        '--backend', 'mlm', '--task', task, '--data_path', data_path,
        '--revision_name', f'step350_{arm}_{col}', '--save_predictions'
    ]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=os.environ.copy(), capture_output=True, text=True, timeout=1200)
    log_path.write_text(r.stdout + '\n--- STDERR ---\n' + r.stderr, encoding='utf-8')
    return {'score': parse_sentence_score(r.stdout), 'returncode': r.returncode, 'elapsed_sec': time.time()-t0, 'log': str(log_path)}


def run_reading(arm: str, model_path: pathlib.Path):
    log_path = OUT_DIR / 'logs' / f'{arm}_Reading.log'
    cmd = [
        sys.executable, '-m', 'evaluation_pipeline.reading.run',
        '--model_path_or_name', str(model_path.resolve()), '--backend', 'mlm',
        '--data_path', 'evaluation_data/fast_eval/reading/reading_data.csv',
        '--revision_name', f'step350_{arm}_Reading'
    ]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=os.environ.copy(), capture_output=True, text=True, timeout=1200)
    log_path.write_text(r.stdout + '\n--- STDERR ---\n' + r.stderr, encoding='utf-8')
    return {'scores': parse_reading(r.stdout), 'returncode': r.returncode, 'elapsed_sec': time.time()-t0, 'log': str(log_path)}


def evaluate_all():
    results = {}
    for arm, path in ARMS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        results[arm] = {}
        for col, task, dpath in ZERO_SHOT_TASKS:
            print(json.dumps({'event':'eval_start','arm':arm,'column':col}), flush=True)
            rec = run_sentence_eval(arm, path, col, task, dpath)
            results[arm][col] = rec
            print(json.dumps({'event':'eval_done','arm':arm,'column':col,'score':rec['score'],'rc':rec['returncode']}), flush=True)
        print(json.dumps({'event':'reading_start','arm':arm}), flush=True)
        rec = run_reading(arm, path)
        results[arm]['Reading'] = rec
        print(json.dumps({'event':'reading_done','arm':arm,'scores':rec['scores'],'rc':rec['returncode']}), flush=True)
    return results


def make_table(results):
    table = {}
    for arm, rec in results.items():
        gp = None
        if rec['GlobalPIQA_parallel']['score'] is not None and rec['GlobalPIQA_nonparallel']['score'] is not None:
            gp = (rec['GlobalPIQA_parallel']['score'] + rec['GlobalPIQA_nonparallel']['score']) / 2.0
        table[arm] = {
            'BLiMP': rec['BLiMP']['score'],
            'Supplement': rec['Supplement']['score'],
            'EWoK': rec['EWoK']['score'],
            'Entity': rec['Entity']['score'],
            'COMPS': rec['COMPS']['score'],
            'GlobalPIQA_parallel': rec['GlobalPIQA_parallel']['score'],
            'GlobalPIQA_nonparallel': rec['GlobalPIQA_nonparallel']['score'],
            'GlobalPIQA_mean': gp,
            'Reading': rec['Reading']['scores'].get('Reading'),
            'Reading_eye': rec['Reading']['scores'].get('reading_eye_tracking'),
            'Reading_self_paced': rec['Reading']['scores'].get('reading_self_paced'),
        }
    return table


def delta(table, a, b):
    return {k: (None if table[a].get(k) is None or table[b].get(k) is None else table[a][k] - table[b][k]) for k in table[a]}


def proxy(delta_rec):
    nlp_sum = sum(delta_rec[k] for k in NLP_FOR_PROXY if delta_rec.get(k) is not None)
    read = delta_rec.get('Reading') or 0.0
    return (3.0/28.0) * nlp_sum + (1.0/8.0) * read


def mean_table(table, arms):
    out = {}
    keys = table[arms[0]].keys()
    for k in keys:
        vals = [table[a][k] for a in arms if table[a].get(k) is not None]
        out[k] = sum(vals)/len(vals) if vals else None
    return out


def delta_against_mean(table, a, mean_name):
    return {k: (None if table[a].get(k) is None or table[mean_name].get(k) is None else table[a][k] - table[mean_name][k]) for k in table[a]}


def main():
    setup_env(); OUT_DIR.mkdir(parents=True, exist_ok=True); t0 = time.time()
    results = evaluate_all()
    table = make_table(results)
    table['random_mean'] = mean_table(table, ['random_a','random_b'])
    deltas = {
        'curriculum_minus_random_a': delta(table, 'curriculum', 'random_a'),
        'curriculum_minus_random_b': delta(table, 'curriculum', 'random_b'),
        'curriculum_minus_random_mean': delta_against_mean(table, 'curriculum', 'random_mean'),
        'random_b_minus_random_a': delta(table, 'random_b', 'random_a'),
    }
    proxies = {k: proxy(v) for k, v in deltas.items()}
    payload = {
        'status': 'CURRICULUM_ORDER_EVAL_DONE',
        'note': 'Fast screen only: SuperGLUE and AoA not evaluated; proxy treats them as 0 delta/missing.',
        'model_paths': {k: str(v) for k,v in ARMS.items()},
        'results': results,
        'table': table,
        'deltas': deltas,
        'weighted_proxy': proxies,
        'elapsed_sec': time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

    def fmt(x):
        return 'NA' if x is None else f'{x:.3f}'
    lines = [
        '# research — Replicated same-content curriculum-order evaluation', '',
        f'JSON: `{OUT_JSON}`', '',
        'All arms are the same 4M content, trained with `--no_shuffle`; only file order differs.', '',
        '## 4M endpoint scores', '',
        '| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA mean | Reading |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'
    ]
    for arm in ['random_a','random_b','random_mean','curriculum']:
        s = table[arm]
        lines.append(f"| {arm} | {fmt(s['BLiMP'])} | {fmt(s['Supplement'])} | {fmt(s['EWoK'])} | {fmt(s['Entity'])} | {fmt(s['COMPS'])} | {fmt(s['GlobalPIQA_mean'])} | {fmt(s['Reading'])} |")
    lines += ['', '## Deltas and weighted proxy', '', '| contrast | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | proxy |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['curriculum_minus_random_a','curriculum_minus_random_b','curriculum_minus_random_mean','random_b_minus_random_a']:
        d = deltas[name]
        lines.append(f"| {name} | {fmt(d['BLiMP'])} | {fmt(d['Supplement'])} | {fmt(d['EWoK'])} | {fmt(d['Entity'])} | {fmt(d['COMPS'])} | {fmt(d['GlobalPIQA_mean'])} | {fmt(d['Reading'])} | {fmt(proxies[name])} |")
    lines += ['', 'Proxy = (3/28) * delta(BLiMP+Supplement+EWoK+Entity+COMPS+GlobalPIQA_mean) + (1/8) * delta(Reading). SuperGLUE/AoA are not evaluated in this fast screen.', '']
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status':'done','out':str(OUT_JSON),'note':str(OUT_NOTE),'elapsed_sec':payload['elapsed_sec'],'weighted_proxy':proxies}, indent=2), flush=True)


if __name__ == '__main__':
    main()
