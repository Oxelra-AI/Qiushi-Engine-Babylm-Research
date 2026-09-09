#!/usr/bin/env python3
"""research corrected official-fast transfer diagnostic for BSM.

Evaluates the non-compliant research mature-checkpoint BSM adaptation against the
baseline chck_100M on official fast zero-shot columns. This is diagnostic only:
the BSM model starts from 100M and receives additional binding exposure, so it is
not a legal Strict-Small submission coordinate.
"""
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
BASELINE = ROOT / 'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M'
BSM = ROOT / 'training/runs/bsm_transfer_diagnostic/hf_model'
OUT_JSON = ROOT / 'data/bsm_transfer_official_fast.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/bsm_transfer_official_fast.md')

TASKS = [
    ('entity_tracking_fast', 'entity_tracking', 'evaluation_data/fast_eval/entity_tracking_fast'),
    ('ewok_fast', 'ewok', 'evaluation_data/fast_eval/ewok_fast'),
    ('blimp_fast', 'blimp', 'evaluation_data/fast_eval/blimp_fast'),
    ('supplement_fast', 'blimp', 'evaluation_data/fast_eval/supplement_fast'),
]


def setup_env():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def parse_score(stdout: str):
    # Official sentence_zero_shot.run prints first line like: "1.0\t22.40"
    for line in stdout.splitlines():
        s = line.strip()
        m = re.match(r'^(?:[0-9.]+)\s+([+-]?[0-9]+(?:\.[0-9]+)?)$', s)
        if m:
            return float(m.group(1))
    # Fallback: last AVERAGE ACCURACY number
    vals = re.findall(r'AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)', stdout)
    if vals:
        return float(vals[-1])
    return None


def run_eval(model_path: pathlib.Path, revision: str, task_type: str, data_path: str):
    cmd = [
        sys.executable, '-m', 'evaluation_pipeline.sentence_zero_shot.run',
        '--model_path_or_name', str(model_path.resolve()),
        '--backend', 'mlm',
        '--task', task_type,
        '--data_path', data_path,
        '--revision_name', revision,
        '--save_predictions',
    ]
    r = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=os.environ.copy(), capture_output=True, text=True, timeout=600)
    return {
        'returncode': r.returncode,
        'score': parse_score(r.stdout),
        'stdout': r.stdout,
        'stderr_tail': r.stderr[-2000:],
        'cmd': cmd,
    }


def main():
    setup_env()
    t0 = time.time()
    if not BSM.exists():
        raise FileNotFoundError(BSM)
    results = {'baseline': {}, 'bsm': {}}
    for task_name, task_type, data_path in TASKS:
        print(json.dumps({'event': 'eval_start', 'model': 'baseline', 'task': task_name}), flush=True)
        results['baseline'][task_name] = run_eval(BASELINE, f'baseline_{task_name}', task_type, data_path)
        print(json.dumps({'event': 'eval_done', 'model': 'baseline', 'task': task_name, 'score': results['baseline'][task_name]['score'], 'returncode': results['baseline'][task_name]['returncode']}), flush=True)
        print(json.dumps({'event': 'eval_start', 'model': 'bsm', 'task': task_name}), flush=True)
        results['bsm'][task_name] = run_eval(BSM, f'bsm_{task_name}', task_type, data_path)
        print(json.dumps({'event': 'eval_done', 'model': 'bsm', 'task': task_name, 'score': results['bsm'][task_name]['score'], 'returncode': results['bsm'][task_name]['returncode']}), flush=True)
    deltas = {}
    for task_name, _, _ in TASKS:
        a = results['baseline'][task_name]['score']
        b = results['bsm'][task_name]['score']
        deltas[task_name] = None if a is None or b is None else b - a
    payload = {
        'status': 'BSM_TRANSFER_OFFICIAL_FAST_DONE',
        'diagnostic_only_non_compliant': True,
        'reason_non_compliant': 'BSM model starts from 100M checkpoint and receives additional binding exposure; it cannot be a Strict-Small submission coordinate.',
        'baseline_model': str(BASELINE),
        'bsm_model': str(BSM),
        'tasks': TASKS,
        'results': results,
        'deltas_bsm_minus_baseline': deltas,
        'elapsed_sec': time.time() - t0,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = [
        '# research — BSM official-fast transfer diagnostic', '',
        '**Diagnostic only; not a legal Strict-Small coordinate.** The BSM model starts from the 100M checkpoint and receives extra binding exposure.', '',
        f'Evidence JSON: `{OUT_JSON}`', '',
        '| task | baseline | BSM diagnostic | delta |', '|---|---:|---:|---:|'
    ]
    for task_name, _, _ in TASKS:
        a = results['baseline'][task_name]['score']
        b = results['bsm'][task_name]['score']
        d = deltas[task_name]
        lines.append(f"| {task_name} | {a if a is not None else 'ERR'} | {b if b is not None else 'ERR'} | {d if d is not None else 'ERR'} |")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'done', 'deltas': deltas, 'out': str(OUT_JSON), 'note': str(OUT_NOTE)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
