#!/usr/bin/env python3
"""research repaired direct-checkpoint evaluation for research 3M relation trajectory.

research evaluated local parent hf_model directories with --revision_name chck_*. For
local paths, transformers ignores revision and loads the parent/final model. This
script does no retraining. It evaluates each exact local checkpoint directory:
  .../hf_model/chck_1M, chck_2M, chck_3M
with no revision_name, records model.safetensors hashes, and aggregates the
corrected relation_explicit - random_quality trajectory.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT/'repos/babylm-eval/strict'
RUNS = {
    'random_quality': ROOT/'training/runs/fineweb_random_quality_debertav2_8x480_3M_b128_seed42',
    'relation_explicit': ROOT/'training/runs/fineweb_relation_explicit_debertav2_8x480_3M_b128_seed42',
}
CKPTS = ['chck_1M', 'chck_2M', 'chck_3M']
TASKS = [
    ('blimp_fast','blimp','evaluation_data/fast_eval/blimp_fast','blimp_fast'),
    ('supplement_fast','blimp','evaluation_data/fast_eval/supplement_fast','supplement_fast'),
    ('ewok_fast','ewok','evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast','ewok_fast'),
    ('entity_tracking_fast','entity_tracking','evaluation_data/fast_eval/entity_tracking_fast','entity_tracking_fast'),
    ('comps','comps','evaluation_data/full_eval/comps','comps'),
]
COLS = ['blimp_fast','supplement_fast','ewok_fast','entity_tracking_fast','comps','reading_eye_tracking','reading_self_paced','Reading_mean']
OUT_JSON = ROOT/'data/fineweb_relation_vs_random_3m_direct_checkpoint_trajectory.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/fineweb_relation_vs_random_3m_direct_checkpoint_trajectory.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_3m_direct_checkpoint_eval.log')


def setup_env():
    env = os.environ.copy()
    hf = ROOT/'training/hf_home'
    env['HF_HOME'] = str(hf.resolve())
    env['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    env['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    env['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    env['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    for k in ['HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def sha16(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def run(cmd, env, logf, cwd=None):
    line = '$ ' + ' '.join(map(str, cmd))
    print(line, flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-2500:], flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f'failed {p.returncode}: {line}\n{p.stdout[-8000:]}')


def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding='utf-8', errors='replace')
    m = re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if not m:
        raise RuntimeError(f'cannot parse average from {report}\n{txt[:1200]}')
    return float(m.group(1))


def read_reading(report: pathlib.Path) -> dict:
    txt = report.read_text(encoding='utf-8', errors='replace')
    out = {}
    for label, key in [('EYE TRACKING SCORE','reading_eye_tracking'), ('SELF-PACED READING SCORE','reading_self_paced')]:
        m = re.search(re.escape(label)+r':\s*([0-9.\-]+)', txt)
        if not m:
            raise RuntimeError(f'cannot parse {label} from {report}\n{txt}')
        out[key] = float(m.group(1))
    out['Reading_mean'] = (out['reading_eye_tracking'] + out['reading_self_paced']) / 2.0
    return out


def eval_one(arm: str, ckpt: str, env, logf) -> dict:
    run_dir = RUNS[arm]
    cp = run_dir/'hf_model'/ckpt
    if not (cp/'config.json').exists():
        raise FileNotFoundError(cp/'config.json')
    h = sha16(cp/'model.safetensors')
    outdir = run_dir/f'eval_step300_direct_{ckpt}'
    scores = {}; reports = {}
    for task_name, task, data_path, ds_name in TASKS:
        run([sys.executable, '-m', 'evaluation_pipeline.sentence_zero_shot.run',
             '--model_path_or_name', str(cp.resolve()), '--backend', 'mlm',
             '--task', task, '--data_path', data_path, '--save_predictions',
             '--batch_size', '64', '--output_dir', str(outdir.resolve())], env, logf, cwd=str(STRICT))
        report = outdir/cp.name/'main'/'zero_shot'/'mlm'/task/ds_name/'best_temperature_report.txt'
        scores[task_name] = read_avg(report); reports[task_name] = str(report)
    run([sys.executable, '-m', 'evaluation_pipeline.reading.run',
         '--model_path_or_name', str(cp.resolve()), '--backend', 'mlm',
         '--data_path', 'evaluation_data/fast_eval/reading/reading_data.csv',
         '--output_dir', str(outdir.resolve())], env, logf, cwd=str(STRICT))
    rr = outdir/cp.name/'main'/'zero_shot'/'mlm'/'reading'/'report.txt'
    scores.update(read_reading(rr)); reports['reading'] = str(rr)
    return {'checkpoint_path': str(cp), 'model_safetensors_sha16': h, 'scores': scores, 'reports': reports}


def load_training_metrics(arm: str) -> dict:
    p = RUNS[arm]/'scientific_metrics.json'
    m = json.loads(p.read_text(encoding='utf-8'))
    s = m.get('tokenization_coupling_summary') or {}
    return {
        'word_exposure': m.get('word_exposure'),
        'steps': m.get('actual_training_steps'),
        'loss_first': m.get('loss_first'),
        'loss_last': m.get('loss_last'),
        'truncated_fraction': s.get('truncated_example_fraction'),
        'kept_tokens_per_word': s.get('kept_tokens_per_whitespace_word'),
        'masked_tokens_per_word': m.get('masked_tokens_per_whitespace_word'),
        'saved_checkpoints': m.get('saved_checkpoints'),
    }


def main():
    t0 = time.time(); env = setup_env(); LOG.parent.mkdir(parents=True, exist_ok=True)
    rows = {arm: {} for arm in RUNS}
    with LOG.open('a', encoding='utf-8') as logf:
        logf.write(f'\n===== research direct checkpoint eval start {time.ctime()} =====\n')
        for ckpt in CKPTS:
            for arm in ['random_quality','relation_explicit']:
                logf.write(f'\n--- evaluating {arm} {ckpt} direct path ---\n'); logf.flush()
                rows[arm][ckpt] = eval_one(arm, ckpt, env, logf)
    trajectory = {}
    for ckpt in CKPTS:
        r = rows['random_quality'][ckpt]['scores']
        e = rows['relation_explicit'][ckpt]['scores']
        trajectory[ckpt] = {
            'random_quality': r,
            'relation_explicit': e,
            'relation_minus_random': {k: round(e[k]-r[k], 4) for k in COLS},
            'hashes': {
                'random_quality': rows['random_quality'][ckpt]['model_safetensors_sha16'],
                'relation_explicit': rows['relation_explicit'][ckpt]['model_safetensors_sha16'],
            }
        }
    payload = {
        'status': 'FINEWEB_RELATION_VS_RANDOM_3M_DIRECT_CHECKPOINT_TRAJECTORY',
        'source_step298_profiles': {
            'random': 'experiments/archive/initial_model_studies/data/fineweb_random_quality_3m_profile.json',
            'relation': 'experiments/archive/initial_model_studies/data/fineweb_relation_explicit_3m_profile.json',
        },
        'method_repair': 'Exact local checkpoint directories were passed as model_path_or_name with no revision_name; this avoids transformers ignoring revision for local parent directories.',
        'arms': rows,
        'trajectory': trajectory,
        'training_metrics': {arm: load_training_metrics(arm) for arm in RUNS},
        'elapsed_sec': time.time() - t0,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# research repaired direct-checkpoint FineWeb relation 3M trajectory','',f'Evidence JSON: `{OUT_JSON}`','',
             'Evaluation repair: each local checkpoint directory is passed directly as `model_path_or_name`; no `revision_name` is used. This corrects the research local-parent/revision loading artifact.','',
             '| checkpoint | ΔBLiMP | ΔSupplement | ΔEWoK | ΔEntity | ΔCOMPS | ΔReading | random hash | relation hash |',
             '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for ckpt in CKPTS:
        d = trajectory[ckpt]['relation_minus_random']; h = trajectory[ckpt]['hashes']
        lines.append(f"| {ckpt} | {d['blimp_fast']:+.4f} | {d['supplement_fast']:+.4f} | {d['ewok_fast']:+.4f} | {d['entity_tracking_fast']:+.4f} | {d['comps']:+.4f} | {d['Reading_mean']:+.4f} | {h['random_quality']} | {h['relation_explicit']} |")
    lines += ['','## Absolute scores','']
    for ckpt in CKPTS:
        lines += [f'### {ckpt}', '', '| arm | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |', '|---|---:|---:|---:|---:|---:|---:|']
        for arm in ['random_quality','relation_explicit']:
            s = trajectory[ckpt][arm]
            lines.append(f"| {arm} | {s['blimp_fast']:.4f} | {s['supplement_fast']:.4f} | {s['ewok_fast']:.4f} | {s['entity_tracking_fast']:.4f} | {s['comps']:.4f} | {s['Reading_mean']:.4f} |")
        lines.append('')
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'out': str(OUT_JSON), 'note': str(OUT_NOTE), 'trajectory_delta': {c: trajectory[c]['relation_minus_random'] for c in CKPTS}, 'elapsed_sec': payload['elapsed_sec']}, indent=2))

if __name__ == '__main__':
    main()
