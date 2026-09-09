#!/usr/bin/env python3
"""Run and aggregate research FineWeb relation-vs-random 3M trajectory.

Launches the two matched arms concurrently on separate H100s using
relation_3m_train_eval_arm.py, then aggregates relation_explicit -
random_quality at chck_1M/chck_2M/chck_3M.
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
ARM_SCRIPT = ROOT/'scripts/relation_3m_train_eval_arm.py'
ARMS = [('random_quality','0'), ('relation_explicit','1')]
CKPTS = ['chck_1M','chck_2M','chck_3M']
COLS = ['blimp_fast','supplement_fast','ewok_fast','entity_tracking_fast','comps','reading_eye_tracking','reading_self_paced','Reading_mean']
OUT_JSON = ROOT/'data/fineweb_relation_vs_random_3m_trajectory.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/fineweb_relation_vs_random_3m_trajectory.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_3m_run_both.log')


def launch(arm: str, gpu: str, logf):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    cmd = [sys.executable, str(ARM_SCRIPT), '--arm', arm]
    logf.write(f"\nLAUNCH {arm} gpu={gpu}: {' '.join(cmd)}\n"); logf.flush()
    out = open(ROOT/f'tasks/step298_{arm}_stdout.log', 'w', encoding='utf-8')
    proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, text=True)
    return proc, out


def read_arm(arm: str):
    path = ROOT/f'data/fineweb_{arm}_3m_profile.json'
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    t0 = time.time(); LOG.parent.mkdir(parents=True, exist_ok=True); (ROOT/'tasks').mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as logf:
        logf.write(f"\n===== research both-arms start {time.ctime()} =====\n")
        procs = [(arm, gpu, *launch(arm, gpu, logf)) for arm, gpu in ARMS]
        failed = []
        for arm, gpu, proc, out in procs:
            rc = proc.wait(); out.close()
            logf.write(f"ARM_DONE {arm} gpu={gpu} rc={rc}\n"); logf.flush()
            if rc != 0:
                failed.append((arm, rc, str(ROOT/f'tasks/step298_{arm}_stdout.log')))
        if failed:
            raise RuntimeError(f'failed arms: {failed}')
    random = read_arm('random_quality')
    relation = read_arm('relation_explicit')
    trajectory = {}
    for ckpt in CKPTS:
        r = random['checkpoints'][ckpt]['scores']
        e = relation['checkpoints'][ckpt]['scores']
        trajectory[ckpt] = {
            'random_quality': r,
            'relation_explicit': e,
            'relation_minus_random': {k: round(e[k]-r[k],4) for k in COLS},
        }
    payload = {
        'status': 'FINEWEB_RELATION_VS_RANDOM_3M_TRAJECTORY',
        'random_profile': 'experiments/archive/initial_model_studies/data/fineweb_random_quality_3m_profile.json',
        'relation_profile': 'experiments/archive/initial_model_studies/data/fineweb_relation_explicit_3m_profile.json',
        'trajectory': trajectory,
        'random_metrics': random['metrics_summary'],
        'relation_metrics': relation['metrics_summary'],
        'elapsed_sec': time.time()-t0,
        'design_note': 'Matched same-source FineWeb-Edu random-quality vs relation-explicit, exact 3M exposure; checkpoints at 1M/2M/3M; same research DeBERTa-v2 WWM configuration.'
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# research FineWeb relation-explicit vs random-quality 3M trajectory','',f'Evidence JSON: `{OUT_JSON}`','',
             '| checkpoint | ΔBLiMP | ΔSupplement | ΔEWoK | ΔEntity | ΔCOMPS | ΔReading |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for ckpt in CKPTS:
        d = trajectory[ckpt]['relation_minus_random']
        lines.append(f"| {ckpt} | {d['blimp_fast']:+.4f} | {d['supplement_fast']:+.4f} | {d['ewok_fast']:+.4f} | {d['entity_tracking_fast']:+.4f} | {d['comps']:+.4f} | {d['Reading_mean']:+.4f} |")
    lines += ['','## Absolute scores','']
    for ckpt in CKPTS:
        lines += [f'### {ckpt}', '', '| arm | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |', '|---|---:|---:|---:|---:|---:|---:|']
        for arm in ['random_quality','relation_explicit']:
            s = trajectory[ckpt][arm]
            lines.append(f"| {arm} | {s['blimp_fast']:.4f} | {s['supplement_fast']:.4f} | {s['ewok_fast']:.4f} | {s['entity_tracking_fast']:.4f} | {s['comps']:.4f} | {s['Reading_mean']:.4f} |")
        lines.append('')
    lines += ['## Training/tokenization summaries', '', '```json', json.dumps({'random': random['metrics_summary'], 'relation': relation['metrics_summary']}, indent=2), '```']
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'out': str(OUT_JSON), 'note': str(OUT_NOTE), 'trajectory_delta': {c: trajectory[c]['relation_minus_random'] for c in CKPTS}, 'elapsed_sec': payload['elapsed_sec']}, indent=2))

if __name__ == '__main__':
    main()
