#!/usr/bin/env python3
"""research exposure-trajectory evaluator for WWM vs AMLM checkpoint ladders.

Runs the research zero-shot/Reading evaluator over selected checkpoints and writes
a resumable trajectory JSON. The scientific target is Delta_m(e)=AMLM-WWM as a
function of word exposure, especially whether the 10M EWoK/GlobalPIQA AMLM signal
reverses by 100M.
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, time

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
EVAL = ROOT/'scripts/eval_final_mlm_coordinate.py'
DATA_OUT_DEFAULT = ROOT/'data/existing_runs_exposure_trajectory.json'
NOTE_OUT_DEFAULT = (ROOT.parents[2] / 'research/notes/initial_model_studies/existing_runs_exposure_trajectory.md')
CKPTS = ['chck_5M','chck_10M','chck_20M','chck_40M','chck_60M','chck_80M','chck_100M']
RUNS = {
    'wwm_seed42': ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model',
    'wwm_seed43': ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model',
    'amlm_seed42': ROOT/'training/runs/fullcycle_official_amlm_debertav2_8x480_seed42_100M_b256/hf_model',
    'amlm_seed43': ROOT/'training/runs/fullcycle_official_amlm_debertav2_8x480_seed43_100M_b256/hf_model',
}
KEYS = ['blimp','supplement','entity_tracking','ewok','comps','GlobalPIQA_mean','Reading_mean','NLP_mean_no_superglue_aoa']

def load_json(p: pathlib.Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}

def save_json(p: pathlib.Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj, indent=2)+'\n')

def eval_one(run_name: str, ckpt: str, model_path: pathlib.Path, gpu: int, force: bool=False) -> dict:
    out_dir = model_path.parent / f'eval_step263_traj_{run_name}_{ckpt}'
    out_json = ROOT/'data/traj_scores'/f'{run_name}_{ckpt}.json'
    out_note = (ROOT.parents[2] / 'research/notes/initial_model_studies/traj_scores')/f'{run_name}_{ckpt}.md'
    log = (ROOT.parents[2] / 'research/notes/initial_model_studies/traj_scores')/f'{run_name}_{ckpt}.log'
    if out_json.exists() and not force:
        return json.loads(out_json.read_text())
    cmd=[sys.executable, str(EVAL), '--model_path', str(model_path), '--run_name', f'{run_name}_{ckpt}', '--out_json', str(out_json), '--out_note', str(out_note), '--log', str(log), '--output_dir', str(out_dir), '--revision', ckpt, '--gpu', str(gpu)]
    print('[eval]', ' '.join(cmd), flush=True)
    p=subprocess.run(cmd, cwd='.', text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-4000:], flush=True)
    if p.returncode:
        raise RuntimeError(f'eval failed {run_name} {ckpt} rc={p.returncode}\n{p.stdout[-8000:]}')
    return json.loads(out_json.read_text())

def summarize(payload: dict) -> str:
    lines=['# research existing-run exposure trajectory','','Input runs: protected WWM seed42; official AMLM seed42/seed43.','', '## Scores by exposure','']
    for run, recs in payload.get('runs', {}).items():
        lines += [f'### {run}', '', '| ckpt | '+' | '.join(KEYS)+' |', '|---|'+'|'.join(['---:']*len(KEYS))+'|']
        for ckpt in CKPTS:
            if ckpt in recs:
                s=recs[ckpt]['scores']; lines.append('| '+ckpt+' | '+' | '.join(f"{s.get(k,float('nan')):.3f}" for k in KEYS)+' |')
        lines.append('')
    if 'deltas' in payload:
        lines += ['## AMLM minus WWM(seed42) deltas','']
        for run in ['amlm_seed42','amlm_seed43']:
            if run not in payload['deltas']: continue
            lines += [f'### {run} - wwm_seed42', '', '| ckpt | '+' | '.join(KEYS)+' |', '|---|'+'|'.join(['---:']*len(KEYS))+'|']
            for ckpt in CKPTS:
                if ckpt in payload['deltas'][run]:
                    d=payload['deltas'][run][ckpt]; lines.append('| '+ckpt+' | '+' | '.join(f"{d.get(k,float('nan')):+.3f}" for k in KEYS)+' |')
            lines.append('')
    return '\n'.join(lines)+'\n'

def compute_deltas(payload: dict) -> None:
    base=payload.get('runs',{}).get('wwm_seed42',{})
    payload['deltas']={}
    for run in ['amlm_seed42','amlm_seed43']:
        payload['deltas'][run]={}
        for ckpt, rec in payload.get('runs',{}).get(run,{}).items():
            if ckpt in base:
                a=rec['scores']; b=base[ckpt]['scores']; payload['deltas'][run][ckpt]={k:a[k]-b[k] for k in KEYS if k in a and k in b}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=list(RUNS))
    ap.add_argument('--ckpts', nargs='*', default=CKPTS)
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--out_json', default=str(DATA_OUT_DEFAULT))
    ap.add_argument('--out_note', default=str(NOTE_OUT_DEFAULT))
    ap.add_argument('--force', action='store_true')
    args=ap.parse_args()
    out_json=pathlib.Path(args.out_json); out_note=pathlib.Path(args.out_note)
    payload=load_json(out_json) or {'status':'EXISTING_RUNS_EXPOSURE_TRAJECTORY','runs':{},'inputs':{k:str(v) for k,v in RUNS.items()},'ckpts':args.ckpts}
    t0=time.time()
    for run in args.runs:
        payload['runs'].setdefault(run,{})
        for ckpt in args.ckpts:
            model_path=RUNS[run]/ckpt
            if not model_path.exists():
                raise FileNotFoundError(f'missing {run} {ckpt}: {model_path}')
            if ckpt in payload['runs'][run] and not args.force:
                print(f'[resume] {run} {ckpt}', flush=True); continue
            rec=eval_one(run, ckpt, model_path, args.gpu, force=args.force)
            payload['runs'][run][ckpt]={'scores':rec['scores'],'reports':rec.get('reports',{}),'model_path':str(model_path)}
            compute_deltas(payload); payload['elapsed_sec']=round(time.time()-t0,1); save_json(out_json,payload); out_note.parent.mkdir(parents=True, exist_ok=True); out_note.write_text(summarize(payload))
    compute_deltas(payload); payload['elapsed_sec']=round(time.time()-t0,1); save_json(out_json,payload); out_note.parent.mkdir(parents=True, exist_ok=True); out_note.write_text(summarize(payload))
    print(json.dumps({'out_json':str(out_json),'out_note':str(out_note),'runs':list(payload['runs'])}, indent=2))
if __name__=='__main__': main()
