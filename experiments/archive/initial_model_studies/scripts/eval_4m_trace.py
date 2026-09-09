#!/usr/bin/env python3
"""research: evaluate four-arm 4M paired BSM developmental trace.

Runs two evidence layers:
  1. Continuous binding-switch margins on available checkpoint directories.
     Note: because train_legal_bsm_screen.py names checkpoint directories by
     floor(M), chck_1M/2M/3M were overwritten by later quarter-M saves. Exact
     250k/500k/750k and 4M are reliable; 1M/2M/3M labels are treated as
     late-in-bin checkpoint dirs and interpreted cautiously.
  2. Official fast scores at chck_4M for BLiMP, Supplement, Entity, EWoK,
     GlobalPIQA parallel/nonparallel, Reading.
"""
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time
from typing import Dict, Any

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
OUT_DIR = ROOT / 'data/4m_trace_eval'
OUT_JSON = ROOT / 'data/4m_trace_eval.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/4m_trace_eval.md')
MARGIN_SCRIPT = ROOT / 'scripts/eval_binding_switch_margin_trace.py'

ARMS = {
    'official_standard': ROOT / 'training/runs/4m_official_standard_seed42/hf_model',
    'official_token_matched_reference': ROOT / 'training/runs/4m_official_token_matched_reference_seed42/hf_model',
    'bsm_paired_coherent': ROOT / 'training/runs/4m_bsm_paired_coherent_seed42/hf_model',
    'bsm_paired_swapped': ROOT / 'training/runs/4m_bsm_paired_swapped_seed42/hf_model',
}
CHECKPOINTS = ['chck_250k', 'chck_500k', 'chck_750k', 'chck_1M', 'chck_2M', 'chck_3M', 'chck_4M']
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


def run_margin_checkpoint(ckpt: str, n_pairs: int = 40):
    names=[]; paths=[]
    for arm, root in ARMS.items():
        p = root / ckpt
        if p.exists():
            names.append(arm); paths.append(str(p.resolve()))
    out_path = OUT_DIR / 'margins' / f'{ckpt}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(MARGIN_SCRIPT.resolve()), '--model_paths', *paths, '--model_names', *names,
           '--out_json', str(out_path.resolve()), '--n_pairs', str(n_pairs), '--seed', '42']
    t0=time.time()
    r = subprocess.run(cmd, cwd=str(pathlib.Path('.').resolve()), env=os.environ.copy(), capture_output=True, text=True, timeout=900)
    log_path = OUT_DIR / 'logs' / f'margins_{ckpt}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(r.stdout + '\n--- STDERR ---\n' + r.stderr, encoding='utf-8')
    data = json.loads(out_path.read_text()) if out_path.exists() else None
    return {'returncode': r.returncode, 'elapsed_sec': time.time()-t0, 'out_json': str(out_path), 'log': str(log_path), 'data': data}


def parse_sentence_score(stdout: str):
    for line in stdout.splitlines():
        s=line.strip()
        m=re.match(r'^(?:[0-9.]+)\s+([+-]?[0-9]+(?:\.[0-9]+)?)$', s)
        if m:
            return float(m.group(1))
    m=re.search(r'AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)', stdout)
    return float(m.group(1)) if m else None


def parse_reading(stdout: str):
    out={}
    for label,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(label)+r':\s*([+-]?[0-9]+(?:\.[0-9]+)?)', stdout)
        if m: out[key]=float(m.group(1))
    if len(out)==2: out['reading_mean']=(out['reading_eye_tracking']+out['reading_self_paced'])/2
    return out


def run_sentence_eval(model_name: str, model_path: pathlib.Path, task_name: str, task_type: str, data_path: str):
    log_path = OUT_DIR / 'official_fast_logs' / f'{model_name}_{task_name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd=[sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(model_path.resolve()),
         '--backend','mlm','--task',task_type,'--data_path',data_path,'--revision_name',f'step345_{model_name}_{task_name}','--save_predictions']
    t0=time.time()
    r=subprocess.run(cmd,cwd=str(STRICT.resolve()),env=os.environ.copy(),capture_output=True,text=True,timeout=900)
    log_path.write_text(r.stdout+'\n--- STDERR ---\n'+r.stderr,encoding='utf-8')
    return {'score':parse_sentence_score(r.stdout),'returncode':r.returncode,'elapsed_sec':time.time()-t0,'log':str(log_path)}


def run_reading(model_name: str, model_path: pathlib.Path):
    log_path=OUT_DIR/'official_fast_logs'/f'{model_name}_reading.log'
    cmd=[sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(model_path.resolve()),'--backend','mlm',
         '--data_path','evaluation_data/fast_eval/reading/reading_data.csv','--revision_name',f'step345_{model_name}_reading']
    t0=time.time()
    r=subprocess.run(cmd,cwd=str(STRICT.resolve()),env=os.environ.copy(),capture_output=True,text=True,timeout=900)
    log_path.write_text(r.stdout+'\n--- STDERR ---\n'+r.stderr,encoding='utf-8')
    return {'scores':parse_reading(r.stdout),'returncode':r.returncode,'elapsed_sec':time.time()-t0,'log':str(log_path)}


def run_official_fast_4m():
    out={}
    for arm, root in ARMS.items():
        p=root/'chck_4M'
        out[arm]={}
        for tname,ttype,dpath in ZERO_SHOT_TASKS:
            print(json.dumps({'event':'official_eval_start','arm':arm,'task':tname}), flush=True)
            out[arm][tname]=run_sentence_eval(arm,p,tname,ttype,dpath)
            print(json.dumps({'event':'official_eval_done','arm':arm,'task':tname,'score':out[arm][tname]['score'],'rc':out[arm][tname]['returncode']}), flush=True)
        print(json.dumps({'event':'reading_eval_start','arm':arm}), flush=True)
        out[arm]['reading']=run_reading(arm,p)
        print(json.dumps({'event':'reading_eval_done','arm':arm,'scores':out[arm]['reading']['scores'],'rc':out[arm]['reading']['returncode']}), flush=True)
    return out


def get_metric_table(official):
    table={}
    for arm, rec in official.items():
        gp=None
        if rec['global_piqa_parallel']['score'] is not None and rec['global_piqa_nonparallel']['score'] is not None:
            gp=(rec['global_piqa_parallel']['score']+rec['global_piqa_nonparallel']['score'])/2
        table[arm]={
            'BLiMP': rec['blimp_fast']['score'],
            'Supplement': rec['supplement_fast']['score'],
            'Entity': rec['entity_tracking_fast']['score'],
            'EWoK': rec['ewok_fast']['score'],
            'GlobalPIQA_parallel': rec['global_piqa_parallel']['score'],
            'GlobalPIQA_nonparallel': rec['global_piqa_nonparallel']['score'],
            'GlobalPIQA_mean': gp,
            'Reading': rec['reading']['scores'].get('reading_mean'),
        }
    return table


def subtract_table(table, a, b):
    return {k: (None if table[a][k] is None or table[b][k] is None else table[a][k]-table[b][k]) for k in table[a]}


def margin_summary(margins):
    # compact table for train/natural mean margin and both-correct per checkpoint/arm
    out={}
    for ck, rec in margins.items():
        if not rec.get('data'): continue
        out[ck]={}
        for arm, splits in rec['data']['results'].items():
            out[ck][arm]={
                'train_mean_margin': splits['train']['mean_margin'],
                'train_both_correct': splits['train']['both_correct_frac'],
                'natural_mean_margin': splits['natural_templates']['mean_margin'],
                'natural_both_correct': splits['natural_templates']['both_correct_frac'],
                'heldout_entities_mean_margin': splits['heldout_entities']['mean_margin'],
                'heldout_values_mean_margin': splits['heldout_values']['mean_margin'],
                'heldout_templates_mean_margin': splits['heldout_templates']['mean_margin'],
                'order_flip_mean_margin': splits['heldout_order_flip']['mean_margin'],
            }
    return out


def main():
    setup_env(); OUT_DIR.mkdir(parents=True, exist_ok=True); t0=time.time()
    margins={}
    for ck in CHECKPOINTS:
        print(json.dumps({'event':'margin_start','checkpoint':ck}), flush=True)
        margins[ck]=run_margin_checkpoint(ck, n_pairs=40)
        print(json.dumps({'event':'margin_done','checkpoint':ck,'rc':margins[ck]['returncode'],'elapsed':round(margins[ck]['elapsed_sec'],1)}), flush=True)
    official=run_official_fast_4m()
    table=get_metric_table(official)
    payload={
        'status':'reference_4M_TRACE_EVAL_DONE',
        'checkpoint_warning':'Trainer checkpoint naming with checkpoint_words=250000 overwrote exact 1M/2M/3M dirs; chck_250k/500k/750k/4M are reliable, chck_1M/2M/3M are late-in-bin overwritten directories.',
        'margins':margins,
        'margin_summary':margin_summary(margins),
        'official_fast_4m':official,
        'official_fast_4m_table':table,
        'official_fast_4m_deltas':{
            'coherent_minus_swapped':subtract_table(table,'bsm_paired_coherent','bsm_paired_swapped'),
            'coherent_minus_reference':subtract_table(table,'bsm_paired_coherent','official_token_matched_reference'),
            'swapped_minus_reference':subtract_table(table,'bsm_paired_swapped','official_token_matched_reference'),
            'reference_minus_standard':subtract_table(table,'official_token_matched_reference','official_standard'),
        },
        'elapsed_sec':time.time()-t0,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    # note
    lines=['# research — 4M paired BSM trace evaluation','',f'JSON: `{OUT_JSON}`','',
           '**Checkpoint warning:** exact 1M/2M/3M checkpoint directories were overwritten by the trainer naming bug. Interpret `chck_1M/2M/3M` as late-in-bin available snapshots, not exact exposure checkpoints. Exact 250k/500k/750k/4M are valid.','',
           '## 4M official fast scores','','| arm | BLiMP | Supplement | Entity | EWoK | GPIQA mean | Reading |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ['official_standard','official_token_matched_reference','bsm_paired_coherent','bsm_paired_swapped']:
        s=table[arm]
        lines.append(f"| {arm} | {s['BLiMP']:.2f} | {s['Supplement']:.2f} | {s['Entity']:.2f} | {s['EWoK']:.2f} | {s['GlobalPIQA_mean']:.3f} | {s['Reading']:.2f} |")
    lines += ['','## 4M deltas','','```json',json.dumps(payload['official_fast_4m_deltas'], indent=2),'```','',
              '## Continuous margin summary (train/natural)','','| ckpt | arm | train margin | train both-correct | natural margin | natural both-correct |','|---|---|---:|---:|---:|---:|']
    ms=payload['margin_summary']
    for ck in CHECKPOINTS:
        if ck not in ms: continue
        for arm in ['official_token_matched_reference','bsm_paired_coherent','bsm_paired_swapped']:
            s=ms[ck][arm]
            lines.append(f"| {ck} | {arm} | {s['train_mean_margin']:.4f} | {s['train_both_correct']:.3f} | {s['natural_mean_margin']:.4f} | {s['natural_both_correct']:.3f} |")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':'done','out':str(OUT_JSON),'note':str(OUT_NOTE),'elapsed_sec':payload['elapsed_sec']}, indent=2), flush=True)

if __name__=='__main__':
    main()
