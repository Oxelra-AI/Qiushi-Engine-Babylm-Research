#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
FULL = STRICT / 'evaluation_data/full_eval'
BACKEND = 'mlm'
ARMS = {
    'high_entity_state': {
        'run': ROOT / 'training/runs/babylm_high_entity_state_s1_10M',
        'ckpt': 'chck_9999947w',
        'out': ROOT / 'data/high_entity_state_10m_available_coordinate.json'
    },
    'matched_low': {
        'run': ROOT / 'training/runs/babylm_matched_low_s1_10M',
        'ckpt': 'chck_9999960w',
        'out': ROOT / 'data/matched_low_10m_available_coordinate.json'
    },
}
OUT_COMPARISON = ROOT / 'data/high_entity_state_vs_matched_low_10m_available_comparison.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/high_entity_state_vs_matched_low_available_comparison.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/structure_high_vs_low_available_eval.log')
TASKS = [
    ('blimp', 'blimp', FULL / 'blimp_filtered', 'blimp_filtered'),
    ('supplement', 'blimp', FULL / 'supplement_filtered', 'supplement_filtered'),
    ('entity_tracking', 'entity_tracking', FULL / 'entity_tracking', 'entity_tracking'),
    ('comps', 'comps', FULL / 'comps', 'comps'),
    ('global_piqa_parallel', 'global_piqa_parallel', FULL / 'global_piqa_parallel', 'global_piqa_parallel'),
    ('global_piqa_nonparallel', 'global_piqa_nonparallel', FULL / 'global_piqa_nonparallel', 'global_piqa_nonparallel'),
]

def setup_env():
    env=os.environ.copy()
    hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve())
    env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve())
    env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env

def run(cmd, env, logf):
    line='$ '+' '.join(map(str,cmd)); print(line, flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run([str(x) for x in cmd], cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-2500:], flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode!=0: raise RuntimeError(f'command failed {p.returncode}: {line}\n{p.stdout[-8000:]}')

def parse_avg(path:pathlib.Path)->float:
    txt=path.read_text(encoding='utf-8', errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if m: return float(m.group(1))
    vals=re.findall(r'(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)', txt, flags=re.I)
    if vals:
        v=float(vals[-1]); return v*(100 if v<=1 else 1)
    raise RuntimeError(f'could not parse score from {path}\n{txt[:1000]}')

def parse_reading(path:pathlib.Path)->dict[str,float]:
    txt=path.read_text(encoding='utf-8', errors='replace')
    out={}
    for label,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(label)+r':\s*([0-9.\-]+)', txt)
        if not m: raise RuntimeError(f'could not parse {label} from {path}\n{txt[:1000]}')
        out[key]=float(m.group(1))
    return out

def eval_arm(name, spec, env, logf):
    run_dir=spec['run']; model=(run_dir/'hf_model'/spec['ckpt']).resolve()
    if not model.exists(): raise RuntimeError(f'missing {model}')
    outdir=(run_dir/'eval_results_available_direct').resolve(); outdir.mkdir(parents=True, exist_ok=True)
    scores={}; reports={}
    for col,task,data,dataset_name in TASKS:
        if not data.exists() or not any(data.rglob('*')): raise RuntimeError(f'missing data {col}: {data}')
        run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(model),'--backend',BACKEND,'--task',task,'--data_path',str(data.resolve()),'--save_predictions','--revision_name','direct_'+spec['ckpt'],'--batch_size','64','--output_dir',str(outdir)], env, logf)
        hits=list(outdir.rglob(f'zero_shot/{BACKEND}/{task}/{dataset_name}/best_temperature_report.txt'))
        if len(hits)!=1: raise RuntimeError(f'report not unique {name}/{col}: {hits[:10]}')
        scores[col]=parse_avg(hits[0]); reports[col]=str(hits[0])
    reading_csv=FULL/'reading/reading_data.csv'
    run([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(model),'--backend',BACKEND,'--data_path',str(reading_csv.resolve()),'--revision_name','direct_'+spec['ckpt'],'--output_dir',str(outdir)], env, logf)
    hits=list(outdir.rglob(f'zero_shot/{BACKEND}/reading/report.txt'))
    if not hits: raise RuntimeError(f'missing reading report {name}')
    reports['reading']=str(hits[-1]); scores.update(parse_reading(hits[-1]))
    metrics=json.loads((run_dir/'scientific_metrics.json').read_text())
    payload={'status':f'STEP159_{name.upper()}_STRUCTURE_DENSITY_AVAILABLE_COORDINATE','arm':name,'run':str(run_dir),'model_path_direct':str(model),'scores':scores,'derived_columns':{'GlobalPIQA_mean_parallel_nonparallel':(scores['global_piqa_parallel']+scores['global_piqa_nonparallel'])/2.0,'Reading_mean_eye_selfpaced':(scores['reading_eye_tracking']+scores['reading_self_paced'])/2.0},'training_core':{k:metrics.get(k) for k in ['word_exposure','actual_training_steps','optimizer_steps','loss_first','loss_last','masked_tokens_total','masked_tokens_per_whitespace_word','example_jsonl_label','example_jsonl_total_words','example_jsonl_total_rows']},'tokenization_coupling_summary':metrics.get('tokenization_coupling_summary'),'reports':reports,'missing_columns':['EWoK_full','SuperGLUE','AoA'],'note':'Structure-density route screen; not decisive proof of recoverable relation structure because lexical cue/topic/syntax are still coupled.'}
    spec['out'].parent.mkdir(parents=True, exist_ok=True); spec['out'].write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    return payload

def main():
    env=setup_env(); LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('w', encoding='utf-8') as logf:
        payloads={name:eval_arm(name,spec,env,logf) for name,spec in ARMS.items()}
    hi=payloads['high_entity_state']; lo=payloads['matched_low']
    deltas={k:hi['scores'][k]-lo['scores'][k] for k in hi['scores'] if k in lo['scores']}
    dd={k:hi['derived_columns'][k]-lo['derived_columns'][k] for k in hi['derived_columns']}
    comp={'status':'HIGH_ENTITY_STATE_MINUS_MATCHED_LOW_AVAILABLE_COMPARISON','mechanism':'within-source high entity/state lexical-structure density selection vs matched low composite structure','scores':{'high_entity_state':hi['scores'],'matched_low':lo['scores']},'derived_columns':{'high_entity_state':hi['derived_columns'],'matched_low':lo['derived_columns']},'high_entity_state_minus_matched_low':deltas,'derived_high_minus_low':dd,'training_loss_last':{'high_entity_state':hi['training_core']['loss_last'],'matched_low':lo['training_core']['loss_last'],'high_minus_low':hi['training_core']['loss_last']-lo['training_core']['loss_last']},'jsons':{name:str(spec['out']) for name,spec in ARMS.items()},'interpretation_warning':'Positive result would require relation-breaking control preserving words/length/source; null result requires checking whether scorer measured real recoverable structure.'}
    OUT_COMPARISON.write_text(json.dumps(comp,indent=2,ensure_ascii=False)+'\n')
    rows=[('blimp','BLiMP'),('supplement','Supplement'),('entity_tracking','Entity'),('comps','COMPS'),('global_piqa_parallel','GlobalPIQA parallel'),('global_piqa_nonparallel','GlobalPIQA nonparallel')]
    lines=['# research — high_entity_state vs matched_low available comparison','',f'Comparison JSON: `{OUT_COMPARISON}`','', '| column | high_entity_state | matched_low | high - low |','|---|---:|---:|---:|']
    for k,label in rows: lines.append(f'| {label} | {hi["scores"][k]:.2f} | {lo["scores"][k]:.2f} | {deltas[k]:+.2f} |')
    lines.append(f'| GlobalPIQA mean | {hi["derived_columns"]["GlobalPIQA_mean_parallel_nonparallel"]:.2f} | {lo["derived_columns"]["GlobalPIQA_mean_parallel_nonparallel"]:.2f} | {dd["GlobalPIQA_mean_parallel_nonparallel"]:+.2f} |')
    lines.append(f'| Reading mean | {hi["derived_columns"]["Reading_mean_eye_selfpaced"]:.2f} | {lo["derived_columns"]["Reading_mean_eye_selfpaced"]:.2f} | {dd["Reading_mean_eye_selfpaced"]:+.2f} |')
    lines.append('')
    lines.append('Full local EWoK is evaluated separately. This is a route screen, not a decisive causal localization of recoverable relation structure.')
    NOTE.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':comp['status'],'comparison':str(OUT_COMPARISON),'note':str(NOTE),'deltas':deltas,'derived_deltas':dd}, indent=2))
if __name__=='__main__': main()
