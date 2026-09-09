#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
FULL = STRICT / 'evaluation_data/full_eval'
BACKEND = 'mlm'
RUN = ROOT / 'training/runs/babylm_matched_low_s1_10M'
CKPT = 'chck_9999960w'
MODEL = (RUN / 'hf_model' / CKPT).resolve()
OUTDIR = (RUN / 'eval_results_available_direct').resolve()
OUT_JSON = ROOT / 'data/matched_low_10m_available_coordinate.json'
HIGH_JSON = ROOT / 'data/high_entity_state_10m_available_coordinate.json'
COMP_JSON = ROOT / 'data/high_entity_state_vs_matched_low_10m_available_comparison.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/high_entity_state_vs_matched_low_available_comparison.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/matched_low_available_continue.log')
EXISTING = {
    'blimp': ('blimp', 'blimp_filtered'),
    'supplement': ('blimp', 'supplement_filtered'),
    'entity_tracking': ('entity_tracking', 'entity_tracking'),
}
MISSING_TASKS = [
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

def run(cmd, env, logf):
    line='$ '+' '.join(map(str,cmd)); print(line, flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run([str(x) for x in cmd], cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-2500:], flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode!=0: raise RuntimeError(f'command failed {p.returncode}: {line}\n{p.stdout[-8000:]}')

def find_report(task: str, dataset_name: str) -> pathlib.Path:
    hits=list(OUTDIR.rglob(f'zero_shot/{BACKEND}/{task}/{dataset_name}/best_temperature_report.txt'))
    if len(hits)!=1:
        raise RuntimeError(f'report not unique for {task}/{dataset_name}: {hits[:10]}')
    return hits[0]

def main():
    if not MODEL.exists(): raise RuntimeError(f'missing {MODEL}')
    if not HIGH_JSON.exists(): raise RuntimeError(f'missing high JSON {HIGH_JSON}')
    OUTDIR.mkdir(parents=True, exist_ok=True); LOG.parent.mkdir(parents=True, exist_ok=True)
    env=setup_env(); scores={}; reports={}
    # Parse already completed reports.
    for col,(task,dataset) in EXISTING.items():
        rep=find_report(task,dataset); scores[col]=parse_avg(rep); reports[col]=str(rep)
    with LOG.open('w', encoding='utf-8') as logf:
        # Run only missing zero-shot tasks.
        for col,task,data,dataset in MISSING_TASKS:
            if not data.exists() or not any(data.rglob('*')): raise RuntimeError(f'missing data {col}: {data}')
            run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(MODEL),'--backend',BACKEND,'--task',task,'--data_path',str(data.resolve()),'--save_predictions','--revision_name','direct_'+CKPT,'--batch_size','64','--output_dir',str(OUTDIR)], env, logf)
            rep=find_report(task,dataset); scores[col]=parse_avg(rep); reports[col]=str(rep)
        # Reading.
        reading_csv=FULL/'reading/reading_data.csv'
        run([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(MODEL),'--backend',BACKEND,'--data_path',str(reading_csv.resolve()),'--revision_name','direct_'+CKPT,'--output_dir',str(OUTDIR)], env, logf)
    hits=list(OUTDIR.rglob(f'zero_shot/{BACKEND}/reading/report.txt'))
    if not hits: raise RuntimeError('missing reading report')
    reports['reading']=str(hits[-1]); scores.update(parse_reading(hits[-1]))
    metrics=json.loads((RUN/'scientific_metrics.json').read_text())
    payload={'status':'MATCHED_LOW_STRUCTURE_DENSITY_AVAILABLE_COORDINATE','arm':'matched_low','run':str(RUN),'model_path_direct':str(MODEL),'scores':scores,'derived_columns':{'GlobalPIQA_mean_parallel_nonparallel':(scores['global_piqa_parallel']+scores['global_piqa_nonparallel'])/2.0,'Reading_mean_eye_selfpaced':(scores['reading_eye_tracking']+scores['reading_self_paced'])/2.0},'training_core':{k:metrics.get(k) for k in ['word_exposure','actual_training_steps','optimizer_steps','loss_first','loss_last','masked_tokens_total','masked_tokens_per_whitespace_word','example_jsonl_label','example_jsonl_total_words','example_jsonl_total_rows']},'tokenization_coupling_summary':metrics.get('tokenization_coupling_summary'),'reports':reports,'missing_columns':['EWoK_full','SuperGLUE','AoA'],'note':'Structure-density route screen; completed by research continuation without rerunning tasks already completed in research.'}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True); OUT_JSON.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    high=json.loads(HIGH_JSON.read_text())
    deltas={k:high['scores'][k]-payload['scores'][k] for k in high['scores'] if k in payload['scores']}
    dd={k:high['derived_columns'][k]-payload['derived_columns'][k] for k in high['derived_columns']}
    comp={'status':'HIGH_ENTITY_STATE_MINUS_MATCHED_LOW_AVAILABLE_COMPARISON','mechanism':'within-source high entity/state lexical-structure density selection vs matched low composite structure','scores':{'high_entity_state':high['scores'],'matched_low':payload['scores']},'derived_columns':{'high_entity_state':high['derived_columns'],'matched_low':payload['derived_columns']},'high_entity_state_minus_matched_low':deltas,'derived_high_minus_low':dd,'training_loss_last':{'high_entity_state':high['training_core']['loss_last'],'matched_low':payload['training_core']['loss_last'],'high_minus_low':high['training_core']['loss_last']-payload['training_core']['loss_last']},'jsons':{'high_entity_state':str(HIGH_JSON),'matched_low':str(OUT_JSON)},'interpretation_warning':'Route screen only. Positive result requires relation-breaking control preserving words/length/source; null/negative result requires checking whether scorer measured real recoverable structure.'}
    COMP_JSON.write_text(json.dumps(comp,indent=2,ensure_ascii=False)+'\n')
    rows=[('blimp','BLiMP'),('supplement','Supplement'),('entity_tracking','Entity'),('comps','COMPS'),('global_piqa_parallel','GlobalPIQA parallel'),('global_piqa_nonparallel','GlobalPIQA nonparallel')]
    lines=['# research/160 — high_entity_state vs matched_low available comparison','',f'Comparison JSON: `{COMP_JSON}`','', '| column | high_entity_state | matched_low | high - low |','|---|---:|---:|---:|']
    for k,label in rows: lines.append(f'| {label} | {high["scores"][k]:.2f} | {payload["scores"][k]:.2f} | {deltas[k]:+.2f} |')
    lines.append(f'| GlobalPIQA mean | {high["derived_columns"]["GlobalPIQA_mean_parallel_nonparallel"]:.2f} | {payload["derived_columns"]["GlobalPIQA_mean_parallel_nonparallel"]:.2f} | {dd["GlobalPIQA_mean_parallel_nonparallel"]:+.2f} |')
    lines.append(f'| Reading mean | {high["derived_columns"]["Reading_mean_eye_selfpaced"]:.2f} | {payload["derived_columns"]["Reading_mean_eye_selfpaced"]:.2f} | {dd["Reading_mean_eye_selfpaced"]:+.2f} |')
    lines += ['', 'Full local EWoK is evaluated separately. This is a route screen, not a decisive causal localization of recoverable relation structure.']
    NOTE.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':comp['status'],'matched_json':str(OUT_JSON),'comparison':str(COMP_JSON),'deltas':deltas,'derived_deltas':dd},indent=2))
if __name__=='__main__': main()
