#!/usr/bin/env python3
"""research matched cadence evaluation: fixed-256 reference vs combined schedule.

No training. Verifies manifests/configs/log summaries and evaluates exact chck_1M
checkpoint directories on fast BLiMP/Supplement/EWoK/Entity/COMPS/Reading.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, re, subprocess, sys, statistics

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'
RUNS={
 'fixed256_ref': ROOT/'training/runs/cadence_fixed256_ref_1M_b128_seed42',
 'combined': ROOT/'training/runs/cadence_combined_1M_b128_seed42',
}
OUT=ROOT/'data/cadence_fixed_vs_combined_fast_profile.json'
NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/cadence_fixed_vs_combined_fast_profile.md')
LOG=(ROOT.parents[2] / 'research/notes/initial_model_studies/cadence_fixed_vs_combined_fast_profile.log')
TASKS=[
 ('blimp_fast','blimp','evaluation_data/fast_eval/blimp_fast','blimp_fast'),
 ('supplement_fast','blimp','evaluation_data/fast_eval/supplement_fast','supplement_fast'),
 ('ewok_fast','ewok','evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast','ewok_fast'),
 ('entity_tracking_fast','entity_tracking','evaluation_data/fast_eval/entity_tracking_fast','entity_tracking_fast'),
 ('comps','comps','evaluation_data/full_eval/comps','comps'),
]

def env_setup():
    env=os.environ.copy(); hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve())
    env['HF_DATASETS_CACHE']=str((hf/'datasets').resolve()); env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve())
    env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve()); env.setdefault('TOKENIZERS_PARALLELISM','false')
    return env

def sha16_path(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()[:16]

def list_hash(xs):
    h=hashlib.sha256(json.dumps(xs,separators=(',',':')).encode()).hexdigest()
    return h[:16]

def read_json(p): return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def read_logs(run):
    rows=[]
    for line in (run/'training_log.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip(): rows.append(json.loads(line))
    return rows

def run_cmd(cmd, env, logf, cwd=None):
    line='$ '+' '.join(map(str,cmd)); print(line, flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run(cmd,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(p.stdout[-1600:], flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode!=0: raise RuntimeError(line+'\n'+p.stdout[-6000:])

def parse_avg(report):
    txt=pathlib.Path(report).read_text(errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if not m: raise RuntimeError(f'parse avg failed {report}')
    return float(m.group(1))

def parse_reading(report):
    txt=pathlib.Path(report).read_text(errors='replace'); out={}
    for lab,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(lab)+r':\s*([0-9.\-]+)',txt)
        if not m: raise RuntimeError(f'parse reading failed {report}')
        out[key]=float(m.group(1))
    out['Reading_mean']=(out['reading_eye_tracking']+out['reading_self_paced'])/2
    return out

def eval_profile(label, run, env, logf):
    ckpt=run/'hf_model/chck_1M'
    outdir=run/f'eval_step321_direct_{label}'
    scores={}; reports={}
    for task_name,task,data_path,ds_name in TASKS:
        run_cmd([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(ckpt.resolve()),'--backend','mlm','--task',task,'--data_path',data_path,'--save_predictions','--batch_size','64','--output_dir',str(outdir.resolve())], env, logf, cwd=str(STRICT))
        report=outdir/ckpt.name/'main'/'zero_shot'/'mlm'/task/ds_name/'best_temperature_report.txt'
        scores[task_name]=parse_avg(report); reports[task_name]=str(report)
    run_cmd([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(ckpt.resolve()),'--backend','mlm','--data_path','evaluation_data/fast_eval/reading/reading_data.csv','--output_dir',str(outdir.resolve())], env, logf, cwd=str(STRICT))
    rr=outdir/ckpt.name/'main'/'zero_shot'/'mlm'/'reading'/'report.txt'
    scores.update(parse_reading(rr)); reports['reading']=str(rr)
    return {'path':str(ckpt),'sha16':sha16_path(ckpt/'model.safetensors'),'scores':scores,'reports':reports}

def verify():
    out={}
    manifests={k:read_json(v/'example_order_manifest.json') for k,v in RUNS.items()}
    metrics={k:read_json(v/'scientific_metrics.json') for k,v in RUNS.items()}
    configs={k:read_json(v/'hf_model/chck_1M/config.json') for k,v in RUNS.items()}
    logs={k:read_logs(v) for k,v in RUNS.items()}
    order_hash={k:list_hash(manifests[k].get('consumed_example_ids_in_order', manifests[k].get('consumed_example_ids', []))) for k in RUNS}
    # research trainer's manifest is compact and lacks full example order; compare source words/seed/num examples as available.
    out['manifests']={k:manifests[k] for k in RUNS}
    out['metrics']={k:metrics[k] for k in RUNS}
    out['configs']={k:{kk:configs[k].get(kk) for kk in ['vocab_size','hidden_size','num_hidden_layers','num_attention_heads','intermediate_size','max_position_embeddings','max_relative_positions','position_buckets','relative_attention','pos_att_type']} for k in RUNS}
    out['config_equal']=out['configs']['fixed256_ref']==out['configs']['combined']
    out['param_counts']={k:metrics[k].get('parameter_count') for k in RUNS}
    out['word_exposure']={k:metrics[k].get('word_exposure') for k in RUNS}
    out['steps']={k:metrics[k].get('total_steps') for k in RUNS}
    out['loss_last']={k:metrics[k].get('loss_last') for k in RUNS}
    out['seq_len_first_last']={k:(metrics[k].get('seq_len_first'),metrics[k].get('seq_len_last')) for k in RUNS}
    out['mask_first_last']={k:(metrics[k].get('mask_prob_first'),metrics[k].get('mask_prob_last')) for k in RUNS}
    out['log_summaries']={}
    for k,rs in logs.items():
        out['log_summaries'][k]={
            'n_log_rows':len(rs),
            'active_tokens_total':sum(r['n_active_tokens'] for r in rs),
            'masked_targets_total':sum(r['n_masked_targets'] for r in rs),
            'active_tokens_mean':statistics.mean(r['n_active_tokens'] for r in rs),
            'masked_targets_mean':statistics.mean(r['n_masked_targets'] for r in rs),
            'unique_seq_lens':sorted(set(r['cur_seq_len'] for r in rs)),
            'mask_prob_minmax':[min(r['cur_mask_prob'] for r in rs),max(r['cur_mask_prob'] for r in rs)],
        }
    return out

def main():
    env=env_setup(); LOG.parent.mkdir(parents=True,exist_ok=True)
    verification=verify(); profiles={}
    with LOG.open('a',encoding='utf-8') as logf:
        logf.write('\n===== research cadence eval =====\n')
        for label,run in RUNS.items(): profiles[label]=eval_profile(label,run,env,logf)
    keys=list(profiles['fixed256_ref']['scores'].keys())
    delta={k:profiles['combined']['scores'][k]-profiles['fixed256_ref']['scores'][k] for k in keys}
    payload={'status':'CADENCE_FIXED_VS_COMBINED_FAST_PROFILE','verification':verification,'profiles':profiles,'combined_minus_fixed256_ref':delta}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research cadence fixed-256 reference vs combined schedule','',f'JSON: `{OUT}`',f'Log: `{LOG}`','','## Matching and training summaries','',f"Config equal: {verification['config_equal']}",f"Param counts: {verification['param_counts']}",f"Word exposure: {verification['word_exposure']}",f"Steps: {verification['steps']}",f"Log summaries: `{json.dumps(verification['log_summaries'], ensure_ascii=False)}`",'','## Fast profile: combined - fixed256_ref','','| BLiMP | Supplement | EWoK | Entity | COMPS | Reading |','|---:|---:|---:|---:|---:|---:|',f"| {delta['blimp_fast']:+.2f} | {delta['supplement_fast']:+.2f} | {delta['ewok_fast']:+.2f} | {delta['entity_tracking_fast']:+.2f} | {delta['comps']:+.2f} | {delta['Reading_mean']:+.3f} |",'','## Raw scores','','| model | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |','|---|---:|---:|---:|---:|---:|---:|']
    for label in ['fixed256_ref','combined']:
        s=profiles[label]['scores']; lines.append(f"| {label} | {s['blimp_fast']:.2f} | {s['supplement_fast']:.2f} | {s['ewok_fast']:.2f} | {s['entity_tracking_fast']:.2f} | {s['comps']:.2f} | {s['Reading_mean']:.3f} |")
    NOTE.parent.mkdir(parents=True,exist_ok=True); NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(OUT),'note':str(NOTE),'delta':delta,'verification':verification},indent=2)[:8000])
if __name__=='__main__': main()
