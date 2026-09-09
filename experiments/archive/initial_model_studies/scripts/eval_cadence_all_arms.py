#!/usr/bin/env python3
"""research matched cadence evaluation for all four 1M arms.

Arms:
- fixed256_ref: fixed len=256, mask=0.15, max_seq_length=512/model config=520
- length_only: 128->256->512, mask=0.15
- mask_decay_only: len=256, mask 0.30->0.15
- combined: length schedule + mask decay

Uses exact chck_1M checkpoint directories. Reuses existing reports if present.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, re, statistics, subprocess, sys
ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'
RUNS={
 'fixed256_ref': ROOT/'training/runs/cadence_fixed256_ref_1M_b128_seed42',
 'length_only': ROOT/'training/runs/cadence_length_only_1M_b128_seed42',
 'mask_decay_only': ROOT/'training/runs/cadence_mask_decay_only_1M_b128_seed42',
 'combined': ROOT/'training/runs/cadence_combined_1M_b128_seed42',
}
OUT=ROOT/'data/cadence_all_arms_fast_profile.json'
NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/cadence_all_arms_fast_profile.md')
LOG=(ROOT.parents[2] / 'research/notes/initial_model_studies/cadence_all_arms_fast_profile.log')
TASKS=[
 ('blimp_fast','blimp','evaluation_data/fast_eval/blimp_fast','blimp_fast'),
 ('supplement_fast','blimp','evaluation_data/fast_eval/supplement_fast','supplement_fast'),
 ('ewok_fast','ewok','evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast','ewok_fast'),
 ('entity_tracking_fast','entity_tracking','evaluation_data/fast_eval/entity_tracking_fast','entity_tracking_fast'),
 ('comps','comps','evaluation_data/full_eval/comps','comps'),
]
def env_setup():
    env=os.environ.copy(); hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve()); env['HF_DATASETS_CACHE']=str((hf/'datasets').resolve()); env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve()); env.setdefault('TOKENIZERS_PARALLELISM','false')
    return env
def sha16(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()[:16]
def read_json(p): return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def read_logs(run):
    return [json.loads(l) for l in (run/'training_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
def parse_avg(report):
    txt=pathlib.Path(report).read_text(errors='replace'); m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)',txt)
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
def run_cmd(cmd,env,logf,cwd=None):
    line='$ '+' '.join(map(str,cmd)); print(line,flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run(cmd,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(p.stdout[-1200:],flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode!=0: raise RuntimeError(line+'\n'+p.stdout[-6000:])
def eval_profile(label,run,env,logf):
    ckpt=run/'hf_model/chck_1M'; outdir=run/f'eval_step321_direct_{label}'; scores={}; reports={}
    for task_name,task,data_path,ds_name in TASKS:
        report=outdir/ckpt.name/'main'/'zero_shot'/'mlm'/task/ds_name/'best_temperature_report.txt'
        if not report.exists():
            run_cmd([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(ckpt.resolve()),'--backend','mlm','--task',task,'--data_path',data_path,'--save_predictions','--batch_size','64','--output_dir',str(outdir.resolve())],env,logf,cwd=str(STRICT))
        scores[task_name]=parse_avg(report); reports[task_name]=str(report)
    rr=outdir/ckpt.name/'main'/'zero_shot'/'mlm'/'reading'/'report.txt'
    if not rr.exists():
        run_cmd([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(ckpt.resolve()),'--backend','mlm','--data_path','evaluation_data/fast_eval/reading/reading_data.csv','--output_dir',str(outdir.resolve())],env,logf,cwd=str(STRICT))
    scores.update(parse_reading(rr)); reports['reading']=str(rr)
    return {'path':str(ckpt),'sha16':sha16(ckpt/'model.safetensors'),'scores':scores,'reports':reports}
def summarize_training(run):
    m=read_json(run/'scientific_metrics.json'); cfg=read_json(run/'hf_model/chck_1M/config.json'); logs=read_logs(run)
    return {
        'parameter_count':m.get('parameter_count'),'word_exposure':m.get('word_exposure'),'steps':m.get('total_steps'),
        'loss_last':m.get('loss_last'),'seq_len_schedule':m.get('seq_len_schedule'),'mask_prob_start':m.get('mask_prob_start'),'mask_prob_end':m.get('mask_prob_end'),
        'seq_len_first':m.get('seq_len_first'),'seq_len_last':m.get('seq_len_last'),'mask_prob_first':m.get('mask_prob_first'),'mask_prob_last':m.get('mask_prob_last'),
        'config_subset':{k:cfg.get(k) for k in ['vocab_size','hidden_size','num_hidden_layers','num_attention_heads','intermediate_size','max_position_embeddings','max_relative_positions','position_buckets','relative_attention','pos_att_type']},
        'source_words':read_json(run/'example_order_manifest.json').get('source_words'),
        'active_tokens_total':sum(r['n_active_tokens'] for r in logs),'masked_targets_total':sum(r['n_masked_targets'] for r in logs),
        'active_tokens_mean':statistics.mean(r['n_active_tokens'] for r in logs),'masked_targets_mean':statistics.mean(r['n_masked_targets'] for r in logs),
        'unique_seq_lens':sorted(set(r['cur_seq_len'] for r in logs)),'mask_prob_minmax':[min(r['cur_mask_prob'] for r in logs),max(r['cur_mask_prob'] for r in logs)],
    }
def main():
    env=env_setup(); LOG.parent.mkdir(parents=True,exist_ok=True); profiles={}; training={}
    with LOG.open('a',encoding='utf-8') as logf:
        logf.write('\n===== research all cadence arms eval =====\n')
        for label,run in RUNS.items():
            training[label]=summarize_training(run)
            profiles[label]=eval_profile(label,run,env,logf)
    base=profiles['fixed256_ref']['scores']; deltas={}
    for label in RUNS:
        if label=='fixed256_ref': continue
        deltas[label]={k:profiles[label]['scores'][k]-base[k] for k in base}
    config_equal=all(training[label]['config_subset']==training['fixed256_ref']['config_subset'] for label in RUNS)
    payload={'status':'CADENCE_ALL_ARMS_FAST_PROFILE','config_equal_all':config_equal,'training':training,'profiles':profiles,'deltas_vs_fixed256_ref':deltas}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research matched cadence all-arms fast profile','',f'JSON: `{OUT}`',f'Log: `{LOG}`','',f'All configs equal to fixed reference: {config_equal}','','## Training target totals','', '| arm | active tokens total | masked targets total | loss last | seq lens | mask minmax |','|---|---:|---:|---:|---|---|']
    for label in RUNS:
        t=training[label]; lines.append(f"| {label} | {t['active_tokens_total']} | {t['masked_targets_total']} | {t['loss_last']:.4f} | {t['unique_seq_lens']} | {t['mask_prob_minmax']} |")
    lines += ['','## Raw scores','','| arm | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |','|---|---:|---:|---:|---:|---:|---:|']
    for label in RUNS:
        s=profiles[label]['scores']; lines.append(f"| {label} | {s['blimp_fast']:.2f} | {s['supplement_fast']:.2f} | {s['ewok_fast']:.2f} | {s['entity_tracking_fast']:.2f} | {s['comps']:.2f} | {s['Reading_mean']:.3f} |")
    lines += ['','## Deltas vs fixed256_ref','','| arm | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |','|---|---:|---:|---:|---:|---:|---:|']
    for label,d in deltas.items(): lines.append(f"| {label} | {d['blimp_fast']:+.2f} | {d['supplement_fast']:+.2f} | {d['ewok_fast']:+.2f} | {d['entity_tracking_fast']:+.2f} | {d['comps']:+.2f} | {d['Reading_mean']:+.3f} |")
    NOTE.parent.mkdir(parents=True,exist_ok=True); NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(OUT),'note':str(NOTE),'config_equal_all':config_equal,'deltas':deltas},indent=2))
if __name__=='__main__': main()
