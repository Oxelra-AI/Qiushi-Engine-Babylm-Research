#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT/'repos/babylm-eval/strict'
RUNS = {
    'random_quality': ROOT/'training/runs/fineweb_random_quality_debertav2_8x480_1M_b128_seed42',
    'relation_explicit': ROOT/'training/runs/fineweb_relation_explicit_debertav2_8x480_1M_b128_seed42',
}
OUT_JSON = ROOT/'data/fineweb_relation_vs_random_1m_profile.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/fineweb_relation_vs_random_1m_profile.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/fineweb_relation_vs_random_1m_profile.log')
TASKS = [
    ('blimp_fast','blimp','evaluation_data/fast_eval/blimp_fast','blimp_fast'),
    ('supplement_fast','blimp','evaluation_data/fast_eval/supplement_fast','supplement_fast'),
    ('ewok_fast','ewok','evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast','ewok_fast'),
    ('entity_tracking_fast','entity_tracking','evaluation_data/fast_eval/entity_tracking_fast','entity_tracking_fast'),
    ('comps','comps','evaluation_data/full_eval/comps','comps'),
]
COLS = ['blimp_fast','supplement_fast','ewok_fast','entity_tracking_fast','comps','reading_eye_tracking','reading_self_paced','Reading_mean']

def setup_env():
    env=os.environ.copy(); hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve())
    env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env

def run(cmd, env, logf):
    line='$ '+' '.join(map(str,cmd)); print(line, flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-3000:], flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode!=0:
        raise RuntimeError(f'failed {p.returncode}: {line}\n{p.stdout[-8000:]}')

def read_avg(report:pathlib.Path):
    txt=report.read_text(encoding='utf-8', errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if not m: raise RuntimeError(f'cannot parse average from {report}\n{txt[:1200]}')
    return float(m.group(1))

def read_reading(report:pathlib.Path):
    txt=report.read_text(encoding='utf-8', errors='replace'); out={}
    for label,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(label)+r':\s*([0-9.\-]+)', txt)
        if not m: raise RuntimeError(f'cannot parse {label} from {report}\n{txt}')
        out[key]=float(m.group(1))
    out['Reading_mean']=(out['reading_eye_tracking']+out['reading_self_paced'])/2.0
    return out

def loadj(p): return json.loads(p.read_text(encoding='utf-8'))

def profile_one(name, run_dir, env, logf):
    ckpt=run_dir/'hf_model'/'chck_1M'
    if not (ckpt/'config.json').exists(): raise FileNotFoundError(ckpt/'config.json')
    outdir=run_dir/'eval_fast_profile'
    scores={}; reports={}
    for task_name, task, data_path, ds_name in TASKS:
        run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str((run_dir/'hf_model').resolve()),'--backend','mlm','--task',task,'--data_path',data_path,'--save_predictions','--revision_name','chck_1M','--batch_size','64','--output_dir',str(outdir.resolve())], env, logf)
        report=outdir/'hf_model'/'chck_1M'/'zero_shot'/'mlm'/task/ds_name/'best_temperature_report.txt'
        scores[task_name]=read_avg(report); reports[task_name]=str(report)
    run([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str((run_dir/'hf_model').resolve()),'--backend','mlm','--data_path','evaluation_data/fast_eval/reading/reading_data.csv','--revision_name','chck_1M','--output_dir',str(outdir.resolve())], env, logf)
    rr=outdir/'hf_model'/'chck_1M'/'zero_shot'/'mlm'/'reading'/'report.txt'
    scores.update(read_reading(rr)); reports['reading']=str(rr)
    metrics=loadj(run_dir/'scientific_metrics.json')
    return {'run_dir':str(run_dir),'scores':scores,'reports':reports,'metrics_summary':{
        'parameter_count':metrics['parameter_count'],'word_exposure':metrics['word_exposure'],'steps':metrics['actual_training_steps'],
        'loss_first':metrics['loss_first'],'loss_last':metrics['loss_last'],'batch_size':128,
        'truncated_fraction':metrics['tokenization_coupling_summary']['truncated_example_fraction'],
        'tokens_per_word_kept':metrics['tokenization_coupling_summary']['kept_tokens_per_whitespace_word'],
        'masked_tokens_per_word':metrics['masked_tokens_per_whitespace_word']}}

def main():
    t0=time.time(); env=setup_env(); LOG.parent.mkdir(parents=True, exist_ok=True)
    rows={}
    with LOG.open('a', encoding='utf-8') as logf:
        for name, run_dir in RUNS.items():
            rows[name]=profile_one(name, run_dir, env, logf)
    r=rows['random_quality']['scores']; e=rows['relation_explicit']['scores']
    delta={k: round(e[k]-r[k],4) for k in COLS}
    payload={'status':'FINEWEB_RELATION_VS_RANDOM_1M_PROFILE','arms':rows,'relation_minus_random':delta,'elapsed_sec':time.time()-t0,
             'interpretation_note':'Matched same-source FineWeb-Edu 1M b128 short run; b128 changes update count relative to protected b256, but both arms are matched to each other.'}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True); OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research FineWeb relation-explicit vs random-quality 1M profile','',f'Evidence JSON: `{OUT_JSON}`','', 'Both arms: FineWeb-Edu sample-10BT, 1M words, DeBERTa-v2 8x480, baseline16k, WWM p=0.15, seed/init fixed, batch 128 after batch-256 OOM. Difference is same-source relation-explicit filtering versus random-quality FineWeb text.','', '| column | random_quality | relation_explicit | relation-random |','|---|---:|---:|---:|']
    for k in COLS:
        lines.append(f'| {k} | {r[k]:.4f} | {e[k]:.4f} | {delta[k]:+.4f} |')
    lines += ['','## Training summary','', '| arm | loss first | loss last | steps | trunc frac | kept tokens/word | masked tokens/word |','|---|---:|---:|---:|---:|---:|---:|']
    for name,d in rows.items():
        m=d['metrics_summary']; lines.append(f"| {name} | {m['loss_first']:.4f} | {m['loss_last']:.4f} | {m['steps']} | {m['truncated_fraction']:.4f} | {m['tokens_per_word_kept']:.4f} | {m['masked_tokens_per_word']:.4f} |")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True); OUT_NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'out':str(OUT_JSON),'note':str(OUT_NOTE),'delta':delta,'elapsed_sec':payload['elapsed_sec']}, indent=2))
if __name__=='__main__': main()
