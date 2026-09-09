#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'
FULL=STRICT/'evaluation_data/full_eval'
RUN=ROOT/'training/runs/babylm_true_s2_100M_wordclock_wwm70_len64256'
MODEL=(RUN/'hf_model/chck_100M').resolve()
OUTDIR=(RUN/'eval_results_chck100M_available_direct').resolve()
OUT_JSON=ROOT/'data/true_s2_100m_available_coordinate.json'
OUT_NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/true_s2_100m_available_coordinate.md')
LOG=(ROOT.parents[2] / 'research/notes/initial_model_studies/true_s2_100m_eval.log')
BACKEND='mlm'; REV='direct_chck_100M'
TASKS=[
 ('blimp','blimp',FULL/'blimp_filtered','blimp_filtered'),
 ('supplement','blimp',FULL/'supplement_filtered','supplement_filtered'),
 ('entity_tracking','entity_tracking',FULL/'entity_tracking','entity_tracking'),
 ('comps','comps',FULL/'comps','comps'),
 ('global_piqa_parallel','global_piqa_parallel',FULL/'global_piqa_parallel','global_piqa_parallel'),
 ('global_piqa_nonparallel','global_piqa_nonparallel',FULL/'global_piqa_nonparallel','global_piqa_nonparallel')]

def env_setup():
    env=os.environ.copy(); hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve()); env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve()); env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']: pathlib.Path(env[k]).mkdir(parents=True,exist_ok=True)
    return env

def run(cmd, env, logf):
    line='$ '+' '.join(map(str,cmd)); print(line,flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run([str(x) for x in cmd], cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-2500:],flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode: raise RuntimeError(f'command failed {p.returncode}: {line}\n{p.stdout[-8000:]}')

def read_avg(report):
    txt=pathlib.Path(report).read_text(encoding='utf-8',errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if m: return float(m.group(1))
    vals=re.findall(r'(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)', txt, flags=re.I)
    if vals:
        v=float(vals[-1]); return v*(100 if v<=1 else 1)
    raise RuntimeError(f'could not parse {report}\n{txt[:1000]}')

def read_reading(report):
    txt=pathlib.Path(report).read_text(encoding='utf-8',errors='replace'); out={}
    for label,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(label)+r':\s*([0-9.\-]+)', txt)
        if not m: raise RuntimeError(f'could not parse {label} from {report}')
        out[key]=float(m.group(1))
    return out

def main():
    if not MODEL.exists(): raise RuntimeError(f'missing {MODEL}')
    OUTDIR.mkdir(parents=True,exist_ok=True); LOG.parent.mkdir(parents=True,exist_ok=True); env=env_setup()
    scores={}; reports={}; paths={}; prov={}
    with LOG.open('a',encoding='utf-8') as logf:
        for col,task,data,dataset in TASKS:
            paths[col]={'path':str(data),'exists':data.exists(),'num_files':sum(1 for _ in data.rglob('*')) if data.exists() else 0}
            run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(MODEL),'--backend',BACKEND,'--task',task,'--data_path',str(data.resolve()),'--save_predictions','--revision_name',REV,'--batch_size','64','--output_dir',str(OUTDIR)],env,logf)
            cand=sorted(OUTDIR.rglob(f'zero_shot/{BACKEND}/{task}/{dataset}/best_temperature_report.txt'), key=lambda p:str(p))
            if not cand: raise RuntimeError(f'no report {col}')
            reports[col]=str(cand[-1]); scores[col]=read_avg(cand[-1]); prov[col]='direct_checkpoint_eval'
        reading=FULL/'reading/reading_data.csv'; paths['reading']={'path':str(reading),'exists':reading.exists(),'bytes':reading.stat().st_size if reading.exists() else 0}
        run([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(MODEL),'--backend',BACKEND,'--data_path',str(reading.resolve()),'--revision_name',REV,'--output_dir',str(OUTDIR)],env,logf)
        cand=sorted(OUTDIR.rglob(f'zero_shot/{BACKEND}/reading/report.txt'), key=lambda p:str(p))
        if not cand: raise RuntimeError('no reading report')
        reports['reading']=str(cand[-1]); scores.update(read_reading(cand[-1])); prov['reading']='direct_checkpoint_eval'
    gp=(scores['global_piqa_parallel']+scores['global_piqa_nonparallel'])/2; rd=(scores['reading_eye_tracking']+scores['reading_self_paced'])/2
    metrics=json.loads((RUN/'scientific_metrics.json').read_text()); cfg=json.loads((MODEL/'config.json').read_text())
    payload={'status':'TRUE_S2_100M_AVAILABLE_COORDINATE','run':str(RUN),'model_path_direct':str(MODEL),'backend':BACKEND,'revision_label':REV,'training_core':{k:metrics.get(k) for k in ['parameter_count','embedding_parameter_count','non_embedding_parameter_count','word_exposure','actual_training_steps','optimizer_steps','effective_batch_size','micro_batch_size','lr_schedule_total_steps','loss_first','loss_last','masked_tokens_total','hidden_size','n_layer','n_head','intermediate_size','vocab_size']},'config_core':{k:cfg.get(k) for k in ['model_type','hidden_size','num_hidden_layers','num_attention_heads','intermediate_size','vocab_size','relative_attention','pos_att_type','position_buckets','max_position_embeddings']},'scores':scores,'derived_columns':{'GlobalPIQA_mean_parallel_nonparallel':gp,'Reading_mean_eye_selfpaced':rd},'reports':reports,'provenance':prov,'path_status':paths,'missing_columns':['EWoK_full_if_not_run_here','SuperGLUE','AoA'],'note':'Direct checkpoint path evaluation; true S2 word-clock curriculum on S1 base with official corpus/baseline16k, not leader reproduction.'}
    OUT_JSON.parent.mkdir(parents=True,exist_ok=True); OUT_JSON.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research — true S2 word-clock curriculum 100M available coordinate','',f'Evidence JSON: `{OUT_JSON}`','',f'Direct checkpoint: `{MODEL}`','','| column/task | score |','|---|---:|']
    for key,label in [('blimp','BLiMP'),('supplement','Supplement'),('entity_tracking','Entity'),('comps','COMPS'),('global_piqa_parallel','GlobalPIQA parallel'),('global_piqa_nonparallel','GlobalPIQA nonparallel')]: lines.append(f'| {label} | {scores[key]:.2f} |')
    lines += [f'| GlobalPIQA mean | {gp:.2f} |',f'| Reading eye | {scores["reading_eye_tracking"]:.2f} |',f'| Reading self-paced | {scores["reading_self_paced"]:.2f} |',f'| Reading mean | {rd:.2f} |']
    OUT_NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out':str(OUT_JSON),'note':str(OUT_NOTE),'scores':scores,'global_piqa_mean':gp,'reading_mean':rd},indent=2))
if __name__=='__main__': main()
