#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
FULL = STRICT / 'evaluation_data/full_eval'
FAST = STRICT / 'evaluation_data/fast_eval'
RUN = ROOT / 'training/runs/babylm_lamb_refdefault_s1_10M'
MODEL = (RUN / 'hf_model/chck_10M').resolve()
OUTDIR = (RUN / 'eval_results_available_direct').resolve()
OUT_JSON = ROOT / 'data/lamb_s1_10m_available_coordinate.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/lamb_s1_10m_available_coordinate.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/lamb_s1_10m_eval.log')
BACKEND = 'mlm'
REV = 'direct_chck_10M'
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
    hf_home=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf_home.resolve()); env['HF_HUB_CACHE']=str((hf_home/'hub').resolve())
    env['TRANSFORMERS_CACHE']=str((hf_home/'transformers').resolve())
    env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env

def run(cmd, env, cwd=STRICT, logf=None):
    line='$ '+' '.join(map(str,cmd)); print(line, flush=True)
    if logf: logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run([str(x) for x in cmd], cwd=str(cwd), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-2500:], flush=True)
    if logf: logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode!=0:
        raise RuntimeError(f'command failed {p.returncode}: {line}\n{p.stdout[-8000:]}')

def read_avg(report: pathlib.Path)->float:
    txt=report.read_text(encoding='utf-8', errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if m: return float(m.group(1))
    vals=re.findall(r'(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)', txt, flags=re.I)
    if vals:
        v=float(vals[-1]); return v*(100 if v<=1 else 1)
    raise RuntimeError(f'could not parse avg from {report}\n{txt[:1200]}')

def read_reading(report: pathlib.Path)->dict[str,float]:
    txt=report.read_text(encoding='utf-8', errors='replace')
    out={}
    for label,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(label)+r':\s*([0-9.\-]+)', txt)
        if not m: raise RuntimeError(f'could not parse {label} from {report}\n{txt[:1200]}')
        out[key]=float(m.group(1))
    return out

def main():
    if not MODEL.exists(): raise RuntimeError(f'Missing direct checkpoint {MODEL}')
    env=setup_env(); OUTDIR.mkdir(parents=True, exist_ok=True); LOG.parent.mkdir(parents=True, exist_ok=True)
    path_status={}
    scores={}; reports={}; provenance={}
    with LOG.open('a', encoding='utf-8') as logf:
        for col, task, data_path, dataset_name in TASKS:
            path_status[col]={'path':str(data_path),'exists':data_path.exists(),'num_files':sum(1 for _ in data_path.rglob('*')) if data_path.exists() else 0}
            if not data_path.exists() or not any(data_path.rglob('*')):
                raise RuntimeError(f'missing eval data for {col}: {data_path}')
            run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run',
                 '--model_path_or_name', str(MODEL), '--backend', BACKEND, '--task', task,
                 '--data_path', str(data_path.resolve()), '--save_predictions', '--revision_name', REV,
                 '--batch_size','64','--output_dir',str(OUTDIR)], env, logf=logf)
            report=OUTDIR / 'chck_10M' / REV / 'zero_shot' / BACKEND / task / dataset_name / 'best_temperature_report.txt'
            if not report.exists():
                # Some eval versions use checkpoint basename twice or model path stem differently; recover unique report.
                candidates=list(OUTDIR.rglob(f'zero_shot/{BACKEND}/{task}/{dataset_name}/best_temperature_report.txt'))
                if len(candidates)!=1:
                    raise RuntimeError(f'report not found for {col}; expected {report}; candidates={candidates[:5]}')
                report=candidates[0]
            reports[col]=str(report); scores[col]=read_avg(report); provenance[col]='direct_checkpoint_eval'
        reading_csv=FULL/'reading/reading_data.csv'
        path_status['reading']={'path':str(reading_csv),'exists':reading_csv.exists(),'bytes':reading_csv.stat().st_size if reading_csv.exists() else 0}
        run([sys.executable,'-m','evaluation_pipeline.reading.run', '--model_path_or_name', str(MODEL), '--backend', BACKEND,
             '--data_path', str(reading_csv.resolve()), '--revision_name', REV, '--output_dir', str(OUTDIR)], env, logf=logf)
        candidates=list(OUTDIR.rglob(f'zero_shot/{BACKEND}/reading/report.txt'))
        if not candidates: raise RuntimeError('reading report not found')
        reading_report=candidates[-1]
        reports['reading']=str(reading_report); scores.update(read_reading(reading_report)); provenance['reading']='direct_checkpoint_eval'
    gp_mean=(scores['global_piqa_parallel']+scores['global_piqa_nonparallel'])/2.0
    reading_mean=(scores['reading_eye_tracking']+scores['reading_self_paced'])/2.0
    metrics=json.loads((RUN/'scientific_metrics.json').read_text())
    cfg=json.loads((MODEL/'config.json').read_text())
    payload={'status':'LAMB_S1_10M_AVAILABLE_COORDINATE','run':str(RUN),'model_path_direct':str(MODEL),'backend':BACKEND,'revision_label':REV,
             'training_core':{k:metrics.get(k) for k in ['parameter_count','embedding_parameter_count','non_embedding_parameter_count','word_exposure','actual_training_steps','optimizer_steps','effective_batch_size','micro_batch_size','lr_schedule_total_steps','loss_first','loss_last','masked_tokens_total','masked_tokens_per_whitespace_word','hidden_size','n_layer','n_head','intermediate_size','vocab_size']},
             'config_core':{k:cfg.get(k) for k in ['model_type','hidden_size','num_hidden_layers','num_attention_heads','intermediate_size','vocab_size','relative_attention','pos_att_type','position_buckets','max_position_embeddings']},
             'scores':scores,'derived_columns':{'GlobalPIQA_mean_parallel_nonparallel':gp_mean,'Reading_mean_eye_selfpaced':reading_mean},
             'reports':reports,'provenance':provenance,'path_status':path_status,
             'missing_columns':['EWoK_full_if_not_run_here','SuperGLUE','AoA'],
             'note':'Direct checkpoint path evaluation; LAMB S1 10M is a stability/early-dynamics screen using reference-default LAMB on official corpus/baseline16k/flat WWM, not a full 100M trajectory or leader reproduction.'}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True); OUT_JSON.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research — LAMB reference-default S1 10M available coordinate','',f'Evidence JSON: `{OUT_JSON}`','',f'Direct checkpoint: `{MODEL}`','', '| column/task | score |','|---|---:|']
    for key,label in [('blimp','BLiMP'),('supplement','Supplement'),('entity_tracking','Entity'),('comps','COMPS'),('global_piqa_parallel','GlobalPIQA parallel'),('global_piqa_nonparallel','GlobalPIQA nonparallel')]:
        lines.append(f'| {label} | {scores[key]:.2f} |')
    lines += [f'| GlobalPIQA mean | {gp_mean:.2f} |', f'| Reading eye | {scores["reading_eye_tracking"]:.2f} |', f'| Reading self-paced | {scores["reading_self_paced"]:.2f} |', f'| Reading mean | {reading_mean:.2f} |']
    lines += ['', 'This is a 10M LAMB early-dynamics screen on official data with baseline16k tokenizer, S1 12x384 shape, and flat WWM; exact leader data remains gated and this does not substitute for the possible full 100M LAMB trajectory.']
    OUT_NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out':str(OUT_JSON),'note':str(OUT_NOTE),'scores':scores,'global_piqa_mean':gp_mean,'reading_mean':reading_mean},indent=2))
if __name__=='__main__': main()
