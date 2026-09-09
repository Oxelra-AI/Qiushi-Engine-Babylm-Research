#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
EWOK_OUT = STRICT / 'evaluation_data/full_eval/ewok_filtered_word_tokenize'
SUMMARY = ROOT / 'data/aligned_shuffled_10m_ewok_word_tokenize_filter_summary.json'
RUN = ROOT / 'training/runs/babylm_leadershape_s1_10M_aligned_micro128'
MODEL = (RUN / 'hf_model/chck_10M').resolve()
EVAL_OUT = RUN / 'eval_results_full_ewok_word_tokenize_direct'
SCORE_JSON = ROOT / 'data/s1_10m_full_ewok_word_tokenize_score.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/s1_10m_full_ewok_word_tokenize_score.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/s1_10m_full_ewok_eval.log')

def parse_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding='utf-8', errors='replace')
    m = re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if not m:
        raise RuntimeError(f'Could not parse average from {report}\n{txt[:1200]}')
    return float(m.group(1))

def setup_env():
    env = os.environ.copy()
    env['NLTK_DATA'] = str((ROOT / 'data/nltk_data').resolve())
    hf = ROOT / 'training/hf_home'
    env['HF_HOME'] = str(hf.resolve())
    env['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    env['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    env['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    for k in ['HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_MODULES_CACHE']:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env

def main():
    t0=time.time(); LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('w', encoding='utf-8') as logf:
        def log(msg: str):
            print(msg, flush=True); logf.write(msg+'\n'); logf.flush()
        if not MODEL.exists(): raise RuntimeError(f'Missing S1 10M checkpoint {MODEL}')
        if not SUMMARY.exists(): raise RuntimeError(f'Missing EWoK filter summary {SUMMARY}')
        if not EWOK_OUT.exists() or not any(EWOK_OUT.glob('*.jsonl')):
            raise RuntimeError(f'Missing filtered EWoK data {EWOK_OUT}')
        EVAL_OUT.mkdir(parents=True, exist_ok=True)
        cmd=[sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(MODEL),'--backend','mlm','--task','ewok','--data_path',str(EWOK_OUT.resolve()),'--save_predictions','--batch_size','64','--output_dir',str(EVAL_OUT.resolve())]
        log('$ '+' '.join(cmd))
        p=subprocess.run(cmd, cwd=str(STRICT), env=setup_env(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log(p.stdout[-10000:])
        if p.returncode != 0: raise RuntimeError(f'S1 10M EWoK eval failed {p.returncode}\n{p.stdout[-12000:]}')
        hits=sorted(EVAL_OUT.rglob('best_temperature_report.txt'))
        if len(hits) != 1: raise RuntimeError(f'Expected one report under {EVAL_OUT}, got {hits[:10]}')
        report=hits[0]; pred=report.parent/'predictions.json'; score=parse_avg(report)
        metrics=json.loads((RUN/'scientific_metrics.json').read_text())
        payload={'status':'S1_10M_FULL_EWOK_WORD_TOKENIZE','model':'S1 12x384 DeBERTa-v2 baseline16k WWM official-corpus 10M','model_path_direct':str(MODEL),'backend':'mlm','ewok_full_score':score,'training_core':{k:metrics.get(k) for k in ['word_exposure','actual_training_steps','loss_first','loss_last','optimizer_steps','lr_schedule_total_steps']},'filter_summary':str(SUMMARY),'report':str(report),'predictions':str(pred),'eval_output_dir':str(EVAL_OUT),'elapsed_sec':time.time()-t0}
        SCORE_JSON.parent.mkdir(parents=True, exist_ok=True); SCORE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        NOTE.write_text('\n'.join(['# research — S1 10M full EWoK baseline', '', f'Evidence JSON: `{SCORE_JSON}`', f'Report: `{report}`', '', f'Full EWoK score: **{score:.2f}**'])+'\n', encoding='utf-8')
        log(json.dumps({'status':payload['status'],'score':score,'score_json':str(SCORE_JSON),'elapsed_sec':payload['elapsed_sec']}, indent=2))
if __name__=='__main__': main()
