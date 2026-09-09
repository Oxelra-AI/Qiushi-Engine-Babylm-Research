#!/usr/bin/env python3
"""research: EWoK live/disabled/rescaled adapter inference ablation.

Complements the research GlobalPIQA inference ablation. It creates lightweight
proxy checkpoint dirs for the scale1.75 80M checkpoint with adapter_scale set
to live/disabled/rescaled values, then calls the validated research EWoK four-cell
interaction reader on each proxy. No model training is performed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, shutil, subprocess, sys, time, os

def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT=find_user_root()
OUT=ROOT/'experiments/archive/representation_and_objectives/data/scale1p75_ewok_inference_ablation'
PROXY=OUT/'proxy_models'
READ=OUT/'readouts'
SCRIPT=ROOT/'experiments/archive/representation_and_objectives/scripts/ewok_interaction_reader.py'
SRC=ROOT/'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M'
ANCH=ROOT/'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M'
VARIANTS={
 'scale1p75_live': {'src': SRC, 'adapter_enabled': True, 'adapter_scale': 1.75},
 'scale1p75_disabled': {'src': SRC, 'adapter_enabled': False, 'adapter_scale': 0.0},
 'scale1p75_scale0p5': {'src': SRC, 'adapter_enabled': True, 'adapter_scale': 0.5},
 'scale1p75_scale1p0': {'src': SRC, 'adapter_enabled': True, 'adapter_scale': 1.0},
 'scale1p75_scale2p5': {'src': SRC, 'adapter_enabled': True, 'adapter_scale': 2.5},
 'matched_legal16k_80M': {'src': ANCH, 'adapter_enabled': None, 'adapter_scale': None},
}

def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)

def proxy_dir(label, spec):
    src=pathlib.Path(spec['src'])
    dst=PROXY/label
    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in {'pytorch_model.bin','model.safetensors'} or item.is_file() or item.is_symlink():
            if item.name=='config.json':
                cfg=json.loads(item.read_text(encoding='utf-8'))
                if spec['adapter_enabled'] is not None:
                    cfg['adapter_enabled']=spec['adapter_enabled']
                if spec['adapter_scale'] is not None:
                    cfg['adapter_scale']=spec['adapter_scale']
                (dst/'config.json').write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')
            else:
                os.symlink(item.resolve(), dst/item.name)
        elif item.is_dir() and item.name in {'tokenizer','sentencepiece.bpe.model'}:
            os.symlink(item.resolve(), dst/item.name)
    # Copy Python custom modules as symlinks if not already included by file loop; config auto_map needs them.
    for py in src.glob('*.py'):
        target=dst/py.name
        if not target.exists(): os.symlink(py.resolve(), target)
    return dst

def run_reader(label, model):
    out=READ/label
    out.mkdir(parents=True, exist_ok=True)
    cache=OUT/'hf_cache'/label
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    os.environ['HF_HOME']=str((cache/'hf_home').resolve())
    os.environ['HF_HUB_CACHE']=str((cache/'hf_home'/'hub').resolve())
    os.environ['TRANSFORMERS_CACHE']=str((cache/'transformers').resolve())
    os.environ['HF_MODULES_CACHE']=str((cache/'modules').resolve())
    os.environ['HF_DATASETS_CACHE']=str((cache/'datasets').resolve())
    os.environ['TOKENIZERS_PARALLELISM']='false'
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE','HF_DATASETS_CACHE']:
        pathlib.Path(os.environ[k]).mkdir(parents=True, exist_ok=True)
    script_dir=ROOT/'experiments/archive/representation_and_objectives/scripts'
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import torch
    import fw_ewok_interaction_reader as ew
    t0=time.time()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res=ew.run_target(label, pathlib.Path(model), device, threads=8, row_limit=0, row_offset=0, row_batch_size=32, masked_batch_size=128)
    records=res.pop('records')
    ew.write_csv(out/'ewok_interaction_records.csv', records)
    ew.write_csv(out/'ewok_interaction_by_domain.csv', res['by_domain'])
    ew.write_csv(out/'ewok_interaction_by_context_diff.csv', res['by_context_diff'])
    summary=out/'ewok_interaction_summary.json'
    summary.write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if device.type=='cuda':
        torch.cuda.empty_cache()
    return {'label':label,'proxy_model':rel(model),'summary_json':rel(summary),'elapsed_sec':round(time.time()-t0,1),'summary':res['summary']}

def main():
    preflight=('--preflight-only' in sys.argv) or ('--preflight' in sys.argv)
    OUT.mkdir(parents=True, exist_ok=True); PROXY.mkdir(parents=True, exist_ok=True); READ.mkdir(parents=True, exist_ok=True)
    if preflight:
        print(json.dumps({'status':'preflight','src_exists':SRC.exists(),'anchor_exists':ANCH.exists(),'script_exists':SCRIPT.exists(),'out':rel(OUT)},indent=2)); return
    results={}
    for label,spec in VARIANTS.items():
        print(json.dumps({'event':'variant_start','label':label,'utc':now()}), flush=True)
        p=proxy_dir(label,spec)
        results[label]=run_reader(label,p)
        sm=results[label]['summary']
        print(json.dumps({'event':'variant_done','label':label,'accuracy':sm.get('accuracy'),'stable_failure':sm.get('stable_failure')}), flush=True)
    base=results['matched_legal16k_80M']['summary']
    live=results['scale1p75_live']['summary']
    disabled=results['scale1p75_disabled']['summary']
    def delta(a,b):
        out={}
        for k in ['accuracy','stable_failure','stable_failure_frac_all','stable_failure_frac_wrong','within_both_positive_wrong_frac','local_both_actual_over_swapped_positive_wrong_frac']:
            if a.get(k) is not None and b.get(k) is not None: out[k]=a[k]-b[k]
        for k in ['interaction_sum_all','interaction_sum_wrong','interaction_mean_wrong']:
            if isinstance(a.get(k),dict) and isinstance(b.get(k),dict): out[k+'_mean']=a[k].get('mean')-b[k].get('mean')
        return out
    combined={'status':'SCALE1P75_EWOK_INFERENCE_ABLATION_DONE','created_utc':now(),'variants':results,
              'live_minus_disabled':delta(live,disabled),'disabled_minus_matched_step35':delta(disabled,base),'live_minus_matched_step35':delta(live,base),
              'interpretation':'Inference-only EWoK decomposition: live-disabled is immediate adapter output; disabled-matched_step35 is co-trained trajectory plus architecture effects under same legal16k lineage.'}
    outj=OUT/'scale1p75_ewok_inference_ablation_summary.json'
    outj.write_text(json.dumps(combined,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':combined['status'],'summary':rel(outj),'variants':list(results)},indent=2), flush=True)
if __name__=='__main__': main()
