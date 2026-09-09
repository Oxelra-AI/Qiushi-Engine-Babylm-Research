#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, sys, time
import torch
from transformers import AutoModelForMaskedLM, AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'
sys.path.insert(0,str(STRICT.resolve()))
from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, StepConfig, load_eval  # noqa:E402
from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor  # noqa:E402
from evaluation_pipeline.utils import AoAEvaluator  # noqa:E402

WORD_PATH=(STRICT/'evaluation_data/full_eval/aoa/cdi_childes.json').resolve()
CDI_HUMAN=(STRICT/'evaluation_data/full_eval/aoa/cdi_human.csv').resolve()
OUT=ROOT/'data/s1_s2_100m_aoa_results.json'
LOG=(ROOT.parents[2] / 'research/notes/initial_model_studies/s1_s2_aoa_eval.log')
MODELS={
 's1': ROOT/'training/runs/babylm_leadershape_s1_100M_aligned_micro128/hf_model',
 's2': ROOT/'training/runs/babylm_true_s2_100M_wordclock_wwm70_len64256/hf_model',
}
OUTDIRS={
 's1': ROOT/'training/runs/babylm_leadershape_s1_100M_aligned_micro128/eval_results_step203_aoa_local_ckpts',
 's2': ROOT/'training/runs/babylm_true_s2_100M_wordclock_wwm70_len64256/eval_results_step203_aoa_local_ckpts',
}

def setup_env():
    hf=ROOT/'training/hf_home'
    os.environ['HF_HOME']=str(hf.resolve()); os.environ['HF_HUB_CACHE']=str((hf/'hub').resolve())
    os.environ['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

class LocalCheckpointSurprisalExtractor(StepSurprisalExtractor):
    def _step_path(self, step:str)->pathlib.Path:
        p=pathlib.Path(self.model_name)/str(step)
        if not p.exists(): raise FileNotFoundError(f'Missing local checkpoint path: {p}')
        return p
    def load_model_for_step(self, step:str):
        m=AutoModelForMaskedLM.from_pretrained(self._step_path(step),trust_remote_code=True)
        m=m.to(self.device); m.eval(); return m
    def load_tokenizer_for_step(self, step:str):
        p=self._step_path(step)
        try: processor=AutoProcessor.from_pretrained(p,trust_remote_code=True,padding_side='right')
        except (ValueError,KeyError): processor=PreTrainedTokenizerFast.from_pretrained(p,padding_side='right')
        tok=processor.tokenizer if hasattr(processor,'tokenizer') else processor
        return processor,tok

def summarize_steps(rows):
    counts={}; mean={}
    for r in rows:
        st=r['step']; counts[st]=counts.get(st,0)+1; mean[st]=mean.get(st,0.0)+float(r['surprisal'])
    return counts,{k:mean[k]/counts[k] for k in counts}

def run_one(name,model_root,outdir,device,target_words,contexts,cfg):
    missing=[s for s in cfg.steps if not (model_root/s).exists()]
    if missing: raise RuntimeError(f'{name} missing checkpoints: {missing}')
    outdir.mkdir(parents=True,exist_ok=True)
    extractor=LocalCheckpointSurprisalExtractor(config=cfg,model_name=str(model_root.resolve()),backend='mlm',device=device)
    result_dir=outdir/'hf_model_local_ckpts/main/zero_shot/mlm/AoA_word'
    score_path=result_dir/'aoa_score.json'; surprisal_path=result_dir/'surprisal.json'
    if score_path.exists() and surprisal_path.exists():
        score=json.loads(score_path.read_text()).get('aoa',json.loads(score_path.read_text()).get('curve_fitness'))
        data=json.loads(surprisal_path.read_text())
    else:
        data=extractor.analyze_steps(contexts=contexts,target_words=target_words,resume_path=None)
        result_dir.mkdir(parents=True,exist_ok=True); JsonProcessor.save_json(data,surprisal_path)
        tok=AutoTokenizer.from_pretrained(model_root,trust_remote_code=True)
        score=AoAEvaluator(CDI_HUMAN).compute_curve_fitness(data,tok)['curve_fitness']
        JsonProcessor.save_json({'aoa':score},score_path)
    rows=data.get('results',[]); counts,means=summarize_steps(rows)
    return {'model_root':str(model_root),'output_dir':str(outdir),'score_path':str(score_path),'surprisal_path':str(surprisal_path),'aoa':float(score),'num_rows':len(rows),'num_steps':len(counts),'step_counts':counts,'step_mean_surprisal':means,'missing_checkpoints':missing}

def main():
    setup_env(); t0=time.time(); LOG.parent.mkdir(parents=True,exist_ok=True); OUT.parent.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    target_words,contexts=load_eval(WORD_PATH,20,False)
    cfg=StepConfig(resume=False,track='strict-small',file_path=None,debug=False)
    payload={'status':'S1_S2_AOA','device':device,'word_path':str(WORD_PATH),'cdi_human':str(CDI_HUMAN),'expected_steps':cfg.steps,'models':{}}
    for name,root in MODELS.items():
        rec=run_one(name,root,OUTDIRS[name],device,target_words,contexts,cfg)
        payload['models'][name]=rec; OUT.write_text(json.dumps(payload,indent=2)+'\n')
    payload['elapsed_sec']=round(time.time()-t0,1)
    OUT.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'out':str(OUT),'aoa':{k:v['aoa'] for k,v in payload['models'].items()},'elapsed_sec':payload['elapsed_sec']},indent=2))
if __name__=='__main__': main()
