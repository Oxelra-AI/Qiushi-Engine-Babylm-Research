#!/usr/bin/env python3
"""research matched RTD-vs-WWM 1M evaluation and context-intervention probe.

This script has two independent evidence outputs:
1) Direct-checkpoint fast profile for the matched research RTD+MLM and WWM-only
   seed42 chck_1M checkpoints (no local parent+revision artifact).
2) Corrected context intervention probe: the SAME earlier-context
   deletion/shuffle/unrelated-prefix perturbations are applied to both matched
   WWM and RTD+MLM discriminators, and the comparison is on fixed later-token
   MLM logits. This tests whether RTD changed credit assignment beyond the WWM
   model's pre-existing recoverable history sensitivity.
"""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, random, re, subprocess, sys, time
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT/'repos/babylm-eval/strict'
RTD_CKPT = ROOT/'training/runs/hybrid_rtd_mlm_debertav2_8x480_official_1M_b128_seed42/hf_model/chck_1M'
WWM_CKPT = ROOT/'training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched/hf_model/chck_1M'
OUT_JSON = ROOT/'data/rtd_wwm_1m_eval_and_context_probe.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/rtd_wwm_1m_eval_and_context_probe.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/rtd_wwm_1m_eval_and_context_probe.log')
TASKS = [
    ('blimp_fast','blimp','evaluation_data/fast_eval/blimp_fast','blimp_fast'),
    ('supplement_fast','blimp','evaluation_data/fast_eval/supplement_fast','supplement_fast'),
    ('ewok_fast','ewok','evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast','ewok_fast'),
    ('entity_tracking_fast','entity_tracking','evaluation_data/fast_eval/entity_tracking_fast','entity_tracking_fast'),
    ('comps','comps','evaluation_data/full_eval/comps','comps'),
]
COLS = ['blimp_fast','supplement_fast','ewok_fast','entity_tracking_fast','comps','reading_eye_tracking','reading_self_paced','Reading_mean']


def setup_env():
    env = os.environ.copy()
    hf = ROOT/'training/hf_home'
    env['HF_HOME'] = str(hf.resolve())
    env['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    env['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    env['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    env['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def sha16(p: pathlib.Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1<<20), b''):
            h.update(c)
    return h.hexdigest()[:16]


def run(cmd, env, logf, cwd=None):
    line='$ '+' '.join(map(str,cmd)); print(line, flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run(cmd,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(p.stdout[-2000:], flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode != 0: raise RuntimeError(f'failed {p.returncode}: {line}\n{p.stdout[-8000:]}')


def read_avg(report: pathlib.Path) -> float:
    txt=report.read_text(encoding='utf-8',errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)',txt)
    if not m: raise RuntimeError(f'cannot parse avg {report}\n{txt[:1000]}')
    return float(m.group(1))


def read_reading(report: pathlib.Path) -> dict:
    txt=report.read_text(encoding='utf-8',errors='replace'); out={}
    for label,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(label)+r':\s*([0-9.\-]+)',txt)
        if not m: raise RuntimeError(f'cannot parse {label} {report}\n{txt}')
        out[key]=float(m.group(1))
    out['Reading_mean']=(out['reading_eye_tracking']+out['reading_self_paced'])/2
    return out


def fast_profile_one(label: str, ckpt: pathlib.Path, env, logf) -> dict:
    outdir = ckpt.parent.parent/f'eval_step305_direct_{label}'
    scores={}; reports={}
    for task_name,task,data_path,ds_name in TASKS:
        run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run',
             '--model_path_or_name',str(ckpt.resolve()),'--backend','mlm','--task',task,
             '--data_path',data_path,'--save_predictions','--batch_size','64','--output_dir',str(outdir.resolve())],
            env,logf,cwd=str(STRICT))
        report=outdir/ckpt.name/'main'/'zero_shot'/'mlm'/task/ds_name/'best_temperature_report.txt'
        scores[task_name]=read_avg(report); reports[task_name]=str(report)
    run([sys.executable,'-m','evaluation_pipeline.reading.run',
         '--model_path_or_name',str(ckpt.resolve()),'--backend','mlm',
         '--data_path','evaluation_data/fast_eval/reading/reading_data.csv','--output_dir',str(outdir.resolve())],
        env,logf,cwd=str(STRICT))
    rr=outdir/ckpt.name/'main'/'zero_shot'/'mlm'/'reading'/'report.txt'
    scores.update(read_reading(rr)); reports['reading']=str(rr)
    return {'checkpoint_path':str(ckpt),'model_safetensors_sha16':sha16(ckpt/'model.safetensors'),'scores':scores,'reports':reports}

# ---- context probe ----
@dataclass
class ProbeCase:
    case_id: int
    prefix_sents: list[str]
    later_sent: str
    target_word: str
    target_word_index: int
    unrelated_prefix: str

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]{2,}$")
STOP = set('the a an and or but if then than as of in on at to from for with without by about into over under is are was were be been being do does did have has had this that these those it its they them he she we you i not no yes can could would should will may might'.split())

def split_sents(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if len(s.strip().split()) >= 5]

def candidate_target(sent: str, tokenizer) -> tuple[str,int] | None:
    toks = sent.split()
    # prefer later content words that are single baseline-BPE tokens with leading-space form
    for idx in range(max(0,len(toks)//3), len(toks)):
        w = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", toks[idx])
        if not _WORD_RE.match(w) or w.lower() in STOP: continue
        ids = tokenizer(' '+w, add_special_tokens=False)['input_ids']
        if len(ids)==1:
            return w, idx
    return None

def make_masked_later(sent: str, target_index: int, tokenizer) -> str:
    toks=sent.split(); toks[target_index]=tokenizer.mask_token
    return ' '.join(toks)

def load_probe_cases(tokenizer, n_cases=160, seed=305) -> list[ProbeCase]:
    # Use official raw dataset already downloaded by research WWM baseline; non-evaluation training text.
    raw_dir = ROOT/'training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched/raw_dataset'
    files = list(raw_dir.glob('*.train.txt'))
    if not files:
        raw_dir = ROOT/'training/runs/hybrid_rtd_mlm_debertav2_8x480_official_1M_b128_seed42/raw_dataset'
        files = list(raw_dir.glob('*.train.txt'))
    passages=[]
    for fp in sorted(files):
        with fp.open('r',encoding='utf-8',errors='replace') as f:
            for line in f:
                sents=split_sents(line)
                if len(sents)>=3 and len(line.split()) <= 180:
                    passages.append((fp.name,sents,line.strip()))
                if len(passages) > 4000: break
        if len(passages) > 4000: break
    rng=random.Random(seed); rng.shuffle(passages)
    unrelated_pool=[' '.join(x[1][:-1]) for x in passages if len(x[1])>=2]
    cases=[]
    for _,sents,_ in passages:
        prefix=sents[:-1]; later=sents[-1]
        if len(prefix)<2: continue
        cand=candidate_target(later, tokenizer)
        if cand is None: continue
        unrelated = rng.choice(unrelated_pool)
        if unrelated.strip() == ' '.join(prefix).strip(): continue
        cases.append(ProbeCase(len(cases), prefix, later, cand[0], cand[1], unrelated))
        if len(cases)>=n_cases: break
    if len(cases)<20: raise RuntimeError(f'too few cases {len(cases)}')
    return cases

def logprob_target(model, tokenizer, text: str, target_word: str, device) -> float:
    enc=tokenizer(text, add_special_tokens=False, truncation=True, max_length=256, return_tensors='pt')
    input_ids=enc['input_ids'].to(device); attn=enc['attention_mask'].to(device)
    mask_positions=(input_ids==tokenizer.mask_token_id).nonzero(as_tuple=False)
    if mask_positions.numel()==0: return float('nan')
    pos=int(mask_positions[-1,1].item())
    tid=tokenizer(' '+target_word, add_special_tokens=False)['input_ids'][0]
    with torch.no_grad():
        logits=model(input_ids=input_ids, attention_mask=attn).logits[0,pos]
        lp=torch.log_softmax(logits, dim=-1)[tid].item()
    return float(lp)

def context_probe(env) -> dict:
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer=AutoTokenizer.from_pretrained(str(WWM_CKPT.resolve()), use_fast=True)
    if tokenizer.mask_token_id is None: raise RuntimeError('no mask token')
    cases=load_probe_cases(tokenizer, n_cases=160)
    models={
        'wwm': AutoModelForMaskedLM.from_pretrained(str(WWM_CKPT.resolve()), trust_remote_code=True).to(device).eval(),
        'rtd_mlm': AutoModelForMaskedLM.from_pretrained(str(RTD_CKPT.resolve()), trust_remote_code=True).to(device).eval(),
    }
    rows=[]
    rng=random.Random(3051)
    for c in cases:
        prefix=' '.join(c.prefix_sents)
        shuf=list(c.prefix_sents); rng.shuffle(shuf); shuf=' '.join(shuf)
        masked_later=make_masked_later(c.later_sent, c.target_word_index, tokenizer)
        variants={
            'full': prefix+' '+masked_later,
            'deleted': masked_later,
            'shuffled': shuf+' '+masked_later,
            'unrelated': c.unrelated_prefix+' '+masked_later,
        }
        rec={'case_id':c.case_id,'target_word':c.target_word,'later_sent':c.later_sent,'prefix_words':len(prefix.split()),'unrelated_words':len(c.unrelated_prefix.split())}
        for mname,model in models.items():
            vals={k:logprob_target(model,tokenizer,v,c.target_word,device) for k,v in variants.items()}
            rec[mname]=vals
            rec[f'{mname}_delta_full_minus_deleted']=vals['full']-vals['deleted']
            rec[f'{mname}_delta_full_minus_shuffled']=vals['full']-vals['shuffled']
            rec[f'{mname}_delta_full_minus_unrelated']=vals['full']-vals['unrelated']
            rec[f'{mname}_specificity_vs_unrelated']=(vals['full']-vals['deleted'])-(vals['full']-vals['unrelated'])
        rec['rtd_minus_wwm_delta_deleted']=rec['rtd_mlm_delta_full_minus_deleted']-rec['wwm_delta_full_minus_deleted']
        rec['rtd_minus_wwm_delta_shuffled']=rec['rtd_mlm_delta_full_minus_shuffled']-rec['wwm_delta_full_minus_shuffled']
        rec['rtd_minus_wwm_delta_unrelated']=rec['rtd_mlm_delta_full_minus_unrelated']-rec['wwm_delta_full_minus_unrelated']
        rec['rtd_minus_wwm_specificity_vs_unrelated']=rec['rtd_mlm_specificity_vs_unrelated']-rec['wwm_specificity_vs_unrelated']
        rows.append(rec)
    # aggregate robustly
    def mean(xs): return sum(xs)/len(xs) if xs else float('nan')
    def median(xs):
        xs=sorted(xs); n=len(xs); return xs[n//2] if n%2 else 0.5*(xs[n//2-1]+xs[n//2])
    keys=['wwm_delta_full_minus_deleted','rtd_mlm_delta_full_minus_deleted','rtd_minus_wwm_delta_deleted',
          'wwm_delta_full_minus_shuffled','rtd_mlm_delta_full_minus_shuffled','rtd_minus_wwm_delta_shuffled',
          'wwm_delta_full_minus_unrelated','rtd_mlm_delta_full_minus_unrelated','rtd_minus_wwm_delta_unrelated',
          'wwm_specificity_vs_unrelated','rtd_mlm_specificity_vs_unrelated','rtd_minus_wwm_specificity_vs_unrelated']
    agg={}
    for k in keys:
        vals=[r[k] for r in rows if not (isinstance(r[k],float) and (r[k]!=r[k]))]
        agg[k]={'mean':mean(vals),'median':median(vals),'pos_fraction':sum(1 for x in vals if x>0)/len(vals),'n':len(vals)}
    csv_path=ROOT/'data/context_probe_rows.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        fieldnames=['case_id','target_word','prefix_words','unrelated_words']+keys
        w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k) for k in fieldnames})
    return {'n_cases':len(rows),'rows_csv':str(csv_path),'aggregate':agg,'example_cases':rows[:5],
            'design':'Same full/deleted/shuffled/unrelated earlier-context interventions are applied to matched WWM and RTD+MLM; metric is original target log-prob at fixed masked later token. The route needs RTD_minus_WWM context deltas and specificity vs unrelated to be positive, not just single-model sensitivity.'}

def main():
    t0=time.time(); env=setup_env(); LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a',encoding='utf-8') as logf:
        logf.write(f'\n===== research eval+probe start {time.ctime()} =====\n')
        profiles={
            'wwm':fast_profile_one('wwm_chck_1M', WWM_CKPT, env, logf),
            'rtd_mlm':fast_profile_one('rtd_mlm_chck_1M', RTD_CKPT, env, logf),
        }
    deltas={k:round(profiles['rtd_mlm']['scores'][k]-profiles['wwm']['scores'][k],4) for k in COLS}
    probe=context_probe(env)
    rtd_metrics=json.loads((RTD_CKPT.parent.parent/'scientific_metrics.json').read_text(encoding='utf-8'))
    wwm_metrics=json.loads((WWM_CKPT.parent.parent/'scientific_metrics.json').read_text(encoding='utf-8'))
    payload={'status':'RTD_WWM_1M_EVAL_AND_CONTEXT_PROBE','profiles':profiles,'rtd_minus_wwm_fast_profile':deltas,
             'context_probe':probe,'rtd_training_metrics':rtd_metrics,'wwm_training_metrics':wwm_metrics,
             'elapsed_sec':time.time()-t0}
    OUT_JSON.parent.mkdir(parents=True,exist_ok=True); OUT_JSON.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research matched RTD+MLM vs WWM 1M validation','',f'Evidence JSON: `{OUT_JSON}`','',
           '## Fast profile: RTD+MLM - matched WWM','',
           '| BLiMP | Supplement | EWoK | Entity | COMPS | Reading |','|---:|---:|---:|---:|---:|---:|',
           f"| {deltas['blimp_fast']:+.4f} | {deltas['supplement_fast']:+.4f} | {deltas['ewok_fast']:+.4f} | {deltas['entity_tracking_fast']:+.4f} | {deltas['comps']:+.4f} | {deltas['Reading_mean']:+.4f} |",'',
           '## RTD shortcut metrics','',
           f"Final replaced recall: {rtd_metrics.get('rtd_replaced_recall_last'):.4f}; above-majority accuracy: {rtd_metrics.get('rtd_above_majority_last'):.4f}; pred-original rate: {rtd_metrics.get('rtd_pred_original_rate_last'):.4f}; label-original rate: {rtd_metrics.get('rtd_label_original_rate_last'):.4f}.",'',
           '## Context intervention probe aggregate','',
           '| metric | mean | median | pos frac |','|---|---:|---:|---:|']
    for k,v in probe['aggregate'].items():
        lines.append(f"| {k} | {v['mean']:+.5f} | {v['median']:+.5f} | {v['pos_fraction']:.3f} |")
    OUT_NOTE.parent.mkdir(parents=True,exist_ok=True); OUT_NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(OUT_JSON),'note':str(OUT_NOTE),'fast_delta':deltas,'probe_aggregate':probe['aggregate'],'elapsed_sec':payload['elapsed_sec']},indent=2))
if __name__=='__main__': main()
