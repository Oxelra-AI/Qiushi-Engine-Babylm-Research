#!/usr/bin/env python3
"""research CPC diagnostics: fast profiles + prefix-sensitivity-stratified MLM-logit probe.

No training. Evaluates two CPC 1M variants against matched WWM:
- same-source/prefer_same CPC from research
- forced-cross-source CPC from research

Then reconstructs exact research/309 consumed official examples and runs a token-level
intervention probe on fixed later targets. It stratifies cases by whether WWM itself
shows any true-prefix signal (full vs deleted/shuffled/unrelated), to test the concern that all-suffix CPC loss may be drowned by locally predictable targets.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, pathlib, random, re, statistics, subprocess, sys, time
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'
WWM=ROOT/'training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched/hf_model/chck_1M'
CPC_SAME=ROOT/'training/runs/cpc_mlm_debertav2_8x480_official_1M_b128_seed42/hf_model/chck_1M'
CPC_CROSS=ROOT/'training/runs/cpc_mlm_cross_debertav2_8x480_official_1M_b128_seed42/hf_model/chck_1M'
RUN_WWM=ROOT/'training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched'
RUN_SAME=CPC_SAME.parents[1]
RUN_CROSS=CPC_CROSS.parents[1]
OUT_JSON=ROOT/'data/cpc_profiles_and_prefix_probe.json'
OUT_CSV=ROOT/'data/cpc_prefix_probe_rows.csv'
OUT_NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/cpc_profiles_and_prefix_probe.md')
LOG=(ROOT.parents[2] / 'research/notes/initial_model_studies/cpc_profiles_and_prefix_probe.log')
sys.path.insert(0,str((ROOT/'training/scripts').resolve()))
from babylm_masked_train import TRAIN_FILES, iter_examples, Example

TASKS=[
 ('blimp_fast','blimp','evaluation_data/fast_eval/blimp_fast','blimp_fast'),
 ('supplement_fast','blimp','evaluation_data/fast_eval/supplement_fast','supplement_fast'),
 ('ewok_fast','ewok','evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast','ewok_fast'),
 ('entity_tracking_fast','entity_tracking','evaluation_data/fast_eval/entity_tracking_fast','entity_tracking_fast'),
 ('comps','comps','evaluation_data/full_eval/comps','comps'),
]
STOP=set('the a an and or but if then than as of in on at to from for with without by about into over under is are was were be been being do does did have has had this that these those it its they them he she we you i me my your our their his her not no yes can could would should will may might there here very just also only'.split())

def env_setup():
    env=os.environ.copy(); hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve())
    env['HF_DATASETS_CACHE']=str((hf/'datasets').resolve()); env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve())
    env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']:
        pathlib.Path(env[k]).mkdir(parents=True,exist_ok=True)
    os.environ.update({k:env[k] for k in ['HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE','TOKENIZERS_PARALLELISM']})
    return env

def sha16(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()[:16]

def run(cmd, env, logf, cwd=None):
    line='$ '+' '.join(map(str,cmd)); print(line, flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run(cmd,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(p.stdout[-1500:], flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode!=0: raise RuntimeError(line+'\n'+p.stdout[-6000:])

def read_avg(report):
    txt=pathlib.Path(report).read_text(errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)',txt)
    if not m: raise RuntimeError(f'parse avg failed {report}')
    return float(m.group(1))

def read_reading(report):
    txt=pathlib.Path(report).read_text(errors='replace'); out={}
    for lab,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(lab)+r':\s*([0-9.\-]+)',txt)
        if not m: raise RuntimeError(f'parse reading failed {report}')
        out[key]=float(m.group(1))
    out['Reading_mean']=(out['reading_eye_tracking']+out['reading_self_paced'])/2
    return out

def fast_profile(label, ckpt, env, logf):
    outdir=ckpt.parent.parent/f'eval_step311_direct_{label}'
    scores={}; reports={}
    for task_name,task,data_path,ds_name in TASKS:
        run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(ckpt.resolve()),'--backend','mlm','--task',task,'--data_path',data_path,'--save_predictions','--batch_size','64','--output_dir',str(outdir.resolve())],env,logf,cwd=str(STRICT))
        report=outdir/ckpt.name/'main'/'zero_shot'/'mlm'/task/ds_name/'best_temperature_report.txt'
        scores[task_name]=read_avg(report); reports[task_name]=str(report)
    run([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(ckpt.resolve()),'--backend','mlm','--data_path','evaluation_data/fast_eval/reading/reading_data.csv','--output_dir',str(outdir.resolve())],env,logf,cwd=str(STRICT))
    rr=outdir/ckpt.name/'main'/'zero_shot'/'mlm'/'reading'/'report.txt'
    scores.update(read_reading(rr)); reports['reading']=str(rr)
    return {'path':str(ckpt),'sha16':sha16(ckpt/'model.safetensors'),'scores':scores,'reports':reports}

def reconstruct_examples():
    raw=RUN_WWM/'raw_dataset'; files=[raw/n for n in TRAIN_FILES]
    pool=list(iter_examples(files,1_000_000,160))
    for i,e in enumerate(pool): e.example_id=i
    rng=random.Random(42); rng.shuffle(pool); out=[]; actual=0
    for ex in pool:
        if actual>=1_000_000: break
        if actual+ex.words<=1_000_000: out.append(ex); actual+=ex.words
        else:
            take=1_000_000-actual; out.append(Example(' '.join(ex.text.split()[:take]),take,ex.example_id,ex.source)); actual+=take
    if actual!=1_000_000: raise RuntimeError(actual)
    return out

def ids(text,tok): return tok(text,add_special_tokens=False)['input_ids']
def good_token(tokstr):
    s=str(tokstr).replace('Ġ','').replace('▁','').strip(); s=re.sub(r'^[^A-Za-z]+|[^A-Za-z]+$','',s)
    return len(s)>=3 and s.lower() not in STOP and re.match(r'^[A-Za-z][A-Za-z\'-]*$',s)

def make_cases(examples,tok,n=240):
    rng=random.Random(311); cands=[]
    for ex in examples:
        words=ex.text.split()
        if len(words)<70: continue
        split=max(30,min(len(words)-25,len(words)//2))
        pids=ids(' '.join(words[:split]),tok); lids=ids(' '+' '.join(words[split:]),tok)
        if len(pids)<20 or len(lids)<20: continue
        pids=pids[-min(len(pids),120):]; lids=lids[:min(len(lids),120)]
        toks=tok.convert_ids_to_tokens(lids); target=None
        for j in range(max(5,len(lids)//4), min(len(lids),120)):
            if lids[j] in tok.all_special_ids: continue
            if good_token(toks[j]): target=j; break
        if target is None: continue
        cands.append({'id':ex.example_id,'source':ex.source,'pids':pids,'lids':lids,'target_idx':target,'target_id':lids[target],'token':toks[target]})
    rng.shuffle(cands); cases=[]
    for c in cands:
        same=[o for o in cands if o['id']!=c['id'] and o['source']==c['source'] and abs(len(o['pids'])-len(c['pids']))<=20]
        cross=[o for o in cands if o['id']!=c['id'] and o['source']!=c['source']]
        anyo=[o for o in cands if o['id']!=c['id']]
        if not anyo: continue
        c=dict(c); c['same_pids']=(rng.choice(same if same else anyo)['pids'])[-len(c['pids']):]
        c['cross_pids']=(rng.choice(cross if cross else anyo)['pids'])[-len(c['pids']):]
        cases.append(c)
        if len(cases)>=n: break
    return cases

def block_shuffle(x,rng):
    blocks=[x[i:i+8] for i in range(0,len(x),8)]; rng.shuffle(blocks); return [z for b in blocks for z in b]
def logp(model,pids,lids,target_idx,target_id,tok,device):
    seq=(pids+lids)[:256]; pos=len(pids)+target_idx
    if pos>=len(seq): return float('nan')
    seq=list(seq); seq[pos]=tok.mask_token_id
    inp=torch.tensor([seq],device=device); attn=torch.ones_like(inp)
    with torch.no_grad(): return float(torch.log_softmax(model(input_ids=inp,attention_mask=attn).logits[0,pos],dim=-1)[target_id].item())
def mean(xs): return sum(xs)/len(xs) if xs else float('nan')
def med(xs): return statistics.median(xs) if xs else float('nan')
def pf(xs): return sum(v>0 for v in xs)/len(xs) if xs else float('nan')
def agg_group(rows):
    keys=[k for k in rows[0] if k.startswith(('wwm_','cpc_same_','cpc_cross_','same_minus_wwm_','cross_minus_wwm_')) and not k.endswith(('full','deleted','shuffled','same_near','cross_near'))]
    out={}
    for k in keys:
        vals=[float(r[k]) for r in rows if not math.isnan(float(r[k]))]
        out[k]={'mean':mean(vals),'median':med(vals),'pos_fraction':pf(vals),'n':len(vals)}
    return out

def prefix_probe():
    tok=AutoTokenizer.from_pretrained(str(WWM.resolve()),use_fast=True)
    cases=make_cases(reconstruct_examples(),tok,240)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models={'wwm':AutoModelForMaskedLM.from_pretrained(str(WWM.resolve()),trust_remote_code=True).to(device).eval(),
            'cpc_same':AutoModelForMaskedLM.from_pretrained(str(CPC_SAME.resolve()),trust_remote_code=True).to(device).eval(),
            'cpc_cross':AutoModelForMaskedLM.from_pretrained(str(CPC_CROSS.resolve()),trust_remote_code=True).to(device).eval()}
    rng=random.Random(3111); rows=[]
    for i,c in enumerate(cases):
        variants={'full':c['pids'],'deleted':[],'shuffled':block_shuffle(c['pids'],rng),'same_near':c['same_pids'],'cross_near':c['cross_pids']}
        r={'case_id':i,'source':c['source'],'target_token':c['token'],'prefix_tokens':len(c['pids']),'later_tokens':len(c['lids']),'target_idx':c['target_idx']}
        for mn,m in models.items():
            vals={vn:logp(m,p,c['lids'],c['target_idx'],c['target_id'],tok,device) for vn,p in variants.items()}
            for vn,v in vals.items(): r[f'{mn}_lp_{vn}']=v
            for vn in ['deleted','shuffled','same_near','cross_near']:
                r[f'{mn}_delta_{vn}']=vals['full']-vals[vn]
            r[f'{mn}_spec_same_vs_cross']=r[f'{mn}_delta_same_near']-r[f'{mn}_delta_cross_near']
            r[f'{mn}_spec_deleted_vs_cross']=r[f'{mn}_delta_deleted']-r[f'{mn}_delta_cross_near']
        for pref,base in [('same','cpc_same'),('cross','cpc_cross')]:
            for key in ['delta_deleted','delta_shuffled','delta_same_near','delta_cross_near','spec_same_vs_cross','spec_deleted_vs_cross']:
                r[f'{pref}_minus_wwm_{key}']=r[f'{base}_{key}']-r[f'wwm_{key}']
        # WWM sensitivity score: max absolute non-cross true-prefix dependence excluding cross source.
        r['wwm_sensitivity_score']=max(abs(r['wwm_delta_deleted']),abs(r['wwm_delta_shuffled']),abs(r['wwm_delta_same_near']))
        rows.append(r)
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with OUT_CSV.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    sorted_rows=sorted(rows,key=lambda r:r['wwm_sensitivity_score'])
    q=max(1,len(rows)//4); low=sorted_rows[:q]; high=sorted_rows[-q:]
    return {'n_cases':len(rows),'rows_csv':str(OUT_CSV),'aggregate_all':agg_group(rows),'aggregate_low_wwm_sensitivity':agg_group(low),'aggregate_high_wwm_sensitivity':agg_group(high),'example_rows':rows[:5]}

def main():
    t0=time.time(); env=env_setup(); LOG.parent.mkdir(parents=True,exist_ok=True)
    profiles={}
    with LOG.open('a',encoding='utf-8') as logf:
        logf.write(f'\n===== research start {time.ctime()} =====\n')
        for label,ckpt in [('wwm',WWM),('cpc_same',CPC_SAME),('cpc_cross',CPC_CROSS)]:
            profiles[label]=fast_profile(label,ckpt,env,logf)
    cols=list(profiles['wwm']['scores'].keys())
    deltas={label:{k:profiles[label]['scores'][k]-profiles['wwm']['scores'][k] for k in cols} for label in ['cpc_same','cpc_cross']}
    probe=prefix_probe()
    metrics={'cpc_same':json.loads((RUN_SAME/'scientific_metrics.json').read_text()),'cpc_cross':json.loads((RUN_CROSS/'scientific_metrics.json').read_text())}
    payload={'status':'CPC_PROFILES_AND_PREFIX_PROBE','profiles':profiles,'deltas_vs_wwm':deltas,'prefix_probe':probe,'training_metrics':metrics,'elapsed_sec':time.time()-t0}
    OUT_JSON.parent.mkdir(parents=True,exist_ok=True); OUT_JSON.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research CPC fast profiles and prefix-sensitivity probe','',f'JSON: `{OUT_JSON}`',f'Rows: `{OUT_CSV}`','','## Fast profile deltas vs matched WWM','', '| model | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |','|---|---:|---:|---:|---:|---:|---:|']
    for label in ['cpc_same','cpc_cross']:
        d=deltas[label]; lines.append(f"| {label} | {d['blimp_fast']:+.2f} | {d['supplement_fast']:+.2f} | {d['ewok_fast']:+.2f} | {d['entity_tracking_fast']:+.2f} | {d['comps']:+.2f} | {d['Reading_mean']:+.3f} |")
    lines += ['','## Training margin endpoints','']
    for label,m in metrics.items():
        lines.append(f"- {label}: mode={m.get('negative_source_mode')}, same={m.get('neg_same_source_frac_last')}, cross={m.get('neg_cross_source_frac_last')}, cpc_loss={m.get('cpc_loss_last'):.4f}, active={m.get('margin_active_frac_last'):.4f}, lp_diff={m.get('mean_lp_diff_last'):.5f}")
    for sect in ['aggregate_all','aggregate_low_wwm_sensitivity','aggregate_high_wwm_sensitivity']:
        lines += ['',f'## Probe {sect}','', '| metric | mean | median | pos frac | n |','|---|---:|---:|---:|---:|']
        for k,v in probe[sect].items():
            if k in ['same_minus_wwm_delta_same_near','same_minus_wwm_spec_same_vs_cross','cross_minus_wwm_delta_cross_near','cross_minus_wwm_spec_same_vs_cross','wwm_delta_same_near','wwm_delta_cross_near','cpc_same_delta_same_near','cpc_cross_delta_cross_near']:
                lines.append(f"| {k} | {v['mean']:+.5f} | {v['median']:+.5f} | {v['pos_fraction']:.3f} | {v['n']} |")
    OUT_NOTE.parent.mkdir(parents=True,exist_ok=True); OUT_NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(OUT_JSON),'note':str(OUT_NOTE),'deltas':deltas,'probe_keys':{k:probe['aggregate_all'][k] for k in probe['aggregate_all'] if k.startswith(('same_minus_wwm','cross_minus_wwm'))},'elapsed_sec':payload['elapsed_sec']},indent=2))
if __name__=='__main__': main()
