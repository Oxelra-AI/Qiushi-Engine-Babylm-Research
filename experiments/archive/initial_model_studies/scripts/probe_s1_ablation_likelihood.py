#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, random, re, math
from collections import defaultdict

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
DEFAULT_MODEL = ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M'
DEFAULT_JSONL = ROOT/'data/counterfactual_revision_109/linked_definition_counterfactual_v3_seed1093_target100000_actual.jsonl'
OUT = ROOT/'data/s1_ablation_likelihood_probe.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/s1_ablation_likelihood_probe.md')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?")
STOP = {'the','a','an','and','or','but','of','for','to','in','on','at','by','with','from','as','is','are','was','were','be','been','being','it','its','they','them','their','this','that','these','those','there','here','then','than','which','who','what','when','where','why','how','not','no','yes','can','could','will','would','should','may','might','has','have','had'}

def words_with_spans(text: str):
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]

def norm(w: str) -> str:
    return w.lower().strip("-'")

def content_words(text: str):
    return [(w,a,b) for w,a,b in words_with_spans(text) if len(norm(w))>=4 and norm(w) not in STOP and not norm(w).isdigit()]

def load_rows(path: pathlib.Path, n: int, split: str, seed: int):
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj=json.loads(line)
            if split!='all' and obj.get('split')!=split: continue
            rows.append(obj)
    rng=random.Random(seed); rng.shuffle(rows)
    return rows[:n]

def target_spans(row):
    s2=row['s2']
    spans=[]
    dep=row.get('linked_span',{}).get('s2_dependent')
    if dep and 'dependent_span' in dep:
        a,b=dep['dependent_span']; spans.append({'type':'initial_dependent','text':s2[a:b], 'span':[a,b]})
    cws=content_words(s2)
    if cws:
        w,a,b=cws[0]; spans.append({'type':'first_s2_content','text':w,'span':[a,b]})
        w,a,b=cws[-1]; spans.append({'type':'last_s2_content','text':w,'span':[a,b]})
    # first two content words as short semantic span if close enough
    if len(cws)>=2 and cws[1][2]-cws[0][1] <= 60:
        spans.append({'type':'first_two_s2_content_window','text':s2[cws[0][1]:cws[1][2]],'span':[cws[0][1],cws[1][2]]})
    # Deduplicate identical spans/types can overlap but keep types separate.
    return spans

def encode_masked(tokenizer, text: str, target_abs_span: tuple[int,int], max_len: int):
    enc=tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len, return_offsets_mapping=True, return_tensors='pt')
    ids=enc['input_ids'][0]
    attn=enc['attention_mask'][0]
    offsets=enc['offset_mapping'][0]
    mask_positions=[]; labels=[]
    a0,b0=target_abs_span
    for i,(a,b) in enumerate(offsets.tolist()):
        if attn[i].item()==0: continue
        if b>a0 and a<b0:
            mask_positions.append(i); labels.append(int(ids[i].item()))
    if not mask_positions:
        return None
    masked=ids.clone(); masked[mask_positions]=tokenizer.mask_token_id
    return {'input_ids':masked.unsqueeze(0),'attention_mask':attn.unsqueeze(0),'positions':mask_positions,'labels':labels,'num_target_tokens':len(labels)}

@torch.no_grad()
def score(model, tokenizer, text: str, target_abs_span: tuple[int,int], device, max_len: int):
    item=encode_masked(tokenizer,text,target_abs_span,max_len)
    if item is None: return None
    inp=item['input_ids'].to(device); attn=item['attention_mask'].to(device)
    logits=model(input_ids=inp, attention_mask=attn).logits[0]
    logp=torch.log_softmax(logits, dim=-1)
    vals=[]
    for pos,lab in zip(item['positions'], item['labels']):
        vals.append(float(logp[pos, lab].detach().cpu()))
    return {'sum_logprob':sum(vals),'mean_logprob':sum(vals)/len(vals),'num_target_tokens':len(vals),'token_logprobs':vals}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', default=str(DEFAULT_MODEL))
    ap.add_argument('--jsonl', default=str(DEFAULT_JSONL))
    ap.add_argument('--rows', type=int, default=96)
    ap.add_argument('--split', default='heldout')
    ap.add_argument('--seed', type=int, default=114)
    ap.add_argument('--max_len', type=int, default=256)
    args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer=AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model=AutoModelForMaskedLM.from_pretrained(args.model).to(device).eval()
    rows=load_rows(pathlib.Path(args.jsonl), args.rows, args.split, args.seed)
    if len(rows)<args.rows and args.split!='all':
        # heldout is small; include all heldout rather than silently sampling too few
        pass
    rng=random.Random(args.seed+17)
    all_rows=[]; per_type=defaultdict(list)
    skipped=0
    for idx,row in enumerate(rows):
        # random wrong s1 from another row, keep the target s2 fixed.
        wrong=row
        for _ in range(20):
            cand=rng.choice(rows)
            if cand.get('line_no')!=row.get('line_no'):
                wrong=cand; break
        for t in target_spans(row):
            a,b=t['span']; s1=row['s1']; s2=row['s2']; wrong_s1=wrong['s1']
            texts={
                'true_s1': s1+' '+s2,
                'no_s1': s2,
                'wrong_s1': wrong_s1+' '+s2,
            }
            spans={
                'true_s1': (len(s1)+1+a, len(s1)+1+b),
                'no_s1': (a,b),
                'wrong_s1': (len(wrong_s1)+1+a, len(wrong_s1)+1+b),
            }
            scores={}
            for k in texts:
                sc=score(model, tokenizer, texts[k], spans[k], device, args.max_len)
                if sc is None:
                    scores=None; break
                scores[k]=sc
            if scores is None:
                skipped+=1; continue
            rec={'row_index':idx,'example_id':row.get('example_id'),'target_type':t['type'],'target_text':t['text'],'target_span_s2':t['span'],
                 'source':row.get('source'),'line_no':row.get('line_no'),'s1':s1,'s2':s2,
                 'wrong_s1_example_id':wrong.get('example_id'),'wrong_s1':wrong_s1,
                 'scores':scores,
                 'delta_true_minus_no_mean':scores['true_s1']['mean_logprob']-scores['no_s1']['mean_logprob'],
                 'delta_true_minus_wrong_mean':scores['true_s1']['mean_logprob']-scores['wrong_s1']['mean_logprob'],
                 'delta_true_minus_no_sum':scores['true_s1']['sum_logprob']-scores['no_s1']['sum_logprob'],
                 'delta_true_minus_wrong_sum':scores['true_s1']['sum_logprob']-scores['wrong_s1']['sum_logprob']}
            all_rows.append(rec); per_type[t['type']].append(rec)
    summary={}
    for typ,recs in per_type.items():
        def stats(field):
            vals=[r[field] for r in recs]
            return {'n':len(vals),'mean':sum(vals)/len(vals),'min':min(vals),'max':max(vals),'positive_frac':sum(v>0 for v in vals)/len(vals)} if vals else None
        summary[typ]={'delta_true_minus_no_mean':stats('delta_true_minus_no_mean'), 'delta_true_minus_wrong_mean':stats('delta_true_minus_wrong_mean'),
                      'mean_target_tokens': sum(r['scores']['true_s1']['num_target_tokens'] for r in recs)/len(recs) if recs else None}
    payload={'status':'S1_ABLATION_LIKELIHOOD_PROBE','model':args.model,'jsonl':args.jsonl,'split':args.split,'rows_requested':args.rows,'rows_loaded':len(rows),'num_scored_targets':len(all_rows),'skipped_targets':skipped,'summary_by_target_type':summary,'records':all_rows[:200]}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research — primary-likelihood s1 ablation probe','',f'Model: `{args.model}`',f'Data: `{args.jsonl}`',f'Evidence JSON: `{OUT}`','', 'Deltas are target-token mean logprob with true s1 minus no-s1 or wrong-s1 contexts. Positive means the true previous sentence helps predict the masked s2 target.','', '| target type | n | true-no mean | true-no positive frac | true-wrong mean | true-wrong positive frac | mean target tokens |','|---|---:|---:|---:|---:|---:|---:|']
    for typ,d in summary.items():
        a=d['delta_true_minus_no_mean']; b=d['delta_true_minus_wrong_mean']
        lines.append(f"| {typ} | {a['n']} | {a['mean']:+.4f} | {a['positive_frac']:.3f} | {b['mean']:+.4f} | {b['positive_frac']:.3f} | {d['mean_target_tokens']:.2f} |")
    lines += ['', 'Interpretation note: this probes whether the existing protected WWM model already uses true s1 for candidate s2 spans. Strongly positive dependency-specific deltas would motivate XSpan/Prefix-LM construction; near-zero or negative deltas mean the current span design does not create load-bearing cross-sentence likelihood.']
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out':str(OUT),'num_scored_targets':len(all_rows),'summary_by_target_type':summary}, indent=2))
if __name__=='__main__': main()
