#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, random, re
from collections import defaultdict, Counter

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
DEFAULT_MODEL = ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M'
DEFAULT_JSONL = ROOT/'data/xspan_revision_118/relational_xspan_compact_v4_from_v3_seed117_target200000_actual.jsonl'
OUT = ROOT/'data/xspan_compact_v4_s1_likelihood_probe.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/xspan_compact_v4_s1_likelihood_probe.md')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?")


def load_rows(path: pathlib.Path, split: str, limit: int, seed: int):
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            if split != 'all' and r.get('split') != split:
                continue
            rows.append(r)
    rng=random.Random(seed)
    rng.shuffle(rows)
    if limit and len(rows) > limit:
        rows=rows[:limit]
    return rows


def encode_masked(tokenizer, text: str, span: tuple[int,int], max_len: int):
    enc=tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len, return_offsets_mapping=True, return_tensors='pt')
    ids=enc['input_ids'][0]
    attn=enc['attention_mask'][0]
    offsets=enc['offset_mapping'][0]
    a0,b0=span
    pos=[]; labels=[]
    for i,(a,b) in enumerate(offsets.tolist()):
        if int(attn[i].item()) == 0:
            continue
        if b > a0 and a < b0:
            pos.append(i); labels.append(int(ids[i].item()))
    if not pos:
        return None
    masked=ids.clone(); masked[pos]=tokenizer.mask_token_id
    return {'input_ids':masked.unsqueeze(0),'attention_mask':attn.unsqueeze(0),'positions':pos,'labels':labels}


@torch.no_grad()
def score(model, tokenizer, text: str, span: tuple[int,int], device, max_len: int):
    item=encode_masked(tokenizer, text, span, max_len)
    if item is None:
        return None
    logits=model(input_ids=item['input_ids'].to(device), attention_mask=item['attention_mask'].to(device)).logits[0]
    logp=torch.log_softmax(logits, dim=-1)
    vals=[]
    for p,l in zip(item['positions'], item['labels']):
        vals.append(float(logp[p,l].detach().cpu()))
    return {'sum_logprob':sum(vals), 'mean_logprob':sum(vals)/len(vals), 'num_target_tokens':len(vals), 'token_logprobs':vals}


def first_words(txt: str, n: int = 2) -> str:
    ws=[m.group(0).lower() for m in WORD_RE.finditer(txt)]
    return ' '.join(ws[:n]) if ws else '<empty>'


def stats(vals):
    if not vals:
        return None
    sv=sorted(vals)
    return {'n':len(vals),'mean':sum(vals)/len(vals),'median':sv[len(sv)//2] if len(sv)%2 else 0.5*(sv[len(sv)//2-1]+sv[len(sv)//2]),'min':sv[0],'max':sv[-1],'positive_frac':sum(v>0 for v in vals)/len(vals)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', default=str(DEFAULT_MODEL))
    ap.add_argument('--jsonl', default=str(DEFAULT_JSONL))
    ap.add_argument('--split', default='heldout')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=118)
    ap.add_argument('--max_len', type=int, default=256)
    args=ap.parse_args()
    rows=load_rows(pathlib.Path(args.jsonl), args.split, args.limit, args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer=AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model=AutoModelForMaskedLM.from_pretrained(args.model).to(device).eval()
    records=[]; skipped=[]
    by_type=defaultdict(list); by_prefix=defaultdict(list)
    for r in rows:
        contexts={
            'true_s1': (r['text'], tuple(r['target_span_text'])),
            'wrong_s1': (r['text_wrong_s1'], tuple(r['target_span_wrong_s1'])),
            'no_s1': (r['text_no_s1'], tuple(r['target_span_no_s1'])),
        }
        scores={}
        ok=True
        for k,(txt,span) in contexts.items():
            sc=score(model, tokenizer, txt, span, device, args.max_len)
            if sc is None:
                skipped.append({'example_id':r.get('example_id'), 'reason':f'no_mask_positions_{k}', 'target_text':r.get('target_text')})
                ok=False; break
            scores[k]=sc
        if not ok:
            continue
        rec={
            'example_id':r.get('example_id'), 'old_example_id':r.get('old_example_id'), 'split':r.get('split'),
            'target_type':r.get('target_type'), 'target_text':r.get('target_text'), 'target_prefix2':first_words(r.get('target_text',''),2),
            'source':r.get('source'), 'line_no':r.get('line_no'), 's1':r.get('s1'), 's2':r.get('s2'), 'wrong_s1':r.get('wrong_s1',{}).get('s1'),
            'num_target_tokens':scores['true_s1']['num_target_tokens'], 'scores':scores,
            'delta_true_minus_wrong_mean':scores['true_s1']['mean_logprob']-scores['wrong_s1']['mean_logprob'],
            'delta_true_minus_no_mean':scores['true_s1']['mean_logprob']-scores['no_s1']['mean_logprob'],
            'delta_wrong_minus_no_mean':scores['wrong_s1']['mean_logprob']-scores['no_s1']['mean_logprob'],
            'delta_true_minus_wrong_sum':scores['true_s1']['sum_logprob']-scores['wrong_s1']['sum_logprob'],
            'delta_true_minus_no_sum':scores['true_s1']['sum_logprob']-scores['no_s1']['sum_logprob'],
        }
        records.append(rec); by_type[rec['target_type']].append(rec); by_prefix[rec['target_prefix2']].append(rec)
    summary={}
    for typ,rs in by_type.items():
        summary[typ]={
            'true_minus_wrong_mean':stats([r['delta_true_minus_wrong_mean'] for r in rs]),
            'true_minus_no_mean':stats([r['delta_true_minus_no_mean'] for r in rs]),
            'wrong_minus_no_mean':stats([r['delta_wrong_minus_no_mean'] for r in rs]),
            'mean_target_tokens':sum(r['num_target_tokens'] for r in rs)/len(rs) if rs else None,
            'top5':pack_examples(sorted(rs,key=lambda r:r['delta_true_minus_wrong_mean'], reverse=True)[:5]),
            'bottom5':pack_examples(sorted(rs,key=lambda r:r['delta_true_minus_wrong_mean'])[:5]),
        }
    # Template/prefix concentration among positive examples and top decile.
    positives=[r for r in records if r['delta_true_minus_wrong_mean']>0]
    top=sorted(records, key=lambda r:r['delta_true_minus_wrong_mean'], reverse=True)[:max(1, len(records)//10)] if records else []
    prefix_counts=Counter(r['target_prefix2'] for r in records)
    pos_prefix_counts=Counter(r['target_prefix2'] for r in positives)
    top_prefix_counts=Counter(r['target_prefix2'] for r in top)
    payload={
        'status':'XSPAN_COMPACT_V4_S1_LIKELIHOOD_PROBE',
        'model':args.model, 'jsonl':args.jsonl, 'split':args.split, 'limit':args.limit, 'rows_loaded':len(rows), 'records_scored':len(records), 'skipped':skipped[:100],
        'overall':{
            'true_minus_wrong_mean':stats([r['delta_true_minus_wrong_mean'] for r in records]),
            'true_minus_no_mean':stats([r['delta_true_minus_no_mean'] for r in records]),
            'wrong_minus_no_mean':stats([r['delta_wrong_minus_no_mean'] for r in records]),
            'mean_target_tokens':sum(r['num_target_tokens'] for r in records)/len(records) if records else None,
        },
        'summary_by_target_type':summary,
        'target_prefix2_counts_top20':prefix_counts.most_common(20),
        'positive_prefix2_counts_top20':pos_prefix_counts.most_common(20),
        'top_decile_prefix2_counts_top20':top_prefix_counts.most_common(20),
        'top_decile_examples':pack_examples(top[:20]),
        'bottom_decile_examples':pack_examples(sorted(records,key=lambda r:r['delta_true_minus_wrong_mean'])[:20]),
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research — compact v4 XSpan true/wrong/no-s1 likelihood probe','',f'Model: `{args.model}`',f'XSpan JSONL: `{args.jsonl}`',f'Evidence JSON: `{OUT}`','',
           'The compact v4 rows were selected by rule-based text filters only; these scores are for mechanism judgment, not training-row selection. Deltas are target-token mean log-probabilities.','',
           '| group | n | true-wrong mean | true-wrong median | positive frac | true-no mean | wrong-no mean | mean target tokens |','|---|---:|---:|---:|---:|---:|---:|---:|']
    ov=payload['overall']; a=ov['true_minus_wrong_mean']; b=ov['true_minus_no_mean']; c=ov['wrong_minus_no_mean']
    lines.append(f"| overall | {a['n']} | {a['mean']:+.4f} | {a['median']:+.4f} | {a['positive_frac']:.3f} | {b['mean']:+.4f} | {c['mean']:+.4f} | {ov['mean_target_tokens']:.2f} |")
    for typ,d in summary.items():
        a=d['true_minus_wrong_mean']; b=d['true_minus_no_mean']; c=d['wrong_minus_no_mean']
        lines.append(f"| {typ} | {a['n']} | {a['mean']:+.4f} | {a['median']:+.4f} | {a['positive_frac']:.3f} | {b['mean']:+.4f} | {c['mean']:+.4f} | {d['mean_target_tokens']:.2f} |")
    lines += ['', '## Top target prefixes among top-decile true-minus-wrong examples', '']
    for k,v in payload['top_decile_prefix2_counts_top20'][:12]:
        lines.append(f'- `{k}`: {v}')
    lines += ['', '## Top positive examples', '']
    for e in payload['top_decile_examples'][:8]:
        lines.append(f"- Δ={e['delta_true_minus_wrong_mean']:+.3f} {e['target_type']} `{e['target_text']}` :: {e['s1'][:80]} / {e['s2'][:120]}")
    lines += ['', '## Bottom examples', '']
    for e in payload['bottom_decile_examples'][:8]:
        lines.append(f"- Δ={e['delta_true_minus_wrong_mean']:+.3f} {e['target_type']} `{e['target_text']}` :: {e['s1'][:80]} / {e['s2'][:120]}")
    lines += ['', '## Mechanism interpretation rule', '', 'If true-minus-wrong is weak, concentrated in a few templates, or wrong-s1 already captures most true-s1 benefit over no-s1, do not add more boundary rules; change target representation or route.']
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out':str(OUT),'records_scored':len(records),'overall':payload['overall'],'summary_by_target_type':{k:{'true_minus_wrong_mean':v['true_minus_wrong_mean']} for k,v in summary.items()}}, indent=2))


def pack_examples(rs):
    return [{k:r.get(k) for k in ['example_id','target_type','target_text','target_prefix2','delta_true_minus_wrong_mean','delta_true_minus_no_mean','delta_wrong_minus_no_mean','num_target_tokens','s1','s2','wrong_s1']} for r in rs]


if __name__=='__main__':
    main()
