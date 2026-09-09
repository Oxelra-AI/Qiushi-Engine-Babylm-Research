#!/usr/bin/env python3
"""research FineWeb-Edu matched random-vs-relation corpus materializer.

Purpose: separate FineWeb-Edu source distribution from relation-explicit experience.
Before any large training, build two same-source, same-word-budget JSONL corpora:
  A. random_quality: high-quality FineWeb-Edu documents after basic cleaning.
  B. relation_explicit: documents from the same streamed prefix but filtered for
     explicit entity/relation experience.

Both arms are packed to exact word counts in trainer-compatible JSONL rows:
  {example_id, source, text, words, ...metadata}
The existing BabyLM masked trainer consumes these rows exactly in file order.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, pathlib, random, re, statistics, sys, time
from dataclasses import dataclass, asdict
from typing import Iterable

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
OUT_DIR_DEFAULT = ROOT/'data/fineweb_relation_matched'
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3})\b")
RELATION_VERBS = set('''went moved stayed lived born died traveled travelled visited returned arrived left entered crossed followed met joined worked studied taught founded built made created opened closed carried held took gave brought found lost owned kept bought sold used broke repaired changed became called named known located based situated contains contained includes has had owns held carries brought gave put placed stored filled covered killed saved helped led ruled served married wrote published discovered developed invented designed elected appointed announced said told asked explained described showed'''.split())
STATE_PREPS = set('in at on inside outside near beside behind under over across through from to into onto around within with without by for'.split())
BAD_SUBSTR = ['{', '}', '<script', '</', 'cookie policy', 'privacy policy', 'terms of use', 'click here', 'subscribe', 'copyright', 'all rights reserved']


def setup_env():
    hf = ROOT/'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']:
        pathlib.Path(os.environ[k]).mkdir(parents=True, exist_ok=True)


def wc(text: str) -> int:
    return len(text.split())


def sha256_file(path: pathlib.Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def clean_text(text: str) -> str:
    text = text.replace('\r',' ').replace('\t',' ')
    lines=[]
    for line in text.split('\n'):
        s=' '.join(line.strip().split())
        if not s: continue
        if len(s) < 20: continue
        lines.append(s)
    return ' '.join(lines)


def sentence_words(text: str) -> list[int]:
    sents=[s.strip() for s in SENT_RE.split(text) if s.strip()]
    return [wc(s) for s in sents if wc(s)>0]


def entities(text: str) -> list[str]:
    bad = {'The','This','That','These','Those','There','When','Where','What','How','Why','Because','For','And','But','New'}
    out=[]
    for m in ENTITY_RE.finditer(text):
        e=m.group(0).strip()
        if e.split()[0] in bad: continue
        if len(e) < 4: continue
        out.append(e)
    return out


def relation_count(text: str) -> int:
    toks=[t.lower() for t in WORD_RE.findall(text)]
    return sum(1 for t in toks if t in RELATION_VERBS or t in STATE_PREPS)


def explicit_relation_hits(text: str) -> list[str]:
    # High precision-ish snippets: Entity within a 14-token span of relation cue and content word.
    words=[(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]
    ents=set(e.split()[0] for e in entities(text))
    hits=[]
    for i,(w,s,e) in enumerate(words):
        if w in ents:
            span=words[i:min(len(words), i+16)]
            lows=[x[0].lower() for x in span]
            if any(x in RELATION_VERBS for x in lows) and any(x in STATE_PREPS for x in lows):
                hits.append(text[s:span[-1][2]])
            elif any(x in RELATION_VERBS for x in lows) and len(span) >= 5:
                hits.append(text[s:span[-1][2]])
    return hits[:5]


def basic_quality(text: str, min_words: int, max_words: int) -> tuple[bool, dict]:
    words=wc(text)
    sw=sentence_words(text)
    ent=entities(text)
    lower=text.lower()
    bad=sum(1 for b in BAD_SUBSTR if b in lower)
    avg_sent=statistics.mean(sw) if sw else 999
    alpha=sum(ch.isalpha() for ch in text); total=max(1,len(text))
    metrics={'words':words,'sentences':len(sw),'avg_sentence_words':avg_sent,'entities':len(ent),'unique_entities':len(set(ent)),'relation_cues':relation_count(text),'bad_substrings':bad,'alpha_ratio':alpha/total}
    ok=(min_words <= words <= max_words and len(sw)>=2 and 7 <= avg_sent <= 35 and bad==0 and metrics['alpha_ratio']>0.65)
    return ok, metrics


def relation_quality(text: str, base_metrics: dict) -> tuple[bool, dict]:
    hits=explicit_relation_hits(text)
    metrics=dict(base_metrics)
    metrics['explicit_relation_hits']=len(hits)
    metrics['relation_hit_samples']=hits
    # Require multiple entities and dense explicit relation signal.
    ok=(base_metrics['unique_entities'] >= 2 and base_metrics['relation_cues'] >= 8 and len(hits) >= 1)
    return ok, metrics

@dataclass
class DocRec:
    doc_id: int
    text: str
    words: int
    arm_source: str
    metrics: dict


def stream_fineweb(dataset_name: str, config: str, max_stream_docs: int):
    from datasets import load_dataset
    ds=load_dataset(dataset_name, name=config, split='train', streaming=True)
    for i,row in enumerate(ds):
        if i>=max_stream_docs: break
        txt=row.get('text','') if isinstance(row,dict) else ''
        yield i, txt


def pack_records(records: list[DocRec], target_words: int, words_per_example: int, source_label: str, seed: int):
    rng=random.Random(seed)
    # Preserve doc order after deterministic shuffle of selected records to avoid source-order confound.
    recs=list(records)
    rng.shuffle(recs)
    rows=[]; buf=[]; bw=0; ex_id=0; consumed=0; doc_ids=[]
    for r in recs:
        toks=r.text.split()
        pos=0
        while pos < len(toks) and consumed < target_words:
            need=min(words_per_example-bw, target_words-consumed, len(toks)-pos)
            if need<=0: break
            buf.extend(toks[pos:pos+need]); bw += need; consumed += need; pos += need
            doc_ids.append(r.doc_id)
            if bw==words_per_example or consumed==target_words:
                rows.append({'example_id':ex_id,'source':source_label,'text':' '.join(buf),'words':bw,'doc_ids':sorted(set(doc_ids))[:20]})
                ex_id += 1; buf=[]; bw=0; doc_ids=[]
    if consumed != target_words:
        raise RuntimeError(f'Could not pack exact target words {consumed} vs {target_words} for {source_label}; records={len(records)}')
    return rows


def write_jsonl(rows: list[dict], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')


def summarize_records(records: list[DocRec]) -> dict:
    if not records:
        return {'docs':0}
    keys=['words','sentences','avg_sentence_words','unique_entities','relation_cues','explicit_relation_hits']
    out={'docs':len(records),'words_total':sum(r.words for r in records)}
    for k in keys:
        vals=[r.metrics.get(k,0) for r in records]
        out[k+'_mean']=float(statistics.mean(vals)) if vals else None
        out[k+'_median']=float(statistics.median(vals)) if vals else None
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset_name',default='HuggingFaceFW/fineweb-edu')
    ap.add_argument('--config',default='sample-10BT')
    ap.add_argument('--max_stream_docs',type=int,default=20000)
    ap.add_argument('--target_words',type=int,default=1_000_000)
    ap.add_argument('--words_per_example',type=int,default=160)
    ap.add_argument('--min_doc_words',type=int,default=80)
    ap.add_argument('--max_doc_words',type=int,default=800)
    ap.add_argument('--seed',type=int,default=277)
    ap.add_argument('--out_dir',default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--sample_n',type=int,default=30)
    args=ap.parse_args()
    setup_env(); t0=time.time(); out_dir=pathlib.Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    random_quality=[]; relation=[]; streamed=0; basic_pass=0; rel_pass=0
    samples={'random_quality':[],'relation_explicit':[]}
    for doc_id, raw in stream_fineweb(args.dataset_name,args.config,args.max_stream_docs):
        streamed += 1
        text=clean_text(raw)
        ok,m=basic_quality(text,args.min_doc_words,args.max_doc_words)
        if not ok: continue
        basic_pass += 1
        d=DocRec(doc_id,text,wc(text),'fineweb_edu_random_quality',m)
        random_quality.append(d)
        rok,rm=relation_quality(text,m)
        if rok:
            rel_pass += 1
            relation.append(DocRec(doc_id,text,wc(text),'fineweb_edu_relation_explicit',rm))
            if len(samples['relation_explicit'])<args.sample_n:
                samples['relation_explicit'].append({'doc_id':doc_id,'words':wc(text),'metrics':rm,'text':text[:1200]})
        if len(samples['random_quality'])<args.sample_n:
            samples['random_quality'].append({'doc_id':doc_id,'words':wc(text),'metrics':m,'text':text[:1200]})
        # Stop once both arms have enough words with some margin.
        if sum(r.words for r in random_quality) >= args.target_words*1.25 and sum(r.words for r in relation) >= args.target_words*1.25:
            break
    rand_words=sum(r.words for r in random_quality); rel_words=sum(r.words for r in relation)
    if rand_words < args.target_words or rel_words < args.target_words:
        # Still write summaries/samples for yield diagnosis.
        meta={'status':'INSUFFICIENT_YIELD','streamed_docs':streamed,'basic_pass_docs':basic_pass,'relation_pass_docs':rel_pass,'random_words':rand_words,'relation_words':rel_words,'target_words':args.target_words,'elapsed_sec':time.time()-t0,'random_summary':summarize_records(random_quality),'relation_summary':summarize_records(relation)}
        (out_dir/'fineweb_relation_materialization_meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        (out_dir/'fineweb_relation_samples.json').write_text(json.dumps(samples,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        raise RuntimeError(f'Insufficient yield random={rand_words} relation={rel_words} target={args.target_words}; meta={out_dir}')
    rand_rows=pack_records(random_quality,args.target_words,args.words_per_example,'fineweb_edu_random_quality',args.seed)
    rel_rows=pack_records(relation,args.target_words,args.words_per_example,'fineweb_edu_relation_explicit',args.seed)
    rand_path=out_dir/f'fineweb_random_quality_{args.target_words}w.jsonl'
    rel_path=out_dir/f'fineweb_relation_explicit_{args.target_words}w.jsonl'
    write_jsonl(rand_rows,rand_path); write_jsonl(rel_rows,rel_path)
    meta={
        'status':'FINEWEB_RELATION_MATCHED_MATERIALIZED',
        'dataset_name':args.dataset_name,'config':args.config,'max_stream_docs':args.max_stream_docs,
        'streamed_docs':streamed,'basic_pass_docs':basic_pass,'relation_pass_docs':rel_pass,
        'target_words_per_arm':args.target_words,'words_per_example':args.words_per_example,'seed':args.seed,
        'random_jsonl':str(rand_path),'relation_jsonl':str(rel_path),
        'random_jsonl_sha256':sha256_file(rand_path),'relation_jsonl_sha256':sha256_file(rel_path),
        'random_rows':len(rand_rows),'relation_rows':len(rel_rows),
        'random_words_in_pool':rand_words,'relation_words_in_pool':rel_words,
        'random_summary':summarize_records(random_quality),'relation_summary':summarize_records(relation),
        'validation':{
            'random_exact_words':sum(r['words'] for r in rand_rows)==args.target_words,
            'relation_exact_words':sum(r['words'] for r in rel_rows)==args.target_words,
            'row_words_max_random':max(r['words'] for r in rand_rows),
            'row_words_max_relation':max(r['words'] for r in rel_rows),
            'same_words_per_arm':sum(r['words'] for r in rand_rows)==sum(r['words'] for r in rel_rows),
        },
        'elapsed_sec':time.time()-t0,
    }
    meta_path=out_dir/'fineweb_relation_materialization_meta.json'
    sample_path=out_dir/'fineweb_relation_samples.json'
    meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    sample_path.write_text(json.dumps(samples,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'meta':str(meta_path),'samples':str(sample_path),'random_jsonl':str(rand_path),'relation_jsonl':str(rel_path),'streamed_docs':streamed,'relation_pass_docs':rel_pass,'elapsed_sec':meta['elapsed_sec']},indent=2))
    sys.stdout.flush(); sys.stderr.flush()
    # HuggingFace streaming/pyarrow occasionally trips a Python finalization crash after all files are written.
    # Exit immediately after successful materialization so the artifact status is not contaminated by cleanup.
    os._exit(0)

if __name__=='__main__':
    main()
