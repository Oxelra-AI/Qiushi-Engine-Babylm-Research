#!/usr/bin/env python3
"""research v2: materialize low-truncation stagewise JSONL for Phase-2.

Corrects the old Phase-2 sequence curriculum where 160-word examples were
truncated to seq64/128 while all 160 words were counted.  Stage v2 chunks are:
  stage1 0-30M:  <=32 words, seq_len 64, batch512
  stage2 30-60M: <=64 words, seq_len 128, batch256
  stage3 60-100M: 160 words, seq_len 256, batch128
The first two stages minimize hidden token loss while preserving the official
100M word exposure schedule and constant 32768 padded positions/update.
"""
from __future__ import annotations
import json, hashlib, time
from pathlib import Path

ROOT = Path("experiments/archive/compact_experience")
OUT = ROOT / "data/phase2_stagewise_v2"
INPUTS = {
    "official": ROOT / "data/fixedinit_replication/official_100M.jsonl",
    "mix25": ROOT / "data/mixture/training_files/mix_25pct_100M.jsonl",
}
STAGES = [
    {"stage": 1, "row_start": 0, "row_end": 187_500, "max_chunk_words": 32, "seq_len": 64, "batch_size": 512, "target_words": 30_000_000},
    {"stage": 2, "row_start": 187_500, "row_end": 375_000, "max_chunk_words": 64, "seq_len": 128, "batch_size": 256, "target_words": 30_000_000},
    {"stage": 3, "row_start": 375_000, "row_end": 625_000, "max_chunk_words": 160, "seq_len": 256, "batch_size": 128, "target_words": 40_000_000},
]

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1<<20), b''): h.update(c)
    return h.hexdigest()

def iter_chunks(words: list[str], max_n: int):
    for i in range(0, len(words), max_n):
        yield i // max_n, words[i:i+max_n]

def materialize_one(name: str, src: Path) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    stage_out={s['stage']: OUT / f"{name}_stage{s['stage']}_seq{s['seq_len']}_{s['target_words']//1_000_000}M_v2.jsonl" for s in STAGES}
    handles={k:p.open('w',encoding='utf-8') for k,p in stage_out.items()}
    stats={s['stage']:{'rows':0,'words':0,'seq_len':s['seq_len'],'batch_size':s['batch_size'],'max_chunk_words':s['max_chunk_words'],'target_words':s['target_words'],'path':str(stage_out[s['stage']])} for s in STAGES}
    try:
        with src.open(encoding='utf-8') as f:
            for row_idx,line in enumerate(f):
                r=json.loads(line); words=r['text'].split(); field=int(r.get('words',len(words)))
                if field != 160 or len(words) != 160:
                    raise RuntimeError(f"{src} row {row_idx}: expected 160 words, field={field}, actual={len(words)}")
                stage=None
                for s in STAGES:
                    if s['row_start'] <= row_idx < s['row_end']:
                        stage=s; break
                if stage is None: raise RuntimeError(f"row {row_idx} not assigned")
                st=stage['stage']
                for chunk_id,ww in iter_chunks(words, stage['max_chunk_words']):
                    out={'text':' '.join(ww),'words':len(ww),'example_id':stats[st]['rows'],
                         'source':f"{r.get('source','')}::phase2_stage{st}::row{row_idx}::chunk{chunk_id}",
                         'orig_example_id':r.get('example_id'),'orig_row_index':row_idx,'chunk_id':chunk_id}
                    handles[st].write(json.dumps(out, ensure_ascii=False)+'\n')
                    stats[st]['rows']+=1; stats[st]['words']+=len(ww)
    finally:
        for h in handles.values(): h.close()
    total=sum(v['words'] for v in stats.values())
    if total != 100_000_000: raise RuntimeError(f"{name} total {total}")
    for s in STAGES:
        st=s['stage']
        if stats[st]['words'] != s['target_words']:
            raise RuntimeError(f"{name} stage {st} words {stats[st]['words']} != {s['target_words']}")
        stats[st]['sha256']=sha256_file(stage_out[st])
    return {'input':str(src),'input_sha256':sha256_file(src),'stages':stats,'total_words':total}

def main():
    t0=time.time(); payload={'status':'PHASE2_STAGEWISE_V2_MATERIALIZED','design':'low-truncation chunks 32w->64, 64w->128, 160w->256; batch 512/256/128; no aligned text in shared tokenizer','inputs':{},'stages':STAGES}
    for name,src in INPUTS.items():
        print('materializing',name,src,flush=True); payload['inputs'][name]=materialize_one(name,src)
    payload['elapsed_sec']=round(time.time()-t0,1)
    out=OUT/'stagewise_v2_summary.json'; out.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(payload,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
