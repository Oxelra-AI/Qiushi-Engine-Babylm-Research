#!/usr/bin/env python3
"""research: corpus-internal source-phase analysis for the repeated 10M stream.

The scale1.75 run repeats the exact legal 10M pool. A checkpoint such as 82M is
not arbitrary in corpus coordinates: it is 2M words into a repeated pass. This
script maps checkpoint word counts to source-stratified recent exposure windows.
It uses only legal corpus metadata and no evaluation labels.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("experiments/archive/frontier_consolidation")
POOL = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
OUT_DIR = ROOT / "data/stream_order_source_phase"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TOTAL_WORDS = 10_000_000
CHECKPOINTS = [77_000_000,78_000_000,79_000_000,80_000_000,81_000_000,82_000_000,83_000_000,84_000_000,85_000_000,90_000_000,100_000_000]
WINDOWS = [250_000, 500_000, 1_000_000, 2_000_000]


def load_rows():
    rows=[]; cum=0
    with open(POOL) as f:
        for idx,line in enumerate(f):
            o=json.loads(line)
            w=int(o["words"])
            rows.append({"idx":idx,"start":cum,"end":cum+w,"words":w,"source":o["source"],"example_id":o.get("example_id")})
            cum += w
    assert cum == TOTAL_WORDS, cum
    return rows


def source_distribution_interval(rows, start, end):
    # Interval on circular 10M pass coordinates [start,end), allowing wrap.
    if end <= start:
        parts=[(start,TOTAL_WORDS),(0,end)]
    else:
        parts=[(start,end)]
    ctr=Counter(); row_ctr=Counter()
    for a,b in parts:
        for r in rows:
            ov=max(0, min(b,r["end"])-max(a,r["start"]))
            if ov>0:
                ctr[r["source"]]+=ov
                row_ctr[r["source"]]+=1
    return {s:{"words":int(w),"fraction":w/sum(ctr.values()),"overlap_rows":row_ctr[s]} for s,w in ctr.most_common()}


def segment_source_runs(rows):
    segs=[]
    cur=None
    for r in rows:
        if cur is None or r["source"] != cur["source"]:
            if cur is not None: segs.append(cur)
            cur={"source":r["source"],"start":r["start"],"end":r["end"],"rows":1,"words":r["words"]}
        else:
            cur["end"]=r["end"]; cur["rows"]+=1; cur["words"]+=r["words"]
    if cur: segs.append(cur)
    return segs


def main():
    rows=load_rows()
    total_by_source=Counter()
    for r in rows: total_by_source[r["source"]]+=r["words"]
    segs=segment_source_runs(rows)
    ckpt_records=[]
    for ck in CHECKPOINTS:
        pos=ck % TOTAL_WORDS
        if pos==0: pos=TOTAL_WORDS
        rec={"checkpoint_words":ck,"checkpoint_M":ck//1_000_000,"position_in_10M_pass":pos,"recent_windows":{},"next_windows":{}}
        for win in WINDOWS:
            start=(pos-win)%TOTAL_WORDS
            end=pos%TOTAL_WORDS
            if end==0: end=TOTAL_WORDS
            rec["recent_windows"][str(win)] = source_distribution_interval(rows,start,end)
            rec["next_windows"][str(win)] = source_distribution_interval(rows,pos%TOTAL_WORDS,(pos+win)%TOTAL_WORDS)
        ckpt_records.append(rec)
    result={
        "status":"STREAM_ORDER_SOURCE_PHASE",
        "created_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "pool":str(POOL),
        "total_words":TOTAL_WORDS,
        "total_by_source":dict(total_by_source),
        "source_runs":segs,
        "checkpoint_records":ckpt_records,
        "interpretation_basis":"Corpus metadata only; no official labels. Checkpoint phase determines which sources were recently rehearsed immediately before a saved model.",
    }
    out_json=OUT_DIR/"stream_order_source_phase.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines=[]
    lines.append("# research stream-order source phase")
    lines.append("")
    lines.append("The 10M legal pool is repeated. A checkpoint at `X` words has phase `X mod 10M`; this table gives recently rehearsed source mass before each late checkpoint.")
    lines.append("")
    lines.append("## Source runs in one 10M pass")
    lines.append("| source | start words | end words | rows | words |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in segs:
        lines.append(f"| {s['source']} | {s['start']} | {s['end']} | {s['rows']} | {s['words']} |")
    lines.append("")
    for win in WINDOWS:
        lines.append(f"## Recent {win//1000}k words before checkpoint")
        lines.append("| ckpt | phase | top sources (fraction) |")
        lines.append("|---:|---:|---|")
        for rec in ckpt_records:
            dist=rec["recent_windows"][str(win)]
            top=', '.join([f"{s} {v['fraction']:.2f}" for s,v in list(dist.items())[:4]])
            lines.append(f"| {rec['checkpoint_M']}M | {rec['position_in_10M_pass']} | {top} |")
        lines.append("")
    lines.append(f"JSON: `{out_json}`")
    out_md=(OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/stream_order_source_phase/stream_order_source_phase.md')
    out_md.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({"status":result["status"],"out_json":str(out_json),"out_md":str(out_md),"source_runs":segs[:10]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
