#!/usr/bin/env python3
"""research: repaired ultra-clean transition probe assets (30k).

The first ultra-clean attempt correctly showed that the very strict candidate pool is
only ~38k words, but its exact-50k selector returned a tiny slice.  This repair uses
a 30k word target, fills word budgets primarily by word coverage, and reuses the
same strict no-evaluation-item filters.
"""
from __future__ import annotations
import csv, importlib.util, json, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
BASE_SCRIPT = A01_WS / "scripts/ultraclean_transition_probe_assets.py"
OUT = A01_WS / "data/ultraclean_transition_probe_v2"
NOTE = A01_WS / "notes/ultraclean_transition_probe_assets_v2.md"
TARGET = 30_000

spec = importlib.util.spec_from_file_location("ultra_v1", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)  # type: ignore[union-attr]
base.TARGET = TARGET
core = base.core

QUOTAS = {
    "physical_material_transition": 18_000,
    "spatial_transition": 8_000,
    "temporal_quantity_transition": 3_500,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")


def subset_fill(rows: list[dict[str,Any]], target: int) -> list[dict[str,Any]]:
    """Select an exact or nearest-under subset, prioritizing word coverage then score."""
    if target <= 0 or not rows: return []
    score=[-10**9]*(target+1); prev=[None]*(target+1); score[0]=0
    for i,r in enumerate(rows):
        w=int(r["words"])
        if w>target: continue
        val=int(round(float(r.get("core_score",0))*100))+10
        for s in range(target,w-1,-1):
            if score[s-w] <= -10**8: continue
            nv=score[s-w]+val
            if nv>score[s]: score[s]=nv; prev[s]=(s-w,i)
    reachable=[s for s,v in enumerate(score) if v>-10**8]
    if not reachable: return []
    best_s=max(reachable)  # fill first; score only decides each filled size
    used=[]; s=best_s; ids=set()
    while s>0 and prev[s] is not None:
        ps,i=prev[s]
        if i in ids: break
        ids.add(i); used.append(rows[i]); s=ps
    used.reverse(); return used


def build_treatment(pool: list[dict[str,Any]]) -> list[dict[str,Any]]:
    by=defaultdict(list)
    for r in pool:
        by[str(r.get("selected_route_bucket"))].append(r)
    for arr in by.values():
        arr.sort(key=lambda r:(-float(r.get("core_score",0)), abs(int(r["words"])-36), r.get("text_sha256","")))
    selected=[]; used=set(); total=0
    # Fill physical/spatial/temporal quotas first, then top off with best remaining.
    for bucket, quota in QUOTAS.items():
        take=subset_fill(by[bucket], min(quota, sum(int(r["words"]) for r in by[bucket])))
        for r in take:
            k=str(r.get("text_sha256") or r.get("text"))
            if k not in used and total+int(r["words"])<=TARGET:
                selected.append(r); used.add(k); total+=int(r["words"])
    remaining=[r for r in pool if str(r.get("text_sha256") or r.get("text")) not in used]
    remaining.sort(key=lambda r:(-float(r.get("core_score",0)), abs(int(r["words"])-36), r.get("text_sha256","")))
    extra=subset_fill(remaining, TARGET-total)
    for r in extra:
        k=str(r.get("text_sha256") or r.get("text"))
        if k not in used and total+int(r["words"])<=TARGET:
            selected.append(r); used.add(k); total+=int(r["words"])
    selected.sort(key=lambda r:(str(r.get("selected_route_bucket")), -float(r.get("core_score",0)), r.get("text_sha256","")))
    for r in selected:
        r["ultraclean_v2_required_capability"] = base.required_cap(r)
    return selected


def counter_words(rows, keyfunc):
    c=Counter()
    for r in rows: c[str(keyfunc(r))]+=int(r["words"])
    return c


def summarize(rows, arm):
    total=sum(int(r["words"]) for r in rows)
    sm={"sentences":len(rows),"words":total,"mean_words":total/len(rows) if rows else 0,"median_words":statistics.median([int(r['words']) for r in rows]) if rows else 0,"source_words":dict(counter_words(rows,lambda r:r.get('source_label',''))),"length_bin_words":dict(counter_words(rows,lambda r:r.get('length_bin') or core.len_bin(int(r['words']))))}
    if arm=='treatment':
        sm['bucket_words']=dict(counter_words(rows,lambda r:r.get('selected_route_bucket','')))
        sm['required_capability_words']=dict(counter_words(rows,base.required_cap))
        sm['mean_core_score']=statistics.mean([float(r.get('core_score',0)) for r in rows]) if rows else 0
        rel=[base.relation_marker_sum(r) for r in rows]
        sm['mean_relation_markers']=statistics.mean(rel) if rel else 0
    else:
        sm['required_capability_words']=dict(counter_words(rows,lambda r:r.get('required_capability') or 'topup'))
        cap=Counter()
        for r in rows:
            for c in r.get('anchor_capabilities',[]): cap[c]+=int(r['words'])
        sm['all_capability_words']=dict(cap)
        sm['match_modes']=dict(counter_words(rows,lambda r:r.get('match_mode','')))
    return sm


def l1(a,b):
    keys=set(a)|set(b); ta=sum(a.values()) or 1; tb=sum(b.values()) or 1
    return sum(abs(a.get(k,0)/ta-b.get(k,0)/tb) for k in keys)


def main():
    t0=time.time(); OUT.mkdir(parents=True, exist_ok=True); NOTE.parent.mkdir(parents=True, exist_ok=True)
    raw=load_jsonl(base.CANDIDATES)
    pool=[r for r in raw if base.treatment_keep(r)]
    treatment=build_treatment(pool)
    control_pool, scan=base.collect_control_pool()
    controls=base.pick_matched_controls(treatment, control_pool)
    write_jsonl(OUT/'ultraclean_transition_candidate_pool_v2.jsonl', pool)
    write_jsonl(OUT/'ultraclean_transition_30k.jsonl', treatment)
    write_jsonl(OUT/'ultraclean_anchor_control_30k.jsonl', controls)
    st=summarize(treatment,'treatment'); sc=summarize(controls,'control')
    metrics={'word_diff_control_minus_treatment':sc['words']-st['words'],'source_l1':l1(st['source_words'],sc['source_words']),'length_bin_l1':l1(st['length_bin_words'],sc['length_bin_words']),'required_capability_l1':l1(st['required_capability_words'],sc['required_capability_words'])}
    csv_path=OUT/'ultraclean_probe_v2_summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        fields=['arm','sentences','words','mean_words','median_words','source_words','length_bin_words','bucket_words','required_capability_words','all_capability_words','match_modes','mean_core_score','mean_relation_markers']
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for arm,sm in [('treatment',st),('anchor_control',sc)]:
            row={'arm':arm}
            for k in fields[1:]:
                v=sm.get(k,'')
                row[k]=json.dumps(v,sort_keys=True) if isinstance(v,dict) else v
            w.writerow(row)
    summary={'status':'ULTRACLEAN_TRANSITION_PROBE_ASSETS_V2_BUILT','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),'repair_over_v1':'50k target was impossible after strict filters; v2 uses 30k and word-coverage-first subset fill','ultraclean_candidate_count':len(pool),'ultraclean_candidate_words':sum(int(r['words']) for r in pool),'treatment_summary':st,'control_summary':sc,'match_metrics':metrics,'control_pool_count':len(control_pool),'control_pool_words':sum(int(r['words']) for r in control_pool),'control_scan':scan,'files':{'candidate_pool':str(OUT/'ultraclean_transition_candidate_pool_v2.jsonl'),'treatment_30k':str(OUT/'ultraclean_transition_30k.jsonl'),'anchor_control_30k':str(OUT/'ultraclean_anchor_control_30k.jsonl'),'summary_csv':str(csv_path),'note':str(NOTE)},'elapsed_sec':round(time.time()-t0,3)}
    (OUT/'ultraclean_transition_probe_assets_v2.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    note=['# research — ultra-clean transition probe assets v2','', '## Purpose','', 'The first ultra-clean attempt showed that the strict pool is only about 38k words and the 50k exact selector failed. This repaired version creates a 30k-word treatment slice and a no-explicit-relation anchored control. It remains a future cheap probe asset, not a full route.', '', '## Counts','', f"- Ultra-clean candidate pool: {len(pool)} sentences / {summary['ultraclean_candidate_words']} words.", f"- Treatment: {st['sentences']} sentences / {st['words']} words; bucket words {st.get('bucket_words')}.", f"- Anchor control: {sc['sentences']} sentences / {sc['words']} words; word difference {metrics['word_diff_control_minus_treatment']}; source L1 {metrics['source_l1']:.4f}; length-bin L1 {metrics['length_bin_l1']:.4f}; required-capability L1 {metrics['required_capability_l1']:.4f}.", '', '## Scientific reading','', 'This is the cleanest transition-substrate probe asset produced so far, but it is only 30k words. It can test whether explicit corpus-derived transition material has a detectable direction relative to anchored non-transition content after a pretrained shared checkpoint, but it cannot by itself establish a SOTA route. Because older INITIAL_MODEL_STUDIES BSM screens raised EWoK while hurting GlobalPIQA, any future use must include the GlobalPIQA margin reader and EWoK interaction reader.', '', '## Files','', f"- summary JSON: `{OUT/'ultraclean_transition_probe_assets_v2.json'}`", f"- treatment: `{OUT/'ultraclean_transition_30k.jsonl'}`", f"- anchor control: `{OUT/'ultraclean_anchor_control_30k.jsonl'}`", f"- candidate pool: `{OUT/'ultraclean_transition_candidate_pool_v2.jsonl'}`", f"- summary CSV: `{csv_path}`"]
    NOTE.write_text('\n'.join(note)+'\n', encoding='utf-8')
    print(json.dumps({'status':summary['status'],'candidate_words':summary['ultraclean_candidate_words'],'treatment_words':st['words'],'control_words':sc['words'],'bucket_words':st.get('bucket_words'),'match_metrics':metrics,'json':str(OUT/'ultraclean_transition_probe_assets_v2.json'),'note':str(NOTE),'elapsed_sec':summary['elapsed_sec']}, indent=2), flush=True)

if __name__=='__main__': main()
