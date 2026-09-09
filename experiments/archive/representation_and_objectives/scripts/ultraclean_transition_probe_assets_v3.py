#!/usr/bin/env python3
"""research v3: 30k ultra-clean transition treatment and anchored control."""
from __future__ import annotations
import csv, importlib.util, json, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
BASE_SCRIPT = A01_WS / "scripts/ultraclean_transition_probe_assets.py"
OUT = A01_WS / "data/ultraclean_transition_probe_v3"
NOTE = A01_WS / "notes/ultraclean_transition_probe_assets_v3.md"
TARGET = 30_000

spec = importlib.util.spec_from_file_location("ultra_base_fixed", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)  # type: ignore[union-attr]
base.TARGET = TARGET
core = base.core

QUOTAS = {
    "physical_material_transition": 20_000,
    "spatial_transition": 8_000,
    "temporal_quantity_transition": 2_000,
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
    """Reachability subset fill with stable parent links.

    Rows are already sorted by desired priority.  We add each newly reachable word
    count only once, so parent chains are not invalidated by later updates.  This
    maximizes filled words under the target and tends to retain earlier high-priority
    rows without the reconstruction-cycle bug of a 1D score DP.
    """
    if target <= 0 or not rows: return []
    reachable={0}
    parent: dict[int, tuple[int,int]] = {}
    for i,r in enumerate(rows):
        w=int(r["words"])
        if w>target: continue
        for s in sorted(list(reachable), reverse=True):
            ns=s+w
            if ns>target or ns in reachable:
                continue
            reachable.add(ns); parent[ns]=(s,i)
    best_s=max(reachable)
    out=[]; s=best_s; seen=set()
    while s>0 and s in parent:
        ps,i=parent[s]
        if i in seen: break
        seen.add(i); out.append(rows[i]); s=ps
    out.reverse(); return out


def build_treatment(pool: list[dict[str,Any]]) -> list[dict[str,Any]]:
    by=defaultdict(list)
    for r in pool:
        by[str(r.get("selected_route_bucket"))].append(r)
    for arr in by.values():
        arr.sort(key=lambda r:(-float(r.get("core_score",0)), abs(int(r["words"])-36), r.get("text_sha256","")))
    selected=[]; used=set(); total=0
    for bucket, quota in QUOTAS.items():
        take=subset_fill(by[bucket], min(quota, sum(int(r["words"]) for r in by[bucket])))
        for r in take:
            key=str(r.get("text_sha256") or r.get("text"))
            if key not in used and total+int(r["words"])<=TARGET:
                selected.append(r); used.add(key); total+=int(r["words"])
    remaining=[r for r in pool if str(r.get("text_sha256") or r.get("text")) not in used]
    remaining.sort(key=lambda r:(-float(r.get("core_score",0)), abs(int(r["words"])-36), r.get("text_sha256","")))
    extra=subset_fill(remaining, TARGET-total)
    for r in extra:
        key=str(r.get("text_sha256") or r.get("text"))
        if key not in used and total+int(r["words"])<=TARGET:
            selected.append(r); used.add(key); total+=int(r["words"])
    selected.sort(key=lambda r:(str(r.get("selected_route_bucket")), -float(r.get("core_score",0)), r.get("text_sha256","")))
    for r in selected:
        r["required_capability"] = base.required_cap(r)
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
    pool.sort(key=lambda r:(str(r.get('selected_route_bucket')), -float(r.get('core_score',0)), abs(int(r['words'])-36), r.get('text_sha256','')))
    treatment=build_treatment(pool)
    control_pool, scan=base.collect_control_pool()
    controls=base.pick_matched_controls(treatment, control_pool)
    write_jsonl(OUT/'ultraclean_transition_candidate_pool_v3.jsonl', pool)
    write_jsonl(OUT/'ultraclean_transition_30k.jsonl', treatment)
    write_jsonl(OUT/'ultraclean_anchor_control_30k.jsonl', controls)
    st=summarize(treatment,'treatment'); sc=summarize(controls,'control')
    metrics={'word_diff_control_minus_treatment':sc['words']-st['words'],'source_l1':l1(st['source_words'],sc['source_words']),'length_bin_l1':l1(st['length_bin_words'],sc['length_bin_words']),'required_capability_l1':l1(st['required_capability_words'],sc['required_capability_words'])}
    csv_path=OUT/'ultraclean_probe_v3_summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        fields=['arm','sentences','words','mean_words','median_words','source_words','length_bin_words','bucket_words','required_capability_words','all_capability_words','match_modes','mean_core_score','mean_relation_markers']
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for arm,sm in [('treatment',st),('anchor_control',sc)]:
            row={'arm':arm}
            for k in fields[1:]:
                v=sm.get(k,'')
                row[k]=json.dumps(v,sort_keys=True) if isinstance(v,dict) else v
            w.writerow(row)
    summary={'status':'ULTRACLEAN_TRANSITION_PROBE_ASSETS_V3_BUILT','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),'repair_over_v1_v2':'fixed subset-fill to maximize filled words, lowered target to 30k after strict filter showed ~35k-word pool','ultraclean_candidate_count':len(pool),'ultraclean_candidate_words':sum(int(r['words']) for r in pool),'treatment_summary':st,'control_summary':sc,'match_metrics':metrics,'control_pool_count':len(control_pool),'control_pool_words':sum(int(r['words']) for r in control_pool),'control_scan':scan,'files':{'candidate_pool':str(OUT/'ultraclean_transition_candidate_pool_v3.jsonl'),'treatment_30k':str(OUT/'ultraclean_transition_30k.jsonl'),'anchor_control_30k':str(OUT/'ultraclean_anchor_control_30k.jsonl'),'summary_csv':str(csv_path),'note':str(NOTE)},'elapsed_sec':round(time.time()-t0,3)}
    (OUT/'ultraclean_transition_probe_assets_v3.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    note=['# research — ultra-clean transition probe assets v3','', '## Purpose','', 'After the v1/v2 exact-subset bugs, this repaired asset builds the intended 30k-word transition treatment slice and a no-explicit-relation anchored control. It is still only a future cheap probe asset: the active FW compact/breadth endpoints must be read before any new GPU route is considered.', '', '## Counts','', f"- Ultra-clean candidate pool: {len(pool)} sentences / {summary['ultraclean_candidate_words']} words.", f"- Treatment: {st['sentences']} sentences / {st['words']} words; bucket words {st.get('bucket_words')}.", f"- Anchor control: {sc['sentences']} sentences / {sc['words']} words; word difference {metrics['word_diff_control_minus_treatment']}; source L1 {metrics['source_l1']:.4f}; length-bin L1 {metrics['length_bin_l1']:.4f}; required-capability L1 {metrics['required_capability_l1']:.4f}.", '', '## Scientific reading','', 'This is the cleanest transition-substrate probe asset in this comparison. It deliberately trades volume for purity. If the running FW compact/breadth endpoints do not move relation-conditioned failures, the lowest-cost meaningful test is to compare this treatment against the anchored control after a shared pretrained checkpoint, while reading EWoK interaction failures and GlobalPIQA hard-row margins. Do not scale it directly to 100M.', '', '## Files','', f"- summary JSON: `{OUT/'ultraclean_transition_probe_assets_v3.json'}`", f"- treatment: `{OUT/'ultraclean_transition_30k.jsonl'}`", f"- anchor control: `{OUT/'ultraclean_anchor_control_30k.jsonl'}`", f"- candidate pool: `{OUT/'ultraclean_transition_candidate_pool_v3.jsonl'}`", f"- summary CSV: `{csv_path}`"]
    NOTE.write_text('\n'.join(note)+'\n', encoding='utf-8')
    print(json.dumps({'status':summary['status'],'candidate_words':summary['ultraclean_candidate_words'],'treatment_words':st['words'],'control_words':sc['words'],'bucket_words':st.get('bucket_words'),'match_metrics':metrics,'json':str(OUT/'ultraclean_transition_probe_assets_v3.json'),'note':str(NOTE),'elapsed_sec':summary['elapsed_sec']}, indent=2), flush=True)

if __name__=='__main__': main()
