#!/usr/bin/env python3
from __future__ import annotations
import collections, itertools, json, pathlib, random, time
ROOT=pathlib.Path('.').resolve(); WS=ROOT/'experiments/archive/frontier_consolidation'
PAIRS=WS/'data/dose_distribution_select/selected_matched_max_pairs.jsonl'
META=WS/'data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_changed_block_rows_meta.jsonl'
OUT=WS/'data/dose_2p64x_permuted_companion_rowholdout_pools/all_exception_repair_search.json'
by={}
for line in PAIRS.open(encoding='utf-8'):
    if line.strip():
        d=json.loads(line); by[d['pair_id']]=d
rows=[]
for line in META.open(encoding='utf-8'):
    if not line.strip(): continue
    m=json.loads(line); ids=m.get('pair_ids') or []
    if not ids: continue
    slots=[]
    for pid in ids:
        p=by[pid]; slots.append({'pid':pid,'rw':int(p['rewrite_words']),'doc':str(p.get('doc_id') or ''),'row':int(m['row_index'])})
    rows.append({'row':int(m['row_index']),'slots':slots,'sum':sum(s['rw'] for s in slots),'k':len(slots),'docs':set(s['doc'] for s in slots)})
rowmap={r['row']:r for r in rows}
length_counts=collections.Counter(s['rw'] for r in rows for s in r['slots'])
singleton_lengths={L for L,c in length_counts.items() if c==1}
exception=[r for r in rows if any(s['rw'] in singleton_lengths for s in r['slots'])]
print('singleton_lengths',singleton_lengths,'exception',[(r['row'],r['sum'],r['k'],[s['rw'] for s in r['slots']]) for r in exception])

def valid(row, donor_slots):
    if len(donor_slots)!=row['k']: return False
    if sum(s['rw'] for s in donor_slots)!=row['sum']: return False
    for s in donor_slots:
        if s['row']==row['row']: return False
        if s['doc'] in row['docs']: return False
    return True

def cands_for(row, donors):
    out=[]
    for comb in itertools.combinations(donors,row['k']):
        if valid(row,comb): out.append(tuple(comb))
    return out

def solve(subrows, seed=0, limit=1000000):
    rng=random.Random(seed)
    donors=[s for r in subrows for s in r['slots']]
    cand={r['row']:cands_for(r,donors) for r in subrows}
    if any(not cand[r['row']] for r in subrows):
        return False,{},0,[r['row'] for r in subrows],{r['row']:len(cand[r['row']]) for r in subrows}
    order=sorted(subrows,key=lambda r:(len(cand[r['row']]),r['k'],r['sum'],r['row']))
    used=set(); assign={}; nodes=0
    def rec(i):
        nonlocal nodes
        nodes+=1
        if nodes>limit: return False
        if i==len(order): return True
        r=order[i]
        cs=[]
        for comb in cand[r['row']]:
            if any(s['pid'] in used for s in comb): continue
            cs.append(comb)
        rng.shuffle(cs)
        cs.sort(key=lambda comb:(len({s['row'] for s in comb}), sum(abs(s['row']-r['row']) for s in comb)))
        for comb in cs:
            for s in comb: used.add(s['pid'])
            assign[r['row']]=comb
            if rec(i+1): return True
            for s in comb: used.remove(s['pid'])
            assign.pop(r['row'],None)
        return False
    ok=rec(0)
    return ok,assign,nodes,[r['row'] for r in order],{r['row']:len(cand[r['row']]) for r in order}

helpers=[]
exception_rows={r['row'] for r in exception}
for r in rows:
    if r['row'] in exception_rows: continue
    if r['k'] in {2,3,4} and 25 <= r['sum'] <= 95:
        lens=[s['rw'] for s in r['slots']]
        if any(8 <= x <= 45 for x in lens): helpers.append(r)
# prioritize rows around needed sums and with small k
helpers=sorted(helpers,key=lambda r:(abs(r['sum']-60),r['k'],r['row']))
rng=random.Random(2701)
for m in range(3,11):
    trials=[]
    trials.append(helpers[:m])
    # include rows from previous partial solution when available
    seed_rows=[rowmap[x] for x in [4015,4889] if x in rowmap and x not in exception_rows]
    if len(seed_rows)<=m:
        trials.append(seed_rows+helpers[:max(0,m-len(seed_rows))])
    for _ in range(2000): trials.append(rng.sample(helpers,m))
    seen=set()
    for hs in trials:
        key=tuple(sorted(r['row'] for r in hs))
        if key in seen: continue
        seen.add(key)
        sub=exception+hs
        ok,assign,nodes,order,cnts=solve(sub,seed=m*100000+len(seen),limit=500000)
        if ok:
            sol={'helper_count':m,'rows':[r['row'] for r in sub],'row_sums':{r['row']:r['sum'] for r in sub},'row_k':{r['row']:r['k'] for r in sub},'row_lens':{r['row']:[s['rw'] for s in r['slots']] for r in sub},'assign':{str(row):[(s['pid'],s['rw'],s['doc'],s['row']) for s in comb] for row,comb in assign.items()},'nodes':nodes,'order':order,'candidate_counts':cnts}
            print(json.dumps({'FOUND':sol},indent=2))
            OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps({'status':'found','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'solution':sol},indent=2),encoding='utf-8')
            raise SystemExit(0)
    print('tried',m,'helpers; no solution')
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps({'status':'not_found','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'helpers':len(helpers)},indent=2),encoding='utf-8')
raise SystemExit(2)
