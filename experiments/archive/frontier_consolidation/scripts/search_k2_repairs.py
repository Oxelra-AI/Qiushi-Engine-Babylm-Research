#!/usr/bin/env python3
from __future__ import annotations
import collections, itertools, json, pathlib, random, time

ROOT=pathlib.Path('.').resolve()
WS=ROOT/'experiments/archive/frontier_consolidation'
PAIRS=WS/'data/dose_distribution_select/selected_matched_max_pairs.jsonl'
META=WS/'data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_changed_block_rows_meta.jsonl'
OUT=WS/'data/dose_2p64x_permuted_companion_rowholdout_pools/k2_repair_search.json'

def wc(x): return len(str(x or '').split())
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
        p=by[pid]
        slots.append({'pid':pid,'rw':int(p['rewrite_words']),'doc':str(p.get('doc_id') or ''),'row':int(m['row_index'])})
    rows.append({'row':int(m['row_index']),'slots':slots,'sum':sum(s['rw'] for s in slots),'k':len(slots),'docs':set(s['doc'] for s in slots)})
rowmap={r['row']:r for r in rows}
k2=[r for r in rows if r['k']==2]

def valid(row, donor_slots):
    if len(donor_slots)!=row['k']: return False
    if sum(s['rw'] for s in donor_slots)!=row['sum']: return False
    if any(s['row']==row['row'] for s in donor_slots): return False
    if any(s['doc'] in row['docs'] for s in donor_slots): return False
    return True

def solve(subrows, seed=0, limit=500000):
    rng=random.Random(seed)
    donors=[s for r in subrows for s in r['slots']]
    target_order=sorted(subrows, key=lambda r: (len(cands_for_row(r, donors)), r['sum'], r['row']))
    used=set(); assign={}; nodes=0
    cand_cache={r['row']:cands_for_row(r, donors) for r in subrows}
    def rec(i):
        nonlocal nodes
        nodes+=1
        if nodes>limit: return False
        if i==len(target_order): return True
        r=target_order[i]
        cands=[]
        for pair in cand_cache[r['row']]:
            if pair[0]['pid'] in used or pair[1]['pid'] in used: continue
            cands.append(pair)
        # Prefer non-trivial length pairs and avoid using two donors from same source row if possible.
        rng.shuffle(cands)
        cands.sort(key=lambda pair: (pair[0]['row']==pair[1]['row'], abs(pair[0]['rw']-pair[1]['rw'])))
        for pair in cands:
            used.add(pair[0]['pid']); used.add(pair[1]['pid']); assign[r['row']]=pair
            if rec(i+1): return True
            used.remove(pair[0]['pid']); used.remove(pair[1]['pid']); assign.pop(r['row'],None)
        return False
    ok=rec(0)
    return ok, assign, nodes, [r['row'] for r in target_order]

def cands_for_row(row, donors):
    out=[]
    # combinations over small subproblem donors
    for a,b in itertools.combinations(donors,2):
        if a['pid']==b['pid']: continue
        if a['rw']+b['rw']!=row['sum']: continue
        if valid(row,[a,b]): out.append((a,b))
    return out

# Search for a compact helper subset around the two hard rows.
targets=[rowmap[4023], rowmap[4025]]
# Candidate helper k2 rows whose slot lengths can contribute to sums 73/74 or absorb singleton lengths.
helpers=[]
for r in k2:
    if r['row'] in (4023,4025): continue
    lens=sorted(s['rw'] for s in r['slots'])
    if 27 <= r['sum'] <= 90:
        # Keep rows that contain lengths near the hard rows or allow common 73/74 complements.
        if any(x in {31,32,33,34,35,36,37,38,39,40,41,42,43,44} for x in lens) or r['sum'] in {63,64,65,66,67,68,69,70,71,72,73,74,75,76,77}:
            helpers.append(r)
helpers=helpers[:]
print('candidate_helpers',len(helpers))
solutions=[]
# Try increasing helper counts. Deterministic random subsets plus targeted rows by sum.
rng=random.Random(270)
priority=sorted(helpers, key=lambda r: (abs(r['sum']-73.5), r['row']))
for m in range(2,9):
    trials=[]
    trials.append(priority[:m])
    # include diverse exact sums around hard values
    bysum=collections.defaultdict(list)
    for r in helpers: bysum[r['sum']].append(r)
    near=[]
    for s in [58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,75,76,77,78,79,80,81,82,83,84,85,86,87,88]:
        if bysum[s]: near.append(bysum[s][0])
    if len(near)>=m: trials.append(near[:m])
    for _ in range(1000):
        trials.append(rng.sample(helpers,m))
    seen=set()
    for hs in trials:
        key=tuple(sorted(r['row'] for r in hs))
        if key in seen: continue
        seen.add(key)
        sub=targets+hs
        # total donor multiset automatically same subrows; total row sums match total donor lengths.
        ok,assign,nodes,order=solve(sub,seed=m*1000+len(seen),limit=300000)
        if ok:
            sol={'helper_count':m,'rows':[r['row'] for r in sub],'row_sums':{r['row']:r['sum'] for r in sub},'row_lens':{r['row']:[s['rw'] for s in r['slots']] for r in sub},'assign':{str(row):[(s['pid'],s['rw'],s['doc'],s['row']) for s in pair] for row,pair in assign.items()},'nodes':nodes,'order':order}
            solutions.append(sol)
            print(json.dumps({'FOUND':sol},indent=2))
            OUT.parent.mkdir(parents=True,exist_ok=True)
            OUT.write_text(json.dumps({'status':'found','solution':sol,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2),encoding='utf-8')
            raise SystemExit(0)
    print('tried helper_count',m,'no solution')
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps({'status':'not_found','helpers':len(helpers),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2),encoding='utf-8')
raise SystemExit(2)
