#!/usr/bin/env python3
"""research v2: exact prediction-geometry scaffold.

Scientific contrasts:
  A. Own compact × direction topology
     - compact_oneway: source→rewrite every epoch
     - compact_reciprocal: source→rewrite on even epochs, rewrite→source on odd epochs
     Exact same pair-token multiset, dose, and pair boundaries; only prediction direction differs.

  B. Copy-matched extractive comparison
     - semantic_extract: source span of rewrite length maximizing lexical overlap with rewrite
     - random_extract: random source span of same length
     Both are 100% literal source tokens and exactly length matched. Difference measures whether
     rewrite-guided semantic selection adds value beyond generic literal replay.
     Each is also crossed with one-way/reciprocal direction.

  C. Adjbreak external perturbation
     - same-length rewrite derangement; excluded from central topology inference because
       correspondence and literal overlap are structurally entangled.

All arms reference the same 9,576,489-word filler and neutral tokenizer. Main six arms each
contain 423,511 pair words + 9,576,489 filler words = exactly 10,000,000 words per epoch.
"""
import json, pathlib, hashlib, random, statistics, collections, re

PAIR_FILE = pathlib.Path("experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl")
FILLER_FILE = pathlib.Path("experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/filler_rows.jsonl")
TOKENIZER_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/neutral_tokenizer")
OUT = pathlib.Path("experiments/archive/representation_and_objectives/data/prediction_geometry_scaffold_v2")
WORD_RE = re.compile(r"[A-Za-z0-9']+")

def norm_words(s): return [x.lower() for x in WORD_RE.findall(s)]
def overlap_frac(source, side):
    ss=set(norm_words(source)); w=norm_words(side)
    return sum(x in ss for x in w)/max(len(w),1)
def rewrite_overlap(span, rewrite):
    a=set(norm_words(span)); b=set(norm_words(rewrite))
    return len(a&b)/max(len(b),1)
def windows(words,n):
    if len(words)<=n: return [(0,words)]
    return [(i,words[i:i+n]) for i in range(len(words)-n+1)]
def extracts(p):
    sw=p['source_text'].split(); n=p['rewrite_words']; ws=windows(sw,n)
    # semantic: highest content-word overlap with rewrite; deterministic earliest tie
    sem=max(ws,key=lambda x:(rewrite_overlap(' '.join(x[1]),p['rewrite_text']),-x[0]))[1]
    seed=int(hashlib.sha256((p['pair_id']+'|random_extract').encode()).hexdigest()[:8],16)
    rnd=random.Random(seed).choice(ws)[1]
    return ' '.join(sem),' '.join(rnd)
def derangement(pairs):
    by=collections.defaultdict(list)
    for p in pairs: by[p['rewrite_words']].append(p)
    out={}; excl=[]
    for n,g in by.items():
        if len(g)<2: excl += [p['pair_id'] for p in g]; continue
        for i,p in enumerate(g): out[p['pair_id']]=g[(i+1)%len(g)]['rewrite_text']
    return out,excl
def pct(v,p):
    s=sorted(v); return s[min(int(len(s)*p),len(s)-1)]
def stats(v): return {'mean':statistics.mean(v),'median':statistics.median(v),'p10':pct(v,.1),'p90':pct(v,.9),'min':min(v),'max':max(v)}
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pairs=[json.loads(l) for l in PAIR_FILE.open()]
    ex={}; der,excl=derangement(pairs)
    for p in pairs: ex[p['pair_id']]=extracts(p)
    types={
      'compact':lambda p:p['rewrite_text'],
      'semantic_extract':lambda p:ex[p['pair_id']][0],
      'random_extract':lambda p:ex[p['pair_id']][1],
      'adjbreak':lambda p:der.get(p['pair_id'])}
    arms={}
    overlap={}
    rewrite_overlap_stats={}
    for typ,fn in types.items():
        vals=[]; ro=[]
        for p in pairs:
            side=fn(p)
            if side is None: continue
            vals.append(overlap_frac(p['source_text'],side))
            ro.append(rewrite_overlap(side,p['rewrite_text']))
        overlap[typ]=stats(vals); rewrite_overlap_stats[typ]=stats(ro)
    for typ in types:
      for topo in ['oneway','reciprocal']:
        name=f'{typ}_{topo}'
        path=OUT/f'{name}.jsonl'; lines=[]; pair_words=0; skipped=0
        for p in pairs:
            side=types[typ](p)
            if side is None: skipped+=1; continue
            assert len(side.split())==p['rewrite_words'], (typ,p['pair_id'],len(side.split()),p['rewrite_words'])
            lines.append(json.dumps({'pair_id':p['pair_id'],'source_text':p['source_text'],
              'side_text':side,'source_words':p['source_words'],'side_words':p['rewrite_words'],
              'data_type':typ,'topology':topo},ensure_ascii=False))
            pair_words += p['source_words']+p['rewrite_words']
        path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        arms[name]={'rows':len(lines),'skipped':skipped,'pair_words_per_epoch':pair_words,
          'filler_words_per_epoch':9576489,'total_words_per_epoch':pair_words+9576489,
          'path':str(path),'sha256':sha(path)}
        print(name,arms[name])
    # Main arms exact 10M; adjbreak differs only by three excluded singleton pairs.
    for n,a in arms.items():
        if not n.startswith('adjbreak'):
            assert a['total_words_per_epoch']==10_000_000,(n,a)
    manifest={'status':'PREDICTION_GEOMETRY_SCAFFOLD_V2_BUILT',
      'central_inference':('Same pair-token multiset and exact 10M words/pass; reciprocal reverses every pair on odd epochs, '
        'oneway repeats source→side. Two epochs therefore compare forward+forward against forward+reverse.'),
      'copy_matched_control':('semantic_extract and random_extract are contiguous source spans, 100% literal copy, exact rewrite length; '
        'they differ only in rewrite-guided semantic selection.'),
      'adjbreak_role':'External correspondence perturbation only; not used alone to separate correspondence from copying.',
      'pair_source':str(PAIR_FILE),'pair_source_sha256':sha(PAIR_FILE),'filler':str(FILLER_FILE),
      'filler_sha256':sha(FILLER_FILE),'tokenizer':str(TOKENIZER_DIR),'pairs':len(pairs),
      'overlap_with_source':overlap,'overlap_with_rewrite':rewrite_overlap_stats,
      'adjbreak_excluded':excl,'arms':arms,
      'planned_screen':{'epochs':2,'exact_words_exposure':20_000_000,
        'primary_arms':['compact_oneway','compact_reciprocal','semantic_extract_oneway','semantic_extract_reciprocal','random_extract_oneway','random_extract_reciprocal'],
        'external_arms':['adjbreak_oneway','adjbreak_reciprocal']}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'status':manifest['status'],'overlap_source':overlap,'overlap_rewrite':rewrite_overlap_stats},indent=2))
if __name__=='__main__': main()
