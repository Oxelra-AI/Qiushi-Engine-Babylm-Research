#!/usr/bin/env python3
"""research semantic/lexical audit of research active relation-pair pool.

This is CPU-only and model-free. It helps interpret four-cell calibration by asking
whether the pool mostly pairs arbitrary sentence-specific targets whose cross-cell
substitution is trivially implausible, rather than contexts that create genuinely
competing relation-conditioned alternatives.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path('.').resolve()
PAIR_POOL = ROOT/'experiments/archive/representation_and_objectives/data/relation_active_pair_funnel/active_relation_pair_pool.jsonl'
OUT = ROOT/'experiments/archive/representation_and_objectives/data/relation_pair_pool_semantic_audit'
NOTE = ROOT/'research/notes/representation_and_objectives/relation_pair_pool_semantic_audit.md'
PUNCT = "\"'“”‘’.,!?;:()[]{}<>«»‹›*_-=+/\\|`~"
WORD_RE = re.compile(r"\S+")
GENERIC_STOP = {
    'a','an','the','and','or','but','if','then','of','to','in','on','for','with','as','by','at','from','that','this','these','those',
    'is','are','was','were','be','been','being','am','do','does','did','have','has','had','can','could','will','would','shall','should',
    'may','might','must','not','no','yes','you','i','he','she','it','we','they','me','him','her','us','them','my','your','his','its','our','their',
    'one','two','three','four','five','six','seven','eight','nine','ten','first','last','new','old','good','bad','great','small','large','little',
    'big','many','much','more','most','less','least','other','same','own','very','really','actually','well','also','just','still','only','even',
    'up','down','out','off','over','under','into','about','after','before','during','than','so','because','though','although','while','when',
    'where','who','what','which','why','how','there','here','thing','things','people','person','man','woman','child','time','year','day','way',
    'work','world','life','part','place','house','city','school','state','water','table','give','take','make','get','go','come','see','look','know',
    'think','say','said','use','used','made','put','set','found','left','right','real','live','show','try','want','need','like'
}
ANTONYM_HINTS = [
    ('increase','decrease'),('increased','decreased'),('increases','decreases'),('higher','lower'),('high','low'),('more','less'),
    ('larger','smaller'),('large','small'),('greater','lesser'),('before','after'),('up','down'),('above','below'),('over','under'),
    ('inside','outside'),('in','out'),('open','closed'),('hot','cold'),('warm','cool'),('young','old'),('new','old'),('left','right'),
    ('start','end'),('started','ended'),('begin','end'),('first','last'),('early','late'),('earlier','later'),('positive','negative'),
    ('true','false'),('yes','no'),('on','off'),('present','absent'),('near','far'),('front','back')
]


def norm(s: Any) -> str:
    return str(s).lower().strip(PUNCT)


def toks(text: str) -> list[str]:
    return [norm(m.group(0)) for m in WORD_RE.finditer(text) if norm(m.group(0))]


def load_pairs() -> list[dict[str, Any]]:
    out=[]
    with PAIR_POOL.open('r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def levenshtein(a: str, b: str, max_cap: int = 50) -> int:
    a=a[:max_cap]; b=b[:max_cap]
    if len(a) < len(b): a,b=b,a
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(prev[j]+1, cur[-1]+1, prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]


def char_jaccard(a: str, b: str) -> float:
    A=set(a); B=set(b)
    return len(A&B)/len(A|B) if A or B else 0.0


def word_context(pair: dict[str,Any], side: str, radius: int = 6) -> list[str]:
    text=pair[f'text_{side}']; gid=int(pair[f'target_gid_{side}'])
    ts=toks(text)
    lo=max(0,gid-radius); hi=min(len(ts),gid+radius+1)
    return ts[lo:hi]


def maybe_antonym(a: str, b: str) -> bool:
    a=norm(a); b=norm(b)
    for x,y in ANTONYM_HINTS:
        if (a==x and b==y) or (a==y and b==x): return True
    return False


def qstats(vals: list[float]) -> dict[str,Any]:
    xs=sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs: return {'n':0}
    def q(p):
        if len(xs)==1: return xs[0]
        idx=p*(len(xs)-1); lo=int(math.floor(idx)); hi=int(math.ceil(idx))
        return xs[lo] if lo==hi else xs[lo]*(hi-idx)+xs[hi]*(idx-lo)
    return {'n':len(xs),'mean':statistics.fmean(xs),'median':q(0.5),'p05':q(0.05),'p95':q(0.95),'min':xs[0],'max':xs[-1]}


def analyze_pair(p: dict[str,Any]) -> dict[str,Any]:
    ta=norm(p['target_norm_a']); tb=norm(p['target_norm_b'])
    pa=norm(p.get('pivot_norm_a','')); pb=norm(p.get('pivot_norm_b',''))
    texta=toks(p['text_a']); textb=toks(p['text_b'])
    ctxa=[x for x in texta if x!=ta]; ctxb=[x for x in textb if x!=tb]
    ca=set(ctxa); cb=set(ctxb)
    cross_leak = int(tb in ca) + int(ta in cb)
    same_prefix3 = ta[:3] == tb[:3]
    same_suffix3 = ta[-3:] == tb[-3:] if len(ta)>=3 and len(tb)>=3 else False
    lev=levenshtein(ta,tb); mx=max(len(ta),len(tb),1)
    local_a=word_context(p,'a'); local_b=word_context(p,'b')
    local_j=len(set(local_a)&set(local_b))/len(set(local_a)|set(local_b)) if set(local_a)|set(local_b) else 0.0
    global_j=len(ca&cb)/len(ca|cb) if ca|cb else 0.0
    return {
        'pair_id':p['pair_id'],'category':p['category'],'match_level':p.get('match_level'),
        'target_a':ta,'target_b':tb,'pivot_a':pa,'pivot_b':pb,'source_a':p.get('source_a'),'source_b':p.get('source_b'),
        'target_class':p.get('target_class'),'target_token_len':p.get('target_token_len'),'target_freq_bin_a':p.get('target_freq_bin_a'),'target_freq_bin_b':p.get('target_freq_bin_b'),
        'a_generic':ta in GENERIC_STOP,'b_generic':tb in GENERIC_STOP,'either_generic':ta in GENERIC_STOP or tb in GENERIC_STOP,
        'both_generic':ta in GENERIC_STOP and tb in GENERIC_STOP,'maybe_antonym':maybe_antonym(ta,tb),'same_prefix3':same_prefix3,'same_suffix3':same_suffix3,
        'levenshtein':lev,'normalized_levenshtein':lev/mx,'char_jaccard':char_jaccard(ta,tb),'cross_context_target_leaks':cross_leak,
        'context_word_jaccard':global_j,'local_window_jaccard':local_j,'target_len_chars_a':len(ta),'target_len_chars_b':len(tb)
    }


def main():
    t0=time.time(); OUT.mkdir(parents=True,exist_ok=True)
    pairs=load_pairs(); rows=[analyze_pair(p) for p in pairs]
    bycat=defaultdict(list)
    for r in rows: bycat[r['category']].append(r)
    summary={'status':'RELATION_PAIR_POOL_SEMANTIC_AUDIT','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pair_pool':str(PAIR_POOL),'pairs':len(rows),'overall':{},'by_category':{},'examples':{}}
    bool_fields=['either_generic','both_generic','maybe_antonym','same_prefix3','same_suffix3']
    num_fields=['normalized_levenshtein','char_jaccard','context_word_jaccard','local_window_jaccard','cross_context_target_leaks']
    def pack(rs):
        d={'n':len(rs),'target_class_counts':dict(Counter(r['target_class'] for r in rs).most_common()),'source_pair_top10':dict(Counter(str(r['source_a'])+'||'+str(r['source_b']) for r in rs).most_common(10))}
        for f in bool_fields: d[f+'_rate']=sum(1 for r in rs if r[f])/len(rs) if rs else None
        for f in num_fields: d[f]=qstats([r[f] for r in rs])
        d['target_top20']=Counter([r['target_a'] for r in rs]+[r['target_b'] for r in rs]).most_common(20)
        d['pivot_top20']=Counter([r['pivot_a'] for r in rs]+[r['pivot_b'] for r in rs]).most_common(20)
        return d
    summary['overall']=pack(rows)
    for cat,rs in sorted(bycat.items()): summary['by_category'][cat]=pack(rs)
    # representative cases: generic, antonym-like, low lexical similarity, high local/context overlap.
    selectors={
        'generic_examples': lambda r: r['either_generic'],
        'antonym_hint_examples': lambda r: r['maybe_antonym'],
        'high_target_similarity_examples': lambda r: r['char_jaccard']>=0.6 or r['normalized_levenshtein']<=0.35,
        'high_local_overlap_examples': lambda r: r['local_window_jaccard']>=0.25,
        'low_context_overlap_examples': lambda r: r['context_word_jaccard']<=0.02,
    }
    for name,fn in selectors.items():
        xs=[r for r in rows if fn(r)]
        # stable deterministic order by pair_id for inspectable samples.
        summary['examples'][name]=xs[:30]
    json_path=OUT/'relation_pair_pool_semantic_audit.json'; json_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    csv_path=OUT/'relation_pair_pool_semantic_audit_rows.csv'
    with csv_path.open('w',encoding='utf-8',newline='') as f:
        keys=list(rows[0].keys()); w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    lines=['# research — active relation pair pool semantic audit','', 'CPU-only, model-free audit of whether the research active-pair reservoir is likely to be a hard conditional-alternative pool or mostly arbitrary sentence-specific target recovery.', '', f'Pairs: **{len(rows)}**', '', '## Overall indicators', '']
    o=summary['overall']
    lines.append(f"Generic target rate: either={o['either_generic_rate']:.3f}, both={o['both_generic_rate']:.3f}; antonym-hint rate={o['maybe_antonym_rate']:.4f}")
    lines.append(f"Target char similarity: {o['char_jaccard']}; normalized edit distance: {o['normalized_levenshtein']}")
    lines.append(f"Context word Jaccard: {o['context_word_jaccard']}; local target-window Jaccard: {o['local_window_jaccard']}")
    lines.append('')
    lines.append('## By family')
    lines.append('| family | n | either generic | antonym hint | mean target char J | mean context J | mean local-window J | top targets |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---|')
    for cat,d in summary['by_category'].items():
        tops=', '.join([f'{t}:{c}' for t,c in d['target_top20'][:6]])
        lines.append(f"| {cat} | {d['n']} | {d['either_generic_rate']:.3f} | {d['maybe_antonym_rate']:.4f} | {d['char_jaccard']['mean']:.3f} | {d['context_word_jaccard']['mean']:.3f} | {d['local_window_jaccard']['mean']:.3f} | {tops} |")
    lines += ['', 'Interpretation aid:', '- A useful training reservoir for the current problem should not merely show large own-context target recovery. It should contain cross-targets that are competitive enough for four-cell margins to differ across checkpoints in the same direction as EWoK/GlobalPIQA relation movement.', '- Low antonym/semantic-alternative hints, low context overlap, and generic high-frequency targets would support the suspicion that the reservoir is dominated by ordinary attested-coherence rather than the hard conditional-choice structure.', '', f'JSON: `{json_path}`', f'CSV: `{csv_path}`']
    NOTE.parent.mkdir(parents=True,exist_ok=True); NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'pairs':len(rows),'json':str(json_path),'note':str(NOTE),'runtime_sec':round(time.time()-t0,2)},indent=2),flush=True)

if __name__=='__main__':
    main()
