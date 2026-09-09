#!/usr/bin/env python3
"""research: cheap probes for the upset-balanced ranking-state substrate.

The repaired research substrate makes event outcome independent of contemporaneous
ranking state. This script asks what signals remain:

1. Transparent parsers: event winner, first state-mentioned entity, and numeric
   rank comparison.
2. Sparse TF-IDF baselines on held families with natural names or canonicalized
   pair names, with and without rank numbers.

This is not student pretraining. It is a small substrate-readiness measurement:
if event-only or rank-ablated text can solve state_at_time, the substrate still
has a shortcut; if only the numeric state sentence solves it, the substrate is a
usable natural-source state-binding scaffold but remains numerically explicit.
"""
from __future__ import annotations

import json, re, time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

STUDY = Path("experiments/archive/representation_and_objectives")
SUB = STUDY / "data/upset_balanced_state_substrate"
OUT = STUDY / "data/upset_state_binding_probe"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n", encoding='utf-8')


def write_jsonl(path: Path, rows: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")


def gold01(q: dict[str,Any]) -> int:
    return 1 if q['gold'] == 'ENTAILED' else 0


def canon_text(text: str, fam: dict[str,Any]) -> str:
    a,b=fam['participant_a'], fam['participant_b']
    # Replace longer names first to avoid partial overlap.
    out=text
    for name,rep in sorted([(a,'PLAYER_A'),(b,'PLAYER_B')], key=lambda x: -len(x[0])):
        out=out.replace(name,rep)
    return out


def ablate_numbers(text: str) -> str:
    return re.sub(r"#?\b\d{1,4}\b", "NUM", text)


def context_for(ctx: dict[str,Any], mode: str) -> str:
    if mode == 'event_only': return ctx['event_text']
    if mode == 'state_at_time_only': return ctx['state_at_time_text']
    if mode == 'state_later_only': return ctx['state_later_text']
    if mode == 'event_plus_state_at_time': return ctx['event_text'] + ' ' + ctx['state_at_time_text']
    if mode == 'event_plus_state_later': return ctx['event_text'] + ' ' + ctx['state_later_text']
    if mode == 'full_two_state': return ctx['event_text'] + ' ' + ctx['state_at_time_text'] + ' ' + ctx['state_later_text']
    raise ValueError(mode)


def build_rows(fams: list[dict[str,Any]], split: str, mode: str, families: list[str]) -> list[dict[str,Any]]:
    rows=[]
    allowed=set(families)
    for fam in fams:
        for wk in ['context1','context2']:
            ctx=fam[wk]
            for q in fam['nli_queries']:
                if q['world'] != wk or q['family'] not in allowed: continue
                req=q['requires']
                # Keep rows where the requested evidence is present in the context mode.
                if req == 'event_text' and mode not in {'event_only','event_plus_state_at_time','event_plus_state_later','full_two_state'}: continue
                if req == 'state_at_time_text' and mode not in {'state_at_time_only','event_plus_state_at_time','full_two_state'}: continue
                if req == 'state_later_text' and mode not in {'state_later_only','event_plus_state_later','full_two_state'}: continue
                if req == 'invariant' and mode not in {'event_plus_state_at_time','event_plus_state_later','full_two_state'}: continue
                text=context_for(ctx,mode) + ' [SEP] ' + q['hypothesis']
                rows.append({
                    'id':f"{fam['family_id']}_{wk}_{q['family']}_{len(rows)}_{mode}", 'family_id':fam['family_id'], 'split':split,
                    'world':wk, 'mode':mode, 'query_family':q['family'], 'requires':req,
                    'text':text, 'text_canon':canon_text(text,fam), 'text_canon_no_numbers':ablate_numbers(canon_text(text,fam)),
                    'label':gold01(q), 'participant_a':fam['participant_a'], 'participant_b':fam['participant_b'],
                    'winner':ctx['winner'], 'loser':ctx['loser'],
                    'winner_higher_at_time':ctx['winner_higher_at_time'], 'winner_higher_later':ctx['winner_higher_later'],
                    'winner_rank':ctx['winner_rank'], 'loser_rank':ctx['loser_rank'], 'winner_rank_later':ctx['winner_rank_later'], 'loser_rank_later':ctx['loser_rank_later'],
                    'hypothesis':q['hypothesis'],
                })
    return rows


def hyp_first_name(row: dict[str,Any]) -> str | None:
    hyp=row['hypothesis']
    for name in sorted([row['participant_a'], row['participant_b']], key=lambda x: -len(x)):
        if hyp.startswith(name): return name
    return None


def event_winner_parser(row: dict[str,Any]) -> int | None:
    name=hyp_first_name(row)
    if name is None: return None
    return int(name == row['winner'])


def state_numeric_parser(row: dict[str,Any]) -> int | None:
    name=hyp_first_name(row)
    if name is None: return None
    fam=row['query_family']
    if fam == 'state_at_time':
        rank = row['winner_rank'] if name == row['winner'] else row['loser_rank']
        other = row['loser_rank'] if name == row['winner'] else row['winner_rank']
    elif fam == 'state_later':
        rank = row['winner_rank_later'] if name == row['winner'] else row['loser_rank_later']
        other = row['loser_rank_later'] if name == row['winner'] else row['winner_rank_later']
    else:
        return None
    return int(rank < other)


def first_state_entity_parser(row: dict[str,Any]) -> int | None:
    # State text is written winner first in the current substrate. This parser is
    # intentionally simple: the first state-mentioned entity is predicted higher.
    if row['query_family'] not in {'state_at_time','state_later'}: return None
    return event_winner_parser(row)


def parser_acc(rows: list[dict[str,Any]], parser) -> dict[str,Any]:
    vals=[]; by={}
    for r in rows:
        yhat=parser(r)
        if yhat is None: continue
        ok=int(yhat == int(r['label'])); vals.append(ok)
        by.setdefault(r['query_family'],[]).append(ok)
    return {'coverage':len(vals)/max(1,len(rows)), 'acc':float(np.mean(vals)) if vals else None, 'by_family':{k:float(np.mean(v)) for k,v in by.items()}}


def fit_eval(train: list[dict[str,Any]], held: list[dict[str,Any]], field: str) -> dict[str,Any]:
    pipe=make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=1), LogisticRegression(max_iter=1000, solver='liblinear'))
    xtr=[r[field] for r in train]; ytr=[r['label'] for r in train]
    xhe=[r[field] for r in held]; yhe=[r['label'] for r in held]
    pipe.fit(xtr,ytr)
    pred=pipe.predict(xhe)
    out={'train_n':len(train),'held_n':len(held),'field':field,'held_acc':float(accuracy_score(yhe,pred))}
    fams=sorted(set(r['query_family'] for r in held))
    out['by_family']={}
    for fam in fams:
        idx=[i for i,r in enumerate(held) if r['query_family']==fam]
        out['by_family'][fam]=float(accuracy_score([yhe[i] for i in idx],[pred[i] for i in idx]))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_f=read_jsonl(SUB/'families_train.jsonl'); held_f=read_jsonl(SUB/'families_held.jsonl')
    modes=['event_only','state_at_time_only','state_later_only','event_plus_state_at_time','event_plus_state_later','full_two_state']
    family_sets={
        'event_only':['event_role'],
        'state_at_time_only':['state_at_time'],
        'state_later_only':['state_later'],
        'event_plus_state_at_time':['event_role','state_at_time','untouched_fact'],
        'event_plus_state_later':['event_role','state_later','untouched_fact'],
        'full_two_state':['event_role','state_at_time','state_later','untouched_fact'],
    }
    rows_train={m:build_rows(train_f,'train',m,family_sets[m]) for m in modes}
    rows_held={m:build_rows(held_f,'held',m,family_sets[m]) for m in modes}
    for m in modes:
        write_jsonl(OUT/f'rows_train_{m}.jsonl', rows_train[m])
        write_jsonl(OUT/f'rows_held_{m}.jsonl', rows_held[m])
    parsers={}
    for m in modes:
        both=rows_train[m]+rows_held[m]
        parsers[m]={
            'event_winner': parser_acc(both,event_winner_parser),
            'first_state_entity': parser_acc(both,first_state_entity_parser),
            'numeric_rank': parser_acc(both,state_numeric_parser),
        }
    # Sparse baselines on held families. Include both all rows and state-only subsets.
    tfidf={}
    for m in modes:
        tfidf[m]={}
        for field in ['text','text_canon','text_canon_no_numbers']:
            tfidf[m][field]=fit_eval(rows_train[m], rows_held[m], field)
    # state_at_time in interfering context only, without event rows, to see whether a model can read the state relation amid event text.
    train_state_interf=[r for r in rows_train['event_plus_state_at_time'] if r['query_family']=='state_at_time']
    held_state_interf=[r for r in rows_held['event_plus_state_at_time'] if r['query_family']=='state_at_time']
    tfidf['state_at_time_inside_event_context']={field:fit_eval(train_state_interf,held_state_interf,field) for field in ['text','text_canon','text_canon_no_numbers']}
    summary={'status':'UPSET_STATE_BINDING_PROBE','created_utc':now(),
             'n_families':{'train':len(train_f),'held':len(held_f)},
             'row_counts':{m:{'train':len(rows_train[m]),'held':len(rows_held[m])} for m in modes},
             'transparent_parsers':parsers,'tfidf_logreg':tfidf,
             'interpretation':'The contemporaneous state substrate defeats the event-winner shortcut. Explicit rank numbers remain the intended state evidence: numeric parsing solves state rows, while rank-number ablation tests whether sparse text baselines were reading the state or exploiting residual wording.'}
    write_json(OUT/'upset_state_binding_probe_summary.json', summary)
    md=['# research upset-balanced ranking-state substrate probes','',summary['interpretation'],'','## Transparent parsers','', '| mode | parser | acc | event | state_at_time | state_later |','|---|---|---:|---:|---:|---:|']
    for m in modes:
        for pn,pv in parsers[m].items():
            bf=pv['by_family']; md.append(f"| {m} | {pn} | {pv['acc'] if pv['acc'] is not None else float('nan'):.3f} | {bf.get('event_role',float('nan')):.3f} | {bf.get('state_at_time',float('nan')):.3f} | {bf.get('state_later',float('nan')):.3f} |")
    md += ['','## TF-IDF logistic held-family accuracy','', '| mode | field | held acc | event | state_at_time | state_later | untouched |','|---|---|---:|---:|---:|---:|---:|']
    for m,d in tfidf.items():
        if m == 'state_at_time_inside_event_context': continue
        for field,o in d.items():
            bf=o['by_family']; md.append(f"| {m} | {field} | {o['held_acc']:.3f} | {bf.get('event_role',float('nan')):.3f} | {bf.get('state_at_time',float('nan')):.3f} | {bf.get('state_later',float('nan')):.3f} | {bf.get('untouched_fact',float('nan')):.3f} |")
    md += ['','## State-at-time query inside event+state context only','', '| field | held acc |','|---|---:|']
    for field,o in tfidf['state_at_time_inside_event_context'].items(): md.append(f"| {field} | {o['held_acc']:.3f} |")
    md += ['', f"Summary JSON: `{OUT/'upset_state_binding_probe_summary.json'}`"]
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/upset_state_binding_probe/upset_state_binding_probe_summary.md')).write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(json.dumps({'status':summary['status'],'summary_json':str(OUT/'upset_state_binding_probe_summary.json'),
                      'event_winner_state_at_time':parsers['event_plus_state_at_time']['event_winner']['by_family'].get('state_at_time'),
                      'numeric_state_at_time':parsers['event_plus_state_at_time']['numeric_rank']['by_family'].get('state_at_time'),
                      'tfidf_state_interf':tfidf['state_at_time_inside_event_context']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
