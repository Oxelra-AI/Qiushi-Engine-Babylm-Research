#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib, re, statistics, collections

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
DATA = ROOT/'data/fineweb_relation_matched_1M'
ARMS = {
    'random_quality': DATA/'fineweb_random_quality_1000000w.jsonl',
    'relation_explicit': DATA/'fineweb_relation_explicit_1000000w.jsonl',
}
OUT_JSON = ROOT/'data/fineweb_experience_characterization.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/fineweb_experience_characterization.md')

SENT_RE = re.compile(r'(?<=[.!?])\s+')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3})\b")
LEAD_BAD = {'The','This','That','These','Those','There','When','Where','What','How','Why','Because','For','And','But','New','All','Most','Some','Many','First','After','Before','During','Then','Each','Every','Page','Pages'}
TRANSITION = set('''went moved stayed traveled travelled visited returned arrived left entered crossed followed met joined worked studied taught built made created opened closed carried held took gave brought found lost kept bought sold used broke repaired changed became turned killed saved helped led ruled served married sent received placed stored picked dropped raised lowered packed put came got took brought carried walked ran drove flew sailed climbed fell escaped fled deposed established succeeded inherited refused accepted'''.split())
LOCATION_PREP = set('''in at on inside outside near beside behind under over across through from to into onto around within toward towards between among by with without'''.split())
STATIC = set('''is was are were been being known called named born died located based situated includes contains consisted accounted described published wrote says said showed meant became'''.split())
POSSESSION = set('''has had have owns owned held carried took brought found gave kept bought sold received put placed stored packed lost'''.split())
STATE_WORDS = set('''open closed broken repaired full empty alive dead free trapped married deposed elected appointed popular unpopular infected located based prepared missing ready available unavailable'''.split())
NOISE_TERMS = ['amazon.com','purchase through','click this link','reload this page','all books','paperback','pages','privacy policy','subscribe','copyright','bestcase scenario']

def words(text: str):
    return [w.lower() for w in WORD_RE.findall(text)]

def sentences(text: str):
    ss = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    return ss if ss else [text]

def entities(sent: str):
    out=[]
    for m in ENTITY_RE.finditer(sent):
        e=' '.join(m.group(0).split())
        first=e.split()[0]
        if first in LEAD_BAD: continue
        if len(e) < 4: continue
        out.append(e.lower())
    return out

def sent_flags(sent: str):
    ws=set(words(sent))
    return {
        'transition': bool(ws & TRANSITION),
        'location': bool(ws & LOCATION_PREP),
        'static': bool(ws & STATIC),
        'possession': bool(ws & POSSESSION),
        'state_word': bool(ws & STATE_WORDS),
        'yearish': bool(re.search(r'\b(1[5-9][0-9]{2}|20[0-2][0-9])\b', sent)),
    }

def characterize_row(text: str):
    ss=sentences(text)
    ent_by_sent=[entities(s) for s in ss]
    flags=[sent_flags(s) for s in ss]
    all_ents=[e for xs in ent_by_sent for e in xs]
    counts=collections.Counter(all_ents)
    repeated=[e for e,c in counts.items() if c>=2]
    cross_repeat=[]
    for e in repeated:
        idx=[i for i,xs in enumerate(ent_by_sent) if e in xs]
        if len(set(idx))>=2:
            cross_repeat.append(e)
    transition_threads=0
    location_threads=0
    possession_threads=0
    state_threads=0
    static_threads=0
    for e in cross_repeat:
        idx=[i for i,xs in enumerate(ent_by_sent) if e in xs]
        # later occurrence must have an event/state cue; this approximates trackable state use.
        for j in idx[1:]:
            fl=flags[j]
            if fl['transition']:
                transition_threads += 1
                if fl['location']: location_threads += 1
                if fl['possession']: possession_threads += 1
                if fl['state_word']: state_threads += 1
                break
            if fl['static']:
                static_threads += 1
                break
    static_relation_sents=sum(1 for xs,fl in zip(ent_by_sent,flags) if xs and (fl['static'] or fl['yearish']))
    transition_sents=sum(1 for xs,fl in zip(ent_by_sent,flags) if xs and fl['transition'])
    loc_poss_sents=sum(1 for xs,fl in zip(ent_by_sent,flags) if xs and (fl['location'] or fl['possession']))
    lower=text.lower()
    noise=sum(1 for n in NOISE_TERMS if n in lower)
    bulletish=(text.count(' - ') + text.count('•') + text.count('|')) / max(1, len(ss))
    return {
        'words': len(text.split()),
        'sentences': len(ss),
        'unique_entities': len(counts),
        'entity_mentions': len(all_ents),
        'repeated_entities': len(repeated),
        'cross_sentence_repeated_entities': len(cross_repeat),
        'transition_threads': transition_threads,
        'location_threads': location_threads,
        'possession_threads': possession_threads,
        'state_threads': state_threads,
        'static_threads': static_threads,
        'static_relation_sents': static_relation_sents,
        'transition_sents': transition_sents,
        'loc_poss_sents': loc_poss_sents,
        'noise_hits': noise,
        'bulletish': bulletish,
        'has_cross_repeat': int(len(cross_repeat)>0),
        'has_transition_thread': int(transition_threads>0),
        'has_static_relation': int(static_relation_sents>0),
        'has_noise': int(noise>0 or bulletish>0.6),
    }

def summarize(rows):
    keys=list(rows[0].keys()) if rows else []
    out={'n_examples': len(rows)}
    for k in keys:
        vals=[r[k] for r in rows]
        out[k+'_mean']=float(statistics.mean(vals)) if vals else None
        if k.startswith('has_'):
            out[k+'_frac']=float(statistics.mean(vals)) if vals else None
    return out

def main():
    payload={'status':'FINEWEB_EXPERIENCE_CHARACTERIZATION','arms':{},'arm_delta_relation_minus_random':{}}
    examples={}
    for arm,path in ARMS.items():
        metrics=[]; sample_transition=[]; sample_static=[]; sample_noise=[]
        for line in path.open(encoding='utf-8'):
            obj=json.loads(line); m=characterize_row(obj['text']); metrics.append(m)
            if m['has_transition_thread'] and len(sample_transition)<5:
                sample_transition.append({'example_id':obj['example_id'],'metrics':m,'text':obj['text'][:700]})
            if m['has_static_relation'] and not m['has_transition_thread'] and len(sample_static)<5:
                sample_static.append({'example_id':obj['example_id'],'metrics':m,'text':obj['text'][:700]})
            if m['has_noise'] and len(sample_noise)<5:
                sample_noise.append({'example_id':obj['example_id'],'metrics':m,'text':obj['text'][:700]})
        payload['arms'][arm]={'summary': summarize(metrics), 'samples': {'transition_thread':sample_transition,'static_relation':sample_static,'noise':sample_noise}}
    r=payload['arms']['random_quality']['summary']; e=payload['arms']['relation_explicit']['summary']
    for k in sorted(set(r) & set(e)):
        if isinstance(r[k], (int,float)) and isinstance(e[k], (int,float)):
            payload['arm_delta_relation_minus_random'][k]=e[k]-r[k]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research FineWeb experience characterization','',f'Evidence JSON: `{OUT_JSON}`','', 'This characterizes the research/279 1M FineWeb JSONL examples as training examples, focusing on whether the relation filter created repeated-entity state-transition experience or mostly static/factual relation density.','', '| metric | random_quality | relation_explicit | relation-random |','|---|---:|---:|---:|']
    show=['unique_entities_mean','entity_mentions_mean','repeated_entities_mean','cross_sentence_repeated_entities_mean','has_cross_repeat_frac','transition_threads_mean','has_transition_thread_frac','location_threads_mean','possession_threads_mean','static_relation_sents_mean','has_static_relation_frac','has_noise_frac','bulletish_mean']
    for k in show:
        rv=r.get(k); ev=e.get(k); dv=payload['arm_delta_relation_minus_random'].get(k)
        lines.append(f'| {k} | {rv:.4f} | {ev:.4f} | {dv:+.4f} |')
    lines += ['', '## Interpretation', '', '- research showed relation-explicit FineWeb improved EWoK fast by +3.27 but did not improve Entity fast. The characterization below tests whether the selected examples actually contain repeated-entity transition threads rather than static facts.', '- If relation-explicit only increases static relation sentences and not transition threads, the next corpus arm should require repeated entities plus state-changing actions across adjacent sentences.', '', '## Example snippets', '']
    for arm in ['random_quality','relation_explicit']:
        lines += [f'### {arm}', '']
        for typ,label in [('transition_thread','transition-thread candidates'),('static_relation','static-relation examples'),('noise','source-noise examples')]:
            lines.append(f'#### {label}')
            for ex in payload['arms'][arm]['samples'][typ][:3]:
                txt=' '.join(ex['text'].split())
                lines.append(f"- ex {ex['example_id']} metrics={ex['metrics']} text={txt[:450]}")
            lines.append('')
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'out_json':str(OUT_JSON),'out_note':str(OUT_NOTE),'deltas':{k:payload['arm_delta_relation_minus_random'].get(k) for k in show}}, indent=2))
if __name__=='__main__': main()
