#!/usr/bin/env python3
"""research: mine a general consequence/contrast substrate from existing allowed reservoirs.

This CPU-only script prepares a future research asset after the GlobalPIQA/EWoK anatomy.
It does NOT read official evaluation item text and should not be treated as evaluation-item
shaping.  It applies broad, predeclared physical/temporal/spatial/affordance/causal lexicons
to existing permitted training-data reservoirs and summarizes whether enough high-precision
general consequence material exists for a later small factorial probe.
"""
from __future__ import annotations
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple

ROOT = Path('experiments/archive/representation_and_objectives')
OUT = ROOT/'data/general_consequence_substrate'
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/general_consequence_substrate_miner.md')

INPUTS = {
    'compact_experience_aligned_10m_rows': {
        'path': Path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl'),
        'text_fields': ['text'],
        'kind': 'current_lineage_10M_pool',
    },
    'fw_frozen_sources_38167': {
        'path': ROOT/'data/fw_mechanism_source_selection/fw_mechanism_frozen_sources.jsonl',
        'text_fields': ['text'],
        'kind': 'fineweb_candidate_source_reservoir',
    },
    'fw_compact_pair_sources': {
        'path': ROOT/'data/fw_full_preservation/full26k_usable_pairs_for_materializer.jsonl',
        'text_fields': ['source_text'],
        'kind': 'selected_fineweb_common_sources',
    },
    'fw_compact_rewrites': {
        'path': ROOT/'data/fw_full_preservation/full26k_usable_pairs_for_materializer.jsonl',
        'text_fields': ['rewrite_text'],
        'kind': 'selected_qwen35_compact_rewrites',
    },
    'fw_breadth_whole_sentence_companions': {
        'path': ROOT/'data/fw_source_breadth_wholesentence_arm/source_breadth_wholesentence_companion_sources.jsonl',
        'text_fields': ['text'],
        'kind': 'selected_fineweb_breadth_companions',
    },
}

WORD_RE = re.compile(r"\b\w+(?:['’-]\w+)?\b", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘\[])|\n+")

LEX = {
    'causal_conditional_marker': [
        r"\bif\b", r"\bwhen\b", r"\bwhenever\b", r"\bbecause\b", r"\bsince\b", r"\bdue to\b",
        r"\bas a result\b", r"\btherefore\b", r"\bso that\b", r"\bin order to\b", r"\bleads? to\b",
        r"\bcaus(?:e|es|ed|ing)\b", r"\bresult(?:s|ed|ing)?\b", r"\bprevent(?:s|ed|ing)?\b",
        r"\ballow(?:s|ed|ing)?\b", r"\bmake(?:s| made)?\b", r"\bmeans?\b",
    ],
    'physical_state_action': [
        r"\bpush(?:ed|es|ing)?\b", r"\bpull(?:ed|s|ing)?\b", r"\bdrop(?:ped|s|ping)?\b", r"\bfall(?:s|ing|en)?\b",
        r"\bbounce(?:s|d|ing)?\b", r"\bbreak(?:s|ing)?\b", r"\bbroken\b", r"\bspill(?:s|ed|ing)?\b",
        r"\broll(?:s|ed|ing)?\b", r"\bfloat(?:s|ed|ing)?\b", r"\bsink(?:s|ing)?\b", r"\bskid(?:s|ded|ding)?\b",
        r"\bslide(?:s|d|ing)?\b", r"\btwist(?:s|ed|ing)?\b", r"\bturn(?:s|ed|ing)?\b", r"\bmove(?:s|d|ing)?\b",
        r"\bopen(?:s|ed|ing)?\b", r"\bclose(?:s|d|ing)?\b", r"\bheat(?:s|ed|ing)?\b", r"\bcool(?:s|ed|ing)?\b",
        r"\bfreeze(?:s|ing)?\b", r"\bmelt(?:s|ed|ing)?\b", r"\bdissolve(?:s|d|ing)?\b", r"\bshatter(?:s|ed|ing)?\b",
    ],
    'material_object_property': [
        r"\bwater\b", r"\bair\b", r"\bglass\b", r"\bmetal\b", r"\bplastic\b", r"\bwood(?:en)?\b", r"\bpaper\b",
        r"\bfabric\b", r"\bcloth\b", r"\bceramic\b", r"\brubber\b", r"\bliquid\b", r"\bsolid\b", r"\bgas\b",
        r"\bhot\b", r"\bcold\b", r"\bwet\b", r"\bdry\b", r"\bsoft\b", r"\bhard\b", r"\bheavy\b", r"\blight\b",
        r"\bfull\b", r"\bempty\b", r"\bsealed\b", r"\btransparent\b", r"\bvisible\b", r"\bdented\b",
    ],
    'spatial_direction': [
        r"\bleft\b", r"\bright\b", r"\bup\b", r"\bdown\b", r"\babove\b", r"\bbelow\b", r"\bunder\b", r"\bover\b",
        r"\binside\b", r"\boutside\b", r"\bfront\b", r"\bback\b", r"\bnorth\b", r"\bsouth\b", r"\beast\b", r"\bwest\b",
        r"\bthrough\b", r"\bacross\b", r"\baround\b", r"\bbehind\b", r"\btowards?\b", r"\baway from\b", r"\bangle\b",
    ],
    'time_count_update': [
        r"\byear(?:s)?\b", r"\bmonth(?:s)?\b", r"\bweek(?:s)?\b", r"\bday(?:s)?\b", r"\bhour(?:s)?\b", r"\bminute(?:s)?\b",
        r"\bbefore\b", r"\bafter\b", r"\bnext\b", r"\blast\b", r"\bearlier\b", r"\blater\b", r"\bevery\b", r"\bmore\b", r"\bfewer\b",
        r"\bless\b", r"\badd(?:s|ed|ing)?\b", r"\bsubtract(?:s|ed|ing)?\b", r"\bdivide(?:s|d|ing)?\b", r"\b\d{1,4}\b",
    ],
    'affordance_procedure': [
        r"\bhow to\b", r"\buse(?:d|s|ing)?\b", r"\btool(?:s)?\b", r"\butensil(?:s)?\b", r"\bwear(?:s|ing)?\b",
        r"\bhold(?:s|ing)?\b", r"\bstore(?:s|d|ing)?\b", r"\bcook(?:s|ed|ing)?\b", r"\bcut(?:s|ting)?\b", r"\bwrite(?:s|ing)?\b",
        r"\bclean(?:s|ed|ing)?\b", r"\bserve(?:s|d|ing)?\b", r"\bmake sure\b", r"\bbe careful\b", r"\bsafe(?:ly|ty)?\b", r"\bbest\b",
    ],
}

CONTRAST_PAIRS = [
    ('more','less'), ('more','fewer'), ('full','empty'), ('hot','cold'), ('wet','dry'), ('heavy','light'), ('hard','soft'),
    ('push','pull'), ('open','close'), ('up','down'), ('above','below'), ('inside','outside'), ('front','back'),
    ('left','right'), ('north','south'), ('east','west'), ('before','after'), ('first','last'), ('increase','decrease'),
    ('float','sink'), ('break','bend'), ('visible','hidden'), ('transparent','opaque'), ('fast','slow'), ('near','far'),
]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_sentences(text: str) -> List[str]:
    parts=[]
    for s in SENT_SPLIT_RE.split(text):
        s=s.strip()
        if not s: continue
        # long transcript rows are not great as whole passages; also split on speaker markers.
        sub=re.split(r"\s+(?=\*[A-Z]{2,4}:)|\s+(?=-\s+[A-Z])", s)
        for x in sub:
            x=x.strip()
            if x: parts.append(x)
    return parts


def hits(text: str) -> Dict[str,int]:
    out={}
    for cat,pats in LEX.items():
        n=sum(len(re.findall(p,text,re.I)) for p in pats)
        out[cat]=n
    lo=' '+re.sub(r"[^a-z0-9]+", ' ', text.lower())+' '
    cp=[]
    for a,b in CONTRAST_PAIRS:
        if f' {a} ' in lo and f' {b} ' in lo:
            cp.append(f'{a}/{b}')
    out['contrast_pair_count']=len(cp)
    return out, cp


def classify(text: str) -> Tuple[List[str], int, List[str]]:
    h, cp = hits(text)
    cats=[k for k,v in h.items() if k!='contrast_pair_count' and v>0]
    # high precision categories for route design
    routes=[]
    if h['causal_conditional_marker'] and (h['physical_state_action'] or h['material_object_property']):
        routes.append('physical_causal_consequence')
    if h['causal_conditional_marker'] and h['spatial_direction']:
        routes.append('spatial_causal_relation')
    if h['causal_conditional_marker'] and h['time_count_update']:
        routes.append('temporal_quant_update')
    if h['affordance_procedure'] and (h['physical_state_action'] or h['material_object_property'] or h['spatial_direction']):
        routes.append('affordance_action_consequence')
    if cp:
        routes.append('explicit_contrast_pair')
    if h['time_count_update'] >= 2 and (h['causal_conditional_marker'] or cp):
        routes.append('algorithmic_time_count_candidate')
    if not routes and cats:
        routes.append('broad_related_only')
    score=(3*len([r for r in routes if r!='broad_related_only']) + sum(min(v,3) for k,v in h.items() if k!='contrast_pair_count') + 2*len(cp))
    return routes, score, cp


def iter_records(label: str, spec: Dict[str,Any]):
    p=Path(spec['path'])
    if not p.exists():
        return
    for i,line in enumerate(p.open()):
        if not line.strip(): continue
        obj=json.loads(line)
        for field in spec['text_fields']:
            text=str(obj.get(field,'')).strip()
            if not text: continue
            for si,sent in enumerate(split_sentences(text)):
                wc=word_count(sent)
                if wc < 6 or wc > 80: continue
                yield {
                    'source_label': label,
                    'source_kind': spec['kind'],
                    'row_index': i,
                    'sentence_index': si,
                    'field': field,
                    'text': sent,
                    'words': wc,
                    'doc_id': obj.get('doc_id'),
                    'norm_hash': obj.get('norm_hash'),
                    'domains': obj.get('domains'),
                    'origin_source': obj.get('source') or obj.get('pool') or obj.get('source_kind'),
                }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0=time.time()
    candidate_rows=[]
    summaries={}
    for label,spec in INPUTS.items():
        n_sent=0; total_words=0; route_counts=Counter(); route_words=Counter(); lex_counts=Counter(); contrast_counts=Counter(); top=[]
        for rec in iter_records(label,spec):
            n_sent+=1; total_words+=rec['words']
            route,score,cp=classify(rec['text'])
            h,_=hits(rec['text'])
            for k,v in h.items():
                if k!='contrast_pair_count': lex_counts[k]+=v
            for c in cp: contrast_counts[c]+=1
            if route:
                for r in route:
                    route_counts[r]+=1; route_words[r]+=rec['words']
                if score>=8 and 'broad_related_only' not in route:
                    out=dict(rec)
                    out.update({'routes':route,'score':score,'contrast_pairs':cp})
                    candidate_rows.append(out)
                    top.append(out)
        top=sorted(top,key=lambda r:(-r['score'], r['source_label'], r['row_index'], r['sentence_index']))[:50]
        summaries[label]={
            'input_path': str(spec['path']),
            'source_kind': spec['kind'],
            'sentences_scanned': n_sent,
            'sentence_words_scanned': total_words,
            'route_counts': dict(route_counts),
            'route_word_counts': dict(route_words),
            'route_sentence_fractions': {k: route_counts[k]/n_sent for k in route_counts} if n_sent else {},
            'route_word_fractions': {k: route_words[k]/total_words for k in route_words} if total_words else {},
            'lexical_hits': dict(lex_counts),
            'lexical_hits_per_10k_words': {k: lex_counts[k]/total_words*10000 for k in lex_counts} if total_words else {},
            'contrast_pair_counts_top': contrast_counts.most_common(30),
            'top_examples': top,
        }
    # Deduplicate candidate rows by normalized text.
    seen=set(); dedup=[]
    for r in sorted(candidate_rows,key=lambda x:(-x['score'], x['source_label'], x['row_index'], x['sentence_index'])):
        key=re.sub(r"\s+",' ',r['text'].strip().lower())
        if key in seen: continue
        seen.add(key); dedup.append(r)
    total_candidate_words=sum(r['words'] for r in dedup)
    # Keep a practical full list; it is a research substrate, not a final corpus.
    with (OUT/'general_consequence_candidates.jsonl').open('w') as f:
        for r in dedup:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary_rows=[]
    for label,s in summaries.items():
        all_routes=sorted(set(s['route_counts']) | {'physical_causal_consequence','spatial_causal_relation','temporal_quant_update','affordance_action_consequence','explicit_contrast_pair','algorithmic_time_count_candidate'})
        for route in all_routes:
            summary_rows.append({
                'source_label': label,
                'route': route,
                'sentences': s['route_counts'].get(route,0),
                'words': s['route_word_counts'].get(route,0),
                'sentence_fraction': s['route_sentence_fractions'].get(route,0.0),
                'word_fraction': s['route_word_fractions'].get(route,0.0),
                'sentences_scanned': s['sentences_scanned'],
                'words_scanned': s['sentence_words_scanned'],
            })
    with (OUT/'general_consequence_substrate_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    payload={
        'status':'GENERAL_CONSEQUENCE_SUBSTRATE_MINED',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'boundary':'Broad lexicon mining of existing allowed reservoirs only; no official evaluation item text used.',
        'inputs': {k:str(v['path']) for k,v in INPUTS.items()},
        'summaries': summaries,
        'candidate_count_dedup': len(dedup),
        'candidate_words_dedup': total_candidate_words,
        'candidate_jsonl': str(OUT/'general_consequence_candidates.jsonl'),
        'summary_csv': str(OUT/'general_consequence_substrate_summary.csv'),
        'scientific_reading': {
            'if_large_reservoir':'A future low-cost factorial can draw 50k-200k words from this reservoir to test procedural/consequence substrate without building another huge teacher pipeline.',
            'if_small_reservoir':'If high-precision physical/temporal/spatial/affordance reservoir is too small, future work must either generate general programmatic curricula under the word budget or shift to objectives on existing text rather than corpus replacement.',
            'relationship_to_running_fw_pair':'This miner should not interrupt the running compact-vs-breadth experiment. It prepares the next route only after official vectors/margins show whether compact restatement or breadth improves hard GlobalPIQA/EWoK rows.',
        },
        'elapsed_sec': round(time.time()-t0,2),
    }
    (OUT/'general_consequence_substrate_miner.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
    lines=[]
    lines.append('# research — General consequence/contrast substrate miner')
    lines.append('')
    lines.append('## Purpose')
    lines.append('')
    lines.append('The GlobalPIQA margin reader shows the lineage usually puts the correct four-choice physical/temporal/spatial/affordance answer at rank 3 or 4 on the hardest rows. This CPU-only miner asks whether existing allowed reservoirs contain enough broad consequence/contrast material for a later strict-compliant low-cost probe. It does not read official evaluation items and does not construct a final training corpus.')
    lines.append('')
    lines.append('## Reservoir summary')
    lines.append('')
    lines.append('| reservoir | scanned sentences | scanned words | physical-causal words | spatial-causal words | temporal-quant words | affordance-consequence words | explicit-contrast words |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for label,s in summaries.items():
        rc=s['route_word_counts']
        lines.append(f"| `{label}` | {s['sentences_scanned']} | {s['sentence_words_scanned']} | {rc.get('physical_causal_consequence',0)} | {rc.get('spatial_causal_relation',0)} | {rc.get('temporal_quant_update',0)} | {rc.get('affordance_action_consequence',0)} | {rc.get('explicit_contrast_pair',0)} |")
    lines.append('')
    lines.append(f"Deduplicated high-scoring candidate sentences: {len(dedup)} totaling {total_candidate_words} words. This is a candidate reservoir for a later probe, not a selected training arm.")
    lines.append('')
    lines.append('## Scientific use')
    lines.append('')
    lines.append('- If the ongoing compact-vs-breadth official vectors show GlobalPIQA_parallel or EWoK movement, reuse the margin reader and this reservoir summary to interpret whether the movement came from dense same-proposition restatement or added consequence substrate.')
    lines.append('- If the running FW arms do not move the hard rows, the next efficient route is a small factorial probe that swaps a controlled 50k–200k word slice from this general reservoir and/or adds a corpus-derived paired contrastive objective. It should be tested from shared checkpoints before any 100M commitment.')
    lines.append('- Because many GlobalPIQA_parallel hard rows are spatial/direction and time/counting, a future route should keep physical, spatial, temporal/quantitative, and affordance channels separate rather than calling them one undifferentiated common-sense bucket.')
    lines.append('')
    lines.append('## Files')
    lines.append('')
    lines.append(f"- JSON: `{OUT/'general_consequence_substrate_miner.json'}`")
    lines.append(f"- summary CSV: `{OUT/'general_consequence_substrate_summary.csv'}`")
    lines.append(f"- candidate JSONL: `{OUT/'general_consequence_candidates.jsonl'}`")
    NOTE.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':payload['status'],'json':str(OUT/'general_consequence_substrate_miner.json'),'csv':str(OUT/'general_consequence_substrate_summary.csv'),'candidates':str(OUT/'general_consequence_candidates.jsonl'),'candidate_count':len(dedup),'candidate_words':total_candidate_words,'note':str(NOTE)},indent=2))

if __name__ == '__main__':
    main()
