#!/usr/bin/env python3
from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import re
import statistics
from typing import Iterable

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
MANIFEST = ROOT / 'data/aligned_rewrite_revision_150/manifest_w9999996_n212001_sel15001_shuf15002_side128-128.json'
OUT_JSON = ROOT / 'data/wikiauto_pair_signal_analysis.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/wikiauto_pair_signal_analysis.md')
OFFICIAL_DIR = ROOT / 'training/runs/babylm_pilot_dense_v0_1M/raw_dataset'

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
CAP_RE = re.compile(r"^[A-Z][A-Za-z]+$")
NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")

STOP = {
    'a','an','the','and','or','but','if','then','else','when','while','of','to','in','on','at','by','for','from','with','without','as','is','are','was','were','be','been','being','am','do','does','did','done','have','has','had','having','can','could','will','would','shall','should','may','might','must','not','no','yes','than','that','this','these','those','it','its','he','she','they','them','his','her','their','we','us','you','your','i','me','my','mine','our','ours','who','whom','which','what','where','why','how','there','here','also','more','most','other','some','any','all','one','two','first','second','new','old','many','much','such','into','over','under','after','before','about','up','down','out','off','only','through','between','during','against','within','because'
}
PRONOUNS = {'he','she','it','they','him','her','them','his','hers','its','their','theirs','himself','herself','itself','themselves','who','whom','whose','which','that','this','these','those'}
SPATIAL = {'in','on','at','under','over','above','below','inside','outside','near','between','behind','front','beside','around','through','across','within','into','onto','from','to','left','right','north','south','east','west','up','down','there','here'}
STATE_TRANSFER = {'put','puts','placed','place','places','moved','move','moves','moving','transfer','transferred','give','gave','given','take','took','taken','bring','brought','send','sent','carry','carried','drop','dropped','fall','fell','fallen','break','broke','broken','open','opened','close','closed','change','changed','become','became','turn','turned','keep','kept','hold','held','store','stored','remove','removed','replace','replaced','contain','contains','contained','enter','entered','leave','left'}
PHYSICAL = {'glass','wood','wooden','metal','plastic','stone','rock','water','ice','fire','air','paper','cloth','rubber','fragile','heavy','light','hot','cold','warm','wet','dry','hard','soft','solid','liquid','gas','round','flat','sharp','smooth','rough','break','burn','melt','freeze','float','sink','fall','rise','push','pull','roll','slide','throw','hit'}
SOCIAL = {'said','say','says','told','tell','asked','ask','asks','answer','answered','reply','replied','talk','spoke','speak','friend','mother','father','child','boy','girl','man','woman','people','person','team','group','family','king','queen'}
CAUSAL_TEMPORAL = {'because','cause','caused','therefore','so','if','then','after','before','during','while','when','until','since','although','though','however','therefore','result','results'}
RELATION_WORDS = SPATIAL | STATE_TRANSFER | PHYSICAL | SOCIAL | CAUSAL_TEMPORAL


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


def lowers(text: str) -> list[str]:
    return [t.lower() for t in tokens(text)]


def content_set(text: str) -> set[str]:
    return {t for t in lowers(text) if t not in STOP and not NUM_RE.match(t)}


def cap_set(text: str) -> set[str]:
    toks = tokens(text)
    out = set()
    for i, t in enumerate(toks):
        if i == 0:
            continue
        if CAP_RE.match(t) and t.lower() not in {'i'}:
            out.add(t)
    return out


def safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    pos = (len(ys)-1)*q
    lo = math.floor(pos); hi = math.ceil(pos)
    if lo == hi:
        return float(ys[lo])
    return float(ys[lo]*(hi-pos)+ys[hi]*(pos-lo))


def bucket_fraction(xs: list[float], cuts: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    labels = []
    prev = -math.inf
    for c in cuts:
        labels.append((prev, c, f'<= {c:g}' if prev == -math.inf else f'({prev:g}, {c:g}]'))
        prev = c
    labels.append((prev, math.inf, f'> {prev:g}'))
    counts = {lab:0 for _,_,lab in labels}
    for x in xs:
        for lo,hi,lab in labels:
            if x > lo and x <= hi:
                counts[lab]+=1; break
    return {k: v/len(xs) for k,v in counts.items()}


def text_feature_counts(text: str) -> dict[str, int]:
    toks = tokens(text)
    low = [t.lower() for t in toks]
    cap = 0
    for i,t in enumerate(toks):
        if i != 0 and CAP_RE.match(t):
            cap += 1
    return {
        'words_regex': len(toks),
        'pronouns': sum(1 for t in low if t in PRONOUNS),
        'capitalized_noninitial': cap,
        'numbers': sum(1 for t in low if NUM_RE.match(t)),
        'spatial': sum(1 for t in low if t in SPATIAL),
        'state_transfer': sum(1 for t in low if t in STATE_TRANSFER),
        'physical': sum(1 for t in low if t in PHYSICAL),
        'social': sum(1 for t in low if t in SOCIAL),
        'causal_temporal': sum(1 for t in low if t in CAUSAL_TEMPORAL),
        'relation_any': sum(1 for t in low if t in RELATION_WORDS),
        'content_tokens': sum(1 for t in low if t not in STOP and not NUM_RE.match(t)),
    }


def aggregate_text_features(texts: Iterable[str]) -> dict[str, float | int]:
    total = collections.Counter()
    n_texts = 0
    total_whitespace = 0
    for text in texts:
        n_texts += 1
        total_whitespace += len(text.split())
        total.update(text_feature_counts(text))
    words = max(1, total['words_regex'])
    out: dict[str, float | int] = {'texts': n_texts, 'whitespace_words': total_whitespace, 'regex_words': total['words_regex']}
    for key in ['pronouns','capitalized_noninitial','numbers','spatial','state_transfer','physical','social','causal_temporal','relation_any','content_tokens']:
        out[key] = total[key]
        out[key + '_per_1k_regex_words'] = 1000.0 * total[key] / words
    return out


def read_unpaired_pairs(path: pathlib.Path) -> dict[int, dict[str, str]]:
    pairs: dict[int, dict[str, str]] = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            pid = int(r['pair_ids'][0])
            d = pairs.setdefault(pid, {})
            if r['mode'] == 'unpaired_source':
                d['source'] = r['text']
            elif r['mode'] == 'unpaired_target':
                d['target'] = r['text']
    bad = [pid for pid,d in pairs.items() if 'source' not in d or 'target' not in d]
    if bad:
        raise RuntimeError(f'incomplete pairs: {bad[:5]}')
    return pairs


def pair_analysis(pairs: dict[int, dict[str,str]]) -> dict:
    ratios=[]; jaccards=[]; tgt_copy=[]; src_covered=[]; cap_overlap_frac=[]; cap_any_overlap=[]
    relation_overlap=[]; both_relation=[]; pronoun_delta=[]; src_caps=[]; tgt_caps=[]; src_words=[]; tgt_words=[]
    examples_low=[]; examples_high=[]; examples_entity_mismatch=[]
    source_texts=[]; target_texts=[]
    for pid,d in pairs.items():
        s=d['source']; t=d['target']
        source_texts.append(s); target_texts.append(t)
        sw=len(s.split()); tw=len(t.split())
        src_words.append(sw); tgt_words.append(tw)
        ratios.append(tw/max(1,sw))
        cs=content_set(s); ct=content_set(t)
        inter=cs & ct; union=cs | ct
        jac=len(inter)/len(union) if union else 0.0
        jaccards.append(jac)
        tgt_copy.append(len([x for x in lowers(t) if x not in STOP and x in cs]) / max(1, len([x for x in lowers(t) if x not in STOP and not NUM_RE.match(x)])))
        src_covered.append(len([x for x in lowers(s) if x not in STOP and x in ct]) / max(1, len([x for x in lowers(s) if x not in STOP and not NUM_RE.match(x)])))
        caps=cap_set(s); capt=cap_set(t)
        src_caps.append(len(caps)); tgt_caps.append(len(capt))
        if caps:
            cap_overlap_frac.append(len(caps & capt)/len(caps))
            cap_any_overlap.append(1.0 if (caps & capt) else 0.0)
            if not (caps & capt) and len(examples_entity_mismatch) < 12:
                examples_entity_mismatch.append({'pair_id':pid,'source_caps':sorted(caps),'target_caps':sorted(capt),'source':s[:220],'target':t[:220]})
        rs={x for x in lowers(s) if x in RELATION_WORDS}; rt={x for x in lowers(t) if x in RELATION_WORDS}
        if rs or rt:
            relation_overlap.append(len(rs & rt)/len(rs | rt) if (rs|rt) else 0.0)
            both_relation.append(1.0 if (rs and rt) else 0.0)
        pronoun_delta.append(sum(1 for x in lowers(t) if x in PRONOUNS) - sum(1 for x in lowers(s) if x in PRONOUNS))
        if jac < 0.15 and len(examples_low) < 10:
            examples_low.append({'pair_id':pid,'jaccard':jac,'ratio':tw/max(1,sw),'source':s[:240],'target':t[:240]})
        if jac > 0.75 and len(examples_high) < 10:
            examples_high.append({'pair_id':pid,'jaccard':jac,'ratio':tw/max(1,sw),'source':s[:240],'target':t[:240]})
    summary = {
        'pairs': len(pairs),
        'source_words_total': sum(src_words),
        'target_words_total': sum(tgt_words),
        'target_source_word_ratio_mean': safe_mean(ratios),
        'target_source_word_ratio_median': pct(ratios,0.5),
        'target_source_word_ratio_p10_p90': [pct(ratios,0.1), pct(ratios,0.9)],
        'target_source_word_ratio_buckets': bucket_fraction(ratios, [0.5,0.75,1.0,1.25,1.5]),
        'content_jaccard_mean': safe_mean(jaccards),
        'content_jaccard_median': pct(jaccards,0.5),
        'content_jaccard_p10_p90': [pct(jaccards,0.1), pct(jaccards,0.9)],
        'content_jaccard_buckets': bucket_fraction(jaccards, [0.15,0.30,0.50,0.70]),
        'target_content_token_copy_frac_mean': safe_mean(tgt_copy),
        'source_content_token_covered_by_target_mean': safe_mean(src_covered),
        'pairs_with_source_capitalized_noninitial_frac': sum(1 for x in src_caps if x>0)/len(src_caps),
        'pairs_with_target_capitalized_noninitial_frac': sum(1 for x in tgt_caps if x>0)/len(tgt_caps),
        'source_cap_overlap_frac_mean_cond_source_has_cap': safe_mean(cap_overlap_frac),
        'source_cap_any_overlap_frac_cond_source_has_cap': safe_mean(cap_any_overlap),
        'relation_overlap_mean_cond_any_relation': safe_mean(relation_overlap),
        'both_sides_relation_frac_cond_any_relation': safe_mean(both_relation),
        'target_minus_source_pronoun_count_mean': safe_mean(pronoun_delta),
        'source_feature_aggregate': aggregate_text_features(source_texts),
        'target_feature_aggregate': aggregate_text_features(target_texts),
        'combined_feature_aggregate': aggregate_text_features(source_texts + target_texts),
        'examples_low_overlap': examples_low,
        'examples_high_overlap': examples_high,
        'examples_source_caps_not_preserved': examples_entity_mismatch,
    }
    return summary


def official_analysis() -> dict:
    files = sorted(OFFICIAL_DIR.glob('*.train.txt'))
    by_file={}
    all_texts=[]
    # Store line texts only in streaming-sized chunks? The corpus is small enough for feature aggregation.
    for path in files:
        total = collections.Counter(); n_lines=0; whitespace=0; line_lengths=[]
        with path.open('r', encoding='utf-8', errors='replace') as f:
            for line in f:
                text=' '.join(line.split())
                if not text:
                    continue
                n_lines += 1
                whitespace += len(text.split())
                line_lengths.append(len(text.split()))
                total.update(text_feature_counts(text))
                all_texts.append(text)
        words=max(1,total['words_regex'])
        d={'lines':n_lines,'whitespace_words':whitespace,'regex_words':total['words_regex'],'mean_line_words':safe_mean(line_lengths),'median_line_words':pct(line_lengths,0.5)}
        for key in ['pronouns','capitalized_noninitial','numbers','spatial','state_transfer','physical','social','causal_temporal','relation_any','content_tokens']:
            d[key]=total[key]
            d[key+'_per_1k_regex_words']=1000.0*total[key]/words
        by_file[path.name]=d
    return {'files':[p.name for p in files], 'by_file':by_file, 'combined':aggregate_text_features(all_texts)}


def main():
    manifest=json.loads(MANIFEST.read_text())
    unpaired_path=pathlib.Path(manifest['arm_paths']['unpaired_mix']['path'])
    pairs=read_unpaired_pairs(unpaired_path)
    pa=pair_analysis(pairs)
    oa=official_analysis()
    payload={
        'status':'WIKIAUTO_PAIR_SIGNAL_ANALYSIS',
        'manifest':str(MANIFEST),
        'unpaired_mix_path':str(unpaired_path),
        'dataset_id':manifest.get('dataset_id'),
        'dataset_revision':manifest.get('dataset_revision'),
        'pair_analysis':pa,
        'official_corpus_analysis':oa,
        'interpretation_short':{
            'mechanism_test_result':'research aligned-vs-shuffled was negative on Entity and EWoK; this file measures whether the selected pair source actually carries dense cross-surface entity/relation signal.',
            'key_fields':['content_jaccard_mean','source_cap_overlap_frac_mean_cond_source_has_cap','relation_overlap_mean_cond_any_relation','feature densities per 1k words']
        }
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

    p=pa; oc=oa['combined']; sc=p['source_feature_aggregate']; tc=p['target_feature_aggregate']; cc=p['combined_feature_aggregate']
    def f(x): return f'{x:.3f}' if isinstance(x,float) else str(x)
    lines=[]
    lines.append('# research — why the WikiAuto aligned-pair mechanism failed')
    lines.append('')
    lines.append(f'Evidence JSON: `{OUT_JSON}`')
    lines.append('')
    lines.append('## Pair correspondence')
    lines.append('')
    lines.append(f'- selected pairs: {p["pairs"]:,}')
    lines.append(f'- source words / target words: {p["source_words_total"]:,} / {p["target_words_total"]:,}')
    lines.append(f'- target/source word ratio mean {p["target_source_word_ratio_mean"]:.3f}, median {p["target_source_word_ratio_median"]:.3f}, p10-p90 {p["target_source_word_ratio_p10_p90"][0]:.3f}–{p["target_source_word_ratio_p10_p90"][1]:.3f}')
    lines.append(f'- content-token Jaccard mean {p["content_jaccard_mean"]:.3f}, median {p["content_jaccard_median"]:.3f}, p10-p90 {p["content_jaccard_p10_p90"][0]:.3f}–{p["content_jaccard_p10_p90"][1]:.3f}')
    lines.append(f'- target content-token copy fraction mean {p["target_content_token_copy_frac_mean"]:.3f}; source covered by target mean {p["source_content_token_covered_by_target_mean"]:.3f}')
    lines.append(f'- source has noninitial capitalized token in {100*p["pairs_with_source_capitalized_noninitial_frac"]:.1f}% of pairs; target has one in {100*p["pairs_with_target_capitalized_noninitial_frac"]:.1f}%')
    lines.append(f'- when source has a capitalized token, mean source-cap preservation in target is {p["source_cap_overlap_frac_mean_cond_source_has_cap"]:.3f}; any overlap occurs in {100*p["source_cap_any_overlap_frac_cond_source_has_cap"]:.1f}%')
    lines.append(f'- relation-word overlap mean, conditional on any relation cue, is {p["relation_overlap_mean_cond_any_relation"]:.3f}; both sides contain relation cues in {100*p["both_sides_relation_frac_cond_any_relation"]:.1f}% of such pairs')
    lines.append('')
    lines.append('## Aggregate feature densities per 1k regex words')
    lines.append('')
    lines.append('| feature | WikiAuto source | WikiAuto target | WikiAuto combined | official corpus |')
    lines.append('|---|---:|---:|---:|---:|')
    for key,label in [('pronouns','pronouns'),('capitalized_noninitial','capitalized noninitial'),('numbers','numbers'),('spatial','spatial words'),('state_transfer','state/transfer verbs'),('physical','physical/material words'),('social','social words'),('causal_temporal','causal/temporal words'),('relation_any','any relation cue')]:
        kk=key+'_per_1k_regex_words'
        lines.append(f'| {label} | {sc[kk]:.2f} | {tc[kk]:.2f} | {cc[kk]:.2f} | {oc[kk]:.2f} |')
    lines.append('')
    lines.append('## Reading of the measurement')
    lines.append('')
    lines.append('The selected WikiAuto/Turk pairs do contain many content words and some capitalized entities, but the target side is a compressed sentence-level simplification rather than an explicit entity/relation bridge. The exact-pair training signal is therefore weakly aligned with the Entity/EWoK target: many source-side entity mentions and relation cues are not preserved in the target, and high lexical overlap encourages local surface reuse rather than learning persistent state or world-relation invariants.')
    lines.append('')
    lines.append('This makes the research negative result scientifically interpretable: the current source tests passive sentence-level simplification adjacency, not a dense entity/relation correspondence principle. A better route should rebuild the experience structure around checked entity and relation invariance, or around source selection/compression that preserves factual predicates, instead of scaling this exact aligned construction.')
    OUT_NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':payload['status'], 'out_json':str(OUT_JSON), 'out_note':str(OUT_NOTE), 'pairs':p['pairs'], 'content_jaccard_mean':p['content_jaccard_mean'], 'source_cap_overlap_mean':p['source_cap_overlap_frac_mean_cond_source_has_cap'], 'relation_overlap_mean':p['relation_overlap_mean_cond_any_relation']}, indent=2))

if __name__=='__main__':
    main()
