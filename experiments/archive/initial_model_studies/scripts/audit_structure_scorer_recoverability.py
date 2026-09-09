#!/usr/bin/env python3
from __future__ import annotations
import collections, json, pathlib, random, re, math
from typing import Any

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
MANIFEST = ROOT / 'data/structure_density_revision_157/manifest.json'
OUT = ROOT / 'data/structure_scorer_recoverability_audit.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/structure_scorer_recoverability_audit.md')
ARMS = ['high_entity_state','matched_low','uniform']

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
CAP_RE = re.compile(r"^[A-Z][a-z]+")
STOP = {
    'a','an','the','and','or','but','if','then','else','when','while','of','to','in','on','at','by','for','from','with','without','as','is','are','was','were','be','been','being','am','do','does','did','done','have','has','had','having','can','could','will','would','shall','should','may','might','must','not','no','yes','than','that','this','these','those','it','its','he','she','they','them','his','her','their','we','us','you','your','i','me','my','mine','our','ours','who','whom','which','what','where','why','how','there','here','also','more','most','other','some','any','all','one','two','first','second','new','old','many','much','such','into','over','under','after','before','about','up','down','out','off','only','through','between','during','against','within','because'
}
FIRST_SECOND = {'i','me','my','mine','we','us','our','ours','you','your','yours'}
THIRD_PRON = {'he','she','it','they','him','her','them','his','hers','its','their','theirs','himself','herself','itself','themselves'}
DEICTIC_REL = {'this','that','these','those','which','who','whom','whose'}
PRONOUNS = FIRST_SECOND | THIRD_PRON | DEICTIC_REL
STATE_TRANSFER = {'put','puts','placed','place','places','moved','move','moves','moving','transfer','transferred','give','gave','given','take','took','taken','bring','brought','send','sent','carry','carried','drop','dropped','fall','fell','fallen','break','broke','broken','open','opened','close','closed','change','changed','become','became','turn','turned','keep','kept','hold','held','store','stored','remove','removed','replace','replaced','contain','contains','contained','enter','entered','leave','left','push','pushed','pull','pulled','throw','threw','thrown','pick','picked','catch','caught','grab','grabbed','lift','lifted','pour','poured','fill','filled','empty','emptied','add','added','lose','lost','find','found','hide','hid','hidden','set','sit','sat','stand','stood','lay','laid','hang','hung','wear','wore','worn'}
SPATIAL = {'in','on','at','under','over','above','below','inside','outside','near','between','behind','front','beside','around','through','across','within','into','onto','from','to','left','right','north','south','east','west','up','down','there','here','top','bottom','middle','next','far','close','away','back','forward','toward','towards'}
PHYSICAL = {'glass','wood','wooden','metal','plastic','stone','rock','water','ice','fire','air','paper','cloth','rubber','fragile','heavy','light','hot','cold','warm','wet','dry','hard','soft','solid','liquid','gas','round','flat','sharp','smooth','rough','break','burn','melt','freeze','float','sink','fall','rise','push','pull','roll','slide','throw','hit','cut','bend','stretch','squeeze','bounce','crack','spill','pour','mix','stick','tear'}
SOCIAL = {'said','say','says','told','tell','asked','ask','asks','answer','answered','reply','replied','talk','spoke','speak','friend','mother','father','child','boy','girl','man','woman','people','person','team','group','family','king','queen','want','wanted','think','thought','know','knew','feel','felt','like','love','hate','need','help','helped','thank','please','sorry','happy','sad','angry','afraid','scared','surprised','worried'}
CAUSAL_TEMPORAL = {'because','cause','caused','therefore','so','if','then','after','before','during','while','when','until','since','although','though','however','result','results','finally','suddenly','immediately','soon','later','earlier','already','still','yet','now','again','once','first','next','last','begin','began','begun','start','started','end','ended','finish','finished','happen','happened','make','made','let'}
REL = STATE_TRANSFER | SPATIAL | PHYSICAL | SOCIAL | CAUSAL_TEMPORAL


def toks(text: str) -> list[str]:
    return WORD_RE.findall(text)

def lowers(text: str) -> list[str]:
    return [t.lower() for t in toks(text)]

def mean(xs): return sum(xs)/len(xs) if xs else 0.0

def frac(xs): return sum(1 for x in xs if x)/len(xs) if xs else 0.0

def pct(xs, q):
    if not xs: return 0.0
    ys=sorted(xs); pos=(len(ys)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    if lo==hi: return ys[lo]
    return ys[lo]*(hi-pos)+ys[hi]*(pos-lo)


def features(text: str) -> dict[str, Any]:
    T=toks(text); L=[x.lower() for x in T]; n=max(1,len(T))
    caps=[]
    for i,t in enumerate(T):
        if i>0 and CAP_RE.match(t) and t.lower()!='i': caps.append(t.lower())
    cap_counts=collections.Counter(caps)
    content=[x for x in L if x not in STOP and not re.fullmatch(r'\d+(?:\.\d+)?', x)]
    cont_counts=collections.Counter(content)
    repeated_content_types=[x for x,c in cont_counts.items() if c>=2]
    repeated_caps=[x for x,c in cap_counts.items() if c>=2]
    first_second=sum(1 for x in L if x in FIRST_SECOND)
    third=sum(1 for x in L if x in THIRD_PRON)
    deictic=sum(1 for x in L if x in DEICTIC_REL)
    pron=first_second+third+deictic
    state=sum(1 for x in L if x in STATE_TRANSFER)
    spatial=sum(1 for x in L if x in SPATIAL)
    physical=sum(1 for x in L if x in PHYSICAL)
    social=sum(1 for x in L if x in SOCIAL)
    causal=sum(1 for x in L if x in CAUSAL_TEMPORAL)
    # Original approximate scores.
    entity_state_score=(sum(cap_counts[x] for x in repeated_caps)+pron+state)/n
    # Recoverability proxies: deliberately stricter than lexical cue count.
    has_entity_chain = bool(repeated_caps) or (len(cap_counts)>=1 and third>=1) or (len(repeated_content_types)>=2 and third>=1)
    has_state_relation = state>=1 and (spatial>=1 or third>=1 or bool(repeated_caps) or len(repeated_content_types)>=2)
    entity_state_recoverable = has_entity_chain and state>=1
    coref_candidate = has_entity_chain
    state_transition_candidate = state>=2 and spatial>=1
    physical_process_candidate = (physical>=1 and (state>=1 or spatial>=1))
    social_causal_candidate = social>=1 and (causal>=1 or third>=1)
    # Potential false-positive mode: pronoun-rich dialogue without named/repeated entity anchor.
    pronoun_dialogue_like = first_second >= max(3, third + len(cap_counts) + 1)
    cue_without_anchor = (pron+state)>=3 and not has_entity_chain
    return {
        'words': len(text.split()), 'regex_words': n,
        'cap_count': len(caps), 'cap_types': len(cap_counts), 'repeated_cap_types': len(repeated_caps),
        'repeated_content_types': len(repeated_content_types),
        'first_second_pronouns': first_second, 'third_pronouns': third, 'deictic_pronouns': deictic, 'pronouns': pron,
        'state_transfer': state, 'spatial': spatial, 'physical': physical, 'social': social, 'causal_temporal': causal,
        'entity_state_score': entity_state_score,
        'coref_candidate': coref_candidate, 'entity_state_recoverable': entity_state_recoverable,
        'state_transition_candidate': state_transition_candidate, 'physical_process_candidate': physical_process_candidate,
        'social_causal_candidate': social_causal_candidate, 'pronoun_dialogue_like': pronoun_dialogue_like,
        'cue_without_anchor': cue_without_anchor,
        'top_repeated_content': repeated_content_types[:8], 'top_repeated_caps': repeated_caps[:8],
    }


def load_unique(path: pathlib.Path) -> list[dict[str, Any]]:
    out=[]; seen=set()
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r.get('epoch') != 0: continue
            key=(r.get('source_file'), r.get('start_line'), r.get('end_line'), r.get('text'))
            if key in seen: continue
            seen.add(key)
            ft=features(r['text']); ft.update({k:r.get(k) for k in ['source_file','start_line','end_line','words']}); ft['text_preview']=r['text'][:360]
            out.append(ft)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n=len(rows); words=sum(r['words'] for r in rows)
    numeric=['entity_state_score','cap_count','repeated_cap_types','repeated_content_types','first_second_pronouns','third_pronouns','pronouns','state_transfer','spatial','physical','social','causal_temporal']
    bools=['coref_candidate','entity_state_recoverable','state_transition_candidate','physical_process_candidate','social_causal_candidate','pronoun_dialogue_like','cue_without_anchor']
    top_sources=collections.Counter(r['source_file'] for r in rows)
    cue_tokens=collections.Counter()
    for r in rows:
        # reconstruct from preview? skip exact; top false-positive examples carry enough.
        pass
    return {
        'unique_windows': n, 'unique_words': words,
        'means_per_window': {k: mean([r[k] for r in rows]) for k in numeric},
        'per_1k_words': {k: 1000*sum(r[k] for r in rows)/max(1,words) for k in numeric if k!='entity_state_score'},
        'fractions': {k: frac([r[k] for r in rows]) for k in bools},
        'entity_state_score_quantiles': {'p10':pct([r['entity_state_score'] for r in rows],0.1),'p50':pct([r['entity_state_score'] for r in rows],0.5),'p90':pct([r['entity_state_score'] for r in rows],0.9)},
        'source_counts': dict(top_sources),
        'examples_high_score_low_recoverability': sorted([r for r in rows if r['entity_state_score']>0.22 and not r['entity_state_recoverable']], key=lambda x:-x['entity_state_score'])[:12],
        'examples_recoverable': sorted([r for r in rows if r['entity_state_recoverable']], key=lambda x:-x['entity_state_score'])[:12],
    }


def main():
    manifest=json.loads(MANIFEST.read_text())
    arms={}
    for arm in ARMS:
        rows=load_unique(pathlib.Path(manifest['arm_paths'][arm]['path']))
        arms[arm]={'summary':summarize(rows)}
    # Add ratios high/low and high/uniform for interpretable fields.
    ratios={}
    for denom in ['matched_low','uniform']:
        ratios[f'high_entity_state_over_{denom}']={}
        for field in ['coref_candidate','entity_state_recoverable','state_transition_candidate','pronoun_dialogue_like','cue_without_anchor']:
            h=arms['high_entity_state']['summary']['fractions'][field]; d=arms[denom]['summary']['fractions'][field]
            ratios[f'high_entity_state_over_{denom}'][field]=(h/max(1e-9,d), h, d)
    payload={'status':'STRUCTURE_SCORER_RECOVERABILITY_AUDIT','manifest':str(MANIFEST),'arms':arms,'ratios':ratios,
             'interpretation':'Audits whether the research entity_state scorer separated plausible recoverable entity/state structure or mainly lexical cues such as pronouns/state words without anchors.'}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    s={a:arms[a]['summary'] for a in ARMS}
    lines=['# research — structure-density scorer recoverability audit','',f'Evidence JSON: `{OUT}`','',
           'This audits whether the high_entity_state scorer measured recoverable entity/state structure or mostly cue-token density. The proxies are heuristic, but they separate anchored cross-mention/state candidates from pronoun-rich or cue-rich text without an entity anchor.','',
           '| arm | unique windows | entity_state score mean | coref-candidate frac | entity+state recoverable frac | state-transition frac | pronoun-dialogue-like frac | cue-without-anchor frac |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        sm=s[a]
        lines.append(f"| {a} | {sm['unique_windows']} | {sm['means_per_window']['entity_state_score']:.3f} | {100*sm['fractions']['coref_candidate']:.1f}% | {100*sm['fractions']['entity_state_recoverable']:.1f}% | {100*sm['fractions']['state_transition_candidate']:.1f}% | {100*sm['fractions']['pronoun_dialogue_like']:.1f}% | {100*sm['fractions']['cue_without_anchor']:.1f}% |")
    lines += ['', '## Reading', '']
    h=s['high_entity_state']['fractions']; l=s['matched_low']['fractions']; u=s['uniform']['fractions']
    lines.append(f"High_entity_state raises the heuristic entity+state recoverable fraction from {100*l['entity_state_recoverable']:.1f}% (matched_low) and {100*u['entity_state_recoverable']:.1f}% (uniform) to {100*h['entity_state_recoverable']:.1f}%, but it also changes pronoun/cue distributions and still failed to improve Entity/EWoK. This suggests either the proxy captures only shallow recoverability, or ordinary WWM does not force use of the structure.")
    lines.append('')
    lines.append('The next decision should be made from this audit plus the three-arm evaluation, not from score labels alone.')
    NOTE.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':payload['status'],'out':str(OUT),'note':str(NOTE),'fractions':{a:s[a]['fractions'] for a in ARMS}}, indent=2))

if __name__=='__main__': main()
