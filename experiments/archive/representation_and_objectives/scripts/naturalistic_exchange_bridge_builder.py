#!/usr/bin/env python3
"""research: build an evaluation-independent naturalistic exchange bridge.

The bridge is a held-out readout, not training data.  It uses naturalistic
role-exchange contexts with constructions that are intentionally different from
the research packet grammar and from official BabyLM evaluation rows.  The goal is
not to tune to official tasks: it is to test whether a small representation-forming
exchange update transfers beyond the synthetic training grammar before any long
endpoint is considered.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import hashlib
import itertools
import json
import os
import pathlib
import random
import re
import time
from typing import Any

ROOT = pathlib.Path('.')
WS = ROOT / 'experiments/archive/representation_and_objectives'
OUT = WS / 'data/naturalistic_exchange_bridge'
OFFICIAL_ROOT = WS / 'data/pristine_official_coordinate/babylm-eval/strict/evaluation_data'
TRAIN = WS / 'data/role_switch_expanded_packets/train_treatment.jsonl'
SEED = 151151
N_PER_TEMPLATE = 24

PEOPLE = [
    'Mira','Noah','Leah','Owen','Iris','Theo','Rosa','Caleb','Nina','Omar','Sofia','Eli',
    'Zoe','Arun','Maya','Liam','Pia','Jonah','Tara','Leo','Ava','Miles','Nora','Felix',
    'Ruby','Simon','Clara','Jules','Hana','Dylan','Mina','Oscar'
]
OBJECTS = [
    'mug','spoon','pencil','button','wallet','ticket','scarf','lantern','marble','notebook',
    'brush','shell','ring','coin','glove','candle','map','photo','rope','key','cup','bowl',
    'feather','stone','badge','ribbon','card','fork','toy','sock','drum','book'
]
THINGS = ['ticket','scarf','map','key','notebook','coin','badge','photo','ring','ribbon','card','shell']
CONTAINERS = ['tin','basket','drawer','backpack','box','jar','cabinet','bucket']
HI_LO = [(18, 7), (21, 9), (16, 5), (24, 11), (14, 6), (19, 8), (22, 10), (17, 4)]
PLACES = ['window','bench','sink','door','lamp','rug','shelf','counter']

# Each template uses x/y as the exchange alternatives.  correct_role says which
# alternative is correct in context_AB; context_BA is rendered by swapping x/y.
TEMPLATES: list[dict[str, Any]] = [
    # possession / transfer; target is the recipient or final holder (y)
    {'family':'natural_transfer','id':'nt01','pool':'people','correct_role':'y',
     'ctx':'During cleanup, {x} slipped the {thing} into {y}\'s backpack while the hallway was noisy.',
     'con':'By the time cleanup ended, __TARGET__ was carrying the {thing}.'},
    {'family':'natural_transfer','id':'nt02','pool':'people','correct_role':'y',
     'ctx':'{x} mailed the {thing} to {y} after lunch, and the package arrived before dinner.',
     'con':'After the delivery, __TARGET__ had the {thing}.'},
    {'family':'natural_transfer','id':'nt03','pool':'people','correct_role':'y',
     'ctx':'The class passed the {thing} from {x} over to {y} during the game.',
     'con':'At the end of that pass, the holder of the {thing} was __TARGET__.'},
    {'family':'natural_transfer','id':'nt04','pool':'people','correct_role':'y',
     'ctx':'Before leaving, {x} returned the borrowed {thing} to {y}.',
     'con':'The person who got the {thing} back was __TARGET__.'},
    {'family':'heard_or_received','id':'hr01','pool':'people','correct_role':'y',
     'ctx':'During rehearsal, {x} whispered the clue to {y} near the curtain.',
     'con':'The person who heard the clue was __TARGET__.'},
    {'family':'heard_or_received','id':'hr02','pool':'people','correct_role':'y',
     'ctx':'After the vase broke, {x} blamed {y} in front of the group.',
     'con':'The person being blamed was __TARGET__.'},

    # spatial, containment, and label matching; target is object x
    {'family':'spatial_path','id':'spn01','pool':'objects','correct_role':'x',
     'ctx':'When the tray tilted, the {x} slid onto the napkin and the {y} stayed near the cup.',
     'con':'The object on the napkin was the __TARGET__.'},
    {'family':'spatial_path','id':'spn02','pool':'objects','correct_role':'x',
     'ctx':'After the boxes were sorted, the {x} rested on the upper shelf while the {y} was on the lower shelf.',
     'con':'The item on the upper shelf was the __TARGET__.'},
    {'family':'spatial_path','id':'spn03','pool':'objects','correct_role':'x',
     'ctx':'In the photo, the {x} was just to the left of the {y} beside the {place}.',
     'con':'The left-hand object was the __TARGET__.'},
    {'family':'spatial_path','id':'spn04','pool':'objects','correct_role':'x',
     'ctx':'After the spill, the {x} landed inside the chalk circle and the {y} landed outside it.',
     'con':'The object inside the circle was the __TARGET__.'},
    {'family':'containment_natural','id':'cnr01','pool':'objects','correct_role':'x',
     'ctx':'{P} tucked the {x} inside the {container}, but kept the {y} on the table.',
     'con':'Inside the {container}, {P} would find the __TARGET__.'},
    {'family':'containment_natural','id':'cnr02','pool':'objects','correct_role':'x',
     'ctx':'The {x} was sealed in the {container}; the {y} remained beside the lid.',
     'con':'The sealed object was the __TARGET__.'},
    {'family':'label_binding','id':'lb01','pool':'objects','correct_role':'x',
     'ctx':'The red tag was tied to the {x}, and the blue tag was tied to the {y}.',
     'con':'The red-tagged item was the __TARGET__.'},

    # quantities and temporal order; target is person x
    {'family':'quantity_comparison','id':'qc01','pool':'people','correct_role':'x',
     'ctx':'At the booth, {x} collected {hi} tokens and {y} collected {lo}.',
     'con':'The larger token pile belonged to __TARGET__.'},
    {'family':'quantity_comparison','id':'qc02','pool':'people','correct_role':'x',
     'ctx':'For the puzzle, {x} solved {hi} clues, whereas {y} solved {lo} clues.',
     'con':'The person with more solved clues was __TARGET__.'},
    {'family':'quantity_comparison','id':'qc03','pool':'people','correct_role':'x',
     'ctx':'On the hike, {x} carried {hi} shells and {y} carried only {lo}.',
     'con':'The one carrying more shells was __TARGET__.'},
    {'family':'temporal_order_repaired','id':'tor01','pool':'people','correct_role':'x',
     'ctx':'Before the lights dimmed, {x} signed the log; after the announcement, {y} signed it.',
     'con':'The earlier signature was from __TARGET__.'},
    {'family':'temporal_order_repaired','id':'tor02','pool':'people','correct_role':'x',
     'ctx':'{x} reached the gate in the morning, and {y} reached it in the afternoon.',
     'con':'The first person at the gate was __TARGET__.'},
    {'family':'temporal_order_repaired','id':'tor03','pool':'people','correct_role':'x',
     'ctx':'While the cookies cooled, {x} washed a cup before {y} washed one.',
     'con':'The earlier washer was __TARGET__.'},

    # object state and simple causal affordance; target is x
    {'family':'object_state','id':'os01','pool':'objects','correct_role':'x',
     'ctx':'Under the awning, the {x} stayed dry while the {y} got wet in the rain.',
     'con':'The dry thing was the __TARGET__.'},
    {'family':'object_state','id':'os02','pool':'objects','correct_role':'x',
     'ctx':'{P} polished the {x} until it shone, but left the {y} dusty.',
     'con':'The shiny object was the __TARGET__.'},
    {'family':'object_state','id':'os03','pool':'objects','correct_role':'x',
     'ctx':'After the freezer broke, the {x} stayed frozen in the cooler while the {y} thawed on the counter.',
     'con':'The still-frozen thing was the __TARGET__.'},
    {'family':'ability_cause','id':'ac01','pool':'people','correct_role':'x',
     'ctx':'{x} knew the safe code, but {y} had never seen it.',
     'con':'The person able to open the safe was __TARGET__.'},
    {'family':'ability_cause','id':'ac02','pool':'people','correct_role':'x',
     'ctx':'{x} packed the map for the trip, while {y} forgot it at home.',
     'con':'The person with the map was __TARGET__.'},
]

WORD_RE = re.compile(r"[A-Za-z0-9']+")


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def words(s: str) -> list[str]:
    return WORD_RE.findall(s.lower())


def ngrams(s: str, n: int) -> set[tuple[str, ...]]:
    w = words(s)
    return {tuple(w[i:i+n]) for i in range(max(0, len(w) - n + 1))}


def read_texts_jsonish(root: pathlib.Path) -> list[str]:
    out: list[str] = []
    if not root.exists():
        return out
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(('.jsonl', '.json', '.txt', '.csv', '.tsv')):
                continue
            fp = pathlib.Path(dirpath) / fn
            try:
                text = fp.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            if fn.endswith('.jsonl'):
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            for v in obj.values():
                                if isinstance(v, str):
                                    out.append(v)
                                elif isinstance(v, list):
                                    out.extend(str(x) for x in v if isinstance(x, (str, int, float)))
                        elif isinstance(obj, list):
                            out.extend(str(x) for x in obj if isinstance(x, (str, int, float)))
                    except Exception:
                        out.append(line)
            elif fn.endswith('.json'):
                try:
                    obj = json.loads(text)
                    stack = [obj]
                    while stack:
                        cur = stack.pop()
                        if isinstance(cur, str):
                            out.append(cur)
                        elif isinstance(cur, dict):
                            stack.extend(cur.values())
                        elif isinstance(cur, list):
                            stack.extend(cur)
                    continue
                except Exception:
                    pass
                out.append(text)
            else:
                out.append(text)
    return out


def official_ngram_sets() -> dict[int, set[tuple[str, ...]]]:
    texts = read_texts_jsonish(OFFICIAL_ROOT)
    sets = {7: set(), 8: set(), 10: set()}
    for t in texts:
        w = words(t)
        for n in sets:
            for i in range(max(0, len(w) - n + 1)):
                sets[n].add(tuple(w[i:i+n]))
    return sets


def ngram_sets() -> dict[int, set[tuple[str, ...]]]:
    sets = {7: set(), 8: set(), 10: set()}
    if not TRAIN.exists():
        return sets
    with TRAIN.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                text = json.loads(line)['text']
            except Exception:
                text = line
            w = words(text)
            for n in sets:
                for i in range(max(0, len(w) - n + 1)):
                    sets[n].add(tuple(w[i:i+n]))
    return sets


def render_pair_texts(p: dict[str, Any]) -> list[str]:
    texts = []
    for ctx_key in ['AB', 'BA']:
        ctx = p[f'context_{ctx_key}']
        for alt_key in ['alt_0', 'alt_1']:
            alt = p[alt_key]
            texts.append(ctx + ' ' + p['consequence_masked'].replace('__TARGET__', alt))
    return texts


def make_pairs() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    pairs: list[dict[str, Any]] = []
    pid = 151000
    for tmpl in TEMPLATES:
        pool = PEOPLE if tmpl['pool'] == 'people' else OBJECTS
        combos = list(itertools.permutations(pool, 2))
        rng.shuffle(combos)
        for a, b in combos[:N_PER_TEMPLATE]:
            P = rng.choice([x for x in PEOPLE if x not in (a, b)])
            hi, lo = rng.choice(HI_LO)
            kw = {
                'thing': rng.choice(THINGS),
                'container': rng.choice(CONTAINERS),
                'place': rng.choice(PLACES),
                'P': P,
                'hi': str(hi),
                'lo': str(lo),
            }
            ctx_ab = tmpl['ctx'].format(x=a, y=b, **kw)
            ctx_ba = tmpl['ctx'].format(x=b, y=a, **kw)
            con = tmpl['con'].format(x=a, y=b, **kw)
            if tmpl['correct_role'] == 'x':
                correct_ab, correct_ba = a, b
            else:
                correct_ab, correct_ba = b, a
            p = {
                'pair_id': pid,
                'family': tmpl['family'],
                'template_id': tmpl['id'],
                'style': 'bridge_unseen',
                'entity_split': 'bridge_people' if tmpl['pool'] == 'people' else 'bridge_objects',
                'alt_type': tmpl['pool'][:-1] if tmpl['pool'].endswith('s') else tmpl['pool'],
                'alt_0': a,
                'alt_1': b,
                'context_AB': ctx_ab,
                'context_BA': ctx_ba,
                'consequence_masked': con,
                'correct_AB': correct_ab,
                'correct_BA': correct_ba,
                'source': 'naturalistic_exchange_bridge',
                'vars': kw,
            }
            p['word_count_scored_text_mean'] = sum(len(words(t)) for t in render_pair_texts(p)) / 4.0
            pairs.append(p)
            pid += 1
    return pairs


def count_overlaps(pairs: list[dict[str, Any]], ref_sets: dict[int, set[tuple[str, ...]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n, ref in sorted(ref_sets.items()):
        hits = []
        for p in pairs:
            hit_grams = set()
            for t in render_pair_texts(p):
                hit_grams |= (ngrams(t, n) & ref)
            if hit_grams:
                hits.append({'pair_id': p['pair_id'], 'n': n, 'hit_count': len(hit_grams), 'hits_head': [' '.join(g) for g in sorted(hit_grams)[:5]]})
        out[str(n)] = {'n_pairs_with_overlap': len(hits), 'hits_head': hits[:20]}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = make_pairs()

    pair_path = OUT / 'naturalistic_exchange_bridge_pairs.jsonl'
    with pair_path.open('w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    official_sets = official_ngram_sets()
    sets = ngram_sets()
    family_counts = dict(sorted(collections.Counter(p['family'] for p in pairs).items()))
    template_counts = dict(sorted(collections.Counter(p['template_id'] for p in pairs).items()))
    token_words = [p['word_count_scored_text_mean'] for p in pairs]
    summary = {
        'status': 'NATURALISTIC_EXCHANGE_BRIDGE',
        'created_utc': now_utc(),
        'scientific_use': 'Held-out bridge for transfer beyond the research packet training grammar and outside official BabyLM tasks; not used for training.',
        'pair_path': str(pair_path),
        'n_pairs': len(pairs),
        'n_templates': len(TEMPLATES),
        'n_per_template': N_PER_TEMPLATE,
        'family_counts': family_counts,
        'template_counts': template_counts,
        'scored_text_word_count_mean': sum(token_words) / len(token_words),
        'scored_text_word_count_min': min(token_words),
        'scored_text_word_count_max': max(token_words),
        'official_ngram_overlap': count_overlaps(pairs, official_sets),
        'train_ngram_overlap': count_overlaps(pairs, sets),
        'pools': {
            'people': PEOPLE,
            'objects': OBJECTS,
            'things': THINGS,
            'containers': CONTAINERS,
        },
        'code_hash': sha256_file(_public_path('experiments/archive/representation_and_objectives/scripts/naturalistic_exchange_bridge_builder.py')),
    }
    spath = OUT / 'naturalistic_exchange_bridge_summary.json'
    spath.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'BRIDGE_BUILT', 'pairs': len(pairs), 'pair_path': str(pair_path), 'summary': str(spath), 'family_counts': family_counts}, indent=2), flush=True)


if __name__ == '__main__':
    main()
