#!/usr/bin/env python3
"""Step 167b — Broader binding probe with entity-position swap.

Fixes Step167a's bottleneck: instead of requiring same-length predicate objects
for positional swap (which yielded only 5 examples), swap ENTITY NAMES at their
sentence positions. This preserves the word bag perfectly when both entities are
single words.

Design:
  Original: "Alice put the ball in the box. Bob put the cup on the shelf."
  Swapped:  "Bob put the ball in the box. Alice put the cup on the shelf."
  Target: "box" when asking about Alice's location.
  In original → Alice is associated with "box" (correct).
  In swapped → Alice is associated with "shelf" (wrong).
  Bag is identical; only entity-predicate binding changed.

Scans the FULL official corpus with larger sentence-pair budget and relaxed criteria.
"""
from __future__ import annotations
import collections, json, os, pathlib, random, re, time
from typing import Any
import torch
import torch.nn.functional as F

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
CORPUS_DIR = ROOT / 'training/runs/babylm_pilot_dense_v0_1M/raw_dataset'
MODEL_DIR = (ROOT / 'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M').resolve()
OUT = ROOT / 'data/broader_binding_probe_v2.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/broader_binding_probe_v2.md')
MAX_SEQ = 256
SEED = 1670

# ── Entity and predicate detection ──

CAP_RE = re.compile(r'^[A-Z][a-z]{2,}$')
STOP_ENT = {'The','And','But','Or','If','When','While','He','She','It','They',
    'We','You','His','Her','Their','This','That','These','Those','So','As','At',
    'In','On','Of','To','For','With','My','Your','Its','What','Who','Where','How',
    'Why','Then','Now','Here','There','After','Before','Not','No','Yes','Well','Oh',
    'Mr','Mrs','Miss','Dr','Sir','Just','Even','Still','Yet','Once','Also','Very',
    'About','Because','However','Although','Since','Already','Many','Some','All',
    'Each','Every','Another','Other','New','Old','More','Most','Much','Such',
    'Chapter','Section','Part','Book','Page','Table','Figure'}

SPATIAL_PREPS = {'in','on','at','under','over','above','below','inside','outside',
    'near','between','behind','beside','around','into','onto','from','to','toward',
    'towards','beneath','next'}

STATE_VERBS = {'put','puts','placed','place','moved','move','took','take','brought',
    'bring','gave','give','sent','send','left','leave','dropped','drop','picked',
    'carried','carry','held','hold','kept','keep','hid','hide','found','find',
    'set','went','go','came','come','lived','live','returned','return','sat','sit',
    'stood','stand','lay','hung','wore','wear','threw','throw','pushed','pulled',
    'is','was','are','were','lives','lived','stays','stayed','born','died','works',
    'worked','played','plays','studied','studies','teaches','taught'}

def find_single_word_cap_entities(words: list[str]) -> list[tuple[int, str]]:
    """Find single-word capitalized entities (non-initial, non-stop)."""
    ents = []
    for i, w in enumerate(words):
        clean = re.sub(r'[^A-Za-z]', '', w)
        if CAP_RE.match(clean) and clean not in STOP_ENT:
            ents.append((i, clean))
    return ents

def find_predicate_for_entity(words: list[str], ent_pos: int) -> dict | None:
    """Find spatial/state predicate associated with entity at ent_pos."""
    lower = [w.lower().strip('.,!?;:\'"()[]{}') for w in words]
    n = len(words)
    for i in range(ent_pos + 1, min(n, ent_pos + 7)):
        if lower[i] in STATE_VERBS:
            for j in range(i + 1, min(n, i + 5)):
                if lower[j] in SPATIAL_PREPS:
                    obj_words = []
                    obj_positions = []
                    for k in range(j + 1, min(n, j + 5)):
                        lw = lower[k]
                        if lw in SPATIAL_PREPS or lw in ('.','!','?',',','and','but','or','then','so','who','which','that','where'):
                            break
                        if len(lw) >= 2 and lw not in ('the','a','an','his','her','its','their','my','your'):
                            obj_words.append(lw)
                            obj_positions.append(k)
                    if obj_words:
                        return {'verb': lower[i], 'prep': lower[j],
                                'object': ' '.join(obj_words), 'object_positions': obj_positions}
                    break
            break
    return None

def scan_corpus_for_binding_windows(corpus_dir: pathlib.Path, seed: int) -> list[dict]:
    """Scan full corpus for windows with ≥2 single-word entities having distinct predicates."""
    rng = random.Random(seed)
    results = []
    
    for path in sorted(corpus_dir.glob('*.train.txt')):
        lines = []
        with path.open('r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Slide a 3-sentence window across each line block
        buf = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                buf = []
                continue
            # Split into sentences
            sents = re.split(r'(?<=[.!?])\s+', stripped)
            for s in sents:
                words = s.split()
                if len(words) >= 4:
                    buf.append(words)
                if len(buf) >= 3:
                    # Analyze the 2-3 sentence window
                    combined = []
                    offsets = []
                    for sent_words in buf[-3:]:
                        offsets.append(len(combined))
                        combined.extend(sent_words)
                    
                    # Find entities across the window
                    ents = find_single_word_cap_entities(combined)
                    # Group by surface
                    by_surf = collections.defaultdict(list)
                    for pos, surf in ents:
                        by_surf[surf].append(pos)
                    
                    # Find distinct entities with predicates
                    ent_with_pred = []
                    seen_objs = set()
                    for surf, positions in by_surf.items():
                        pred = find_predicate_for_entity(combined, positions[0])
                        if pred and pred['object'] not in seen_objs:
                            ent_with_pred.append({'entity': surf, 'entity_pos': positions[0], 'pred': pred})
                            seen_objs.add(pred['object'])
                    
                    if len(ent_with_pred) >= 2:
                        results.append({
                            'combined_words': combined,
                            'ent_with_pred': ent_with_pred[:4],
                            'source': path.name,
                        })
                    
                    buf = buf[-2:]  # slide window
    
    rng.shuffle(results)
    return results

def construct_entity_swap(result: dict) -> dict | None:
    """Swap two entity names at their positions to create a binding-broken version."""
    combined = list(result['combined_words'])
    ewp = result['ent_with_pred']
    if len(ewp) < 2:
        return None
    
    e1 = ewp[0]
    e2 = ewp[1]
    
    # Both entities must be single words at known positions
    pos1 = e1['entity_pos']
    pos2 = e2['entity_pos']
    name1 = e1['entity']
    name2 = e2['entity']
    
    if pos1 >= len(combined) or pos2 >= len(combined):
        return None
    
    # Verify the words at those positions match
    clean1 = re.sub(r'[^A-Za-z]', '', combined[pos1])
    clean2 = re.sub(r'[^A-Za-z]', '', combined[pos2])
    if clean1 != name1 or clean2 != name2:
        return None
    
    # Swap entity names (preserving punctuation)
    swapped = list(combined)
    # Replace only the alphabetic part, keeping any attached punctuation
    swapped[pos1] = combined[pos1].replace(name1, name2)
    swapped[pos2] = combined[pos2].replace(name2, name1)
    
    # Also swap ALL other occurrences of these entity names in the window
    for i in range(len(swapped)):
        if i == pos1 or i == pos2:
            continue
        ci = re.sub(r'[^A-Za-z]', '', combined[i])
        if ci == name1:
            swapped[i] = combined[i].replace(name1, name2)
        elif ci == name2:
            swapped[i] = combined[i].replace(name2, name1)
    
    # Target: the object words of e1's predicate (state/location, not entity name)
    target_positions = e1['pred']['object_positions']
    target_words = [combined[p].lower().strip('.,!?;:\'"()[]{}') for p in target_positions if p < len(combined)]
    
    if not target_words:
        return None
    
    # Verify bag preservation: sorted lowercase words should be identical
    bag_orig = sorted(w.lower() for w in combined)
    bag_swap = sorted(w.lower() for w in swapped)
    if bag_orig != bag_swap:
        return None
    
    return {
        'correct_text': ' '.join(combined),
        'swapped_text': ' '.join(swapped),
        'target_words': target_words,
        'target_positions': target_positions,
        'entity1': name1,
        'entity2': name2,
        'e1_predicate': e1['pred']['object'],
        'e2_predicate': e2['pred']['object'],
        'source': result['source'],
    }

def setup_env():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

@torch.no_grad()
def probe_binding(model, tokenizer, correct_text: str, swapped_text: str,
                  target_words: list[str], device) -> dict | None:
    """Measure target token likelihood under correct vs swapped binding."""
    mask_id = tokenizer.mask_token_id
    
    enc_c = tokenizer(correct_text, add_special_tokens=False, truncation=True, max_length=MAX_SEQ)
    enc_s = tokenizer(swapped_text, add_special_tokens=False, truncation=True, max_length=MAX_SEQ)
    ids_c = enc_c['input_ids']
    ids_s = enc_s['input_ids']
    
    if len(ids_c) != len(ids_s):
        return None
    
    # Find target tokens in the correct text
    target_text = ' '.join(target_words)
    target_ids = tokenizer(target_text, add_special_tokens=False)['input_ids']
    if not target_ids:
        return None
    
    # Search for target token sequence in correct ids
    tgt_positions = []
    for i in range(len(ids_c) - len(target_ids) + 1):
        if ids_c[i:i+len(target_ids)] == target_ids:
            tgt_positions = list(range(i, i + len(target_ids)))
            break
    
    if not tgt_positions:
        # Try matching with space prefix
        target_ids_sp = tokenizer(' ' + target_text, add_special_tokens=False)['input_ids']
        for i in range(len(ids_c) - len(target_ids_sp) + 1):
            if ids_c[i:i+len(target_ids_sp)] == target_ids_sp:
                tgt_positions = list(range(i, i + len(target_ids_sp)))
                break
    
    if not tgt_positions:
        return None
    
    # The target tokens should be identical in both versions at the same positions
    # (entity swap shouldn't change the predicate object tokens)
    for p in tgt_positions:
        if p >= len(ids_s) or ids_c[p] != ids_s[p]:
            return None  # Target tokens differ → not target-identical
    
    target_token_ids = [ids_c[p] for p in tgt_positions]
    
    # Mask targets in both
    masked_c = list(ids_c)
    masked_s = list(ids_s)
    for p in tgt_positions:
        masked_c[p] = mask_id
        masked_s[p] = mask_id
    
    def get_ll(masked_ids):
        inp = torch.tensor([masked_ids], device=device)
        att = torch.ones_like(inp)
        logits = model(input_ids=inp, attention_mask=att).logits[0]
        lp = F.log_softmax(logits, dim=-1)
        total = 0.0
        for p, tid in zip(tgt_positions, target_token_ids):
            if p < logits.shape[0]:
                total += float(lp[p, tid])
        return total / max(1, len(tgt_positions))
    
    ll_c = get_ll(masked_c)
    ll_s = get_ll(masked_s)
    
    return {'ll_correct': round(ll_c, 4), 'll_swapped': round(ll_s, 4),
            'delta': round(ll_c - ll_s, 4)}

def main():
    setup_env()
    t0 = time.time()
    rng = random.Random(SEED)
    
    print("Scanning full corpus for binding windows...")
    results = scan_corpus_for_binding_windows(CORPUS_DIR, SEED)
    print(f"  Found {len(results)} windows with ≥2 entities + predicates")
    
    print("Constructing entity-position swaps...")
    swap_examples = []
    for r in results:
        swap = construct_entity_swap(r)
        if swap:
            swap_examples.append(swap)
    print(f"  Constructed {len(swap_examples)} bag-preserving entity-swap examples")
    
    if len(swap_examples) == 0:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({'status': 'NO_SWAP_EXAMPLES', 'windows': len(results),
                                    'sample_windows': [{'text': ' '.join(r['combined_words'][:40]),
                                                        'ents': [e['entity'] for e in r['ent_with_pred']]}
                                                       for r in results[:20]]}, indent=2) + '\n')
        print("No valid swap examples. See output for diagnostics.")
        return
    
    # Probe
    rng.shuffle(swap_examples)
    probe_n = min(300, len(swap_examples))
    probe_subset = swap_examples[:probe_n]
    
    print(f"Loading model for probe ({probe_n} examples)...")
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(MODEL_DIR))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    print(f"  Model on {device}")
    
    valid = []
    deltas = []
    for i, ex in enumerate(probe_subset):
        r = probe_binding(model, tokenizer, ex['correct_text'], ex['swapped_text'],
                         ex['target_words'], device)
        if r is not None:
            valid.append({**{k: ex[k] for k in ['entity1','entity2','e1_predicate','e2_predicate','source']},
                         'correct_preview': ex['correct_text'][:250],
                         'swapped_preview': ex['swapped_text'][:250],
                         'target_words': ex['target_words'], **r})
            deltas.append(r['delta'])
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{probe_n}, valid: {len(valid)}")
    
    n = len(deltas)
    if n > 0:
        sorted_d = sorted(deltas)
        stats = {'n': n, 'mean': round(sum(deltas)/n, 4),
                 'median': round(sorted_d[n//2], 4),
                 'positive_fraction': round(sum(1 for d in deltas if d > 0)/n, 4),
                 'p10': round(sorted_d[max(0,n//10)], 4),
                 'p90': round(sorted_d[min(n-1,9*n//10)], 4)}
    else:
        stats = {'n': 0}
    
    payload = {
        'status': 'BROADER_BINDING_PROBE_V2',
        'binding_windows_found': len(results),
        'entity_swap_examples': len(swap_examples),
        'probed': probe_n, 'valid_probes': n,
        'target_type': 'state/location tokens (not entity names)',
        'bag_preserved': True,
        'swap_method': 'entity-position swap (all occurrences of E1↔E2 in window)',
        'correct_minus_swapped_stats': stats,
        'examples': valid[:15],
        'elapsed_sec': round(time.time()-t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text('\n'.join([
        '# research — broader binding-substrate probe (v2, entity-position swap)',
        '', f'Binding windows (≥2 entities + predicates): {len(results)}',
        f'Entity-swap examples (bag-preserved): {len(swap_examples)}',
        f'Valid probe results: {n}', '',
        '## Probe results (correct binding minus swapped binding target LL)',
        f'  mean = {stats.get("mean")}', f'  median = {stats.get("median")}',
        f'  positive fraction = {stats.get("positive_fraction")}',
        f'  p10 = {stats.get("p10")}, p90 = {stats.get("p90")}', '',
        '## Interpretation',
        'If positive fraction > 0.6: official corpus has usable binding signal.',
        'If near 0.5 or below: model does not use entity-state binding for state prediction.',
    ]) + '\n')
    print(json.dumps({'status': payload['status'], 'windows': len(results),
                      'swaps': len(swap_examples), 'valid': n, 'stats': stats}, indent=2))

if __name__ == '__main__':
    main()
