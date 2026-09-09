#!/usr/bin/env python3
"""research — Broader binding-substrate scan and probe.

Goes beyond research by:
1. Using adjacent sentence pairs/triples from the FULL official corpus.
2. Including pronoun/deictic chains, lowercase nominal entities, aliases.
3. Using bag-preserving entity→predicate swap construction.
4. Running a target-identical likelihood probe on the protected 100M model:
   mask a STATE/RELATION token and compare correct binding vs. swapped binding
   context (same words, same bag, only assignment changed).

Design:
  For each qualifying pair of sentences:
    - Detect ≥2 distinct referents with distinct predicates/locations/states.
    - Construct the bag-preserving swap by exchanging predicate assignments.
    - The TARGET is a state/relation/location token (not the entity name).
    - Both conditions contain exactly the same word multiset.
  Probe (using protected 100M DeBERTa):
    - Mask the target token(s).
    - Measure target log-likelihood under correct binding vs swapped binding.
    - If correct > swapped: binding-dependent signal exists in the corpus and
      the model already uses it to some degree.
"""
from __future__ import annotations
import collections, json, os, pathlib, random, re, sys, time
from typing import Any
import torch
import torch.nn.functional as F

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
CORPUS_DIR = ROOT / 'training/runs/babylm_pilot_dense_v0_1M/raw_dataset'
MODEL_DIR = (ROOT / 'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M').resolve()
OUT = ROOT / 'data/broader_binding_substrate_probe.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/broader_binding_substrate_probe.md')
MAX_SEQ = 256
SEED = 167
MAX_CANDIDATE_PAIRS = 8000   # scan budget
MAX_PROBE_EXAMPLES = 300     # probe budget

# ──── Entity and predicate detection ────

CAP_RE = re.compile(r'^[A-Z][a-z]{2,}$')
STOP_ENT = {'The','And','But','Or','If','When','While','He','She','It','They',
    'We','You','His','Her','Their','This','That','These','Those','So','As','At',
    'In','On','Of','To','For','With','My','Your','Its','What','Who','Where','How',
    'Why','Then','Now','Here','There','After','Before','Not','No','Yes','Well','Oh',
    'Mr','Mrs','Miss','Dr','Sir','Just','Even','Still','Yet','Once','Also','Very',
    'About','Because','However','Although','Since','Already','Many','Some','All',
    'Each','Every','Another','Other'}

# Lowercase nominal entity patterns
NOMINAL_RE = re.compile(r'\b(the|a|an) ((?:red|blue|green|big|small|old|young|little|other|first|second|black|white|new) )?(boy|girl|man|woman|dog|cat|ball|cup|box|bag|hat|car|book|toy|baby|child|mother|father|king|queen|prince|princess|teacher|doctor|bird|fish|bear|rabbit|fox|mouse|lion|tiger|monkey|elephant)\b', re.I)

SPATIAL_PREPS = {'in','on','at','under','over','above','below','inside','outside',
    'near','between','behind','beside','around','into','onto','from','to','toward',
    'towards','beneath','next'}

LOCATION_WORDS = {'box','table','shelf','room','house','garden','kitchen','school',
    'store','park','forest','car','bag','basket','pocket','drawer','cupboard','bed',
    'chair','floor','ground','tree','river','lake','street','city','town','country',
    'home','office','hospital','church','library','museum','farm','field','mountain',
    'cave','boat','ship','train','bus','island','village','castle','palace'}

STATE_VERBS = {'put','puts','placed','place','moved','move','took','take','brought',
    'bring','gave','give','sent','send','left','leave','dropped','drop','picked',
    'carried','carry','held','hold','kept','keep','hid','hide','found','find',
    'set','went','go','came','come','lived','live','returned','return','sat','sit',
    'stood','stand','lay','hung','wore','wear','threw','throw','pushed','pulled'}

def find_entities_in_sent(words: list[str]) -> list[tuple[int, str, str]]:
    """Return (position, surface, type) for entities: cap names, nominal phrases."""
    entities = []
    for i, w in enumerate(words):
        clean = re.sub(r'[^A-Za-z]', '', w)
        if i > 0 and CAP_RE.match(clean) and clean not in STOP_ENT:
            entities.append((i, clean, 'cap'))
    # Nominal entities from the original sentence text
    text = ' '.join(words)
    for m in NOMINAL_RE.finditer(text):
        # Find word position
        start_char = m.start()
        word_pos = len(text[:start_char].split())
        entities.append((word_pos, m.group(0).lower().strip(), 'nominal'))
    return entities

def find_predicate(words: list[str], ent_pos: int) -> dict | None:
    """Find a spatial/state predicate associated with entity at ent_pos."""
    lower = [w.lower().strip('.,!?;:\'"()[]') for w in words]
    n = len(words)
    # Look for verb + prep + location/object in a window after entity
    for i in range(ent_pos + 1, min(n, ent_pos + 6)):
        if lower[i] in STATE_VERBS or lower[i] in ('is','was','are','were','lives','lived','stays','stayed'):
            for j in range(i + 1, min(n, i + 4)):
                if lower[j] in SPATIAL_PREPS:
                    # Collect object words after prep
                    obj_words = []
                    obj_positions = []
                    for k in range(j + 1, min(n, j + 4)):
                        lw = lower[k]
                        if lw in SPATIAL_PREPS or lw in ('.','!','?',',','and','but','or','then','so'):
                            break
                        if len(lw) >= 2 and lw not in ('the','a','an','his','her','its','their','my','your'):
                            obj_words.append(lw)
                            obj_positions.append(k)
                    if obj_words and any(w in LOCATION_WORDS or len(w) >= 3 for w in obj_words):
                        return {
                            'verb': lower[i], 'verb_pos': i,
                            'prep': lower[j], 'prep_pos': j,
                            'object': ' '.join(obj_words),
                            'object_words': obj_words,
                            'object_positions': obj_positions,
                        }
                    break
            break
    return None

def extract_sentence_pairs(corpus_dir: pathlib.Path, max_pairs: int) -> list[dict]:
    """Extract adjacent sentence pairs from the full official corpus."""
    rng = random.Random(SEED)
    pairs = []
    for path in sorted(corpus_dir.glob('*.train.txt')):
        prev_sent = None
        with path.open('r', encoding='utf-8', errors='replace') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    prev_sent = None
                    continue
                # Split into sentences (simple heuristic: period/question/exclamation)
                sents = re.split(r'(?<=[.!?])\s+', stripped)
                for s in sents:
                    words = s.split()
                    if len(words) < 5:
                        prev_sent = None
                        continue
                    if prev_sent is not None:
                        pairs.append({
                            'sent1_words': prev_sent,
                            'sent2_words': words,
                            'source': path.name,
                        })
                    prev_sent = words
                    if len(pairs) >= max_pairs * 4:  # oversample, filter later
                        break
            if len(pairs) >= max_pairs * 4:
                break
    rng.shuffle(pairs)
    return pairs[:max_pairs * 2]

def find_binding_pair(pair: dict) -> dict | None:
    """Check if adjacent sentences have ≥2 entities with distinct swappable predicates."""
    combined = pair['sent1_words'] + pair['sent2_words']
    s1_len = len(pair['sent1_words'])
    
    # Find all entities across both sentences
    ents1 = find_entities_in_sent(pair['sent1_words'])
    ents2 = [(pos + s1_len, surf, typ) for pos, surf, typ in find_entities_in_sent(pair['sent2_words'])]
    all_ents = ents1 + ents2
    
    # Group by surface (deduplicate)
    by_surface: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for pos, surf, typ in all_ents:
        by_surface[surf].append((pos, typ))
    
    # Find entities with predicates
    ent_preds = []
    seen_objects = set()
    for surf, occurrences in by_surface.items():
        first_pos = occurrences[0][0]
        pred = find_predicate(combined, first_pos)
        if pred and pred['object'] not in seen_objects:
            ent_preds.append({'entity': surf, 'entity_pos': first_pos, 'predicate': pred})
            seen_objects.add(pred['object'])
    
    # Need ≥2 entities with different predicates for a swappable binding
    if len(ent_preds) < 2:
        return None
    
    # Take first two as the binding pair
    b1, b2 = ent_preds[0], ent_preds[1]
    
    # Construct the bag-preserving swap
    # Original text has: E1→P1, E2→P2
    # Swapped text has: E1→P2, E2→P1 (same words, different assignment)
    # Target: P1's object words when referenced after E1
    return {
        'combined_words': combined,
        'combined_text': ' '.join(combined),
        'binding1': b1,
        'binding2': b2,
        'source': pair['source'],
        's1_len': s1_len,
    }

def construct_swap(binding_result: dict) -> dict | None:
    """Construct a bag-preserving swapped version and identify target tokens."""
    combined = list(binding_result['combined_words'])
    b1 = binding_result['binding1']
    b2 = binding_result['binding2']
    p1 = b1['predicate']
    p2 = b2['predicate']
    
    # The target is p1's object words (what we'll mask and try to predict)
    target_positions = p1['object_positions']
    target_words = p1['object_words']
    
    if not target_positions or not target_words:
        return None
    
    # Construct swapped version: swap the object words between b1 and b2
    # We need both objects to have the same token count for clean swap
    obj1_positions = p1['object_positions']
    obj2_positions = p2['object_positions']
    
    if len(obj1_positions) != len(obj2_positions):
        # Can still swap if we're careful about word count
        # Use the simpler approach: swap entity names if same length
        return None  # for now, require same-length objects
    
    # Build swapped text: exchange object words at their positions
    swapped = list(combined)
    for i, (pos1, pos2) in enumerate(zip(obj1_positions, obj2_positions)):
        if pos1 < len(swapped) and pos2 < len(swapped):
            swapped[pos1], swapped[pos2] = swapped[pos2], swapped[pos1]
    
    # Verify bag preservation: same multiset of words
    if sorted(w.lower() for w in combined) != sorted(w.lower() for w in swapped):
        return None  # bag not preserved, skip
    
    return {
        'correct_text': ' '.join(combined),
        'swapped_text': ' '.join(swapped),
        'target_words': target_words,
        'target_positions': target_positions,
        'entity1': b1['entity'],
        'entity2': b2['entity'],
        'pred1_object': p1['object'],
        'pred2_object': p2['object'],
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
    
    # Tokenize both versions
    enc_c = tokenizer(correct_text, add_special_tokens=False, truncation=True, max_length=MAX_SEQ)
    enc_s = tokenizer(swapped_text, add_special_tokens=False, truncation=True, max_length=MAX_SEQ)
    
    ids_c = enc_c['input_ids']
    ids_s = enc_s['input_ids']
    
    if len(ids_c) != len(ids_s):
        return None  # length mismatch after tokenization, skip
    
    # Find target token positions by matching the target word tokens
    target_tok_ids = tokenizer(' '.join(target_words), add_special_tokens=False)['input_ids']
    if not target_tok_ids:
        return None
    
    # Find positions of target tokens in correct text
    target_positions = []
    for i in range(len(ids_c) - len(target_tok_ids) + 1):
        if ids_c[i:i+len(target_tok_ids)] == target_tok_ids:
            target_positions = list(range(i, i + len(target_tok_ids)))
            break
    
    if not target_positions:
        return None
    
    # Mask target positions in both versions
    masked_c = list(ids_c)
    masked_s = list(ids_s)
    for p in target_positions:
        masked_c[p] = mask_id
        masked_s[p] = mask_id
    
    # Get likelihoods
    def get_target_ll(masked_ids, target_ids_at_pos):
        inp = torch.tensor([masked_ids], device=device)
        att = torch.ones_like(inp)
        logits = model(input_ids=inp, attention_mask=att).logits[0]
        lp = F.log_softmax(logits, dim=-1)
        total = 0.0
        for p, tid in zip(target_positions, target_ids_at_pos):
            if p < logits.shape[0]:
                total += float(lp[p, tid])
        return total / max(1, len(target_positions))
    
    # Target tokens are the SAME in both conditions (from the correct version)
    target_ids = [ids_c[p] for p in target_positions]
    
    ll_correct = get_target_ll(masked_c, target_ids)
    ll_swapped = get_target_ll(masked_s, target_ids)
    
    return {
        'll_correct': ll_correct,
        'll_swapped': ll_swapped,
        'delta': ll_correct - ll_swapped,
    }


def main():
    setup_env()
    t0 = time.time()
    rng = random.Random(SEED)
    
    print("Extracting adjacent sentence pairs from full corpus...")
    pairs = extract_sentence_pairs(CORPUS_DIR, MAX_CANDIDATE_PAIRS)
    print(f"  Extracted {len(pairs)} candidate pairs")
    
    print("Finding binding pairs with swappable predicates...")
    binding_results = []
    for pair in pairs:
        result = find_binding_pair(pair)
        if result:
            binding_results.append(result)
    print(f"  Found {len(binding_results)} binding pairs")
    
    print("Constructing bag-preserving swaps...")
    swap_examples = []
    for br in binding_results:
        swap = construct_swap(br)
        if swap:
            swap_examples.append(swap)
    print(f"  Constructed {len(swap_examples)} bag-preserving swap examples")
    
    if not swap_examples:
        print("WARNING: No valid swap examples found. Saving diagnostic output.")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            'status': 'NO_SWAP_EXAMPLES',
            'candidate_pairs': len(pairs),
            'binding_results': len(binding_results),
            'swap_examples': 0,
            'note': 'The bag-preserving same-length swap constraint was too strict. Relax or use different construction.',
            'binding_result_samples': [{'combined_text': br['combined_text'][:300],
                                         'b1_entity': br['binding1']['entity'],
                                         'b1_object': br['binding1']['predicate']['object'],
                                         'b2_entity': br['binding2']['entity'],
                                         'b2_object': br['binding2']['predicate']['object']}
                                        for br in binding_results[:20]],
        }, indent=2) + '\n')
        NOTE.write_text(f'# research binding probe\n\nNo valid bag-preserving swap examples found.\n'
                        f'Binding pairs detected: {len(binding_results)}\n'
                        f'Same-length swap constraint too strict. Relax the swap construction.\n')
        return
    
    # Run likelihood probe
    rng.shuffle(swap_examples)
    probe_subset = swap_examples[:MAX_PROBE_EXAMPLES]
    
    print(f"Loading model for probe ({len(probe_subset)} examples)...")
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(MODEL_DIR))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    print(f"  Model on {device}")
    
    results = []
    deltas = []
    for i, ex in enumerate(probe_subset):
        r = probe_binding(model, tokenizer, ex['correct_text'], ex['swapped_text'],
                         ex['target_words'], device)
        if r is not None:
            results.append({**ex, **r, 'correct_text': ex['correct_text'][:300],
                           'swapped_text': ex['swapped_text'][:300]})
            deltas.append(r['delta'])
        if (i + 1) % 50 == 0:
            print(f"  Probed {i+1}/{len(probe_subset)}, valid: {len(results)}")
    
    # Statistics
    n = len(deltas)
    if n > 0:
        mean_delta = sum(deltas) / n
        sorted_d = sorted(deltas)
        median_delta = sorted_d[n // 2]
        pos_frac = sum(1 for d in deltas if d > 0) / n
        stats = {'n': n, 'mean': round(mean_delta, 4), 'median': round(median_delta, 4),
                 'positive_fraction': round(pos_frac, 4),
                 'p10': round(sorted_d[max(0, n//10)], 4),
                 'p90': round(sorted_d[min(n-1, 9*n//10)], 4)}
    else:
        stats = {'n': 0}
    
    payload = {
        'status': 'BROADER_BINDING_SUBSTRATE_PROBE',
        'model': str(MODEL_DIR),
        'corpus_dir': str(CORPUS_DIR),
        'candidate_pairs_scanned': len(pairs),
        'binding_pairs_found': len(binding_results),
        'bag_preserving_swaps': len(swap_examples),
        'probed_examples': len(probe_subset),
        'valid_probe_results': n,
        'binding_dependent_target': True,
        'target_type': 'state/location/relation tokens (not entity names)',
        'bag_preserved': True,
        'correct_minus_swapped_stats': stats,
        'examples': results[:15],
        'elapsed_sec': time.time() - t0,
        'interpretation': (
            f'From {len(pairs)} adjacent sentence pairs, found {len(binding_results)} with ≥2 entities and '
            f'swappable predicates, of which {len(swap_examples)} have same-length objects for clean bag swap. '
            f'Probed {n} valid examples. Correct-minus-swapped target LL: mean {stats.get("mean")}, '
            f'positive fraction {stats.get("positive_fraction")}. '
            f'If positive fraction >0.6 and mean >0.1, the model uses binding for state/relation prediction.'
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    
    lines = [
        '# research — broader binding-substrate probe',
        '',
        f'Model: protected 100M DeBERTa. Corpus: full official BabyLM.',
        f'Adjacent sentence pairs scanned: {len(pairs)}',
        f'Binding pairs (≥2 entities, swappable predicates): {len(binding_results)}',
        f'Bag-preserving swap examples: {len(swap_examples)}',
        f'Valid probe results: {n}',
        '',
        '## Probe design',
        '',
        'Target: state/location/relation tokens (NOT entity names).',
        'Both conditions have identical word multiset (bag preserved).',
        'Only entity→predicate binding assignment is swapped.',
        '',
        '## Results',
        '',
        f'Correct-minus-swapped target LL:',
        f'  mean = {stats.get("mean")}',
        f'  median = {stats.get("median")}',
        f'  positive fraction = {stats.get("positive_fraction")}',
        f'  p10 = {stats.get("p10")}, p90 = {stats.get("p90")}',
        '',
        '## Interpretation',
        '',
        'If mean > 0 and positive fraction > 0.6: the protected model already uses',
        'entity-state binding for prediction → training objective can amplify this.',
        'If near zero: binding signal is real in corpus but model does not use it yet',
        '→ training must teach binding from scratch (harder).',
        'If negative or very sparse: official corpus does not support this mechanism.',
    ]
    NOTE.write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': payload['status'], 'binding_pairs': len(binding_results),
                      'swap_examples': len(swap_examples), 'valid_probes': n,
                      'stats': stats}, indent=2))

if __name__ == '__main__':
    main()
