#!/usr/bin/env python3
"""research — Binding-dependency probe on ideal procedural state-change text.

Mechanism-test requirement: before any data-route training, prove that target-token
prediction genuinely depends on correct entity→state propositions, not local word
frequency. Use bag-preserving, relation-swapped controls.

Design:
- Construct ~200 synthetic procedural passages with explicit entity→state bindings
  (2-3 entities, distinct locations/states, downstream targets).
- For each, construct a RELATION-SWAPPED control that preserves word surface and
  domain but shuffles entity→state assignment (e.g., swap which entity is in which
  location).
- Mask downstream state/location TARGET tokens.
- Measure target log-likelihood under correct vs. swapped context using the
  protected 100M DeBERTa model.
- If correct > swapped: binding-dependent signal exists even in the protected model
  → proposition-dense text would provide learnable binding supervision under WWM.
- If correct ≈ swapped: the model cannot use binding even in ideal text → the
  bottleneck is the objective/architecture, not data.

This tests the UPPER BOUND of what proposition-dense data can provide.
"""
from __future__ import annotations
import json, os, pathlib, random, time
import torch
import torch.nn.functional as F

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
MODEL_DIR = (ROOT / 'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M').resolve()
OUT = ROOT / 'data/ideal_binding_dependency_probe.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/ideal_binding_dependency_probe.md')
SEED = 173

# ──── Procedural passage templates ────
# Each template has:
# - Two entities with distinct states/locations
# - A downstream target that depends on the correct entity→state binding
# - The SAME words appear in both correct and swapped versions

TEMPLATES = [
    # Location tracking
    {"setup": "{E1} put the ball in the {L1}. {E2} put the cup in the {L2}.",
     "query": "Later, {E1} went back to get the ball from the",
     "target": "{L1}", "swap_target": "{L2}"},
    {"setup": "{E1} left the book on the {L1}. {E2} left the key on the {L2}.",
     "query": "{E1} returned for the book and looked on the",
     "target": "{L1}", "swap_target": "{L2}"},
    {"setup": "{E1} moved to the {L1}. {E2} moved to the {L2}.",
     "query": "To find {E1} you would go to the",
     "target": "{L1}", "swap_target": "{L2}"},
    {"setup": "{E1} hid the treasure in the {L1}. {E2} hid the map in the {L2}.",
     "query": "{E1} needed to retrieve the treasure from the",
     "target": "{L1}", "swap_target": "{L2}"},
    {"setup": "The {L1} contained {E1}'s coat. The {L2} contained {E2}'s hat.",
     "query": "{E1} opened the {L1} and took out the",
     "target": "coat", "swap_target": "hat"},
    # Possession/state
    {"setup": "{E1} bought a red car. {E2} bought a blue car.",
     "query": "{E1} drove away in the",
     "target": "red", "swap_target": "blue"},
    {"setup": "{E1} ordered coffee. {E2} ordered tea.",
     "query": "The waiter brought {E1} the",
     "target": "coffee", "swap_target": "tea"},
    {"setup": "{E1} wore a black jacket. {E2} wore a white jacket.",
     "query": "Everyone noticed {E1} in the",
     "target": "black", "swap_target": "white"},
    # Process/action state
    {"setup": "{E1} started cooking dinner. {E2} started cleaning the house.",
     "query": "An hour later {E1} finished",
     "target": "cooking", "swap_target": "cleaning"},
    {"setup": "{E1} planted roses in the garden. {E2} planted tulips in the garden.",
     "query": "In spring {E1}'s",
     "target": "roses", "swap_target": "tulips"},
    # Transfer
    {"setup": "{E1} gave the letter to {E2}. {E2} gave the package to {E1}.",
     "query": "{E2} now held the",
     "target": "letter", "swap_target": "package"},
    {"setup": "{E1} sent a message to the doctor. {E2} sent a message to the teacher.",
     "query": "The doctor received the message from",
     "target": "{E1}", "swap_target": "{E2}"},
]

ENTITY_PAIRS = [
    ("Alice", "Bob"), ("John", "Mary"), ("Tom", "Sarah"), ("David", "Emma"),
    ("James", "Lucy"), ("Peter", "Anna"), ("Mark", "Kate"), ("Paul", "Jane"),
    ("Sam", "Lisa"), ("Mike", "Helen"), ("Dan", "Susan"), ("Chris", "Laura"),
    ("Jack", "Emily"), ("Alex", "Rachel"), ("Ben", "Diana"), ("Luke", "Grace"),
]

LOCATION_PAIRS = [
    ("box", "basket"), ("shelf", "drawer"), ("table", "chair"),
    ("kitchen", "bedroom"), ("garden", "garage"), ("closet", "cabinet"),
    ("desk", "bookshelf"), ("attic", "basement"), ("bag", "pocket"),
    ("cupboard", "pantry"), ("bench", "counter"), ("floor", "roof"),
]

def generate_examples(n: int, seed: int) -> list[dict]:
    """Generate n procedural binding examples with correct and swapped versions."""
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        template = rng.choice(TEMPLATES)
        e1, e2 = rng.choice(ENTITY_PAIRS)
        l1, l2 = rng.choice(LOCATION_PAIRS)
        
        def fill(text, entities=(e1, e2), locs=(l1, l2)):
            return text.replace("{E1}", entities[0]).replace("{E2}", entities[1]) \
                       .replace("{L1}", locs[0]).replace("{L2}", locs[1])
        
        # Correct version: E1→L1, E2→L2
        correct_setup = fill(template["setup"])
        query = fill(template["query"])
        target = fill(template["target"])
        
        # Swapped version: swap entity assignments (E1↔E2 in setup only)
        # This preserves all words but changes which entity is associated with which state
        swapped_setup = fill(template["setup"], entities=(e2, e1))
        swapped_query = fill(template["query"])  # query still asks about E1
        swap_target = fill(template["swap_target"])
        
        correct_text = correct_setup + " " + query
        swapped_text = swapped_setup + " " + swapped_query
        
        # Verify bag preservation: same word multiset in setup+query
        # (query is identical; setup has same words with E1/E2 swapped in sentences)
        correct_words = sorted(correct_text.lower().split())
        swapped_words = sorted(swapped_text.lower().split())
        bag_preserved = (correct_words == swapped_words)
        
        examples.append({
            'correct_text': correct_text,
            'swapped_text': swapped_text,
            'target_word': target,
            'swap_target_word': swap_target,
            'bag_preserved': bag_preserved,
            'template_idx': TEMPLATES.index(template),
            'entities': (e1, e2),
            'locations': (l1, l2),
        })
    return examples

def setup_env():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

@torch.no_grad()
def probe_binding(model, tokenizer, correct_text: str, swapped_text: str,
                  target_word: str, device) -> dict | None:
    """Measure target token likelihood under correct vs swapped binding context."""
    mask_id = tokenizer.mask_token_id
    
    # Tokenize target word
    target_ids = tokenizer(target_word, add_special_tokens=False)['input_ids']
    if not target_ids:
        return None
    
    # Tokenize both versions
    enc_c = tokenizer(correct_text, add_special_tokens=False, truncation=True, max_length=256)
    enc_s = tokenizer(swapped_text, add_special_tokens=False, truncation=True, max_length=256)
    ids_c = enc_c['input_ids']
    ids_s = enc_s['input_ids']
    
    if len(ids_c) != len(ids_s):
        return None  # Length mismatch
    
    # Find target token positions in correct text (should be at the end)
    tgt_positions = []
    for i in range(len(ids_c) - len(target_ids) + 1):
        if ids_c[i:i+len(target_ids)] == target_ids:
            tgt_positions = list(range(i, i + len(target_ids)))
    
    if not tgt_positions:
        # Try with space prefix
        target_ids_sp = tokenizer(' ' + target_word, add_special_tokens=False)['input_ids']
        for i in range(len(ids_c) - len(target_ids_sp) + 1):
            if ids_c[i:i+len(target_ids_sp)] == target_ids_sp:
                tgt_positions = list(range(i, i + len(target_ids_sp)))
                target_ids = target_ids_sp
                break
    
    if not tgt_positions:
        return None
    
    # Use the LAST occurrence (query target)
    last_start = -1
    for i in range(len(ids_c) - len(target_ids), -1, -1):
        if ids_c[i:i+len(target_ids)] == target_ids:
            tgt_positions = list(range(i, i + len(target_ids)))
            break
    
    if not tgt_positions:
        return None
    
    # Mask target in both versions and measure likelihood
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
        for p, tid in zip(tgt_positions, target_ids):
            if p < logits.shape[0]:
                total += float(lp[p, tid])
        return total / max(1, len(tgt_positions))
    
    ll_correct = get_ll(masked_c)
    ll_swapped = get_ll(masked_s)
    
    return {
        'll_correct': round(ll_correct, 4),
        'll_swapped': round(ll_swapped, 4),
        'delta': round(ll_correct - ll_swapped, 4),
    }

def main():
    setup_env()
    t0 = time.time()
    
    print("Generating ideal procedural binding examples...")
    examples = generate_examples(200, SEED)
    bag_preserved_count = sum(1 for e in examples if e['bag_preserved'])
    print(f"  Generated {len(examples)} examples, {bag_preserved_count} bag-preserved")
    
    print("Loading protected 100M DeBERTa model...")
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(MODEL_DIR))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    print(f"  Model on {device}")
    
    print("Running binding-dependency probe...")
    results = []
    deltas = []
    for i, ex in enumerate(examples):
        if not ex['bag_preserved']:
            continue  # Skip non-bag-preserved examples
        r = probe_binding(model, tokenizer, ex['correct_text'], ex['swapped_text'],
                         ex['target_word'], device)
        if r is not None:
            results.append({
                'correct_text': ex['correct_text'],
                'swapped_text': ex['swapped_text'],
                'target_word': ex['target_word'],
                'template_idx': ex['template_idx'],
                **r
            })
            deltas.append(r['delta'])
        if (i + 1) % 50 == 0:
            print(f"  Probed {i+1}/{len(examples)}, valid: {len(results)}")
    
    n = len(deltas)
    if n > 0:
        sorted_d = sorted(deltas)
        stats = {
            'n': n,
            'mean': round(sum(deltas) / n, 4),
            'median': round(sorted_d[n // 2], 4),
            'positive_fraction': round(sum(1 for d in deltas if d > 0) / n, 4),
            'p10': round(sorted_d[max(0, n // 10)], 4),
            'p90': round(sorted_d[min(n - 1, 9 * n // 10)], 4),
            'max': round(max(deltas), 4),
            'min': round(min(deltas), 4),
        }
    else:
        stats = {'n': 0}
    
    payload = {
        'status': 'IDEAL_BINDING_DEPENDENCY_PROBE',
        'model': str(MODEL_DIR),
        'total_generated': len(examples),
        'bag_preserved': bag_preserved_count,
        'valid_probes': n,
        'target_type': 'state/location/property tokens (not entity names)',
        'bag_preserved_in_probes': True,
        'swap_method': 'entity-assignment swap in setup sentences (E1↔E2)',
        'correct_minus_swapped_stats': stats,
        'examples': results[:20],
        'elapsed_sec': round(time.time() - t0, 1),
        'interpretation': (
            f'Probed {n} ideal procedural binding examples on the protected 100M model. '
            f'Correct-minus-swapped mean = {stats.get("mean")}, positive fraction = {stats.get("positive_fraction")}. '
            f'If mean > 0.2 and positive fraction > 0.65: protected model already uses binding in clean text → '
            f'proposition-dense data would provide learnable binding training signal under WWM. '
            f'If near zero: model cannot use binding even in ideal text → objective/architecture is the bottleneck, not data.'
        ),
    }
    
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text('\n'.join([
        '# research — Ideal binding-dependency probe',
        '',
        '## Purpose',
        'Before committing to any proposition-dense data route, prove that target-token',
        'prediction genuinely depends on correct entity→state propositions. Tests the',
        'UPPER BOUND of what proposition-dense data can provide under plain WWM.',
        '',
        '## Design',
        '- ~200 synthetic procedural passages with maximally clear entity→state bindings',
        '- Bag-preserving entity-assignment swap controls (same words, same domain)',
        '- Target: downstream state/location/property tokens',
        '- Model: protected 100M DeBERTa (trained on official corpus only)',
        '',
        '## Results',
        f'- Valid probes: {n}',
        f'- Mean (correct - swapped): {stats.get("mean")}',
        f'- Median: {stats.get("median")}',
        f'- Positive fraction: {stats.get("positive_fraction")}',
        f'- p10, p90: {stats.get("p10")}, {stats.get("p90")}',
        '',
        '## Interpretation',
        'If positive: the model uses entity-state binding for prediction in clean text →',
        '  proposition-dense data provides learnable binding signal under WWM.',
        'If null: binding cannot be exploited even in ideal conditions → bottleneck is',
        '  objective/architecture, not data content.',
    ]) + '\n')
    
    print(json.dumps({
        'status': payload['status'],
        'valid_probes': n,
        'stats': stats,
    }, indent=2))

if __name__ == '__main__':
    main()
