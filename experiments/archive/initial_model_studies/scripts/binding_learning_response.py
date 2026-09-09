#!/usr/bin/env python3
"""research — Same-word-bag binding learning-response experiment.

Tests whether a loss that explicitly depends on correct entity→state assignment
can induce binding sensitivity in a small model. Uses four matched arms:
1. wwm_correct: WWM on correct passages only
2. wwm_both: WWM on correct + swapped passages (no binding labels)
3. random_pair: WWM + contrastive loss but random +/- orientation
4. true_binding: WWM + true contrastive loss (correct = positive)

The contrastive loss masks the downstream state/location target and pushes
p(target | correct) > p(target | swapped) with a margin.

Key design requirements (from research contract):
- No entity-name targets
- Target is state/location/property word
- Query does NOT contain an object word that locally determines the target
- Same word multiset in both versions
- Query references the entity directly ("Alice returned to the ___")
"""
from __future__ import annotations
import copy, json, os, pathlib, random, time
from dataclasses import dataclass, field
from typing import Any
import torch
import torch.nn.functional as F

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
OUT_DIR = ROOT / 'data/binding_learning_response'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/175_binding_learning_response.md')
SEED = 175

# ──── Template system ────

TEMPLATES = [
    # Pattern: E1 verb obj1 in L1. E2 verb obj2 in L2. Later E1 returned to the [L1].
    # Target: L1 (location). Query mentions entity only, not object.
    {"setup": "{E1} placed a stone in the {L1}. {E2} placed a shell in the {L2}.",
     "query": "Later {E1} returned to the", "target": "{L1}"},
    {"setup": "{E1} left a note in the {L1}. {E2} left a coin in the {L2}.",
     "query": "Then {E1} went back to the", "target": "{L1}"},
    {"setup": "{E1} hid something in the {L1}. {E2} hid something in the {L2}.",
     "query": "{E1} came back to the", "target": "{L1}"},
    {"setup": "{E1} sat down in the {L1}. {E2} sat down in the {L2}.",
     "query": "You could find {E1} in the", "target": "{L1}"},
    {"setup": "{E1} moved to the {L1}. {E2} moved to the {L2}.",
     "query": "To visit {E1} go to the", "target": "{L1}"},
    {"setup": "{E1} was working in the {L1}. {E2} was working in the {L2}.",
     "query": "{E1} stayed in the", "target": "{L1}"},
    # State/property patterns
    {"setup": "{E1} picked up a {L1} ball. {E2} picked up a {L2} ball.",
     "query": "{E1} was holding the", "target": "{L1}"},
    {"setup": "{E1} wore a {L1} coat. {E2} wore a {L2} coat.",
     "query": "{E1} looked nice in the", "target": "{L1}"},
]

ENTITY_PAIRS_TRAIN = [
    ("Alice", "Bob"), ("John", "Mary"), ("Tom", "Sarah"), ("David", "Emma"),
    ("James", "Lucy"), ("Peter", "Anna"), ("Mark", "Kate"), ("Paul", "Jane"),
]
ENTITY_PAIRS_HELDOUT = [
    ("Sam", "Lisa"), ("Mike", "Helen"), ("Dan", "Susan"), ("Chris", "Laura"),
]

LOCATION_PAIRS_TRAIN = [
    ("box", "basket"), ("shelf", "drawer"), ("kitchen", "bedroom"),
    ("garden", "garage"), ("closet", "cabinet"), ("attic", "basement"),
]
LOCATION_PAIRS_HELDOUT = [
    ("desk", "bookshelf"), ("bench", "counter"), ("pocket", "bag"),
]

# State/property pairs (for templates 6-7)
STATE_PAIRS_TRAIN = [
    ("red", "blue"), ("green", "yellow"), ("black", "white"),
    ("large", "small"), ("old", "new"), ("dark", "light"),
]
STATE_PAIRS_HELDOUT = [
    ("silver", "golden"), ("thick", "thin"), ("warm", "cold"),
]


def generate_pairs(templates, entity_pairs, location_pairs, n_per_combo, rng):
    """Generate binding pairs with correct and swapped versions."""
    pairs = []
    for _ in range(n_per_combo):
        for template in templates:
            e1, e2 = rng.choice(entity_pairs)
            if template.get("target") in ["{L1}"] and "ball" not in template["setup"]:
                l1, l2 = rng.choice(location_pairs)
            else:
                l1, l2 = rng.choice(STATE_PAIRS_TRAIN if "TRAIN" in str(location_pairs) else location_pairs)
            
            def fill(text, ents=(e1, e2), locs=(l1, l2)):
                return (text.replace("{E1}", ents[0]).replace("{E2}", ents[1])
                        .replace("{L1}", locs[0]).replace("{L2}", locs[1]))
            
            correct_text = fill(template["setup"]) + " " + fill(template["query"])
            swapped_text = fill(template["setup"], ents=(e2, e1)) + " " + fill(template["query"])
            target = fill(template["target"])
            
            # Verify bag preservation
            c_words = sorted(correct_text.lower().split())
            s_words = sorted(swapped_text.lower().split())
            if c_words != s_words:
                continue
            
            pairs.append({
                'correct': correct_text,
                'swapped': swapped_text,
                'target': target,
                'entity1': e1,
                'entity2': e2,
                'loc1': l1,
                'loc2': l2,
            })
    rng.shuffle(pairs)
    return pairs


def setup_env():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


@torch.no_grad()
def evaluate_binding(model, tokenizer, pairs, device, max_n=200):
    """Measure binding sensitivity: margin = log p(target | correct) - log p(target | swapped)."""
    mask_id = tokenizer.mask_token_id
    margins = []
    for pair in pairs[:max_n]:
        target_ids = tokenizer(pair['target'], add_special_tokens=False)['input_ids']
        if not target_ids:
            continue
        # Also try with space prefix
        target_ids_sp = tokenizer(' ' + pair['target'], add_special_tokens=False)['input_ids']
        
        def get_ll(text, tgt_ids):
            enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=96,
                           return_tensors='pt').to(device)
            ids = enc['input_ids'][0].tolist()
            # Find last occurrence of target
            pos = []
            for start in range(len(ids) - len(tgt_ids), -1, -1):
                if ids[start:start+len(tgt_ids)] == tgt_ids:
                    pos = list(range(start, start + len(tgt_ids)))
                    break
            if not pos:
                return None
            masked = list(ids)
            for p in pos:
                masked[p] = mask_id
            inp = torch.tensor([masked], device=device)
            att = torch.ones_like(inp)
            logits = model(input_ids=inp, attention_mask=att).logits[0]
            lp = F.log_softmax(logits, dim=-1)
            return sum(float(lp[p, tid]) for p, tid in zip(pos, tgt_ids)) / len(pos)
        
        # Try both tokenizations
        ll_c = get_ll(pair['correct'], target_ids)
        ll_s = get_ll(pair['swapped'], target_ids)
        if ll_c is None or ll_s is None:
            ll_c = get_ll(pair['correct'], target_ids_sp)
            ll_s = get_ll(pair['swapped'], target_ids_sp)
        if ll_c is not None and ll_s is not None:
            margins.append(ll_c - ll_s)
    
    if not margins:
        return {'n': 0}
    n = len(margins)
    sorted_m = sorted(margins)
    return {
        'n': n,
        'mean_margin': round(sum(margins) / n, 4),
        'median_margin': round(sorted_m[n // 2], 4),
        'binding_accuracy': round(sum(1 for m in margins if m > 0) / n, 4),
        'p10': round(sorted_m[max(0, n // 10)], 4),
        'p90': round(sorted_m[min(n - 1, 9 * n // 10)], 4),
    }


def train_arm(arm_name, model, tokenizer, train_pairs, contrastive_mode, n_steps, 
              batch_size, lr, margin_m, lambda_c, device, rng_seed):
    """Train one arm and return loss history."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    mask_id = tokenizer.mask_token_id
    rng = random.Random(rng_seed)
    losses = []
    
    for step in range(n_steps):
        optimizer.zero_grad()
        batch_pairs = [rng.choice(train_pairs) for _ in range(batch_size)]
        total_loss = torch.tensor(0.0, device=device)
        valid = 0
        
        for pair in batch_pairs:
            # Encode correct passage
            enc_c = tokenizer(pair['correct'], add_special_tokens=False, truncation=True,
                            max_length=96, padding='max_length', return_tensors='pt').to(device)
            ids_c = enc_c['input_ids'][0].tolist()
            
            # Find target and mask it
            target_ids = tokenizer(pair['target'], add_special_tokens=False)['input_ids']
            if not target_ids:
                target_ids = tokenizer(' ' + pair['target'], add_special_tokens=False)['input_ids']
            if not target_ids:
                continue
            
            # Find last occurrence
            pos = []
            for start in range(len(ids_c) - len(target_ids), -1, -1):
                if ids_c[start:start+len(target_ids)] == target_ids:
                    pos = list(range(start, start + len(target_ids)))
                    break
            if not pos:
                continue
            
            # WWM loss on correct passage (mask target + random ~15% of other tokens)
            masked_c = list(ids_c)
            labels_c = [-100] * len(ids_c)
            for p in pos:
                labels_c[p] = masked_c[p]
                masked_c[p] = mask_id
            # Also mask ~15% of non-target tokens for WWM signal
            att = enc_c['attention_mask'][0].tolist()
            for i in range(len(masked_c)):
                if i not in pos and att[i] == 1 and rng.random() < 0.15:
                    labels_c[i] = masked_c[i]
                    masked_c[i] = mask_id
            
            inp_c = torch.tensor([masked_c], device=device)
            att_c = enc_c['attention_mask']
            lab_c = torch.tensor([labels_c], device=device)
            out_c = model(input_ids=inp_c, attention_mask=att_c, labels=lab_c)
            wwm_loss = out_c.loss if out_c.loss is not None else torch.tensor(0.0, device=device)
            
            # Contrastive component
            if contrastive_mode in ('true', 'random'):
                enc_s = tokenizer(pair['swapped'], add_special_tokens=False, truncation=True,
                                max_length=96, padding='max_length', return_tensors='pt').to(device)
                ids_s = enc_s['input_ids'][0].tolist()
                
                # Find target in swapped
                pos_s = []
                for start in range(len(ids_s) - len(target_ids), -1, -1):
                    if ids_s[start:start+len(target_ids)] == target_ids:
                        pos_s = list(range(start, start + len(target_ids)))
                        break
                
                if pos_s:
                    # Mask target in swapped
                    masked_s = list(ids_s)
                    for p in pos_s:
                        masked_s[p] = mask_id
                    
                    inp_s = torch.tensor([masked_s], device=device)
                    att_s = enc_s['attention_mask']
                    logits_c = model(input_ids=inp_c, attention_mask=att_c).logits[0]
                    logits_s = model(input_ids=inp_s, attention_mask=att_s).logits[0]
                    
                    lp_c = F.log_softmax(logits_c, dim=-1)
                    lp_s = F.log_softmax(logits_s, dim=-1)
                    
                    # Keep as tensors for gradient flow
                    ll_pos = sum(lp_c[p, tid] for p, tid in zip(pos, target_ids)) / len(pos)
                    ll_neg = sum(lp_s[p, tid] for p, tid in zip(pos_s, target_ids)) / len(pos_s)
                    
                    if contrastive_mode == 'random':
                        # Random orientation: randomly swap which is positive
                        if rng.random() < 0.5:
                            ll_pos, ll_neg = ll_neg, ll_pos
                    
                    contrast_loss = torch.clamp(torch.tensor(margin_m, device=device) - ll_pos + ll_neg, min=0.0)
                    total_loss = total_loss + wwm_loss + lambda_c * contrast_loss
                else:
                    total_loss = total_loss + wwm_loss
            else:
                total_loss = total_loss + wwm_loss
            valid += 1
        
        if valid > 0:
            (total_loss / valid).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(total_loss.detach().cpu()) / valid)
    
    model.eval()
    return losses


def main():
    setup_env()
    t0 = time.time()
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print("Generating training and held-out pairs...")
    # Training: use train entities + locations, templates 0-5 (location) + 6-7 (state)
    loc_templates = TEMPLATES[:6]
    state_templates = TEMPLATES[6:]
    
    train_loc = generate_pairs(loc_templates, ENTITY_PAIRS_TRAIN, LOCATION_PAIRS_TRAIN, 8, rng)
    train_state = generate_pairs(state_templates, ENTITY_PAIRS_TRAIN, STATE_PAIRS_TRAIN, 8, rng)
    train_pairs = train_loc + train_state
    rng.shuffle(train_pairs)
    
    # Held-out: new entities + new locations (generalization test)
    heldout_loc = generate_pairs(loc_templates, ENTITY_PAIRS_HELDOUT, LOCATION_PAIRS_HELDOUT, 4, rng)
    heldout_state = generate_pairs(state_templates, ENTITY_PAIRS_HELDOUT, STATE_PAIRS_HELDOUT, 4, rng)
    heldout_pairs = heldout_loc + heldout_state
    rng.shuffle(heldout_pairs)
    
    # Also held-out with TRAINING entities but HELD-OUT locations (location generalization)
    heldout_loc_only = generate_pairs(loc_templates, ENTITY_PAIRS_TRAIN, LOCATION_PAIRS_HELDOUT, 4, rng)
    
    print(f"  Training pairs: {len(train_pairs)}")
    print(f"  Held-out pairs (new entities + locations): {len(heldout_pairs)}")
    print(f"  Held-out pairs (train entities, new locations): {len(heldout_loc_only)}")
    
    # Build small model
    print("Building small DeBERTa-v2 model...")
    from transformers import DebertaV2Config, DebertaV2ForMaskedLM, AutoTokenizer
    tok_path = str((ROOT / 'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M').resolve())
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    
    config = DebertaV2Config(
        vocab_size=len(tokenizer), hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=256,
        max_position_embeddings=128, position_buckets=64,
        relative_attention=True, pos_att_type=['p2c', 'c2p'],
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training params
    N_STEPS = 150
    BATCH_SIZE = 8
    LR = 5e-4
    MARGIN = 0.5
    LAMBDA = 0.5
    
    arms = ['wwm_correct', 'wwm_both', 'random_pair', 'true_binding']
    results = {}
    
    for arm in arms:
        print(f"\n{'='*60}")
        print(f"ARM: {arm}")
        print(f"{'='*60}")
        
        # Fresh model from same init
        torch.manual_seed(42)
        model = DebertaV2ForMaskedLM(config).to(device)
        model.eval()
        
        # Pre-training evaluation
        pre_train = evaluate_binding(model, tokenizer, train_pairs[:100], device)
        pre_heldout = evaluate_binding(model, tokenizer, heldout_pairs, device)
        pre_heldout_loc = evaluate_binding(model, tokenizer, heldout_loc_only, device)
        print(f"  Pre-train binding accuracy: train={pre_train.get('binding_accuracy')}, "
              f"heldout={pre_heldout.get('binding_accuracy')}, heldout_loc={pre_heldout_loc.get('binding_accuracy')}")
        
        # Determine contrastive mode
        if arm == 'wwm_correct':
            mode = 'none'
        elif arm == 'wwm_both':
            mode = 'none'  # sees both but no contrastive signal
        elif arm == 'random_pair':
            mode = 'random'
        elif arm == 'true_binding':
            mode = 'true'
        
        # For wwm_both: train on interleaved correct+swapped as separate examples
        if arm == 'wwm_both':
            both_pairs = []
            for p in train_pairs:
                both_pairs.append(p)
                # Add swapped as a separate "correct" example (no binding signal)
                both_pairs.append({**p, 'correct': p['swapped']})
            train_data = both_pairs
        else:
            train_data = train_pairs
        
        # Train
        losses = train_arm(arm, model, tokenizer, train_data, mode, N_STEPS,
                          BATCH_SIZE, LR, MARGIN, LAMBDA, device, rng_seed=SEED + hash(arm) % 1000)
        
        # Post-training evaluation
        model.eval()
        post_train = evaluate_binding(model, tokenizer, train_pairs[:100], device)
        post_heldout = evaluate_binding(model, tokenizer, heldout_pairs, device)
        post_heldout_loc = evaluate_binding(model, tokenizer, heldout_loc_only, device)
        
        print(f"  Post-train binding accuracy: train={post_train.get('binding_accuracy')}, "
              f"heldout={post_heldout.get('binding_accuracy')}, heldout_loc={post_heldout_loc.get('binding_accuracy')}")
        print(f"  Margin change (train): {pre_train.get('mean_margin', 0):.4f} → {post_train.get('mean_margin', 0):.4f}")
        print(f"  Margin change (heldout): {pre_heldout.get('mean_margin', 0):.4f} → {post_heldout.get('mean_margin', 0):.4f}")
        
        results[arm] = {
            'pre_train': pre_train,
            'pre_heldout': pre_heldout,
            'pre_heldout_loc_only': pre_heldout_loc,
            'post_train': post_train,
            'post_heldout': post_heldout,
            'post_heldout_loc_only': post_heldout_loc,
            'loss_first': losses[0] if losses else None,
            'loss_last': losses[-1] if losses else None,
            'n_steps': len(losses),
        }
    
    # Save results
    payload = {
        'status': 'BINDING_LEARNING_RESPONSE',
        'model_config': {'hidden': 128, 'layers': 2, 'heads': 4, 'intermediate': 256},
        'training': {'n_steps': N_STEPS, 'batch_size': BATCH_SIZE, 'lr': LR,
                    'margin': MARGIN, 'lambda': LAMBDA, 'init_seed': 42},
        'data': {'train_pairs': len(train_pairs), 'heldout_pairs': len(heldout_pairs),
                'heldout_loc_only': len(heldout_loc_only)},
        'results': results,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    (OUT_DIR / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Binding learning response")
    print("=" * 60)
    print(f"{'arm':<15} {'pre_margin':<12} {'post_margin':<12} {'pre_acc':<10} {'post_acc':<10} {'Δ_margin':<10}")
    for arm in arms:
        r = results[arm]
        pm = r['pre_heldout'].get('mean_margin', 0)
        qm = r['post_heldout'].get('mean_margin', 0)
        pa = r['pre_heldout'].get('binding_accuracy', 0)
        qa = r['post_heldout'].get('binding_accuracy', 0)
        print(f"{arm:<15} {pm:<12.4f} {qm:<12.4f} {pa:<10.4f} {qa:<10.4f} {qm-pm:<10.4f}")
    
    print(json.dumps({'status': payload['status'], 'summary': {
        arm: {'pre_margin': results[arm]['pre_heldout'].get('mean_margin'),
              'post_margin': results[arm]['post_heldout'].get('mean_margin'),
              'pre_acc': results[arm]['pre_heldout'].get('binding_accuracy'),
              'post_acc': results[arm]['post_heldout'].get('binding_accuracy')}
        for arm in arms}}, indent=2))

if __name__ == '__main__':
    main()
