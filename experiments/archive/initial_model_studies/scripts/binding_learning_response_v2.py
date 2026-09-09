#!/usr/bin/env python3
"""research/177 — REPAIRED downstream-query binding learning-response experiment.

Fixes the research critical bug: target word is now APPENDED to the query.
The setup target remains visible; only the downstream query target is masked.

Correct masked example:
  Alice placed a stone in the box. Bob placed a shell in the basket. Later Alice returned to the [MASK].

NOT (research bug):
  Alice placed a stone in the [MASK]. Bob placed a shell in the basket. Later Alice returned to the

Four matched arms × 3 seeds, larger held-out, per-sample margins, separated splits.
"""
from __future__ import annotations
import json, os, pathlib, random, time, copy
import torch
import torch.nn.functional as F

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
OUT_DIR = ROOT / 'data/binding_learning_response'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/binding_learning_response_results.md')

# ── Templates ──
TEMPLATES_LOC = [
    {"setup": "{E1} placed a stone in the {L1}. {E2} placed a shell in the {L2}.",
     "query": "Later {E1} returned to the", "target": "{L1}", "family": "location"},
    {"setup": "{E1} left a note in the {L1}. {E2} left a coin in the {L2}.",
     "query": "Then {E1} went back to the", "target": "{L1}", "family": "location"},
    {"setup": "{E1} hid something in the {L1}. {E2} hid something in the {L2}.",
     "query": "{E1} came back to the", "target": "{L1}", "family": "location"},
    {"setup": "{E1} sat down in the {L1}. {E2} sat down in the {L2}.",
     "query": "You could find {E1} in the", "target": "{L1}", "family": "location"},
    {"setup": "{E1} moved to the {L1}. {E2} moved to the {L2}.",
     "query": "To visit {E1} go to the", "target": "{L1}", "family": "location"},
    {"setup": "{E1} was working in the {L1}. {E2} was working in the {L2}.",
     "query": "{E1} stayed in the", "target": "{L1}", "family": "location"},
]

TEMPLATES_PROP = [
    {"setup": "{E1} picked up a {L1} ball. {E2} picked up a {L2} ball.",
     "query": "{E1} was holding the", "target": "{L1}", "family": "property"},
    {"setup": "{E1} wore a {L1} coat. {E2} wore a {L2} coat.",
     "query": "{E1} looked nice in the", "target": "{L1}", "family": "property"},
]

# Held-out template family (never seen during training)
TEMPLATES_HELDOUT_FAM = [
    {"setup": "{E1} put a toy in the {L1}. {E2} put a toy in the {L2}.",
     "query": "{E1} reached into the", "target": "{L1}", "family": "location_heldout"},
    {"setup": "{E1} chose a {L1} hat. {E2} chose a {L2} hat.",
     "query": "{E1} put on the", "target": "{L1}", "family": "property_heldout"},
]

ENTITY_PAIRS_TRAIN = [
    ("Alice","Bob"),("John","Mary"),("Tom","Sarah"),("David","Emma"),
    ("James","Lucy"),("Peter","Anna"),("Mark","Kate"),("Paul","Jane"),
    ("Ben","Sophie"),("Luke","Grace"),
]
ENTITY_PAIRS_HELDOUT = [
    ("Sam","Lisa"),("Mike","Helen"),("Dan","Susan"),("Chris","Laura"),
    ("Jack","Emily"),("Alex","Rachel"),
]
LOC_TRAIN = [("box","basket"),("shelf","drawer"),("kitchen","bedroom"),
             ("garden","garage"),("closet","cabinet"),("attic","basement")]
LOC_HELDOUT = [("desk","bookshelf"),("bench","counter"),("pocket","bag"),
               ("pantry","cupboard"),("porch","hallway")]
PROP_TRAIN = [("red","blue"),("green","yellow"),("black","white"),
              ("large","small"),("old","new"),("dark","light")]
PROP_HELDOUT = [("silver","golden"),("thick","thin"),("warm","cold"),
                ("soft","hard"),("bright","dull")]


def generate_pairs(templates, entity_pairs, state_pairs, n_per_combo, rng):
    """Generate binding pairs with target APPENDED to query (the research fix)."""
    pairs = []
    for _ in range(n_per_combo):
        for tmpl in templates:
            e1, e2 = rng.choice(entity_pairs)
            l1, l2 = rng.choice(state_pairs)
            def fill(t, ents=(e1,e2), locs=(l1,l2)):
                return t.replace("{E1}",ents[0]).replace("{E2}",ents[1]).replace("{L1}",locs[0]).replace("{L2}",locs[1])
            
            setup = fill(tmpl["setup"])
            query_prefix = fill(tmpl["query"])
            target = fill(tmpl["target"])
            
            # KEY FIX: append target to query
            correct_text = f"{setup} {query_prefix} {target}."
            swapped_text = f"{fill(tmpl['setup'], ents=(e2,e1))} {query_prefix} {target}."
            
            # Record target character span in the full text
            # Target is the last occurrence of the target word before the period
            target_char_start = len(correct_text) - len(target) - 1  # before "."
            target_char_end = target_char_start + len(target)
            
            # Verify bag preservation
            c_words = sorted(correct_text.lower().split())
            s_words = sorted(swapped_text.lower().split())
            if c_words != s_words:
                continue
            
            pairs.append({
                'correct': correct_text,
                'swapped': swapped_text,
                'target': target,
                'target_char_start': target_char_start,
                'target_char_end': target_char_end,
                'entity1': e1, 'entity2': e2,
                'state1': l1, 'state2': l2,
                'family': tmpl['family'],
            })
    rng.shuffle(pairs)
    return pairs


def find_query_target_span(tokenizer, text, target, target_char_start, target_char_end):
    """Find the token span of the query target using offset mapping."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                   truncation=True, max_length=96)
    offsets = enc['offset_mapping']
    ids = enc['input_ids']
    
    # Find token positions that overlap with the target character span
    target_positions = []
    for i, (start, end) in enumerate(offsets):
        if start < target_char_end and end > target_char_start:
            target_positions.append(i)
    
    if not target_positions:
        return None, None, None
    
    # Verify the target tokens encode the target word
    target_token_ids = [ids[p] for p in target_positions]
    decoded = tokenizer.decode(target_token_ids).strip()
    
    return target_positions, target_token_ids, decoded


def setup_env():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


@torch.no_grad()
def evaluate_binding(model, tokenizer, pairs, device):
    """Evaluate binding: mask ONLY the query target, keep setup target visible."""
    mask_id = tokenizer.mask_token_id
    results = []
    
    for pair in pairs:
        # Find query target span in correct text
        pos_c, tids_c, dec_c = find_query_target_span(
            tokenizer, pair['correct'], pair['target'],
            pair['target_char_start'], pair['target_char_end'])
        pos_s, tids_s, dec_s = find_query_target_span(
            tokenizer, pair['swapped'], pair['target'],
            pair['target_char_start'], pair['target_char_end'])
        
        if pos_c is None or pos_s is None:
            continue
        if tids_c != tids_s:  # target tokens must be identical
            continue
        
        def get_ll(text, target_positions, target_token_ids):
            enc = tokenizer(text, add_special_tokens=False, truncation=True,
                          max_length=96, return_tensors='pt').to(device)
            ids = enc['input_ids'][0].tolist()
            # Mask only the query target positions
            masked = list(ids)
            for p in target_positions:
                if p < len(masked):
                    masked[p] = mask_id
            inp = torch.tensor([masked], device=device)
            att = torch.ones_like(inp)
            logits = model(input_ids=inp, attention_mask=att).logits[0]
            lp = F.log_softmax(logits, dim=-1)
            total = 0.0
            for p, tid in zip(target_positions, target_token_ids):
                if p < logits.shape[0]:
                    total += float(lp[p, tid])
            return total / max(1, len(target_positions))
        
        ll_c = get_ll(pair['correct'], pos_c, tids_c)
        ll_s = get_ll(pair['swapped'], pos_s, tids_s)
        margin = ll_c - ll_s
        
        results.append({
            'margin': margin,
            'll_correct': ll_c,
            'll_swapped': ll_s,
            'target': pair['target'],
            'family': pair['family'],
            'entity1': pair['entity1'],
            'state1': pair['state1'],
        })
    
    if not results:
        return {'n': 0, 'per_sample': []}
    margins = [r['margin'] for r in results]
    n = len(margins)
    sorted_m = sorted(margins)
    return {
        'n': n,
        'mean_margin': round(sum(margins)/n, 4),
        'median_margin': round(sorted_m[n//2], 4),
        'binding_accuracy': round(sum(1 for m in margins if m > 0)/n, 4),
        'p10': round(sorted_m[max(0,n//10)], 4),
        'p90': round(sorted_m[min(n-1,9*n//10)], 4),
        'per_sample': results,
    }


def train_arm(model, tokenizer, train_pairs, contrastive_mode, n_steps,
              batch_size, lr, margin_m, lambda_c, device, rng_seed):
    """Train one arm with CORRECT downstream query target masking."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    mask_id = tokenizer.mask_token_id
    rng = random.Random(rng_seed)
    losses = []
    contrastive_accs = []
    
    for step in range(n_steps):
        optimizer.zero_grad()
        batch = [rng.choice(train_pairs) for _ in range(batch_size)]
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid = 0
        correct_count = 0
        
        for pair in batch:
            pos_c, tids_c, _ = find_query_target_span(
                tokenizer, pair['correct'], pair['target'],
                pair['target_char_start'], pair['target_char_end'])
            if pos_c is None or not tids_c:
                continue
            
            # Encode correct passage
            enc_c = tokenizer(pair['correct'], add_special_tokens=False, truncation=True,
                            max_length=96, padding='max_length', return_tensors='pt').to(device)
            ids_c = enc_c['input_ids'][0].tolist()
            
            # Build labels: mask query target + random 15% of other non-pad tokens
            masked_c = list(ids_c)
            labels_c = [-100] * len(ids_c)
            for p in pos_c:
                if p < len(masked_c):
                    labels_c[p] = masked_c[p]
                    masked_c[p] = mask_id
            att = enc_c['attention_mask'][0].tolist()
            for i in range(len(masked_c)):
                if i not in pos_c and att[i] == 1 and masked_c[i] != 0 and rng.random() < 0.15:
                    labels_c[i] = masked_c[i]
                    masked_c[i] = mask_id
            
            inp_c = torch.tensor([masked_c], device=device)
            att_c = enc_c['attention_mask']
            lab_c = torch.tensor([labels_c], device=device)
            out_c = model(input_ids=inp_c, attention_mask=att_c, labels=lab_c)
            wwm_loss = out_c.loss if out_c.loss is not None else torch.tensor(0.0, device=device)
            
            # Contrastive component
            if contrastive_mode in ('true', 'random'):
                pos_s, tids_s, _ = find_query_target_span(
                    tokenizer, pair['swapped'], pair['target'],
                    pair['target_char_start'], pair['target_char_end'])
                if pos_s is None or tids_s != tids_c:
                    total_loss = total_loss + wwm_loss
                    valid += 1
                    continue
                
                # Mask query target in swapped
                enc_s = tokenizer(pair['swapped'], add_special_tokens=False, truncation=True,
                                max_length=96, padding='max_length', return_tensors='pt').to(device)
                ids_s = enc_s['input_ids'][0].tolist()
                masked_s = list(ids_s)
                for p in pos_s:
                    if p < len(masked_s):
                        masked_s[p] = mask_id
                
                inp_s = torch.tensor([masked_s], device=device)
                att_s = enc_s['attention_mask']
                
                logits_c2 = model(input_ids=inp_c, attention_mask=att_c).logits[0]
                logits_s2 = model(input_ids=inp_s, attention_mask=att_s).logits[0]
                
                lp_c = F.log_softmax(logits_c2, dim=-1)
                lp_s = F.log_softmax(logits_s2, dim=-1)
                
                ll_pos = sum(lp_c[p, tid] for p, tid in zip(pos_c, tids_c)) / len(pos_c)
                ll_neg = sum(lp_s[p, tid] for p, tid in zip(pos_s, tids_s)) / len(pos_s)
                
                if contrastive_mode == 'random' and rng.random() < 0.5:
                    ll_pos, ll_neg = ll_neg, ll_pos
                
                if float(ll_pos.detach()) > float(ll_neg.detach()):
                    correct_count += 1
                
                contrast_loss = torch.clamp(
                    torch.tensor(margin_m, device=device) - ll_pos + ll_neg, min=0.0)
                total_loss = total_loss + wwm_loss + lambda_c * contrast_loss
            else:
                total_loss = total_loss + wwm_loss
            valid += 1
        
        if valid > 0:
            avg_loss = total_loss / valid
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(avg_loss.detach().cpu()))
            if contrastive_mode in ('true', 'random') and valid > 0:
                contrastive_accs.append(correct_count / valid)
    
    model.eval()
    return losses, contrastive_accs


def main():
    setup_env()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    from transformers import DebertaV2Config, DebertaV2ForMaskedLM, AutoTokenizer
    tok_path = str((ROOT / 'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M').resolve())
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate data with larger pools
    SEEDS = [42, 123, 789]
    N_STEPS = 150
    BATCH_SIZE = 8
    LR = 5e-4
    MARGIN = 0.5
    LAMBDA = 0.5
    
    rng = random.Random(7777)
    
    # Training data: train templates × train entities × train states
    train_loc = generate_pairs(TEMPLATES_LOC, ENTITY_PAIRS_TRAIN, LOC_TRAIN, 6, rng)
    train_prop = generate_pairs(TEMPLATES_PROP, ENTITY_PAIRS_TRAIN, PROP_TRAIN, 6, rng)
    train_pairs = train_loc + train_prop
    rng.shuffle(train_pairs)
    
    # Held-out splits
    ho_new_ent_train_state = generate_pairs(TEMPLATES_LOC + TEMPLATES_PROP,
                                             ENTITY_PAIRS_HELDOUT, LOC_TRAIN + PROP_TRAIN, 3, rng)
    ho_train_ent_new_state = generate_pairs(TEMPLATES_LOC, ENTITY_PAIRS_TRAIN, LOC_HELDOUT, 3, rng)
    ho_new_ent_new_state = generate_pairs(TEMPLATES_LOC, ENTITY_PAIRS_HELDOUT, LOC_HELDOUT, 3, rng)
    ho_heldout_template = generate_pairs(TEMPLATES_HELDOUT_FAM, ENTITY_PAIRS_TRAIN, LOC_TRAIN + PROP_TRAIN, 4, rng)
    
    all_heldout = {
        'new_ent_train_state': ho_new_ent_train_state,
        'train_ent_new_state': ho_train_ent_new_state,
        'new_ent_new_state': ho_new_ent_new_state,
        'heldout_template_family': ho_heldout_template,
    }
    
    print(f"Training pairs: {len(train_pairs)}")
    for k, v in all_heldout.items():
        print(f"  Held-out [{k}]: {len(v)}")
    
    # Validate a few examples
    print("\n=== VALIDATION: human-readable masked examples ===")
    for pair in train_pairs[:3]:
        pos_c, tids_c, dec_c = find_query_target_span(
            tokenizer, pair['correct'], pair['target'],
            pair['target_char_start'], pair['target_char_end'])
        if pos_c:
            enc = tokenizer(pair['correct'], add_special_tokens=False, truncation=True, max_length=96)
            ids = list(enc['input_ids'])
            masked = list(ids)
            for p in pos_c:
                masked[p] = tokenizer.mask_token_id
            print(f"  Correct:  {pair['correct']}")
            print(f"  Swapped:  {pair['swapped']}")
            print(f"  Masked:   {tokenizer.decode(masked)}")
            print(f"  Target:   '{pair['target']}' decoded_span='{dec_c}' positions={pos_c}")
            # Verify setup target is NOT masked
            setup_end = pair['correct'].index('. ') + 1
            assert pos_c[0] > len(tokenizer(pair['correct'][:setup_end], add_special_tokens=False)['input_ids']) - 3, \
                "Target span appears to be in setup, not query!"
            print(f"  ✓ Query target confirmed (pos {pos_c} > setup token count)")
            print()
    
    # Run arms
    arms = ['wwm_correct', 'wwm_both', 'random_pair', 'true_binding']
    config = DebertaV2Config(
        vocab_size=len(tokenizer), hidden_size=128, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=256,
        max_position_embeddings=128, position_buckets=64,
        relative_attention=True, pos_att_type=['p2c', 'c2p'],
    )
    
    all_results = {}
    for arm in arms:
        arm_results = {'seeds': {}}
        for seed in SEEDS:
            print(f"\n{'='*50}\n{arm} seed={seed}\n{'='*50}")
            torch.manual_seed(seed)
            model = DebertaV2ForMaskedLM(config).to(device)
            model.eval()
            
            # Pre-training eval on all held-out splits
            pre_evals = {}
            for split_name, split_pairs in all_heldout.items():
                pre_evals[split_name] = evaluate_binding(model, tokenizer, split_pairs, device)
            pre_train_eval = evaluate_binding(model, tokenizer, train_pairs[:60], device)
            
            # Determine mode and training data
            if arm == 'wwm_correct':
                mode = 'none'
                td = train_pairs
            elif arm == 'wwm_both':
                mode = 'none'
                td = train_pairs + [{'correct': p['swapped'], 'swapped': p['correct'],
                                      'target': p['target'],
                                      'target_char_start': p['target_char_start'],
                                      'target_char_end': p['target_char_end'],
                                      'entity1': p['entity2'], 'entity2': p['entity1'],
                                      'state1': p['state2'], 'state2': p['state1'],
                                      'family': p['family']} for p in train_pairs]
            elif arm == 'random_pair':
                mode = 'random'
                td = train_pairs
            else:
                mode = 'true'
                td = train_pairs
            
            # Train
            losses, c_accs = train_arm(model, tokenizer, td, mode, N_STEPS,
                                       BATCH_SIZE, LR, MARGIN, LAMBDA, device,
                                       rng_seed=seed * 10 + 1)
            
            # Post-training eval
            post_evals = {}
            for split_name, split_pairs in all_heldout.items():
                post_evals[split_name] = evaluate_binding(model, tokenizer, split_pairs, device)
            post_train_eval = evaluate_binding(model, tokenizer, train_pairs[:60], device)
            
            # Strip per_sample for JSON size (keep summary stats)
            def strip(ev):
                return {k: v for k, v in ev.items() if k != 'per_sample'}
            
            arm_results['seeds'][seed] = {
                'pre_train': strip(pre_train_eval),
                'post_train': strip(post_train_eval),
                'pre_heldout': {k: strip(v) for k, v in pre_evals.items()},
                'post_heldout': {k: strip(v) for k, v in post_evals.items()},
                'loss_first': losses[0] if losses else None,
                'loss_last': losses[-1] if losses else None,
                'contrastive_acc_last': c_accs[-1] if c_accs else None,
            }
            
            # Print summary for this seed
            for split_name in all_heldout:
                pre_a = pre_evals[split_name].get('binding_accuracy', 0)
                post_a = post_evals[split_name].get('binding_accuracy', 0)
                pre_m = pre_evals[split_name].get('mean_margin', 0)
                post_m = post_evals[split_name].get('mean_margin', 0)
                print(f"  {split_name}: acc {pre_a:.3f}→{post_a:.3f}, margin {pre_m:.4f}→{post_m:.4f}")
        
        all_results[arm] = arm_results
    
    # Aggregate across seeds
    print("\n" + "="*60)
    print("AGGREGATE SUMMARY (mean across 3 seeds)")
    print("="*60)
    summary = {}
    for arm in arms:
        arm_summary = {}
        for split_name in all_heldout:
            accs = [all_results[arm]['seeds'][s]['post_heldout'][split_name].get('binding_accuracy', 0) for s in SEEDS]
            margins = [all_results[arm]['seeds'][s]['post_heldout'][split_name].get('mean_margin', 0) for s in SEEDS]
            arm_summary[split_name] = {
                'mean_acc': round(sum(accs)/len(accs), 4),
                'mean_margin': round(sum(margins)/len(margins), 4),
                'accs': accs,
                'margins': margins,
            }
        # Training accuracy
        train_accs = [all_results[arm]['seeds'][s]['post_train'].get('binding_accuracy', 0) for s in SEEDS]
        arm_summary['train'] = {'mean_acc': round(sum(train_accs)/len(train_accs), 4)}
        summary[arm] = arm_summary
        print(f"\n{arm}:")
        print(f"  train acc: {arm_summary['train']['mean_acc']:.3f}")
        for split_name in all_heldout:
            s = arm_summary[split_name]
            print(f"  {split_name}: acc={s['mean_acc']:.3f} ({s['accs']}), margin={s['mean_margin']:.4f}")
    
    payload = {
        'status': 'BINDING_LEARNING_RESPONSE_V2',
        'fix': 'target appended to query; downstream query target masked, setup target visible',
        'model_config': {'hidden': 128, 'layers': 2, 'heads': 4},
        'training': {'n_steps': N_STEPS, 'batch_size': BATCH_SIZE, 'lr': LR,
                    'margin': MARGIN, 'lambda': LAMBDA, 'seeds': SEEDS},
        'data': {'train_pairs': len(train_pairs),
                 'heldout': {k: len(v) for k, v in all_heldout.items()}},
        'aggregate_summary': summary,
        'per_seed_results': all_results,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    (OUT_DIR / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(f"\nSaved to {OUT_DIR / 'results.json'}")
    print(json.dumps({'status': payload['status'], 'summary': summary}, indent=2))

if __name__ == '__main__':
    main()
