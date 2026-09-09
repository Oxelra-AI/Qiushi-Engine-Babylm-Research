#!/usr/bin/env python3
"""research: Raw-text anchor transfer experiment.

Tests whether correctly aligned anchor evidence produces disproportionate
transfer to held predicates in raw-text role parsing.

The factorized role-coordinate analysis (research) showed:
  - True anchors give perfect held-predicate transfer
  - Shuffled anchors give held-both=1.0 but mixed held=0.0
  - Exposure-only gives intermediate results

This script tests whether the same principle operates in raw natural text:
  - 10 anchor templates: always fully labeled in training
  - 5 probe templates: varying amounts of true/shuffled/exposure labels
  - 5 held templates: never in training (strongest transfer test)

Both single-context (no interference) and dual-context (interference) tasks.
"""

import json, hashlib, random, time, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE = Path("experiments/archive/representation_and_objectives")
DATA_DIR = WORKSPACE / "data/raw_text_anchor_transfer"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PILOT_DIR = WORKSPACE / "data/paired_world_pilot"

# Template split (balanced w/l/m in each group)
ANCHOR_TEMPLATES = {1, 2, 3, 6, 7, 8, 11, 12, 13, 14}   # 3w+3l+4m
PROBE_TEMPLATES  = {4, 5, 9, 10, 15}                       # 2w+2l+1m
HELD_TEMPLATES   = {16, 17, 18, 19, 20}                    # 2w+2l+1m

COVERAGE_LEVELS = [0, 5, 10, 20, 50, 100]  # percent of probe rows
ARMS = ['true', 'shuffled', 'exposure']
N_SEEDS = 5
DEVICE = 'cpu'

# Model hyperparameters
EMB_DIM = 48
HIDDEN_DIM = 48
MAX_LEN = 80
EPOCHS = 40
BATCH_SIZE = 32
LR = 0.002
PATIENCE = 8  # early stop patience

# ──────────────────────────────────────────────
# 1. Data loading
# ──────────────────────────────────────────────

def load_families():
    """Load all families from train and held splits."""
    families = []
    for split in ['train', 'held']:
        p = PILOT_DIR / f"families_{split}.jsonl"
        for line in p.read_text().strip().split('\n'):
            fam = json.loads(line)
            families.append(fam)
    return families

def extract_nli_event_rows(families):
    """Extract single-context NLI event rows from families."""
    rows = []
    for fam in families:
        fid = fam['family_id']
        fsplit = fam['family_split']
        pa = fam['participant_a']
        pb = fam['participant_b']
        
        for cx_key in ['context1', 'context2']:
            cx = fam[cx_key]
            tid = cx['template_id']
            tsplit = cx['template_split']
            text_ablated = cx['text_score_ablated']
            winner_label = cx['winner_label']  # 'A' or 'B'
            winner_name = cx['winner_name']
            loser_name = cx['loser_name']
            
            # Two hypothesis directions: "A defeated B" and "B defeated A"
            hyp_ab = f"{pa} defeated {pb}."
            hyp_ba = f"{pb} defeated {pa}."
            
            # Label depends on who won
            label_ab = 1 if winner_label == 'A' else 0
            label_ba = 1 if winner_label == 'B' else 0
            
            rows.append({
                'family_id': fid, 'family_split': fsplit,
                'context_key': cx_key, 'template_id': tid,
                'template_split': tsplit,
                'context_text': text_ablated,
                'hypothesis': hyp_ab, 'label': label_ab,
                'hyp_direction': 'AB',
                'winner_label': winner_label,
                'participant_a': pa, 'participant_b': pb,
            })
            rows.append({
                'family_id': fid, 'family_split': fsplit,
                'context_key': cx_key, 'template_id': tid,
                'template_split': tsplit,
                'context_text': text_ablated,
                'hypothesis': hyp_ba, 'label': label_ba,
                'hyp_direction': 'BA',
                'winner_label': winner_label,
                'participant_a': pa, 'participant_b': pb,
            })
    return rows

def extract_interference_rows(families):
    """Extract dual-context interference NLI rows from families."""
    rows = []
    for fam in families:
        fid = fam['family_id']
        fsplit = fam['family_split']
        pa = fam['participant_a']
        pb = fam['participant_b']
        
        c1 = fam['context1']
        c2 = fam['context2']
        t1 = c1['text_score_ablated']
        t2 = c2['text_score_ablated']
        tid1 = c1['template_id']
        tid2 = c2['template_id']
        w1 = c1['winner_label']
        w2 = c2['winner_label']
        
        # Determine template group for the family
        # Use the "harder" template (probe > anchor > held for classification)
        tids = {tid1, tid2}
        
        hyp_ab = f"{pa} defeated {pb}."
        hyp_ba = f"{pb} defeated {pa}."
        
        # For each queried context and each hypothesis direction
        for queried_cx, queried_tid, winner in [('context1', tid1, w1), ('context2', tid2, w2)]:
            for hyp, hyp_dir in [(hyp_ab, 'AB'), (hyp_ba, 'BA')]:
                label = 1 if (hyp_dir == 'AB' and winner == 'A') or (hyp_dir == 'BA' and winner == 'B') else 0
                
                # Input: both contexts + query indicator + hypothesis
                if queried_cx == 'context1':
                    input_text = f"Event 1: {t1} Event 2: {t2} [QUERY_EVENT_1] {hyp}"
                else:
                    input_text = f"Event 1: {t1} Event 2: {t2} [QUERY_EVENT_2] {hyp}"
                
                rows.append({
                    'family_id': fid, 'family_split': fsplit,
                    'queried_cx': queried_cx,
                    'template_id': queried_tid,
                    'template_ids': (tid1, tid2),
                    'input_text': input_text,
                    'label': label,
                    'hyp_direction': hyp_dir,
                    'winner_label': winner,
                    'participant_a': pa, 'participant_b': pb,
                })
    return rows

# ──────────────────────────────────────────────
# 2. Vocabulary and encoding
# ──────────────────────────────────────────────

def build_vocab(texts, min_freq=2):
    """Build word vocabulary from texts."""
    word_counts = Counter()
    for text in texts:
        for word in text.lower().split():
            word_counts[word] += 1
    
    vocab = {'<PAD>': 0, '<UNK>': 1, '<SEP>': 2}
    for word, count in word_counts.most_common():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode_text(text, vocab, max_len):
    """Encode text to integer indices."""
    words = text.lower().split()[:max_len]
    ids = [vocab.get(w, vocab['<UNK>']) for w in words]
    # Pad
    ids = ids + [vocab['<PAD>']] * (max_len - len(ids))
    return ids

# ──────────────────────────────────────────────
# 3. Model
# ──────────────────────────────────────────────

class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        mask = (x != 0).float()  # (B, L)
        emb = self.emb(x)       # (B, L, E)
        out, _ = self.gru(emb)  # (B, L, 2H)
        
        # Attention pooling
        scores = self.attn(out).squeeze(-1)   # (B, L)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)  # (B, L)
        pooled = (out * weights.unsqueeze(-1)).sum(dim=1)  # (B, 2H)
        
        logits = self.classifier(pooled).squeeze(-1)  # (B,)
        return logits

# ──────────────────────────────────────────────
# 4. Training and evaluation
# ──────────────────────────────────────────────

def train_and_eval(train_ids, train_labels, eval_sets, vocab_size, seed, max_len=MAX_LEN):
    """Train a BiGRU classifier and evaluate on multiple sets."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    train_x = torch.tensor(train_ids, dtype=torch.long)
    train_y = torch.tensor(train_labels, dtype=torch.float)
    
    model = BiGRUClassifier(vocab_size, EMB_DIM, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    best_train_acc = 0.0
    patience_counter = 0
    best_state = None
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct = 0
        total = 0
        for xb, yb in loader:
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = (logits > 0).float()
            correct += (preds == yb).sum().item()
            total += len(yb)
        
        train_acc = correct / total if total > 0 else 0
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE and best_train_acc > 0.6:
            break
        if best_train_acc >= 0.999:
            break
    
    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Evaluate
    model.eval()
    results = {'train_acc': best_train_acc, 'epochs': epoch}
    with torch.no_grad():
        for name, (eval_ids, eval_labels) in eval_sets.items():
            if len(eval_ids) == 0:
                results[name] = {'acc': float('nan'), 'n': 0}
                continue
            ex = torch.tensor(eval_ids, dtype=torch.long)
            ey = torch.tensor(eval_labels, dtype=torch.float)
            logits = model(ex)
            preds = (logits > 0).float()
            acc = (preds == ey).float().mean().item()
            results[name] = {'acc': acc, 'n': len(eval_labels)}
    
    return results

# ──────────────────────────────────────────────
# 5. Experiment runner
# ──────────────────────────────────────────────

def classify_template(tid):
    if tid in ANCHOR_TEMPLATES:
        return 'anchor'
    elif tid in PROBE_TEMPLATES:
        return 'probe'
    elif tid in HELD_TEMPLATES:
        return 'held'
    return 'unknown'

def run_single_context_experiment(nli_rows, vocab, max_len):
    """Run the single-context anchor transfer experiment."""
    print("\n=== SINGLE-CONTEXT EXPERIMENT ===")
    
    # Classify rows by template group and family split
    anchor_train_rows = [r for r in nli_rows 
                         if classify_template(r['template_id']) == 'anchor' 
                         and r['family_split'] == 'train']
    probe_train_rows = [r for r in nli_rows 
                        if classify_template(r['template_id']) == 'probe'
                        and r['family_split'] == 'train']
    
    # Evaluation sets (from held families)
    anchor_eval_rows = [r for r in nli_rows 
                        if classify_template(r['template_id']) == 'anchor'
                        and r['family_split'] == 'held']
    probe_eval_rows = [r for r in nli_rows 
                       if classify_template(r['template_id']) == 'probe'
                       and r['family_split'] == 'held']
    held_eval_rows = [r for r in nli_rows 
                      if classify_template(r['template_id']) == 'held']
    
    # Also create probe eval from train split (for transfer within training families)
    probe_train_eval = [r for r in probe_train_rows]  # will be used as eval when not in training
    
    print(f"Anchor train rows: {len(anchor_train_rows)}")
    print(f"Probe train rows:  {len(probe_train_rows)}")
    print(f"Anchor eval rows:  {len(anchor_eval_rows)}")
    print(f"Probe eval rows:   {len(probe_eval_rows)}")
    print(f"Held eval rows:    {len(held_eval_rows)}")
    
    # Encode anchor training rows (always in training)
    anchor_ids = [encode_text(f"{r['context_text']} [SEP] {r['hypothesis']}", vocab, max_len) for r in anchor_train_rows]
    anchor_labels = [r['label'] for r in anchor_train_rows]
    
    # Encode probe rows
    probe_ids = [encode_text(f"{r['context_text']} [SEP] {r['hypothesis']}", vocab, max_len) for r in probe_train_rows]
    probe_labels = [r['label'] for r in probe_train_rows]
    
    # Encode eval sets
    def encode_eval(rows):
        if not rows:
            return [], []
        ids = [encode_text(f"{r['context_text']} [SEP] {r['hypothesis']}", vocab, max_len) for r in rows]
        labels = [r['label'] for r in rows]
        return ids, labels
    
    eval_anchor = encode_eval(anchor_eval_rows)
    eval_probe_held = encode_eval(probe_eval_rows)
    eval_held = encode_eval(held_eval_rows)
    eval_probe_train = encode_eval(probe_train_rows)
    
    # Also per-template eval for probe templates
    probe_template_evals = {}
    for tid in sorted(PROBE_TEMPLATES):
        tid_rows = [r for r in probe_train_rows if r['template_id'] == tid]
        if tid_rows:
            probe_template_evals[f'probe_T{tid:02d}'] = encode_eval(tid_rows)
    
    vocab_size = len(vocab)
    results_all = []
    
    for coverage_pct in COVERAGE_LEVELS:
        for arm in ARMS:
            if coverage_pct == 0 and arm != 'true':
                continue  # 0% is the same for all arms
            
            arm_label = f"{arm}_{coverage_pct}pct" if coverage_pct > 0 else "zero_coverage"
            
            for seed in range(N_SEEDS):
                rng = random.Random(seed * 1000 + coverage_pct * 10 + hash(arm) % 100)
                
                # Select probe rows for this arm
                n_probe = int(len(probe_train_rows) * coverage_pct / 100)
                selected_indices = list(range(len(probe_train_rows)))
                rng.shuffle(selected_indices)
                selected_indices = selected_indices[:n_probe]
                
                # Construct probe training data
                probe_train_ids = [probe_ids[i] for i in selected_indices]
                
                if arm == 'true':
                    probe_train_labels = [probe_labels[i] for i in selected_indices]
                elif arm == 'shuffled':
                    # Shuffle labels: flip each label with 50% probability
                    probe_train_labels = [1 - probe_labels[i] if rng.random() < 0.5 else probe_labels[i] 
                                          for i in selected_indices]
                elif arm == 'exposure':
                    # Random 50/50 labels (no correlation with text)
                    probe_train_labels = [rng.randint(0, 1) for _ in selected_indices]
                
                # Combine anchor + selected probe rows
                all_train_ids = anchor_ids + probe_train_ids
                all_train_labels = anchor_labels + probe_train_labels
                
                # Evaluation: probe rows NOT selected for training
                not_selected = [i for i in range(len(probe_train_rows)) if i not in set(selected_indices)]
                eval_probe_unseen = ([probe_ids[i] for i in not_selected],
                                     [probe_labels[i] for i in not_selected])
                
                eval_sets = {
                    'anchor_held_fam': eval_anchor,
                    'probe_held_fam': eval_probe_held,
                    'held_templates': eval_held,
                    'probe_unseen_train': eval_probe_unseen,
                    'probe_all_train': eval_probe_train,
                }
                # Add per-template evals
                eval_sets.update(probe_template_evals)
                
                result = train_and_eval(all_train_ids, all_train_labels, eval_sets, 
                                        vocab_size, seed + coverage_pct * 17 + hash(arm) % 100)
                result['arm'] = arm_label
                result['arm_type'] = arm
                result['coverage_pct'] = coverage_pct
                result['seed'] = seed
                result['n_probe_train'] = n_probe
                result['n_total_train'] = len(all_train_ids)
                results_all.append(result)
                
                # Progress
                probe_unseen_acc = result.get('probe_unseen_train', {}).get('acc', float('nan'))
                held_acc = result.get('held_templates', {}).get('acc', float('nan'))
                print(json.dumps({
                    'arm': arm_label, 'seed': seed,
                    'train_acc': round(result['train_acc'], 4),
                    'probe_unseen': round(probe_unseen_acc, 4) if not np.isnan(probe_unseen_acc) else 'nan',
                    'held': round(held_acc, 4) if not np.isnan(held_acc) else 'nan',
                }), flush=True)
    
    return results_all

def run_interference_experiment(families, vocab, max_len):
    """Run the dual-context interference experiment."""
    print("\n=== DUAL-CONTEXT INTERFERENCE EXPERIMENT ===")
    
    # We need families where we know both templates
    # Extract interference rows from train-split families
    train_fams = [f for f in families if f['family_split'] == 'train']
    held_fams = [f for f in families if f['family_split'] == 'held']
    
    def make_interference_rows(fam_list):
        rows = []
        for fam in fam_list:
            fid = fam['family_id']
            pa = fam['participant_a']
            pb = fam['participant_b']
            c1 = fam['context1']
            c2 = fam['context2']
            t1 = c1['text_score_ablated']
            t2 = c2['text_score_ablated']
            tid1 = c1['template_id']
            tid2 = c2['template_id']
            w1 = c1['winner_label']
            w2 = c2['winner_label']
            
            hyp_ab = f"{pa} defeated {pb}."
            hyp_ba = f"{pb} defeated {pa}."
            
            for q_cx, q_tid, winner in [('c1', tid1, w1), ('c2', tid2, w2)]:
                for hyp, hyp_dir in [(hyp_ab, 'AB'), (hyp_ba, 'BA')]:
                    label = 1 if (hyp_dir == 'AB' and winner == 'A') or (hyp_dir == 'BA' and winner == 'B') else 0
                    
                    if q_cx == 'c1':
                        text = f"Event 1: {t1} Event 2: {t2} [QUERY_1] {hyp}"
                    else:
                        text = f"Event 1: {t1} Event 2: {t2} [QUERY_2] {hyp}"
                    
                    # Classify: template group of the QUERIED context
                    tgroup = classify_template(q_tid)
                    
                    rows.append({
                        'family_id': fid,
                        'template_id': q_tid,
                        'template_group': tgroup,
                        'text': text,
                        'label': label,
                        'q_cx': q_cx,
                        'winner': winner,
                        'tid_pair': (tid1, tid2),
                    })
        return rows
    
    train_rows = make_interference_rows(train_fams)
    held_rows = make_interference_rows(held_fams)
    
    # Classify rows
    anchor_train = [r for r in train_rows if r['template_group'] == 'anchor']
    probe_train = [r for r in train_rows if r['template_group'] == 'probe']
    
    anchor_eval = [r for r in held_rows if r['template_group'] == 'anchor']
    probe_eval = [r for r in held_rows if r['template_group'] == 'probe']
    held_eval = [r for r in held_rows if r['template_group'] == 'held']
    # Also include train-split held-template rows
    held_from_train = [r for r in train_rows if r['template_group'] == 'held']
    
    print(f"Interference anchor train: {len(anchor_train)}")
    print(f"Interference probe train:  {len(probe_train)}")
    print(f"Interference held eval:    {len(held_eval) + len(held_from_train)}")
    
    # Encode
    max_len_int = max_len * 3  # longer for dual context
    
    def enc_rows(rows):
        ids = [encode_text(r['text'], vocab, max_len_int) for r in rows]
        labels = [r['label'] for r in rows]
        return ids, labels
    
    anchor_ids, anchor_labels = enc_rows(anchor_train)
    probe_ids, probe_labels = enc_rows(probe_train)
    
    eval_anchor_enc = enc_rows(anchor_eval)
    eval_probe_enc = enc_rows(probe_eval)
    eval_held_enc = enc_rows(held_eval + held_from_train)
    eval_probe_train_enc = enc_rows(probe_train)
    
    vocab_size = len(vocab)
    results_all = []
    
    # Subset of coverage levels for interference (it's slower)
    int_coverage = [0, 10, 50, 100]
    
    for coverage_pct in int_coverage:
        for arm in ARMS:
            if coverage_pct == 0 and arm != 'true':
                continue
            
            arm_label = f"int_{arm}_{coverage_pct}pct" if coverage_pct > 0 else "int_zero"
            
            for seed in range(N_SEEDS):
                rng = random.Random(seed * 2000 + coverage_pct * 10 + hash(arm) % 100)
                
                n_probe = int(len(probe_train) * coverage_pct / 100)
                selected = list(range(len(probe_train)))
                rng.shuffle(selected)
                selected = selected[:n_probe]
                
                p_ids = [probe_ids[i] for i in selected]
                if arm == 'true':
                    p_labels = [probe_labels[i] for i in selected]
                elif arm == 'shuffled':
                    p_labels = [1 - probe_labels[i] if rng.random() < 0.5 else probe_labels[i] 
                               for i in selected]
                elif arm == 'exposure':
                    p_labels = [rng.randint(0, 1) for _ in selected]
                
                all_ids = anchor_ids + p_ids
                all_labels = anchor_labels + p_labels
                
                not_selected = [i for i in range(len(probe_train)) if i not in set(selected)]
                eval_probe_unseen = ([probe_ids[i] for i in not_selected],
                                     [probe_labels[i] for i in not_selected])
                
                eval_sets = {
                    'anchor_held': eval_anchor_enc,
                    'probe_held': eval_probe_enc,
                    'held_templates': eval_held_enc,
                    'probe_unseen': eval_probe_unseen,
                    'probe_all': eval_probe_train_enc,
                }
                
                result = train_and_eval(all_ids, all_labels, eval_sets,
                                        vocab_size, seed + coverage_pct * 19 + hash(arm) % 100,
                                        max_len=max_len_int)
                result['arm'] = arm_label
                result['arm_type'] = arm
                result['coverage_pct'] = coverage_pct
                result['seed'] = seed
                result['task'] = 'interference'
                results_all.append(result)
                
                probe_unseen_acc = result.get('probe_unseen', {}).get('acc', float('nan'))
                held_acc = result.get('held_templates', {}).get('acc', float('nan'))
                print(json.dumps({
                    'task': 'interference',
                    'arm': arm_label, 'seed': seed,
                    'train_acc': round(result['train_acc'], 4),
                    'probe_unseen': round(probe_unseen_acc, 4) if not np.isnan(probe_unseen_acc) else 'nan',
                    'held': round(held_acc, 4) if not np.isnan(held_acc) else 'nan',
                }), flush=True)
    
    return results_all

# ──────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────

def summarize_results(results, task_name):
    """Aggregate results by arm across seeds."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        arm = r['arm']
        for key in ['train_acc', 'epochs']:
            agg[arm][key].append(r[key])
        for eval_name in ['anchor_held_fam', 'probe_held_fam', 'held_templates', 
                          'probe_unseen_train', 'probe_all_train',
                          'anchor_held', 'probe_held', 'probe_unseen', 'probe_all']:
            if eval_name in r and isinstance(r[eval_name], dict):
                acc = r[eval_name].get('acc', float('nan'))
                if not np.isnan(acc):
                    agg[arm][f'{eval_name}_acc'].append(acc)
        # Per-template
        for tid in sorted(PROBE_TEMPLATES):
            key = f'probe_T{tid:02d}'
            if key in r and isinstance(r[key], dict):
                acc = r[key].get('acc', float('nan'))
                if not np.isnan(acc):
                    agg[arm][f'{key}_acc'].append(acc)
    
    summary = {}
    for arm, metrics in agg.items():
        summary[arm] = {}
        for metric, values in metrics.items():
            values = [v for v in values if not np.isnan(v)]
            if values:
                summary[arm][metric] = {
                    'mean': round(np.mean(values), 4),
                    'std': round(np.std(values), 4),
                    'n': len(values),
                }
    return summary

def main():
    t0 = time.time()
    
    print("Loading families...", flush=True)
    families = load_families()
    print(f"Total families: {len(families)}")
    
    # Extract NLI rows
    nli_rows = extract_nli_event_rows(families)
    print(f"Total single-context NLI event rows: {len(nli_rows)}")
    
    # Template distribution
    template_dist = Counter(r['template_id'] for r in nli_rows)
    for tid in sorted(template_dist):
        group = classify_template(tid)
        print(f"  T{tid:02d} ({group:6s}): {template_dist[tid]} rows")
    
    # Build vocabulary from ALL text (no information leak — just word set)
    all_texts = []
    for r in nli_rows:
        all_texts.append(f"{r['context_text']} [SEP] {r['hypothesis']}")
    # Also from interference format
    int_rows = extract_interference_rows(families)
    for r in int_rows:
        all_texts.append(r['input_text'])
    
    vocab = build_vocab(all_texts, min_freq=1)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Run single-context experiment
    sc_results = run_single_context_experiment(nli_rows, vocab, MAX_LEN)
    sc_summary = summarize_results(sc_results, 'single_context')
    
    # Run interference experiment
    int_results = run_interference_experiment(families, vocab, MAX_LEN)
    int_summary = summarize_results(int_results, 'interference')
    
    elapsed = time.time() - t0
    
    # ── Save results ──
    full_output = {
        'status': 'RAW_TEXT_ANCHOR_TRANSFER',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'config': {
            'anchor_templates': sorted(ANCHOR_TEMPLATES),
            'probe_templates': sorted(PROBE_TEMPLATES),
            'held_templates': sorted(HELD_TEMPLATES),
            'coverage_levels': COVERAGE_LEVELS,
            'arms': ARMS,
            'n_seeds': N_SEEDS,
            'emb_dim': EMB_DIM, 'hidden_dim': HIDDEN_DIM,
            'max_len': MAX_LEN, 'epochs': EPOCHS,
            'batch_size': BATCH_SIZE, 'lr': LR,
        },
        'single_context': {
            'n_rows': len(nli_rows),
            'summary': sc_summary,
            'raw_results': sc_results,
        },
        'interference': {
            'n_rows': len(int_rows),
            'summary': int_summary,
            'raw_results': int_results,
        },
        'elapsed_seconds': round(elapsed, 1),
    }
    
    out_json = DATA_DIR / 'anchor_transfer_summary.json'
    out_json.write_text(json.dumps(full_output, indent=2, default=str))
    
    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"SINGLE-CONTEXT ANCHOR TRANSFER (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")
    
    # Key comparison: probe accuracy by arm type and coverage
    print(f"\n{'arm':<25s} {'train':>8s} {'probe_unseen':>12s} {'probe_all':>10s} {'held':>8s}")
    print('-' * 65)
    for arm_name in sorted(sc_summary.keys()):
        s = sc_summary[arm_name]
        train = s.get('train_acc', {}).get('mean', float('nan'))
        pu = s.get('probe_unseen_train_acc', {}).get('mean', float('nan'))
        pa = s.get('probe_all_train_acc', {}).get('mean', float('nan'))
        h = s.get('held_templates_acc', {}).get('mean', float('nan'))
        print(f"{arm_name:<25s} {train:>8.4f} {pu:>12.4f} {pa:>10.4f} {h:>8.4f}")
    
    print(f"\n{'='*60}")
    print(f"INTERFERENCE ANCHOR TRANSFER")
    print(f"{'='*60}")
    print(f"\n{'arm':<25s} {'train':>8s} {'probe_unseen':>12s} {'probe_all':>10s} {'held':>8s}")
    print('-' * 65)
    for arm_name in sorted(int_summary.keys()):
        s = int_summary[arm_name]
        train = s.get('train_acc', {}).get('mean', float('nan'))
        pu = s.get('probe_unseen_acc', {}).get('mean', float('nan'))
        pa = s.get('probe_all_acc', {}).get('mean', float('nan'))
        h = s.get('held_templates_acc', {}).get('mean', float('nan'))
        print(f"{arm_name:<25s} {train:>8.4f} {pu:>12.4f} {pa:>10.4f} {h:>8.4f}")
    
    print(f"\nSaved to {out_json}")
    print(json.dumps({
        'status': full_output['status'],
        'elapsed': full_output['elapsed_seconds'],
        'summary_json': str(out_json),
    }, indent=2), flush=True)

if __name__ == '__main__':
    main()
