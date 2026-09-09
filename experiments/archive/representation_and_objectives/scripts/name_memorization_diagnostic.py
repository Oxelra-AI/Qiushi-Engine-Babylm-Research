#!/usr/bin/env python3
"""research name-memorization diagnostic.

Critical question: is the below-chance probe accuracy from predicate 
role-coordinate inversion, or from name-winner memorization across 
the paired-world family structure?

The paired-world design ensures:
  - Context 1: entity A wins
  - Context 2: entity B wins (role reversal)
  
If the model memorizes "A is the winner" from anchor-context training,
it will SYSTEMATICALLY predict wrong on probe-context evaluation 
(where B wins), producing below-chance accuracy.

This diagnostic tests:
1. Per-family: does the probe context always have the opposite winner?
2. Name isolation: split families so anchor and probe groups share NO names
3. Compare within-family (shared names) vs cross-family (new names) probe accuracy

If name memorization drives the below-chance: within-family probe ~0.30, cross-family ~0.50
If predicate inversion drives it: both ~0.30
"""

import json, random, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE = Path("experiments/archive/representation_and_objectives")
DATA_DIR = WORKSPACE / "data/name_memorization_diagnostic"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PILOT_DIR = WORKSPACE / "data/paired_world_pilot"

ANCHOR_TEMPLATES = {1, 2, 3, 6, 7, 8, 11, 12, 13, 14}
PROBE_TEMPLATES  = {4, 5, 9, 10, 15}

MAX_LEN = 80; EMB_DIM = 48; HIDDEN_DIM = 48; EPOCHS = 25; BATCH_SIZE = 32; LR = 0.002
N_SEEDS = 3

def load_train_families():
    p = PILOT_DIR / "families_train.jsonl"
    return [json.loads(line) for line in p.read_text().strip().split('\n')]

def classify_tid(tid):
    if tid in ANCHOR_TEMPLATES: return 'anchor'
    if tid in PROBE_TEMPLATES: return 'probe'
    return 'held'

def get_nli_rows(fam, context_key):
    """Extract NLI rows for one context of a family."""
    cx = fam[context_key]
    pa, pb = fam['participant_a'], fam['participant_b']
    text = cx['text_score_ablated']
    wl = cx['winner_label']
    rows = []
    for hyp, hd in [(f"{pa} defeated {pb}.", 'AB'), (f"{pb} defeated {pa}.", 'BA')]:
        label = 1 if (hd=='AB' and wl=='A') or (hd=='BA' and wl=='B') else 0
        rows.append({'tid': cx['template_id'], 'text': f"{text} [SEP] {hyp}", 'label': label,
                     'family_id': fam['family_id'], 'cx': context_key,
                     'winner': cx['winner_label']})
    return rows

def build_vocab(texts):
    wc = Counter()
    for t in texts:
        for w in t.lower().split(): wc[w] += 1
    v = {'<PAD>': 0, '<UNK>': 1}
    for w in wc: v[w] = len(v)
    return v

def encode(text, vocab):
    words = text.lower().split()[:MAX_LEN]
    return [vocab.get(w, 1) for w in words] + [0]*(MAX_LEN-len(words))

class BiGRU(nn.Module):
    def __init__(self, vs):
        super().__init__()
        self.emb = nn.Embedding(vs, EMB_DIM, padding_idx=0)
        self.gru = nn.GRU(EMB_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(HIDDEN_DIM*2, 1)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM),
                                  nn.ReLU(), nn.Dropout(0.1), nn.Linear(HIDDEN_DIM, 1))
    def forward(self, x):
        mask = (x!=0).float()
        out, _ = self.gru(self.emb(x))
        s = self.attn(out).squeeze(-1).masked_fill(mask==0, -1e9)
        w = torch.softmax(s, dim=-1)
        return self.head((out*w.unsqueeze(-1)).sum(1)).squeeze(-1)

def train_model(model, ids, labels, epochs=EPOCHS):
    tx = torch.tensor(ids, dtype=torch.long)
    ty = torch.tensor(labels, dtype=torch.float)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH_SIZE, shuffle=True)
    best_acc, best_state = 0, None
    for ep in range(epochs):
        model.train()
        c = t = 0
        for xb, yb in loader:
            F.binary_cross_entropy_with_logits(model(xb), yb).backward()
            opt.step(); opt.zero_grad()
            c += ((model(xb)>0).float()==yb).sum().item(); t += len(yb)
        acc = c/t
        if acc > best_acc:
            best_acc = acc
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
        if best_acc >= 0.999: break
    if best_state: model.load_state_dict(best_state)
    return best_acc

def eval_acc(model, ids, labels):
    if not ids: return float('nan'), 0
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(ids, dtype=torch.long))
        preds = (logits > 0).float()
        gt = torch.tensor(labels, dtype=torch.float)
        return (preds==gt).float().mean().item(), len(labels)

def main():
    t0 = time.time()
    families = load_train_families()
    print(f"Train families: {len(families)}")
    
    # ── 1. Check paired-world structure ──
    print("\n=== Family structure analysis ===")
    reversal_count = 0
    same_count = 0
    for fam in families:
        w1 = fam['context1']['winner_label']
        w2 = fam['context2']['winner_label']
        if w1 != w2: reversal_count += 1
        else: same_count += 1
    print(f"Role reversal families: {reversal_count}/{len(families)} ({reversal_count/len(families):.1%})")
    print(f"Same winner families: {same_count}/{len(families)}")
    
    # ── 2. Classify families by template usage ──
    fam_types = {'both_anchor': [], 'mixed': [], 'both_probe': []}
    for fam in families:
        g1 = classify_tid(fam['context1']['template_id'])
        g2 = classify_tid(fam['context2']['template_id'])
        if g1 == 'anchor' and g2 == 'anchor':
            fam_types['both_anchor'].append(fam)
        elif g1 == 'probe' and g2 == 'probe':
            fam_types['both_probe'].append(fam)
        else:
            fam_types['mixed'].append(fam)
    
    print(f"\nBoth anchor: {len(fam_types['both_anchor'])}")
    print(f"Mixed (anchor+probe): {len(fam_types['mixed'])}")
    print(f"Both probe: {len(fam_types['both_probe'])}")
    
    # ── 3. Name-isolated split ──
    # Split families into TRAIN and NAMEHOLD groups with zero name overlap
    all_names = set()
    train_fams = []
    namehold_fams = []
    
    # First pass: collect name usage
    name_to_fams = defaultdict(list)
    for fam in families:
        for name in [fam['participant_a'], fam['participant_b']]:
            name_to_fams[name].append(fam['family_id'])
    
    # Sort families by ID for determinism, then greedily assign
    random.seed(42)
    shuffled = list(families)
    random.shuffle(shuffled)
    
    train_names = set()
    namehold_names = set()
    
    for fam in shuffled:
        names = {fam['participant_a'], fam['participant_b']}
        # Check overlap with existing groups
        overlaps_train = bool(names & train_names)
        overlaps_namehold = bool(names & namehold_names)
        
        if not overlaps_namehold and len(train_fams) < 300:
            train_fams.append(fam)
            train_names |= names
        elif not overlaps_train:
            namehold_fams.append(fam)
            namehold_names |= names
        elif not overlaps_namehold:
            train_fams.append(fam)
            train_names |= names
        # else: skip (names overlap both groups)
    
    overlap = train_names & namehold_names
    print(f"\nName-isolated split: train={len(train_fams)}, namehold={len(namehold_fams)}, name_overlap={len(overlap)}")
    
    # ── 4. Collect rows ──
    all_texts = []
    
    # Train rows: only anchor-template contexts from train families
    train_rows = []
    for fam in train_fams:
        for ck in ['context1', 'context2']:
            if classify_tid(fam[ck]['template_id']) == 'anchor':
                train_rows.extend(get_nli_rows(fam, ck))
    for r in train_rows: all_texts.append(r['text'])
    
    # WITHIN-family probe rows: probe-template contexts from TRAIN families (shared names)
    within_probe_rows = []
    for fam in train_fams:
        for ck in ['context1', 'context2']:
            if classify_tid(fam[ck]['template_id']) == 'probe':
                within_probe_rows.extend(get_nli_rows(fam, ck))
    for r in within_probe_rows: all_texts.append(r['text'])
    
    # CROSS-family probe rows: ALL template contexts from NAMEHOLD families (new names)
    cross_anchor_rows = []
    cross_probe_rows = []
    for fam in namehold_fams:
        for ck in ['context1', 'context2']:
            rows = get_nli_rows(fam, ck)
            for r in rows: all_texts.append(r['text'])
            if classify_tid(fam[ck]['template_id']) == 'anchor':
                cross_anchor_rows.extend(rows)
            elif classify_tid(fam[ck]['template_id']) == 'probe':
                cross_probe_rows.extend(rows)
    
    # WITHIN-family anchor rows from other contexts (sanity check: trained template, same names)
    within_anchor_rows = []
    for fam in train_fams:
        for ck in ['context1', 'context2']:
            if classify_tid(fam[ck]['template_id']) == 'anchor':
                # Only include rows NOT in training (from the other context)
                pass  # These ARE in training, so we need a separate check
    
    # Actually: within_anchor_unseen = anchor rows from contexts NOT in training
    # But all anchor contexts from train families ARE in training
    # So "within anchor" eval = training accuracy
    
    vocab = build_vocab(all_texts)
    vs = len(vocab)
    print(f"Vocab: {vs}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Within-family probe rows: {len(within_probe_rows)}")
    print(f"Cross-family anchor rows: {len(cross_anchor_rows)}")
    print(f"Cross-family probe rows: {len(cross_probe_rows)}")
    
    # Encode
    tr_ids = [encode(r['text'], vocab) for r in train_rows]
    tr_lab = [r['label'] for r in train_rows]
    wp_ids = [encode(r['text'], vocab) for r in within_probe_rows]
    wp_lab = [r['label'] for r in within_probe_rows]
    ca_ids = [encode(r['text'], vocab) for r in cross_anchor_rows]
    ca_lab = [r['label'] for r in cross_anchor_rows]
    cp_ids = [encode(r['text'], vocab) for r in cross_probe_rows]
    cp_lab = [r['label'] for r in cross_probe_rows]
    
    # ── 5. Train and evaluate ──
    results = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = BiGRU(vs)
        train_acc = train_model(model, tr_ids, tr_lab)
        
        within_probe_acc, n_wp = eval_acc(model, wp_ids, wp_lab)
        cross_anchor_acc, n_ca = eval_acc(model, ca_ids, ca_lab)
        cross_probe_acc, n_cp = eval_acc(model, cp_ids, cp_lab)
        
        r = {'seed': seed, 'train_acc': round(train_acc, 4),
             'within_probe': round(within_probe_acc, 4),
             'cross_anchor': round(cross_anchor_acc, 4),
             'cross_probe': round(cross_probe_acc, 4)}
        results.append(r)
        print(json.dumps(r), flush=True)
    
    # ── 6. Summary ──
    print(f"\n{'='*70}")
    print(f"NAME-MEMORIZATION DIAGNOSTIC")
    print(f"{'='*70}")
    print(f"  Train rows: {len(train_rows)} (anchor templates, train families)")
    print(f"  Name overlap between groups: {len(overlap)}")
    
    metrics = {}
    for key in ['train_acc', 'within_probe', 'cross_anchor', 'cross_probe']:
        vals = [r[key] for r in results]
        m, s = np.mean(vals), np.std(vals)
        metrics[key] = {'mean': round(m, 4), 'std': round(s, 4)}
    
    print(f"\n{'metric':<20s} {'mean':>8s} {'std':>8s} {'interpretation'}")
    print(f"-"*70)
    print(f"{'train_acc':<20s} {metrics['train_acc']['mean']:>8.4f} {metrics['train_acc']['std']:>8.4f}  model masters anchor training")
    print(f"{'within_probe':<20s} {metrics['within_probe']['mean']:>8.4f} {metrics['within_probe']['std']:>8.4f}  same names, reversed roles, probe templates")
    print(f"{'cross_anchor':<20s} {metrics['cross_anchor']['mean']:>8.4f} {metrics['cross_anchor']['std']:>8.4f}  NEW names, anchor templates (predicate transfer)")
    print(f"{'cross_probe':<20s} {metrics['cross_probe']['mean']:>8.4f} {metrics['cross_probe']['std']:>8.4f}  NEW names, probe templates (full transfer)")
    
    within_vs_cross = metrics['within_probe']['mean'] - metrics['cross_probe']['mean']
    anchor_vs_probe = metrics['cross_anchor']['mean'] - metrics['cross_probe']['mean']
    print(f"\n  Within-family vs cross-family probe gap: {within_vs_cross:+.4f}")
    print(f"  Cross-family anchor vs probe gap:        {anchor_vs_probe:+.4f}")
    
    print(f"\nINTERPRETATION:")
    if metrics['within_probe']['mean'] < 0.45 and metrics['cross_probe']['mean'] > 0.45:
        print("  >>> Name memorization dominates: below-chance is from shared names, not predicate inversion")
    elif metrics['within_probe']['mean'] < 0.45 and metrics['cross_probe']['mean'] < 0.45:
        print("  >>> Predicate inversion confirmed: below-chance persists with new names")
    elif metrics['cross_anchor']['mean'] > metrics['cross_probe']['mean'] + 0.05:
        print("  >>> Predicate anchoring effect: anchor templates transfer better than probe templates")
    else:
        print("  >>> No clear predicate anchoring: anchor and probe templates similar on new names")
    
    elapsed = time.time() - t0
    output = {
        'status': 'NAME_MEMORIZATION_DIAGNOSTIC',
        'elapsed': round(elapsed, 1),
        'family_split': {
            'train_fams': len(train_fams), 'namehold_fams': len(namehold_fams),
            'name_overlap': len(overlap),
        },
        'row_counts': {
            'train': len(train_rows), 'within_probe': len(within_probe_rows),
            'cross_anchor': len(cross_anchor_rows), 'cross_probe': len(cross_probe_rows),
        },
        'metrics': metrics,
        'raw': results,
    }
    out = DATA_DIR / 'name_memorization_diagnostic.json'
    out.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nElapsed: {elapsed:.1f}s. Saved: {out}")

if __name__ == '__main__':
    main()
