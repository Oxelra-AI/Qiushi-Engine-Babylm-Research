#!/usr/bin/env python3
"""research diagnostic: Per-template inversion analysis + BoW control.

The partial BiGRU results showed probe_unseen ~0.30 (below chance).
This diagnostic determines:
1. Which specific probe templates are inverted vs correct vs chance
2. Whether a BoW model also shows inversion (sequence-sensitivity check)
3. Whether the inversion is consistent across seeds (systematic vs random)

Uses a tiny model (16-dim) for speed.
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
DATA_DIR = WORKSPACE / "data/raw_text_anchor_transfer"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PILOT_DIR = WORKSPACE / "data/paired_world_pilot"

ANCHOR_TEMPLATES = {1, 2, 3, 6, 7, 8, 11, 12, 13, 14}
PROBE_TEMPLATES  = {4, 5, 9, 10, 15}
HELD_TEMPLATES   = {16, 17, 18, 19, 20}

# Template metadata
TEMPLATE_INFO = {
    1: ('w', 'defeated'), 2: ('w', 'beat'), 3: ('w', 'won against'),
    4: ('w', 'overcame'), 5: ('w', 'proved too strong for'),
    6: ('l', 'lost to'), 7: ('l', 'fell to'), 8: ('l', 'was defeated by'),
    9: ('l', 'was beaten by'), 10: ('l', 'was unable to overcome'),
    11: ('m', 'triumph over'), 12: ('m', 'emerged victorious over'),
    13: ('m', 'prevailed against'), 14: ('m', 'came out on top against'),
    15: ('m', 'victorious over'),
    16: ('w', 'edged out'), 17: ('l', 'succumbed to'),
    18: ('m', 'resulted in victory for'), 19: ('w', 'claimed the win against'),
    20: ('l', 'went down to'),
}

MAX_LEN = 80; N_SEEDS = 5

def load_nli():
    rows = []
    for split in ['train', 'held']:
        p = PILOT_DIR / f"families_{split}.jsonl"
        for line in p.read_text().strip().split('\n'):
            fam = json.loads(line)
            pa, pb = fam['participant_a'], fam['participant_b']
            for ck in ['context1', 'context2']:
                cx = fam[ck]
                text = cx['text_score_ablated']
                wl = cx['winner_label']
                for hyp, hd in [(f"{pa} defeated {pb}.", 'AB'), (f"{pb} defeated {pa}.", 'BA')]:
                    label = 1 if (hd=='AB' and wl=='A') or (hd=='BA' and wl=='B') else 0
                    rows.append({'tid': cx['template_id'], 'fsplit': fam['family_split'],
                                'text': f"{text} [SEP] {hyp}", 'label': label})
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

# ── Tiny BiGRU (16-dim for speed) ──
class TinyGRU(nn.Module):
    def __init__(self, vs, edim=16, hdim=16):
        super().__init__()
        self.emb = nn.Embedding(vs, edim, padding_idx=0)
        self.gru = nn.GRU(edim, hdim, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hdim*2, 1)
    def forward(self, x):
        mask = (x != 0).float()
        out, _ = self.gru(self.emb(x))
        # Mean pool over non-padded
        pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled).squeeze(-1)

# ── BoW baseline ──
class BoWModel(nn.Module):
    def __init__(self, vs, hdim=32):
        super().__init__()
        self.emb = nn.EmbeddingBag(vs, hdim, mode='mean', padding_idx=0)
        self.head = nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU(), nn.Linear(hdim, 1))
    def forward(self, x):
        return self.head(self.emb(x)).squeeze(-1)

def train_model(model, train_ids, train_labels, epochs=15, lr=0.003, bs=64):
    tx = torch.tensor(train_ids, dtype=torch.long)
    ty = torch.tensor(train_labels, dtype=torch.float)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=bs, shuffle=True)
    best_acc, best_state = 0, None
    for ep in range(epochs):
        model.train()
        c = t = 0
        for xb, yb in loader:
            l = F.binary_cross_entropy_with_logits(model(xb), yb)
            l.backward(); opt.step(); opt.zero_grad()
            c += ((model(xb)>0).float()==yb).sum().item(); t += len(yb)
        acc = c/t
        if acc > best_acc:
            best_acc = acc
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
        if best_acc >= 0.999: break
    if best_state: model.load_state_dict(best_state)
    return best_acc

def eval_per_template(model, eval_groups):
    model.eval()
    results = {}
    with torch.no_grad():
        for tid, (ids, labels) in eval_groups.items():
            if not ids: continue
            logits = model(torch.tensor(ids, dtype=torch.long))
            preds = (logits > 0).float()
            gt = torch.tensor(labels, dtype=torch.float)
            acc = (preds == gt).float().mean().item()
            # Also compute: fraction predicting 1
            pred1_frac = preds.mean().item()
            # And the true label fraction
            true1_frac = gt.mean().item()
            results[tid] = {'acc': round(acc, 4), 'n': len(labels),
                           'pred1_frac': round(pred1_frac, 4),
                           'true1_frac': round(true1_frac, 4)}
    return results

def main():
    t0 = time.time()
    nli = load_nli()
    vocab = build_vocab([r['text'] for r in nli])
    vs = len(vocab)
    print(f"NLI rows: {len(nli)}, Vocab: {vs}")
    
    # Split rows
    def classify(tid):
        if tid in ANCHOR_TEMPLATES: return 'anchor'
        if tid in PROBE_TEMPLATES: return 'probe'
        return 'held'
    
    anchor_train = [r for r in nli if classify(r['tid'])=='anchor' and r['fsplit']=='train']
    
    # Eval groups by template
    eval_groups = {}
    for r in nli:
        if r['fsplit'] == 'train':
            tid = r['tid']
            if tid not in eval_groups: eval_groups[tid] = ([], [])
            eval_groups[tid][0].append(encode(r['text'], vocab))
            eval_groups[tid][1].append(r['label'])
    
    # Train IDs
    a_ids = [encode(r['text'], vocab) for r in anchor_train]
    a_lab = [r['label'] for r in anchor_train]
    
    print(f"\nAnchor train: {len(anchor_train)} rows")
    print(f"Label balance: {Counter(r['label'] for r in anchor_train)}")
    
    # ── Run GRU and BoW across seeds ──
    all_results = {'gru': defaultdict(list), 'bow': defaultdict(list)}
    
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        
        # Train tiny GRU
        gru = TinyGRU(vs)
        gru_train_acc = train_model(gru, a_ids, a_lab, epochs=15)
        gru_results = eval_per_template(gru, eval_groups)
        
        # Train BoW
        bow = BoWModel(vs)
        bow_train_acc = train_model(bow, a_ids, a_lab, epochs=15)
        bow_results = eval_per_template(bow, eval_groups)
        
        for tid in sorted(eval_groups.keys()):
            g = classify(tid)
            info = TEMPLATE_INFO.get(tid, ('?', '?'))
            
            if tid in gru_results:
                all_results['gru'][tid].append(gru_results[tid]['acc'])
            if tid in bow_results:
                all_results['bow'][tid].append(bow_results[tid]['acc'])
        
        print(f"\nSeed {seed}: GRU train={gru_train_acc:.4f}, BoW train={bow_train_acc:.4f}")
        for tid in sorted(eval_groups.keys()):
            g = classify(tid)
            fm, pred = TEMPLATE_INFO.get(tid, ('?', '?'))
            ga = gru_results.get(tid, {}).get('acc', float('nan'))
            ba = bow_results.get(tid, {}).get('acc', float('nan'))
            marker = '***' if g == 'probe' else ('   ' if g == 'anchor' else '+++')
            print(f"  {marker} T{tid:02d} ({g:6s} {fm}) {pred:30s} GRU={ga:.3f} BoW={ba:.3f}")
    
    # Aggregate
    print(f"\n{'='*80}")
    print(f"AGGREGATED PER-TEMPLATE ACCURACY (anchor-only training, {N_SEEDS} seeds)")
    print(f"{'='*80}")
    print(f"{'TID':>4s} {'group':>7s} {'fm':>3s} {'predicate':>30s} {'GRU_mean':>9s} {'GRU_std':>8s} {'BoW_mean':>9s} {'BoW_std':>8s} {'n':>4s}")
    print('-'*85)
    
    summary_table = {}
    for tid in sorted(eval_groups.keys()):
        g = classify(tid)
        fm, pred = TEMPLATE_INFO.get(tid, ('?', '?'))
        n = len(eval_groups[tid][1])
        
        gru_vals = all_results['gru'].get(tid, [])
        bow_vals = all_results['bow'].get(tid, [])
        
        gm = round(np.mean(gru_vals), 4) if gru_vals else float('nan')
        gs = round(np.std(gru_vals), 4) if gru_vals else float('nan')
        bm = round(np.mean(bow_vals), 4) if bow_vals else float('nan')
        bs_val = round(np.std(bow_vals), 4) if bow_vals else float('nan')
        
        marker = '>>>' if g == 'probe' else ('...' if g == 'held' else '   ')
        print(f"{marker}T{tid:02d} {g:>7s} {fm:>3s} {pred:>30s} {gm:>9.4f} {gs:>8.4f} {bm:>9.4f} {bs_val:>8.4f} {n:>4d}")
        
        summary_table[tid] = {
            'group': g, 'first_mention': fm, 'predicate': pred, 'n': n,
            'gru_mean': gm, 'gru_std': gs, 'bow_mean': bm, 'bow_std': bs_val
        }
    
    # Group summaries
    print(f"\nGroup averages:")
    for group in ['anchor', 'probe', 'held']:
        tids = [t for t in summary_table if summary_table[t]['group'] == group]
        gru_means = [summary_table[t]['gru_mean'] for t in tids if not np.isnan(summary_table[t]['gru_mean'])]
        bow_means = [summary_table[t]['bow_mean'] for t in tids if not np.isnan(summary_table[t]['bow_mean'])]
        print(f"  {group:7s}: GRU={np.mean(gru_means):.4f}±{np.std(gru_means):.4f}  BoW={np.mean(bow_means):.4f}±{np.std(bow_means):.4f}")
    
    # By first_mention within probe templates
    print(f"\nProbe templates by first_mention:")
    for fm in ['w', 'l', 'm']:
        tids = [t for t in PROBE_TEMPLATES if TEMPLATE_INFO[t][0] == fm]
        for tid in tids:
            s = summary_table[tid]
            direction = 'winner-first' if fm == 'w' else ('loser-first' if fm == 'l' else 'embedded')
            print(f"  T{tid:02d} ({direction:12s}, {s['predicate']:30s}): GRU={s['gru_mean']:.4f}±{s['gru_std']:.4f}")
    
    elapsed = time.time() - t0
    
    output = {
        'status': 'TEMPLATE_INVERSION_DIAGNOSTIC',
        'elapsed': round(elapsed, 1),
        'per_template': summary_table,
    }
    out = DATA_DIR / 'template_inversion_diagnostic.json'
    out.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nElapsed: {elapsed:.1f}s. Saved: {out}")

if __name__ == '__main__':
    main()
