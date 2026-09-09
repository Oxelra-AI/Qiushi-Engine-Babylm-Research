#!/usr/bin/env python3
"""research FAST: Raw-text anchor transfer — focused on the decisive comparison.

Key finding from partial run: zero-coverage probe accuracy is ~0.32, BELOW chance.
This means the model learns probe predicate structure but assigns the wrong role
direction — the inversion ambiguity from the factorized analysis.

This fast version focuses on:
1. Single-context only (interference deferred)
2. Fewer seeds (3) and coverage levels (0, 5, 20, 100)
3. Shorter training (25 epochs with early stop)
4. The decisive comparison: true vs shuffled vs exposure anchor quality
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

COVERAGE_LEVELS = [0, 5, 20, 100]
ARMS = ['true', 'shuffled', 'exposure']
N_SEEDS = 3
EMB_DIM = 48; HIDDEN_DIM = 48; MAX_LEN = 80
EPOCHS = 25; BATCH_SIZE = 32; LR = 0.002; PATIENCE = 6

def load_families():
    families = []
    for split in ['train', 'held']:
        p = PILOT_DIR / f"families_{split}.jsonl"
        for line in p.read_text().strip().split('\n'):
            families.append(json.loads(line))
    return families

def extract_nli_rows(families):
    rows = []
    for fam in families:
        pa, pb = fam['participant_a'], fam['participant_b']
        for cx_key in ['context1', 'context2']:
            cx = fam[cx_key]
            text = cx['text_score_ablated']
            wl = cx['winner_label']
            for hyp_text, hyp_dir in [(f"{pa} defeated {pb}.", 'AB'), 
                                       (f"{pb} defeated {pa}.", 'BA')]:
                label = 1 if (hyp_dir == 'AB' and wl == 'A') or (hyp_dir == 'BA' and wl == 'B') else 0
                rows.append({
                    'fam_id': fam['family_id'], 'fam_split': fam['family_split'],
                    'tid': cx['template_id'], 'tsplit': cx['template_split'],
                    'text': f"{text} [SEP] {hyp_text}", 'label': label,
                })
    return rows

def classify_tid(tid):
    if tid in ANCHOR_TEMPLATES: return 'anchor'
    if tid in PROBE_TEMPLATES:  return 'probe'
    if tid in HELD_TEMPLATES:   return 'held'
    return 'unknown'

def build_vocab(texts):
    wc = Counter()
    for t in texts:
        for w in t.lower().split(): wc[w] += 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for w, c in wc.most_common():
        if c >= 1: vocab[w] = len(vocab)
    return vocab

def encode(text, vocab):
    words = text.lower().split()[:MAX_LEN]
    ids = [vocab.get(w, 1) for w in words]
    return ids + [0] * (MAX_LEN - len(ids))

class BiGRU(nn.Module):
    def __init__(self, vs):
        super().__init__()
        self.emb = nn.Embedding(vs, EMB_DIM, padding_idx=0)
        self.gru = nn.GRU(EMB_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(HIDDEN_DIM*2, 1)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM), 
                                  nn.ReLU(), nn.Dropout(0.1), nn.Linear(HIDDEN_DIM, 1))
    def forward(self, x):
        mask = (x != 0).float()
        out, _ = self.gru(self.emb(x))
        s = self.attn(out).squeeze(-1).masked_fill(mask==0, -1e9)
        w = torch.softmax(s, dim=-1)
        return self.head((out * w.unsqueeze(-1)).sum(1)).squeeze(-1)

def train_eval(train_ids, train_labels, evals, vs, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    tx = torch.tensor(train_ids, dtype=torch.long)
    ty = torch.tensor(train_labels, dtype=torch.float)
    model = BiGRU(vs)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH_SIZE, shuffle=True)
    
    best_acc, patience, best_state = 0.0, 0, None
    for ep in range(1, EPOCHS+1):
        model.train()
        correct = total = 0
        for xb, yb in loader:
            logits = model(xb)
            F.binary_cross_entropy_with_logits(logits, yb).backward()
            opt.step(); opt.zero_grad()
            correct += ((logits>0).float()==yb).sum().item()
            total += len(yb)
        acc = correct/total
        if acc > best_acc:
            best_acc = acc; patience = 0
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= PATIENCE and best_acc > 0.6: break
        if best_acc >= 0.999: break
    
    if best_state: model.load_state_dict(best_state)
    model.eval()
    res = {'train_acc': best_acc, 'epochs': ep}
    with torch.no_grad():
        for name, (ei, el) in evals.items():
            if not ei: res[name] = float('nan'); continue
            logits = model(torch.tensor(ei, dtype=torch.long))
            res[name] = ((logits>0).float() == torch.tensor(el, dtype=torch.float)).float().mean().item()
    return res

def main():
    t0 = time.time()
    families = load_families()
    nli = extract_nli_rows(families)
    
    vocab = build_vocab([r['text'] for r in nli])
    vs = len(vocab)
    print(f"Families: {len(families)}, NLI rows: {len(nli)}, Vocab: {vs}")
    
    # Partition rows
    anchor_train = [r for r in nli if classify_tid(r['tid'])=='anchor' and r['fam_split']=='train']
    probe_train  = [r for r in nli if classify_tid(r['tid'])=='probe'  and r['fam_split']=='train']
    anchor_eval  = [r for r in nli if classify_tid(r['tid'])=='anchor' and r['fam_split']=='held']
    probe_eval   = [r for r in nli if classify_tid(r['tid'])=='probe'  and r['fam_split']=='held']
    held_eval    = [r for r in nli if classify_tid(r['tid'])=='held']
    
    print(f"Anchor train: {len(anchor_train)}, Probe train: {len(probe_train)}")
    print(f"Anchor eval: {len(anchor_eval)}, Probe eval: {len(probe_eval)}, Held eval: {len(held_eval)}")
    
    # Encode fixed sets
    a_ids = [encode(r['text'], vocab) for r in anchor_train]
    a_lab = [r['label'] for r in anchor_train]
    p_ids = [encode(r['text'], vocab) for r in probe_train]
    p_lab = [r['label'] for r in probe_train]
    
    def enc_set(rows):
        if not rows: return [], []
        return [encode(r['text'], vocab) for r in rows], [r['label'] for r in rows]
    
    eval_anchor = enc_set(anchor_eval)
    eval_probe_held = enc_set(probe_eval)
    eval_held = enc_set(held_eval)
    eval_probe_all = enc_set(probe_train)
    
    # Per-template probe eval
    per_tid_eval = {}
    for tid in sorted(PROBE_TEMPLATES):
        rows = [r for r in probe_train if r['tid']==tid]
        if rows: per_tid_eval[f'T{tid:02d}'] = enc_set(rows)
    
    results = []
    for cov in COVERAGE_LEVELS:
        for arm in ARMS:
            if cov == 0 and arm != 'true': continue
            arm_label = f"{arm}_{cov}pct" if cov > 0 else "zero"
            
            for seed in range(N_SEEDS):
                rng = random.Random(seed*1000 + cov*10 + hash(arm)%100)
                n_sel = int(len(probe_train) * cov / 100)
                indices = list(range(len(probe_train)))
                rng.shuffle(indices)
                sel = indices[:n_sel]
                
                sel_ids = [p_ids[i] for i in sel]
                if arm == 'true':
                    sel_lab = [p_lab[i] for i in sel]
                elif arm == 'shuffled':
                    sel_lab = [1-p_lab[i] if rng.random()<0.5 else p_lab[i] for i in sel]
                elif arm == 'exposure':
                    sel_lab = [rng.randint(0,1) for _ in sel]
                
                not_sel = [i for i in range(len(probe_train)) if i not in set(sel)]
                eval_probe_unseen = ([p_ids[i] for i in not_sel], [p_lab[i] for i in not_sel])
                
                evals = {
                    'anchor_eval': eval_anchor,
                    'probe_unseen': eval_probe_unseen,
                    'probe_all': eval_probe_all,
                    'held': eval_held,
                }
                evals.update(per_tid_eval)
                
                r = train_eval(a_ids + sel_ids, a_lab + sel_lab, evals, vs, 
                              seed + cov*17 + hash(arm)%100)
                r['arm'] = arm_label; r['arm_type'] = arm; r['cov'] = cov; r['seed'] = seed
                r['n_probe'] = n_sel; r['n_train'] = len(a_ids)+n_sel
                results.append(r)
                
                pu = r.get('probe_unseen', float('nan'))
                h = r.get('held', float('nan'))
                print(json.dumps({'arm':arm_label,'seed':seed,
                    'train':round(r['train_acc'],4),
                    'probe_unseen':round(pu,4) if not np.isnan(pu) else 'nan',
                    'held':round(h,4) if not np.isnan(h) else 'nan'}), flush=True)
    
    # Aggregate
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        a = r['arm']
        agg[a]['train_acc'].append(r['train_acc'])
        for k in ['anchor_eval','probe_unseen','probe_all','held']:
            v = r.get(k, float('nan'))
            if not np.isnan(v): agg[a][k].append(v)
        for tid in sorted(PROBE_TEMPLATES):
            v = r.get(f'T{tid:02d}', float('nan'))
            if not np.isnan(v): agg[a][f'T{tid:02d}'].append(v)
    
    summary = {}
    for arm, metrics in agg.items():
        summary[arm] = {}
        for k, vals in metrics.items():
            vals = [v for v in vals if not np.isnan(v)]
            if vals: summary[arm][k] = {'mean': round(np.mean(vals),4), 'std': round(np.std(vals),4), 'n': len(vals)}
    
    elapsed = time.time() - t0
    output = {
        'status': 'RAW_TEXT_ANCHOR_TRANSFER_FAST',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'elapsed_seconds': round(elapsed,1),
        'config': {
            'anchor_templates': sorted(ANCHOR_TEMPLATES),
            'probe_templates': sorted(PROBE_TEMPLATES),
            'held_templates': sorted(HELD_TEMPLATES),
            'coverage_levels': COVERAGE_LEVELS, 'arms': ARMS,
            'n_seeds': N_SEEDS, 'model': 'BiGRU_48_48',
        },
        'summary': summary,
        'raw': results,
    }
    
    out = DATA_DIR / 'anchor_transfer_fast_summary.json'
    out.write_text(json.dumps(output, indent=2, default=str))
    
    # Print table
    print(f"\n{'='*70}")
    print(f"RAW-TEXT ANCHOR TRANSFER (elapsed: {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"\n{'arm':<22s} {'train':>7s} {'anchor':>7s} {'p_unseen':>9s} {'p_all':>7s} {'held':>7s}")
    print('-'*55)
    for arm_name in ['zero'] + [f'{a}_{c}pct' for c in [5,20,100] for a in ARMS]:
        if arm_name not in summary: continue
        s = summary[arm_name]
        vals = [s.get(k,{}).get('mean',float('nan')) for k in ['train_acc','anchor_eval','probe_unseen','probe_all','held']]
        print(f"{arm_name:<22s}" + "".join(f" {v:>7.4f}" if not np.isnan(v) else f" {'nan':>7s}" for v in vals))
    
    # Per-template breakdown for key arms
    print(f"\n--- Per-template probe accuracy ---")
    print(f"{'arm':<22s}" + "".join(f" {'T'+str(t):>7s}" for t in sorted(PROBE_TEMPLATES)))
    print('-'*57)
    for arm_name in ['zero'] + [f'{a}_{c}pct' for c in [5,20,100] for a in ARMS]:
        if arm_name not in summary: continue
        s = summary[arm_name]
        vals = [s.get(f'T{t:02d}',{}).get('mean',float('nan')) for t in sorted(PROBE_TEMPLATES)]
        print(f"{arm_name:<22s}" + "".join(f" {v:>7.4f}" if not np.isnan(v) else f" {'nan':>7s}" for v in vals))
    
    print(f"\nSaved: {out}")
    print(json.dumps({'status': output['status'], 'elapsed': elapsed, 'summary_json': str(out)}, indent=2))

if __name__ == '__main__':
    main()
