#!/usr/bin/env python3
"""research: Entity-anonymized predicate learning test.

The name-memorization diagnostic showed:
  - BiGRU achieves 98% training accuracy through name memorization
  - Zero predicate-semantic transfer to new names
  - Below-chance within-family accuracy from entity-binding interference

This test removes the name shortcut by replacing all entity names with
Entity_A / Entity_B. The question: can the model learn predicate semantics
when name memorization is impossible?

If yes → anchoring principle becomes testable in raw text
If no → predicate semantics require richer representation (pretrained, factorized)
"""

import json, random, time, re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE = Path("experiments/archive/representation_and_objectives")
DATA_DIR = WORKSPACE / "data/anonymized_predicate_test"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PILOT_DIR = WORKSPACE / "data/paired_world_pilot"

ANCHOR_TEMPLATES = {1, 2, 3, 6, 7, 8, 11, 12, 13, 14}
PROBE_TEMPLATES  = {4, 5, 9, 10, 15}

MAX_LEN = 80; EMB_DIM = 48; HIDDEN_DIM = 48; EPOCHS = 30; BATCH_SIZE = 32; LR = 0.002
N_SEEDS = 5

def load_train_families():
    p = PILOT_DIR / "families_train.jsonl"
    return [json.loads(line) for line in p.read_text().strip().split('\n')]

def anonymize_text(text, pa, pb):
    """Replace participant names with Entity_A / Entity_B."""
    # Replace longer name first to avoid partial matches
    names = sorted([(pa, 'Entity_A'), (pb, 'Entity_B')], key=lambda x: -len(x[0]))
    for real_name, anon_name in names:
        text = text.replace(real_name, anon_name)
    return text

def extract_anon_nli_rows(families):
    """Extract NLI rows with anonymized names."""
    rows = []
    for fam in families:
        pa, pb = fam['participant_a'], fam['participant_b']
        for ck in ['context1', 'context2']:
            cx = fam[ck]
            text_orig = cx['text_score_ablated']
            text_anon = anonymize_text(text_orig, pa, pb)
            wl = cx['winner_label']
            tid = cx['template_id']
            
            # Hypotheses with anonymized names
            hyp_ab = "Entity_A defeated Entity_B."
            hyp_ba = "Entity_B defeated Entity_A."
            
            for hyp, hd in [(hyp_ab, 'AB'), (hyp_ba, 'BA')]:
                label = 1 if (hd=='AB' and wl=='A') or (hd=='BA' and wl=='B') else 0
                rows.append({
                    'tid': tid, 'fsplit': fam['family_split'],
                    'fam_id': fam['family_id'], 'cx': ck,
                    'text_anon': f"{text_anon} [SEP] {hyp}",
                    'text_orig': f"{text_orig} [SEP] {hyp.replace('Entity_A', pa).replace('Entity_B', pb)}",
                    'label': label, 'winner': wl,
                })
    return rows

def classify_tid(tid):
    if tid in ANCHOR_TEMPLATES: return 'anchor'
    if tid in PROBE_TEMPLATES: return 'probe'
    return 'held'

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

def train_eval_model(train_ids, train_labels, eval_sets, vs, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = BiGRU(vs)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(
        torch.tensor(train_ids, dtype=torch.long),
        torch.tensor(train_labels, dtype=torch.float)),
        batch_size=BATCH_SIZE, shuffle=True)
    
    best_acc, best_state = 0, None
    for ep in range(EPOCHS):
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
    model.eval()
    res = {'train_acc': round(best_acc, 4), 'epochs': ep+1}
    with torch.no_grad():
        for name, (ei, el) in eval_sets.items():
            if not ei: res[name] = float('nan'); continue
            logits = model(torch.tensor(ei, dtype=torch.long))
            preds = (logits > 0).float()
            gt = torch.tensor(el, dtype=torch.float)
            res[name] = round((preds==gt).float().mean().item(), 4)
    return res

def main():
    t0 = time.time()
    families = load_train_families()
    rows = extract_anon_nli_rows(families)
    print(f"Total anonymized NLI rows: {len(rows)}")
    
    # Show anonymization examples
    for r in rows[:3]:
        print(f"  ANON: {r['text_anon'][:100]}...")
        print(f"  ORIG: {r['text_orig'][:100]}...")
        print(f"  label={r['label']}, winner={r['winner']}, tid={r['tid']}")
        print()
    
    # Split by template group
    anchor_rows = [r for r in rows if classify_tid(r['tid']) == 'anchor']
    probe_rows  = [r for r in rows if classify_tid(r['tid']) == 'probe']
    
    print(f"Anchor rows: {len(anchor_rows)}, Probe rows: {len(probe_rows)}")
    print(f"Label balance anchor: {Counter(r['label'] for r in anchor_rows)}")
    print(f"Label balance probe: {Counter(r['label'] for r in probe_rows)}")
    
    # Build vocab from anonymized text
    vocab_anon = build_vocab([r['text_anon'] for r in rows])
    vocab_orig = build_vocab([r['text_orig'] for r in rows])
    
    print(f"Vocab (anon): {len(vocab_anon)}, Vocab (orig): {len(vocab_orig)}")
    
    # Encode
    a_anon = [encode(r['text_anon'], vocab_anon) for r in anchor_rows]
    a_orig = [encode(r['text_orig'], vocab_orig) for r in anchor_rows]
    a_lab = [r['label'] for r in anchor_rows]
    
    p_anon = [encode(r['text_anon'], vocab_anon) for r in probe_rows]
    p_orig = [encode(r['text_orig'], vocab_orig) for r in probe_rows]
    p_lab = [r['label'] for r in probe_rows]
    
    # Split anchor rows: 80% train, 20% held (by family)
    fam_ids = sorted(set(r['fam_id'] for r in anchor_rows))
    random.seed(42)
    random.shuffle(fam_ids)
    train_fams = set(fam_ids[:int(len(fam_ids)*0.8)])
    held_fams = set(fam_ids[int(len(fam_ids)*0.8):])
    
    anchor_train = [i for i, r in enumerate(anchor_rows) if r['fam_id'] in train_fams]
    anchor_held  = [i for i, r in enumerate(anchor_rows) if r['fam_id'] in held_fams]
    
    # Results collection
    all_results = []
    
    for mode_name, vocab, enc_a, enc_p in [
        ('anonymized', vocab_anon, a_anon, p_anon),
        ('original', vocab_orig, a_orig, p_orig),
    ]:
        vs = len(vocab)
        
        tr_ids = [enc_a[i] for i in anchor_train]
        tr_lab = [a_lab[i] for i in anchor_train]
        he_ids = [enc_a[i] for i in anchor_held]
        he_lab = [a_lab[i] for i in anchor_held]
        
        print(f"\n=== {mode_name.upper()} ===")
        print(f"Train: {len(tr_ids)}, Anchor held: {len(he_ids)}, Probe: {len(enc_p)}")
        
        for seed in range(N_SEEDS):
            evals = {
                'anchor_held': (he_ids, he_lab),
                'probe_all': (enc_p, p_lab),
            }
            
            # Per-template probe
            per_tid = defaultdict(lambda: ([], []))
            for i, r in enumerate(probe_rows):
                per_tid[r['tid']][0].append(enc_p[i])
                per_tid[r['tid']][1].append(p_lab[i])
            for tid, (ids, labs) in per_tid.items():
                evals[f'T{tid:02d}'] = (ids, labs)
            
            r = train_eval_model(tr_ids, tr_lab, evals, vs, seed)
            r['mode'] = mode_name
            r['seed'] = seed
            all_results.append(r)
            
            print(json.dumps({
                'mode': mode_name, 'seed': seed,
                'train': r['train_acc'],
                'anchor_held': r['anchor_held'],
                'probe_all': r['probe_all'],
            }), flush=True)
    
    # Aggregate
    print(f"\n{'='*70}")
    print(f"ANONYMIZED vs ORIGINAL: PREDICATE LEARNING TEST")
    print(f"{'='*70}")
    
    summary = {}
    for mode in ['anonymized', 'original']:
        mode_res = [r for r in all_results if r['mode'] == mode]
        s = {}
        for key in ['train_acc', 'anchor_held', 'probe_all']:
            vals = [r[key] for r in mode_res if not np.isnan(r[key])]
            s[key] = {'mean': round(np.mean(vals), 4), 'std': round(np.std(vals), 4)}
        for tid in sorted(PROBE_TEMPLATES):
            vals = [r.get(f'T{tid:02d}', float('nan')) for r in mode_res]
            vals = [v for v in vals if not np.isnan(v)]
            if vals: s[f'T{tid:02d}'] = {'mean': round(np.mean(vals), 4), 'std': round(np.std(vals), 4)}
        summary[mode] = s
    
    print(f"\n{'metric':<18s} {'ANON mean':>10s} {'ANON std':>9s} {'ORIG mean':>10s} {'ORIG std':>9s}")
    print('-'*60)
    for key in ['train_acc', 'anchor_held', 'probe_all']:
        am = summary['anonymized'].get(key, {}).get('mean', float('nan'))
        astd = summary['anonymized'].get(key, {}).get('std', float('nan'))
        om = summary['original'].get(key, {}).get('mean', float('nan'))
        ostd = summary['original'].get(key, {}).get('std', float('nan'))
        print(f"{key:<18s} {am:>10.4f} {astd:>9.4f} {om:>10.4f} {ostd:>9.4f}")
    
    # Per-template breakdown
    print(f"\nPer-template probe accuracy:")
    for tid in sorted(PROBE_TEMPLATES):
        key = f'T{tid:02d}'
        am = summary['anonymized'].get(key, {}).get('mean', float('nan'))
        om = summary['original'].get(key, {}).get('mean', float('nan'))
        print(f"  {key}: ANON={am:.4f}, ORIG={om:.4f}")
    
    elapsed = time.time() - t0
    
    output = {
        'status': 'ANONYMIZED_PREDICATE_TEST',
        'elapsed': round(elapsed, 1),
        'summary': summary,
        'raw': all_results,
    }
    out = DATA_DIR / 'anonymized_predicate_test_summary.json'
    out.write_text(json.dumps(output, indent=2, default=str))
    
    print(f"\n  INTERPRETATION:")
    a_held = summary['anonymized']['anchor_held']['mean']
    a_probe = summary['anonymized']['probe_all']['mean']
    o_held = summary['original']['anchor_held']['mean']
    
    if a_held > 0.55:
        print(f"  ANON anchor_held={a_held:.3f} > 0.55: model learns predicate semantics!")
        if a_probe > 0.55:
            print(f"  ANON probe_all={a_probe:.3f} > 0.55: transfer to new predicates!")
        else:
            print(f"  ANON probe_all={a_probe:.3f} ≈ 0.50: no cross-predicate transfer yet")
    else:
        print(f"  ANON anchor_held={a_held:.3f} ≈ 0.50: model cannot learn predicate semantics")
        if o_held > 0.55:
            print(f"  ORIG anchor_held={o_held:.3f} > 0.55: but original uses name memorization")
        print(f"  → Predicate learning from raw text requires richer representation than BiGRU")
    
    print(f"\nElapsed: {elapsed:.1f}s. Saved: {out}")

if __name__ == '__main__':
    main()
