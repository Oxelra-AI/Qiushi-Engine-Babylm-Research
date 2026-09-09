#!/usr/bin/env python3
"""research targeted convergence probe on saved repaired datasets.

The broad e30/s6 convergence run was too slow. This script is deliberately
narrow: it reloads the saved research repaired train/eval rows and tests whether
specific failed seeds from the first diagnostic can learn with more epochs. It
prints progress and writes a compact summary, so a timeout is interpretable as an
operational cost problem rather than a silent scientific result.
"""
from __future__ import annotations

import argparse, collections, copy, importlib.util, json, os, random, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

STUDY = Path("experiments/archive/representation_and_objectives")
BASE_DIR = STUDY / "data/equivariance_diagnostics"
MODULE_PATH = STUDY / "scripts/equivariance_truth_gradient_and_composition.py"
OUT_DIR = STUDY / "data/targeted_convergence_probe"


def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

def load_base_module():
    spec = importlib.util.spec_from_file_location("base", MODULE_PATH)
    if spec is None or spec.loader is None: raise RuntimeError(MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod

class TextBiGRU(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 64, hid: int = 96, dropout: float = 0.02):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(2*hid), nn.Dropout(dropout), nn.Linear(2*hid, 64), nn.ReLU(), nn.Linear(64, 2))
    def encode_repr(self, x):
        e = self.emb(x); _, h = self.gru(e); return torch.cat([h[-2], h[-1]], dim=-1)
    def forward(self, x): return self.head(self.encode_repr(x))

def acc(model, x, y, batch=1024):
    model.eval(); ok=0; n=0
    with torch.no_grad():
        for s in range(0, len(x), batch):
            pred = model(x[s:s+batch]).argmax(1)
            ok += int((pred == y[s:s+batch]).sum().item()); n += len(pred)
    return ok/max(1,n)

def build_eq_pairs(rows, max_pairs=20000, seed=25404):
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if r.get('source') != 'composition_train_pair': continue
        groups[(r['family_id'], r['world'], r['hp'], r['orient'], int(r['label']))].append(i)
    pairs=[]
    for idxs in groups.values():
        for a in range(len(idxs)):
            for b in range(a+1,len(idxs)):
                if rows[idxs[a]]['cp'] != rows[idxs[b]]['cp']:
                    pairs.append((idxs[a], idxs[b]))
    random.Random(seed).shuffle(pairs)
    return pairs[:max_pairs]

def train_one(base, train_rows, eval_rows, seed, mode, epochs, lr, lam):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    vocab = base.Vocab(); vocab.fit(train_rows)
    max_len = min(96, max(len(base.toks(r['text'])) for r in train_rows + eval_rows))
    tr_x, tr_y = base.encode_rows(train_rows, vocab, max_len)
    ev_x, ev_y = base.encode_rows(eval_rows, vocab, max_len)
    model = TextBiGRU(len(vocab.w2i))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
    eq_t = torch.tensor(build_eq_pairs(train_rows), dtype=torch.long) if mode == 'equivariant' and lam > 0 else None
    hist=[]; best={'eval_acc':-1}; best_state=None
    t0=time.time()
    for ep in range(1, epochs+1):
        model.train(); total=0.0; n=0
        for xb,yb in loader:
            logits=model(xb); loss=F.cross_entropy(logits,yb)
            if eq_t is not None:
                pi=torch.randint(0,len(eq_t),(min(128,len(eq_t)),))
                p=eq_t[pi]
                ha=model.encode_repr(tr_x[p[:,0]]); hb=model.encode_repr(tr_x[p[:,1]])
                loss = loss + lam*F.mse_loss(F.normalize(ha,dim=-1), F.normalize(hb,dim=-1))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total += float(loss.item())*len(xb); n += len(xb)
        if ep <= 5 or ep % 2 == 0 or ep == epochs:
            ta=acc(model,tr_x,tr_y); ea=acc(model,ev_x,ev_y)
            rec={'epoch':ep,'train_acc':ta,'eval_acc':ea,'seconds':time.time()-t0,'loss_proxy':total/max(1,n)}
            hist.append(rec); print(json.dumps({'mode':mode,'seed':seed,**rec}), flush=True)
            if ea > best['eval_acc']:
                best=rec; best_state=copy.deepcopy(model.state_dict())
            if ta >= 0.999 and ea >= 0.95:
                break
    if best_state: model.load_state_dict(best_state)
    return {'mode':mode,'seed':seed,'epochs_requested':epochs,'best':best,'history':hist,'final_train_acc':acc(model,tr_x,tr_y),'final_eval_acc':acc(model,ev_x,ev_y),'vocab_size':len(vocab.w2i),'max_len':max_len}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=24)
    ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--lam',type=float,default=0.35)
    ap.add_argument('--cases',type=str,default='standard:254,standard:256,equivariant:254')
    args=ap.parse_args()
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    base=load_base_module()
    support=read_jsonl(BASE_DIR/'semantic_support_train.jsonl')
    comp=read_jsonl(BASE_DIR/'composition_train_pairs.jsonl')
    train_rows=support+comp
    eval_rows=read_jsonl(BASE_DIR/'held_pair_held_family_eval.jsonl')
    rows=[]
    for case in args.cases.split(','):
        mode, seed_s = case.split(':')
        rows.append(train_one(base, train_rows, eval_rows, int(seed_s), mode, args.epochs, args.lr, args.lam))
    summary={'status':'TARGETED_CONVERGENCE_PROBE','created_utc':now(),'base_dir':str(BASE_DIR),'eval_surface':'held_pair_held_family','cases':rows,'interpretation_boundary':'This tests whether the failed seeds from the first research run are merely slow/nonconverged under the same repaired predicate-supported task.'}
    write_json(OUT_DIR/'targeted_convergence_summary.json',summary)
    md=['# research targeted convergence probe','','| mode | seed | best epoch | best train | best held-pair-held-family | final train | final eval |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        b=r['best']; md.append(f"| {r['mode']} | {r['seed']} | {b.get('epoch')} | {b.get('train_acc', float('nan')):.3f} | {b.get('eval_acc', float('nan')):.3f} | {r['final_train_acc']:.3f} | {r['final_eval_acc']:.3f} |")
    md += ['', f"Summary JSON: `{OUT_DIR/'targeted_convergence_summary.json'}`"]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/targeted_convergence_probe/targeted_convergence_summary.md')).write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'summary_json':str(OUT_DIR/'targeted_convergence_summary.json'),'case_best_eval':[r['best']['eval_acc'] for r in rows]}, indent=2), flush=True)

if __name__=='__main__': main()
