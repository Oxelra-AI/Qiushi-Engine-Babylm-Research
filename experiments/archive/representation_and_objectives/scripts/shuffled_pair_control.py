#!/usr/bin/env python3
"""research shuffled-pair control for the equivariance advantage.

The targeted convergence probe showed that standard CE stays at chance for seeds
254/256 through 24 epochs while the equivariance objective converges to 0.984 on
held predicate-pair compositions. That could be a genuine role-equivariance
effect, or a generic auxiliary-gradient/regularization effect.

Controls compared here, all with identical CE data, budget, and lambda:
  standard        : CE only
  equivariant     : match rows with same family/world/hypothesis/orientation/label
                    that differ only in context predicate (true role-equivalence)
  shuffled_label  : match random row pairs that merely share the label
  shuffled_any    : match fully random row pairs
  same_cp_control : match rows that share the context predicate but differ in
                    world/query (wrong invariance: collapses distinctions that
                    the task needs)
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
OUT_DIR = STUDY / "data/shuffled_pair_control"


def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

def load_base():
    spec = importlib.util.spec_from_file_location("base", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
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
            ok += int((model(x[s:s+batch]).argmax(1) == y[s:s+batch]).sum().item()); n += len(x[s:s+batch])
    return ok/max(1,n)

def comp_indices(rows):
    return [i for i,r in enumerate(rows) if r.get('source') == 'composition_train_pair']

def pairs_true(rows, cap, seed):
    g=collections.defaultdict(list)
    for i in comp_indices(rows):
        r=rows[i]; g[(r['family_id'], r['world'], r['hp'], r['orient'], int(r['label']))].append(i)
    out=[]
    for idxs in g.values():
        for a in range(len(idxs)):
            for b in range(a+1,len(idxs)):
                if rows[idxs[a]]['cp'] != rows[idxs[b]]['cp']: out.append((idxs[a], idxs[b]))
    random.Random(seed).shuffle(out); return out[:cap]

def pairs_shuffled_label(rows, cap, seed):
    rng=random.Random(seed)
    byl=collections.defaultdict(list)
    for i in comp_indices(rows): byl[int(rows[i]['label'])].append(i)
    out=[]
    while len(out) < cap:
        y=rng.choice([0,1]); a,b=rng.choice(byl[y]), rng.choice(byl[y])
        if a!=b: out.append((a,b))
    return out

def pairs_shuffled_any(rows, cap, seed):
    rng=random.Random(seed); idx=comp_indices(rows); out=[]
    while len(out) < cap:
        a,b=rng.choice(idx), rng.choice(idx)
        if a!=b: out.append((a,b))
    return out

def pairs_same_cp(rows, cap, seed):
    rng=random.Random(seed)
    g=collections.defaultdict(list)
    for i in comp_indices(rows): g[rows[i]['cp']].append(i)
    out=[]
    guard=0
    while len(out) < cap and guard < cap*40:
        guard+=1
        cp=rng.choice(list(g)); a,b=rng.choice(g[cp]), rng.choice(g[cp])
        if a==b: continue
        ra,rb=rows[a],rows[b]
        if (ra['family_id'],ra['world'],ra['hp'],ra['orient']) != (rb['family_id'],rb['world'],rb['hp'],rb['orient']):
            out.append((a,b))
    return out

PAIR_BUILDERS = {
    'equivariant': pairs_true,
    'shuffled_label': pairs_shuffled_label,
    'shuffled_any': pairs_shuffled_any,
    'same_cp_control': pairs_same_cp,
}

def run_case(base, train_rows, eval_sets, mode, seed, epochs, lr, lam, cap):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    vocab=base.Vocab(); vocab.fit(train_rows)
    all_eval=[r for rs in eval_sets.values() for r in rs]
    max_len=min(96, max(len(base.toks(r['text'])) for r in train_rows+all_eval))
    tr_x,tr_y=base.encode_rows(train_rows, vocab, max_len)
    enc={k: base.encode_rows(v, vocab, max_len) for k,v in eval_sets.items()}
    model=TextBiGRU(len(vocab.w2i))
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader=DataLoader(TensorDataset(tr_x,tr_y), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
    eq=None
    n_pairs=0
    if mode != 'standard':
        pl=PAIR_BUILDERS[mode](train_rows, cap, seed)
        n_pairs=len(pl)
        eq=torch.tensor(pl, dtype=torch.long) if pl else None
    hist=[]; best=-1.0; best_state=None; t0=time.time()
    sel='held_pair_held_family'
    for ep in range(1, epochs+1):
        model.train()
        for xb,yb in loader:
            loss=F.cross_entropy(model(xb), yb)
            if eq is not None and lam>0:
                p=eq[torch.randint(0,len(eq),(min(128,len(eq)),))]
                ha=model.encode_repr(tr_x[p[:,0]]); hb=model.encode_repr(tr_x[p[:,1]])
                loss=loss+lam*F.mse_loss(F.normalize(ha,dim=-1), F.normalize(hb,dim=-1))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if ep<=3 or ep%2==0 or ep==epochs:
            ta=acc(model,tr_x,tr_y); sa=acc(model,*enc[sel])
            hist.append({'epoch':ep,'train_acc':ta,sel:sa,'seconds':time.time()-t0})
            print(json.dumps({'mode':mode,'seed':seed,**hist[-1]}), flush=True)
            if sa>best: best=sa; best_state=copy.deepcopy(model.state_dict())
            if ta>=0.999 and sa>=0.95: break
    if best_state: model.load_state_dict(best_state)
    return {'mode':mode,'seed':seed,'pairs':n_pairs,'best_selected':best,'history':hist,
            'eval':{k: acc(model,*enc[k]) for k in enc},
            'final_train_acc':acc(model,tr_x,tr_y)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=18)
    ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--lam',type=float,default=0.35)
    ap.add_argument('--cap',type=int,default=11520)
    ap.add_argument('--seeds',type=str,default='254,256')
    ap.add_argument('--modes',type=str,default='standard,equivariant,shuffled_label,shuffled_any,same_cp_control')
    args=ap.parse_args()
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    base=load_base()
    train_rows=read_jsonl(BASE_DIR/'semantic_support_train.jsonl')+read_jsonl(BASE_DIR/'composition_train_pairs.jsonl')
    eval_sets={'held_pair_held_family':read_jsonl(BASE_DIR/'held_pair_held_family_eval.jsonl'),
               'held_pair_held_domain':read_jsonl(BASE_DIR/'held_pair_held_domain_eval.jsonl')}
    cases=[]
    for mode in [m for m in args.modes.split(',') if m.strip()]:
        for seed in [int(s) for s in args.seeds.split(',') if s.strip()]:
            cases.append(run_case(base, train_rows, eval_sets, mode, seed, args.epochs, args.lr, args.lam, args.cap))
    agg=collections.defaultdict(dict)
    for mode in {c['mode'] for c in cases}:
        sub=[c for c in cases if c['mode']==mode]
        for k in ['held_pair_held_family','held_pair_held_domain']:
            vals=[c['eval'][k] for c in sub]
            agg[mode][k]={'mean':float(np.mean(vals)),'std':float(np.std(vals)),'seeds':vals}
        agg[mode]['final_train_acc']={'mean':float(np.mean([c['final_train_acc'] for c in sub])),'seeds':[c['final_train_acc'] for c in sub]}
        agg[mode]['learned_rate_ge_0p90']=float(np.mean([c['eval']['held_pair_held_family']>=0.90 for c in sub]))
    summary={'status':'SHUFFLED_PAIR_CONTROL','created_utc':now(),
             'config':{'epochs':args.epochs,'lr':args.lr,'lambda':args.lam,'pair_cap':args.cap,'seeds':args.seeds,'modes':args.modes},
             'aggregates':dict(agg),'cases':cases,
             'interpretation_boundary':'If shuffled/label-only/same-predicate matching reproduces the equivariant advantage, the effect is generic auxiliary-gradient regularization. If only true role-equivalence pairing unlocks held-composition learning, the objective is doing role-specific work.'}
    write_json(OUT_DIR/'shuffled_pair_control_summary.json', summary)
    md=['# research shuffled-pair control','','| mode | pairs | final train | held-pair-held-family | held-pair-held-domain | learned-rate |','|---|---:|---:|---:|---:|---:|']
    for mode in ['standard','equivariant','shuffled_label','shuffled_any','same_cp_control']:
        if mode not in agg: continue
        a=agg[mode]; pr=[c['pairs'] for c in cases if c['mode']==mode]
        md.append(f"| {mode} | {pr[0] if pr else 0} | {a['final_train_acc']['mean']:.3f} | {a['held_pair_held_family']['mean']:.3f}±{a['held_pair_held_family']['std']:.3f} | {a['held_pair_held_domain']['mean']:.3f}±{a['held_pair_held_domain']['std']:.3f} | {a['learned_rate_ge_0p90']:.2f} |")
    md += ['', f"Summary JSON: `{OUT_DIR/'shuffled_pair_control_summary.json'}`"]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/shuffled_pair_control/shuffled_pair_control_summary.md')).write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'summary_json':str(OUT_DIR/'shuffled_pair_control_summary.json'),
                      'held_pair_held_family_by_mode':{m:agg[m]['held_pair_held_family']['mean'] for m in agg}}, indent=2), flush=True)

if __name__=='__main__': main()
