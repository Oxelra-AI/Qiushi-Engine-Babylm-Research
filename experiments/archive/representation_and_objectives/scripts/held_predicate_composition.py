#!/usr/bin/env python3
"""research decisive test: predicates held out of composition training.

The shuffled-pair control showed that any auxiliary representation-matching loss
breaks the symmetric plateau equally well, and that held cp x hp *cells* are
trivially generalized once the training predicates are learned. So that surface
cannot discriminate an equivariance objective.

This script builds the discriminating surface instead. Four predicates (2 subject=
winner, 2 subject=loser) receive independent semantic evidence only through
role-word support rows ("X was the winner/loser"), and never appear in any
composition (context-predicate x hypothesis-predicate) training row. Evaluation
then requires composing those predicates in both context and hypothesis position,
on held families and a held source domain.

Comparison, matched in data/budget/lambda:
  standard      : CE only
  equivariant   : CE + true role-equivalence matching over training composition rows
  shuffled_any  : CE + random-pair matching (symmetry-breaking control)

Only runs that actually learn the training distribution are scientifically
comparable, so the summary reports both raw means and converged-only means.
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
MODULE_PATH = STUDY / "scripts/equivariance_truth_gradient_and_composition.py"
OUT_DIR = STUDY / "data/held_predicate_composition"

# Predicates never used in composition training; semantics supplied only by
# role-word support rows.
HELD_COMP_PREDS = ["overcame", "fell_to", "prevailed", "was_defeated"]


def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

def write_jsonl(p: Path, rows: list[dict[str, Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")

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


def true_pairs(rows, cap, seed):
    g=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if r.get('source') != 'composition_train_pair': continue
        g[(r['family_id'], r['world'], r['hp'], r['orient'], int(r['label']))].append(i)
    out=[]
    for idxs in g.values():
        for a in range(len(idxs)):
            for b in range(a+1,len(idxs)):
                if rows[idxs[a]]['cp'] != rows[idxs[b]]['cp']: out.append((idxs[a], idxs[b]))
    random.Random(seed).shuffle(out); return out[:cap]

def random_pairs(rows, cap, seed):
    rng=random.Random(seed)
    idx=[i for i,r in enumerate(rows) if r.get('source') == 'composition_train_pair']
    out=[]
    while len(out)<cap:
        a,b=rng.choice(idx), rng.choice(idx)
        if a!=b: out.append((a,b))
    return out


def build_data(base, args):
    all_train = base.read_jsonl(base.TRAIN)
    all_held = base.read_jsonl(base.HELD)
    train_fams, held_fams, held_domain_fams, held_domain, domain_counts = base.choose_family_splits(
        all_train, all_held, args.train_fams, args.held_fams, args.held_domain_fams)
    seen = [p for p in base.PRED_LIST if p not in HELD_COMP_PREDS]
    # Support rows cover every predicate (including held-composition ones).
    support = base.make_support_rows(train_fams, "train")
    train_pairs = {(cp, hp) for cp in seen for hp in seen}
    comp_train = base.make_pair_rows(train_fams, "train", train_pairs, "composition_train_pair")
    train_rows = support + comp_train

    def pairs_with(cps, hps):
        return {(cp, hp) for cp in cps for hp in hps}

    eval_sets = {
        # sanity: seen-predicate composition on held families
        "seen_comp_held_family": base.make_pair_rows(held_fams, "held", train_pairs, "eval_seen_comp_held_family"),
        # decisive: held-composition predicate in hypothesis position only
        "heldpred_hyp_only": base.make_pair_rows(held_fams, "held", pairs_with(seen, HELD_COMP_PREDS), "eval_heldpred_hyp_only"),
        # decisive: held-composition predicate in context position only
        "heldpred_ctx_only": base.make_pair_rows(held_fams, "held", pairs_with(HELD_COMP_PREDS, seen), "eval_heldpred_ctx_only"),
        # hardest: both positions held out of composition
        "heldpred_both": base.make_pair_rows(held_fams, "held", pairs_with(HELD_COMP_PREDS, HELD_COMP_PREDS), "eval_heldpred_both"),
        # hardest + held source domain
        "heldpred_both_held_domain": base.make_pair_rows(held_domain_fams, "held_domain", pairs_with(HELD_COMP_PREDS, HELD_COMP_PREDS), "eval_heldpred_both_held_domain"),
    }
    meta = {
        "held_domain": held_domain, "domain_counts": domain_counts,
        "family_counts": {"train": len(train_fams), "held": len(held_fams), "held_domain": len(held_domain_fams)},
        "seen_predicates": seen, "held_composition_predicates": HELD_COMP_PREDS,
        "train_pair_cells": len(train_pairs),
        "row_counts": {"train_total": len(train_rows), "support": len(support), "composition_train": len(comp_train), **{k: len(v) for k,v in eval_sets.items()}},
        "design_note": "Held-composition predicates appear only in role-word support rows, so their word meaning is learnable but their use in composition is genuinely new.",
    }
    return train_rows, eval_sets, meta


def run_case(base, train_rows, eval_sets, mode, seed, args):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    vocab=base.Vocab(); vocab.fit(train_rows)
    all_eval=[r for rs in eval_sets.values() for r in rs]
    # Held-composition predicate words appear in support rows, so they are in-vocab.
    max_len=min(96, max(len(base.toks(r['text'])) for r in train_rows+all_eval))
    tr_x,tr_y=base.encode_rows(train_rows, vocab, max_len)
    enc={k: base.encode_rows(v, vocab, max_len) for k,v in eval_sets.items()}
    model=TextBiGRU(len(vocab.w2i))
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader=DataLoader(TensorDataset(tr_x,tr_y), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
    eq=None; n_pairs=0
    if mode=='equivariant':
        pl=true_pairs(train_rows, args.cap, seed); n_pairs=len(pl); eq=torch.tensor(pl,dtype=torch.long)
    elif mode=='shuffled_any':
        pl=random_pairs(train_rows, args.cap, seed); n_pairs=len(pl); eq=torch.tensor(pl,dtype=torch.long)
    hist=[]; t0=time.time(); best_train=-1.0; best_state=None
    for ep in range(1, args.epochs+1):
        model.train()
        for xb,yb in loader:
            loss=F.cross_entropy(model(xb), yb)
            if eq is not None and args.lam>0:
                p=eq[torch.randint(0,len(eq),(min(128,len(eq)),))]
                ha=model.encode_repr(tr_x[p[:,0]]); hb=model.encode_repr(tr_x[p[:,1]])
                loss=loss+args.lam*F.mse_loss(F.normalize(ha,dim=-1), F.normalize(hb,dim=-1))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if ep<=3 or ep%2==0 or ep==args.epochs:
            ta=acc(model,tr_x,tr_y)
            rec={'epoch':ep,'train_acc':ta,'heldpred_both':acc(model,*enc['heldpred_both']),'seconds':time.time()-t0}
            hist.append(rec); print(json.dumps({'mode':mode,'seed':seed,**rec}), flush=True)
            # Select by TRAIN fit only, so held-composition surfaces stay untouched.
            if ta>best_train: best_train=ta; best_state=copy.deepcopy(model.state_dict())
            if ta>=0.999: break
    if best_state: model.load_state_dict(best_state)
    return {'mode':mode,'seed':seed,'pairs':n_pairs,'final_train_acc':acc(model,tr_x,tr_y),
            'converged': bool(acc(model,tr_x,tr_y)>=0.95),
            'eval':{k: acc(model,*enc[k]) for k in enc}, 'history':hist}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-fams',type=int,default=24)
    ap.add_argument('--held-fams',type=int,default=12)
    ap.add_argument('--held-domain-fams',type=int,default=12)
    ap.add_argument('--epochs',type=int,default=26)
    ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--lam',type=float,default=0.35)
    ap.add_argument('--cap',type=int,default=11520)
    ap.add_argument('--seeds',type=str,default='254,255,256')
    ap.add_argument('--modes',type=str,default='standard,equivariant,shuffled_any')
    args=ap.parse_args()
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    base=load_base()
    train_rows, eval_sets, meta = build_data(base, args)
    write_jsonl(OUT_DIR/'train_rows.jsonl', train_rows)
    for k,v in eval_sets.items(): write_jsonl(OUT_DIR/f'eval_{k}.jsonl', v)
    audits={k: base.audit_rows(v,k) for k,v in eval_sets.items()}
    audits['train_rows']=base.audit_rows(train_rows,'train_rows')

    cases=[]
    for mode in [m for m in args.modes.split(',') if m.strip()]:
        for seed in [int(s) for s in args.seeds.split(',') if s.strip()]:
            cases.append(run_case(base, train_rows, eval_sets, mode, seed, args))

    surfaces=list(eval_sets.keys())
    agg={}
    for mode in {c['mode'] for c in cases}:
        sub=[c for c in cases if c['mode']==mode]
        conv=[c for c in sub if c['converged']]
        agg[mode]={'n_runs':len(sub),'n_converged':len(conv),
                   'raw':{s:{'mean':float(np.mean([c['eval'][s] for c in sub])),'seeds':[c['eval'][s] for c in sub]} for s in surfaces},
                   'converged_only':{s:{'mean':float(np.mean([c['eval'][s] for c in conv])) if conv else float('nan'),'seeds':[c['eval'][s] for c in conv]} for s in surfaces},
                   'train_acc':{'mean':float(np.mean([c['final_train_acc'] for c in sub])),'seeds':[c['final_train_acc'] for c in sub]}}
    summary={'status':'HELD_PREDICATE_COMPOSITION','created_utc':now(),
             'config':vars(args),'design':meta,'audits':audits,'aggregates':agg,'cases':cases,
             'interpretation_boundary':'Composition surfaces here use predicates that never appear in composition training, so a converged-only difference between standard and equivariant learning is a genuine objective effect. A null converged-only difference means role-composition transfer is not created by this matching objective at this scale.'}
    write_json(OUT_DIR/'held_predicate_composition_summary.json', summary)

    md=['# research held-predicate composition test','',
        f"Seen predicates: {meta['seen_predicates']}",
        f"Held-composition predicates (support-only): {meta['held_composition_predicates']}",
        f"Held source domain: {meta['held_domain']}",'',
        '| mode | runs | converged | train acc | seen_comp_held_family | heldpred_hyp_only | heldpred_ctx_only | heldpred_both | heldpred_both_held_domain |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in ['standard','equivariant','shuffled_any']:
        if mode not in agg: continue
        a=agg[mode]; c=a['converged_only']
        md.append(f"| {mode} | {a['n_runs']} | {a['n_converged']} | {a['train_acc']['mean']:.3f} | "
                  f"{c['seen_comp_held_family']['mean']:.3f} | {c['heldpred_hyp_only']['mean']:.3f} | "
                  f"{c['heldpred_ctx_only']['mean']:.3f} | {c['heldpred_both']['mean']:.3f} | {c['heldpred_both_held_domain']['mean']:.3f} |")
    md += ['','Converged-only means are the scientifically comparable numbers; runs stuck on the symmetric plateau are reported separately in the JSON.','',
           f"Summary JSON: `{OUT_DIR/'held_predicate_composition_summary.json'}`"]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/held_predicate_composition/held_predicate_composition_summary.md')).write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'summary_json':str(OUT_DIR/'held_predicate_composition_summary.json'),
                      'converged_counts':{m:agg[m]['n_converged'] for m in agg},
                      'heldpred_both_converged':{m:agg[m]['converged_only']['heldpred_both']['mean'] for m in agg}}, indent=2), flush=True)

if __name__=='__main__': main()
