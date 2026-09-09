#!/usr/bin/env python3
"""research: 3-seed replication of section vs natural vs separated formats."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

STUDY = Path("experiments/archive/representation_and_objectives")
WS = STUDY
ATP_DIR = WS / "data/upset_balanced_state_substrate"
PATH = WS / "training/scripts/contrastive_wording_cross_probe.py"
spec = importlib.util.spec_from_file_location("s262", PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
ALIAS_POOL = m.ALIAS_POOL
DEFAULT_MODEL = m.DEFAULT_MODEL

def _si(s): return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)
def als(key, n): return random.Random(_si(f"a266r|{key}")).sample(ALIAS_POOL, n)

def load_changed_worlds():
    changed_tr, changed_he = [], []
    for sp, out_list in [("train", changed_tr), ("held", changed_he)]:
        p = ATP_DIR / f"families_{sp}.jsonl"
        for line in p.read_text("utf-8").splitlines():
            if not line.strip(): continue
            f = json.loads(line)
            a, b = f["participant_a"], f["participant_b"]
            for wk in ["context1", "context2"]:
                c = f[wk]
                if not c.get("state_changed_between_snapshots"): continue
                w, l = c["winner"], c["loser"]
                hi0, lo0 = (w, l) if c["winner_higher_at_time"] else (l, w)
                hi1, lo1 = (w, l) if c["winner_higher_later"] else (l, w)
                out_list.append(dict(
                    family_id=f["family_id"], world_id=f"{f['family_id']}::{wk}",
                    a=a, b=b, hi0=hi0, lo0=lo0, hi1=hi1, lo1=lo1,
                ))
    return changed_tr, changed_he

FORMATS = {
    "separated": {
        "ctx_fn": lambda w, am: am[w["hi0"]] + " was ranked above " + am[w["lo0"]] + " in the ATP.",
        "ctx_fn_after": lambda w, am: am[w["hi1"]] + " was ranked above " + am[w["lo1"]] + " in the ATP.",
        "hyp_b": "{X} held the higher ranking than {Y}.",
        "hyp_a": "{X} held the higher ranking than {Y}.",
        "dual_ctx": False,
    },
    "section": {
        "init": "Background: {H} was ranked above {Lo}.",
        "later": "Update: {H} was ranked above {Lo}.",
        "hyp_b": "Based on the background, {X} was ranked higher than {Y}.",
        "hyp_a": "Based on the update, {X} was ranked higher than {Y}.",
        "dual_ctx": True,
    },
    "natural": {
        "init": "In the earlier ATP ranking, {H} was above {Lo}.",
        "later": "In the later ATP ranking, {H} was above {Lo}.",
        "hyp_b": "Before the later ranking update, {X} held the higher ranking than {Y}.",
        "hyp_a": "After the later ranking update, {X} held the higher ranking than {Y}.",
        "dual_ctx": True,
    },
}

def build_rows(worlds, fmt_name, ns, split):
    fmt = FORMATS[fmt_name]
    rows = []
    for i, w in enumerate(worlds):
        al = als(f"{ns}|{w['world_id']}", 2)
        am = {w["a"]: al[0], w["b"]: al[1]}
        for hd in ["AB", "BA"]:
            x = am[w["a"]] if hd == "AB" else am[w["b"]]
            y = am[w["b"]] if hd == "AB" else am[w["a"]]
            hi0a = (w["hi0"] == w["a"])
            hi1a = (w["hi1"] == w["a"])
            lb = int((hd == "AB" and hi0a) or (hd == "BA" and not hi0a))
            la = int((hd == "AB" and hi1a) or (hd == "BA" and not hi1a))
            
            if not fmt["dual_ctx"]:
                ctx_b = fmt["ctx_fn"](w, am)
                ctx_a = fmt["ctx_fn_after"](w, am)
                rows.append({"text": f"{ctx_b} [SEP] {fmt['hyp_b'].format(X=x,Y=y)}", "label": lb, "query": "before"})
                rows.append({"text": f"{ctx_a} [SEP] {fmt['hyp_a'].format(X=x,Y=y)}", "label": la, "query": "after"})
            else:
                ci = fmt["init"].format(H=am[w["hi0"]], Lo=am[w["lo0"]])
                cl = fmt["later"].format(H=am[w["hi1"]], Lo=am[w["lo1"]])
                ctx = f"{ci} {cl}"
                rows.append({"text": f"{ctx} [SEP] {fmt['hyp_b'].format(X=x,Y=y)}", "label": lb, "query": "before"})
                rows.append({"text": f"{ctx} [SEP] {fmt['hyp_a'].format(X=x,Y=y)}", "label": la, "query": "after"})
    return rows

def train_eval_one(train_rows, eval_rows, model_path, device, seed, epochs=12, lr=1e-4, bs=8):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    head = torch.nn.Linear(model.config.hidden_size, 2).to(device)
    
    def enc(rows):
        return tokenizer([r["text"] for r in rows], padding=True, truncation=True, max_length=160, return_tensors="pt"), \
               torch.tensor([r["label"] for r in rows], dtype=torch.long)
    
    tr_e, tr_l = enc(train_rows)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-3)
    
    for ep in range(1, epochs + 1):
        model.train(); head.train()
        perm = torch.randperm(len(train_rows))
        c, t = 0, 0
        for s in range(0, len(perm), bs):
            idx = perm[s:s+bs]
            b = {k: v[idx].to(device) for k, v in tr_e.items()}
            l = tr_l[idx].to(device)
            logits = head(model(**b).last_hidden_state[:, 0, :])
            F.cross_entropy(logits, l).backward()
            opt.step(); opt.zero_grad()
            c += (logits.argmax(-1) == l).sum().item()
            t += len(idx)
        if c / t >= 0.999 and ep >= 3: break
    
    model.eval(); head.eval()
    def eval_by_q(rows, e_data, e_lab):
        bq = collections.defaultdict(lambda: [0, 0])
        with torch.no_grad():
            for s in range(0, len(rows), 32):
                end = min(s + 32, len(rows))
                b = {k: v[s:end].to(device) for k, v in e_data.items()}
                l = e_lab[s:end].to(device)
                p = head(model(**b).last_hidden_state[:, 0, :]).argmax(-1)
                for j in range(end - s):
                    q = rows[s + j]["query"]
                    bq[q][0] += int(p[j] == l[j]); bq[q][1] += 1
        return {q: v[0]/v[1] for q, v in bq.items()}
    
    tr_bq = eval_by_q(train_rows, tr_e, tr_l)
    ev_e, ev_l = enc(eval_rows)
    ev_bq = eval_by_q(eval_rows, ev_e, ev_l)
    
    del model, head; torch.cuda.empty_cache()
    return {"train_acc": c / t, "train_by_q": dict(tr_bq), "eval_by_q": dict(ev_bq)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--out_dir", default=str(WS / "data/section_replication_3seed"))
    parser.add_argument("--n_train", type=int, default=40)
    parser.add_argument("--n_eval", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    changed_tr, changed_he = load_changed_worlds()
    
    seeds = [26600, 26601, 26602]
    fmt_names = ["separated", "section", "natural"]
    
    results = collections.defaultdict(list)
    for seed in seeds:
        rng = random.Random(seed)
        tr_w = changed_tr[:]; rng.shuffle(tr_w); tr_w = tr_w[:args.n_train]
        ev_w = changed_he[:]; rng.shuffle(ev_w); ev_w = ev_w[:args.n_eval]
        
        for fn in fmt_names:
            tr = build_rows(tr_w, fn, f"tr_{fn}_{seed}", "train")
            ev = build_rows(ev_w, fn, f"ev_{fn}_{seed}", "held")
            r = train_eval_one(tr, ev, args.model_path, args.device, seed)
            r["seed"] = seed; r["fmt"] = fn
            results[fn].append(r)
            print(json.dumps({"seed": seed, "fmt": fn, "train": r["train_acc"],
                              "before": r["train_by_q"].get("before"),
                              "after": r["train_by_q"].get("after"),
                              "eval_b": r["eval_by_q"].get("before"),
                              "eval_a": r["eval_by_q"].get("after")}), flush=True)
    
    md = ["# research section replication (3 seeds)\n"]
    md.append("| Format | Seed | Train | Before | After | Eval before | Eval after |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for fn in fmt_names:
        for r in results[fn]:
            md.append(f"| {fn} | {r['seed']} | {r['train_acc']:.3f} | "
                      f"{r['train_by_q'].get('before', 0):.3f} | {r['train_by_q'].get('after', 0):.3f} | "
                      f"{r['eval_by_q'].get('before', 0):.3f} | {r['eval_by_q'].get('after', 0):.3f} |")
    
    md.append("\n## Means\n| Format | Train | Eval before | Eval after |")
    md.append("|---|---:|---:|---:|")
    for fn in fmt_names:
        ta = np.mean([r["train_acc"] for r in results[fn]])
        eb = np.mean([r["eval_by_q"].get("before", 0.5) for r in results[fn]])
        ea = np.mean([r["eval_by_q"].get("after", 0.5) for r in results[fn]])
        md.append(f"| {fn} | {ta:.3f} | {eb:.3f} | {ea:.3f} |")
    
    (out / "section_replication.md").write_text("\n".join(md))
    summary = {"results": {fn: results[fn] for fn in fmt_names}}
    (out / "section_replication.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": "DONE", "md": str(out / "section_replication.md")}), flush=True)

if __name__ == "__main__":
    main()
