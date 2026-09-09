#!/usr/bin/env python3
"""research v2: Identity-shortcut bridge with RANDOMIZED repeat positions.

v1 used fixed MODIFY_IDX=[0,3], creating a position-based copy shortcut.
v2 randomizes which 2 base events are repeated per sequence, forcing
content-based entity matching. REPEAT_MASKED uses per-example 3D masks.

Same 5 conditions, same eval probes, same model.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ VOCABULARY ============
N_ENT = 32
N_ATTR = 32
ENTITIES = [f"e{i}" for i in range(N_ENT)]
ATTRIBUTES = [f"a{i}" for i in range(N_ATTR)]
SPECIALS = ["<pad>", "<bos>", "<eos>"]
VERBS = ["has", "is"]
ALL_TOKENS = SPECIALS + VERBS + ["."] + ENTITIES + ATTRIBUTES


class Vocab:
    def __init__(self):
        self.itos = list(ALL_TOKENS)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad = self.stoi["<pad>"]
        self.bos = self.stoi["<bos>"]
        self.eos = self.stoi["<eos>"]
    def encode(self, tokens): return [self.stoi[t] for t in tokens]
    def __len__(self): return len(self.itos)


# ============ DATA ============
EV_TOTAL = 8
BASE_N = 6
MOD_N = 2
TOKS_PER_EV = 4
SEQ_LEN = 1 + EV_TOTAL * TOKS_PER_EV + 1  # 34

def ev_start(s): return 1 + s * TOKS_PER_EV
def ev_end(s): return ev_start(s) + TOKS_PER_EV
def attr_pos(s): return ev_start(s) + 2
def ev_tok(e, v, a): return [e, v, a, "."]


def gen_sequence_v2(condition, vocab, rng):
    """Generate one sequence with RANDOMIZED modify indices."""
    used = set()
    def pick():
        e = rng.choice([x for x in ENTITIES if x not in used])
        used.add(e)
        return e

    base = []
    for _ in range(BASE_N):
        base.append((pick(), rng.choice(VERBS), rng.choice(ATTRIBUTES)))

    # Randomly select which 2 base events to modify
    mod_idx = sorted(rng.sample(range(BASE_N), MOD_N))

    extra = []
    if condition == "unique":
        for _ in range(MOD_N):
            extra.append((pick(), rng.choice(VERBS), rng.choice(ATTRIBUTES)))
    elif condition in ("repeat_full", "repeat_masked"):
        for i in mod_idx:
            extra.append(base[i])
    elif condition == "varied":
        for i in mod_idx:
            e, v, a = base[i]
            extra.append((e, "is" if v == "has" else "has", a))
    elif condition == "wrong":
        for i in mod_idx:
            e, v, a = base[i]
            extra.append((e, v, rng.choice([x for x in ATTRIBUTES if x != a])))

    toks = ["<bos>"]
    for e, v, a in base:
        toks.extend(ev_tok(e, v, a))
    for e, v, a in extra:
        toks.extend(ev_tok(e, v, a))
    toks.append("<eos>")
    assert len(toks) == SEQ_LEN

    # Repeat block info for attention masking
    blocks = []
    if condition in ("repeat_full", "repeat_masked"):
        for slot, bidx in enumerate(mod_idx):
            blocks.append((ev_start(bidx), ev_end(bidx),
                           ev_start(BASE_N + slot), ev_end(BASE_N + slot)))

    return vocab.encode(toks), blocks, mod_idx


def gen_batch_v2(n, condition, vocab, rng):
    """Generate batch with per-example repeat block info."""
    all_ids, all_blocks, all_midx = [], [], []
    for _ in range(n):
        ids, blocks, midx = gen_sequence_v2(condition, vocab, rng)
        all_ids.append(ids)
        all_blocks.append(blocks)
        all_midx.append(midx)
    return torch.tensor(all_ids, dtype=torch.long), all_blocks, all_midx


# ============ EVAL PROBES ============
def gen_eval_probes(n, vocab, rng):
    """Generate probes with context at RANDOM slot positions."""
    probes = []
    used = set()
    for _ in range(n):
        while True:
            ent = rng.choice(ENTITIES)
            attr = rng.choice(ATTRIBUTES)
            if (ent, attr) not in used:
                used.add((ent, attr))
                break

        # Random context position among slots 0-6 (not the last slot = probe)
        ctx_slot = rng.randint(0, EV_TOTAL - 2)
        ctx_toks = ev_tok(ent, "has", attr)

        # Filler for baseline (replaces context)
        fent = rng.choice([e for e in ENTITIES if e != ent])
        filler_ctx = ev_tok(fent, rng.choice(VERBS), rng.choice(ATTRIBUTES))

        # Other fillers
        excl = {ent, fent}
        fillers = []
        for s in range(EV_TOTAL - 1):  # 7 slots before probe
            if s == ctx_slot:
                continue  # skip context slot
            fe = rng.choice([e for e in ENTITIES if e not in excl])
            excl.add(fe)
            fillers.append(ev_tok(fe, rng.choice(VERBS), rng.choice(ATTRIBUTES)))

        # Build sequences: 7 events + 1 probe
        probe_has = ev_tok(ent, "has", attr)
        probe_is = ev_tok(ent, "is", attr)

        def mkseq(use_ctx, probe):
            t = ["<bos>"]
            fi = 0
            for s in range(EV_TOTAL - 1):
                if s == ctx_slot:
                    t.extend(use_ctx)
                else:
                    t.extend(fillers[fi])
                    fi += 1
            t.extend(probe)
            t.append("<eos>")
            assert len(t) == SEQ_LEN, f"len={len(t)}"
            return vocab.encode(t)

        probes.append({
            "entity": ent, "attr": attr, "ctx_slot": ctx_slot,
            "copy": mkseq(ctx_toks, probe_has),
            "content": mkseq(ctx_toks, probe_is),
            "base_has": mkseq(filler_ctx, probe_has),
            "base_is": mkseq(filler_ctx, probe_is),
            "target_pos": attr_pos(EV_TOTAL - 1),
            "target_id": vocab.stoi[attr],
        })
    return probes


# ============ MODEL ============
class Block(nn.Module):
    def __init__(self, d, nh, dff=None):
        super().__init__()
        dff = dff or 4 * d
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))
    def forward(self, x, mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + h
        return x + self.ff(self.ln2(x))

class SmallGPT(nn.Module):
    def __init__(self, V, d=64, nh=2, nl=3, ml=64):
        super().__init__()
        self.d = d
        self.nh = nh
        self.te = nn.Embedding(V, d)
        self.pe = nn.Embedding(ml, d)
        self.layers = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
    def forward(self, x, mask=None):
        B, T = x.shape
        h = self.te(x) + self.pe(torch.arange(T, device=x.device))
        for layer in self.layers: h = layer(h, mask=mask)
        return self.head(self.ln(h))


# ============ MASKS ============
def causal_mask(L, dev):
    return torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), diagonal=1)


def build_3d_mask(batch_blocks, seq_len, nhead, dev):
    """Build per-example 3D attention mask: (batch*nhead, L, L).

    batch_blocks[i] = list of (orig_s, orig_e, rep_s, rep_e)
    """
    B = len(batch_blocks)
    base = causal_mask(seq_len, dev)
    mask = base.unsqueeze(0).expand(B, -1, -1).clone()  # (B, L, L)
    for i, blocks in enumerate(batch_blocks):
        for os, oe, rs, re in blocks:
            mask[i, rs:re, os:oe] = True
    # Expand for multi-head: (B, L, L) → (B*nh, L, L)
    mask = mask.unsqueeze(1).expand(-1, nhead, -1, -1).reshape(B * nhead, seq_len, seq_len)
    return mask


# ============ TRAIN ============
def train_one_epoch(model, condition, n_seqs, bs, vocab, rng, opt, dev):
    model.train()
    data, blocks, midxs = gen_batch_v2(n_seqs, condition, vocab, rng)
    data = data.to(dev)
    cmask = causal_mask(SEQ_LEN, dev)
    tot, nb = 0.0, 0

    for i in range(0, n_seqs, bs):
        b = data[i:i+bs]
        bsz = b.shape[0]

        if condition == "repeat_masked":
            mask = build_3d_mask(blocks[i:i+bs], SEQ_LEN, model.nh, dev)
        else:
            mask = cmask

        logits = model(b, mask=mask)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, len(vocab)), b[:, 1:].reshape(-1)
        )
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    return tot / nb


# ============ EVAL ============
@torch.no_grad()
def eval_probes_fn(model, probes, vocab, dev):
    model.eval()
    cmask = causal_mask(SEQ_LEN, dev)
    copy_nll, cont_nll, bh_nll, bi_nll = [], [], [], []
    all_seqs = []
    for p in probes:
        all_seqs.extend([p["copy"], p["content"], p["base_has"], p["base_is"]])
    batch = torch.tensor(all_seqs, dtype=torch.long, device=dev)
    chunk = 256
    all_logits = []
    for s in range(0, len(batch), chunk):
        all_logits.append(model(batch[s:s+chunk], mask=cmask))
    logits = torch.cat(all_logits, dim=0)
    for idx, p in enumerate(probes):
        base = idx * 4
        tpos = p["target_pos"]
        tid = p["target_id"]
        lp = F.log_softmax(logits[base:base+4, tpos-1], dim=-1)
        copy_nll.append(-lp[0, tid].item())
        cont_nll.append(-lp[1, tid].item())
        bh_nll.append(-lp[2, tid].item())
        bi_nll.append(-lp[3, tid].item())
    n = len(probes)
    a = {
        "copy_nll": round(sum(copy_nll)/n, 5),
        "content_nll": round(sum(cont_nll)/n, 5),
        "base_has_nll": round(sum(bh_nll)/n, 5),
        "base_is_nll": round(sum(bi_nll)/n, 5),
    }
    a["copy_gain"] = round(a["base_has_nll"] - a["copy_nll"], 5)
    a["content_gain"] = round(a["base_is_nll"] - a["content_nll"], 5)
    return a


@torch.no_grad()
def eval_train_positions(model, condition, n_seqs, vocab, rng_state, dev):
    """Measure NLL at modification vs base attribute positions."""
    model.eval()
    rng = random.Random()
    rng.setstate(rng_state)
    data, blocks, midxs = gen_batch_v2(n_seqs, condition, vocab, rng)
    data = data.to(dev)
    cmask = causal_mask(SEQ_LEN, dev)
    logits = model(data, mask=cmask)
    lp = F.log_softmax(logits[:, :-1], dim=-1)
    targets = data[:, 1:]
    tgt_lp = lp.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    base_idx = [attr_pos(s) - 1 for s in range(BASE_N)]
    mod_idx = [attr_pos(BASE_N + s) - 1 for s in range(MOD_N)]
    return {
        "base_attr_nll": round(-tgt_lp[:, base_idx].mean().item(), 5),
        "mod_attr_nll": round(-tgt_lp[:, mod_idx].mean().item(), 5),
    }


# ============ MAIN ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seqs", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=2)
    ap.add_argument("--nlayers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--conditions", nargs="+",
                    default=["unique", "repeat_full", "repeat_masked", "varied", "wrong"])
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    vocab = Vocab()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}, Vocab: {len(vocab)}, SEQ_LEN: {SEQ_LEN}", flush=True)

    eval_rng = random.Random(args.seed + 7777)
    probes = gen_eval_probes(args.n_probes, vocab, eval_rng)
    print(f"Generated {len(probes)} probes (ctx_slots: {set(p['ctx_slot'] for p in probes)})", flush=True)

    torch.manual_seed(args.seed)
    init_model = SmallGPT(len(vocab), args.d_model, args.nhead, args.nlayers, SEQ_LEN + 4)
    n_params = sum(p.numel() for p in init_model.parameters())
    init_state = {k: v.clone() for k, v in init_model.state_dict().items()}
    print(f"Params: {n_params:,}", flush=True)

    # Verify data identity
    rng1, rng2 = random.Random(args.seed), random.Random(args.seed)
    d1, _, _ = gen_batch_v2(5, "repeat_full", vocab, rng1)
    d2, b2, _ = gen_batch_v2(5, "repeat_masked", vocab, rng2)
    assert (d1 == d2).all(), "Data mismatch!"
    print(f"✓ Data identical; sample blocks: {b2[0]}", flush=True)

    all_res = {}
    for cond in args.conditions:
        print(f"\n{'='*55}\n  {cond}\n{'='*55}", flush=True)
        model = SmallGPT(len(vocab), args.d_model, args.nhead, args.nlayers, SEQ_LEN+4).to(dev)
        model.load_state_dict(init_state); model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        dseed = args.seed if cond in ("repeat_full", "repeat_masked") else args.seed + hash(cond)
        drng = random.Random(dseed)
        hist = []; t0 = time.time()

        for ep in range(1, args.epochs + 1):
            rng_st = drng.getstate()
            loss = train_one_epoch(model, cond, args.n_seqs, args.bs, vocab, drng, opt, dev)
            if ep % args.eval_every == 0 or ep == 1 or ep == args.epochs:
                ev = eval_probes_fn(model, probes, vocab, dev)
                tp = eval_train_positions(model, cond, min(args.n_seqs, 200), vocab, rng_st, dev)
                entry = {"epoch": ep, "train_loss": round(loss, 6), **ev, **tp}
                hist.append(entry)
                print(json.dumps(entry), flush=True)

        elapsed = time.time() - t0
        final = hist[-1] if hist else {}
        all_res[cond] = {"condition": cond, "elapsed": round(elapsed, 2), "history": hist,
                         "final": {k: final.get(k) for k in [
                             "copy_gain", "content_gain", "copy_nll", "content_nll",
                             "base_has_nll", "base_is_nll", "base_attr_nll", "mod_attr_nll",
                             "train_loss"]}}
        (out / cond).mkdir(exist_ok=True)
        with open(out / cond / "result.json", "w") as f:
            json.dump(all_res[cond], f, indent=2)

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}", flush=True)
    table = []
    for cond in args.conditions:
        r = all_res[cond]; f = r["final"]
        row = {"condition": cond, **{k: f.get(k) for k in [
            "copy_gain", "content_gain", "copy_nll", "content_nll",
            "base_has_nll", "base_is_nll", "base_attr_nll", "mod_attr_nll",
            "train_loss"]}, "elapsed": r["elapsed"]}
        table.append(row)
        cg = f.get("copy_gain") or 0; ccg = f.get("content_gain") or 0
        tl = f.get("train_loss") or 0; ma = f.get("mod_attr_nll") or 0
        print(f"  {cond:16s}  copy={cg:+.4f}  content={ccg:+.4f}  "
              f"mod_attr={ma:.4f}  loss={tl:.4f}  {r['elapsed']:.0f}s", flush=True)

    print(f"\n  KEY CONTRAST: repeat_full vs repeat_masked", flush=True)
    if "repeat_full" in all_res and "repeat_masked" in all_res:
        rf, rm = all_res["repeat_full"]["final"], all_res["repeat_masked"]["final"]
        dc = (rf.get("copy_gain") or 0) - (rm.get("copy_gain") or 0)
        dcc = (rf.get("content_gain") or 0) - (rm.get("content_gain") or 0)
        print(f"  Δcopy_gain  = {dc:+.5f}", flush=True)
        print(f"  Δcontent_gain = {dcc:+.5f}", flush=True)
        if dc > 0.01 and dcc < -0.01:
            print("  → SUPPORTED: identity shortcut helps copy, hurts content", flush=True)
        elif dc > 0.01 and dcc > 0.01:
            print("  → Identity shortcut helps BOTH copy and content (no competition)", flush=True)
        elif abs(dc) < 0.01 and abs(dcc) < 0.01:
            print("  → No measurable identity shortcut effect", flush=True)
        else:
            print(f"  → Interpret with learning curves", flush=True)

    with open(out / "summary.json", "w") as f:
        json.dump({"status": "V2_DONE", "args": vars(args),
                    "vocab_size": len(vocab), "seq_len": SEQ_LEN,
                    "n_params": n_params, "design": "randomized_modify_positions",
                    "conditions": table}, f, indent=2)
    print(json.dumps({"status": "V2_DONE", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
