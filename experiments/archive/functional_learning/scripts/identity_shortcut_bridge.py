#!/usr/bin/env python3
"""research functional_learning: Identity-shortcut bridge experiment.

Scientific question: Does within-context identity shortcut (attention-based
access to identical content) cause damage to nonidentical content use, beyond
what mere weight-based repetition causes?

Design:
  5 conditions share identical initialization and evaluation probes.
  Key contrast: REPEAT_FULL vs REPEAT_MASKED have IDENTICAL training data.
  MASKED blocks attention from second occurrence to first occurrence of
  repeated events, removing the identity shortcut while preserving all else.

  Conditions:
    unique:        8 unique events per sequence (baseline)
    repeat_full:   6 base + 2 exact repeats, full causal attention
    repeat_masked: same data as repeat_full, repeat→original attention blocked
    varied:        6 base + 2 restatements (flipped verb template)
    wrong:         6 base + 2 contradictions (different attribute)

  Events use two interchangeable templates (both seen by all conditions):
    "[entity] has [attr] ."   and   "[entity] is [attr] ."

  Evaluation probes (held-out, same for all, full causal attention):
    copy_gain:    NLL(fresh has-probe) - NLL(primed has-probe)
    content_gain: NLL(fresh is-probe)  - NLL(primed is-probe)

  A content-conditioning probe presents "eX has aY" then asks to predict aY
  in "eX is ___". Copy probe repeats "eX has aY" verbatim.
  Gains measure how much prior context helps predict the target attribute.

  Prediction:
    If identity shortcut is the mechanism:
      REPEAT_FULL  > REPEAT_MASKED in copy_gain (learned to copy)
      REPEAT_FULL  < REPEAT_MASKED in content_gain (copy competes)
    If not: both repeat conditions should be similar in content_gain
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== VOCABULARY =====================

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

    def encode(self, tokens):
        return [self.stoi[t] for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def __len__(self):
        return len(self.itos)


# ===================== DATA =====================

EVENTS_TOTAL = 8  # per sequence
BASE_N = 6
MODIFY_N = 2
MODIFY_IDX = [0, 3]  # which base events to repeat/vary/wrong
TOKS_PER_EV = 4  # entity verb attr .
SEQ_LEN = 1 + EVENTS_TOTAL * TOKS_PER_EV + 1  # 34


def ev_start(slot):
    return 1 + slot * TOKS_PER_EV


def ev_end(slot):
    return ev_start(slot) + TOKS_PER_EV


def attr_pos(slot):
    return ev_start(slot) + 2


def ev_tokens(entity, verb, attr):
    return [entity, verb, attr, "."]


def gen_sequence(condition, vocab, rng):
    """Generate one training sequence for the given condition."""
    used = set()

    def pick_ent():
        e = rng.choice([x for x in ENTITIES if x not in used])
        used.add(e)
        return e

    base = []
    for _ in range(BASE_N):
        base.append((pick_ent(), rng.choice(VERBS), rng.choice(ATTRIBUTES)))

    extra = []
    if condition == "unique":
        for _ in range(MODIFY_N):
            extra.append((pick_ent(), rng.choice(VERBS), rng.choice(ATTRIBUTES)))
    elif condition in ("repeat_full", "repeat_masked"):
        for i in MODIFY_IDX:
            extra.append(base[i])  # exact copy
    elif condition == "varied":
        for i in MODIFY_IDX:
            e, v, a = base[i]
            nv = "is" if v == "has" else "has"
            extra.append((e, nv, a))
    elif condition == "wrong":
        for i in MODIFY_IDX:
            e, v, a = base[i]
            wa = rng.choice([x for x in ATTRIBUTES if x != a])
            extra.append((e, v, wa))
    else:
        raise ValueError(condition)

    toks = ["<bos>"]
    for e, v, a in base:
        toks.extend(ev_tokens(e, v, a))
    for e, v, a in extra:
        toks.extend(ev_tokens(e, v, a))
    toks.append("<eos>")
    assert len(toks) == SEQ_LEN, f"len={len(toks)}"
    return vocab.encode(toks)


def gen_batch(n, condition, vocab, rng):
    return torch.tensor(
        [gen_sequence(condition, vocab, rng) for _ in range(n)],
        dtype=torch.long,
    )


# ===================== EVAL PROBES =====================

def gen_eval_probes(n, vocab, rng):
    """Generate matched evaluation probes.

    Each probe has 4 sequences of length SEQ_LEN:
      copy:     [eX has aY] [fillers] [eX has aY]     (copy gain)
      content:  [eX has aY] [fillers] [eX is  aY]     (content gain)
      base_has: [filler0]   [fillers] [eX has aY]     (baseline copy)
      base_is:  [filler0]   [fillers] [eX is  aY]     (baseline content)
    """
    probes = []
    used = set()
    for _ in range(n):
        while True:
            ent = rng.choice(ENTITIES)
            attr = rng.choice(ATTRIBUTES)
            if (ent, attr) not in used:
                used.add((ent, attr))
                break

        ctx = ev_tokens(ent, "has", attr)
        fent = rng.choice([e for e in ENTITIES if e != ent])
        fattr = rng.choice(ATTRIBUTES)
        filler0 = ev_tokens(fent, rng.choice(VERBS), fattr)

        excl = {ent, fent}
        fillers = []
        for _ in range(EVENTS_TOTAL - 2):
            fe = rng.choice([e for e in ENTITIES if e not in excl])
            excl.add(fe)
            fillers.append(ev_tokens(fe, rng.choice(VERBS), rng.choice(ATTRIBUTES)))

        probe_has = ev_tokens(ent, "has", attr)
        probe_is = ev_tokens(ent, "is", attr)

        def mkseq(slot0, probe):
            t = ["<bos>"]
            t.extend(slot0)
            for f in fillers:
                t.extend(f)
            t.extend(probe)
            t.append("<eos>")
            assert len(t) == SEQ_LEN, f"eval len={len(t)}"
            return vocab.encode(t)

        probes.append(
            {
                "entity": ent,
                "attr": attr,
                "copy": mkseq(ctx, probe_has),
                "content": mkseq(ctx, probe_is),
                "base_has": mkseq(filler0, probe_has),
                "base_is": mkseq(filler0, probe_is),
                "target_pos": attr_pos(EVENTS_TOTAL - 1),
                "target_id": vocab.stoi[attr],
            }
        )
    return probes


# ===================== MODEL =====================


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
        self.te = nn.Embedding(V, d)
        self.pe = nn.Embedding(ml, d)
        self.layers = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, x, mask=None):
        B, T = x.shape
        h = self.te(x) + self.pe(torch.arange(T, device=x.device))
        for layer in self.layers:
            h = layer(h, mask=mask)
        return self.head(self.ln(h))


# ===================== MASKS =====================


def causal_mask(L, dev):
    """Standard causal mask. True = blocked (can only attend to past)."""
    return torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), diagonal=1)


def repeat_masked_mask(L, dev):
    """Causal mask + block second→first occurrence of repeated events.

    MODIFY_IDX = [0, 3]:
      Repeat slot 0 (event 6, positions 25-28) blocked from event 0 (positions 1-4)
      Repeat slot 1 (event 7, positions 29-32) blocked from event 3 (positions 13-16)
    """
    m = causal_mask(L, dev)
    for slot, bidx in enumerate(MODIFY_IDX):
        os, oe = ev_start(bidx), ev_end(bidx)
        rs, re = ev_start(BASE_N + slot), ev_end(BASE_N + slot)
        m[rs:re, os:oe] = True
    return m


# ===================== TRAIN & EVAL =====================


def train_one_epoch(model, condition, n_seqs, bs, vocab, rng, opt, dev, masks):
    model.train()
    data = gen_batch(n_seqs, condition, vocab, rng).to(dev)
    mask = masks["repeat_masked"] if condition == "repeat_masked" else masks["causal"]
    tot, nb = 0.0, 0
    for i in range(0, n_seqs, bs):
        b = data[i : i + bs]
        logits = model(b, mask=mask)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, len(vocab)), b[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
        nb += 1
    return tot / nb


@torch.no_grad()
def eval_probes_fn(model, probes, vocab, dev, cmask):
    """Evaluate copy_gain and content_gain on held-out probes."""
    model.eval()
    copy_nll, cont_nll, bh_nll, bi_nll = [], [], [], []
    # batch probes for speed
    all_seqs = []
    for p in probes:
        all_seqs.extend([p["copy"], p["content"], p["base_has"], p["base_is"]])
    batch = torch.tensor(all_seqs, dtype=torch.long, device=dev)
    # forward in chunks
    chunk = 256
    all_logits = []
    for s in range(0, len(batch), chunk):
        all_logits.append(model(batch[s : s + chunk], mask=cmask))
    logits = torch.cat(all_logits, dim=0)
    # extract per-probe
    for idx, p in enumerate(probes):
        base = idx * 4
        tpos = p["target_pos"]
        tid = p["target_id"]
        lp = F.log_softmax(logits[base : base + 4, tpos - 1], dim=-1)
        copy_nll.append(-lp[0, tid].item())
        cont_nll.append(-lp[1, tid].item())
        bh_nll.append(-lp[2, tid].item())
        bi_nll.append(-lp[3, tid].item())
    n = len(probes)
    agg = {
        "copy_nll": round(sum(copy_nll) / n, 5),
        "content_nll": round(sum(cont_nll) / n, 5),
        "base_has_nll": round(sum(bh_nll) / n, 5),
        "base_is_nll": round(sum(bi_nll) / n, 5),
    }
    agg["copy_gain"] = round(agg["base_has_nll"] - agg["copy_nll"], 5)
    agg["content_gain"] = round(agg["base_is_nll"] - agg["content_nll"], 5)
    return agg


# ===================== TRAIN-POSITION PROBE =====================


@torch.no_grad()
def eval_train_positions(model, condition, n_seqs, vocab, rng_state, dev, masks):
    """Measure NLL at modification-slot positions on training-like data.

    Returns mean NLL at base positions and at modification positions.
    This reveals how well the model predicts repeated/varied/wrong tokens.
    """
    model.eval()
    rng = random.Random()
    rng.setstate(rng_state)
    data = gen_batch(n_seqs, condition, vocab, rng).to(dev)
    mask = masks["causal"]  # eval with full attention to see content benefit

    logits = model(data, mask=mask)
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    targets = data[:, 1:]

    # gather target log probs
    tgt_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # base attr positions (0-indexed in shifted targets: attr_pos(slot)-1)
    base_attr_idx = [attr_pos(s) - 1 for s in range(BASE_N)]
    mod_attr_idx = [attr_pos(BASE_N + s) - 1 for s in range(MODIFY_N)]

    base_nll = -tgt_lp[:, base_attr_idx].mean().item()
    mod_nll = -tgt_lp[:, mod_attr_idx].mean().item()

    return {"base_attr_nll": round(base_nll, 5), "mod_attr_nll": round(mod_nll, 5)}


# ===================== MAIN =====================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seqs", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=2)
    ap.add_argument("--nlayers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["unique", "repeat_full", "repeat_masked", "varied", "wrong"],
    )
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab = Vocab()
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {dev}, Vocab: {len(vocab)}, SEQ_LEN: {SEQ_LEN}", flush=True)

    # ---- shared eval probes ----
    eval_rng = random.Random(args.seed + 7777)
    probes = gen_eval_probes(args.n_probes, vocab, eval_rng)
    print(f"Generated {len(probes)} eval probes", flush=True)

    # ---- shared initial model state ----
    torch.manual_seed(args.seed)
    init_model = SmallGPT(len(vocab), args.d_model, args.nhead, args.nlayers, SEQ_LEN + 4)
    n_params = sum(p.numel() for p in init_model.parameters())
    init_state = {k: v.clone() for k, v in init_model.state_dict().items()}
    print(f"Model params: {n_params:,}", flush=True)

    # ---- masks ----
    cmask = causal_mask(SEQ_LEN, dev)
    rmask = repeat_masked_mask(SEQ_LEN, dev)
    masks_d = {"causal": cmask, "repeat_masked": rmask}

    # ---- verify data identity ----
    rng1 = random.Random(args.seed)
    rng2 = random.Random(args.seed)
    d1 = gen_batch(5, "repeat_full", vocab, rng1)
    d2 = gen_batch(5, "repeat_masked", vocab, rng2)
    assert (d1 == d2).all(), "Data mismatch between repeat_full and repeat_masked!"
    diff_count = int((rmask & ~cmask).sum().item())
    print(
        f"✓ repeat_full/repeat_masked data identical; mask blocks {diff_count} extra positions",
        flush=True,
    )

    # ---- print mask details ----
    print("Extra blocked positions in REPEAT_MASKED:", flush=True)
    diff = rmask & ~cmask
    for i in range(SEQ_LEN):
        for j in range(SEQ_LEN):
            if diff[i, j]:
                print(f"  pos {i} blocked from attending pos {j}", flush=True)

    # ---- train each condition ----
    all_res = {}
    for cond in args.conditions:
        print(f"\n{'='*55}\n  CONDITION: {cond}\n{'='*55}", flush=True)

        model = SmallGPT(
            len(vocab), args.d_model, args.nhead, args.nlayers, SEQ_LEN + 4
        ).to(dev)
        model.load_state_dict(init_state)
        model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # repeat_full and repeat_masked share the same data seed
        dseed = args.seed if cond in ("repeat_full", "repeat_masked") else args.seed + hash(cond)
        drng = random.Random(dseed)

        hist = []
        t0 = time.time()

        for ep in range(1, args.epochs + 1):
            # save RNG state for optional train-position probe
            rng_state_before = drng.getstate()
            loss = train_one_epoch(
                model, cond, args.n_seqs, args.bs, vocab, drng, opt, dev, masks_d
            )

            if ep % args.eval_every == 0 or ep == 1 or ep == args.epochs:
                ev = eval_probes_fn(model, probes, vocab, dev, cmask)
                tp = eval_train_positions(
                    model, cond, min(args.n_seqs, 200), vocab, rng_state_before, dev, masks_d
                )
                entry = {
                    "epoch": ep,
                    "train_loss": round(loss, 6),
                    **ev,
                    **tp,
                }
                hist.append(entry)
                print(json.dumps(entry), flush=True)

        elapsed = time.time() - t0
        final = hist[-1] if hist else {}

        all_res[cond] = {
            "condition": cond,
            "elapsed": round(elapsed, 2),
            "history": hist,
            "final": {
                k: final.get(k)
                for k in [
                    "copy_gain",
                    "content_gain",
                    "copy_nll",
                    "content_nll",
                    "base_has_nll",
                    "base_is_nll",
                    "base_attr_nll",
                    "mod_attr_nll",
                    "train_loss",
                ]
            },
        }
        (out / cond).mkdir(exist_ok=True)
        with open(out / cond / "result.json", "w") as f:
            json.dump(all_res[cond], f, indent=2)

    # ---- combined summary ----
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}", flush=True)
    table = []
    for cond in args.conditions:
        r = all_res[cond]
        f = r["final"]
        row = {
            "condition": cond,
            "copy_gain": f.get("copy_gain"),
            "content_gain": f.get("content_gain"),
            "copy_nll": f.get("copy_nll"),
            "content_nll": f.get("content_nll"),
            "base_has_nll": f.get("base_has_nll"),
            "base_is_nll": f.get("base_is_nll"),
            "base_attr_nll": f.get("base_attr_nll"),
            "mod_attr_nll": f.get("mod_attr_nll"),
            "train_loss": f.get("train_loss"),
            "elapsed": r["elapsed"],
        }
        table.append(row)
        cg = f.get("copy_gain", 0) or 0
        ccg = f.get("content_gain", 0) or 0
        tl = f.get("train_loss", 0) or 0
        ba = f.get("base_attr_nll", 0) or 0
        ma = f.get("mod_attr_nll", 0) or 0
        print(
            f"  {cond:16s}  copy_gain={cg:+.4f}  content_gain={ccg:+.4f}  "
            f"train_loss={tl:.4f}  base_attr={ba:.4f}  mod_attr={ma:.4f}  "
            f"elapsed={r['elapsed']:.1f}s",
            flush=True,
        )

    # Key contrast
    print(f"\n  KEY CONTRAST: repeat_full vs repeat_masked", flush=True)
    if "repeat_full" in all_res and "repeat_masked" in all_res:
        rf = all_res["repeat_full"]["final"]
        rm = all_res["repeat_masked"]["final"]
        dcopy = (rf.get("copy_gain") or 0) - (rm.get("copy_gain") or 0)
        dcont = (rf.get("content_gain") or 0) - (rm.get("content_gain") or 0)
        print(f"  copy_gain(full) - copy_gain(masked) = {dcopy:+.4f}", flush=True)
        print(f"  content_gain(full) - content_gain(masked) = {dcont:+.4f}", flush=True)
        if dcopy > 0 and dcont < 0:
            print(
                "  → Identity shortcut SUPPORTED: full attention yields higher copy "
                "but lower content conditioning",
                flush=True,
            )
        elif abs(dcont) < 0.01:
            print(
                "  → Identity shortcut NOT SUPPORTED: content_gain similar, "
                "mechanism may not be within-context copy competition",
                flush=True,
            )
        else:
            print(f"  → Ambiguous: interpret with learning curves", flush=True)

    with open(out / "summary.json", "w") as f:
        json.dump(
            {
                "status": "DONE",
                "args": vars(args),
                "vocab_size": len(vocab),
                "seq_len": SEQ_LEN,
                "n_params": n_params,
                "modify_idx": MODIFY_IDX,
                "conditions": table,
            },
            f,
            indent=2,
        )
    print(json.dumps({"status": "DONE", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
