#!/usr/bin/env python3
"""research functional_learning: matched-support source-specific contrast.

Purpose
-------
research v2 showed large absolute degradation of the unpracticed relation, but the
source-specific decomposition had the opposite sign from relation_learning's BabyLM
recurrence cost: the true source still helped more than the unrelated-source
control. research repairs two synthetic support mismatches before interpreting
that degradation:

1. Source positions: train and probes use the same distribution. Each matched
   final event is at slot 6 or 7 and points to a uniformly sampled source slot
   0..5, with no sorting of source indices.
2. Match/no-match support: relation arms train every sequence with one matched
   final event and one unmatched final event, so the unrelated/no-match probe no
   longer violates a deterministic "final slots always repeat earlier entities"
   training expectation.

The paired intervention remains exact: repeat_full and repeat_masked receive
identical token sequences and optimizer updates; only training attention from the
matched target event to its source event is blocked. The same paired condition is
included for varied_full/varied_masked.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Vocabulary -----------------------------
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

    def encode(self, toks: Sequence[str]) -> List[int]:
        return [self.stoi[t] for t in toks]

    def __len__(self) -> int:
        return len(self.itos)


# ----------------------------- Sequence geometry -----------------------------
EV_TOTAL = 8
BASE_N = 6
EXTRA_N = 2
TOKS_PER_EV = 4
SEQ_LEN = 1 + EV_TOTAL * TOKS_PER_EV + 1


def ev_start(slot: int) -> int:
    return 1 + slot * TOKS_PER_EV


def ev_end(slot: int) -> int:
    return ev_start(slot) + TOKS_PER_EV


def attr_pos(slot: int) -> int:
    return ev_start(slot) + 2


def ev_tok(e: str, v: str, a: str) -> List[str]:
    return [e, v, a, "."]


def flip(v: str) -> str:
    return "is" if v == "has" else "has"


COND_OFFSETS = {
    "unique": 0,
    "support_control": 1000,
    "repeat_full": 2000,
    "repeat_masked": 2000,
    "varied_full": 3000,
    "varied_masked": 3000,
    "wrong_full": 4000,
}
DEFAULT_CONDITIONS = [
    "unique",
    "support_control",
    "repeat_full",
    "repeat_masked",
    "varied_full",
    "varied_masked",
    "wrong_full",
]


def condition_family(cond: str) -> str:
    if cond.startswith("repeat"):
        return "repeat"
    if cond.startswith("varied"):
        return "varied"
    if cond.startswith("wrong"):
        return "wrong"
    if cond == "support_control":
        return "support_control"
    if cond == "unique":
        return "unique"
    raise ValueError(f"unknown condition {cond}")


def is_masked_condition(cond: str) -> bool:
    return cond.endswith("_masked")


# ----------------------------- Data generation -----------------------------
# A block tuple is (source_start, source_end, target_start, target_end). The
# attention mask blocks target-event query rows from source-event key columns.
BlockTuple = Tuple[int, int, int, int]


def _pick_unused(rng: random.Random, used: set[str]) -> str:
    choices = [e for e in ENTITIES if e not in used]
    e = rng.choice(choices)
    used.add(e)
    return e


def gen_sequence(condition: str, vocab: Vocab, rng: random.Random):
    """Generate one train sequence.

    For relation families (repeat/varied/wrong/support_control), each sequence
    has one matched final event and one unmatched final event. The matched final
    event slot is 6 or 7 with equal probability; its source slot is uniform 0..5.
    Source-to-target assignment is not sorted. Unique has two unmatched final
    events and is retained as the no-relation/no-match baseline.
    """
    fam = condition_family(condition)
    used: set[str] = set()

    base: List[Tuple[str, str, str]] = []
    for _ in range(BASE_N):
        base.append((_pick_unused(rng, used), rng.choice(VERBS), rng.choice(ATTRIBUTES)))

    extras: List[Optional[Tuple[str, str, str]]] = [None, None]
    meta: Dict[str, Optional[int | str]] = {
        "family": fam,
        "matched_extra_local": None,
        "unmatched_extra_local": None,
        "source_slot": None,
        "target_slot": None,
    }

    if fam == "unique":
        for j in range(EXTRA_N):
            extras[j] = (_pick_unused(rng, used), rng.choice(VERBS), rng.choice(ATTRIBUTES))
    else:
        matched_local = rng.randrange(EXTRA_N)  # target extra local index: 0=>slot6, 1=>slot7
        unmatched_local = 1 - matched_local
        source_slot = rng.randrange(BASE_N)
        e, v, a = base[source_slot]

        if fam == "repeat":
            matched = (e, v, a)
        elif fam == "varied":
            matched = (e, flip(v), a)
        elif fam == "wrong":
            wrong_a = rng.choice([x for x in ATTRIBUTES if x != a])
            matched = (e, v, wrong_a)
        elif fam == "support_control":
            # Same entity-match support, but target attribute is independent of source.
            # Allow accidental equality with probability 1/N_ATTR so the relation is
            # statistically independent rather than adversarially "not equal".
            matched = (e, rng.choice(VERBS), rng.choice(ATTRIBUTES))
        else:
            raise AssertionError(fam)

        extras[matched_local] = matched
        extras[unmatched_local] = (_pick_unused(rng, used), rng.choice(VERBS), rng.choice(ATTRIBUTES))
        meta.update({
            "matched_extra_local": matched_local,
            "unmatched_extra_local": unmatched_local,
            "source_slot": source_slot,
            "target_slot": BASE_N + matched_local,
        })

    toks = ["<bos>"]
    for e, v, a in base:
        toks.extend(ev_tok(e, v, a))
    for ex in extras:
        assert ex is not None
        toks.extend(ev_tok(*ex))
    toks.append("<eos>")
    assert len(toks) == SEQ_LEN

    blocks: List[BlockTuple] = []
    if is_masked_condition(condition) and fam in ("repeat", "varied", "wrong"):
        assert meta["source_slot"] is not None and meta["target_slot"] is not None
        ss = int(meta["source_slot"])
        ts = int(meta["target_slot"])
        blocks.append((ev_start(ss), ev_end(ss), ev_start(ts), ev_end(ts)))

    return vocab.encode(toks), blocks, meta


def gen_batch(n: int, condition: str, vocab: Vocab, rng: random.Random):
    all_ids, all_blocks, metas = [], [], []
    for _ in range(n):
        ids, blocks, meta = gen_sequence(condition, vocab, rng)
        all_ids.append(ids)
        all_blocks.append(blocks)
        metas.append(meta)
    return torch.tensor(all_ids, dtype=torch.long), all_blocks, metas


# ----------------------------- Evaluation probes -----------------------------
def _filler_event(rng: random.Random, excluded: set[str]) -> List[str]:
    e = _pick_unused(rng, excluded)
    return ev_tok(e, rng.choice(VERBS), rng.choice(ATTRIBUTES))


def build_probe_sequence(
    *,
    rng: random.Random,
    ent: str,
    attr: str,
    source_slot: int,
    target_slot: int,
    source_event: List[str],
    target_verb: str,
) -> List[str]:
    """Build one 8-event sequence with a controlled source and target slot."""
    excluded = {ent}
    # If source_event uses an unrelated entity, exclude it too so target entity
    # remains absent outside the target slot for no-match controls.
    if source_event[0] in ENTITIES:
        excluded.add(source_event[0])
    slots: List[Optional[List[str]]] = [None for _ in range(EV_TOTAL)]
    slots[source_slot] = source_event
    slots[target_slot] = ev_tok(ent, target_verb, attr)
    for s in range(EV_TOTAL):
        if slots[s] is None:
            slots[s] = _filler_event(rng, excluded)
    toks = ["<bos>"]
    for s in range(EV_TOTAL):
        toks.extend(slots[s])
    toks.append("<eos>")
    assert len(toks) == SEQ_LEN
    return toks


def gen_eval_probes(n: int, vocab: Vocab, rng: random.Random):
    """Generate source-specific probes aligned to train source/target support.

    Every probe uses source_slot ~ Uniform{0..5} and target_slot ~ Uniform{6,7},
    matching train. For both copy (has->has) and content/restatement (has->is),
    the true-source sequence has entity e with attribute a in the source. The
    no-match unrelated-source sequence replaces the source event by an unrelated
    entity; the target entity has no prior occurrence, which is now in support
    because relation arms train one unmatched final event in every sequence. A
    wrong-match diagnostic keeps the entity match but gives the source a wrong
    attribute.
    """
    probes = []
    used_pairs: set[Tuple[str, str]] = set()
    for _ in range(n):
        while True:
            ent = rng.choice(ENTITIES)
            attr = rng.choice(ATTRIBUTES)
            if (ent, attr) not in used_pairs:
                used_pairs.add((ent, attr))
                break
        source_slot = rng.randrange(BASE_N)
        target_slot = BASE_N + rng.randrange(EXTRA_N)
        wrong_attr = rng.choice([x for x in ATTRIBUTES if x != attr])
        other_ent = rng.choice([e for e in ENTITIES if e != ent])
        other_attr = rng.choice(ATTRIBUTES)

        true_source = ev_tok(ent, "has", attr)
        unrelated_source = ev_tok(other_ent, rng.choice(VERBS), other_attr)
        wrong_match_source = ev_tok(ent, "has", wrong_attr)

        # Use separate RNG streams for each constructed sequence so filler noise
        # does not accidentally make T and U differ in support except at source.
        state = rng.getstate()
        seqs: Dict[str, List[int]] = {}
        for relation, target_verb in (("copy", "has"), ("content", "is")):
            for source_kind, src_ev in (
                ("T", true_source),
                ("U_nomatch", unrelated_source),
                ("U_wrongmatch", wrong_match_source),
            ):
                rng.setstate(state)
                toks = build_probe_sequence(
                    rng=rng,
                    ent=ent,
                    attr=attr,
                    source_slot=source_slot,
                    target_slot=target_slot,
                    source_event=src_ev,
                    target_verb=target_verb,
                )
                seqs[f"{relation}_{source_kind}"] = vocab.encode(toks)
        # Advance the outer RNG after reusing the filler state.
        rng.random()

        probes.append({
            "entity": ent,
            "attr": attr,
            "source_slot": source_slot,
            "target_slot": target_slot,
            "target_pos": attr_pos(target_slot),
            "target_id": vocab.stoi[attr],
            **seqs,
        })
    return probes


# ----------------------------- Model -----------------------------
class Block(nn.Module):
    def __init__(self, d: int, nh: int, dff: Optional[int] = None):
        super().__init__()
        dff = dff or 4 * d
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + h
        return x + self.ff(self.ln2(x))


class SmallGPT(nn.Module):
    def __init__(self, V: int, d: int = 64, nh: int = 2, nl: int = 3, ml: int = 64):
        super().__init__()
        self.d = d
        self.nh = nh
        self.te = nn.Embedding(V, d)
        self.pe = nn.Embedding(ml, d)
        self.layers = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T = x.shape
        h = self.te(x) + self.pe(torch.arange(T, device=x.device))
        for layer in self.layers:
            h = layer(h, mask=mask)
        return self.head(self.ln(h))


# ----------------------------- Masks -----------------------------
def causal_mask(L: int, dev: torch.device):
    return torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), diagonal=1)


def build_3d_mask(batch_blocks: List[List[BlockTuple]], seq_len: int, nhead: int, dev: torch.device):
    B = len(batch_blocks)
    base = causal_mask(seq_len, dev)
    mask = base.unsqueeze(0).expand(B, -1, -1).clone()
    for i, blocks in enumerate(batch_blocks):
        for source_start, source_end, target_start, target_end in blocks:
            mask[i, target_start:target_end, source_start:source_end] = True
    return mask.unsqueeze(1).expand(-1, nhead, -1, -1).reshape(B * nhead, seq_len, seq_len)


# ----------------------------- Training/evaluation -----------------------------
def train_one_epoch(model, condition: str, n_seqs: int, bs: int, vocab: Vocab, rng: random.Random, opt, dev):
    model.train()
    data, blocks, metas = gen_batch(n_seqs, condition, vocab, rng)
    data = data.to(dev)
    cmask = causal_mask(SEQ_LEN, dev)
    total, nb = 0.0, 0
    for i in range(0, n_seqs, bs):
        b = data[i:i + bs]
        if is_masked_condition(condition):
            mask = build_3d_mask(blocks[i:i + len(b)], SEQ_LEN, model.nh, dev)
        else:
            mask = cmask
        logits = model(b, mask=mask)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, len(vocab)), b[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.item())
        nb += 1
    return total / max(nb, 1)


@torch.no_grad()
def eval_probes_fn(model, probes, vocab: Vocab, dev: torch.device):
    model.eval()
    cmask = causal_mask(SEQ_LEN, dev)
    keys = [
        "copy_T", "copy_U_nomatch", "copy_U_wrongmatch",
        "content_T", "content_U_nomatch", "content_U_wrongmatch",
    ]
    all_seqs = []
    for p in probes:
        for k in keys:
            all_seqs.append(p[k])
    batch = torch.tensor(all_seqs, dtype=torch.long, device=dev)
    all_logits = []
    for s in range(0, len(batch), 256):
        all_logits.append(model(batch[s:s + 256], mask=cmask))
    logits = torch.cat(all_logits, dim=0)
    vals = {k: [] for k in keys}
    for idx, p in enumerate(probes):
        base = idx * len(keys)
        tpos = p["target_pos"]
        tid = p["target_id"]
        lp = F.log_softmax(logits[base:base + len(keys), tpos - 1], dim=-1)
        for j, k in enumerate(keys):
            vals[k].append(-float(lp[j, tid].item()))
    out = {f"{k}_nll": round(sum(v) / len(v), 5) for k, v in vals.items()}
    out["copy_gain_nomatch"] = round(out["copy_U_nomatch_nll"] - out["copy_T_nll"], 5)
    out["content_gain_nomatch"] = round(out["content_U_nomatch_nll"] - out["content_T_nll"], 5)
    out["copy_gain_wrongmatch"] = round(out["copy_U_wrongmatch_nll"] - out["copy_T_nll"], 5)
    out["content_gain_wrongmatch"] = round(out["content_U_wrongmatch_nll"] - out["content_T_nll"], 5)
    return out


@torch.no_grad()
def eval_train_positions(model, condition: str, n_seqs: int, vocab: Vocab, rng_state, dev: torch.device):
    model.eval()
    rng = random.Random()
    rng.setstate(rng_state)
    data, blocks, metas = gen_batch(n_seqs, condition, vocab, rng)
    data = data.to(dev)
    logits = model(data, mask=causal_mask(SEQ_LEN, dev))
    lp = F.log_softmax(logits[:, :-1], dim=-1)
    targets = data[:, 1:]
    tgt_lp = lp.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    base_positions = [attr_pos(s) - 1 for s in range(BASE_N)]
    base_nll = -tgt_lp[:, base_positions].mean().item()
    extra_positions = [attr_pos(BASE_N + j) - 1 for j in range(EXTRA_N)]
    extra_nll = -tgt_lp[:, extra_positions].mean().item()

    matched_vals, unmatched_vals = [], []
    for row, meta in enumerate(metas):
        if meta["matched_extra_local"] is None:
            unmatched_vals.extend([-tgt_lp[row, pos].item() for pos in extra_positions])
        else:
            mloc = int(meta["matched_extra_local"])
            uloc = int(meta["unmatched_extra_local"])
            matched_vals.append(-tgt_lp[row, attr_pos(BASE_N + mloc) - 1].item())
            unmatched_vals.append(-tgt_lp[row, attr_pos(BASE_N + uloc) - 1].item())

    def mean_or_nan(xs):
        return float("nan") if not xs else sum(xs) / len(xs)

    return {
        "base_attr_nll": round(base_nll, 5),
        "extra_attr_nll": round(extra_nll, 5),
        "matched_extra_attr_nll": round(mean_or_nan(matched_vals), 5) if matched_vals else None,
        "unmatched_extra_attr_nll": round(mean_or_nan(unmatched_vals), 5) if unmatched_vals else None,
    }


def contrast_terms(table: Dict[str, Dict], a: str, b: str, relation: str = "content", u_kind: str = "nomatch"):
    T_key = f"{relation}_T_nll"
    U_key = f"{relation}_U_{u_kind}_nll"
    gain_key = f"{relation}_gain_{u_kind}"
    fa, fb = table[a]["final"], table[b]["final"]
    t_delta = fa[T_key] - fb[T_key]
    u_delta = fa[U_key] - fb[U_key]
    excess_true_cost = t_delta - u_delta
    gain_delta = fa[gain_key] - fb[gain_key]
    return {
        "A": a,
        "B": b,
        "relation": relation,
        "u_kind": u_kind,
        "T_delta_A_minus_B": round(t_delta, 5),
        "U_delta_A_minus_B": round(u_delta, 5),
        "excess_true_cost": round(excess_true_cost, 5),
        "gain_delta_A_minus_B": round(gain_delta, 5),
    }


def markdown_for_seed(seed: int, args, result: Dict) -> str:
    rows = result["conditions"]
    lines = []
    lines.append(f"# research matched-support source contrast — seed {seed}\n")
    lines.append("Train/probe support repair: one matched and one unmatched final slot per relation-arm sequence; matched target slot is uniformly slot 6/7; source slot is uniform 0..5 and not sorted. Probes use the same support. `U_nomatch` is now in train support for relation arms. `U_wrongmatch` is an extra diagnostic with the same entity but a wrong source attribute.\n")
    lines.append("## Final epoch table\n")
    header = [
        "condition", "copy_gain_no", "content_gain_no", "copy_T", "copy_U_no",
        "content_T", "content_U_no", "content_U_wrong", "matched_train", "unmatched_train", "loss",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in rows:
        f = r
        vals = [
            r["condition"],
            f"{f.get('copy_gain_nomatch', float('nan')):+.3f}",
            f"{f.get('content_gain_nomatch', float('nan')):+.3f}",
            f"{f.get('copy_T_nll', float('nan')):.3f}",
            f"{f.get('copy_U_nomatch_nll', float('nan')):.3f}",
            f"{f.get('content_T_nll', float('nan')):.3f}",
            f"{f.get('content_U_nomatch_nll', float('nan')):.3f}",
            f"{f.get('content_U_wrongmatch_nll', float('nan')):.3f}",
            "" if f.get("matched_extra_attr_nll") is None else f"{f.get('matched_extra_attr_nll'):.3f}",
            "" if f.get("unmatched_extra_attr_nll") is None else f"{f.get('unmatched_extra_attr_nll'):.3f}",
            f"{f.get('train_loss', float('nan')):.3f}",
        ]
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("\n## Source-specific contrasts\n")
    lines.append("Positive `excess_true_cost` means A is worse on true-source use beyond its unrelated-source change: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.\n")
    lines.append("| A | B | relation | U | T_delta | U_delta | excess_true_cost | gain_delta |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for c in result["contrasts"]:
        lines.append(
            f"| {c['A']} | {c['B']} | {c['relation']} | {c['u_kind']} | "
            f"{c['T_delta_A_minus_B']:+.3f} | {c['U_delta_A_minus_B']:+.3f} | "
            f"{c['excess_true_cost']:+.3f} | {c['gain_delta_A_minus_B']:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def run_seed(seed: int, args, out_root: Path):
    seed_out = out_root / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    vocab = Vocab()
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"\n{'#'*80}\nSeed {seed} | device={dev} | vocab={len(vocab)} | seq_len={SEQ_LEN}\n{'#'*80}", flush=True)

    eval_rng = random.Random(seed + 7777)
    probes = gen_eval_probes(args.n_probes, vocab, eval_rng)
    print(f"Generated {len(probes)} probes; source_slots={sorted(set(p['source_slot'] for p in probes))}; target_slots={sorted(set(p['target_slot'] for p in probes))}", flush=True)

    torch.manual_seed(seed)
    init_model = SmallGPT(len(vocab), args.d_model, args.nhead, args.nlayers, SEQ_LEN + 4)
    n_params = sum(p.numel() for p in init_model.parameters())
    init_state = {k: v.clone() for k, v in init_model.state_dict().items()}
    print(f"Params: {n_params:,}", flush=True)

    # Verify paired data identity for full/masked interventions.
    for a, b in (("repeat_full", "repeat_masked"), ("varied_full", "varied_masked")):
        rng_a = random.Random(seed + COND_OFFSETS[a])
        rng_b = random.Random(seed + COND_OFFSETS[b])
        da, _, ma = gen_batch(16, a, vocab, rng_a)
        db, bb, mb = gen_batch(16, b, vocab, rng_b)
        assert torch.equal(da, db), f"paired data mismatch: {a} vs {b}"
        print(f"✓ paired data identical for {a}/{b}; first block={bb[0]}; first meta={mb[0]}", flush=True)

    all_res: Dict[str, Dict] = {}
    for cond in args.conditions:
        print(f"\n{'='*64}\n  {cond}\n{'='*64}", flush=True)
        model = SmallGPT(len(vocab), args.d_model, args.nhead, args.nlayers, SEQ_LEN + 4).to(dev)
        model.load_state_dict(init_state)
        model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        drng = random.Random(seed + COND_OFFSETS[cond])
        hist = []
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            rng_state = drng.getstate()
            loss = train_one_epoch(model, cond, args.n_seqs, args.bs, vocab, drng, opt, dev)
            if ep == 1 or ep == args.epochs or ep % args.eval_every == 0:
                ev = eval_probes_fn(model, probes, vocab, dev)
                tp = eval_train_positions(model, cond, min(args.n_seqs, 256), vocab, rng_state, dev)
                entry = {"epoch": ep, "train_loss": round(loss, 6), **ev, **tp}
                hist.append(entry)
                print(json.dumps(entry), flush=True)
        elapsed = time.time() - t0
        final = hist[-1]
        final_keys = [
            "copy_T_nll", "copy_U_nomatch_nll", "copy_U_wrongmatch_nll",
            "content_T_nll", "content_U_nomatch_nll", "content_U_wrongmatch_nll",
            "copy_gain_nomatch", "content_gain_nomatch",
            "copy_gain_wrongmatch", "content_gain_wrongmatch",
            "base_attr_nll", "extra_attr_nll", "matched_extra_attr_nll", "unmatched_extra_attr_nll",
            "train_loss",
        ]
        all_res[cond] = {
            "condition": cond,
            "elapsed": round(elapsed, 2),
            "history": hist,
            "final": {k: final.get(k) for k in final_keys},
        }
        cdir = seed_out / cond
        cdir.mkdir(exist_ok=True)
        (cdir / "result.json").write_text(json.dumps(all_res[cond], indent=2), encoding="utf-8")

    table = []
    for cond in args.conditions:
        f = all_res[cond]["final"]
        row = {"condition": cond, **f, "elapsed": all_res[cond]["elapsed"]}
        table.append(row)

    contrasts = []
    available = set(args.conditions)
    for rel in ("content", "copy"):
        for u_kind in ("nomatch", "wrongmatch"):
            for A, B in (
                ("repeat_full", "repeat_masked"),
                ("varied_full", "varied_masked"),
                ("repeat_full", "unique"),
                ("repeat_full", "support_control"),
                ("varied_full", "unique"),
                ("varied_full", "support_control"),
                ("repeat_full", "varied_full"),
            ):
                if A in available and B in available:
                    contrasts.append(contrast_terms(all_res, A, B, relation=rel, u_kind=u_kind))

    result = {
        "status": "SEED_DONE",
        "seed": seed,
        "args": vars(args),
        "vocab_size": len(vocab),
        "seq_len": SEQ_LEN,
        "n_params": n_params,
        "design": "matched_source_positions_one_match_one_nomatch_train_support",
        "conditions": table,
        "contrasts": contrasts,
    }
    (seed_out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (seed_out / "analysis.md").write_text(markdown_for_seed(seed, args, result), encoding="utf-8")

    print(f"\n{'='*80}\nSeed {seed} summary\n{'='*80}", flush=True)
    for row in table:
        print(
            f"{row['condition']:17s} copy_gain={row['copy_gain_nomatch']:+.4f} "
            f"content_gain={row['content_gain_nomatch']:+.4f} "
            f"content_T={row['content_T_nll']:.4f} U_no={row['content_U_nomatch_nll']:.4f} "
            f"matched_train={row.get('matched_extra_attr_nll')} loss={row['train_loss']:.4f}",
            flush=True,
        )
    print("\nContent nomatch contrasts:", flush=True)
    for c in contrasts:
        if c["relation"] == "content" and c["u_kind"] == "nomatch":
            print(json.dumps(c), flush=True)
    return result


def aggregate_results(seed_results: List[Dict]) -> Dict:
    # Simple per-condition mean/sd for the most important final metrics and contrasts.
    agg: Dict[str, Dict] = {"conditions": {}, "contrasts": {}}
    metric_keys = [
        "copy_gain_nomatch", "content_gain_nomatch", "copy_T_nll", "copy_U_nomatch_nll",
        "content_T_nll", "content_U_nomatch_nll", "matched_extra_attr_nll", "unmatched_extra_attr_nll",
        "train_loss",
    ]
    conds = [r["condition"] for r in seed_results[0]["conditions"]]
    for cond in conds:
        agg["conditions"][cond] = {}
        for k in metric_keys:
            vals = []
            for sr in seed_results:
                row = next(x for x in sr["conditions"] if x["condition"] == cond)
                v = row.get(k)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    vals.append(float(v))
            if vals:
                mean = sum(vals) / len(vals)
                sd = 0.0 if len(vals) == 1 else math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
                agg["conditions"][cond][k] = {"mean": round(mean, 5), "sd": round(sd, 5), "n": len(vals), "values": vals}

    # Aggregate contrasts by tuple.
    keys = set()
    for sr in seed_results:
        for c in sr["contrasts"]:
            keys.add((c["A"], c["B"], c["relation"], c["u_kind"]))
    for key in sorted(keys):
        vals_by_metric = {"T_delta_A_minus_B": [], "U_delta_A_minus_B": [], "excess_true_cost": [], "gain_delta_A_minus_B": []}
        for sr in seed_results:
            matches = [c for c in sr["contrasts"] if (c["A"], c["B"], c["relation"], c["u_kind"]) == key]
            if matches:
                c = matches[0]
                for m in vals_by_metric:
                    vals_by_metric[m].append(float(c[m]))
        label = "|".join(key)
        agg["contrasts"][label] = {}
        for m, vals in vals_by_metric.items():
            if vals:
                mean = sum(vals) / len(vals)
                sd = 0.0 if len(vals) == 1 else math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
                agg["contrasts"][label][m] = {"mean": round(mean, 5), "sd": round(sd, 5), "n": len(vals), "values": vals}
    return agg


def aggregate_markdown(args, seed_results: List[Dict], agg: Dict) -> str:
    lines = []
    lines.append("# research matched-support source-specific contrast\n")
    lines.append("This experiment repairs the research v2 support mismatch before interpreting absolute degradation. Relation arms train with one matched and one unmatched final event per sequence. Matched target slot is uniformly 6/7; source slot is uniformly 0..5 and is not sorted. Probes use the same positional distribution. `repeat_full`/`repeat_masked` and `varied_full`/`varied_masked` have identical token data within each pair; the only difference is the training attention block from matched target event to its source event.\n")
    lines.append(f"Seeds: {args.seeds}; epochs={args.epochs}; n_seqs={args.n_seqs}; probes={args.n_probes}; model d={args.d_model}, heads={args.nhead}, layers={args.nlayers}.\n")
    lines.append("## Mean final metrics\n")
    headers = ["condition", "copy_gain", "content_gain", "content_T", "content_U", "matched_train", "unmatched_train", "loss"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for cond, d in agg["conditions"].items():
        def fmt(k, signed=False):
            x = d.get(k, {})
            if not x:
                return ""
            mean = x["mean"]
            sd = x["sd"]
            return f"{mean:+.3f}±{sd:.3f}" if signed else f"{mean:.3f}±{sd:.3f}"
        lines.append("| " + " | ".join([
            cond,
            fmt("copy_gain_nomatch", True),
            fmt("content_gain_nomatch", True),
            fmt("content_T_nll"),
            fmt("content_U_nomatch_nll"),
            fmt("matched_extra_attr_nll"),
            fmt("unmatched_extra_attr_nll"),
            fmt("train_loss"),
        ]) + " |")
    lines.append("\n## Mean source-specific contrasts using U_nomatch\n")
    lines.append("Positive excess true-source cost means the true source is worse relative to the unrelated-source control: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.\n")
    lines.append("| contrast | relation | T_delta | U_delta | excess_true_cost | gain_delta |")
    lines.append("|---|---|---:|---:|---:|---:|")
    wanted = [
        ("repeat_full", "repeat_masked"), ("varied_full", "varied_masked"),
        ("repeat_full", "unique"), ("repeat_full", "support_control"),
        ("varied_full", "unique"), ("varied_full", "support_control"),
        ("repeat_full", "varied_full"),
    ]
    for rel in ("content", "copy"):
        for A, B in wanted:
            label = f"{A}|{B}|{rel}|nomatch"
            if label not in agg["contrasts"]:
                continue
            cd = agg["contrasts"][label]
            def cfmt(m):
                x = cd[m]
                return f"{x['mean']:+.3f}±{x['sd']:.3f}"
            lines.append(f"| {A} − {B} | {rel} | {cfmt('T_delta_A_minus_B')} | {cfmt('U_delta_A_minus_B')} | {cfmt('excess_true_cost')} | {cfmt('gain_delta_A_minus_B')} |")
    lines.append("\n## Per-seed files\n")
    for sr in seed_results:
        seed = sr["seed"]
        lines.append(f"- seed {seed}: `seed{seed}/analysis.md`, `seed{seed}/summary.json`")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--n-seqs", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=2)
    ap.add_argument("--nlayers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-probes", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seed_results = []
    t0 = time.time()
    for seed in args.seeds:
        seed_results.append(run_seed(seed, args, out_root))
    agg = aggregate_results(seed_results)
    final = {
        "status": "DONE",
        "args": vars(args),
        "elapsed": round(time.time() - t0, 2),
        "seed_results": seed_results,
        "aggregate": agg,
    }
    (out_root / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (out_root / "analysis.md").write_text(aggregate_markdown(args, seed_results, agg), encoding="utf-8")
    print(json.dumps({"status": "DONE", "out_dir": str(out_root), "elapsed": final["elapsed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
