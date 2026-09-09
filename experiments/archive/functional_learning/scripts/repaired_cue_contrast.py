#!/usr/bin/env python3
"""research repaired effective-input contrast: nuisance-cue invariance.

Scientific purpose
------------------
research's VARY_CONTEXT varied surface names, but PosAlign hard matching replaces
names by candidate/other vectors before the GRU.  This script therefore works at
that effective-input level directly: tokens equal to the current candidate are
canonicalized as <C>, tokens equal to the other entity as <O>.

We test a sharper question: is varied experience useful for acquiring invariance
to a nuisance feature that actually reaches the relational learner?  We inject
visible relation-cue tokens into TRAIN events only and compare:

  clean              : no cue; supplied name-invariance only.
  cue_locked         : a relation-specific cue is prepended to every train event.
                       The cue is predictive in train but absent at no-cue eval.
  cue_locked_noise   : same relation-specific cue plus random neutral filler;
                       diverse strings but the shortcut remains intact.
  cue_varied_dropout : cue identity is randomized independently of relation and
                       sometimes omitted; this breaks the cue shortcut while
                       matching the presence of visible nuisance tokens.

Evaluation is reported along the training budget curve on held-out no-cue events
and, separately, on cue-matched eval events.  If cue_locked succeeds only when
the cue is also present at eval, it has learned the spurious visible coordinate;
if cue_varied_dropout recovers no-cue h1/h3 transport, variation is useful here
because it installs cue invariance, not because diversity is intrinsically good.

This is a fast pilot, not a replacement for the full PosAlign harness: it removes
char-matcher/equality pretraining and simulates the frozen effective coordinate.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

A01_SCRIPT_DIR = Path("experiments/archive/representation_and_objectives/training/scripts")
sys.path.insert(0, str(A01_SCRIPT_DIR))
import raw_span_discovery_probe as base  # noqa: E402

DEFAULT_OUT = Path("experiments/archive/functional_learning/data/repaired_cue_contrast")

REL_CUES = {
    "h0_dax": "relalpha",
    "h1_mep": "relbravo",
    "h2_norp": "relcharlie",
    "h3_ziv": "reldelta",
    "s_give": "relecho",
    "s_receive": "relfoxtrot",
}
CUE_VOCAB = list(REL_CUES.values())
NOISE_TOKENS = [
    "noisecedar", "noisebirch", "noiseelm", "noisefir", "noiseash",
    "noisepine", "noisemaple", "noiseoak", "noiseivy", "noisemoss",
]
CONDITIONS = [
    "clean", "cue_locked", "cue_locked_noise", "cue_varied_dropout",
    # Comp-only masked-verb variants: state anchors remain clean; comparison
    # evidence receives cue manipulations and the true relation verb is hidden in
    # most comparison events, making the visible cue a genuinely attractive
    # shortcut for graph transport.
    "comp_clean_masked", "comp_cue_locked_masked",
    "comp_cue_locked_noise_masked", "comp_cue_varied_dropout_masked",
    # Pair-token shortcut variants: comparison rows get a shared packet token in
    # both events.  This is an actually visible shortcut for the positive
    # comparison task and is absent at eval.
    "packet_locked", "packet_locked_noise", "packet_broken",
]
RELATIONS = ["h0_dax", "h1_mep", "h2_norp", "h3_ziv"]
DIRECT_RELS = {"h0_dax", "h2_norp"}
RELATION_VERB_FORMS = {
    "h0_dax": "daxed",
    "h1_mep": "meped",
    "h2_norp": "norped",
    "h3_ziv": "zived",
    "s_give": "gave",
    "s_receive": "received",
}
MASKED_VERB = "relverbmask"
MASK_PROB = 0.75
PACKET_TOKENS = [f"pkt{a}{b}{c}" for a in "abcdef" for b in "abcdef" for c in "abcdef"]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


@dataclass
class EventItem:
    text: str
    names: Tuple[str, str]
    relation: str
    target_idx: Optional[int] = None
    is_direct_anchor: bool = False
    source_id: str = ""


@dataclass
class CompItem:
    event1: EventItem
    event2: EventItem
    label: bool
    source_id: str = ""


class Vocab:
    def __init__(self):
        self.stoi = {"<pad>": 0, "<unk>": 1, "<C>": 2, "<O>": 3}
        self.itos = ["<pad>", "<unk>", "<C>", "<O>"]

    def add_tokens(self, toks: Sequence[str]) -> None:
        for t in toks:
            if t not in self.stoi:
                self.stoi[t] = len(self.itos)
                self.itos.append(t)

    def encode(self, toks: Sequence[str]) -> List[int]:
        return [self.stoi.get(t, 1) for t in toks]


def raw_tokens(text: str) -> List[str]:
    toks, _ = base.raw_tokenize(text)
    return toks


def cue_mode(condition: str) -> str:
    if condition.startswith("comp_"):
        core = condition[len("comp_"):]
        if core == "clean_masked":
            return "clean"
        if core == "cue_locked_masked":
            return "cue_locked"
        if core == "cue_locked_noise_masked":
            return "cue_locked_noise"
        if core == "cue_varied_dropout_masked":
            return "cue_varied_dropout"
    return condition


def mask_relation_verb(text: str, relation: str) -> str:
    v = RELATION_VERB_FORMS.get(relation)
    if not v:
        return text
    return re_sub_word(text, v, MASKED_VERB)


def re_sub_word(text: str, old: str, new: str) -> str:
    import re
    return re.sub(r"\b" + re.escape(old) + r"\b", new, text)


def add_train_cue(text: str, relation: str, condition: str, rng: random.Random) -> str:
    """Add condition-specific visible nuisance tokens to a train event."""
    condition = cue_mode(condition)
    if condition == "clean":
        return text
    if condition == "cue_locked":
        return f"{REL_CUES[relation]} {text}"
    if condition == "cue_locked_noise":
        return f"{REL_CUES[relation]} {rng.choice(NOISE_TOKENS)} {text} {rng.choice(NOISE_TOKENS)}"
    if condition == "cue_varied_dropout":
        # Include no-cue cases so no-cue eval is inside the learned invariance
        # support, while all cue identities are independent of the relation.
        cue_options = [None] + CUE_VOCAB
        cue = rng.choice(cue_options)
        if cue is None:
            return text
        # Add one neutral token half the time to avoid cue-position becoming a
        # perfectly constant feature.
        if rng.random() < 0.5:
            return f"{cue} {rng.choice(NOISE_TOKENS)} {text}"
        return f"{cue} {text}"
    raise ValueError(condition)


def packet_token(source_id: str) -> str:
    # Stable mapping from row id to one packet token.  Python's hash is salted,
    # so use a simple deterministic character sum instead.
    idx = sum((i + 1) * ord(ch) for i, ch in enumerate(source_id)) % len(PACKET_TOKENS)
    return PACKET_TOKENS[idx]


def add_matched_eval_cue(text: str, relation: str) -> str:
    return f"{REL_CUES[relation]} {text}"


def add_wrong_eval_cue(text: str, relation: str) -> str:
    rels = ["h0_dax", "h1_mep", "h2_norp", "h3_ziv", "s_give", "s_receive"]
    i = rels.index(relation) if relation in rels else 0
    wrong_rel = rels[(i + 1) % len(rels)]
    return f"{REL_CUES[wrong_rel]} {text}"


def effective_tokens(text: str, cand: str, other: str) -> Tuple[str, ...]:
    cand_l = cand.lower(); other_l = other.lower()
    out = []
    for t in raw_tokens(text):
        if t == cand_l:
            out.append("<C>")
        elif t == other_l:
            out.append("<O>")
        else:
            out.append(t)
    return tuple(out)


class SharedEventModel(nn.Module):
    """Effective-input analogue of the shared-trunk PosAlign model."""
    def __init__(self, vocab_size: int, emb_dim: int = 48, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.state_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.cmp_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def encode_batch(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        y, _ = self.gru(x)
        m = mask.unsqueeze(-1).float()
        return (y * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    def score_pair(self, batch_ids: torch.Tensor, batch_mask: torch.Tensor, head: str) -> torch.Tensor:
        h = self.encode_batch(batch_ids, batch_mask)
        hd = self.state_head if head == "state" else self.cmp_head
        return hd(h).squeeze(-1)


class EncCache:
    def __init__(self, vocab: Vocab, device: torch.device):
        self.vocab = vocab
        self.device = device
        self.cache: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def encode_event_two_candidates(self, event: EventItem) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (event.text, event.names[0], event.names[1])
        if key in self.cache:
            return self.cache[key]
        seqs = []
        for i, cand in enumerate(event.names):
            other = event.names[1 - i]
            toks = effective_tokens(event.text, cand, other)
            seqs.append(torch.tensor(self.vocab.encode(toks), dtype=torch.long))
        max_len = max(len(s) for s in seqs)
        ids = torch.zeros((2, max_len), dtype=torch.long)
        mask = torch.zeros((2, max_len), dtype=torch.bool)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = s
            mask[i, :len(s)] = True
        ids = ids.to(self.device); mask = mask.to(self.device)
        self.cache[key] = (ids, mask)
        return ids, mask


def load_items() -> Tuple[List[EventItem], List[CompItem], List[EventItem]]:
    ts, tc, es, ec, pe, counts = base.load_dataset(base.DEFAULT_DATA_ROOT, base.DEFAULT_ARM)
    if pe:
        print(f"Parse warnings: {pe[:3]}", file=sys.stderr)
    train_states = []
    for q in ts:
        # Keep changed event queries. Static unchanged rows are not part of the
        # coordinate-transport mechanism and would require an extra scorer.
        if not getattr(q, "is_changed", False):
            continue
        train_states.append(EventItem(
            text=q.event,
            names=tuple(q.names),
            relation=q.relation,
            target_idx=base.get_target_idx(q),
            is_direct_anchor=bool(q.is_direct_anchor),
            source_id=getattr(q, "row_id", getattr(q, "key", "")),
        ))
    train_comps = []
    for c in tc:
        e1 = EventItem(c.event1, tuple(c.names), c.relation1, source_id=c.row_id + ":1")
        e2 = EventItem(c.event2, tuple(c.names), c.relation2, source_id=c.row_id + ":2")
        train_comps.append(CompItem(e1, e2, bool(c.label), source_id=c.row_id))
    eval_states = []
    for q in es:
        if not getattr(q, "is_changed", False):
            continue
        eval_states.append(EventItem(
            text=q.event,
            names=tuple(q.names),
            relation=q.relation,
            target_idx=base.get_target_idx(q),
            is_direct_anchor=bool(q.is_direct_anchor),
            source_id=getattr(q, "row_id", getattr(q, "key", "")),
        ))
    return train_states, train_comps, eval_states


def transformed_epoch(train_states: List[EventItem], train_comps: List[CompItem], condition: str,
                      seed: int, epoch: int) -> Tuple[List[EventItem], List[CompItem]]:
    rng = random.Random(seed * 1000003 + epoch * 7919)
    comp_only = condition.startswith("comp_") or condition.startswith("packet_")
    masked = condition.startswith("comp_") and condition.endswith("_masked")
    out_states = []
    for e in train_states:
        ee = copy.copy(e)
        # In comp-only and packet variants, direct state anchors stay clean so
        # the intervention specifically targets graph/comparison evidence.
        ee.text = e.text if comp_only else add_train_cue(e.text, e.relation, condition, rng)
        out_states.append(ee)
    out_comps = []
    for c in train_comps:
        e1 = copy.copy(c.event1); e2 = copy.copy(c.event2)
        t1, t2 = e1.text, e2.text
        if masked and rng.random() < MASK_PROB:
            t1 = mask_relation_verb(t1, e1.relation)
        if masked and rng.random() < MASK_PROB:
            t2 = mask_relation_verb(t2, e2.relation)
        if condition == "packet_locked":
            pkt = packet_token(c.source_id)
            e1.text = f"{pkt} {t1}"
            e2.text = f"{pkt} {t2}"
        elif condition == "packet_locked_noise":
            pkt = packet_token(c.source_id)
            e1.text = f"{pkt} {rng.choice(NOISE_TOKENS)} {t1}"
            e2.text = f"{pkt} {rng.choice(NOISE_TOKENS)} {t2}"
        elif condition == "packet_broken":
            # Same amount of packet-like visible material, but independent across
            # the two events and changing by epoch, so it cannot identify the row.
            p1 = rng.choice(PACKET_TOKENS)
            p2 = rng.choice(PACKET_TOKENS)
            e1.text = f"{p1} {rng.choice(NOISE_TOKENS)} {t1}"
            e2.text = f"{p2} {rng.choice(NOISE_TOKENS)} {t2}"
        else:
            e1.text = add_train_cue(t1, e1.relation, condition, rng)
            e2.text = add_train_cue(t2, e2.relation, condition, rng)
        out_comps.append(CompItem(e1, e2, c.label, c.source_id))
    return out_states, out_comps


def eval_variant(eval_states: List[EventItem], variant: str) -> List[EventItem]:
    out = []
    for e in eval_states:
        ee = copy.copy(e)
        if variant == "no_cue":
            pass
        elif variant == "matched_cue":
            ee.text = add_matched_eval_cue(ee.text, ee.relation)
        elif variant == "wrong_cue":
            ee.text = add_wrong_eval_cue(ee.text, ee.relation)
        else:
            raise ValueError(variant)
        out.append(ee)
    return out


def build_vocab(train_states: List[EventItem], train_comps: List[CompItem], eval_states: List[EventItem]) -> Vocab:
    vocab = Vocab()
    all_texts = []
    for e in train_states + eval_states:
        all_texts.append((e.text, e.names))
        all_texts.append((add_matched_eval_cue(e.text, e.relation), e.names))
        for cue in CUE_VOCAB:
            all_texts.append((f"{cue} {e.text}", e.names))
        for nt in NOISE_TOKENS:
            all_texts.append((f"{cue if (cue := CUE_VOCAB[0]) else ''} {nt} {e.text} {nt}", e.names))
    for c in train_comps:
        for e in [c.event1, c.event2]:
            all_texts.append((e.text, e.names))
            all_texts.append((add_matched_eval_cue(e.text, e.relation), e.names))
            for cue in CUE_VOCAB:
                all_texts.append((f"{cue} {e.text}", e.names))
            for nt in NOISE_TOKENS:
                all_texts.append((f"{CUE_VOCAB[0]} {nt} {e.text} {nt}", e.names))
    for text, names in all_texts:
        for cand in names:
            other = names[1 - list(names).index(cand)]
            vocab.add_tokens(effective_tokens(text, cand, other))
    vocab.add_tokens(CUE_VOCAB + NOISE_TOKENS + [MASKED_VERB] + PACKET_TOKENS)
    return vocab


def score_event(model: SharedEventModel, cache: EncCache, event: EventItem, head: str) -> torch.Tensor:
    ids, mask = cache.encode_event_two_candidates(event)
    return model.score_pair(ids, mask, head=head)


def eval_states(model: SharedEventModel, cache: EncCache, states: List[EventItem], bridge_sign: int,
                head: str = "state") -> dict:
    by_rel = defaultdict(lambda: {"n": 0, "correct": 0, "margins": []})
    model.eval()
    with torch.no_grad():
        for e in states:
            sc = score_event(model, cache, e, head=head)
            target = int(e.target_idx)
            if bridge_sign == -1 and e.is_direct_anchor:
                sc_eff = sc.flip(0)
            else:
                sc_eff = sc
            ok = int(sc_eff.argmax().item() == target)
            margin = float((sc_eff[target] - sc_eff[1 - target]).detach().cpu())
            d = by_rel[e.relation]
            d["n"] += 1; d["correct"] += ok; d["margins"].append(margin)
    out = {}
    for rel in sorted(by_rel):
        d = by_rel[rel]
        out[rel] = {
            "n": d["n"],
            "acc": d["correct"] / d["n"] if d["n"] else math.nan,
            "mean_signed_margin": float(np.mean(d["margins"])) if d["margins"] else math.nan,
        }
    direct = [out[r]["acc"] for r in ["h0_dax", "h2_norp"] if r in out]
    graph = [out[r]["acc"] for r in ["h1_mep", "h3_ziv"] if r in out]
    out["_summary"] = {
        "direct_mean": float(np.mean(direct)) if direct else math.nan,
        "graph_mean": float(np.mean(graph)) if graph else math.nan,
    }
    return out


def train_cmp_acc(model: SharedEventModel, cache: EncCache, comps: List[CompItem]) -> float:
    n = ok = 0
    model.eval()
    with torch.no_grad():
        for c in comps:
            s1 = score_event(model, cache, c.event1, "cmp")
            s2 = score_event(model, cache, c.event2, "cmp")
            p1 = F.softmax(s1, dim=0); p2 = F.softmax(s2, dim=0)
            psame = (p1 * p2).sum()
            ok += int(bool(psame >= 0.5) == bool(c.label))
            n += 1
    return ok / n if n else math.nan


def run_cell(condition: str, bridge_sign: int, seed: int, epochs: int, eval_every: int,
             lr: float, wd: float, emb_dim: int, hidden: int, device: torch.device) -> dict:
    train_states_base, train_comps_base, eval_states_base = load_items()
    vocab = build_vocab(train_states_base, train_comps_base, eval_states_base)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = SharedEventModel(len(vocab.itos), emb_dim=emb_dim, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    history = []
    t0 = time.time()

    # Fixed no-cue and matched-cue eval objects; train objects are regenerated per epoch.
    eval_no = eval_variant(eval_states_base, "no_cue")
    eval_matched = eval_variant(eval_states_base, "matched_cue")
    eval_wrong = eval_variant(eval_states_base, "wrong_cue")

    for ep in range(1, epochs + 1):
        model.train()
        cache = EncCache(vocab, device)
        states, comps = transformed_epoch(train_states_base, train_comps_base, condition, seed, ep)
        rng = random.Random(seed + ep)
        rng.shuffle(states); rng.shuffle(comps)
        opt.zero_grad(set_to_none=True)
        losses = []
        for e in states:
            sc = score_event(model, cache, e, "state")
            target = int(e.target_idx)
            if bridge_sign == -1 and e.is_direct_anchor:
                sc = sc.flip(0)
            losses.append(F.cross_entropy(sc.view(1, -1), torch.tensor([target], dtype=torch.long, device=device)))
        for c in comps:
            s1 = score_event(model, cache, c.event1, "cmp")
            s2 = score_event(model, cache, c.event2, "cmp")
            p1 = F.softmax(s1, dim=0); p2 = F.softmax(s2, dim=0)
            psame = (p1 * p2).sum().clamp(1e-6, 1 - 1e-6)
            y = torch.tensor(float(c.label), dtype=torch.float32, device=device)
            losses.append(-(y * torch.log(psame) + (1 - y) * torch.log(1 - psame)))
        total = torch.stack(losses).mean()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if ep == 1 or ep == epochs or ep % eval_every == 0:
            # Evaluate held-out transport along budget curve.  Use a fresh cache
            # for eval variants so we do not mix transformed train strings.
            no_cache = EncCache(vocab, device)
            matched_cache = EncCache(vocab, device)
            wrong_cache = EncCache(vocab, device)
            train_cache = EncCache(vocab, device)
            train_cmp = train_cmp_acc(model, train_cache, comps)
            no = eval_states(model, no_cache, eval_no, bridge_sign)
            matched = eval_states(model, matched_cache, eval_matched, bridge_sign)
            wrong = eval_states(model, wrong_cache, eval_wrong, bridge_sign)
            rec = {
                "epoch": ep,
                "loss": float(total.detach().cpu()),
                "train_cmp_current": train_cmp,
                "no_cue_direct": no["_summary"]["direct_mean"],
                "no_cue_graph": no["_summary"]["graph_mean"],
                "matched_cue_direct": matched["_summary"]["direct_mean"],
                "matched_cue_graph": matched["_summary"]["graph_mean"],
                "wrong_cue_direct": wrong["_summary"]["direct_mean"],
                "wrong_cue_graph": wrong["_summary"]["graph_mean"],
                "no_cue_per_relation": no,
                "matched_cue_per_relation": matched,
                "wrong_cue_per_relation": wrong,
            }
            history.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k not in ["no_cue_per_relation", "matched_cue_per_relation"]}, sort_keys=True), flush=True)

    return {
        "condition": condition,
        "bridge_sign": bridge_sign,
        "seed": seed,
        "epochs": epochs,
        "elapsed_seconds": time.time() - t0,
        "vocab_size": len(vocab.itos),
        "history": history,
        "final_no_cue": history[-1]["no_cue_per_relation"],
        "final_matched_cue": history[-1]["matched_cue_per_relation"],
        "final_wrong_cue": history[-1]["wrong_cue_per_relation"],
    }


def run_all(args) -> dict:
    device = torch.device("cuda:%d" % args.gpu if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    conditions = args.conditions or CONDITIONS
    signs = args.bridge_signs or [1]
    for condition in conditions:
        for sign in signs:
            sign_s = "+1" if sign == 1 else "-1"
            tag = f"{condition}_bs{sign_s}_e{args.epochs}_seed{args.seed}"
            print(f"\n=== {tag} ===", flush=True)
            rec = run_cell(condition, sign, args.seed, args.epochs, args.eval_every,
                           args.lr, args.wd, args.emb_dim, args.hidden, device)
            results[(condition, sign)] = rec
            rd = out / tag
            rd.mkdir(parents=True, exist_ok=True)
            write_json(rd / "result.json", rec)

    # Combined analysis tables.
    combined = {
        "experiment": "repaired_cue_contrast",
        "seed": args.seed,
        "epochs": args.epochs,
        "eval_every": args.eval_every,
        "conditions": conditions,
        "bridge_signs": signs,
        "results": {f"{c}_bs{'+1' if s == 1 else '-1'}": r for (c, s), r in results.items()},
    }
    write_json(out / "combined_results.json", combined)

    lines = [
        "# research repaired cue contrast",
        "",
        "Fast effective-input pilot: names are canonicalized as `<C>/<O>` before the GRU, so name invariance is supplied. Train-only cue tokens test whether variation helps only when a visible nuisance can compete with the relation solution. In `comp_*_masked` conditions, direct state anchors remain clean while comparison-event relation verbs are masked with probability 0.75. In `packet_*` conditions, a row-level packet token is shared by both events in a training comparison row or broken independently across the two events; packet tokens are absent at eval.",
        "",
        f"Seed={args.seed}, epochs={args.epochs}, device={device}",
        "",
        "## Final held-out performance",
        "",
        "| condition | sign | train_cmp_current | no-cue direct | no-cue graph | matched-cue direct | matched-cue graph | wrong-cue direct | wrong-cue graph |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        for sign in signs:
            r = results[(condition, sign)]
            h = r["history"][-1]
            lines.append(f"| {condition} | {'+1' if sign == 1 else '-1'} | {h['train_cmp_current']:.3f} | {h['no_cue_direct']:.3f} | {h['no_cue_graph']:.3f} | {h['matched_cue_direct']:.3f} | {h['matched_cue_graph']:.3f} | {h['wrong_cue_direct']:.3f} | {h['wrong_cue_graph']:.3f} |")
    lines.extend(["", "## Budget curve: no-cue graph transport", "", "| epoch | " + " | ".join(f"{c} bs{'+1' if s == 1 else '-1'}" for c in conditions for s in signs) + " |", "|---:" + "|---:" * (len(conditions) * len(signs)) + "|"])
    epochs = sorted({h["epoch"] for r in results.values() for h in r["history"]})
    for ep in epochs:
        vals = []
        for c in conditions:
            for s in signs:
                hist = {h["epoch"]: h for h in results[(c, s)]["history"]}
                vals.append(f"{hist[ep]['no_cue_graph']:.3f}" if ep in hist else "")
        lines.append(f"| {ep} | " + " | ".join(vals) + " |")
    lines.extend(["", "## Per-relation final no-cue accuracies", "", "| condition | sign | h0 | h1 | h2 | h3 |", "|---|---|---:|---:|---:|---:|"])
    for c in conditions:
        for s in signs:
            r = results[(c, s)]["final_no_cue"]
            lines.append(f"| {c} | {'+1' if s == 1 else '-1'} | {r['h0_dax']['acc']:.3f} | {r['h1_mep']['acc']:.3f} | {r['h2_norp']['acc']:.3f} | {r['h3_ziv']['acc']:.3f} |")
    lines.append("")
    (out / "combined_results.md").write_text("\n".join(lines))

    print(json.dumps({
        "status": "REPAIRED_CUE_CONTRAST_DONE",
        "out_md": str(out / "combined_results.md"),
        "out_json": str(out / "combined_results.json"),
    }, indent=2), flush=True)
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=None)
    ap.add_argument("--bridge-signs", nargs="+", type=int, choices=[1, -1], default=None)
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--emb-dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
