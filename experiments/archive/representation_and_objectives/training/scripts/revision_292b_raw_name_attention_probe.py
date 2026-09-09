#!/usr/bin/env python3
"""Step292b: raw-name binding with query-attention — fix binding bottleneck.

Step292a showed the mean-pooled GRU cannot learn comparison binding with
character-encoded names (shared_trunk train_cmp stuck at 0.5). This adds a
query-to-event cross-attention module that explicitly locates the queried
name in the event text, providing the binding primitive needed for both
state and comparison tasks.

Changes from raw_name_binding_probe.py:
1. QueryAttentionScorer: finds <QRY> in token IDs, separates event/query
   positions, computes cross-attention from query to event, feeds attended
   representation to the scalar head.
2. Fresh model per bridge_sign: fixes sequential training contamination.
3. Simpler run: single seed, primary conditions only, longer timeout budget.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_DATA_ROOT = PROJECT / "data/information_budget_substrate/replace_k16_spread"
DEFAULT_OUT = PROJECT / "data/step292b_raw_name_attention"
DEFAULT_ARM = "aligned_state_bridge"
DIRECT_RELS = {"h0_dax", "h2_norp"}
GRAPH_RELS = {"h1_mep", "h3_ziv"}

TOKEN_RE = re.compile(r"[A-Za-z_]+|[0-9]+|[.,;:?]")
CAP_RE = re.compile(r"\b[A-Z][a-z]+\b")
NOT_NAMES = {
    "During", "Event", "Did", "After", "At", "The", "A", "B", "From", "This",
    "First", "Then", "Before", "Question", "Answer",
}


def project_rel(path: Path) -> str:
    try: return str(path.relative_to(Path.cwd()))
    except Exception: return str(path)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists(): return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s: rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


# ── Name extraction ──
def extract_names(text: str) -> List[str]:
    names = []
    for m in CAP_RE.findall(text):
        if m not in NOT_NAMES and m not in names: names.append(m)
    return names

def event_names(event_text: str) -> List[str]:
    return extract_names(event_text)[:2]

def parse_comparison_text(row):
    e1, e2 = str(row.get("event1","")), str(row.get("event2",""))
    if e1 and e2: return e1, e2
    text = str(row.get("text",""))
    m = re.search(r"Event A:\s*(.*?)\s*Event B:\s*(.*?)\s*Did Event A", text)
    if not m: raise ValueError(f"could not parse comparison text: {text[:120]}")
    return m.group(1).strip(), m.group(2).strip()

def parse_state_event(row):
    ce = str(row.get("cause_event",""))
    if ce: return ce
    prem = str(row.get("premise",""))
    marker = "The event was this:"
    if marker in prem: return prem.split(marker,1)[1].strip()
    raise ValueError(f"could not parse state event: {prem[:120]}")

def premise_before_event(row):
    prem = str(row.get("premise",""))
    marker = "The event was this:"
    return prem.split(marker,1)[0].strip() if marker in prem else prem

def parse_hypothesis(row):
    hyp = str(row.get("hypothesis",""))
    m = re.search(r"After the event,\s+([A-Z][a-z]+)\s+had the\s+([A-Za-z_]+)\.?", hyp)
    if not m: raise ValueError(f"could not parse hypothesis: {hyp}")
    return m.group(1), m.group(2).lower()

def event_object(event_text):
    m = re.search(r"During the\s+([A-Za-z_]+)\s+episode", event_text)
    return m.group(1).lower() if m else None

def rel_family(rel):
    if rel in DIRECT_RELS: return "direct_anchor"
    if rel in GRAPH_RELS: return "graph_transfer"
    return "seen_or_other" if rel else "unknown"


# ── Character encoding with query ──
def name_to_chars(name: str) -> List[str]:
    return ["<N>"] + list(name.lower()) + ["</N>"]

def tokenize_with_char_names(text: str, names: Sequence[str]) -> List[str]:
    toks = TOKEN_RE.findall(text)
    out = []
    for t in toks:
        if t in names: out.extend(name_to_chars(t))
        else: out.append(t.lower())
    return out

def encode_event_with_query(event_text, candidate, names):
    event_toks = tokenize_with_char_names(event_text, names)
    query_toks = name_to_chars(candidate)
    return ["<bos>"] + event_toks + ["<QRY>"] + query_toks + ["<eos>"]

def encode_static_with_query(prefix_text, hyp_text, candidate, names):
    prefix_toks = tokenize_with_char_names(prefix_text, names)
    hyp_toks = tokenize_with_char_names(hyp_text, names)
    query_toks = name_to_chars(candidate)
    return ["<bos>"] + prefix_toks + ["<hyp>"] + hyp_toks + ["<QRY>"] + query_toks + ["<eos>"]


# ── Data structures ──
@dataclass
class StateQuery:
    key: str; suite: str; split: str; arm: str
    event: str; prefix: str; hypothesis_text_by_name: Dict[str,str]
    names: Tuple[str,str]; object_name: str; is_changed: bool
    label_by_name: Dict[str,bool]; inverted_target_by_name: Dict[str,bool]
    relation: Optional[str]; relation_family: str
    initial_pattern: Optional[str]; static_slot: Optional[int]
    query_kind: str; metadata: Dict[str,Any]
    @property
    def is_direct_anchor(self): return self.is_changed and self.relation in DIRECT_RELS

@dataclass
class ComparisonExample:
    row_id: str; suite: str; split: str; arm: str
    event1: str; event2: str; names: Tuple[str,str]
    label: bool; inverted_target: bool
    relation1: Optional[str]; relation2: Optional[str]; metadata: Dict[str,Any]

def state_query_key(row):
    if row.get("pair_id") is not None:
        return f"{row.get('suite')}|{row.get('pair_id')}|{row.get('query_kind')}"
    return re.sub(r"_(0|1)$", "", str(row.get("row_id")))

def build_state_queries(rows, arm, parse_errors):
    by = defaultdict(list)
    for r in rows:
        if r.get("task") == "state_query": by[state_query_key(r)].append(r)
    out = []
    for k, group in by.items():
        if len(group) < 2:
            parse_errors.append(f"state group {k} has {len(group)} rows"); continue
        try:
            ev = parse_state_event(group[0]); ns = event_names(ev)
            if len(ns) != 2: parse_errors.append(f"state group {k} names={ns}"); continue
            prefix = premise_before_event(group[0])
            hyp_by_name, label_by_name, inv_by_name = {}, {}, {}; obj = None
            for r in group:
                cand, obj_i = parse_hypothesis(r)
                hyp_by_name[cand] = str(r.get("hypothesis",""))
                label = bool(r.get("label")); flip = bool(r.get("global_swap_changes_label",False))
                label_by_name[cand] = label; inv_by_name[cand] = bool(label) ^ bool(flip); obj = obj_i
            if not all(n in label_by_name for n in ns): continue
            ev_obj = event_object(ev); is_changed = (obj == ev_obj); r0 = group[0]
            out.append(StateQuery(key=k, suite=str(r0.get("suite","")), split=str(r0.get("split","")), arm=arm,
                event=ev, prefix=prefix, hypothesis_text_by_name=hyp_by_name, names=(ns[0],ns[1]),
                object_name=str(obj), is_changed=bool(is_changed), label_by_name=label_by_name,
                inverted_target_by_name=inv_by_name, relation=r0.get("relation") or r0.get("cause_relation"),
                relation_family=rel_family(r0.get("relation") or r0.get("cause_relation")),
                initial_pattern=r0.get("initial_pattern"), static_slot=r0.get("static_slot"),
                query_kind="changed" if is_changed else "unchanged",
                metadata={"pair_id": r0.get("pair_id"), "row_ids": [rr.get("row_id") for rr in group],
                          "global_swap_changes_label": bool(r0.get("global_swap_changes_label",False)),
                          "raw_query_kind": r0.get("query_kind")}))
        except Exception as e:
            parse_errors.append(f"state group {k}: {type(e).__name__}: {e}")
    return out

def build_comparisons(rows, arm, parse_errors):
    out = []
    for r in rows:
        if r.get("task") != "relation_comparison": continue
        try:
            e1, e2 = parse_comparison_text(r)
            ns = []
            for n in event_names(e1) + event_names(e2):
                if n not in ns: ns.append(n)
            if len(ns) != 2: continue
            label = bool(r.get("label")); flip = bool(r.get("global_swap_changes_label",False))
            out.append(ComparisonExample(row_id=str(r.get("row_id")), suite=str(r.get("suite","")),
                split=str(r.get("split","")), arm=arm, event1=e1, event2=e2, names=(ns[0],ns[1]),
                label=label, inverted_target=bool(label) ^ bool(flip),
                relation1=r.get("relation1"), relation2=r.get("relation2"),
                metadata={"global_swap_changes_label": flip, "orientation_dependency": r.get("orientation_dependency")}))
        except Exception as e:
            parse_errors.append(f"comparison row {r.get('row_id')}: {e}")
    return out


# ── Vocabulary ──
class Vocab:
    def __init__(self):
        self.itos = ["<pad>", "<unk>", "<bos>", "<eos>", "<N>", "</N>", "<QRY>", "<hyp>"]
        for c in "abcdefghijklmnopqrstuvwxyz": self.itos.append(c)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
    def add(self, toks):
        for t in toks:
            if t not in self.stoi:
                self.stoi[t] = len(self.itos); self.itos.append(t)
    def encode(self, toks): return [self.stoi.get(t, 1) for t in toks]

def collect_vocab(vocab, states, comps):
    for q in states:
        for c in q.names:
            if q.is_changed: vocab.add(encode_event_with_query(q.event, c, q.names))
            else: vocab.add(encode_static_with_query(q.prefix, q.hypothesis_text_by_name[c], c, q.names))
    for c in comps:
        for cn in c.names:
            vocab.add(encode_event_with_query(c.event1, cn, c.names))
            vocab.add(encode_event_with_query(c.event2, cn, c.names))


# ── Model with query-attention ──

class TrunkEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.out_dim = 2 * hidden

    def forward_all(self, ids, mask):
        """Return per-position hidden states."""
        x = self.emb(ids)
        y, _ = self.gru(x)
        return y  # [batch, seq_len, 2*hidden]


class ScalarHead(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, h): return self.net(h).squeeze(-1)


class QueryAttentionScorer(nn.Module):
    """Score event with query-conditioned cross-attention.

    1. Run trunk over full sequence [event <QRY> query_name]
    2. Separate event positions (before <QRY>) and query positions (after <QRY>)
    3. Mean-pool query positions → query vector
    4. Cross-attention from query to event positions
    5. Feed attended event repr to scalar head
    """
    def __init__(self, trunk, head, qry_token_id, attn_dim=32):
        super().__init__()
        self.trunk = trunk
        self.head = head
        self.qry_id = qry_token_id
        self.W_q = nn.Linear(trunk.out_dim, attn_dim)
        self.W_k = nn.Linear(trunk.out_dim, attn_dim)

    def forward_ids(self, ids, mask):
        y = self.trunk.forward_all(ids, mask)  # [batch, seq, hidden]
        batch, seq_len, hdim = y.shape
        device = ids.device

        # Find <QRY> position for each sequence
        is_qry = (ids == self.qry_id)  # [batch, seq]
        qry_pos = is_qry.long().argmax(dim=1)  # [batch]

        pos = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq]
        event_mask = mask & (pos < qry_pos.unsqueeze(1))   # before <QRY>
        query_mask = mask & (pos > qry_pos.unsqueeze(1))   # after <QRY>

        # Query representation: mean-pool query positions
        q_float = query_mask.unsqueeze(-1).float()
        q_rep = (y * q_float).sum(1) / q_float.sum(1).clamp_min(1.0)  # [batch, hidden]

        # Cross-attention: query → event
        Q = self.W_q(q_rep).unsqueeze(1)  # [batch, 1, attn_dim]
        K = self.W_k(y)                    # [batch, seq, attn_dim]
        attn_logits = (Q * K).sum(-1) / math.sqrt(Q.shape[-1])  # [batch, seq]
        attn_logits = attn_logits.masked_fill(~event_mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=-1)  # [batch, seq]

        # Attended event representation
        context = (y * attn_weights.unsqueeze(-1)).sum(1)  # [batch, hidden]

        return self.head(context)


def create_single_model(vocab_size, mode, qry_token_id, seed, emb_dim=48, hidden=64, attn_dim=32):
    """Create a single model with fresh initialization from the given seed."""
    torch.manual_seed(seed)
    ref_trunk = TrunkEncoder(vocab_size, emb_dim, hidden)
    ref_head = ScalarHead(ref_trunk.out_dim, hidden)
    ref_static_trunk = TrunkEncoder(vocab_size, emb_dim, hidden)
    ref_static_head = ScalarHead(ref_static_trunk.out_dim, hidden)
    ref_attn_q = nn.Linear(ref_trunk.out_dim, attn_dim)
    ref_attn_k = nn.Linear(ref_trunk.out_dim, attn_dim)

    class PM(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.mode = m
            if m == "tied":
                trunk_c = copy.deepcopy(ref_trunk); head_c = copy.deepcopy(ref_head)
                scorer = QueryAttentionScorer(trunk_c, head_c, qry_token_id, attn_dim)
                scorer.W_q = copy.deepcopy(ref_attn_q); scorer.W_k = copy.deepcopy(ref_attn_k)
                self.event_state = scorer; self.event_cmp = self.event_state
            elif m == "shared_trunk":
                trunk_c = copy.deepcopy(ref_trunk)
                s_state = QueryAttentionScorer(trunk_c, copy.deepcopy(ref_head), qry_token_id, attn_dim)
                s_state.W_q = copy.deepcopy(ref_attn_q); s_state.W_k = copy.deepcopy(ref_attn_k)
                s_cmp = QueryAttentionScorer(trunk_c, copy.deepcopy(ref_head), qry_token_id, attn_dim)
                s_cmp.W_q = copy.deepcopy(ref_attn_q); s_cmp.W_k = copy.deepcopy(ref_attn_k)
                self.event_state = s_state; self.event_cmp = s_cmp
            elif m == "untied":
                s_state = QueryAttentionScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head), qry_token_id, attn_dim)
                s_state.W_q = copy.deepcopy(ref_attn_q); s_state.W_k = copy.deepcopy(ref_attn_k)
                s_cmp = QueryAttentionScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head), qry_token_id, attn_dim)
                s_cmp.W_q = copy.deepcopy(ref_attn_q); s_cmp.W_k = copy.deepcopy(ref_attn_k)
                self.event_state = s_state; self.event_cmp = s_cmp
            else: raise ValueError(m)
            # Static scorer also gets attention
            s_static = QueryAttentionScorer(copy.deepcopy(ref_static_trunk), copy.deepcopy(ref_static_head),
                                            qry_token_id, attn_dim)
            s_static.W_q = copy.deepcopy(ref_attn_q); s_static.W_k = copy.deepcopy(ref_attn_k)
            self.static = s_static

    return PM(mode)


# ── Scoring ──
def pad_batch(seqs, device):
    max_len = max(len(s) for s in seqs) if seqs else 1
    arr = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        arr[i,:len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        mask[i,:len(s)] = True
    return arr, mask

def score_sequences(scorer, seqs, device):
    ids, mask = pad_batch(seqs, device)
    return scorer.forward_ids(ids, mask)

def state_scores(model, vocab, q, device):
    if q.is_changed:
        seqs = [vocab.encode(encode_event_with_query(q.event, c, q.names)) for c in q.names]
        return score_sequences(model.event_state, seqs, device)
    seqs = [vocab.encode(encode_static_with_query(q.prefix, q.hypothesis_text_by_name[c], c, q.names)) for c in q.names]
    return score_sequences(model.static, seqs, device)

def comparison_prob_same(model, vocab, c, device):
    seqs1 = [vocab.encode(encode_event_with_query(c.event1, n, c.names)) for n in c.names]
    seqs2 = [vocab.encode(encode_event_with_query(c.event2, n, c.names)) for n in c.names]
    s1 = score_sequences(model.event_cmp, seqs1, device)
    s2 = score_sequences(model.event_cmp, seqs2, device)
    p1, p2 = F.softmax(s1, dim=0), F.softmax(s2, dim=0)
    psame = (p1 * p2).sum().clamp(1e-6, 1 - 1e-6)
    return psame, s1, s2

def bce_prob(p, target):
    y = torch.tensor(float(target), dtype=torch.float32, device=p.device)
    return -(y * torch.log(p) + (1-y) * torch.log(1-p))


# ── Training ──
def get_target_idx(q): return 1 if (q.label_by_name[q.names[1]] and not q.label_by_name[q.names[0]]) else 0

def train_one(model, vocab, train_states, train_comps, device, bridge_sign, epochs, lr,
              weight_decay, seed, cmp_weight=1.0, state_weight=1.0, static_weight=1.0, print_every=0):
    model.to(device)
    seen_ids = set(); all_params = []; clip_groups = []
    def add_mod(mod, name):
        gp = []
        for p in mod.parameters():
            pid = id(p)
            if pid not in seen_ids:
                seen_ids.add(pid); all_params.append(p); gp.append(p)
        if gp: clip_groups.append(gp)
    add_mod(model.event_state, "event_state")
    if model.event_cmp is not model.event_state:
        add_mod(model.event_cmp, "event_cmp")
    add_mod(model.static, "static")

    opt = torch.optim.AdamW([{"params": all_params}], lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed); history = []; t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        states = list(train_states); comps = list(train_comps)
        rng.shuffle(states); rng.shuffle(comps)
        opt.zero_grad(set_to_none=True); losses = []

        for q in states:
            scores = state_scores(model, vocab, q, device)
            target_idx = get_target_idx(q)
            if bridge_sign == -1 and q.is_direct_anchor: scores = scores.flip(0)
            loss = F.cross_entropy(scores.view(1,-1), torch.tensor([target_idx], dtype=torch.long, device=device))
            losses.append((state_weight if q.is_changed else static_weight) * loss)

        for c in comps:
            p, _, _ = comparison_prob_same(model, vocab, c, device)
            losses.append(cmp_weight * bce_prob(p, c.label))

        if not losses: raise RuntimeError("no losses")
        loss_total = torch.stack(losses).mean(); loss_total.backward()
        for cg in clip_groups: torch.nn.utils.clip_grad_norm_(cg, 5.0)
        opt.step()

        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            with torch.no_grad():
                model.eval()
                n_s=c_s=n_ch=c_ch=n_un=c_un=0
                for q in train_states:
                    sc = state_scores(model, vocab, q, device)
                    if bridge_sign == -1 and q.is_direct_anchor: sc_eff = sc.flip(0)
                    else: sc_eff = sc
                    tgt = get_target_idx(q); pred = int(torch.argmax(sc_eff).item())
                    n_s += 1; c_s += int(pred == tgt)
                    if q.is_changed: n_ch += 1; c_ch += int(pred == tgt)
                    else: n_un += 1; c_un += int(pred == tgt)
                n_c = c_c = 0
                for ex in train_comps:
                    p2, _, _ = comparison_prob_same(model, vocab, ex, device)
                    n_c += 1; c_c += int(float(p2.detach().cpu()) >= 0.5) == int(ex.label)
                metrics = {"train_state_acc": c_s/n_s if n_s else float("nan"),
                           "train_changed_acc": c_ch/n_ch if n_ch else float("nan"),
                           "train_unchanged_acc": c_un/n_un if n_un else float("nan"),
                           "train_cmp_acc": c_c/n_c if n_c else float("nan")}
            rec = {"epoch": ep, "loss": float(loss_total.detach().cpu()), **{k:float(v) for k,v in metrics.items()}}
            history.append(rec)
            if print_every: print(json.dumps(rec, sort_keys=True), flush=True)
    return {"elapsed_seconds": time.time() - t0, "history": history}


# ── Evaluation ──
def eval_predictions(model, vocab, states, comps, device, condition, arm, seed, bridge_sign):
    model.eval(); state_rows = []; comp_rows = []
    with torch.no_grad():
        for q in states:
            scores_t = state_scores(model, vocab, q, device)
            scores = [float(x) for x in scores_t.detach().cpu().tolist()]
            probs = [float(x) for x in F.softmax(scores_t, dim=0).detach().cpu().tolist()]
            d_e = scores[0] - scores[1]
            for i, cand in enumerate(q.names):
                label_true = bool(q.label_by_name[cand])
                state_rows.append({"condition":condition, "arm":arm, "seed":seed, "bridge_sign":bridge_sign,
                    "suite":q.suite, "split":q.split, "query_key":q.key, "task":"state_query",
                    "candidate":cand, "candidate_index":i, "score":scores[i], "prob":probs[i], "d_e":d_e,
                    "label_true":label_true, "is_changed":q.is_changed, "query_kind":q.query_kind,
                    "relation":q.relation, "relation_family":q.relation_family,
                    "initial_pattern":q.initial_pattern, "static_slot":q.static_slot,
                    "is_direct_anchor":q.is_direct_anchor, "names":list(q.names), "object":q.object_name,
                    "global_swap_changes_label":q.metadata.get("global_swap_changes_label")})
        for c in comps:
            p, s1, s2 = comparison_prob_same(model, vocab, c, device)
            p_f = float(p.detach().cpu())
            logit = math.log(p_f/(1.0-p_f)) if 0 < p_f < 1 else 0.0
            comp_rows.append({"condition":condition, "arm":arm, "seed":seed, "bridge_sign":bridge_sign,
                "suite":c.suite, "split":c.split, "row_id":c.row_id, "task":"relation_comparison",
                "prob_same":p_f, "logit_same":logit, "label":bool(c.label), "pred":bool(p_f>=0.5),
                "correct":bool(p_f>=0.5) == bool(c.label),
                "signed_margin": logit if bool(c.label) else -logit,
                "d_e1":float(s1[0])-float(s1[1]), "d_e2":float(s2[0])-float(s2[1]),
                "relation1":c.relation1, "relation2":c.relation2, "names":list(c.names)})
    return state_rows, comp_rows


def central_readout(state_rows, comp_rows, bridge_sign, arm):
    mean = lambda xs: sum(xs)/len(xs) if xs else float("nan")
    def canonical_target(row):
        if bridge_sign == 1: return row["label_true"]
        if row.get("is_direct_anchor"): return not row["label_true"]
        return row["label_true"]

    changed_psc = [r for r in state_rows if r["is_changed"] and r.get("suite","").startswith("paired_state_conservation") and r["candidate_index"]==0]
    unchanged_psc = [r for r in state_rows if not r["is_changed"] and r.get("suite","").startswith("paired_state_conservation") and r["candidate_index"]==0]
    direct_same = [r for r in changed_psc if r["relation_family"]=="direct_anchor" and r.get("initial_pattern")=="same"]
    graph_same = [r for r in changed_psc if r["relation_family"]=="graph_transfer" and r.get("initial_pattern")=="same"]

    def choice_acc(rows): return mean([float(r["d_e"]>0)==canonical_target(r) for r in rows]) if rows else float("nan")
    def state_margin(rows): return mean([abs(r["d_e"])*(1 if (float(r["d_e"]>0)==canonical_target(r)) else -1) for r in rows]) if rows else float("nan")

    psc0 = [r for r in state_rows if r.get("suite","").startswith("paired_state_conservation") and r["candidate_index"]==0]
    psc_pairs = defaultdict(list)
    for r in psc0: psc_pairs[r.get("query_key","").rsplit("|",1)[0]].append(r)
    gsm_pairs = {k: rs for k,rs in psc_pairs.items() if any(r["relation_family"]=="graph_transfer" and r.get("initial_pattern")=="same" for r in rs)}
    def pair_both(pairs):
        both = []
        for k, rs in pairs.items():
            ch = [r for r in rs if r["is_changed"]]; un = [r for r in rs if not r["is_changed"]]
            if ch and un:
                ch_ok = all((float(r["d_e"]>0)==canonical_target(r)) for r in ch)
                un_ok = all((float(r["d_e"]>0)==canonical_target(r)) for r in un)
                both.append(float(ch_ok and un_ok))
        return mean(both) if both else float("nan")

    hh = [r for r in comp_rows if r.get("suite","").startswith("heldheld")]
    mixed = [r for r in comp_rows if r.get("suite","").startswith("mixed_held_seen")]
    return {
        "direct_same": choice_acc(direct_same), "graph_same": choice_acc(graph_same),
        "graph_same_margin": state_margin(graph_same), "unchanged": choice_acc(unchanged_psc),
        "pair_both_graph_same": pair_both(gsm_pairs),
        "hh_closure": mean([float(r["correct"]) for r in hh]) if hh else float("nan"),
        "hh_margin": mean([r.get("signed_margin",0) for r in hh]) if hh else float("nan"),
        "mixed_acc": mean([float(r["correct"]) for r in mixed]) if mixed else float("nan"),
        "mixed_margin": mean([r.get("signed_margin",0) for r in mixed]) if mixed else float("nan"),
    }


# ── Main ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[29200])
    ap.add_argument("--conditions", nargs="+", default=["tied", "shared_trunk", "untied"])
    ap.add_argument("--bridge-signs", type=int, nargs="+", default=[1, -1])
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--emb-dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--attn-dim", type=int, default=32)
    ap.add_argument("--print-every", type=int, default=55)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_root = args.data_root; arm_dir = data_root / "arms" / args.arm; eval_dir = data_root / "eval"
    parse_errors = []
    arm_rows = load_jsonl(arm_dir / "train_supervised.jsonl")
    common_rows = load_jsonl(data_root / "common_seen_train.jsonl")
    train_states_arm = build_state_queries(arm_rows, args.arm, parse_errors)
    train_comps_arm = build_comparisons(arm_rows, args.arm, parse_errors)
    train_states_common = build_state_queries(common_rows, "common", parse_errors)
    train_states = train_states_arm + train_states_common
    train_comps = train_comps_arm

    eval_suites = {}
    for fp in sorted(eval_dir.iterdir()):
        if fp.suffix != ".jsonl": continue
        rows = load_jsonl(fp)
        eval_suites[fp.stem] = (build_state_queries(rows, args.arm, parse_errors),
                                build_comparisons(rows, args.arm, parse_errors))

    vocab = Vocab()
    collect_vocab(vocab, train_states, train_comps)
    qry_id = vocab.stoi["<QRY>"]
    print(f"Vocabulary: {len(vocab.itos)} tokens, <QRY> id={qry_id}", flush=True)

    # Sample encoding check
    sq = train_states_arm[0] if train_states_arm else train_states_common[0]
    for cn in sq.names:
        toks = encode_event_with_query(sq.event, cn, sq.names)
        ids = vocab.encode(toks)
        print(f"  {cn}: {' '.join(toks[:30])} ... ({len(toks)} tok, {sum(1 for x in ids if x==1)} unk)", flush=True)

    device = torch.device(args.device)
    all_results = []

    for seed in args.seeds:
        for cond in args.conditions:
            for bs in args.bridge_signs:
                # Fresh model for EACH (cond, bridge_sign) — no sequential contamination
                model = create_single_model(len(vocab.itos), cond, qry_id, seed,
                                            args.emb_dim, args.hidden, args.attn_dim)
                cell_tag = f"bs{'+' if bs>0 else ''}{bs}"
                run_tag = f"{cond}_{cell_tag}_seed{seed}"
                print(f"\n=== {cond}/bs={bs:+d}/seed={seed} ===", flush=True)

                info = train_one(model, vocab, train_states, train_comps, device, bs,
                                 args.epochs, args.lr, args.weight_decay, seed,
                                 print_every=args.print_every)

                all_state_rows, all_comp_rows = [], []
                for suite_name, (sq_s, cmp_s) in eval_suites.items():
                    sr, cr = eval_predictions(model, vocab, sq_s, cmp_s, device, cond, args.arm, seed, bs)
                    all_state_rows.extend(sr); all_comp_rows.extend(cr)

                central = central_readout(all_state_rows, all_comp_rows, bs, args.arm)
                central["train_state_acc"] = info["history"][-1]["train_state_acc"]
                central["train_cmp_acc"] = info["history"][-1]["train_cmp_acc"]

                result = {"arm": args.arm, "bridge_sign": bs, "cell_tag": cell_tag,
                    "central_eval": central, "condition": cond, "seed": seed,
                    "elapsed_seconds": info["elapsed_seconds"], "epochs": args.epochs,
                    "encoding": "raw_name_char_query_attention", "architecture": "query_attention",
                    "history": {"sampled": info["history"]}, "vocab_size": len(vocab.itos)}
                all_results.append(result)
                print(json.dumps({k:v for k,v in result.items() if k!="history"}, sort_keys=True), flush=True)

                run_dir = args.out / run_tag
                write_json(run_dir / "result.json", result)
                write_jsonl(run_dir / "state_predictions.jsonl", all_state_rows)
                write_jsonl(run_dir / "comp_predictions.jsonl", all_comp_rows)

    # Summary
    lines = ["# Step292b raw-name binding with query-attention\n\n",
             "## Per-run readout\n\n",
             "| condition | bs | train_state | train_cmp | direct_same | graph_same | pair_both | unchanged | hh_closure | mixed_acc | mixed_margin |\n",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"]
    for r in all_results:
        c = r["central_eval"]
        lines.append(f"| {r['condition']} | {r['bridge_sign']:+d} | "
                     f"{c.get('train_state_acc',float('nan')):.3f} | {c.get('train_cmp_acc',float('nan')):.3f} | "
                     f"{c.get('direct_same',float('nan')):.3f} | {c.get('graph_same',float('nan')):.3f} | "
                     f"{c.get('pair_both_graph_same',float('nan')):.3f} | {c.get('unchanged',float('nan')):.3f} | "
                     f"{c.get('hh_closure',float('nan')):.3f} | {c.get('mixed_acc',float('nan')):.3f} | "
                     f"{c.get('mixed_margin',float('nan')):.1f} |\n")

    md_path = args.out / "raw_name_attention_summary.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("".join(lines), encoding="utf-8")
    write_json(args.out / "raw_name_attention_summary.json", all_results)

    print(json.dumps({"device": str(device), "n_results": len(all_results),
        "no_official_evaluation_upload_or_leaderboard": True,
        "status": "STEP292B_RAW_NAME_ATTENTION_COMPLETE",
        "summary": project_rel(md_path)}, indent=2, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
