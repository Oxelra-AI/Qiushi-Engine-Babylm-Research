#!/usr/bin/env python3
"""research: causal gauge-transport probe.

Scientific purpose
------------------
research showed that a tied candidate-event scorer transports h0/h2 state-anchor
orientation to h1/h3 graph-transfer relations, while an untied model does not.
The result holds in the controlled experiment. This probe tests
the **causal** mechanism: does the shared coordinate transport an absolute gauge,
or is generic parameter sharing sufficient?

Design
------
1. **bridge_sign**: apply a +1/-1 score permutation *only* at h0/h2 changed-state
   anchor training loss.  Comparison rows, unchanged/static rows, h1/h3, and eval
   are never permuted.  bridge_sign=-1 reverses the anchor convention.  The tied
   model predicts h1/h3 canonical signs will follow bridge_sign (gauge transport).
   The untied model predicts no coherent response.

2. **shared_trunk vs tied vs untied**: shared_trunk shares the embedding/GRU trunk
   but uses separate scalar output heads for comparison and state.  This separates
   gauge identity (scalar tie) from generic representation sharing.

3. **Paired initialization**: all model variants for a given seed start from cloned
   random weights.  Dropout=0 for deterministic comparison.

4. **Data filters**: --no-comparisons removes held-held comparison rows (bridge-only);
   --no-bridge-anchors removes h0/h2 state anchors (comparison-only).  These create
   the graph×anchor factorial without new substrate generation.

Uses fixed aligned_state_bridge arm and train-only vocabulary throughout.
No BabyLM official evaluation or upload.  This is a small controlled probe.
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
DEFAULT_OUT = PROJECT / "data/causal_gauge_probe"
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
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return str(path)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def extract_names(text: str) -> List[str]:
    names: List[str] = []
    for m in CAP_RE.findall(text):
        if m in NOT_NAMES:
            continue
        if m not in names:
            names.append(m)
    return names


def event_names(event_text: str) -> List[str]:
    return extract_names(event_text)[:2]


def parse_comparison_text(row: Dict[str, Any]) -> Tuple[str, str]:
    e1, e2 = str(row.get("event1", "")), str(row.get("event2", ""))
    if e1 and e2:
        return e1, e2
    text = str(row.get("text", ""))
    m = re.search(r"Event A:\s*(.*?)\s*Event B:\s*(.*?)\s*Did Event A", text)
    if not m:
        raise ValueError(f"could not parse comparison text: {text[:120]}")
    return m.group(1).strip(), m.group(2).strip()


def parse_state_event(row: Dict[str, Any]) -> str:
    ce = str(row.get("cause_event", ""))
    if ce:
        return ce
    prem = str(row.get("premise", ""))
    marker = "The event was this:"
    if marker in prem:
        return prem.split(marker, 1)[1].strip()
    raise ValueError(f"could not parse state event: {prem[:120]}")


def premise_before_event(row: Dict[str, Any]) -> str:
    prem = str(row.get("premise", ""))
    marker = "The event was this:"
    return prem.split(marker, 1)[0].strip() if marker in prem else prem


def parse_hypothesis(row: Dict[str, Any]) -> Tuple[str, str]:
    hyp = str(row.get("hypothesis", ""))
    m = re.search(r"After the event,\s+([A-Z][a-z]+)\s+had the\s+([A-Za-z_]+)\.?", hyp)
    if not m:
        raise ValueError(f"could not parse hypothesis: {hyp}")
    return m.group(1), m.group(2).lower()


def event_object(event_text: str) -> Optional[str]:
    m = re.search(r"During the\s+([A-Za-z_]+)\s+episode", event_text)
    return m.group(1).lower() if m else None


def normalize_event_for_candidate(event_text: str, candidate: str, names: Sequence[str]) -> List[str]:
    toks = TOKEN_RE.findall(event_text)
    out: List[str] = ["<bos>"]
    for t in toks:
        if t == candidate:
            out.append("<cand>")
        elif t in names:
            out.append("<other>")
        else:
            out.append(t.lower())
    out.append("<eos>")
    return out


def normalize_static_for_candidate(prefix_text: str, hyp_text: str, candidate: str, names: Sequence[str]) -> List[str]:
    toks = TOKEN_RE.findall(prefix_text + " <hyp> " + hyp_text)
    out: List[str] = ["<bos>"]
    for t in toks:
        if t == candidate:
            out.append("<cand>")
        elif t in names:
            out.append("<other>")
        else:
            out.append(t.lower())
    out.append("<eos>")
    return out


def rel_family(rel: Optional[str]) -> str:
    if rel in DIRECT_RELS:
        return "direct_anchor"
    if rel in GRAPH_RELS:
        return "graph_transfer"
    return "seen_or_other" if rel else "unknown"


# ── Data structures ──

@dataclass
class StateQuery:
    key: str
    suite: str
    split: str
    arm: str
    event: str
    prefix: str
    hypothesis_text_by_name: Dict[str, str]
    names: Tuple[str, str]
    object_name: str
    is_changed: bool
    label_by_name: Dict[str, bool]
    inverted_target_by_name: Dict[str, bool]
    relation: Optional[str]
    relation_family: str
    initial_pattern: Optional[str]
    static_slot: Optional[int]
    query_kind: str
    metadata: Dict[str, Any]

    @property
    def is_direct_anchor(self) -> bool:
        return self.is_changed and self.relation in DIRECT_RELS


@dataclass
class ComparisonExample:
    row_id: str
    suite: str
    split: str
    arm: str
    event1: str
    event2: str
    names: Tuple[str, str]
    label: bool
    inverted_target: bool
    relation1: Optional[str]
    relation2: Optional[str]
    metadata: Dict[str, Any]


def state_query_key(row: Dict[str, Any]) -> str:
    if row.get("pair_id") is not None:
        return f"{row.get('suite')}|{row.get('pair_id')}|{row.get('query_kind')}"
    rid = str(row.get("row_id"))
    return re.sub(r"_(0|1)$", "", rid)


def build_state_queries(rows: Sequence[Dict[str, Any]], arm: str, parse_errors: List[str]) -> List[StateQuery]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("task") == "state_query":
            by[state_query_key(r)].append(r)
    out: List[StateQuery] = []
    for k, group in by.items():
        if len(group) < 2:
            parse_errors.append(f"state group {k} has {len(group)} rows")
            continue
        try:
            ev = parse_state_event(group[0])
            ns = event_names(ev)
            if len(ns) != 2:
                parse_errors.append(f"state group {k} event names={ns}")
                continue
            prefix = premise_before_event(group[0])
            hyp_by_name, label_by_name, inv_by_name = {}, {}, {}
            obj = None
            for r in group:
                cand, obj_i = parse_hypothesis(r)
                hyp_by_name[cand] = str(r.get("hypothesis", ""))
                label = bool(r.get("label"))
                flip = bool(r.get("global_swap_changes_label", False))
                label_by_name[cand] = label
                inv_by_name[cand] = bool(label) ^ bool(flip)
                obj = obj_i
            if not all(n in label_by_name for n in ns):
                parse_errors.append(f"state group {k} missing labels for names={ns}")
                continue
            ev_obj = event_object(ev)
            is_changed = (obj == ev_obj)
            r0 = group[0]
            out.append(StateQuery(
                key=k, suite=str(r0.get("suite", "")), split=str(r0.get("split", "")), arm=arm,
                event=ev, prefix=prefix, hypothesis_text_by_name=hyp_by_name, names=(ns[0], ns[1]),
                object_name=str(obj), is_changed=bool(is_changed), label_by_name=label_by_name,
                inverted_target_by_name=inv_by_name, relation=r0.get("relation") or r0.get("cause_relation"),
                relation_family=rel_family(r0.get("relation") or r0.get("cause_relation")),
                initial_pattern=r0.get("initial_pattern"), static_slot=r0.get("static_slot"),
                query_kind="changed" if is_changed else "unchanged",
                metadata={"pair_id": r0.get("pair_id"), "row_ids": [rr.get("row_id") for rr in group],
                          "global_swap_changes_label": bool(r0.get("global_swap_changes_label", False)),
                          "raw_query_kind": r0.get("query_kind")}))
        except Exception as e:
            parse_errors.append(f"state group {k}: {type(e).__name__}: {e}")
    return out


def build_comparisons(rows: Sequence[Dict[str, Any]], arm: str, parse_errors: List[str]) -> List[ComparisonExample]:
    out: List[ComparisonExample] = []
    for r in rows:
        if r.get("task") != "relation_comparison":
            continue
        try:
            e1, e2 = parse_comparison_text(r)
            ns: List[str] = []
            for n in event_names(e1) + event_names(e2):
                if n not in ns:
                    ns.append(n)
            if len(ns) != 2:
                parse_errors.append(f"comparison row {r.get('row_id')} names={ns}")
                continue
            label = bool(r.get("label"))
            flip = bool(r.get("global_swap_changes_label", False))
            out.append(ComparisonExample(
                row_id=str(r.get("row_id")), suite=str(r.get("suite", "")), split=str(r.get("split", "")), arm=arm,
                event1=e1, event2=e2, names=(ns[0], ns[1]), label=label,
                inverted_target=bool(label) ^ bool(flip), relation1=r.get("relation1"), relation2=r.get("relation2"),
                metadata={"global_swap_changes_label": flip, "orientation_dependency": r.get("orientation_dependency")}))
        except Exception as e:
            parse_errors.append(f"comparison row {r.get('row_id')}: {type(e).__name__}: {e}")
    return out


# ── Vocabulary ──

class Vocab:
    def __init__(self) -> None:
        self.itos = ["<pad>", "<unk>", "<bos>", "<eos>", "<cand>", "<other>", "<hyp>"]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def add(self, toks: Sequence[str]) -> None:
        for t in toks:
            if t not in self.stoi:
                self.stoi[t] = len(self.itos)
                self.itos.append(t)

    def encode(self, toks: Sequence[str]) -> List[int]:
        return [self.stoi.get(t, 1) for t in toks]


def collect_vocab(vocab: Vocab, states: Sequence[StateQuery], comps: Sequence[ComparisonExample]) -> None:
    for q in states:
        for cand in q.names:
            if q.is_changed:
                vocab.add(normalize_event_for_candidate(q.event, cand, q.names))
            else:
                vocab.add(normalize_static_for_candidate(q.prefix, q.hypothesis_text_by_name[cand], cand, q.names))
    for c in comps:
        for cand in c.names:
            vocab.add(normalize_event_for_candidate(c.event1, cand, c.names))
            vocab.add(normalize_event_for_candidate(c.event2, cand, c.names))


# ── Model ──

class TrunkEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.out_dim = 2 * hidden

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        y, _ = self.gru(x)
        m = mask.unsqueeze(-1).float()
        return (y * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class ScalarHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class TextScorer(nn.Module):
    def __init__(self, trunk: TrunkEncoder, head: ScalarHead) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = head

    def forward_ids(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(ids, mask))


def make_probe_model(vocab_size: int, mode: str, emb_dim: int = 48, hidden: int = 64) -> nn.Module:
    """Build a ProbeModel with the specified sharing mode.

    Modes:
      tied:         one trunk + one head for both comparison and state
      shared_trunk: one trunk shared, separate scalar heads
      untied:       fully independent comparison and state scorers
    """
    # Build reference components
    ref_trunk = TrunkEncoder(vocab_size, emb_dim, hidden)
    ref_head = ScalarHead(ref_trunk.out_dim, hidden)

    class ProbeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode
            if mode == "tied":
                self.event_state = TextScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head))
                self.event_cmp = self.event_state
            elif mode == "shared_trunk":
                trunk = copy.deepcopy(ref_trunk)
                self.event_state = TextScorer(trunk, copy.deepcopy(ref_head))
                self.event_cmp = TextScorer(trunk, copy.deepcopy(ref_head))
            elif mode == "untied":
                self.event_state = TextScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head))
                self.event_cmp = TextScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head))
            else:
                raise ValueError(mode)
            self.static = TextScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head))

    return ProbeModel()


def create_paired_models(vocab_size: int, modes: List[str], seed: int,
                         emb_dim: int = 48, hidden: int = 64) -> Dict[str, nn.Module]:
    """Create models with paired initialization: all modes start from identical weights."""
    torch.manual_seed(seed)
    ref_trunk = TrunkEncoder(vocab_size, emb_dim, hidden)
    ref_head = ScalarHead(ref_trunk.out_dim, hidden)
    ref_static_trunk = TrunkEncoder(vocab_size, emb_dim, hidden)
    ref_static_head = ScalarHead(ref_static_trunk.out_dim, hidden)

    models: Dict[str, nn.Module] = {}
    for mode in modes:
        class PM(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.mode = m
                if m == "tied":
                    trunk_c = copy.deepcopy(ref_trunk)
                    head_c = copy.deepcopy(ref_head)
                    self.event_state = TextScorer(trunk_c, head_c)
                    self.event_cmp = self.event_state
                elif m == "shared_trunk":
                    trunk_c = copy.deepcopy(ref_trunk)
                    self.event_state = TextScorer(trunk_c, copy.deepcopy(ref_head))
                    self.event_cmp = TextScorer(trunk_c, copy.deepcopy(ref_head))
                elif m == "untied":
                    self.event_state = TextScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head))
                    self.event_cmp = TextScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head))
                else:
                    raise ValueError(m)
                self.static = TextScorer(copy.deepcopy(ref_static_trunk), copy.deepcopy(ref_static_head))
        models[mode] = PM(mode)
    return models


# ── Scoring ──

def pad_batch(seqs: Sequence[Sequence[int]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs) if seqs else 1
    arr = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        mask[i, :len(s)] = True
    return arr, mask


def score_sequences(scorer: TextScorer, seqs: Sequence[Sequence[int]], device: torch.device) -> torch.Tensor:
    ids, mask = pad_batch(seqs, device)
    return scorer.forward_ids(ids, mask)


def state_scores(model: nn.Module, vocab: Vocab, q: StateQuery, device: torch.device) -> torch.Tensor:
    if q.is_changed:
        seqs = [vocab.encode(normalize_event_for_candidate(q.event, c, q.names)) for c in q.names]
        return score_sequences(model.event_state, seqs, device)
    seqs = [vocab.encode(normalize_static_for_candidate(q.prefix, q.hypothesis_text_by_name[c], c, q.names)) for c in q.names]
    return score_sequences(model.static, seqs, device)


def comparison_prob_same(model: nn.Module, vocab: Vocab, c: ComparisonExample, device: torch.device):
    seqs1 = [vocab.encode(normalize_event_for_candidate(c.event1, n, c.names)) for n in c.names]
    seqs2 = [vocab.encode(normalize_event_for_candidate(c.event2, n, c.names)) for n in c.names]
    s1 = score_sequences(model.event_cmp, seqs1, device)
    s2 = score_sequences(model.event_cmp, seqs2, device)
    p1, p2 = F.softmax(s1, dim=0), F.softmax(s2, dim=0)
    psame = (p1 * p2).sum().clamp(1e-6, 1 - 1e-6)
    return psame, s1, s2


def bce_prob(p: torch.Tensor, target: bool) -> torch.Tensor:
    y = torch.tensor(float(target), dtype=torch.float32, device=p.device)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p))


# ── Training with bridge_sign ──

def get_target_idx(q: StateQuery) -> int:
    return 1 if (q.label_by_name[q.names[1]] and not q.label_by_name[q.names[0]]) else 0


def train_one(model: nn.Module, vocab: Vocab,
              train_states: Sequence[StateQuery], train_comps: Sequence[ComparisonExample],
              device: torch.device, bridge_sign: int, epochs: int, lr: float,
              weight_decay: float, seed: int, cmp_weight: float, state_weight: float,
              static_weight: float, print_every: int = 0) -> Dict[str, Any]:
    model.to(device)
    # Collect unique parameter groups avoiding duplicate parameters from shared trunk.
    # For per-module clipping, we track which parameters belong to which clip group.
    seen_ids = set()
    all_params = []
    clip_groups: List[List[nn.Parameter]] = []  # one per logical module

    def add_module_params(mod: nn.Module, group_name: str):
        group_ps = []
        for p in mod.parameters():
            pid = id(p)
            if pid not in seen_ids:
                seen_ids.add(pid)
                all_params.append(p)
                group_ps.append(p)
        if group_ps:
            clip_groups.append(group_ps)

    add_module_params(model.event_state, "event_state")
    if model.event_cmp is not model.event_state:
        add_module_params(model.event_cmp, "event_cmp")
    add_module_params(model.static, "static")

    opt = torch.optim.AdamW([{"params": all_params}], lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    history: List[Dict[str, float]] = []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        states = list(train_states); comps = list(train_comps)
        rng.shuffle(states); rng.shuffle(comps)
        opt.zero_grad(set_to_none=True)
        losses: List[torch.Tensor] = []

        for q in states:
            scores = state_scores(model, vocab, q, device)
            target_idx = get_target_idx(q)
            # Apply bridge_sign only to direct anchor changed-state rows
            if bridge_sign == -1 and q.is_direct_anchor:
                scores = scores.flip(0)
            loss = F.cross_entropy(scores.view(1, -1), torch.tensor([target_idx], dtype=torch.long, device=device))
            losses.append((state_weight if q.is_changed else static_weight) * loss)

        for c in comps:
            p, _, _ = comparison_prob_same(model, vocab, c, device)
            losses.append(cmp_weight * bce_prob(p, c.label))

        if not losses:
            raise RuntimeError("no losses")
        loss_total = torch.stack(losses).mean()
        loss_total.backward()
        # Per-module clip
        for cg in clip_groups:
            torch.nn.utils.clip_grad_norm_(cg, 5.0)
        opt.step()

        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            with torch.no_grad():
                metrics = quick_train_metrics(model, vocab, train_states, train_comps, device, bridge_sign)
            rec = {"epoch": ep, "loss": float(loss_total.detach().cpu()), **{k: float(v) for k, v in metrics.items()}}
            history.append(rec)
            if print_every:
                print(json.dumps(rec, sort_keys=True), flush=True)

    return {"elapsed_seconds": time.time() - t0, "history": history}


def quick_train_metrics(model: nn.Module, vocab: Vocab, states: Sequence[StateQuery],
                        comps: Sequence[ComparisonExample], device: torch.device,
                        bridge_sign: int) -> Dict[str, float]:
    model.eval()
    n_s = c_s = n_ch = c_ch = n_un = c_un = 0
    # Report post-connector accuracy (what the model is optimizing)
    for q in states:
        scores = state_scores(model, vocab, q, device)
        if bridge_sign == -1 and q.is_direct_anchor:
            scores_eff = scores.flip(0)
        else:
            scores_eff = scores
        tgt = get_target_idx(q)
        pred = int(torch.argmax(scores_eff).item())
        n_s += 1; c_s += int(pred == tgt)
        if q.is_changed:
            n_ch += 1; c_ch += int(pred == tgt)
        else:
            n_un += 1; c_un += int(pred == tgt)
    n_c = c_c = 0
    for ex in comps:
        p, _, _ = comparison_prob_same(model, vocab, ex, device)
        n_c += 1; c_c += int(float(p.detach().cpu()) >= 0.5) == int(ex.label)
    return {
        "train_state_acc": c_s / n_s if n_s else math.nan,
        "train_changed_acc": c_ch / n_ch if n_ch else math.nan,
        "train_unchanged_acc": c_un / n_un if n_un else math.nan,
        "train_cmp_acc": c_c / n_c if n_c else math.nan,
    }


# ── Evaluation ──

def eval_predictions(model: nn.Module, vocab: Vocab, states: Sequence[StateQuery],
                     comps: Sequence[ComparisonExample], device: torch.device,
                     condition: str, arm: str, seed: int, bridge_sign: int):
    model.eval()
    state_rows, comp_rows = [], []
    with torch.no_grad():
        for q in states:
            scores_t = state_scores(model, vocab, q, device)
            scores = [float(x) for x in scores_t.detach().cpu().tolist()]
            probs = [float(x) for x in F.softmax(scores_t, dim=0).detach().cpu().tolist()]
            d_e = scores[0] - scores[1]  # raw event coordinate
            for i, cand in enumerate(q.names):
                label_true = bool(q.label_by_name[cand])
                state_rows.append({
                    "condition": condition, "arm": arm, "seed": seed, "bridge_sign": bridge_sign,
                    "suite": q.suite, "split": q.split, "query_key": q.key,
                    "task": "state_query", "candidate": cand, "candidate_index": i,
                    "score": scores[i], "prob": probs[i], "d_e": d_e,
                    "label_true": label_true, "is_changed": q.is_changed,
                    "query_kind": q.query_kind, "raw_query_kind": q.metadata.get("raw_query_kind"),
                    "relation": q.relation, "relation_family": q.relation_family,
                    "initial_pattern": q.initial_pattern, "static_slot": q.static_slot,
                    "global_swap_changes_label": q.metadata.get("global_swap_changes_label"),
                    "names": list(q.names), "object": q.object_name,
                    "is_direct_anchor": q.is_direct_anchor,
                })
        for c in comps:
            p, s1, s2 = comparison_prob_same(model, vocab, c, device)
            p_f = float(p.detach().cpu())
            logit = math.log(p_f / (1.0 - p_f)) if 0 < p_f < 1 else 0.0
            d_e1 = float(s1[0]) - float(s1[1])  # event1 coordinate
            d_e2 = float(s2[0]) - float(s2[1])  # event2 coordinate
            comp_rows.append({
                "condition": condition, "arm": arm, "seed": seed, "bridge_sign": bridge_sign,
                "suite": c.suite, "split": c.split, "row_id": c.row_id,
                "task": "relation_comparison", "prob_same": p_f, "logit_same": logit,
                "label": bool(c.label), "pred": bool(p_f >= 0.5),
                "correct": bool(p_f >= 0.5) == bool(c.label),
                "signed_margin": logit if bool(c.label) else -logit,
                "relation1": c.relation1, "relation2": c.relation2,
                "d_e1": d_e1, "d_e2": d_e2,
                "event1_scores": [float(x) for x in s1.detach().cpu().tolist()],
                "event2_scores": [float(x) for x in s2.detach().cpu().tolist()],
                "names": list(c.names),
                "global_swap_changes_label": c.metadata.get("global_swap_changes_label"),
                "orientation_dependency": c.metadata.get("orientation_dependency"),
            })
    return state_rows, comp_rows


# ── Summary helpers ──

def mean(xs: Sequence[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def summarize_vals(xs: Sequence[float]) -> Dict[str, Any]:
    return {"n": len(xs), "mean": mean(xs)} if xs else {"n": 0, "mean": None}


def state_choice_records(state_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in state_rows:
        by[(r["condition"], r["arm"], r["seed"], r["bridge_sign"], r["suite"], r["query_key"])].append(r)
    out: List[Dict[str, Any]] = []
    for key, rows in by.items():
        if len(rows) != 2:
            continue
        rows = sorted(rows, key=lambda x: int(x["candidate_index"]))
        pred_i = 0 if rows[0]["score"] >= rows[1]["score"] else 1
        target_i = 0
        for i, r in enumerate(rows):
            if bool(r["label_true"]):
                target_i = i
        margin = rows[target_i]["score"] - rows[1 - target_i]["score"]
        r0 = rows[0]
        out.append({
            "condition": r0["condition"], "arm": r0["arm"], "seed": r0["seed"],
            "bridge_sign": r0["bridge_sign"], "suite": r0["suite"],
            "query_key": r0["query_key"], "correct": int(pred_i == target_i),
            "margin": float(margin), "d_e": float(r0["d_e"]),
            "is_changed": bool(r0["is_changed"]), "query_kind": r0["query_kind"],
            "relation": r0["relation"], "relation_family": r0["relation_family"],
            "initial_pattern": r0["initial_pattern"], "static_slot": r0["static_slot"],
            "is_direct_anchor": bool(r0.get("is_direct_anchor")),
        })
    return out


def pair_both_records(choice_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_base: Dict[Tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in choice_rows:
        if r.get("suite") not in {"paired_state_conservation", "cross_template_state_readout"}:
            continue
        qk = "changed" if r.get("is_changed") else "unchanged"
        parts = str(r["query_key"]).split("|")
        base = "|".join(parts[:-1]) if len(parts) >= 3 else re.sub(r"_(changed|unchanged)$", "", str(r["query_key"]))
        by_base[(r["condition"], r["arm"], r["seed"], r["bridge_sign"], r["suite"], base)][qk] = r
    out: List[Dict[str, Any]] = []
    for key, d in by_base.items():
        if "changed" not in d or "unchanged" not in d:
            continue
        ch, un = d["changed"], d["unchanged"]
        out.append({
            "condition": key[0], "arm": key[1], "seed": key[2], "bridge_sign": key[3],
            "suite": key[4], "base_key": key[5],
            "both_correct": int(ch["correct"] and un["correct"]),
            "changed_correct": int(ch["correct"]), "unchanged_correct": int(un["correct"]),
            "changed_margin": ch["margin"], "unchanged_margin": un["margin"],
            "relation": ch["relation"], "relation_family": ch["relation_family"],
            "initial_pattern": ch["initial_pattern"],
        })
    return out


def compute_central(choices: List[Dict], boths: List[Dict], comp_rows: List[Dict],
                     condition: str, arm: str, seed: int, bridge_sign: int) -> Dict[str, Any]:
    def filt_c(**kw):
        xs = [r for r in choices if r["condition"] == condition and r["arm"] == arm
              and r["seed"] == seed and r["bridge_sign"] == bridge_sign]
        for k, v in kw.items():
            xs = [r for r in xs if r.get(k) == v]
        return xs

    def filt_b(**kw):
        xs = [r for r in boths if r["condition"] == condition and r["arm"] == arm
              and r["seed"] == seed and r["bridge_sign"] == bridge_sign]
        for k, v in kw.items():
            xs = [r for r in xs if r.get(k) == v]
        return xs

    def acc(xs):
        return mean([float(r["correct"]) for r in xs])

    def both_acc(xs):
        return mean([float(r["both_correct"]) for r in xs])

    def mean_margin(xs):
        return mean([float(r["margin"]) for r in xs])

    def mean_de(xs):
        return mean([float(r["d_e"]) for r in xs])

    c = {
        # Canonical state accuracy (true labels, no connector applied)
        "direct_same": acc(filt_c(suite="paired_state_conservation", is_changed=True, relation_family="direct_anchor", initial_pattern="same")),
        "direct_opp": acc(filt_c(suite="paired_state_conservation", is_changed=True, relation_family="direct_anchor", initial_pattern="opposite")),
        "graph_same": acc(filt_c(suite="paired_state_conservation", is_changed=True, relation_family="graph_transfer", initial_pattern="same")),
        "graph_opp": acc(filt_c(suite="paired_state_conservation", is_changed=True, relation_family="graph_transfer", initial_pattern="opposite")),
        "unchanged": acc(filt_c(suite="paired_state_conservation", is_changed=False)),
        "pair_both_graph_same": both_acc(filt_b(suite="paired_state_conservation", relation_family="graph_transfer", initial_pattern="same")),
        "pair_both_graph_opp": both_acc(filt_b(suite="paired_state_conservation", relation_family="graph_transfer", initial_pattern="opposite")),
        # Canonical margins (sign matters for bridge_sign comparison)
        "direct_same_margin": mean_margin(filt_c(suite="paired_state_conservation", is_changed=True, relation_family="direct_anchor", initial_pattern="same")),
        "graph_same_margin": mean_margin(filt_c(suite="paired_state_conservation", is_changed=True, relation_family="graph_transfer", initial_pattern="same")),
        # Raw event coordinates
        "direct_mean_de": mean_de(filt_c(suite="paired_state_conservation", is_changed=True, relation_family="direct_anchor")),
        "graph_mean_de": mean_de(filt_c(suite="paired_state_conservation", is_changed=True, relation_family="graph_transfer")),
    }
    # Comparison metrics
    for suite in ["mixed_held_seen_orientation", "heldheld_unseen_edge_closure"]:
        cs = [r for r in comp_rows if r["condition"] == condition and r["arm"] == arm
              and r["seed"] == seed and r["bridge_sign"] == bridge_sign and r["suite"] == suite]
        c[f"{suite}_acc"] = mean([float(r["correct"]) for r in cs])
        c[f"{suite}_signed_margin"] = mean([float(r["signed_margin"]) for r in cs])
    return c


# ── Data loading with filters ──

def load_dataset(data_root: Path, arm: str, no_comparisons: bool = False,
                 no_bridge_anchors: bool = False) -> Tuple[List[StateQuery], List[ComparisonExample],
                                                            List[StateQuery], List[ComparisonExample],
                                                            List[str], Dict[str, int]]:
    parse_errors: List[str] = []
    common = load_jsonl(data_root / "common_seen_train.jsonl")
    sup = load_jsonl(data_root / "arms" / arm / "train_supervised.jsonl")

    # Apply data filters
    if no_comparisons:
        sup = [r for r in sup if r.get("task") != "relation_comparison"]
    if no_bridge_anchors:
        sup = [r for r in sup if not (r.get("task") == "state_query")]

    train_rows = common + sup
    eval_rows: List[Dict[str, Any]] = []
    for p in sorted((data_root / "eval").glob("*.jsonl")):
        if p.name == "name_permutation_counterfactual.jsonl":
            continue
        eval_rows.extend(load_jsonl(p))
    train_states = build_state_queries(train_rows, arm, parse_errors)
    train_comps = build_comparisons(train_rows, arm, parse_errors)
    eval_states = build_state_queries(eval_rows, arm, parse_errors)
    eval_comps = build_comparisons(eval_rows, arm, parse_errors)
    counts = {
        "train_raw_rows": len(train_rows), "train_state_queries": len(train_states),
        "train_comparisons": len(train_comps), "eval_raw_rows": len(eval_rows),
        "eval_state_queries": len(eval_states), "eval_comparisons": len(eval_comps),
        "parse_errors": len(parse_errors),
        "no_comparisons": no_comparisons, "no_bridge_anchors": no_bridge_anchors,
    }
    return train_states, train_comps, eval_states, eval_comps, parse_errors, counts


# ── Run one cell ──

def run_one(data_root: Path, out: Path, condition: str, arm: str, seed: int,
            bridge_sign: int, epochs: int, lr: float, weight_decay: float,
            emb_dim: int, hidden: int, device: torch.device,
            cmp_weight: float, state_weight: float, static_weight: float,
            print_every: int, no_comparisons: bool, no_bridge_anchors: bool,
            paired_models: Optional[Dict[str, nn.Module]] = None) -> Dict[str, Any]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_states, train_comps, eval_states, eval_comps, parse_errors, counts = \
        load_dataset(data_root, arm, no_comparisons, no_bridge_anchors)
    vocab = Vocab()
    collect_vocab(vocab, train_states, train_comps)

    if paired_models and condition in paired_models:
        model = copy.deepcopy(paired_models[condition])
    else:
        model = make_probe_model(len(vocab.itos), condition, emb_dim, hidden)

    train_info = train_one(model, vocab, train_states, train_comps, device, bridge_sign,
                           epochs, lr, weight_decay, seed, cmp_weight, state_weight, static_weight,
                           print_every=print_every)
    with torch.no_grad():
        final_train = quick_train_metrics(model, vocab, train_states, train_comps, device, bridge_sign)

    train_state_rows, train_comp_rows = eval_predictions(model, vocab, train_states, train_comps, device, condition, arm, seed, bridge_sign)
    eval_state_rows, eval_comp_rows = eval_predictions(model, vocab, eval_states, eval_comps, device, condition, arm, seed, bridge_sign)
    for r in train_state_rows + train_comp_rows:
        r["prediction_split"] = "train"
    for r in eval_state_rows + eval_comp_rows:
        r["prediction_split"] = "eval"

    choices = state_choice_records(eval_state_rows)
    boths = pair_both_records(choices)
    central = compute_central(choices, boths, eval_comp_rows, condition, arm, seed, bridge_sign)

    cell_tag = f"bs{'+' if bridge_sign == 1 else '-'}1"
    if no_comparisons:
        cell_tag += "_nocmp"
    if no_bridge_anchors:
        cell_tag += "_noanchor"

    rec = {
        "condition": condition, "arm": arm, "seed": seed, "bridge_sign": bridge_sign,
        "cell_tag": cell_tag, "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
        "emb_dim": emb_dim, "hidden": hidden, "dropout": 0.0,
        "no_comparisons": no_comparisons, "no_bridge_anchors": no_bridge_anchors,
        "counts": counts, "parse_errors_sample": parse_errors[:10],
        "vocab_size": len(vocab.itos), "train_info": train_info,
        "final_train_metrics": final_train, "central_eval": central,
    }
    run_dir = out / f"{condition}_{cell_tag}_seed{seed}"
    write_json(run_dir / "result.json", rec)
    write_jsonl(run_dir / "eval_state_predictions.jsonl", eval_state_rows)
    write_jsonl(run_dir / "eval_comparison_predictions.jsonl", eval_comp_rows)
    write_jsonl(run_dir / "train_state_predictions.jsonl", train_state_rows)
    write_jsonl(run_dir / "train_comparison_predictions.jsonl", train_comp_rows)
    return rec


# ── Summary ──

def write_summary_and_json(out_dir: Path, results: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    summary: Dict[str, Any] = {"n_results": len(results), "central_by_run": []}
    for r in results:
        ce = r.get("central_eval", {})
        row = {
            "condition": r["condition"], "bridge_sign": r["bridge_sign"],
            "cell_tag": r["cell_tag"], "seed": r["seed"],
            "train_state_acc": r["final_train_metrics"].get("train_state_acc"),
            "train_cmp_acc": r["final_train_metrics"].get("train_cmp_acc"),
            "direct_same": ce.get("direct_same"),
            "direct_opp": ce.get("direct_opp"),
            "graph_same": ce.get("graph_same"),
            "graph_opp": ce.get("graph_opp"),
            "pair_both_graph_same": ce.get("pair_both_graph_same"),
            "unchanged": ce.get("unchanged"),
            "mixed_acc": ce.get("mixed_held_seen_orientation_acc"),
            "mixed_margin": ce.get("mixed_held_seen_orientation_signed_margin"),
            "hh_closure": ce.get("heldheld_unseen_edge_closure_acc"),
            "direct_same_margin": ce.get("direct_same_margin"),
            "graph_same_margin": ce.get("graph_same_margin"),
            "direct_mean_de": ce.get("direct_mean_de"),
            "graph_mean_de": ce.get("graph_mean_de"),
            "elapsed_sec": r["train_info"].get("elapsed_seconds"),
        }
        summary["central_by_run"].append(row)

    # Bridge-sign paired comparisons (same condition, seed, cell filters)
    paired: Dict[str, Any] = {}
    by_key: Dict[Tuple, Dict[int, Dict]] = defaultdict(dict)
    for row in summary["central_by_run"]:
        k = (row["condition"], row["seed"], row.get("cell_tag", "").replace("bs+1", "").replace("bs-1", ""))
        by_key[k][row["bridge_sign"]] = row
    for k, d in by_key.items():
        if 1 in d and -1 in d:
            p1, m1 = d[1], d[-1]
            pkey = f"{k[0]}|seed{k[1]}|{k[2] or 'full'}"
            paired[pkey] = {
                "bs+1_graph_same": p1.get("graph_same"),
                "bs-1_graph_same": m1.get("graph_same"),
                "bs+1_graph_margin": p1.get("graph_same_margin"),
                "bs-1_graph_margin": m1.get("graph_same_margin"),
                "bs+1_direct_de": p1.get("direct_mean_de"),
                "bs-1_direct_de": m1.get("direct_mean_de"),
                "bs+1_graph_de": p1.get("graph_mean_de"),
                "bs-1_graph_de": m1.get("graph_mean_de"),
                "bs+1_mixed_acc": p1.get("mixed_acc"),
                "bs-1_mixed_acc": m1.get("mixed_acc"),
                "bs+1_mixed_margin": p1.get("mixed_margin"),
                "bs-1_mixed_margin": m1.get("mixed_margin"),
                "bs+1_hh_closure": p1.get("hh_closure"),
                "bs-1_hh_closure": m1.get("hh_closure"),
                "margin_sign_flip": (p1.get("graph_same_margin") is not None and m1.get("graph_same_margin") is not None
                                     and (p1["graph_same_margin"] > 0) != (m1["graph_same_margin"] > 0)),
                "de_sign_flip": (p1.get("graph_mean_de") is not None and m1.get("graph_mean_de") is not None
                                 and (p1["graph_mean_de"] > 0) != (m1["graph_mean_de"] > 0)),
            }
    summary["bridge_sign_paired"] = paired

    write_json(out_dir / "causal_gauge_summary.json", summary)

    # Markdown
    lines = ["# research causal gauge-transport probe", "",
             "Fixed aligned arm, train-only vocabulary, dropout=0, paired initialization.", ""]
    cols = ["condition", "bridge_sign", "cell_tag", "seed", "train_state_acc", "train_cmp_acc",
            "direct_same", "graph_same", "pair_both_graph_same", "unchanged",
            "mixed_acc", "mixed_margin", "hh_closure", "graph_same_margin", "graph_mean_de"]
    lines.append("## Per-run central readout\n")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in summary["central_by_run"]:
        vals = []
        for c in cols:
            v = row.get(c)
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("\n## Bridge-sign paired comparisons\n")
    for pk, pv in sorted(paired.items()):
        lines.append(f"### {pk}")
        for k2, v2 in sorted(pv.items()):
            lines.append(f"  {k2}: {v2}")
        lines.append("")
    lines.append("## Scientific reading\n")
    lines.append("If the tied model shows graph_same_margin and graph_mean_de reversing between bs+1 and bs-1 "
                 "while hh_closure and seen references remain stable, this is evidence for absolute gauge transport "
                 "through the shared coordinate.  shared_trunk and untied models should not show the same "
                 "coherent reversal.  anchor-only and comparison-only cells provide the factorial interaction.")
    (out_dir / "causal_gauge_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", default=["tied", "shared_trunk", "untied"])
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--bridge-signs", nargs="+", type=int, default=[1, -1])
    ap.add_argument("--seeds", nargs="+", type=int, default=[29000])
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--emb-dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--cmp-weight", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=1.0)
    ap.add_argument("--static-weight", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--print-every", type=int, default=55)
    # Factorial data filters
    ap.add_argument("--no-comparisons", action="store_true", help="Remove held-held comparison rows (bridge-only)")
    ap.add_argument("--no-bridge-anchors", action="store_true", help="Remove bridge state anchors (comparison-only)")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    results: List[Dict[str, Any]] = []
    for seed in args.seeds:
        # Paired initialization: create reference models, then deep-copy for each run
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        # Need vocab before creating models; load data once for vocab construction
        ts, tc, es, ec, _, _ = load_dataset(args.data_root, args.arm, args.no_comparisons, args.no_bridge_anchors)
        vocab = Vocab()
        collect_vocab(vocab, ts, tc)
        vsize = len(vocab.itos)

        paired_models = create_paired_models(vsize, args.models, seed, args.emb_dim, args.hidden)

        for bs in args.bridge_signs:
            for cond in args.models:
                print(f"=== {cond}/bs={bs:+d}/seed={seed} ===", flush=True)
                rec = run_one(args.data_root, args.out, cond, args.arm, seed, bs,
                              args.epochs, args.lr, args.weight_decay, args.emb_dim, args.hidden,
                              device, args.cmp_weight, args.state_weight, args.static_weight,
                              args.print_every, args.no_comparisons, args.no_bridge_anchors,
                              paired_models=paired_models)
                results.append(rec)
                ce = rec.get("central_eval", {})
                print(json.dumps({
                    "condition": cond, "bridge_sign": bs, "seed": seed,
                    "train_state_acc": rec["final_train_metrics"].get("train_state_acc"),
                    "train_cmp_acc": rec["final_train_metrics"].get("train_cmp_acc"),
                    "direct_same": ce.get("direct_same"), "graph_same": ce.get("graph_same"),
                    "pair_both_graph_same": ce.get("pair_both_graph_same"),
                    "unchanged": ce.get("unchanged"),
                    "mixed_acc": ce.get("mixed_held_seen_orientation_acc"),
                    "mixed_margin": ce.get("mixed_held_seen_orientation_signed_margin"),
                    "graph_same_margin": ce.get("graph_same_margin"),
                    "graph_mean_de": ce.get("graph_mean_de"),
                    "elapsed_seconds": rec["train_info"].get("elapsed_seconds"),
                }, sort_keys=True), flush=True)

    summary = write_summary_and_json(args.out, results, args)
    print(json.dumps({
        "status": "CAUSAL_GAUGE_PROBE_COMPLETE",
        "summary": project_rel(args.out / "causal_gauge_summary.md"),
        "json": project_rel(args.out / "causal_gauge_summary.json"),
        "n_results": len(results),
        "device": str(device),
        "no_official_evaluation_upload_or_leaderboard": True,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
