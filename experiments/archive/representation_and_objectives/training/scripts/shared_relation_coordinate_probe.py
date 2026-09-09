#!/usr/bin/env python3
"""research: shared latent relation-coordinate probe on raw text.

Scientific purpose
------------------
The research/287 DeBERTa+independent-binary-head probes fit local rows but did not
transport held relation orientation to h1/h3 state updates.  This script tests the
specific inductive-bias hypothesis:

  A learner needs a shared low-dimensional event-output coordinate used by both
  relation-comparison decisions and changed-state decisions.  The role assignment
  itself must remain latent in the raw event text; the model is not given relation
  ids, voices, argument roles, correct slots, or event-role labels.

The contrast is a tied model versus an equally expressive untied model on identical
experience.  Both models see raw event/premise/hypothesis strings only.  A small
regex extracts the two participant *strings* and candidate string from the natural
surface so the model can be queried for each possible final owner; it does not pass
subject/object roles or relation labels to the model.  The event encoder must learn
from the word order and voice morphology which candidate an event gives the object
to.

Data default: research/284 repaired `replace_k16_spread`, because it is already
audited to remove the anti-copy/copy-initial/static-slot equivalences and contains
only informative held-held comparisons, avoiding research's rank-zero neutral rows.

No BabyLM official evaluation or upload.  This is a small controlled probe.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_DATA_ROOT = PROJECT / "data/information_budget_substrate/replace_k16_spread"
DEFAULT_OUT = PROJECT / "data/shared_relation_coordinate_probe"
DEFAULT_ARMS = ["aligned_state_bridge", "inverted_state_bridge"]
DEFAULT_MODELS = ["tied", "untied"]
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
    ns = extract_names(event_text)
    # In this substrate each event has exactly two participant names.  If a parsing
    # issue appears, keep the first two unique names and record it in parse_errors.
    return ns[:2]


def parse_comparison_text(row: Dict[str, Any]) -> Tuple[str, str]:
    # Prefer raw event spans already serialized by the substrate.  These are text
    # fields, not relation/role metadata.  Fall back to parsing the natural prompt.
    e1 = str(row.get("event1", ""))
    e2 = str(row.get("event2", ""))
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
    if marker in prem:
        return prem.split(marker, 1)[0].strip()
    return prem


def parse_hypothesis(row: Dict[str, Any]) -> Tuple[str, str]:
    hyp = str(row.get("hypothesis", ""))
    m = re.search(r"After the event,\s+([A-Z][a-z]+)\s+had the\s+([A-Za-z_]+)\.?", hyp)
    if not m:
        raise ValueError(f"could not parse hypothesis: {hyp}")
    return m.group(1), m.group(2).lower()


def event_object(event_text: str) -> Optional[str]:
    # All controlled event templates begin "During the X episode, ...".
    m = re.search(r"During the\s+([A-Za-z_]+)\s+episode", event_text)
    return m.group(1).lower() if m else None


def tokenize_raw(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


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
    if rel is None:
        return "unknown"
    return "seen_or_other"


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
    # Candidate rows in one binary query share pair_id/query_kind when available.
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
                parse_errors.append(f"state group {k} event names={ns} event={ev}")
                continue
            prefix = premise_before_event(group[0])
            hyp_by_name: Dict[str, str] = {}
            label_by_name: Dict[str, bool] = {}
            inv_by_name: Dict[str, bool] = {}
            obj = None
            for r in group:
                cand, obj_i = parse_hypothesis(r)
                hyp_by_name[cand] = str(r.get("hypothesis", ""))
                label = bool(r.get("label"))
                flip = bool(r.get("global_swap_changes_label", False))
                label_by_name[cand] = label
                inv_by_name[cand] = bool(label) ^ bool(flip)
                obj = obj_i
            # Ensure both candidates have a target.  Some groups may list names in a
            # different order from event surface; use parsed event participants.
            if not all(n in label_by_name for n in ns):
                parse_errors.append(f"state group {k} missing labels for names={ns} labels={list(label_by_name)}")
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
                query_kind="changed" if is_changed else "unchanged", metadata={
                    "pair_id": r0.get("pair_id"), "row_ids": [rr.get("row_id") for rr in group],
                    "global_swap_changes_label": bool(r0.get("global_swap_changes_label", False)),
                    "raw_query_kind": r0.get("query_kind"),
                }))
        except Exception as e:
            parse_errors.append(f"state group {k} parse error: {type(e).__name__}: {e}")
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
                parse_errors.append(f"comparison row {r.get('row_id')} names={ns} e1={e1} e2={e2}")
                continue
            label = bool(r.get("label"))
            flip = bool(r.get("global_swap_changes_label", False))
            out.append(ComparisonExample(
                row_id=str(r.get("row_id")), suite=str(r.get("suite", "")), split=str(r.get("split", "")), arm=arm,
                event1=e1, event2=e2, names=(ns[0], ns[1]), label=label,
                inverted_target=bool(label) ^ bool(flip), relation1=r.get("relation1"), relation2=r.get("relation2"),
                metadata={"global_swap_changes_label": flip, "orientation_dependency": r.get("orientation_dependency")}))
        except Exception as e:
            parse_errors.append(f"comparison row {r.get('row_id')} parse error: {type(e).__name__}: {e}")
    return out


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


class TextScorer(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 48, hidden: int = 64, dropout: float = 0.05) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward_ids(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        y, _ = self.gru(x)
        # masked mean over contextual states is more stable than final state for short templates.
        m = mask.unsqueeze(-1).float()
        pooled = (y * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.out(self.drop(pooled)).squeeze(-1)


class ProbeModel(nn.Module):
    def __init__(self, vocab_size: int, mode: str, emb_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        if mode not in {"tied", "untied"}:
            raise ValueError(mode)
        self.mode = mode
        self.event_state = TextScorer(vocab_size, emb_dim, hidden, dropout)
        self.event_cmp = self.event_state if mode == "tied" else TextScorer(vocab_size, emb_dim, hidden, dropout)
        self.static = TextScorer(vocab_size, emb_dim, hidden, dropout)


def pad_batch(seqs: Sequence[Sequence[int]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs) if seqs else 1
    arr = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        if not s:
            continue
        arr[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        mask[i, :len(s)] = True
    return arr, mask


def score_sequences(scorer: TextScorer, seqs: Sequence[Sequence[int]], device: torch.device) -> torch.Tensor:
    ids, mask = pad_batch(seqs, device)
    return scorer.forward_ids(ids, mask)


def state_scores(model: ProbeModel, vocab: Vocab, q: StateQuery, device: torch.device) -> torch.Tensor:
    seqs: List[List[int]] = []
    if q.is_changed:
        for cand in q.names:
            seqs.append(vocab.encode(normalize_event_for_candidate(q.event, cand, q.names)))
        return score_sequences(model.event_state, seqs, device)
    for cand in q.names:
        seqs.append(vocab.encode(normalize_static_for_candidate(q.prefix, q.hypothesis_text_by_name[cand], cand, q.names)))
    return score_sequences(model.static, seqs, device)


def comparison_prob_same(model: ProbeModel, vocab: Vocab, c: ComparisonExample, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs1 = [vocab.encode(normalize_event_for_candidate(c.event1, cand, c.names)) for cand in c.names]
    seqs2 = [vocab.encode(normalize_event_for_candidate(c.event2, cand, c.names)) for cand in c.names]
    s1 = score_sequences(model.event_cmp, seqs1, device)
    s2 = score_sequences(model.event_cmp, seqs2, device)
    p1 = F.softmax(s1, dim=0)
    p2 = F.softmax(s2, dim=0)
    psame = (p1 * p2).sum().clamp(1e-6, 1 - 1e-6)
    return psame, s1, s2


def bce_prob(p: torch.Tensor, target: bool) -> torch.Tensor:
    y = torch.tensor(float(target), dtype=torch.float32, device=p.device)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p))


def train_one(model: ProbeModel, vocab: Vocab, train_states: Sequence[StateQuery], train_comps: Sequence[ComparisonExample],
              device: torch.device, epochs: int, lr: float, weight_decay: float, seed: int,
              cmp_weight: float, state_weight: float, static_weight: float, print_every: int = 0) -> Dict[str, Any]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    history: List[Dict[str, float]] = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        # Shuffle query order but keep exact full objective each epoch.
        states = list(train_states); comps = list(train_comps)
        rng.shuffle(states); rng.shuffle(comps)
        opt.zero_grad(set_to_none=True)
        losses: List[torch.Tensor] = []
        state_loss_terms = 0; cmp_loss_terms = 0; static_loss_terms = 0
        for q in states:
            scores = state_scores(model, vocab, q, device)
            # one-hot target index under the row labels actually used for training
            target_idx = 0 if q.label_by_name[q.names[0]] else 1
            # Assert the serialized binary labels are one-hot; if not, use CE target of first true.
            if q.label_by_name[q.names[1]] and not q.label_by_name[q.names[0]]:
                target_idx = 1
            loss = F.cross_entropy(scores.view(1, -1), torch.tensor([target_idx], dtype=torch.long, device=device))
            losses.append((state_weight if q.is_changed else static_weight) * loss)
            if q.is_changed:
                state_loss_terms += 1
            else:
                static_loss_terms += 1
        for c in comps:
            p, _, _ = comparison_prob_same(model, vocab, c, device)
            losses.append(cmp_weight * bce_prob(p, c.label))
            cmp_loss_terms += 1
        if not losses:
            raise RuntimeError("no losses")
        loss_total = torch.stack(losses).mean()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            with torch.no_grad():
                metrics = quick_train_metrics(model, vocab, train_states, train_comps, device)
            rec = {"epoch": ep, "loss": float(loss_total.detach().cpu()), **{k: float(v) for k, v in metrics.items()}}
            history.append(rec)
            if print_every:
                print(json.dumps(rec, sort_keys=True), flush=True)
    return {"elapsed_seconds": time.time() - t0, "history": history}


def quick_train_metrics(model: ProbeModel, vocab: Vocab, states: Sequence[StateQuery], comps: Sequence[ComparisonExample], device: torch.device) -> Dict[str, float]:
    model.eval()
    n_s = c_s = n_ch = c_ch = n_un = c_un = 0
    for q in states:
        scores = state_scores(model, vocab, q, device)
        pred = int(torch.argmax(scores).item())
        tgt = 0 if q.label_by_name[q.names[0]] else 1
        if q.label_by_name[q.names[1]] and not q.label_by_name[q.names[0]]:
            tgt = 1
        n_s += 1; c_s += int(pred == tgt)
        if q.is_changed:
            n_ch += 1; c_ch += int(pred == tgt)
        else:
            n_un += 1; c_un += int(pred == tgt)
    n_c = c_c = 0
    for ex in comps:
        p, _, _ = comparison_prob_same(model, vocab, ex, device)
        pred = bool(float(p.detach().cpu()) >= 0.5)
        n_c += 1; c_c += int(pred == ex.label)
    return {
        "train_state_acc": c_s / n_s if n_s else math.nan,
        "train_changed_acc": c_ch / n_ch if n_ch else math.nan,
        "train_unchanged_acc": c_un / n_un if n_un else math.nan,
        "train_cmp_acc": c_c / n_c if n_c else math.nan,
    }


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def eval_predictions(model: ProbeModel, vocab: Vocab, states: Sequence[StateQuery], comps: Sequence[ComparisonExample],
                     device: torch.device, condition: str, arm: str, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    state_rows: List[Dict[str, Any]] = []
    comp_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for q in states:
            scores_t = state_scores(model, vocab, q, device)
            scores = [float(x) for x in scores_t.detach().cpu().tolist()]
            probs = [float(x) for x in F.softmax(scores_t, dim=0).detach().cpu().tolist()]
            for i, cand in enumerate(q.names):
                label_true = bool(q.label_by_name[cand])
                label_inv = bool(q.inverted_target_by_name[cand])
                arm_target = label_inv if arm == "inverted_state_bridge" else label_true
                state_rows.append({
                    "condition": condition, "arm": arm, "seed": seed, "suite": q.suite, "split": q.split,
                    "query_key": q.key, "task": "state_query", "candidate": cand, "candidate_index": i,
                    "score": scores[i], "prob": probs[i], "label_true": label_true,
                    "label_inverted": label_inv, "label_arm_target": arm_target,
                    "is_changed": q.is_changed, "query_kind": q.query_kind, "raw_query_kind": q.metadata.get("raw_query_kind"),
                    "relation": q.relation, "relation_family": q.relation_family,
                    "initial_pattern": q.initial_pattern, "static_slot": q.static_slot,
                    "global_swap_changes_label": q.metadata.get("global_swap_changes_label"),
                    "names": list(q.names), "object": q.object_name,
                })
        for c in comps:
            p, s1, s2 = comparison_prob_same(model, vocab, c, device)
            p_float = float(p.detach().cpu())
            logit = math.log(p_float / (1.0 - p_float))
            for target_name, target_val in [("true", c.label), ("inverted", c.inverted_target), ("arm", c.inverted_target if arm == "inverted_state_bridge" else c.label)]:
                comp_rows.append({
                    "condition": condition, "arm": arm, "seed": seed, "suite": c.suite, "split": c.split,
                    "row_id": c.row_id, "task": "relation_comparison", "target_mode": target_name,
                    "prob_same": p_float, "logit_same": logit, "label": bool(target_val),
                    "pred": bool(p_float >= 0.5), "correct": bool(p_float >= 0.5) == bool(target_val),
                    "signed_margin": logit if bool(target_val) else -logit,
                    "relation1": c.relation1, "relation2": c.relation2,
                    "global_swap_changes_label": c.metadata.get("global_swap_changes_label"),
                    "orientation_dependency": c.metadata.get("orientation_dependency"),
                    "names": list(c.names),
                    "event1_scores": [float(x) for x in s1.detach().cpu().tolist()],
                    "event2_scores": [float(x) for x in s2.detach().cpu().tolist()],
                })
    return state_rows, comp_rows


def mean(xs: Sequence[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def std(xs: Sequence[float]) -> Optional[float]:
    return float(np.std(np.asarray(xs, dtype=float))) if xs else None


def sem(xs: Sequence[float]) -> Optional[float]:
    return float(np.std(np.asarray(xs, dtype=float)) / math.sqrt(len(xs))) if xs else None


def summarize_vals(xs: Sequence[float]) -> Dict[str, Any]:
    return {"n": len(xs), "mean": mean(xs), "std": std(xs), "sem": sem(xs), "values": [float(x) for x in xs]}


def group_key(row: Dict[str, Any], fields: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(row.get(f) for f in fields)


def state_choice_records(state_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in state_rows:
        by[(r["condition"], r["arm"], r["seed"], r["suite"], r["query_key"])].append(r)
    out: List[Dict[str, Any]] = []
    for key, rows in by.items():
        if len(rows) != 2:
            continue
        rows = sorted(rows, key=lambda x: int(x["candidate_index"]))
        pred_i = 0 if rows[0]["score"] >= rows[1]["score"] else 1
        for mode in ["true", "arm"]:
            label_key = "label_true" if mode == "true" else "label_arm_target"
            target_i = None
            for i, r in enumerate(rows):
                if bool(r[label_key]):
                    target_i = i
            if target_i is None:
                continue
            margin = rows[target_i]["score"] - rows[1 - target_i]["score"]
            r0 = rows[0]
            out.append({
                "condition": r0["condition"], "arm": r0["arm"], "seed": r0["seed"], "suite": r0["suite"],
                "query_key": r0["query_key"], "target_mode": mode, "correct": int(pred_i == target_i),
                "margin": float(margin), "pred_index": pred_i, "target_index": target_i,
                "is_changed": bool(r0["is_changed"]), "query_kind": r0["query_kind"],
                "relation": r0["relation"], "relation_family": r0["relation_family"],
                "initial_pattern": r0["initial_pattern"], "static_slot": r0["static_slot"],
                "global_swap_changes_label": r0["global_swap_changes_label"],
            })
    return out


def pair_both_records(choice_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Pair changed and unchanged queries by removing the terminal query-kind marker in the key when pair_id is present.
    by_base: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in choice_rows:
        if r.get("suite") not in {"paired_state_conservation", "cross_template_state_readout"}:
            continue
        qk = "changed" if r.get("is_changed") else "unchanged"
        # query_key is suite|pair_id|query_kind; pair_id contains no query_kind.
        parts = str(r["query_key"]).split("|")
        if len(parts) >= 3:
            base = "|".join(parts[:-1])
        else:
            base = re.sub(r"_(changed|unchanged)$", "", str(r["query_key"]))
        by_base[(r["condition"], r["arm"], r["seed"], r["suite"], r["target_mode"], base)][qk] = r
    out: List[Dict[str, Any]] = []
    for key, d in by_base.items():
        if "changed" not in d or "unchanged" not in d:
            continue
        ch = d["changed"]; un = d["unchanged"]
        out.append({
            "condition": key[0], "arm": key[1], "seed": key[2], "suite": key[3], "target_mode": key[4],
            "base_key": key[5], "both_correct": int(ch["correct"] and un["correct"]),
            "changed_correct": int(ch["correct"]), "unchanged_correct": int(un["correct"]),
            "changed_margin": ch["margin"], "unchanged_margin": un["margin"],
            "relation": ch["relation"], "relation_family": ch["relation_family"],
            "initial_pattern": ch["initial_pattern"], "static_slot": ch["static_slot"],
        })
    return out


def summarize_group(rows: Sequence[Dict[str, Any]], value_field: str = "correct") -> Dict[str, Any]:
    vals = [float(r[value_field]) for r in rows if r.get(value_field) is not None]
    return summarize_vals(vals)


def aggregate_run(state_rows: Sequence[Dict[str, Any]], comp_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    choices = state_choice_records(state_rows)
    boths = pair_both_records(choices)
    agg: Dict[str, Any] = {"state_choice": {}, "pair_both": {}, "comparison": {}, "central": {}}

    for fields, label, source, valfield in [
        (["condition", "arm", "target_mode", "suite", "is_changed", "relation_family", "initial_pattern"], "by_suite_family_pattern", choices, "correct"),
        (["condition", "arm", "target_mode", "suite", "relation", "initial_pattern"], "by_relation_pattern", choices, "correct"),
    ]:
        d: Dict[str, Any] = {}
        groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
        for r in source:
            groups[group_key(r, fields)].append(r)
        for k, rs in groups.items():
            d[json.dumps(dict(zip(fields, k)), sort_keys=True)] = summarize_group(rs, valfield)
        agg["state_choice"][label] = d

    groups2: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in boths:
        groups2[group_key(r, ["condition", "arm", "target_mode", "suite", "relation_family", "initial_pattern"])].append(r)
    for k, rs in groups2.items():
        agg["pair_both"][json.dumps(dict(zip(["condition", "arm", "target_mode", "suite", "relation_family", "initial_pattern"], k)), sort_keys=True)] = summarize_group(rs, "both_correct")

    groups3: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in comp_rows:
        groups3[group_key(r, ["condition", "arm", "target_mode", "suite"])].append(r)
    for k, rs in groups3.items():
        acc = summarize_group(rs, "correct")
        margins = summarize_vals([float(r["signed_margin"]) for r in rs])
        pred_true = summarize_vals([1.0 if r["pred"] else 0.0 for r in rs])
        agg["comparison"][json.dumps(dict(zip(["condition", "arm", "target_mode", "suite"], k)), sort_keys=True)] = {"acc": acc, "signed_margin": margins, "pred_true_frac": pred_true}

    # Central compact table.
    for condition in sorted({r["condition"] for r in state_rows}):
        for arm in sorted({r["arm"] for r in state_rows}):
            for mode in ["true", "arm"]:
                key = f"{condition}|{arm}|{mode}"
                def filt_choice(**kw):
                    xs = [r for r in choices if r["condition"] == condition and r["arm"] == arm and r["target_mode"] == mode]
                    for kk, vv in kw.items():
                        xs = [r for r in xs if r.get(kk) == vv]
                    return xs
                def filt_both(**kw):
                    xs = [r for r in boths if r["condition"] == condition and r["arm"] == arm and r["target_mode"] == mode]
                    for kk, vv in kw.items():
                        xs = [r for r in xs if r.get(kk) == vv]
                    return xs
                def acc(xs):
                    return mean([float(r["correct"]) for r in xs])
                def both_acc(xs):
                    return mean([float(r["both_correct"]) for r in xs])
                central = {
                    "psc_changed_same_direct": acc(filt_choice(suite="paired_state_conservation", is_changed=True, relation_family="direct_anchor", initial_pattern="same")),
                    "psc_changed_same_graph": acc(filt_choice(suite="paired_state_conservation", is_changed=True, relation_family="graph_transfer", initial_pattern="same")),
                    "psc_changed_opp_graph": acc(filt_choice(suite="paired_state_conservation", is_changed=True, relation_family="graph_transfer", initial_pattern="opposite")),
                    "psc_unchanged": acc(filt_choice(suite="paired_state_conservation", is_changed=False)),
                    "psc_pair_both_same_graph": both_acc(filt_both(suite="paired_state_conservation", relation_family="graph_transfer", initial_pattern="same")),
                    "psc_pair_both_opp_graph": both_acc(filt_both(suite="paired_state_conservation", relation_family="graph_transfer", initial_pattern="opposite")),
                    "xt_changed_same_graph": acc(filt_choice(suite="cross_template_state_readout", is_changed=True, relation_family="graph_transfer", initial_pattern="same")),
                    "xt_pair_both_same_graph": both_acc(filt_both(suite="cross_template_state_readout", relation_family="graph_transfer", initial_pattern="same")),
                }
                for suite in ["mixed_held_seen_orientation", "heldheld_unseen_edge_closure"]:
                    cs = [r for r in comp_rows if r["condition"] == condition and r["arm"] == arm and r["target_mode"] == mode and r["suite"] == suite]
                    central[f"{suite}_acc"] = mean([float(r["correct"]) for r in cs])
                    central[f"{suite}_signed_margin"] = mean([float(r["signed_margin"]) for r in cs])
                agg["central"][key] = central
    return agg


def load_dataset(data_root: Path, arm: str) -> Tuple[List[StateQuery], List[ComparisonExample], List[StateQuery], List[ComparisonExample], List[str], Dict[str, int]]:
    parse_errors: List[str] = []
    common = load_jsonl(data_root / "common_seen_train.jsonl")
    sup = load_jsonl(data_root / "arms" / arm / "train_supervised.jsonl")
    train_rows = common + sup
    eval_rows: List[Dict[str, Any]] = []
    for p in sorted((data_root / "eval").glob("*.jsonl")):
        if p.name == "name_permutation_counterfactual.jsonl":
            # Not central for this factorized readout; it can be added later.
            continue
        eval_rows.extend(load_jsonl(p))
    train_states = build_state_queries(train_rows, arm, parse_errors)
    train_comps = build_comparisons(train_rows, arm, parse_errors)
    eval_states = build_state_queries(eval_rows, arm, parse_errors)
    eval_comps = build_comparisons(eval_rows, arm, parse_errors)
    counts = {
        "train_raw_rows": len(train_rows), "train_state_queries": len(train_states), "train_comparisons": len(train_comps),
        "eval_raw_rows": len(eval_rows), "eval_state_queries": len(eval_states), "eval_comparisons": len(eval_comps),
        "parse_errors": len(parse_errors),
    }
    return train_states, train_comps, eval_states, eval_comps, parse_errors, counts


def run_one(data_root: Path, out: Path, condition: str, arm: str, seed: int, epochs: int, lr: float,
            weight_decay: float, emb_dim: int, hidden: int, dropout: float, device: torch.device,
            cmp_weight: float, state_weight: float, static_weight: float, print_every: int,
            vocab_scope: str) -> Dict[str, Any]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_states, train_comps, eval_states, eval_comps, parse_errors, counts = load_dataset(data_root, arm)
    vocab = Vocab()
    if vocab_scope == "train":
        collect_vocab(vocab, train_states, train_comps)
    elif vocab_scope == "train_eval":
        collect_vocab(vocab, train_states + eval_states, train_comps + eval_comps)
    else:
        raise ValueError(f"unknown vocab_scope={vocab_scope}")
    model = ProbeModel(len(vocab.itos), condition, emb_dim, hidden, dropout)
    train_info = train_one(model, vocab, train_states, train_comps, device, epochs, lr, weight_decay, seed,
                           cmp_weight, state_weight, static_weight, print_every=print_every)
    with torch.no_grad():
        final_train = quick_train_metrics(model, vocab, train_states, train_comps, device)
    train_state_rows, train_comp_rows = eval_predictions(model, vocab, train_states, train_comps, device, condition, arm, seed)
    eval_state_rows, eval_comp_rows = eval_predictions(model, vocab, eval_states, eval_comps, device, condition, arm, seed)
    # Mark split explicitly because train/eval prediction rows share fields.
    for r in train_state_rows + train_comp_rows:
        r["prediction_split"] = "train"
    for r in eval_state_rows + eval_comp_rows:
        r["prediction_split"] = "eval"
    agg_train = aggregate_run(train_state_rows, train_comp_rows)
    agg_eval = aggregate_run(eval_state_rows, eval_comp_rows)
    rec: Dict[str, Any] = {
        "condition": condition, "arm": arm, "seed": seed, "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
        "emb_dim": emb_dim, "hidden": hidden, "dropout": dropout, "vocab_scope": vocab_scope,
        "cmp_weight": cmp_weight, "state_weight": state_weight, "static_weight": static_weight,
        "counts": counts, "parse_errors_sample": parse_errors[:20], "vocab_size": len(vocab.itos),
        "train_info": train_info, "final_train_metrics": final_train,
        "central_eval": agg_eval["central"].get(f"{condition}|{arm}|arm", {}),
        "central_eval_true": agg_eval["central"].get(f"{condition}|{arm}|true", {}),
    }
    run_dir = out / f"{condition}_{arm}_seed{seed}"
    write_json(run_dir / "result.json", rec)
    write_json(run_dir / "train_aggregate.json", agg_train)
    write_json(run_dir / "eval_aggregate.json", agg_eval)
    write_jsonl(run_dir / "train_state_predictions.jsonl", train_state_rows)
    write_jsonl(run_dir / "train_comparison_predictions.jsonl", train_comp_rows)
    write_jsonl(run_dir / "eval_state_predictions.jsonl", eval_state_rows)
    write_jsonl(run_dir / "eval_comparison_predictions.jsonl", eval_comp_rows)
    return rec


def summarize_all(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_results": len(results), "groups": {}, "central_by_result": []}
    for r in results:
        ce = r.get("central_eval", {})
        cet = r.get("central_eval_true", {})
        row = {
            "condition": r["condition"], "arm": r["arm"], "seed": r["seed"],
            "train_state_acc": r["final_train_metrics"].get("train_state_acc"),
            "train_cmp_acc": r["final_train_metrics"].get("train_cmp_acc"),
            "arm_psc_same_graph": ce.get("psc_changed_same_graph"),
            "arm_pair_both_same_graph": ce.get("psc_pair_both_same_graph"),
            "arm_mixed_acc": ce.get("mixed_held_seen_orientation_acc"),
            "arm_mixed_margin": ce.get("mixed_held_seen_orientation_signed_margin"),
            "true_psc_same_graph": cet.get("psc_changed_same_graph"),
            "true_pair_both_same_graph": cet.get("psc_pair_both_same_graph"),
            "true_mixed_acc": cet.get("mixed_held_seen_orientation_acc"),
            "true_mixed_margin": cet.get("mixed_held_seen_orientation_signed_margin"),
            "direct_same": ce.get("psc_changed_same_direct"),
            "unchanged": ce.get("psc_unchanged"),
            "hh_closure_acc": ce.get("heldheld_unseen_edge_closure_acc"),
        }
        out["central_by_result"].append(row)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in out["central_by_result"]:
        groups[(row["condition"], row["arm"])].append(row)
    metrics = [k for k in out["central_by_result"][0].keys() if k not in {"condition", "arm", "seed"}] if out["central_by_result"] else []
    for k, rows in groups.items():
        g: Dict[str, Any] = {"n": len(rows)}
        for m in metrics:
            vals = [float(row[m]) for row in rows if row.get(m) is not None and not (isinstance(row[m], float) and math.isnan(row[m]))]
            g[m] = summarize_vals(vals)
        out["groups"][f"{k[0]}|{k[1]}"] = g
    # Difference summaries tied-untied by arm on central graph transfer metrics.
    diffs: Dict[str, Any] = {}
    by_arm_seed: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in out["central_by_result"]:
        by_arm_seed[(row["arm"], int(row["seed"]))][row["condition"]] = row
    for metric in ["arm_psc_same_graph", "arm_pair_both_same_graph", "arm_mixed_acc", "arm_mixed_margin", "true_mixed_margin", "hh_closure_acc"]:
        vals_by_arm: Dict[str, List[float]] = defaultdict(list)
        for (arm, seed), d in by_arm_seed.items():
            if "tied" in d and "untied" in d and d["tied"].get(metric) is not None and d["untied"].get(metric) is not None:
                vals_by_arm[arm].append(float(d["tied"][metric]) - float(d["untied"][metric]))
        for arm, vals in vals_by_arm.items():
            diffs[f"tied_minus_untied|{arm}|{metric}"] = summarize_vals(vals)
    out["paired_differences"] = diffs
    return out


def write_summary(out_dir: Path, summary: Dict[str, Any], args: argparse.Namespace) -> None:
    lines: List[str] = []
    lines.append("# research shared relation-coordinate probe")
    lines.append("")
    lines.append("This controlled probe trains small raw-text models on the repaired research/284 `replace_k16_spread` substrate.  Both tied and untied models receive identical strings and labels.  The tied model uses one candidate-event scorer for both relation comparisons and changed-state updates; the untied model has separate comparison and state event scorers with at least as much capacity.  Neither model receives relation ids, voices, subject/object roles, correct slots, or event-role labels as input; candidate names are extracted from the surface and marked only as `<cand>` versus `<other>` inside the raw event string.")
    lines.append("")
    lines.append(f"- data root: `{args.data_root}`")
    lines.append(f"- conditions: {args.models}")
    lines.append(f"- arms: {args.arms}")
    lines.append(f"- seeds: {args.seeds}")
    lines.append(f"- epochs/lr: {args.epochs} / {args.lr}")
    lines.append(f"- vocab scope: `{args.vocab_scope}`")
    lines.append("")
    lines.append("## Central per-result readout")
    lines.append("")
    cols = ["condition", "arm", "seed", "train_state_acc", "train_cmp_acc", "direct_same", "arm_psc_same_graph", "arm_pair_both_same_graph", "unchanged", "arm_mixed_acc", "arm_mixed_margin", "true_mixed_margin", "hh_closure_acc"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in summary.get("central_by_result", []):
        vals: List[str] = []
        for c in cols:
            v = row.get(c)
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Group means")
    lines.append("")
    gcols = ["group", "n", "train_state_acc", "train_cmp_acc", "direct_same", "arm_psc_same_graph", "arm_pair_both_same_graph", "unchanged", "arm_mixed_acc", "arm_mixed_margin", "hh_closure_acc"]
    lines.append("| " + " | ".join(gcols) + " |")
    lines.append("|" + "|".join(["---"] * len(gcols)) + "|")
    for gname, g in sorted(summary.get("groups", {}).items()):
        vals = [gname, str(g.get("n"))]
        for c in gcols[2:]:
            m = g.get(c, {}).get("mean") if isinstance(g.get(c), dict) else None
            vals.append("n/a" if m is None else f"{m:.3f}")
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Tied minus untied paired differences")
    lines.append("")
    lines.append("| comparison | n | mean | values |")
    lines.append("|---|---:|---:|---|")
    for k, d in sorted(summary.get("paired_differences", {}).items()):
        vals = d.get("values", [])
        lines.append(f"| {k} | {d.get('n')} | {d.get('mean') if d.get('mean') is not None else 'n/a'} | {vals} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("A positive factorization result requires the tied model, but not the untied equal-capacity model, to recover graph-transfer h1/h3 changed-state choices on same-initial rows under the arm's installed coordinate, preserve unchanged facts, and show corresponding signed mixed held-seen orientation.  Direct-anchor success alone is not enough; it only means h0/h2 bridge rows were fitted.  If tied succeeds while untied fails, the missing ingredient in the DeBERTa independent-head probes is shared computational factorization rather than more evidence or supplied role addresses.  If tied also fails after train fit, the surface parser/encoder or objective still lacks the needed latent role assignment; if both tied and untied succeed, the result would indicate extra capacity or easier optimization rather than the sharing constraint.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- summary JSON: `{project_rel(out_dir / 'shared_coordinate_summary.json')}`")
    lines.append(f"- per-run directories: `{project_rel(out_dir)}`")
    (out_dir / "shared_coordinate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[28800])
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--emb-dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--cmp-weight", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=1.0)
    ap.add_argument("--static-weight", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--vocab-scope", choices=["train", "train_eval"], default="train_eval")
    ap.add_argument("--print-every", type=int, default=0)
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # Important for small jobs on shared H100s: deterministic enough for comparison,
    # without requiring expensive exact determinism.
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    results: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for arm in args.arms:
            for model_name in args.models:
                print(f"=== {model_name}/{arm}/seed={seed} ===", flush=True)
                rec = run_one(args.data_root, args.out, model_name, arm, seed, args.epochs, args.lr,
                              args.weight_decay, args.emb_dim, args.hidden, args.dropout, device,
                              args.cmp_weight, args.state_weight, args.static_weight, args.print_every,
                              args.vocab_scope)
                results.append(rec)
                ce = rec.get("central_eval", {})
                print(json.dumps({
                    "condition": model_name, "arm": arm, "seed": seed,
                    "train_state_acc": rec["final_train_metrics"].get("train_state_acc"),
                    "train_cmp_acc": rec["final_train_metrics"].get("train_cmp_acc"),
                    "direct_same": ce.get("psc_changed_same_direct"),
                    "graph_same": ce.get("psc_changed_same_graph"),
                    "pair_both_graph_same": ce.get("psc_pair_both_same_graph"),
                    "unchanged": ce.get("psc_unchanged"),
                    "mixed_acc": ce.get("mixed_held_seen_orientation_acc"),
                    "mixed_margin": ce.get("mixed_held_seen_orientation_signed_margin"),
                    "elapsed_seconds": rec["train_info"].get("elapsed_seconds"),
                }, sort_keys=True), flush=True)
    summary = summarize_all(results)
    summary.update({
        "status": "SHARED_RELATION_COORDINATE_PROBE_COMPLETE",
        "data_root": str(args.data_root),
        "out": str(args.out),
        "device": str(device),
        "no_official_evaluation_upload_or_leaderboard": True,
        "input_restriction": "model inputs use raw event/premise/hypothesis strings with candidate/other surface markers; relation ids, voices, argument roles, correct slots, and event-role labels are not input features",
        "vocab_scope": args.vocab_scope,
    })
    write_json(args.out / "shared_coordinate_summary.json", summary)
    write_summary(args.out, summary, args)
    print(json.dumps({
        "status": summary["status"],
        "summary": project_rel(args.out / "shared_coordinate_summary.md"),
        "json": project_rel(args.out / "shared_coordinate_summary.json"),
        "n_results": len(results),
        "device": str(device),
        "no_official_evaluation_upload_or_leaderboard": True,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
