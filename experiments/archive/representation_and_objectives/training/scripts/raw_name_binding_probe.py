#!/usr/bin/env python3
"""research: raw-name binding probe — remove supplied <cand>/<other> addressing.

Scientific purpose
------------------
research/291 established causal gauge transport inside a harness where participant
names are replaced with <cand>/<other>, supplying candidate addressing.  This
probe removes EXACTLY that interface: names appear as character sequences inside
<N>...</N> delimiters, and a <QRY> token tells the model which participant to
evaluate.  The model must learn to bind the queried character pattern to its
occurrences in the event text.

Design changes from research
---------------------------
1. Names encoded as character sequences: <N> c1 c2 ... </N>
2. Query appended: ... event tokens ... <QRY> <N> candidate_chars </N> <eos>
3. Vocabulary: all 26 lowercase letters always present; <cand>/<other> removed
4. --no-bridge-changed-only: narrow comparison-only filter (only removes changed
   h0/h2 bridge anchors, not unchanged companions)
5. --fresh-rename: evaluate with globally renamed eval participants to test
   whether binding generalises to novel character patterns
6. All other controls identical: bridge_sign, shared_trunk/untied, paired init,
   data filters, dropout=0, per-module clip, print_every, save_predictions

Uses fixed aligned_state_bridge arm and train-only vocabulary throughout.
No BabyLM official evaluation or upload.
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
DEFAULT_OUT = PROJECT / "data/raw_name_binding"
DEFAULT_ARM = "aligned_state_bridge"
DIRECT_RELS = {"h0_dax", "h2_norp"}
GRAPH_RELS = {"h1_mep", "h3_ziv"}

TOKEN_RE = re.compile(r"[A-Za-z_]+|[0-9]+|[.,;:?]")
CAP_RE = re.compile(r"\b[A-Z][a-z]+\b")
NOT_NAMES = {
    "During", "Event", "Did", "After", "At", "The", "A", "B", "From", "This",
    "First", "Then", "Before", "Question", "Answer",
}

# Fresh names for renaming evaluation (disjoint from train and eval names)
FRESH_NAMES = [
    "Bix", "Caz", "Dex", "Fay", "Gol", "Hux", "Ivy", "Jix",
    "Kel", "Lex", "Mox", "Nev", "Pix", "Qay", "Rux", "Wyn",
]


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


# ── Name extraction ──

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


def rel_family(rel: Optional[str]) -> str:
    if rel in DIRECT_RELS:
        return "direct_anchor"
    if rel in GRAPH_RELS:
        return "graph_transfer"
    return "seen_or_other" if rel else "unknown"


# ── NEW: Character-based name encoding with query ──

def name_to_chars(name: str) -> List[str]:
    """Encode a name as a char sequence inside <N>...</N> delimiters."""
    return ["<N>"] + list(name.lower()) + ["</N>"]


def tokenize_with_char_names(text: str, names: Sequence[str]) -> List[str]:
    """Tokenize text, replacing known names with character sequences."""
    toks = TOKEN_RE.findall(text)
    out: List[str] = []
    for t in toks:
        if t in names:
            out.extend(name_to_chars(t))
        else:
            out.append(t.lower())
    return out


def encode_event_with_query(event_text: str, candidate: str, names: Sequence[str]) -> List[str]:
    """Encode event + candidate query for scoring.

    Format: <bos> [event with char-encoded names] <QRY> <N> cand_chars </N> <eos>
    """
    event_toks = tokenize_with_char_names(event_text, names)
    query_toks = name_to_chars(candidate)
    return ["<bos>"] + event_toks + ["<QRY>"] + query_toks + ["<eos>"]


def encode_static_with_query(prefix_text: str, hyp_text: str, candidate: str, names: Sequence[str]) -> List[str]:
    """Encode static premise+hypothesis + candidate query.

    Format: <bos> [prefix with char-encoded names] <hyp> [hyp with char-encoded names] <QRY> <N> cand_chars </N> <eos>
    """
    prefix_toks = tokenize_with_char_names(prefix_text, names)
    hyp_toks = tokenize_with_char_names(hyp_text, names)
    query_toks = name_to_chars(candidate)
    return ["<bos>"] + prefix_toks + ["<hyp>"] + hyp_toks + ["<QRY>"] + query_toks + ["<eos>"]


def rename_text(text: str, name_map: Dict[str, str]) -> str:
    """Replace all occurrences of names in text according to name_map."""
    result = text
    # Sort by length descending to avoid partial replacement
    for old_name in sorted(name_map.keys(), key=len, reverse=True):
        result = result.replace(old_name, name_map[old_name])
    return result


# ── Data structures (same as research) ──

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

    @property
    def is_direct_changed_anchor(self) -> bool:
        """Changed state query for a direct/bridge relation (narrow filter)."""
        return self.is_changed and self.relation in DIRECT_RELS


def state_query_key(row: Dict[str, Any]) -> str:
    if row.get("pair_id") is not None:
        return f"{row.get('suite')}|{row.get('pair_id')}|{row.get('query_kind')}"
    rid = str(row.get("row_id"))
    return re.sub(r"_(0|1)$", "", rid)


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


# ── Vocabulary (MODIFIED: char-based, no <cand>/<other>) ──

class Vocab:
    def __init__(self) -> None:
        self.itos = ["<pad>", "<unk>", "<bos>", "<eos>", "<N>", "</N>", "<QRY>", "<hyp>"]
        # Always include all 26 lowercase letters for character-based name encoding
        for c in "abcdefghijklmnopqrstuvwxyz":
            self.itos.append(c)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def add(self, toks: Sequence[str]) -> None:
        for t in toks:
            if t not in self.stoi:
                self.stoi[t] = len(self.itos)
                self.itos.append(t)

    def encode(self, toks: Sequence[str]) -> List[int]:
        return [self.stoi.get(t, 1) for t in toks]


def collect_vocab(vocab: Vocab, states: Sequence[StateQuery], comps: Sequence[ComparisonExample]) -> None:
    """Add word tokens from training data to vocabulary."""
    for q in states:
        for cand in q.names:
            if q.is_changed:
                vocab.add(encode_event_with_query(q.event, cand, q.names))
            else:
                vocab.add(encode_static_with_query(q.prefix, q.hypothesis_text_by_name[cand], cand, q.names))
    for c in comps:
        for cand in c.names:
            vocab.add(encode_event_with_query(c.event1, cand, c.names))
            vocab.add(encode_event_with_query(c.event2, cand, c.names))


# ── Model (same as research) ──

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


# ── Scoring (MODIFIED: uses char-encoded queries) ──

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
        seqs = [vocab.encode(encode_event_with_query(q.event, c, q.names)) for c in q.names]
        return score_sequences(model.event_state, seqs, device)
    seqs = [vocab.encode(encode_static_with_query(q.prefix, q.hypothesis_text_by_name[c], c, q.names)) for c in q.names]
    return score_sequences(model.static, seqs, device)


def comparison_prob_same(model: nn.Module, vocab: Vocab, c: ComparisonExample, device: torch.device):
    seqs1 = [vocab.encode(encode_event_with_query(c.event1, n, c.names)) for n in c.names]
    seqs2 = [vocab.encode(encode_event_with_query(c.event2, n, c.names)) for n in c.names]
    s1 = score_sequences(model.event_cmp, seqs1, device)
    s2 = score_sequences(model.event_cmp, seqs2, device)
    p1, p2 = F.softmax(s1, dim=0), F.softmax(s2, dim=0)
    psame = (p1 * p2).sum().clamp(1e-6, 1 - 1e-6)
    return psame, s1, s2


def bce_prob(p: torch.Tensor, target: bool) -> torch.Tensor:
    y = torch.tensor(float(target), dtype=torch.float32, device=p.device)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p))


# ── Training with bridge_sign (same logic, narrower filter option) ──

def get_target_idx(q: StateQuery) -> int:
    return 1 if (q.label_by_name[q.names[1]] and not q.label_by_name[q.names[0]]) else 0


def train_one(model: nn.Module, vocab: Vocab,
              train_states: Sequence[StateQuery], train_comps: Sequence[ComparisonExample],
              device: torch.device, bridge_sign: int, epochs: int, lr: float,
              weight_decay: float, seed: int, cmp_weight: float, state_weight: float,
              static_weight: float, print_every: int = 0) -> Dict[str, Any]:
    model.to(device)
    seen_ids = set()
    all_params = []
    clip_groups: List[List[nn.Parameter]] = []

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
            d_e = scores[0] - scores[1]
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
            d_e1 = float(s1[0]) - float(s1[1])
            d_e2 = float(s2[0]) - float(s2[1])
            comp_rows.append({
                "condition": condition, "arm": arm, "seed": seed, "bridge_sign": bridge_sign,
                "suite": c.suite, "split": c.split, "row_id": c.row_id,
                "task": "relation_comparison", "prob_same": p_f, "logit_same": logit,
                "label": bool(c.label), "pred": bool(p_f >= 0.5),
                "correct": bool(p_f >= 0.5) == bool(c.label),
                "signed_margin": logit if bool(c.label) else -logit,
                "d_e1": d_e1, "d_e2": d_e2,
                "relation1": c.relation1, "relation2": c.relation2,
                "names": list(c.names),
                "global_swap_changes_label": c.metadata.get("global_swap_changes_label"),
            })
    return state_rows, comp_rows


def make_fresh_rename_map(eval_names: List[str]) -> Dict[str, str]:
    """Create bijection from eval names to fresh names."""
    assert len(eval_names) <= len(FRESH_NAMES), f"not enough fresh names: need {len(eval_names)}, have {len(FRESH_NAMES)}"
    return dict(zip(sorted(eval_names), FRESH_NAMES[:len(eval_names)]))


def renamed_state_query(q: StateQuery, name_map: Dict[str, str]) -> StateQuery:
    """Create a StateQuery with renamed participants."""
    new_names = tuple(name_map.get(n, n) for n in q.names)
    new_event = rename_text(q.event, name_map)
    new_prefix = rename_text(q.prefix, name_map)
    new_hyp = {name_map.get(k, k): rename_text(v, name_map) for k, v in q.hypothesis_text_by_name.items()}
    new_label = {name_map.get(k, k): v for k, v in q.label_by_name.items()}
    new_inv = {name_map.get(k, k): v for k, v in q.inverted_target_by_name.items()}
    return StateQuery(
        key=q.key + "_renamed", suite=q.suite + "_renamed", split=q.split, arm=q.arm,
        event=new_event, prefix=new_prefix, hypothesis_text_by_name=new_hyp,
        names=new_names, object_name=q.object_name, is_changed=q.is_changed,
        label_by_name=new_label, inverted_target_by_name=new_inv,
        relation=q.relation, relation_family=q.relation_family,
        initial_pattern=q.initial_pattern, static_slot=q.static_slot,
        query_kind=q.query_kind, metadata={**q.metadata, "renamed": True})


def renamed_comparison(c: ComparisonExample, name_map: Dict[str, str]) -> ComparisonExample:
    """Create a ComparisonExample with renamed participants."""
    new_names = tuple(name_map.get(n, n) for n in c.names)
    return ComparisonExample(
        row_id=c.row_id + "_renamed", suite=c.suite + "_renamed", split=c.split, arm=c.arm,
        event1=rename_text(c.event1, name_map), event2=rename_text(c.event2, name_map),
        names=new_names, label=c.label, inverted_target=c.inverted_target,
        relation1=c.relation1, relation2=c.relation2,
        metadata={**c.metadata, "renamed": True})


# ── Central readout (adapted from research) ──

def central_readout(state_rows: List[Dict], comp_rows: List[Dict], bridge_sign: int, arm: str) -> Dict[str, Any]:
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")

    def canonical_target(row):
        """Target under the *canonical* (bs=+1) convention."""
        if bridge_sign == 1:
            return row["label_true"]
        if row.get("is_direct_anchor"):
            return not row["label_true"]
        return row["label_true"]

    # State-based metrics from candidate_index=0 rows
    changed_psc = [r for r in state_rows if r["is_changed"] and r.get("suite", "").startswith("paired_state_conservation") and r["candidate_index"] == 0]
    unchanged_psc = [r for r in state_rows if not r["is_changed"] and r.get("suite", "").startswith("paired_state_conservation") and r["candidate_index"] == 0]
    direct_ch = [r for r in changed_psc if r["relation_family"] == "direct_anchor"]
    graph_ch = [r for r in changed_psc if r["relation_family"] == "graph_transfer"]
    direct_same = [r for r in direct_ch if r.get("initial_pattern") == "same"]
    direct_opp = [r for r in direct_ch if r.get("initial_pattern") == "opposite"]
    graph_same = [r for r in graph_ch if r.get("initial_pattern") == "same"]
    graph_opp = [r for r in graph_ch if r.get("initial_pattern") == "opposite"]

    def choice_acc(rows):
        return mean([float(r["d_e"] > 0) == canonical_target(r) for r in rows]) if rows else float("nan")

    def state_margin(rows):
        return mean([abs(r["d_e"]) * (1 if (float(r["d_e"] > 0) == canonical_target(r)) else -1) for r in rows]) if rows else float("nan")

    def pair_both(pairs_by_key):
        both = []
        for k, rs in pairs_by_key.items():
            ch = [r for r in rs if r["is_changed"]]
            un = [r for r in rs if not r["is_changed"]]
            if ch and un:
                ch_ok = all((float(r["d_e"] > 0) == canonical_target(r)) for r in ch)
                un_ok = all((float(r["d_e"] > 0) == canonical_target(r)) for r in un)
                both.append(float(ch_ok and un_ok))
        return mean(both) if both else float("nan")

    psc0 = [r for r in state_rows if r.get("suite", "").startswith("paired_state_conservation") and r["candidate_index"] == 0]
    psc_pairs: Dict[str, List[Dict]] = defaultdict(list)
    for r in psc0:
        pk = r.get("query_key", "").rsplit("|", 1)[0]
        psc_pairs[pk].append(r)
    graph_same_pairs = {k: rs for k, rs in psc_pairs.items()
                        if any(r["relation_family"] == "graph_transfer" and r.get("initial_pattern") == "same" for r in rs)}

    # Comparison-based metrics from comp_rows
    hh_comps = [r for r in comp_rows if r.get("suite", "").startswith("heldheld")]
    hh_closure = mean([float(r["correct"]) for r in hh_comps]) if hh_comps else float("nan")
    hh_margin = mean([r.get("signed_margin", 0.0) for r in hh_comps]) if hh_comps else float("nan")

    mixed_comps = [r for r in comp_rows if r.get("suite", "").startswith("mixed_held_seen")]
    mixed_acc = mean([float(r["correct"]) for r in mixed_comps]) if mixed_comps else float("nan")
    mixed_margin = mean([r.get("signed_margin", 0.0) for r in mixed_comps]) if mixed_comps else float("nan")

    out = {
        "direct_same": choice_acc(direct_same), "direct_opp": choice_acc(direct_opp),
        "direct_same_margin": state_margin(direct_same),
        "graph_same": choice_acc(graph_same), "graph_opp": choice_acc(graph_opp),
        "graph_same_margin": state_margin(graph_same),
        "unchanged": choice_acc(unchanged_psc),
        "pair_both_graph_same": pair_both(graph_same_pairs),
        "hh_closure": hh_closure, "hh_margin": hh_margin,
        "mixed_acc": mixed_acc, "mixed_margin": mixed_margin,
    }
    return out


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
    ap.add_argument("--cmp-weight", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=1.0)
    ap.add_argument("--static-weight", type=float, default=1.0)
    ap.add_argument("--print-every", type=int, default=55)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-comparisons", action="store_true", help="Remove all comparison rows (bridge-only)")
    ap.add_argument("--no-bridge-changed-only", action="store_true",
                    help="Remove ONLY changed h0/h2 bridge anchors (narrow comparison-only)")
    ap.add_argument("--fresh-rename", action="store_true", help="Also evaluate with fresh-renamed eval participants")
    args = ap.parse_args()

    data_root = args.data_root
    arm_dir = data_root / "arms" / args.arm
    eval_dir = data_root / "eval"

    # Load data
    parse_errors: List[str] = []
    arm_rows = load_jsonl(arm_dir / "train_supervised.jsonl")
    common_rows = load_jsonl(data_root / "common_seen_train.jsonl")
    train_states_arm = build_state_queries(arm_rows, args.arm, parse_errors)
    train_comps_arm = build_comparisons(arm_rows, args.arm, parse_errors)
    train_states_common = build_state_queries(common_rows, "common", parse_errors)

    # Apply data filters
    if args.no_comparisons:
        train_comps_arm = []
    if args.no_bridge_changed_only:
        # Narrow filter: remove ONLY changed h0/h2 bridge anchors
        n_before = len(train_states_arm)
        train_states_arm = [q for q in train_states_arm if not q.is_direct_changed_anchor]
        n_removed = n_before - len(train_states_arm)
        print(f"[narrow bridge filter] removed {n_removed} changed h0/h2 anchors, {len(train_states_arm)} arm states remain", flush=True)

    train_states = train_states_arm + train_states_common
    train_comps = train_comps_arm

    # Load eval suites
    eval_suites: Dict[str, Tuple[List[StateQuery], List[ComparisonExample]]] = {}
    for fp in sorted(eval_dir.iterdir()):
        if fp.suffix != ".jsonl":
            continue
        name = fp.stem
        rows = load_jsonl(fp)
        sq = build_state_queries(rows, args.arm, parse_errors)
        cmp = build_comparisons(rows, args.arm, parse_errors)
        eval_suites[name] = (sq, cmp)

    if parse_errors:
        print(f"WARNING: {len(parse_errors)} parse errors", flush=True)
        for e in parse_errors[:5]:
            print(f"  {e}", file=sys.stderr)

    # Build vocabulary from training data only
    vocab = Vocab()
    collect_vocab(vocab, train_states, train_comps)
    print(f"Vocabulary: {len(vocab.itos)} tokens (train-only + all letters)", flush=True)

    # Encoding sanity check
    sample_q = train_states_arm[0] if train_states_arm else train_states_common[0]
    for cand in sample_q.names:
        toks = encode_event_with_query(sample_q.event, cand, sample_q.names)
        ids = vocab.encode(toks)
        print(f"  Sample encoding for {cand}: {' '.join(toks[:30])} ... ({len(toks)} tokens, {sum(1 for x in ids if x==1)} <unk>)", flush=True)

    # Collect eval names for fresh renaming
    eval_names_set: set = set()
    for name, (sq, cmp) in eval_suites.items():
        for q in sq:
            eval_names_set.update(q.names)
        for c in cmp:
            eval_names_set.update(c.names)
    eval_names_list = sorted(eval_names_set)

    device = torch.device(args.device)
    all_results = []

    for seed in args.seeds:
        models = create_paired_models(len(vocab.itos), args.conditions, seed,
                                      emb_dim=args.emb_dim, hidden=args.hidden)

        for cond in args.conditions:
            model = models[cond]
            for bs in args.bridge_signs:
                cell_tag = f"bs{'+' if bs > 0 else ''}{bs}"
                run_tag = f"{cond}_{cell_tag}_seed{seed}"
                print(f"\n=== {cond}/bs={bs:+d}/seed={seed} ===", flush=True)

                info = train_one(model, vocab, train_states, train_comps, device, bs,
                                 args.epochs, args.lr, args.weight_decay, seed,
                                 args.cmp_weight, args.state_weight, args.static_weight,
                                 print_every=args.print_every)

                # Evaluate on all suites
                all_state_rows, all_comp_rows = [], []
                for suite_name, (sq, cmp) in eval_suites.items():
                    sr, cr = eval_predictions(model, vocab, sq, cmp, device, cond, args.arm, seed, bs)
                    all_state_rows.extend(sr)
                    all_comp_rows.extend(cr)
                # Also eval on train arm state/comp for fit verification
                sr_train, cr_train = eval_predictions(model, vocab, train_states_arm, train_comps_arm, device, cond, args.arm, seed, bs)

                central = central_readout(all_state_rows, all_comp_rows, bs, args.arm)
                central["train_state_acc"] = info["history"][-1]["train_state_acc"]
                central["train_cmp_acc"] = info["history"][-1]["train_cmp_acc"]

                # Fresh renaming evaluation
                rename_central = {}
                renamed_state_rows, renamed_comp_rows = [], []
                if args.fresh_rename and eval_names_list:
                    name_map = make_fresh_rename_map(eval_names_list)
                    renamed_state_rows, renamed_comp_rows = [], []
                    for suite_name, (sq, cmp) in eval_suites.items():
                        rsq = [renamed_state_query(q, name_map) for q in sq]
                        rcmp = [renamed_comparison(c, name_map) for c in cmp]
                        sr, cr = eval_predictions(model, vocab, rsq, rcmp, device, cond, args.arm, seed, bs)
                        renamed_state_rows.extend(sr)
                        renamed_comp_rows.extend(cr)
                    rename_central = central_readout(renamed_state_rows, renamed_comp_rows, bs, args.arm)
                    rename_central = {f"rename_{k}": v for k, v in rename_central.items()}

                result = {
                    "arm": args.arm, "bridge_sign": bs, "cell_tag": cell_tag,
                    "central_eval": {**central, **rename_central},
                    "condition": cond, "seed": seed,
                    "elapsed_seconds": info["elapsed_seconds"],
                    "epochs": args.epochs,
                    "encoding": "raw_name_char_query",
                    "history": {"sampled": info["history"]},
                    "vocab_size": len(vocab.itos),
                    "weight_decay": args.weight_decay,
                    "no_comparisons": args.no_comparisons,
                    "no_bridge_changed_only": args.no_bridge_changed_only,
                    "fresh_rename": args.fresh_rename,
                }
                all_results.append(result)
                print(json.dumps({k: v for k, v in result.items() if k != "history"}, sort_keys=True), flush=True)

                # Save per-run
                run_dir = args.out / run_tag
                write_json(run_dir / "result.json", result)
                write_jsonl(run_dir / "state_predictions.jsonl", all_state_rows)
                write_jsonl(run_dir / "comp_predictions.jsonl", all_comp_rows)
                write_jsonl(run_dir / "train_state_predictions.jsonl", sr_train)
                write_jsonl(run_dir / "train_comp_predictions.jsonl", cr_train)
                if renamed_state_rows:
                    write_jsonl(run_dir / "renamed_state_predictions.jsonl", renamed_state_rows)
                    write_jsonl(run_dir / "renamed_comp_predictions.jsonl", renamed_comp_rows)

    # Write summary
    summary_lines = ["# research raw-name binding probe\n",
                     f"Encoding: character-based names + <QRY> query (no <cand>/<other>)\n",
                     f"Data filters: no_comparisons={args.no_comparisons}, no_bridge_changed_only={args.no_bridge_changed_only}\n",
                     f"Fresh rename: {args.fresh_rename}\n\n",
                     "## Per-run central readout\n\n",
                     "| condition | bridge_sign | seed | train_state | train_cmp | direct_same | graph_same | pair_both | unchanged | hh_closure | mixed_acc | mixed_margin |\n",
                     "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"]
    for r in all_results:
        c = r["central_eval"]
        summary_lines.append(f"| {r['condition']} | {r['bridge_sign']:+d} | {r['seed']} | "
                             f"{c.get('train_state_acc', float('nan')):.3f} | {c.get('train_cmp_acc', float('nan')):.3f} | "
                             f"{c.get('direct_same', float('nan')):.3f} | {c.get('graph_same', float('nan')):.3f} | "
                             f"{c.get('pair_both_graph_same', float('nan')):.3f} | {c.get('unchanged', float('nan')):.3f} | "
                             f"{c.get('hh_closure', float('nan')):.3f} | {c.get('mixed_acc', float('nan')):.3f} | "
                             f"{c.get('mixed_margin', float('nan')):.1f} |\n")
    if args.fresh_rename:
        summary_lines.extend(["\n## Fresh rename readout\n\n",
                              "| condition | bridge_sign | seed | rename_direct_same | rename_graph_same | rename_pair_both | rename_hh_closure | rename_mixed_acc |\n",
                              "|---|---|---:|---:|---:|---:|---:|---:|\n"])
        for r in all_results:
            c = r["central_eval"]
            summary_lines.append(f"| {r['condition']} | {r['bridge_sign']:+d} | {r['seed']} | "
                                 f"{c.get('rename_direct_same', float('nan')):.3f} | {c.get('rename_graph_same', float('nan')):.3f} | "
                                 f"{c.get('rename_pair_both_graph_same', float('nan')):.3f} | "
                                 f"{c.get('rename_hh_closure', float('nan')):.3f} | {c.get('rename_mixed_acc', float('nan')):.3f} |\n")

    md_path = args.out / "raw_name_binding_summary.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("".join(summary_lines), encoding="utf-8")
    write_json(args.out / "raw_name_binding_summary.json", all_results)

    print(json.dumps({
        "device": str(device),
        "json": project_rel(args.out / "raw_name_binding_summary.json"),
        "n_results": len(all_results),
        "no_official_evaluation_upload_or_leaderboard": True,
        "status": "RAW_NAME_BINDING_COMPLETE",
        "summary": project_rel(md_path),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
