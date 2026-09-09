#!/usr/bin/env python3
"""research: Context-free SpanMatcher gauge probe.

Scientific purpose
------------------
research-293 showed that raw-name query-attention fails to generalize candidate
matching to held names, even with full training fit. The diagnosis: the GRU-based
attention entangles name identity with contextual position, preventing compositional
character generalization.

This probe separates candidate binding from relational reasoning:
- SpanMatcher: tiny character-GRU encodes name strings OUTSIDE the main GRU,
  then cosine similarity + softmax produces match probabilities.
- Match-gated embeddings: at name positions, replace token embedding with
  soft interpolation between learned cand_embed and other_embed.
- Standard shared_trunk/tied/untied GRU + heads for gauge transport.
- Oracle mode: deterministic exact character matching (= supplied harness equivalent).

Controls: oracle/learned matcher, bridge_sign ±1, comparison-only filter,
fresh-rename evaluation, matching accuracy reporting.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_DATA_ROOT = PROJECT / "data/information_budget_substrate/replace_k16_spread"
DEFAULT_OUT = PROJECT / "data/span_matcher_gauge"
DEFAULT_ARM = "aligned_state_bridge"
DIRECT_RELS = {"h0_dax", "h2_norp"}
GRAPH_RELS = {"h1_mep", "h3_ziv"}

TOKEN_RE = re.compile(r"[A-Za-z_]+|[0-9]+|[.,;:?]")
CAP_RE = re.compile(r"\b[A-Z][a-z]+\b")
NOT_NAMES = {
    "During", "Event", "Did", "After", "At", "The", "A", "B", "From", "This",
    "First", "Then", "Before", "Question", "Answer",
}


def project_rel(path) -> str:
    try: return str(Path(path).relative_to(Path.cwd()))
    except Exception: return str(path)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists(): return rows
    for line in path.open(encoding="utf-8"):
        s = line.strip()
        if s: rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


# ── Name extraction (same as research) ──

def extract_names(text: str) -> List[str]:
    names = []
    for m in CAP_RE.findall(text):
        if m in NOT_NAMES: continue
        if m not in names: names.append(m)
    return names


def event_names(event_text: str) -> List[str]:
    return extract_names(event_text)[:2]


def parse_comparison_text(row) -> Tuple[str, str]:
    e1, e2 = str(row.get("event1", "")), str(row.get("event2", ""))
    if e1 and e2: return e1, e2
    text = str(row.get("text", ""))
    m = re.search(r"Event A:\s*(.*?)\s*Event B:\s*(.*?)\s*Did Event A", text)
    if not m: raise ValueError(f"parse fail: {text[:120]}")
    return m.group(1).strip(), m.group(2).strip()


def parse_state_event(row) -> str:
    ce = str(row.get("cause_event", ""))
    if ce: return ce
    prem = str(row.get("premise", ""))
    marker = "The event was this:"
    if marker in prem: return prem.split(marker, 1)[1].strip()
    raise ValueError(f"parse fail: {prem[:120]}")


def premise_before_event(row) -> str:
    prem = str(row.get("premise", ""))
    marker = "The event was this:"
    return prem.split(marker, 1)[0].strip() if marker in prem else prem


def parse_hypothesis(row) -> Tuple[str, str]:
    hyp = str(row.get("hypothesis", ""))
    m = re.search(r"After the event,\s+([A-Z][a-z]+)\s+had the\s+([A-Za-z_]+)\.?", hyp)
    if not m: raise ValueError(f"hyp parse fail: {hyp}")
    return m.group(1), m.group(2).lower()


def event_object(event_text: str) -> Optional[str]:
    m = re.search(r"During the\s+([A-Za-z_]+)\s+episode", event_text)
    return m.group(1).lower() if m else None


def rel_family(rel) -> str:
    if rel in DIRECT_RELS: return "direct_anchor"
    if rel in GRAPH_RELS: return "graph_transfer"
    return "seen_or_other" if rel else "unknown"


# ── Data structures (same as research) ──

@dataclass
class StateQuery:
    key: str; suite: str; split: str; arm: str
    event: str; prefix: str; hypothesis_text_by_name: Dict[str, str]
    names: Tuple[str, str]; object_name: str
    is_changed: bool; label_by_name: Dict[str, bool]
    inverted_target_by_name: Dict[str, bool]
    relation: Optional[str]; relation_family: str
    initial_pattern: Optional[str]; static_slot: Optional[int]
    query_kind: str; metadata: Dict[str, Any]

    @property
    def is_direct_anchor(self) -> bool:
        return self.is_changed and self.relation in DIRECT_RELS


@dataclass
class ComparisonExample:
    row_id: str; suite: str; split: str; arm: str
    event1: str; event2: str; names: Tuple[str, str]
    label: bool; inverted_target: bool
    relation1: Optional[str]; relation2: Optional[str]
    metadata: Dict[str, Any]


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
            ev = parse_state_event(group[0])
            ns = event_names(ev)
            if len(ns) != 2:
                parse_errors.append(f"state group {k} names={ns}"); continue
            prefix = premise_before_event(group[0])
            hyp_by_name, label_by_name, inv_by_name = {}, {}, {}
            obj = None
            for r in group:
                cand, obj_i = parse_hypothesis(r)
                hyp_by_name[cand] = str(r.get("hypothesis", ""))
                label = bool(r.get("label"))
                flip = bool(r.get("global_swap_changes_label", False))
                label_by_name[cand] = label; inv_by_name[cand] = bool(label) ^ bool(flip); obj = obj_i
            if not all(n in label_by_name for n in ns):
                parse_errors.append(f"state group {k} missing labels"); continue
            ev_obj = event_object(ev)
            is_changed = (obj == ev_obj)
            r0 = group[0]
            out.append(StateQuery(
                key=k, suite=str(r0.get("suite","")), split=str(r0.get("split","")), arm=arm,
                event=ev, prefix=prefix, hypothesis_text_by_name=hyp_by_name,
                names=(ns[0], ns[1]), object_name=str(obj), is_changed=bool(is_changed),
                label_by_name=label_by_name, inverted_target_by_name=inv_by_name,
                relation=r0.get("relation") or r0.get("cause_relation"),
                relation_family=rel_family(r0.get("relation") or r0.get("cause_relation")),
                initial_pattern=r0.get("initial_pattern"), static_slot=r0.get("static_slot"),
                query_kind="changed" if is_changed else "unchanged",
                metadata={"pair_id":r0.get("pair_id"), "row_ids":[rr.get("row_id") for rr in group],
                          "global_swap_changes_label":bool(r0.get("global_swap_changes_label",False)),
                          "raw_query_kind":r0.get("query_kind")}))
        except Exception as e:
            parse_errors.append(f"state group {k}: {e}")
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
            if len(ns) != 2:
                parse_errors.append(f"cmp {r.get('row_id')} names={ns}"); continue
            label = bool(r.get("label")); flip = bool(r.get("global_swap_changes_label", False))
            out.append(ComparisonExample(
                row_id=str(r.get("row_id")), suite=str(r.get("suite","")),
                split=str(r.get("split","")), arm=arm,
                event1=e1, event2=e2, names=(ns[0], ns[1]), label=label,
                inverted_target=bool(label) ^ bool(flip),
                relation1=r.get("relation1"), relation2=r.get("relation2"),
                metadata={"global_swap_changes_label":flip,
                          "orientation_dependency":r.get("orientation_dependency")}))
        except Exception as e:
            parse_errors.append(f"cmp {r.get('row_id')}: {e}")
    return out


# ── Vocabulary (name-agnostic) ──

class Vocab:
    def __init__(self):
        self.itos = ["<pad>", "<unk>", "<bos>", "<eos>", "<name>", "<hyp>"]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def add(self, toks):
        for t in toks:
            if t not in self.stoi:
                self.stoi[t] = len(self.itos); self.itos.append(t)

    def encode(self, toks):
        return [self.stoi.get(t, 1) for t in toks]


def normalize_event_for_matcher(event_text: str, names: Sequence[str]) -> Tuple[List[str], List[Tuple[int,int,int]]]:
    """Tokenize event text with <name> placeholders at name positions.
    Returns (tokens, name_spans) where name_spans = [(pos, pos+1, name_idx), ...]."""
    toks = TOKEN_RE.findall(event_text)
    out, spans = ["<bos>"], []
    for t in toks:
        if t in names:
            pos = len(out)
            out.append("<name>")
            spans.append((pos, pos + 1, names.index(t)))
        else:
            out.append(t.lower())
    out.append("<eos>")
    return out, spans


def normalize_static_for_matcher(prefix: str, hyp: str, names: Sequence[str]) -> Tuple[List[str], List[Tuple[int,int,int]]]:
    """Tokenize prefix + <hyp> + hypothesis with <name> placeholders."""
    toks = TOKEN_RE.findall(prefix + " <hyp> " + hyp)
    out, spans = ["<bos>"], []
    for t in toks:
        if t in names:
            pos = len(out)
            out.append("<name>")
            spans.append((pos, pos + 1, names.index(t)))
        elif t == "<hyp>":
            out.append("<hyp>")
        else:
            out.append(t.lower())
    out.append("<eos>")
    return out, spans


def collect_vocab(vocab, states, comps):
    for q in states:
        if q.is_changed:
            toks, _ = normalize_event_for_matcher(q.event, q.names)
            vocab.add(toks)
        else:
            for c in q.names:
                toks, _ = normalize_static_for_matcher(q.prefix, q.hypothesis_text_by_name[c], q.names)
                vocab.add(toks)
    for c in comps:
        for ev in [c.event1, c.event2]:
            toks, _ = normalize_event_for_matcher(ev, c.names)
            vocab.add(toks)


# ── Character-level name encoding ──

def name_to_char_ids(name: str) -> List[int]:
    """Convert name to character IDs: a=0, b=1, ..., z=25."""
    return [ord(c) - ord('a') for c in name.lower() if 'a' <= c <= 'z']


# ── SpanMatcher ──

class CharGRUMatcher(nn.Module):
    """Context-free character-level name matcher.
    Encodes names independently via a tiny character GRU, then matches
    query name to event names via scaled dot product + softmax."""

    def __init__(self, n_chars: int = 26, char_dim: int = 16, hidden_dim: int = 16):
        super().__init__()
        self.char_embed = nn.Embedding(n_chars, char_dim)
        self.gru = nn.GRU(char_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def encode_name(self, char_ids: torch.Tensor) -> torch.Tensor:
        if char_ids.numel() == 0:
            return torch.zeros(self.hidden_dim, device=self.char_embed.weight.device)
        x = self.char_embed(char_ids).unsqueeze(0)
        _, h = self.gru(x)
        return h.squeeze(0).squeeze(0)

    def forward(self, query_chars: torch.Tensor,
                event_name_chars: List[torch.Tensor]) -> torch.Tensor:
        """Return softmax match probabilities over event names."""
        q = self.encode_name(query_chars)
        logits = []
        for nc in event_name_chars:
            n = self.encode_name(nc)
            logits.append(torch.dot(q, n))
        if not logits:
            return torch.zeros(0, device=q.device)
        logits_t = torch.stack(logits)
        return torch.softmax(logits_t * 5.0, dim=0)


# ── Model: Match-gated encoder + scalar head ──

class MatchGatedEncoder(nn.Module):
    """GRU encoder with match-gated name embeddings."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden: int,
                 oracle: bool = False, shared_matcher: Optional[CharGRUMatcher] = None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cand_embed = nn.Parameter(torch.randn(emb_dim) * 0.1)
        self.other_embed = nn.Parameter(torch.randn(emb_dim) * 0.1)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.out_dim = 2 * hidden
        self.oracle = oracle
        if not oracle:
            self.matcher = shared_matcher if shared_matcher is not None else CharGRUMatcher()

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor,
                name_spans: List[Tuple[int, int, int]],
                query_chars: torch.Tensor,
                event_chars: List[torch.Tensor],
                oracle_idx: Optional[int] = None) -> torch.Tensor:
        embeds = self.emb(token_ids)  # (1, L, D)
        device = token_ids.device
        n_names = len(set(ni for _, _, ni in name_spans))

        # Compute match probabilities
        if self.oracle:
            match_probs = torch.zeros(n_names, device=device)
            if oracle_idx is not None and oracle_idx < n_names:
                match_probs[oracle_idx] = 1.0
        else:
            match_probs = self.matcher(query_chars, event_chars)

        # Gate name embeddings
        for start, end, name_idx in name_spans:
            p = match_probs[name_idx] if name_idx < len(match_probs) else torch.tensor(0.0, device=device)
            gated = p * self.cand_embed + (1.0 - p) * self.other_embed
            embeds[0, start:end] = gated.unsqueeze(0).expand(end - start, -1)

        y, _ = self.gru(embeds)
        m = mask.unsqueeze(-1).float()
        return (y * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class ScalarHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class MatchScorer(nn.Module):
    def __init__(self, encoder: MatchGatedEncoder, head: ScalarHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def score(self, token_ids, mask, name_spans, query_chars, event_chars, oracle_idx=None):
        h = self.encoder(token_ids, mask, name_spans, query_chars, event_chars, oracle_idx)
        return self.head(h)


def create_paired_models(vocab_size: int, modes: List[str], seed: int,
                         oracle: bool, emb_dim: int = 48, hidden: int = 64):
    """Create models with paired initialization; all modes start from identical weights."""
    torch.manual_seed(seed)
    # Reference components
    ref_encoder = MatchGatedEncoder(vocab_size, emb_dim, hidden, oracle=oracle)
    ref_head = ScalarHead(ref_encoder.out_dim, hidden)
    ref_static_encoder = MatchGatedEncoder(vocab_size, emb_dim, hidden, oracle=oracle)
    ref_static_head = ScalarHead(ref_static_encoder.out_dim, hidden)

    models = {}
    for mode in modes:
        class PM(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.mode = m
                if m == "tied":
                    enc = copy.deepcopy(ref_encoder)
                    hd = copy.deepcopy(ref_head)
                    self.event_state = MatchScorer(enc, hd)
                    self.event_cmp = self.event_state
                elif m == "shared_trunk":
                    enc = copy.deepcopy(ref_encoder)
                    self.event_state = MatchScorer(enc, copy.deepcopy(ref_head))
                    self.event_cmp = MatchScorer(enc, copy.deepcopy(ref_head))
                elif m == "untied":
                    self.event_state = MatchScorer(copy.deepcopy(ref_encoder), copy.deepcopy(ref_head))
                    self.event_cmp = MatchScorer(copy.deepcopy(ref_encoder), copy.deepcopy(ref_head))
                else:
                    raise ValueError(m)
                self.static = MatchScorer(copy.deepcopy(ref_static_encoder), copy.deepcopy(ref_static_head))
        models[mode] = PM(mode)
    return models


# ── Scoring functions ──

def pad_and_mask(seq: List[int], device) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([seq], dtype=torch.long, device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    return ids, mask


def state_scores(model, vocab, q: StateQuery, device, all_name_chars):
    """Score state query for both candidates. Returns (2,) tensor."""
    scores = []
    for cand in q.names:
        if q.is_changed:
            toks, spans = normalize_event_for_matcher(q.event, q.names)
        else:
            toks, spans = normalize_static_for_matcher(q.prefix, q.hypothesis_text_by_name[cand], q.names)
        ids = vocab.encode(toks)
        t_ids, t_mask = pad_and_mask(ids, device)
        q_chars = torch.tensor(name_to_char_ids(cand), dtype=torch.long, device=device)
        e_chars = [torch.tensor(name_to_char_ids(n), dtype=torch.long, device=device) for n in q.names]
        oracle_idx = q.names.index(cand)
        scorer = model.event_state if q.is_changed else model.static
        s = scorer.score(t_ids, t_mask, spans, q_chars, e_chars, oracle_idx)
        scores.append(s.squeeze())
    return torch.stack(scores)


def comparison_prob_same(model, vocab, c: ComparisonExample, device, all_name_chars):
    """Comparison scoring. Returns (prob_same, scores1, scores2)."""
    s1_list, s2_list = [], []
    for cand in c.names:
        toks1, spans1 = normalize_event_for_matcher(c.event1, c.names)
        toks2, spans2 = normalize_event_for_matcher(c.event2, c.names)
        ids1 = vocab.encode(toks1); ids2 = vocab.encode(toks2)
        t1, m1 = pad_and_mask(ids1, device)
        t2, m2 = pad_and_mask(ids2, device)
        q_chars = torch.tensor(name_to_char_ids(cand), dtype=torch.long, device=device)
        e_chars = [torch.tensor(name_to_char_ids(n), dtype=torch.long, device=device) for n in c.names]
        oracle_idx = c.names.index(cand)
        sc1 = model.event_cmp.score(t1, m1, spans1, q_chars, e_chars, oracle_idx).squeeze()
        sc2 = model.event_cmp.score(t2, m2, spans2, q_chars, e_chars, oracle_idx).squeeze()
        s1_list.append(sc1); s2_list.append(sc2)
    s1 = torch.stack(s1_list); s2 = torch.stack(s2_list)
    p1, p2 = F.softmax(s1, dim=0), F.softmax(s2, dim=0)
    psame = (p1 * p2).sum().clamp(1e-6, 1 - 1e-6)
    return psame, s1, s2


def bce_prob(p, target):
    y = torch.tensor(float(target), dtype=torch.float32, device=p.device)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p))


def get_target_idx(q: StateQuery) -> int:
    return 1 if (q.label_by_name[q.names[1]] and not q.label_by_name[q.names[0]]) else 0


# ── Training ──

def train_one(model, vocab, train_states, train_comps, device, bridge_sign,
              epochs, lr, weight_decay, seed, cmp_weight=1.0, state_weight=1.0,
              static_weight=1.0, print_every=0):
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
    rng = random.Random(seed)
    history = []; t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        states = list(train_states); comps = list(train_comps)
        rng.shuffle(states); rng.shuffle(comps)
        opt.zero_grad(set_to_none=True)
        losses = []

        for q in states:
            scores = state_scores(model, vocab, q, device, None)
            target_idx = get_target_idx(q)
            if bridge_sign == -1 and q.is_direct_anchor:
                scores = scores.flip(0)
            loss = F.cross_entropy(scores.view(1, -1), torch.tensor([target_idx], dtype=torch.long, device=device))
            losses.append((state_weight if q.is_changed else static_weight) * loss)

        for c in comps:
            p, _, _ = comparison_prob_same(model, vocab, c, device, None)
            losses.append(cmp_weight * bce_prob(p, c.label))

        if not losses: raise RuntimeError("no losses")
        loss_total = torch.stack(losses).mean()
        loss_total.backward()
        for cg in clip_groups:
            torch.nn.utils.clip_grad_norm_(cg, 5.0)
        opt.step()

        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            with torch.no_grad():
                metrics = quick_train_metrics(model, vocab, train_states, train_comps, device, bridge_sign)
            rec = {"epoch": ep, "loss": float(loss_total.detach().cpu()),
                   **{k: float(v) for k, v in metrics.items()}}
            history.append(rec)
            if print_every: print(json.dumps(rec, sort_keys=True), flush=True)

    return {"elapsed_seconds": time.time() - t0, "history": history}


def quick_train_metrics(model, vocab, states, comps, device, bridge_sign):
    model.eval()
    n_s = c_s = n_ch = c_ch = n_un = c_un = 0
    for q in states:
        scores = state_scores(model, vocab, q, device, None)
        if bridge_sign == -1 and q.is_direct_anchor:
            scores_eff = scores.flip(0)
        else:
            scores_eff = scores
        tgt = get_target_idx(q)
        pred = int(torch.argmax(scores_eff).item())
        n_s += 1; c_s += int(pred == tgt)
        if q.is_changed: n_ch += 1; c_ch += int(pred == tgt)
        else: n_un += 1; c_un += int(pred == tgt)
    n_c = c_c = 0
    for ex in comps:
        p, _, _ = comparison_prob_same(model, vocab, ex, device, None)
        n_c += 1; c_c += int(float(p.detach().cpu()) >= 0.5) == int(ex.label)
    return {
        "train_state_acc": c_s / n_s if n_s else math.nan,
        "train_changed_acc": c_ch / n_ch if n_ch else math.nan,
        "train_unchanged_acc": c_un / n_un if n_un else math.nan,
        "train_cmp_acc": c_c / n_c if n_c else math.nan,
    }


# ── Matching accuracy diagnostic ──

def matching_accuracy(model, vocab, eval_states, eval_comps, device):
    """Report how accurately the SpanMatcher identifies the query candidate."""
    if hasattr(model.event_state.encoder, 'oracle') and model.event_state.encoder.oracle:
        return {"mode": "oracle", "state_match_acc": 1.0, "cmp_match_acc": 1.0}

    model.eval()
    correct_s = total_s = 0
    correct_c = total_c = 0
    with torch.no_grad():
        for q in eval_states:
            for cand in q.names:
                q_chars = torch.tensor(name_to_char_ids(cand), dtype=torch.long, device=device)
                e_chars = [torch.tensor(name_to_char_ids(n), dtype=torch.long, device=device) for n in q.names]
                probs = model.event_state.encoder.matcher(q_chars, e_chars)
                pred_idx = int(torch.argmax(probs).item())
                true_idx = q.names.index(cand)
                total_s += 1; correct_s += int(pred_idx == true_idx)
        for c in eval_comps:
            for cand in c.names:
                q_chars = torch.tensor(name_to_char_ids(cand), dtype=torch.long, device=device)
                e_chars = [torch.tensor(name_to_char_ids(n), dtype=torch.long, device=device) for n in c.names]
                probs = model.event_cmp.encoder.matcher(q_chars, e_chars)
                pred_idx = int(torch.argmax(probs).item())
                true_idx = c.names.index(cand)
                total_c += 1; correct_c += int(pred_idx == true_idx)
    return {
        "mode": "learned",
        "state_match_acc": correct_s / total_s if total_s else math.nan,
        "cmp_match_acc": correct_c / total_c if total_c else math.nan,
        "state_n": total_s, "cmp_n": total_c,
    }


# ── Evaluation with saved predictions ──

def eval_predictions(model, vocab, states, comps, device, condition, arm, seed, bridge_sign):
    model.eval()
    state_rows, comp_rows = [], []
    with torch.no_grad():
        for q in states:
            scores_t = state_scores(model, vocab, q, device, None)
            scores = [float(x) for x in scores_t.detach().cpu().tolist()]
            probs = [float(x) for x in F.softmax(scores_t, dim=0).detach().cpu().tolist()]
            d_e = scores[0] - scores[1]
            for i, cand in enumerate(q.names):
                state_rows.append({
                    "condition": condition, "arm": arm, "seed": seed, "bridge_sign": bridge_sign,
                    "suite": q.suite, "split": q.split, "query_key": q.key,
                    "task": "state_query", "candidate": cand, "candidate_index": i,
                    "score": scores[i], "prob": probs[i], "d_e": d_e,
                    "label_true": bool(q.label_by_name[cand]), "is_changed": q.is_changed,
                    "query_kind": q.query_kind, "relation": q.relation,
                    "relation_family": q.relation_family,
                    "initial_pattern": q.initial_pattern, "static_slot": q.static_slot,
                    "global_swap_changes_label": q.metadata.get("global_swap_changes_label"),
                    "names": list(q.names), "object": q.object_name,
                    "is_direct_anchor": q.is_direct_anchor,
                })
        for c in comps:
            p, s1, s2 = comparison_prob_same(model, vocab, c, device, None)
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
                "relation1": c.relation1, "relation2": c.relation2,
                "d_e1": d_e1, "d_e2": d_e2,
                "names": list(c.names),
            })
    return state_rows, comp_rows


# ── Central readout (adapted from research) ──

def central_eval(state_rows, comp_rows):
    ce = {}
    # State choice accuracy by family and query kind
    by_key = defaultdict(list)
    for r in state_rows:
        by_key[(r["suite"], r["query_key"])].append(r)

    choices = []
    for key, rows in by_key.items():
        if len(rows) != 2: continue
        rows = sorted(rows, key=lambda x: x["candidate_index"])
        pred_i = 0 if rows[0]["score"] >= rows[1]["score"] else 1
        target_i = 0
        for i, r in enumerate(rows):
            if r["label_true"]: target_i = i
        margin = rows[target_i]["score"] - rows[1 - target_i]["score"]
        r0 = rows[0]
        choices.append({
            "correct": pred_i == target_i, "margin": margin,
            "relation_family": r0["relation_family"],
            "query_kind": r0["query_kind"], "suite": r0["suite"],
            "initial_pattern": r0.get("initial_pattern"),
            "d_e": r0["d_e"],
        })

    def acc(items): return sum(x["correct"] for x in items) / len(items) if items else math.nan

    direct_changed = [c for c in choices if c["relation_family"] == "direct_anchor" and c["query_kind"] == "changed"]
    graph_changed = [c for c in choices if c["relation_family"] == "graph_transfer" and c["query_kind"] == "changed"]
    unchanged = [c for c in choices if c["query_kind"] == "unchanged"]
    same_init = [c for c in choices if c.get("initial_pattern") == "same" and c["query_kind"] == "changed"]

    ce["direct_same"] = acc(direct_changed)
    ce["graph_same"] = acc(graph_changed)
    ce["unchanged"] = acc(unchanged)
    ce["same_init_changed"] = acc(same_init)

    # Pair-both for graph changed
    graph_same_init = [c for c in graph_changed if c.get("initial_pattern") == "same"]
    by_suite_key = defaultdict(list)
    for c in graph_same_init:
        by_suite_key[c["suite"]].append(c)
    pair_correct = 0; pair_total = 0
    for sk, items in by_suite_key.items():
        if len(items) >= 2:
            pair_total += 1
            pair_correct += int(all(x["correct"] for x in items))
    ce["pair_both_graph_same"] = pair_correct / pair_total if pair_total else math.nan

    # Graph mean d_e
    graph_des = [c["d_e"] for c in graph_changed]
    ce["graph_mean_de"] = sum(graph_des) / len(graph_des) if graph_des else math.nan
    ce["graph_same_margin"] = sum(c["margin"] for c in graph_changed) / len(graph_changed) if graph_changed else math.nan

    # Comparison metrics
    hh = [r for r in comp_rows if "heldheld" in r.get("suite", "")]
    mixed = [r for r in comp_rows if "mixed" in r.get("suite", "")]

    ce["hh_closure"] = sum(r["correct"] for r in hh) / len(hh) if hh else math.nan
    ce["hh_margin"] = sum(r["signed_margin"] for r in hh) / len(hh) if hh else math.nan
    ce["mixed_acc"] = sum(r["correct"] for r in mixed) / len(mixed) if mixed else math.nan
    ce["mixed_margin"] = sum(r["signed_margin"] for r in mixed) / len(mixed) if mixed else math.nan

    return ce


# ── Fresh-rename evaluation ──

def fresh_rename_state_queries(states, seen_names, rng_seed=42):
    """Generate fresh-renamed eval copies with names unseen during training."""
    rng = random.Random(rng_seed)
    fresh_pool = [f"Xname{i}" for i in range(50)]
    rng.shuffle(fresh_pool)
    out = []
    for q in states:
        old_a, old_b = q.names
        new_a, new_b = fresh_pool.pop(), fresh_pool.pop()
        fresh_pool.extend([new_a, new_b])  # recycle
        new_event = q.event.replace(old_a, new_a).replace(old_b, new_b)
        new_prefix = q.prefix.replace(old_a, new_a).replace(old_b, new_b)
        new_hyp = {new_a if k == old_a else new_b: v.replace(old_a, new_a).replace(old_b, new_b)
                   for k, v in q.hypothesis_text_by_name.items()}
        new_label = {new_a if k == old_a else new_b: v for k, v in q.label_by_name.items()}
        new_inv = {new_a if k == old_a else new_b: v for k, v in q.inverted_target_by_name.items()}
        out.append(StateQuery(
            key=q.key + "_fresh", suite=q.suite + "_fresh", split=q.split, arm=q.arm,
            event=new_event, prefix=new_prefix, hypothesis_text_by_name=new_hyp,
            names=(new_a, new_b), object_name=q.object_name, is_changed=q.is_changed,
            label_by_name=new_label, inverted_target_by_name=new_inv,
            relation=q.relation, relation_family=q.relation_family,
            initial_pattern=q.initial_pattern, static_slot=q.static_slot,
            query_kind=q.query_kind, metadata=q.metadata))
    return out


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--conditions", nargs="+", default=["shared_trunk", "untied"])
    ap.add_argument("--matcher", choices=["oracle", "learned"], default="learned")
    ap.add_argument("--bridge-signs", nargs="+", type=int, default=[1, -1])
    ap.add_argument("--seeds", nargs="+", type=int, default=[29400])
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--emb-dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--print-every", type=int, default=55)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-bridge-changed-only", action="store_true",
                    help="Remove only changed h0/h2 anchors (corrected comparison-only)")
    ap.add_argument("--no-comparisons", action="store_true", help="Remove comparison rows (bridge-only)")
    ap.add_argument("--fresh-rename", action="store_true", help="Also evaluate with fresh-renamed names")
    args = ap.parse_args()

    device = torch.device(args.device)
    data_root = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle = (args.matcher == "oracle")

    print(f"research SpanMatcher gauge probe: matcher={args.matcher} conditions={args.conditions}", flush=True)

    # Load substrate
    arm_dir = data_root / "arms" / args.arm
    eval_dir = data_root / "eval"
    arm_rows = []
    for f in sorted(arm_dir.iterdir()):
        if f.name.endswith(".jsonl"):
            arm_rows.extend(load_jsonl(f))
    eval_rows = []
    for f in sorted(eval_dir.iterdir()):
        if f.name.endswith(".jsonl"):
            eval_rows.extend(load_jsonl(f))

    parse_errors = []
    train_states = build_state_queries(arm_rows, args.arm, parse_errors)
    train_comps = build_comparisons(arm_rows, args.arm, parse_errors)
    eval_states_raw = build_state_queries(eval_rows, "eval", parse_errors)
    eval_comps_raw = build_comparisons(eval_rows, "eval", parse_errors)

    # Data filters
    if args.no_bridge_changed_only:
        before = len(train_states)
        train_states = [q for q in train_states if not q.is_direct_anchor]
        print(f"[narrow bridge filter] removed {before - len(train_states)} changed h0/h2 anchors, "
              f"{len(train_states)} arm states remain", flush=True)

    if args.no_comparisons:
        train_comps = []

    # Build vocabulary from training data only
    vocab = Vocab()
    collect_vocab(vocab, train_states, train_comps)
    print(f"Vocabulary: {len(vocab.itos)} tokens", flush=True)

    # Collect train/eval name sets
    train_names = set()
    for q in train_states: train_names.update(q.names)
    for c in train_comps: train_names.update(c.names)
    eval_names = set()
    for q in eval_states_raw: eval_names.update(q.names)
    for c in eval_comps_raw: eval_names.update(c.names)
    print(f"Train names ({len(train_names)}): {sorted(train_names)}", flush=True)
    print(f"Eval names ({len(eval_names)}): {sorted(eval_names)}", flush=True)
    print(f"Overlap: {sorted(train_names & eval_names)}", flush=True)

    # Run cells
    all_results = []
    for seed in args.seeds:
        # Create paired models
        paired = create_paired_models(len(vocab.itos), args.conditions, seed, oracle,
                                      args.emb_dim, args.hidden)
        # Hash untouched init
        init_hash = hashlib.sha256()
        for k in sorted(paired.keys()):
            for n, p in sorted(paired[k].named_parameters()):
                init_hash.update(p.data.cpu().numpy().tobytes())
        init_hash_str = init_hash.hexdigest()[:16]
        print(f"Paired init hash: {init_hash_str}", flush=True)

        for condition in args.conditions:
            for bs in args.bridge_signs:
                tag = f"{condition}/bs={'+'if bs==1 else '-'}1/seed={seed}"
                print(f"\n=== {args.matcher} {tag} init={init_hash_str} ===", flush=True)

                model = copy.deepcopy(paired[condition])
                train_info = train_one(
                    model, vocab, train_states, train_comps, device, bs,
                    args.epochs, args.lr, args.weight_decay, seed,
                    print_every=args.print_every)

                # Evaluate
                with torch.no_grad():
                    s_rows, c_rows = eval_predictions(
                        model, vocab, eval_states_raw, eval_comps_raw, device,
                        condition, args.arm, seed, bs)
                    ce = central_eval(s_rows, c_rows)
                    match_acc = matching_accuracy(model, vocab, eval_states_raw, eval_comps_raw, device)

                # Fresh rename
                fresh_ce = {}
                if args.fresh_rename:
                    fresh_states = fresh_rename_state_queries(eval_states_raw, train_names)
                    fs_rows, _ = eval_predictions(model, vocab, fresh_states, [], device,
                                                  condition, args.arm, seed, bs)
                    fresh_ce = central_eval(fs_rows, [])

                result = {
                    "condition": condition, "matcher": args.matcher,
                    "bridge_sign": bs, "seed": seed,
                    "init_hash_prefix": init_hash_str,
                    "train_state_acc": train_info["history"][-1].get("train_state_acc") if train_info["history"] else None,
                    "train_cmp_acc": train_info["history"][-1].get("train_cmp_acc") if train_info["history"] else None,
                    "central_eval": ce,
                    "matching_accuracy": match_acc,
                    "fresh_rename_eval": fresh_ce if fresh_ce else None,
                    "elapsed_seconds": train_info["elapsed_seconds"],
                    "epochs": args.epochs,
                }
                all_results.append(result)
                print(json.dumps(result, indent=2, sort_keys=True), flush=True)

                # Save predictions
                pred_dir = out_dir / f"{args.matcher}_{condition}_bs{'+' if bs == 1 else '-'}1_seed{seed}"
                pred_dir.mkdir(parents=True, exist_ok=True)
                write_jsonl(pred_dir / "state_predictions.jsonl", s_rows)
                write_jsonl(pred_dir / "comparison_predictions.jsonl", c_rows)

    # Save summary
    write_json(out_dir / "span_matcher_gauge_summary.json", {
        "all_results": all_results, "parse_errors": parse_errors[:20],
        "vocab_size": len(vocab.itos), "matcher": args.matcher,
        "conditions": args.conditions, "epochs": args.epochs,
    })

    # Write markdown summary
    md_lines = ["# research SpanMatcher gauge probe\n",
                f"Matcher: {args.matcher}, Conditions: {args.conditions}\n",
                f"Epochs: {args.epochs}, Vocab: {len(vocab.itos)}\n\n",
                "| condition | matcher | bs | seed | train_state | train_cmp | graph_same | unchanged | hh_closure | mixed_acc | match_s | match_c |\n",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"]
    for r in all_results:
        ce = r["central_eval"]
        ma = r["matching_accuracy"]
        md_lines.append(f"| {r['condition']} | {r['matcher']} | {r['bridge_sign']:+d} | {r['seed']} "
                        f"| {r.get('train_state_acc',''):.3f} | {r.get('train_cmp_acc',''):.3f} "
                        f"| {ce.get('graph_same',''):.3f} | {ce.get('unchanged',''):.3f} "
                        f"| {ce.get('hh_closure',''):.3f} | {ce.get('mixed_acc',''):.3f} "
                        f"| {ma.get('state_match_acc',''):.3f} | {ma.get('cmp_match_acc',''):.3f} |\n")
    md_path = out_dir / "span_matcher_gauge_summary.md"
    md_path.write_text("".join(md_lines))
    print(json.dumps({
        "status": "SPAN_MATCHER_GAUGE_COMPLETE",
        "summary": project_rel(md_path),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
