#!/usr/bin/env python3
"""research: raw span discovery gauge-transport probe.

Removes supplied entity-span positions (<name> markers and <cand>/<other>).
The model receives raw lowercase text with actual names embedded and must
discover entity positions through context-free character-level matching on
ALL tokens.

Modes:
  oracle  : exact substring matching (hard 0/1 probabilities at true positions)
  learned : CharGRU matcher, trained end-to-end

Architecture: SpanGatedEncoder = CharGRU token matcher + soft gating + shared GRU.
Same bridge_sign, shared_trunk/untied, and evaluation as research.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, re, sys, time
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
DEFAULT_OUT = PROJECT / "data/span_discovery"
DEFAULT_ARM = "aligned_state_bridge"
DIRECT_RELS = {"h0_dax", "h2_norp"}
GRAPH_RELS = {"h1_mep", "h3_ziv"}

TOKEN_RE = re.compile(r"[A-Za-z_]+|[0-9]+|[.,;:?]")
CAP_RE = re.compile(r"\b[A-Z][a-z]+\b")
NOT_NAMES = {
    "During", "Event", "Did", "After", "At", "The", "A", "B", "From", "This",
    "First", "Then", "Before", "Question", "Answer",
}

# ── Shared utilities (from research) ──

def project_rel(p: Path) -> str:
    try: return str(p.relative_to(Path.cwd()))
    except: return str(p)

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists(): return rows
    with path.open() as f:
        for line in f:
            s = line.strip()
            if s: rows.append(json.loads(s))
    return rows

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)+"\n")

def write_jsonl(path: Path, rows: Iterable[Dict[str,Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True)+"\n")

def extract_names(text):
    ns = []
    for m in CAP_RE.findall(text):
        if m in NOT_NAMES: continue
        if m not in ns: ns.append(m)
    return ns

def event_names(t): return extract_names(t)[:2]

def parse_state_event(row):
    ce = str(row.get("cause_event",""))
    if ce: return ce
    prem = str(row.get("premise",""))
    mk = "The event was this:"
    if mk in prem: return prem.split(mk,1)[1].strip()
    raise ValueError(f"no event: {prem[:80]}")

def premise_before_event(row):
    prem = str(row.get("premise",""))
    mk = "The event was this:"
    return prem.split(mk,1)[0].strip() if mk in prem else prem

def parse_hypothesis(row):
    hyp = str(row.get("hypothesis",""))
    m = re.search(r"After the event,\s+([A-Z][a-z]+)\s+had the\s+([A-Za-z_]+)\.?", hyp)
    if not m: raise ValueError(f"no hyp: {hyp}")
    return m.group(1), m.group(2).lower()

def event_object(event_text):
    m = re.search(r"During the\s+([A-Za-z_]+)\s+episode", event_text)
    return m.group(1).lower() if m else None

def rel_family(rel):
    if rel in DIRECT_RELS: return "direct_anchor"
    if rel in GRAPH_RELS: return "graph_transfer"
    return "seen_or_other" if rel else "unknown"

# ── Data structures (from research) ──

@dataclass
class StateQuery:
    key: str; suite: str; split: str; arm: str; event: str; prefix: str
    hypothesis_text_by_name: Dict[str,str]; names: Tuple[str,str]
    object_name: str; is_changed: bool; label_by_name: Dict[str,bool]
    inverted_target_by_name: Dict[str,bool]; relation: Optional[str]
    relation_family: str; initial_pattern: Optional[str]; static_slot: Optional[int]
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
    rid = str(row.get("row_id"))
    return re.sub(r"_(0|1)$", "", rid)

def build_state_queries(rows, arm, pe):
    by = defaultdict(list)
    for r in rows:
        if r.get("task") == "state_query": by[state_query_key(r)].append(r)
    out = []
    for k, grp in by.items():
        if len(grp) < 2: pe.append(f"sg {k} {len(grp)}"); continue
        try:
            ev = parse_state_event(grp[0]); ns = event_names(ev)
            if len(ns) != 2: pe.append(f"sg {k} ns={ns}"); continue
            prefix = premise_before_event(grp[0])
            hyp_n, lab_n, inv_n = {}, {}, {}; obj = None
            for r in grp:
                c, oi = parse_hypothesis(r); hyp_n[c] = str(r.get("hypothesis",""))
                l = bool(r.get("label")); f = bool(r.get("global_swap_changes_label",False))
                lab_n[c] = l; inv_n[c] = l ^ f; obj = oi
            if not all(n in lab_n for n in ns): pe.append(f"sg {k} miss"); continue
            eo = event_object(ev); ic = (obj == eo); r0 = grp[0]
            out.append(StateQuery(key=k,suite=str(r0.get("suite","")),split=str(r0.get("split","")),
                arm=arm,event=ev,prefix=prefix,hypothesis_text_by_name=hyp_n,names=(ns[0],ns[1]),
                object_name=str(obj),is_changed=ic,label_by_name=lab_n,inverted_target_by_name=inv_n,
                relation=r0.get("relation") or r0.get("cause_relation"),
                relation_family=rel_family(r0.get("relation") or r0.get("cause_relation")),
                initial_pattern=r0.get("initial_pattern"),static_slot=r0.get("static_slot"),
                query_kind="changed" if ic else "unchanged",
                metadata={"pair_id":r0.get("pair_id"),"row_ids":[rr.get("row_id") for rr in grp],
                          "global_swap_changes_label":bool(r0.get("global_swap_changes_label",False)),
                          "raw_query_kind":r0.get("query_kind")}))
        except Exception as e: pe.append(f"sg {k}: {e}")
    return out

def build_comparisons(rows, arm, pe):
    out = []
    for r in rows:
        if r.get("task") != "relation_comparison": continue
        try:
            e1, e2 = str(r.get("event1","")), str(r.get("event2",""))
            if not e1 or not e2:
                text = str(r.get("text",""))
                m = re.search(r"Event A:\s*(.*?)\s*Event B:\s*(.*?)\s*Did Event A", text)
                if m: e1, e2 = m.group(1).strip(), m.group(2).strip()
            ns = []
            for n in event_names(e1)+event_names(e2):
                if n not in ns: ns.append(n)
            if len(ns)!=2: pe.append(f"cmp {r.get('row_id')} ns={ns}"); continue
            l=bool(r.get("label")); f=bool(r.get("global_swap_changes_label",False))
            out.append(ComparisonExample(row_id=str(r.get("row_id")),suite=str(r.get("suite","")),
                split=str(r.get("split","")),arm=arm,event1=e1,event2=e2,names=(ns[0],ns[1]),
                label=l,inverted_target=l^f,relation1=r.get("relation1"),relation2=r.get("relation2"),
                metadata={"global_swap_changes_label":f,"orientation_dependency":r.get("orientation_dependency")}))
        except Exception as e: pe.append(f"cmp {r.get('row_id')}: {e}")
    return out

# ── Raw tokenization (no <cand>/<other> substitution) ──

def raw_tokenize(text: str) -> Tuple[List[str], List[str]]:
    """Tokenize text keeping original case forms for character matching.
    Returns (token_list, char_form_list) both with <bos>/<eos> wrappers."""
    toks = TOKEN_RE.findall(text)
    ids = ["<bos>"] + [t.lower() for t in toks] + ["<eos>"]
    forms = [""] + [t.lower() for t in toks] + [""]  # original lowercase for char matching
    return ids, forms

def raw_tokenize_static(prefix: str, hyp: str) -> Tuple[List[str], List[str]]:
    combined = prefix + " <hyp> " + hyp
    toks = TOKEN_RE.findall(combined)
    ids = ["<bos>"] + [t.lower() for t in toks] + ["<eos>"]
    forms = [""] + [t.lower() for t in toks] + [""]
    return ids, forms

# ── Vocabulary (no <cand>/<other>) ──

class RawVocab:
    def __init__(self):
        self.itos = ["<pad>", "<unk>", "<bos>", "<eos>", "<hyp>"]
        self.stoi = {t:i for i,t in enumerate(self.itos)}
    def add(self, toks):
        for t in toks:
            if t not in self.stoi:
                self.stoi[t] = len(self.itos); self.itos.append(t)
    def encode(self, toks): return [self.stoi.get(t,1) for t in toks]

def collect_raw_vocab(vocab, states, comps, exclude_names=None):
    skip = {n.lower() for n in (exclude_names or [])}
    def filtered_add(ids):
        vocab.add([t for t in ids if t not in skip])
    for q in states:
        if q.is_changed:
            ids, _ = raw_tokenize(q.event); filtered_add(ids)
        else:
            ids, _ = raw_tokenize_static(q.prefix, q.hypothesis_text_by_name[q.names[0]])
            filtered_add(ids)
            ids2, _ = raw_tokenize_static(q.prefix, q.hypothesis_text_by_name[q.names[1]])
            filtered_add(ids2)
    for c in comps:
        ids1, _ = raw_tokenize(c.event1); filtered_add(ids1)
        ids2, _ = raw_tokenize(c.event2); filtered_add(ids2)

# ── Character encoding ──

def name_to_char_ids(name: str, max_len: int = 20) -> List[int]:
    """Convert lowercase name to character IDs (a=1..z=26, other=27)."""
    ids = []
    for ch in name.lower()[:max_len]:
        if 'a' <= ch <= 'z': ids.append(ord(ch) - ord('a') + 1)
        else: ids.append(27)
    return ids if ids else [0]

def pad_char_batch(char_seqs: Sequence[Sequence[int]], device: torch.device):
    ml = max(len(s) for s in char_seqs) if char_seqs else 1
    arr = torch.zeros((len(char_seqs), ml), dtype=torch.long, device=device)
    for i, s in enumerate(char_seqs):
        arr[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return arr

# ── Character-level encoder ──

class CharGRUEncoder(nn.Module):
    def __init__(self, n_chars: int = 28, char_dim: int = 16, hidden: int = 16):
        super().__init__()
        self.emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.gru = nn.GRU(char_dim, hidden, batch_first=True)
    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(char_ids)
        _, h = self.gru(x)
        return h.squeeze(0)  # (batch, hidden)

# ── Span-gated trunk encoder ──

class SpanGatedTrunk(nn.Module):
    """Word embeddings + character-matched soft gating → GRU.
    
    For each token, computes character match against candidate/other names.
    Name tokens get soft candidate/other embeddings; non-name tokens keep
    their word embeddings. Same mean-pooled GRU output as research TrunkEncoder.
    """
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int,
                 char_dim: int = 16, char_hidden: int = 16):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cand_param = nn.Parameter(torch.randn(emb_dim) * 0.02)
        self.other_param = nn.Parameter(torch.randn(emb_dim) * 0.02)
        self.char_enc = CharGRUEncoder(28, char_dim, char_hidden)
        self.neither_bias = nn.Parameter(torch.tensor(2.0))  # bias toward "neither" initially
        self.gru = nn.GRU(emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.out_dim = 2 * hidden

    def forward_gated(self, token_ids: torch.Tensor, mask: torch.Tensor,
                      token_char_batch: torch.Tensor, cand_char: torch.Tensor,
                      other_char: torch.Tensor, is_special: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (1, seq_len) - vocabulary IDs
        mask: (1, seq_len) - bool mask
        token_char_batch: (seq_len, max_chars) - char IDs for each token
        cand_char: (1, max_chars) - char IDs for candidate name
        other_char: (1, max_chars) - char IDs for other name
        is_special: (seq_len,) - bool mask for <bos>/<eos>/<hyp> (force neither)
        Returns: (1, out_dim) mean-pooled GRU output
        """
        word_x = self.word_emb(token_ids)  # (1, seq_len, emb_dim)
        sl = token_ids.size(1)
        
        # Character matching
        tok_vecs = self.char_enc(token_char_batch)  # (seq_len, char_hidden)
        cand_vec = self.char_enc(cand_char)          # (1, char_hidden)
        other_vec = self.char_enc(other_char)        # (1, char_hidden)
        
        sc = (tok_vecs * cand_vec).sum(-1)    # (seq_len,)
        so = (tok_vecs * other_vec).sum(-1)   # (seq_len,)
        sn = self.neither_bias.expand(sl)
        
        logits = torch.stack([sc, so, sn], dim=-1)  # (seq_len, 3)
        # Force special tokens to neither
        logits[is_special, 0] = -1e6
        logits[is_special, 1] = -1e6
        
        probs = F.softmax(logits, dim=-1)  # (seq_len, 3)
        
        p_c = probs[:, 0:1]  # (seq_len, 1)
        p_o = probs[:, 1:2]
        p_n = probs[:, 2:3]
        
        gated = (p_n * word_x[0] +
                 p_c * self.cand_param.unsqueeze(0) +
                 p_o * self.other_param.unsqueeze(0))
        gated = gated.unsqueeze(0)  # (1, seq_len, emb_dim)
        
        y, _ = self.gru(gated)
        m = mask.unsqueeze(-1).float()
        return (y * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0), probs

class ScalarHead(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, h): return self.net(h).squeeze(-1)

class SpanGatedScorer(nn.Module):
    def __init__(self, trunk: SpanGatedTrunk, head: ScalarHead):
        super().__init__()
        self.trunk = trunk; self.head = head
    def score_gated(self, token_ids, mask, tcb, cc, oc, isp):
        h, probs = self.trunk.forward_gated(token_ids, mask, tcb, cc, oc, isp)
        return self.head(h), probs

# ── Model construction ──

def make_span_model(vocab_size, mode, emb_dim=48, hidden=64, char_dim=16, char_hidden=16):
    ref_trunk = SpanGatedTrunk(vocab_size, emb_dim, hidden, char_dim, char_hidden)
    ref_head = ScalarHead(ref_trunk.out_dim, hidden)
    ref_static_trunk = SpanGatedTrunk(vocab_size, emb_dim, hidden, char_dim, char_hidden)
    ref_static_head = ScalarHead(ref_static_trunk.out_dim, hidden)
    
    class SpanModel(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.mode = m
            if m == "tied":
                t = copy.deepcopy(ref_trunk); h = copy.deepcopy(ref_head)
                self.event_state = SpanGatedScorer(t, h)
                self.event_cmp = self.event_state
            elif m == "shared_trunk":
                t = copy.deepcopy(ref_trunk)
                self.event_state = SpanGatedScorer(t, copy.deepcopy(ref_head))
                self.event_cmp = SpanGatedScorer(t, copy.deepcopy(ref_head))
            elif m == "untied":
                self.event_state = SpanGatedScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head))
                self.event_cmp = SpanGatedScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head))
            else: raise ValueError(m)
            self.static = SpanGatedScorer(copy.deepcopy(ref_static_trunk), copy.deepcopy(ref_static_head))
    return SpanModel(mode)

def create_paired_span_models(vocab_size, modes, seed, emb_dim=48, hidden=64, char_dim=16, char_hidden=16):
    torch.manual_seed(seed)
    ref_trunk = SpanGatedTrunk(vocab_size, emb_dim, hidden, char_dim, char_hidden)
    ref_head = ScalarHead(ref_trunk.out_dim, hidden)
    ref_static_trunk = SpanGatedTrunk(vocab_size, emb_dim, hidden, char_dim, char_hidden)
    ref_static_head = ScalarHead(ref_static_trunk.out_dim, hidden)
    
    models = {}
    for mode in modes:
        class PM(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.mode = m
                if m == "tied":
                    t = copy.deepcopy(ref_trunk); h = copy.deepcopy(ref_head)
                    self.event_state = SpanGatedScorer(t, h)
                    self.event_cmp = self.event_state
                elif m == "shared_trunk":
                    t = copy.deepcopy(ref_trunk)
                    self.event_state = SpanGatedScorer(t, copy.deepcopy(ref_head))
                    self.event_cmp = SpanGatedScorer(t, copy.deepcopy(ref_head))
                elif m == "untied":
                    self.event_state = SpanGatedScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head))
                    self.event_cmp = SpanGatedScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head))
                else: raise ValueError(m)
                self.static = SpanGatedScorer(copy.deepcopy(ref_static_trunk), copy.deepcopy(ref_static_head))
        models[mode] = PM(mode)
    return models

# ── Scoring with span gating ──

def prepare_span_inputs(token_ids_list: List[int], char_forms: List[str],
                        cand_name: str, other_name: str, device: torch.device,
                        matcher_mode: str = "learned"):
    """Prepare tensors for SpanGatedTrunk.forward_gated."""
    sl = len(token_ids_list)
    ids_t = torch.tensor([token_ids_list], dtype=torch.long, device=device)
    mask = torch.ones((1, sl), dtype=torch.bool, device=device)
    
    # Character IDs for each token
    tok_chars = [name_to_char_ids(f) if f else [0] for f in char_forms]
    tcb = pad_char_batch(tok_chars, device)  # (sl, max_chars)
    
    # Candidate and other char IDs
    cc = pad_char_batch([name_to_char_ids(cand_name)], device)   # (1, max_chars)
    oc = pad_char_batch([name_to_char_ids(other_name)], device)  # (1, max_chars)
    
    # Special token mask (<bos>, <eos>, <hyp> get forced to "neither")
    isp = torch.zeros(sl, dtype=torch.bool, device=device)
    isp[0] = True   # <bos>
    isp[-1] = True   # <eos>
    # Also mark <hyp> positions
    for i, f in enumerate(char_forms):
        if f == "" and i > 0 and i < sl - 1:
            isp[i] = True  # empty char form = special token
    
    if matcher_mode == "oracle":
        # Force exact matching: override char tensors to produce hard 0/1
        # We do this by providing matching char sequences for name tokens
        pass  # oracle mode handled at scoring level
    
    return ids_t, mask, tcb, cc, oc, isp

def hard_oracle_score_gated(scorer: SpanGatedScorer, token_ids_list: List[int],
                            char_forms: List[str], cand: str, other: str,
                            device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact raw-token localization ceiling.

    For matcher_mode='oracle', do not pass through the learned CharGRU matcher.
    Candidate-name tokens receive the learned candidate embedding, other-name
    tokens receive the learned other embedding, and all non-name tokens retain
    their ordinary word embedding. This is the hard-canonicalized raw-substring
    control corresponding to research's supplied <cand>/<other> representation.
    """
    trunk = scorer.trunk
    sl = len(token_ids_list)
    ids_t = torch.tensor([token_ids_list], dtype=torch.long, device=device)
    mask = torch.ones((1, sl), dtype=torch.bool, device=device)
    word_x = trunk.word_emb(ids_t)[0]
    probs = torch.zeros((sl, 3), dtype=word_x.dtype, device=device)
    cand_lc = cand.lower(); other_lc = other.lower()
    for i, f in enumerate(char_forms):
        if f == cand_lc:
            probs[i, 0] = 1.0
        elif f == other_lc:
            probs[i, 1] = 1.0
        else:
            probs[i, 2] = 1.0
    gated = (probs[:, 2:3] * word_x +
             probs[:, 0:1] * trunk.cand_param.unsqueeze(0) +
             probs[:, 1:2] * trunk.other_param.unsqueeze(0))
    y, _ = trunk.gru(gated.unsqueeze(0))
    m = mask.unsqueeze(-1).float()
    h = (y * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    return scorer.head(h), probs


def score_event_span(scorer: SpanGatedScorer, vocab: 'RawVocab',
                     event_text: str, names: Tuple[str,str], device: torch.device,
                     matcher_mode: str = "learned") -> Tuple[torch.Tensor, Dict]:
    """Score one event for both candidate assignments. Returns (scores[2], match_info)."""
    tok_ids_str, char_forms = raw_tokenize(event_text)
    token_ids = vocab.encode(tok_ids_str)
    
    scores = []
    match_info_list = []
    for ci, cand in enumerate(names):
        other = names[1-ci]
        if matcher_mode == "oracle":
            s, probs = hard_oracle_score_gated(scorer, token_ids, char_forms, cand, other, device)
        else:
            ids_t, mask, tcb, cc, oc, isp = prepare_span_inputs(
                token_ids, char_forms, cand, other, device, matcher_mode)
            s, probs = scorer.score_gated(ids_t, mask, tcb, cc, oc, isp)
        scores.append(s.squeeze())
        
        # Record matching diagnostics
        p_np = probs.detach().cpu().numpy()
        cand_lc = cand.lower()
        other_lc = other.lower()
        match_rec = {"cand_name": cand, "other_name": other}
        for pi, f in enumerate(char_forms):
            if f == cand_lc:
                match_rec["cand_pos"] = pi
                match_rec["cand_p_cand"] = float(p_np[pi, 0])
                match_rec["cand_p_other"] = float(p_np[pi, 1])
                match_rec["cand_p_neither"] = float(p_np[pi, 2])
            elif f == other_lc:
                match_rec["other_pos"] = pi
                match_rec["other_p_cand"] = float(p_np[pi, 0])
                match_rec["other_p_other"] = float(p_np[pi, 1])
                match_rec["other_p_neither"] = float(p_np[pi, 2])
        match_info_list.append(match_rec)
    
    return torch.stack(scores), match_info_list


def score_static_span(scorer: SpanGatedScorer, vocab: 'RawVocab',
                      prefix: str, hyp: str, names: Tuple[str,str],
                      cand: str, device: torch.device,
                      matcher_mode: str = "learned") -> torch.Tensor:
    tok_ids_str, char_forms = raw_tokenize_static(prefix, hyp)
    token_ids = vocab.encode(tok_ids_str)
    other = names[1] if cand == names[0] else names[0]
    if matcher_mode == "oracle":
        s, _ = hard_oracle_score_gated(scorer, token_ids, char_forms, cand, other, device)
    else:
        ids_t, mask, tcb, cc, oc, isp = prepare_span_inputs(
            token_ids, char_forms, cand, other, device, matcher_mode)
        s, _ = scorer.score_gated(ids_t, mask, tcb, cc, oc, isp)
    return s.squeeze()

def state_scores_span(model, vocab, q, device, matcher_mode="learned"):
    if q.is_changed:
        return score_event_span(model.event_state, vocab, q.event, q.names, device, matcher_mode)[0]
    scores = []
    for c in q.names:
        s = score_static_span(model.static, vocab, q.prefix,
                              q.hypothesis_text_by_name[c], q.names, c, device, matcher_mode)
        scores.append(s)
    return torch.stack(scores)

def comparison_prob_same_span(model, vocab, c, device, matcher_mode="learned"):
    s1, _ = score_event_span(model.event_cmp, vocab, c.event1, c.names, device, matcher_mode)
    s2, _ = score_event_span(model.event_cmp, vocab, c.event2, c.names, device, matcher_mode)
    p1, p2 = F.softmax(s1, dim=0), F.softmax(s2, dim=0)
    psame = (p1 * p2).sum().clamp(1e-6, 1-1e-6)
    return psame, s1, s2

def bce_prob(p, target):
    y = torch.tensor(float(target), dtype=torch.float32, device=p.device)
    return -(y*torch.log(p) + (1-y)*torch.log(1-p))

# ── Training ──

def get_target_idx(q):
    return 1 if (q.label_by_name[q.names[1]] and not q.label_by_name[q.names[0]]) else 0

def train_one_span(model, vocab, train_states, train_comps, device, bridge_sign, epochs, lr,
                   weight_decay, seed, cmp_w, state_w, static_w, print_every=0, matcher_mode="learned"):
    model.to(device)
    seen = set(); all_params = []; clip_groups = []
    def add_mp(mod, _):
        gp = []
        for p in mod.parameters():
            pid = id(p)
            if pid not in seen: seen.add(pid); all_params.append(p); gp.append(p)
        if gp: clip_groups.append(gp)
    add_mp(model.event_state, "es")
    if model.event_cmp is not model.event_state: add_mp(model.event_cmp, "ec")
    add_mp(model.static, "st")
    
    opt = torch.optim.AdamW([{"params": all_params}], lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)
    history = []; t0 = time.time()
    
    for ep in range(1, epochs+1):
        model.train()
        states = list(train_states); comps = list(train_comps)
        rng.shuffle(states); rng.shuffle(comps)
        opt.zero_grad(set_to_none=True)
        losses = []
        
        for q in states:
            scores = state_scores_span(model, vocab, q, device, matcher_mode)
            tgt = get_target_idx(q)
            if bridge_sign == -1 and q.is_direct_anchor:
                scores = scores.flip(0)
            loss = F.cross_entropy(scores.view(1,-1), torch.tensor([tgt], dtype=torch.long, device=device))
            losses.append((state_w if q.is_changed else static_w) * loss)
        
        for c in comps:
            p, _, _ = comparison_prob_same_span(model, vocab, c, device, matcher_mode)
            losses.append(cmp_w * bce_prob(p, c.label))
        
        if not losses: raise RuntimeError("no losses")
        lt = torch.stack(losses).mean()
        lt.backward()
        for cg in clip_groups: torch.nn.utils.clip_grad_norm_(cg, 5.0)
        opt.step()
        
        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            with torch.no_grad():
                met = quick_train_span(model, vocab, train_states, train_comps, device, bridge_sign, matcher_mode)
            rec = {"epoch": ep, "loss": float(lt.detach().cpu()), **{k:float(v) for k,v in met.items()}}
            history.append(rec)
            if print_every: print(json.dumps(rec, sort_keys=True), flush=True)
    
    return {"elapsed_seconds": time.time()-t0, "history": history}

def quick_train_span(model, vocab, states, comps, device, bs, mm):
    model.eval()
    ns=cs=nc_=cc_=nch=cch=nun=cun=0
    for q in states:
        scores = state_scores_span(model, vocab, q, device, mm)
        if bs == -1 and q.is_direct_anchor: se = scores.flip(0)
        else: se = scores
        tgt = get_target_idx(q); pred = int(torch.argmax(se).item())
        ns+=1; cs+=int(pred==tgt)
        if q.is_changed: nch+=1; cch+=int(pred==tgt)
        else: nun+=1; cun+=int(pred==tgt)
    for ex in comps:
        p,_,_ = comparison_prob_same_span(model, vocab, ex, device, mm)
        nc_+=1; cc_+=int(float(p.detach().cpu())>=0.5)==int(ex.label)
    return {"train_state_acc":cs/ns if ns else math.nan, "train_changed_acc":cch/nch if nch else math.nan,
            "train_unchanged_acc":cun/nun if nun else math.nan, "train_cmp_acc":cc_/nc_ if nc_ else math.nan}

# ── Evaluation (replicates research output format) ──

def eval_predictions_span(model, vocab, states, comps, device, condition, arm, seed, bridge_sign, mm):
    model.eval(); state_rows=[]; comp_rows=[]
    with torch.no_grad():
        for q in states:
            scores_t = state_scores_span(model, vocab, q, device, mm)
            scores = [float(x) for x in scores_t.detach().cpu().tolist()]
            probs = [float(x) for x in F.softmax(scores_t,dim=0).detach().cpu().tolist()]
            d_e = scores[0]-scores[1]
            
            # Matching diagnostics for changed events
            match_diag = None
            if q.is_changed:
                _, mi = score_event_span(model.event_state, vocab, q.event, q.names, device, mm)
                match_diag = mi
            
            for i, cand in enumerate(q.names):
                lt = bool(q.label_by_name[cand])
                row = {"condition":condition,"arm":arm,"seed":seed,"bridge_sign":bridge_sign,
                    "suite":q.suite,"split":q.split,"query_key":q.key,"task":"state_query",
                    "candidate":cand,"candidate_index":i,"score":scores[i],"prob":probs[i],"d_e":d_e,
                    "label_true":lt,"is_changed":q.is_changed,"query_kind":q.query_kind,
                    "raw_query_kind":q.metadata.get("raw_query_kind"),
                    "relation":q.relation,"relation_family":q.relation_family,
                    "initial_pattern":q.initial_pattern,"static_slot":q.static_slot,
                    "global_swap_changes_label":q.metadata.get("global_swap_changes_label"),
                    "names":list(q.names),"object":q.object_name,
                    "is_direct_anchor":q.is_direct_anchor}
                if match_diag and i < len(match_diag):
                    row["matching"] = match_diag[i]
                state_rows.append(row)
        
        for c in comps:
            p, s1, s2 = comparison_prob_same_span(model, vocab, c, device, mm)
            pf = float(p.detach().cpu())
            logit = math.log(pf/(1-pf)) if 0<pf<1 else 0.0
            de1 = float(s1[0])-float(s1[1])
            de2 = float(s2[0])-float(s2[1])
            comp_rows.append({"condition":condition,"arm":arm,"seed":seed,"bridge_sign":bridge_sign,
                "suite":c.suite,"split":c.split,"row_id":c.row_id,"task":"relation_comparison",
                "prob_same":pf,"logit_same":logit,"label":bool(c.label),
                "pred":bool(pf>=0.5),"correct":bool(pf>=0.5)==bool(c.label),
                "signed_margin":logit if bool(c.label) else -logit,
                "relation1":c.relation1,"relation2":c.relation2,
                "d_e1":de1,"d_e2":de2,
                "event1_scores":[float(x) for x in s1.detach().cpu().tolist()],
                "event2_scores":[float(x) for x in s2.detach().cpu().tolist()],
                "names":list(c.names),
                "global_swap_changes_label":c.metadata.get("global_swap_changes_label"),
                "orientation_dependency":c.metadata.get("orientation_dependency")})
    return state_rows, comp_rows

# ── Summary (from research) ──

def mean(xs): return float(sum(xs)/len(xs)) if xs else None

def state_choice_records(state_rows):
    by = defaultdict(list)
    for r in state_rows:
        by[(r["condition"],r["arm"],r["seed"],r["bridge_sign"],r["suite"],r["query_key"])].append(r)
    out = []
    for key, rows in by.items():
        if len(rows) != 2: continue
        rows = sorted(rows, key=lambda x: int(x["candidate_index"]))
        pi = 0 if rows[0]["score"]>=rows[1]["score"] else 1
        ti = 0
        for i,r in enumerate(rows):
            if bool(r["label_true"]): ti = i
        margin = rows[ti]["score"]-rows[1-ti]["score"]
        r0 = rows[0]
        out.append({"condition":r0["condition"],"arm":r0["arm"],"seed":r0["seed"],
            "bridge_sign":r0["bridge_sign"],"suite":r0["suite"],"query_key":r0["query_key"],
            "correct":int(pi==ti),"margin":float(margin),"d_e":float(r0["d_e"]),
            "is_changed":bool(r0["is_changed"]),"query_kind":r0["query_kind"],
            "relation":r0["relation"],"relation_family":r0["relation_family"],
            "initial_pattern":r0["initial_pattern"],"static_slot":r0["static_slot"],
            "is_direct_anchor":bool(r0.get("is_direct_anchor"))})
    return out

def pair_both_records(choices):
    by_base = defaultdict(dict)
    for r in choices:
        if r.get("suite") not in {"paired_state_conservation","cross_template_state_readout"}: continue
        qk = "changed" if r.get("is_changed") else "unchanged"
        parts = str(r["query_key"]).split("|")
        base = "|".join(parts[:-1]) if len(parts)>=3 else re.sub(r"_(changed|unchanged)$","",str(r["query_key"]))
        by_base[(r["condition"],r["arm"],r["seed"],r["bridge_sign"],r["suite"],base)][qk] = r
    out = []
    for key, d in by_base.items():
        if "changed" not in d or "unchanged" not in d: continue
        ch, un = d["changed"], d["unchanged"]
        out.append({"condition":key[0],"arm":key[1],"seed":key[2],"bridge_sign":key[3],
            "suite":key[4],"base_key":key[5],"both_correct":int(ch["correct"] and un["correct"]),
            "changed_correct":int(ch["correct"]),"unchanged_correct":int(un["correct"]),
            "changed_margin":ch["margin"],"unchanged_margin":un["margin"],
            "relation":ch["relation"],"relation_family":ch["relation_family"],
            "initial_pattern":ch["initial_pattern"]})
    return out

def compute_central(choices, boths, comp_rows, condition, arm, seed, bridge_sign):
    def fc(**kw):
        xs = [r for r in choices if r["condition"]==condition and r["arm"]==arm
              and r["seed"]==seed and r["bridge_sign"]==bridge_sign]
        for k,v in kw.items(): xs=[r for r in xs if r.get(k)==v]
        return xs
    def fb(**kw):
        xs = [r for r in boths if r["condition"]==condition and r["arm"]==arm
              and r["seed"]==seed and r["bridge_sign"]==bridge_sign]
        for k,v in kw.items(): xs=[r for r in xs if r.get(k)==v]
        return xs
    def acc(xs): return mean([float(r["correct"]) for r in xs])
    def ba(xs): return mean([float(r["both_correct"]) for r in xs])
    def mm(xs): return mean([float(r["margin"]) for r in xs])
    def mde(xs): return mean([float(r["d_e"]) for r in xs])
    
    c = {
        "direct_same": acc(fc(suite="paired_state_conservation",is_changed=True,relation_family="direct_anchor",initial_pattern="same")),
        "graph_same": acc(fc(suite="paired_state_conservation",is_changed=True,relation_family="graph_transfer",initial_pattern="same")),
        "unchanged": acc(fc(suite="paired_state_conservation",is_changed=False)),
        "pair_both_graph_same": ba(fb(suite="paired_state_conservation",relation_family="graph_transfer",initial_pattern="same")),
        "graph_same_margin": mm(fc(suite="paired_state_conservation",is_changed=True,relation_family="graph_transfer",initial_pattern="same")),
        "graph_mean_de": mde(fc(suite="paired_state_conservation",is_changed=True,relation_family="graph_transfer")),
        "same_init_changed": acc(fc(suite="paired_state_conservation",is_changed=True,initial_pattern="same")),
    }
    for suite in ["heldheld_unseen_edge_closure", "mixed_held_seen_orientation"]:
        cs = [r for r in comp_rows if r["condition"]==condition and r["arm"]==arm
              and r["seed"]==seed and r["bridge_sign"]==bridge_sign and r["suite"]==suite]
        name = "hh_closure" if "heldheld" in suite else "mixed_acc"
        c[name] = mean([float(r["correct"]) for r in cs])
        c[name+"_margin"] = mean([float(r["signed_margin"]) for r in cs])
    return c

# ── Matching accuracy diagnostic ──

def matching_accuracy_from_state_rows(state_rows):
    """How often does the span matcher correctly identify candidate/other?"""
    n_events = correct_cand = correct_other = 0
    for r in state_rows:
        m = r.get("matching")
        if not m or r.get("candidate_index") != 0: continue  # avoid double counting
        n_events += 1
        # Check if cand position got highest cand probability
        if m.get("cand_p_cand", 0) > 0.5: correct_cand += 1
        if m.get("other_p_other", 0) > 0.5: correct_other += 1
    return {"n_events": n_events,
            "cand_match_acc": correct_cand/n_events if n_events else 0,
            "other_match_acc": correct_other/n_events if n_events else 0}

# ── Data loading ──

def load_dataset(data_root, arm, no_cmp=False, no_anchor=False):
    pe = []
    common = load_jsonl(data_root/"common_seen_train.jsonl")
    sup = load_jsonl(data_root/"arms"/arm/"train_supervised.jsonl")
    if no_cmp: sup=[r for r in sup if r.get("task")!="relation_comparison"]
    if no_anchor: sup=[r for r in sup if not (r.get("task")=="state_query")]
    train_rows = common+sup
    eval_rows = []
    for p in sorted((data_root/"eval").glob("*.jsonl")):
        if p.name=="name_permutation_counterfactual.jsonl": continue
        eval_rows.extend(load_jsonl(p))
    ts=build_state_queries(train_rows,arm,pe); tc=build_comparisons(train_rows,arm,pe)
    es=build_state_queries(eval_rows,arm,pe);  ec=build_comparisons(eval_rows,arm,pe)
    counts={"train_state":len(ts),"train_cmp":len(tc),"eval_state":len(es),"eval_cmp":len(ec),"errors":len(pe)}
    return ts,tc,es,ec,pe,counts

# ── Run one cell ──

def run_one(data_root, out, condition, arm, seed, bridge_sign, epochs, lr, wd,
            emb_dim, hidden, char_dim, char_hidden, device, cmp_w, state_w, static_w,
            print_every, matcher_mode, paired_models=None, no_name_vocab=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type=="cuda": torch.cuda.manual_seed_all(seed)
    
    ts,tc,es,ec,pe,counts = load_dataset(data_root, arm)
    vocab = RawVocab()
    if no_name_vocab:
        # Collect all train names. The same exclusion is used in main before paired
        # model construction; keeping it explicit here makes run_one independent
        # and prevents train-name word embeddings from becoming a shortcut.
        atn = set()
        for q in ts:
            for n in q.names: atn.add(n)
        for c in tc:
            for n in c.names: atn.add(n)
        collect_raw_vocab(vocab, ts, tc, exclude_names=atn)
    else:
        collect_raw_vocab(vocab, ts, tc)
    
    if paired_models and condition in paired_models:
        model = copy.deepcopy(paired_models[condition])
    else:
        model = make_span_model(len(vocab.itos), condition, emb_dim, hidden, char_dim, char_hidden)
    
    # Save init hash
    init_hash = hashlib.sha256()
    for p in model.parameters():
        init_hash.update(p.detach().cpu().numpy().tobytes())
    ihp = init_hash.hexdigest()[:16]
    
    info = train_one_span(model, vocab, ts, tc, device, bridge_sign, epochs, lr, wd,
                          seed, cmp_w, state_w, static_w, print_every, matcher_mode)
    
    with torch.no_grad():
        ft = quick_train_span(model, vocab, ts, tc, device, bridge_sign, matcher_mode)
    
    train_sr, train_cr = eval_predictions_span(model, vocab, ts, tc, device, condition, arm, seed, bridge_sign, matcher_mode)
    eval_sr, eval_cr = eval_predictions_span(model, vocab, es, ec, device, condition, arm, seed, bridge_sign, matcher_mode)
    
    # Matching diagnostics on eval
    eval_match = matching_accuracy_from_state_rows(eval_sr)
    train_match = matching_accuracy_from_state_rows(train_sr)
    
    choices = state_choice_records(eval_sr)
    boths = pair_both_records(choices)
    central = compute_central(choices, boths, eval_cr, condition, arm, seed, bridge_sign)
    
    tag = f"{'oracle' if matcher_mode=='oracle' else 'learned'}_{condition}_bs{'+' if bridge_sign==1 else '-'}1_seed{seed}"
    rd = out/tag
    
    rec = {"condition":condition,"arm":arm,"seed":seed,"bridge_sign":bridge_sign,
           "matcher":matcher_mode,"epochs":epochs,"vocab_size":len(vocab.itos),
           "init_hash_prefix":ihp,"counts":counts,"train_info":info,
           "train_state_acc":ft.get("train_state_acc"),"train_cmp_acc":ft.get("train_cmp_acc"),
           "central_eval":central,"matching_accuracy":{"mode":matcher_mode,
               "eval":eval_match,"train":train_match},
           "elapsed_seconds":info["elapsed_seconds"]}
    
    write_json(rd/"result.json", rec)
    write_jsonl(rd/"state_predictions.jsonl", eval_sr)
    write_jsonl(rd/"comparison_predictions.jsonl", eval_cr)
    
    return rec

# ── Pipeline ──

def write_summary(out, results, matcher_mode):
    lines = [f"# research raw span discovery gauge probe",
             f"Matcher: {matcher_mode}, Conditions: {sorted(set(r['condition'] for r in results))}",
             f"Epochs: {results[0]['epochs'] if results else '?'}, Vocab: {results[0].get('vocab_size','?') if results else '?'}","",
             "| condition | matcher | bs | seed | train_state | train_cmp | graph_same | unchanged | hh_closure | mixed_acc | match_cand | match_other |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r.get("central_eval",{})
        ma = r.get("matching_accuracy",{}).get("eval",{})
        lines.append(f"| {r['condition']} | {r['matcher']} | {'+' if r['bridge_sign']==1 else '-'}1 | {r['seed']} "
            f"| {r.get('train_state_acc',0):.3f} | {r.get('train_cmp_acc',0):.3f} "
            f"| {c.get('graph_same',0):.3f} | {c.get('unchanged',0):.3f} "
            f"| {c.get('hh_closure',0):.3f} | {c.get('mixed_acc',0):.3f} "
            f"| {ma.get('cand_match_acc',0):.3f} | {ma.get('other_match_acc',0):.3f} |")
    write_json(out/"span_discovery_summary.json", {"all_results":results})
    (out/"span_discovery_summary.md").write_text("\n".join(lines)+"\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--conditions", nargs="+", default=["shared_trunk", "untied"])
    ap.add_argument("--matcher", default="learned", choices=["oracle","learned"])
    ap.add_argument("--seed", type=int, default=29600)
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--emb-dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--char-dim", type=int, default=16)
    ap.add_argument("--char-hidden", type=int, default=16)
    ap.add_argument("--cmp-weight", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=1.0)
    ap.add_argument("--static-weight", type=float, default=1.0)
    ap.add_argument("--print-every", type=int, default=55)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bridge-signs", nargs="+", type=int, choices=[1, -1], default=[1, -1],
                    help="Bridge signs to run; use a single sign for one recoverable GPU cell")
    ap.add_argument("--no-name-vocab", action="store_true",
                    help="Exclude name tokens from vocab; forces CharGRU as sole identity pathway")
    args = ap.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"research span discovery probe: matcher={args.matcher} conditions={args.conditions} bridge_signs={args.bridge_signs} no_name_vocab={args.no_name_vocab}", flush=True)
    
    # Build paired models
    pe = []
    ts,tc,_,_,_,_ = load_dataset(args.data_root, args.arm)
    
    # Collect all train names for vocab exclusion
    all_train_names = set()
    for q in ts:
        for n in q.names: all_train_names.add(n)
    for c in tc:
        for n in c.names: all_train_names.add(n)
    
    vocab = RawVocab()
    if args.no_name_vocab:
        collect_raw_vocab(vocab, ts, tc, exclude_names=all_train_names)
        print(f"Vocabulary: {len(vocab.itos)} tokens (names excluded)", flush=True)
    else:
        collect_raw_vocab(vocab, ts, tc)
        print(f"Vocabulary: {len(vocab.itos)} tokens", flush=True)
    
    # Collect name sets
    train_names = set(); eval_names_set = set()
    for q in ts:
        for n in q.names: train_names.add(n)
    eval_data = []
    for p in sorted((args.data_root/"eval").glob("*.jsonl")):
        if p.name=="name_permutation_counterfactual.jsonl": continue
        eval_data.extend(load_jsonl(p))
    for r in eval_data:
        for f in ["cause_event","event1","event2"]:
            t = str(r.get(f,""))
            if t:
                for n in extract_names(t)[:2]: eval_names_set.add(n)
    eval_names_set -= train_names
    print(f"Train names ({len(train_names)}): {sorted(train_names)}", flush=True)
    print(f"Eval names ({len(eval_names_set)}): {sorted(eval_names_set)}", flush=True)
    
    # Check which eval names are OOV
    oov = [n for n in sorted(eval_names_set) if n.lower() not in vocab.stoi]
    print(f"OOV eval names: {oov}", flush=True)
    
    results = []
    for bs in args.bridge_signs:
        paired = create_paired_span_models(
            len(vocab.itos), args.conditions, args.seed,
            args.emb_dim, args.hidden, args.char_dim, args.char_hidden)
        
        # Record init hash per condition
        for cond in args.conditions:
            ih = hashlib.sha256()
            for p in paired[cond].parameters(): ih.update(p.detach().cpu().numpy().tobytes())
            print(f"Init hash {cond}/bs{'+' if bs==1 else '-'}1: {ih.hexdigest()[:16]}", flush=True)
        
        for cond in args.conditions:
            print(f"\n=== {args.matcher} {cond}/bs={'+' if bs==1 else '-'}1/seed={args.seed} init={paired[cond].__class__.__name__} ===", flush=True)
            rec = run_one(args.data_root, args.out, cond, args.arm, args.seed, bs,
                           args.epochs, args.lr, args.wd, args.emb_dim, args.hidden,
                           args.char_dim, args.char_hidden, device,
                           args.cmp_weight, args.state_weight, args.static_weight,
                           args.print_every, args.matcher, paired, args.no_name_vocab)
            results.append(rec)
            print(json.dumps(rec, indent=2, sort_keys=True, default=str), flush=True)
    
    write_summary(args.out, results, args.matcher)
    print(json.dumps({"status":"SPAN_DISCOVERY_COMPLETE",
                      "summary": project_rel(args.out/"span_discovery_summary.md")}, indent=2), flush=True)

if __name__ == "__main__":
    main()
