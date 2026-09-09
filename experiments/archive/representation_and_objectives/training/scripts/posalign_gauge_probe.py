#!/usr/bin/env python3
"""research: Position-aligned matcher + gauge-transport probe.

Scientific purpose
------------------
research CharGRU capacity pilot showed that eval hard-assignment accuracy
plateaus ~0.5 regardless of hidden dimension (16/32/64) because the GRU
compresses character sequences into fixed vectors, conflating exact equality
with approximate character similarity. This prevents entity detection
(finding name tokens among all tokens) despite good entity identification
(distinguishing between different names at the same position).

This script replaces the CharGRU with a PositionAlignedMatcher:
  - Compare characters position by position using learned embeddings
  - Soft AND (product) over positions: high only if ALL characters match
  - Length match check: penalize different-length strings
  
This has the correct inductive bias for string equality and should generalize
to unseen names because it doesn't compress the character sequence.

Then runs the standard gauge-transport causal test:
  - shared_trunk bs+1 and bs-1 with frozen position-aligned hard assignment
  - Compares with untied control

The scientific question: does position-aligned matching restore gauge
transport on raw text with no-name vocabulary?
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, hashlib, json, math, os, random, sys, time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(_public_path('experiments/archive/representation_and_objectives/training/scripts')))
import raw_span_discovery_probe as base

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_OUT = PROJECT / "data/posalign_gauge"
DEFAULT_DATA_ROOT = base.DEFAULT_DATA_ROOT
DEFAULT_ARM = base.DEFAULT_ARM

def project_rel(p):
    try: return str(p.relative_to(Path.cwd()))
    except: return str(p)
def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)+"\n")
def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True)+"\n")


# ═════════════════════════════════════════════════════════════════════
# Position-Aligned Matcher
# ═════════════════════════════════════════════════════════════════════

MAX_NAME_LEN = 10  # longest name in the dataset is ~6 chars

class PositionAlignedMatcher(nn.Module):
    """Compare character sequences position-by-position, then soft-AND.
    
    For each (token, query_name) pair:
    1. Embed characters at each position with learned embeddings
    2. Compute per-position similarity (dot product)  
    3. Sigmoid → per-position match probability
    4. Product over positions → overall match score (soft AND)
    5. Length penalty: if different lengths, multiply by a learned penalty
    
    This generalizes to unseen strings because it checks each position
    independently: same character → high similarity, different → low.
    """
    def __init__(self, n_chars=28, char_dim=16, temperature=1.0):
        super().__init__()
        self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.length_penalty = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ≈ 0.05
    
    def match_score(self, tok_chars: torch.Tensor, qry_chars: torch.Tensor) -> torch.Tensor:
        """Compute match score between a token and a query name.
        
        tok_chars: (max_len,) padded character IDs for token
        qry_chars: (max_len,) padded character IDs for query name
        Returns: scalar match score in (0, 1)
        """
        tok_emb = self.char_emb(tok_chars)  # (max_len, char_dim)
        qry_emb = self.char_emb(qry_chars)  # (max_len, char_dim)
        
        # Per-position similarity
        sim = (tok_emb * qry_emb).sum(dim=-1)  # (max_len,)
        
        # Masks: which positions have real characters (not padding)
        tok_mask = tok_chars > 0  # (max_len,)
        qry_mask = qry_chars > 0  # (max_len,)
        both_valid = tok_mask & qry_mask
        
        # Length check: both should be the same length
        tok_len = tok_mask.sum()
        qry_len = qry_mask.sum()
        same_length = (tok_len == qry_len).float()
        
        # Per-position match probability (only at valid positions)
        match_prob = torch.sigmoid(sim / self.temperature.abs().clamp(min=0.1))
        
        # Product over valid positions (soft AND)
        # For positions where only one is valid, use penalty
        valid_match = match_prob * both_valid.float() + (1.0 - both_valid.float())
        
        # Additional penalty for positions where only query has a char (tok shorter)
        only_qry = qry_mask & (~tok_mask)
        # or where only tok has a char (tok longer)  
        only_tok = tok_mask & (~qry_mask)
        
        product = valid_match.prod()
        
        # Length penalty: if lengths differ, scale down
        len_factor = same_length + (1 - same_length) * torch.sigmoid(self.length_penalty)
        
        return product * len_factor
    
    def batch_match_scores(self, tok_chars_batch: torch.Tensor,
                           qry_chars: torch.Tensor) -> torch.Tensor:
        """Compute match scores for all tokens against one query.
        
        tok_chars_batch: (n_tokens, max_len)
        qry_chars: (max_len,)
        Returns: (n_tokens,) match scores
        """
        tok_emb = self.char_emb(tok_chars_batch)  # (n, max_len, dim)
        qry_emb = self.char_emb(qry_chars)        # (max_len, dim)
        
        sim = (tok_emb * qry_emb.unsqueeze(0)).sum(dim=-1)  # (n, max_len)
        
        tok_mask = tok_chars_batch > 0   # (n, max_len)
        qry_mask = qry_chars > 0         # (max_len,)
        both_valid = tok_mask & qry_mask.unsqueeze(0)  # (n, max_len)
        
        tok_len = tok_mask.sum(dim=1)     # (n,)
        qry_len = qry_mask.sum()          # scalar
        same_length = (tok_len == qry_len).float()  # (n,)
        
        match_prob = torch.sigmoid(sim / self.temperature.abs().clamp(min=0.1))
        valid_match = match_prob * both_valid.float() + (1.0 - both_valid.float())
        
        product = valid_match.prod(dim=1)  # (n,)
        len_factor = same_length + (1 - same_length) * torch.sigmoid(self.length_penalty)
        
        return product * len_factor


# ═════════════════════════════════════════════════════════════════════
# Position-aligned gated trunk
# ═════════════════════════════════════════════════════════════════════

def name_to_padded_chars(name: str, max_len: int = MAX_NAME_LEN) -> List[int]:
    """Convert name to padded character ID list."""
    chars = base.name_to_char_ids(name)
    result = chars[:max_len] + [0] * max(0, max_len - len(chars))
    return result


class PosAlignTrunk(nn.Module):
    """Word embeddings + position-aligned matching → GRU."""
    def __init__(self, vocab_size, emb_dim, hidden, char_dim=16):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cand_param = nn.Parameter(torch.randn(emb_dim) * 0.02)
        self.other_param = nn.Parameter(torch.randn(emb_dim) * 0.02)
        self.matcher = PositionAlignedMatcher(28, char_dim, temperature=1.0)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.out_dim = 2 * hidden
    
    def forward_matched(self, token_ids, mask, tok_char_padded, cand_chars, other_chars,
                         is_special, hard=True):
        """
        token_ids: (1, seq_len)
        mask: (1, seq_len) 
        tok_char_padded: (seq_len, max_len) padded char IDs for each token
        cand_chars: (max_len,) padded char IDs for candidate
        other_chars: (max_len,) padded char IDs for other
        is_special: (seq_len,) bool
        hard: if True, argmax hard assignment; if False, soft weighted sum
        """
        word_x = self.word_emb(token_ids)  # (1, seq_len, emb_dim)
        sl = token_ids.size(1)
        
        with torch.no_grad() if hard else torch.enable_grad():
            sc = self.matcher.batch_match_scores(tok_char_padded, cand_chars)  # (sl,)
            so = self.matcher.batch_match_scores(tok_char_padded, other_chars)  # (sl,)
        
        # Mask specials
        sc = sc.clone(); so = so.clone()
        sc[is_special] = 0.0
        so[is_special] = 0.0
        
        if hard:
            # Argmax hard assignment
            sc_np = sc.detach().cpu().numpy()
            so_np = so.detach().cpu().numpy()
            ci = int(sc_np.argmax())
            oi = int(so_np.argmax())
            if ci == oi:
                if sc_np[ci] >= so_np[oi]:
                    so_np2 = so_np.copy(); so_np2[ci] = -1; oi = int(so_np2.argmax())
                else:
                    sc_np2 = sc_np.copy(); sc_np2[oi] = -1; ci = int(sc_np2.argmax())
            
            # Use weighted-sum (not in-place) for clean autograd
            probs = torch.zeros((sl, 3), device=word_x.device)
            probs[:, 2] = 1.0
            probs[ci, 0] = 1.0; probs[ci, 2] = 0.0
            probs[oi, 1] = 1.0; probs[oi, 2] = 0.0
            
            gated = (probs[:, 2:3] * word_x[0] +
                     probs[:, 0:1] * self.cand_param.unsqueeze(0) +
                     probs[:, 1:2] * self.other_param.unsqueeze(0))
        else:
            # Soft weighted sum
            sn = torch.ones(sl, device=sc.device) * 0.01  # small neither baseline
            logits = torch.stack([sc, so, sn], dim=-1)
            logits[is_special, 0] = -1e6; logits[is_special, 1] = -1e6
            probs = F.softmax(logits * 10.0, dim=-1)  # temperature-scaled
            gated = (probs[:, 2:3] * word_x[0] +
                     probs[:, 0:1] * self.cand_param.unsqueeze(0) +
                     probs[:, 1:2] * self.other_param.unsqueeze(0))
        
        gated = gated.unsqueeze(0)
        y, _ = self.gru(gated)
        m = mask.unsqueeze(-1).float()
        h = (y * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return h, probs, (ci if hard else -1, oi if hard else -1)


class PosAlignScorer(nn.Module):
    def __init__(self, trunk, head):
        super().__init__()
        self.trunk = trunk; self.head = head
    def score_matched(self, tok_ids, mask, tcp, cc, oc, isp, hard=True):
        h, probs, assign = self.trunk.forward_matched(tok_ids, mask, tcp, cc, oc, isp, hard)
        return self.head(h), probs, assign


def make_posalign_model(vocab_size, mode, emb_dim=48, hidden=64, char_dim=16):
    ref_trunk = PosAlignTrunk(vocab_size, emb_dim, hidden, char_dim)
    ref_head = base.ScalarHead(ref_trunk.out_dim, hidden)
    ref_st = PosAlignTrunk(vocab_size, emb_dim, hidden, char_dim)
    ref_sh = base.ScalarHead(ref_st.out_dim, hidden)
    
    class PAModel(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.mode = m
            if m == "tied":
                t = copy.deepcopy(ref_trunk); h = copy.deepcopy(ref_head)
                self.event_state = PosAlignScorer(t, h)
                self.event_cmp = self.event_state
            elif m == "shared_trunk":
                t = copy.deepcopy(ref_trunk)
                self.event_state = PosAlignScorer(t, copy.deepcopy(ref_head))
                self.event_cmp = PosAlignScorer(t, copy.deepcopy(ref_head))
            elif m == "untied":
                self.event_state = PosAlignScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head))
                self.event_cmp = PosAlignScorer(copy.deepcopy(ref_trunk), copy.deepcopy(ref_head))
            else: raise ValueError(m)
            self.static = PosAlignScorer(copy.deepcopy(ref_st), copy.deepcopy(ref_sh))
    return PAModel(mode)


def create_paired_posalign(vocab_size, modes, seed, emb_dim=48, hidden=64, char_dim=16):
    torch.manual_seed(seed)
    models = {}
    for mode in modes:
        models[mode] = make_posalign_model(vocab_size, mode, emb_dim, hidden, char_dim)
    return models


# ═════════════════════════════════════════════════════════════════════
# Scoring wrappers
# ═════════════════════════════════════════════════════════════════════

def prepare_posalign_inputs(text, cand, other, device):
    tok_ids_str, char_forms = base.raw_tokenize(text)
    sl = len(tok_ids_str)
    
    tok_padded = torch.zeros((sl, MAX_NAME_LEN), dtype=torch.long, device=device)
    for i, f in enumerate(char_forms):
        if f:
            chars = name_to_padded_chars(f)
            tok_padded[i] = torch.tensor(chars, dtype=torch.long, device=device)
    
    cand_chars = torch.tensor(name_to_padded_chars(cand), dtype=torch.long, device=device)
    other_chars = torch.tensor(name_to_padded_chars(other), dtype=torch.long, device=device)
    
    isp = torch.zeros(sl, dtype=torch.bool, device=device)
    isp[0] = True; isp[-1] = True
    for i, f in enumerate(char_forms):
        if not f or f == "<hyp>": isp[i] = True
    
    return tok_ids_str, char_forms, tok_padded, cand_chars, other_chars, isp


def score_event_pa(scorer, vocab, text, names, device, hard=True):
    tok_ids_str, char_forms, tcp, _, _, isp = prepare_posalign_inputs(text, names[0], names[1], device)
    token_ids = vocab.encode(tok_ids_str)
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
    mask = torch.ones((1, len(token_ids)), dtype=torch.bool, device=device)
    
    scores = []; match_info = []
    for ci, cand in enumerate(names):
        other = names[1 - ci]
        cc = torch.tensor(name_to_padded_chars(cand), dtype=torch.long, device=device)
        oc = torch.tensor(name_to_padded_chars(other), dtype=torch.long, device=device)
        s, probs, assign = scorer.score_matched(ids_t, mask, tcp, cc, oc, isp, hard)
        scores.append(s.squeeze())
        
        cand_lc = cand.lower()
        other_lc = other.lower()
        rec = {"cand_name": cand, "other_name": other}
        cand_pos = next((i for i, f in enumerate(char_forms) if f == cand_lc), -1)
        other_pos = next((i for i, f in enumerate(char_forms) if f == other_lc), -1)
        if hard:
            rec["assigned_cand"] = assign[0]; rec["assigned_other"] = assign[1]
            rec["cand_correct"] = (assign[0] == cand_pos)
            rec["other_correct"] = (assign[1] == other_pos)
            rec["both_correct"] = rec["cand_correct"] and rec["other_correct"]
        match_info.append(rec)
    
    return torch.stack(scores), match_info


def score_static_pa(scorer, vocab, prefix, hyp, names, cand, device, hard=True):
    text = prefix + " " + hyp
    tok_ids_str, char_forms = base.raw_tokenize_static(prefix, hyp)
    token_ids = vocab.encode(tok_ids_str)
    sl = len(token_ids)
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
    mask = torch.ones((1, sl), dtype=torch.bool, device=device)
    
    tcp = torch.zeros((sl, MAX_NAME_LEN), dtype=torch.long, device=device)
    for i, f in enumerate(char_forms):
        if f:
            tcp[i] = torch.tensor(name_to_padded_chars(f), dtype=torch.long, device=device)
    
    other = names[1] if cand == names[0] else names[0]
    cc = torch.tensor(name_to_padded_chars(cand), dtype=torch.long, device=device)
    oc = torch.tensor(name_to_padded_chars(other), dtype=torch.long, device=device)
    isp = torch.zeros(sl, dtype=torch.bool, device=device)
    isp[0] = True; isp[-1] = True
    for i, f in enumerate(char_forms):
        if not f or f == "<hyp>": isp[i] = True
    
    s, _, _ = scorer.score_matched(ids_t, mask, tcp, cc, oc, isp, hard)
    return s.squeeze()


# ═════════════════════════════════════════════════════════════════════
# Equality pretraining for the position-aligned matcher
# ═════════════════════════════════════════════════════════════════════

def unique_train_events(states, comps):
    seen = set(); out = []
    for q in states:
        if q.is_changed:
            k = (q.event, tuple(q.names))
            if k not in seen: seen.add(k); out.append(k)
    for c in comps:
        for ev in [c.event1, c.event2]:
            k = (ev, tuple(c.names))
            if k not in seen: seen.add(k); out.append(k)
    return out


def pretrain_posalign(model, examples, device, epochs, lr, print_every=10):
    """Pretrain the position-aligned matcher with contrastive match scores."""
    # Collect matcher parameters from all trunks
    params = []; seen = set()
    for scorer in [model.event_state, model.event_cmp]:
        m = scorer.trunk.matcher
        if id(m) not in seen:
            for p in m.parameters():
                if id(p) not in seen: params.append(p); seen.add(id(p))
            seen.add(id(m))
    
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(42)
    
    matchers = []
    seen_m = set()
    for scorer in [model.event_state, model.event_cmp]:
        if id(scorer.trunk.matcher) not in seen_m:
            matchers.append(scorer.trunk.matcher)
            seen_m.add(id(scorer.trunk.matcher))
    
    hist = []; t0 = time.time()
    model.to(device)
    
    for ep in range(1, epochs + 1):
        model.train()
        exs = list(examples); rng.shuffle(exs)
        opt.zero_grad(set_to_none=True)
        losses = []
        
        for matcher in matchers:
            for text, names in exs:
                _, char_forms = base.raw_tokenize(text)
                sl = len(char_forms)
                tcp = torch.zeros((sl, MAX_NAME_LEN), dtype=torch.long, device=device)
                for i, f in enumerate(char_forms):
                    if f: tcp[i] = torch.tensor(name_to_padded_chars(f), dtype=torch.long, device=device)
                
                for cand in names:
                    other = names[1] if cand == names[0] else names[0]
                    cc = torch.tensor(name_to_padded_chars(cand), dtype=torch.long, device=device)
                    oc = torch.tensor(name_to_padded_chars(other), dtype=torch.long, device=device)
                    
                    sc = matcher.batch_match_scores(tcp, cc)  # (sl,)
                    so = matcher.batch_match_scores(tcp, oc)  # (sl,)
                    
                    # Targets: for cand query, the token matching cand should be high
                    cand_lc = cand.lower(); other_lc = other.lower()
                    target_c = torch.zeros(sl, device=device)
                    target_o = torch.zeros(sl, device=device)
                    for i, f in enumerate(char_forms):
                        if f == cand_lc: target_c[i] = 1.0
                        elif f == other_lc: target_o[i] = 1.0
                    
                    # BCE loss: push match scores toward targets
                    loss_c = F.binary_cross_entropy(sc.clamp(1e-7, 1-1e-7), target_c, reduction="mean")
                    loss_o = F.binary_cross_entropy(so.clamp(1e-7, 1-1e-7), target_o, reduction="mean")
                    losses.append(loss_c + loss_o)
        
        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        
        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            rec = {"epoch": ep, "loss": float(loss.detach().cpu()) if losses else 0,
                   "elapsed": round(time.time() - t0, 1)}
            hist.append(rec)
            if print_every: print(json.dumps({"posalign_pretrain": rec}), flush=True)
    
    return hist


def eval_matching_accuracy(model, vocab, states, comps, device):
    """Evaluate hard-assignment accuracy of the position-aligned matcher."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for q in states:
            if not q.is_changed: continue
            _, mi = score_event_pa(model.event_state, vocab, q.event, q.names, device, hard=True)
            for m in mi:
                if "both_correct" in m: correct += m["both_correct"]; total += 1
        for c in comps:
            for ev in [c.event1, c.event2]:
                _, mi = score_event_pa(model.event_cmp, vocab, ev, c.names, device, hard=True)
                for m in mi:
                    if "both_correct" in m: correct += m["both_correct"]; total += 1
    return correct / total if total else 0, total


def freeze_matcher(model):
    """Freeze matcher parameters."""
    for scorer in [model.event_state, model.event_cmp, model.static]:
        for p in scorer.trunk.matcher.parameters():
            p.requires_grad_(False)


# ═════════════════════════════════════════════════════════════════════
# Relational training + evaluation (from research hard-assignment)
# ═════════════════════════════════════════════════════════════════════

def train_relational(model, vocab, ts, tc, device, bridge_sign, epochs, lr, wd,
                     seed, cmp_w, state_w, static_w, print_every, hard):
    model.to(device)
    seen = set(); all_params = []; clip_groups = []
    def add_mp(mod):
        gp = []
        for p in mod.parameters():
            if p.requires_grad:
                pid = id(p)
                if pid not in seen: seen.add(pid); all_params.append(p); gp.append(p)
        if gp: clip_groups.append(gp)
    add_mp(model.event_state)
    if model.event_cmp is not model.event_state: add_mp(model.event_cmp)
    add_mp(model.static)

    if not all_params: return {"error": "no trainable params"}
    opt = torch.optim.AdamW([{"params": all_params}], lr=lr, weight_decay=wd)
    rng = random.Random(seed)
    history = []; t0 = time.time()

    for ep in range(1, epochs+1):
        model.train()
        states = list(ts); comps = list(tc)
        rng.shuffle(states); rng.shuffle(comps)
        opt.zero_grad(set_to_none=True)
        losses = []

        for q in states:
            if q.is_changed:
                sc, _ = score_event_pa(model.event_state, vocab, q.event, q.names, device, hard)
            else:
                sc = torch.stack([score_static_pa(model.static, vocab, q.prefix,
                    q.hypothesis_text_by_name[cn], q.names, cn, device, hard) for cn in q.names])
            target = base.get_target_idx(q)
            # Match research/296 bridge_sign semantics exactly: reverse only
            # sparse direct-anchor changed-state rows. Graph-transfer rows,
            # unchanged/static rows, comparison rows, and evaluation labels
            # remain canonical.
            if bridge_sign == -1 and q.is_direct_anchor:
                sc_eff = sc.flip(0)
            else:
                sc_eff = sc
            loss = F.cross_entropy(sc_eff.view(1, -1),
                                   torch.tensor([target], dtype=torch.long, device=device))
            losses.append((state_w if q.is_changed else static_w) * loss)

        for c in comps:
            s1, _ = score_event_pa(model.event_cmp, vocab, c.event1, c.names, device, hard)
            s2, _ = score_event_pa(model.event_cmp, vocab, c.event2, c.names, device, hard)
            p1, p2 = F.softmax(s1, dim=0), F.softmax(s2, dim=0)
            psame = (p1*p2).sum().clamp(1e-6, 1-1e-6)
            y = torch.tensor(float(c.label), dtype=torch.float32, device=psame.device)
            losses.append(cmp_w * (-(y*torch.log(psame) + (1-y)*torch.log(1-psame))))

        total = torch.stack(losses).mean()
        total.backward()
        for gp in clip_groups: torch.nn.utils.clip_grad_norm_(gp, 5.0)
        opt.step()

        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            model.eval()
            with torch.no_grad():
                cs, ns, cch, nch = 0, 0, 0, 0
                for q in ts:
                    if q.is_changed:
                        sc2, _ = score_event_pa(model.event_state, vocab, q.event, q.names, device, hard)
                    else:
                        sc2 = torch.stack([score_static_pa(model.static, vocab, q.prefix,
                            q.hypothesis_text_by_name[cn], q.names, cn, device, hard) for cn in q.names])
                    target = base.get_target_idx(q)
                    if bridge_sign == -1 and q.is_direct_anchor:
                        sc2_eff = sc2.flip(0)
                    else:
                        sc2_eff = sc2
                    c_ok = int(sc2_eff.argmax().item() == target)
                    if q.is_changed: cch += c_ok; nch += 1
                    cs += c_ok; ns += 1

                cc_cmp, nc_cmp = 0, 0
                for c in tc:
                    s1, _ = score_event_pa(model.event_cmp, vocab, c.event1, c.names, device, hard)
                    s2, _ = score_event_pa(model.event_cmp, vocab, c.event2, c.names, device, hard)
                    psame = (F.softmax(s1,0)*F.softmax(s2,0)).sum()
                    cc_cmp += int((psame>=0.5)==c.label); nc_cmp += 1

            rec = {"epoch": ep, "loss": float(total.detach().cpu()),
                   "train_state": cs/ns if ns else 0, "train_changed": cch/nch if nch else 0,
                   "train_cmp": cc_cmp/nc_cmp if nc_cmp else 0}
            history.append(rec)
            if print_every: print(json.dumps(rec), flush=True)

    return {"elapsed_seconds": time.time()-t0, "history": history}
    return {"elapsed_seconds": time.time()-t0, "history": history}


def eval_predictions(model, vocab, states, comps, device, cond, arm, seed, bs, hard):
    model.eval()
    sr = []; cr = []
    with torch.no_grad():
        for q in states:
            if q.is_changed:
                sc, mi = score_event_pa(model.event_state, vocab, q.event, q.names, device, hard)
            else:
                sc = torch.stack([score_static_pa(model.static, vocab, q.prefix,
                    q.hypothesis_text_by_name[cn], q.names, cn, device, hard) for cn in q.names])
                mi = []
            target = base.get_target_idx(q)
            inv_t = q.inverted_target_by_name[q.names[1]]
            scores_np = [float(sc[ci].detach().cpu()) for ci in range(len(q.names))]
            d_e = scores_np[0] - scores_np[1] if len(scores_np) == 2 else 0
            probs_np = [float(F.softmax(sc, dim=0)[ci].detach().cpu()) for ci in range(len(q.names))]
            for ci, cn in enumerate(q.names):
                lt = bool(q.label_by_name[cn])
                sr.append({"row_id": q.key, "suite": q.suite, "split": q.split, "arm": arm,
                    "condition": cond, "bridge_sign": bs, "seed": seed,
                    "candidate": cn, "candidate_index": ci, "is_changed": q.is_changed,
                    "names": list(q.names), "score": scores_np[ci], "prob": probs_np[ci], "d_e": d_e,
                    "target": target, "predicted": int(sc.argmax().item()),
                    "correct": int(sc.argmax().item()==target), "inverted_target": bool(inv_t),
                    "query_key": q.key, "task": "state_query", "label_true": lt,
                    "query_kind": q.query_kind, "relation": q.relation,
                    "relation_family": q.relation_family, "initial_pattern": q.initial_pattern,
                    "static_slot": q.static_slot, "is_direct_anchor": q.is_direct_anchor,
                    "object": q.object_name,
                    "global_swap_changes_label": q.metadata.get("global_swap_changes_label"),
                    "raw_query_kind": q.metadata.get("raw_query_kind")})
        for c in comps:
            s1, _ = score_event_pa(model.event_cmp, vocab, c.event1, c.names, device, hard)
            s2, _ = score_event_pa(model.event_cmp, vocab, c.event2, c.names, device, hard)
            psame = float((F.softmax(s1,0)*F.softmax(s2,0)).sum().detach().cpu())
            logit = math.log(psame/(1-psame)) if 1e-7 < psame < 1-1e-7 else 0.0
            de1 = float(s1[0])-float(s1[1])
            de2 = float(s2[0])-float(s2[1])
            cr.append({"condition": cond, "arm": arm, "seed": seed, "bridge_sign": bs,
                "suite": c.suite, "split": c.split, "row_id": c.row_id,
                "task": "relation_comparison",
                "prob_same": psame, "logit_same": logit, "label": bool(c.label),
                "correct": int(bool(psame>=0.5)==bool(c.label)),
                "signed_margin": logit if c.label else -logit,
                "d_e1": de1, "d_e2": de2, "names": list(c.names),
                "relation1": c.relation1, "relation2": c.relation2})
    return sr, cr


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def run_one(data_root, out, cond, arm, seed, bs, eq_epochs, eq_lr,
            epochs, lr, wd, emb_dim, hidden, char_dim, cmp_w, state_w, static_w,
            print_every, no_name_vocab, gpu, paired):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
    
    ts, tc, es, ec, pe, counts = base.load_dataset(data_root, arm)
    train_names = set()
    for q in ts:
        for n in q.names: train_names.add(n)
    for c in tc:
        for n in c.names: train_names.add(n)
    vocab = base.RawVocab()
    if no_name_vocab:
        base.collect_raw_vocab(vocab, ts, tc, exclude_names=train_names)
    else:
        base.collect_raw_vocab(vocab, ts, tc)
    
    model = copy.deepcopy(paired[cond])
    model.to(device)
    
    ih = hashlib.sha256()
    for p in model.parameters(): ih.update(p.detach().cpu().numpy().tobytes())
    ihp = ih.hexdigest()[:16]
    print(f"Init hash {cond}/bs{'+' if bs==1 else '-'}1: {ihp}", flush=True)
    
    examples = unique_train_events(ts, tc)
    
    # Pre-pretraining matching accuracy
    train_ma_before, _ = eval_matching_accuracy(model, vocab, ts, tc, device)
    eval_ma_before, _ = eval_matching_accuracy(model, vocab, es, ec, device)
    print(f"Matching before pretrain: train={train_ma_before:.4f} eval={eval_ma_before:.4f}", flush=True)
    
    # Equality pretraining
    eq_hist = None
    if eq_epochs > 0:
        eq_hist = pretrain_posalign(model, examples, device, eq_epochs, eq_lr, print_every=max(1, eq_epochs//5))
    
    train_ma_after, _ = eval_matching_accuracy(model, vocab, ts, tc, device)
    eval_ma_after, _ = eval_matching_accuracy(model, vocab, es, ec, device)
    print(f"Matching after pretrain: train={train_ma_after:.4f} eval={eval_ma_after:.4f}", flush=True)
    
    # Freeze matcher
    freeze_matcher(model)
    
    # Relational training
    print(f"\n=== posalign {cond}/bs={'+' if bs==1 else '-'}1/seed={seed} ===", flush=True)
    train_info = train_relational(model, vocab, ts, tc, device, bs, epochs, lr, wd,
                                   seed, cmp_w, state_w, static_w, print_every, hard=True)
    
    # Final eval
    model.eval()
    eval_sr, eval_cr = eval_predictions(model, vocab, es, ec, device, cond, arm, seed, bs, True)
    train_sr, train_cr = eval_predictions(model, vocab, ts, tc, device, cond, arm, seed, bs, True)
    
    eval_ma_final, _ = eval_matching_accuracy(model, vocab, es, ec, device)
    
    # Central metrics
    choices = base.state_choice_records(eval_sr)
    boths = base.pair_both_records(choices)
    central = base.compute_central(choices, boths, eval_cr, cond, arm, seed, bs)
    
    tag = f"posalign_{cond}_bs{'+' if bs==1 else '-'}1_seed{seed}"
    rd = out / tag
    rec = {
        "condition": cond, "arm": arm, "seed": seed, "bridge_sign": bs,
        "mode": "posalign_hard", "eq_pretrain_epochs": eq_epochs,
        "epochs": epochs, "vocab_size": len(vocab.itos), "no_name_vocab": no_name_vocab,
        "init_hash_prefix": ihp, "counts": counts,
        "matching_before": {"train": train_ma_before, "eval": eval_ma_before},
        "matching_after_pretrain": {"train": train_ma_after, "eval": eval_ma_after},
        "matching_final": eval_ma_final,
        "eq_history": eq_hist, "train_info": train_info,
        "central_eval": central,
    }
    write_json(rd / "result.json", rec)
    write_jsonl(rd / "state_predictions.jsonl", eval_sr)
    write_jsonl(rd / "comparison_predictions.jsonl", eval_cr)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arm", default=DEFAULT_ARM)
    ap.add_argument("--conditions", nargs="+", default=["shared_trunk"])
    ap.add_argument("--seed", type=int, default=29800)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--emb-dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--char-dim", type=int, default=16)
    ap.add_argument("--cmp-weight", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=1.0)
    ap.add_argument("--static-weight", type=float, default=1.0)
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bridge-signs", nargs="+", type=int, choices=[1,-1], default=[1,-1])
    ap.add_argument("--no-name-vocab", action="store_true")
    ap.add_argument("--eq-pretrain-epochs", type=int, default=25)
    ap.add_argument("--eq-lr", type=float, default=5e-3)
    args = ap.parse_args()
    
    ts, tc, _, _, _, _ = base.load_dataset(args.data_root, args.arm)
    train_names = set()
    for q in ts:
        for n in q.names: train_names.add(n)
    for c in tc:
        for n in c.names: train_names.add(n)
    vocab = base.RawVocab()
    if args.no_name_vocab:
        base.collect_raw_vocab(vocab, ts, tc, exclude_names=train_names)
    else:
        base.collect_raw_vocab(vocab, ts, tc)
    
    print(f"research posalign probe: conditions={args.conditions} bridge_signs={args.bridge_signs} "
          f"no_name_vocab={args.no_name_vocab} vocab={len(vocab.itos)} eq_pre={args.eq_pretrain_epochs}", flush=True)
    
    results = []
    for bs in args.bridge_signs:
        paired = create_paired_posalign(len(vocab.itos), args.conditions, args.seed,
                                         args.emb_dim, args.hidden, args.char_dim)
        for cond in args.conditions:
            rec = run_one(args.data_root, args.out, cond, args.arm, args.seed, bs,
                          args.eq_pretrain_epochs, args.eq_lr,
                          args.epochs, args.lr, args.wd, args.emb_dim, args.hidden,
                          args.char_dim, args.cmp_weight, args.state_weight, args.static_weight,
                          args.print_every, args.no_name_vocab, args.gpu, paired)
            results.append(rec)
            ce = rec.get("central_eval", {})
            print(json.dumps({"done": f"posalign_{cond}_bs{'+' if bs==1 else '-'}1",
                               "graph_same": ce.get("graph_same"), "unchanged": ce.get("unchanged"),
                               "hh_closure": ce.get("hh_closure"), "mixed_acc": ce.get("mixed_acc"),
                               "matching_final": rec.get("matching_final")}, indent=2), flush=True)
    
    # Summary
    lines = ["# research position-aligned matcher gauge probe", "",
             "| condition | bs | vocab | match_after | match_final | train_state | train_cmp | graph_same | unchanged | hh_closure | mixed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        ce = r.get("central_eval", {})
        ti = r.get("train_info", {}).get("history", [{}])[-1] if r.get("train_info", {}).get("history") else {}
        lines.append(f"| {r['condition']} | {r['bridge_sign']} | {r['vocab_size']} "
                     f"| {r.get('matching_after_pretrain',{}).get('eval',0):.4f} "
                     f"| {r.get('matching_final',0):.4f} "
                     f"| {ti.get('train_state',0):.3f} | {ti.get('train_cmp',0):.3f} "
                     f"| {ce.get('graph_same',0):.3f} | {ce.get('unchanged',0):.3f} "
                     f"| {ce.get('hh_closure',0):.3f} | {ce.get('mixed_acc',0):.3f} |")
    write_json(args.out / "posalign_summary.json", {"all_results": results})
    (args.out / "posalign_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "POSALIGN_COMPLETE",
                      "summary": project_rel(args.out / "posalign_summary.md")}, indent=2))


if __name__ == "__main__":
    main()
