#!/usr/bin/env python3
"""research: hard-assignment gauge-transport probe.

Scientific purpose
------------------
research showed that the CharGRU has good relative discrimination (eval cand>other
0.971) but poor absolute gating (eval both_wins_gate 0.654). The soft three-way
softmax gate fails because the `neither` bias is poorly calibrated for held names.
Meanwhile, the hard oracle (exact string matching) restores the full gauge
fingerprint on the same raw text.

This probe tests whether hard assignment from the *learned* CharGRU — argmax
over tokens for each query name, then hard-substitution — restores gauge
transport. This isolates whether the bottleneck is:
  (a) soft-gate calibration (neither bias) → hard assignment fixes it
  (b) CharGRU representation quality → hard assignment also fails

Conditions:
  1. hard_pretrained_frozen: pretrain CharGRU on equality labels, freeze, hard-assign
  2. soft_pretrained_frozen: pretrain CharGRU on equality labels, freeze, soft-gate
  3. oracle_ceiling: exact string match (replication of research oracle)

Each condition runs shared_trunk bs+1 and bs-1 with fresh initialization.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, hashlib, json, math, os, random, sys, time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import base module ──
sys.path.insert(0, str(_public_path('experiments/archive/representation_and_objectives/training/scripts')))
import raw_span_discovery_probe as base
sys.path.pop(0)

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_OUT = PROJECT / "data/hard_assignment"
DEFAULT_DATA_ROOT = base.DEFAULT_DATA_ROOT
DEFAULT_ARM = base.DEFAULT_ARM


def project_rel(p: Path) -> str:
    try: return str(p.relative_to(Path.cwd()))
    except: return str(p)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


# ═════════════════════════════════════════════════════════════════════
# Hard-assignment scoring — argmax over tokens, no softmax gate
# ═════════════════════════════════════════════════════════════════════

def hard_assignment_score_gated(scorer, token_ids_list, char_forms,
                                 cand, other, device):
    """Learned hard assignment: CharGRU for scoring, argmax for assignment.
    
    1. Compute CharGRU dot-product similarity of each token to cand/other
    2. argmax to find the best-matching token for each name
    3. Hard-substitute: cand_token → cand_param, other_token → other_param
    4. All other tokens → word embedding
    """
    trunk = scorer.trunk
    sl = len(token_ids_list)
    ids_t = torch.tensor([token_ids_list], dtype=torch.long, device=device)
    mask = torch.ones((1, sl), dtype=torch.bool, device=device)
    word_x = trunk.word_emb(ids_t)[0]  # (sl, emb_dim)
    
    # Character encodings
    tok_chars = [base.name_to_char_ids(f) if f else [0] for f in char_forms]
    tcb = base.pad_char_batch(tok_chars, device)
    cc = base.pad_char_batch([base.name_to_char_ids(cand)], device)
    oc = base.pad_char_batch([base.name_to_char_ids(other)], device)
    
    with torch.no_grad():  # CharGRU is frozen during relational training
        tok_vecs = trunk.char_enc(tcb)   # (sl, char_hidden)
        cand_vec = trunk.char_enc(cc)     # (1, char_hidden)
        other_vec = trunk.char_enc(oc)    # (1, char_hidden)
    
    sc = (tok_vecs * cand_vec).sum(-1)   # (sl,)
    so = (tok_vecs * other_vec).sum(-1)  # (sl,)
    
    # Mask out special tokens (can't be names)
    for i, f in enumerate(char_forms):
        if not f or f == "<hyp>":
            sc[i] = -1e6
            so[i] = -1e6
    
    # Joint greedy argmax: assign globally best match first
    sc_np = sc.detach().cpu().numpy()
    so_np = so.detach().cpu().numpy()
    cand_best = int(sc_np.argmax())
    other_best = int(so_np.argmax())
    
    if cand_best == other_best:
        # Conflict: give it to the one with higher score, second-best for other
        if sc_np[cand_best] >= so_np[other_best]:
            so_np_copy = so_np.copy()
            so_np_copy[cand_best] = -1e9
            other_best = int(so_np_copy.argmax())
        else:
            sc_np_copy = sc_np.copy()
            sc_np_copy[other_best] = -1e9
            cand_best = int(sc_np_copy.argmax())
    
    # Build hard-assigned input
    gated = word_x.clone()
    gated[cand_best] = trunk.cand_param
    gated[other_best] = trunk.other_param
    
    # Diagnostic probs
    probs = torch.zeros((sl, 3), dtype=word_x.dtype, device=device)
    probs[:, 2] = 1.0
    probs[cand_best, 0] = 1.0; probs[cand_best, 2] = 0.0
    probs[other_best, 1] = 1.0; probs[other_best, 2] = 0.0
    
    y, _ = trunk.gru(gated.unsqueeze(0))
    m = mask.unsqueeze(-1).float()
    h = (y * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    return scorer.head(h), probs, (cand_best, other_best)


# ═════════════════════════════════════════════════════════════════════
# Scoring wrappers for hard/soft/oracle modes
# ═════════════════════════════════════════════════════════════════════

def score_event_ha(scorer, vocab, event_text, names, device, mode="hard"):
    """Score one event using hard-assignment, soft-gate, or oracle."""
    tok_ids_str, char_forms = base.raw_tokenize(event_text)
    token_ids = vocab.encode(tok_ids_str)
    scores = []; match_info_list = []
    
    for ci, cand in enumerate(names):
        other = names[1 - ci]
        if mode == "oracle":
            s, probs = base.hard_oracle_score_gated(scorer, token_ids, char_forms, cand, other, device)
            assign = None
        elif mode == "hard":
            s, probs, assign = hard_assignment_score_gated(scorer, token_ids, char_forms, cand, other, device)
        else:  # soft
            ids_t, mask, tcb, cc, oc, isp = base.prepare_span_inputs(
                token_ids, char_forms, cand, other, device, "learned")
            s, probs = scorer.score_gated(ids_t, mask, tcb, cc, oc, isp)
            assign = None
        scores.append(s.squeeze())
        
        # Diagnostics
        p_np = probs.detach().cpu().numpy()
        cand_lc, other_lc = cand.lower(), other.lower()
        match_rec = {"cand_name": cand, "other_name": other, "mode": mode}
        for pi, f in enumerate(char_forms):
            if f == cand_lc:
                match_rec["cand_pos"] = pi
                match_rec["cand_p_cand"] = float(p_np[pi, 0])
            elif f == other_lc:
                match_rec["other_pos"] = pi
                match_rec["other_p_other"] = float(p_np[pi, 1])
        if assign is not None:
            match_rec["assigned_cand_pos"] = assign[0]
            match_rec["assigned_other_pos"] = assign[1]
            # Check correctness
            cand_pos = next((i for i, f in enumerate(char_forms) if f == cand_lc), -1)
            other_pos = next((i for i, f in enumerate(char_forms) if f == other_lc), -1)
            match_rec["cand_correct"] = (assign[0] == cand_pos)
            match_rec["other_correct"] = (assign[1] == other_pos)
            match_rec["both_correct"] = (assign[0] == cand_pos and assign[1] == other_pos)
        match_info_list.append(match_rec)
    
    return torch.stack(scores), match_info_list


def score_static_ha(scorer, vocab, prefix, hyp, names, cand, device, mode="hard"):
    tok_ids_str, char_forms = base.raw_tokenize_static(prefix, hyp)
    token_ids = vocab.encode(tok_ids_str)
    other = names[1] if cand == names[0] else names[0]
    if mode == "oracle":
        s, _ = base.hard_oracle_score_gated(scorer, token_ids, char_forms, cand, other, device)
    elif mode == "hard":
        s, _, _ = hard_assignment_score_gated(scorer, token_ids, char_forms, cand, other, device)
    else:
        ids_t, mask, tcb, cc, oc, isp = base.prepare_span_inputs(
            token_ids, char_forms, cand, other, device, "learned")
        s, _ = scorer.score_gated(ids_t, mask, tcb, cc, oc, isp)
    return s.squeeze()


# ═════════════════════════════════════════════════════════════════════
# Hard-assignment evaluation: matching accuracy on raw events
# ═════════════════════════════════════════════════════════════════════

def eval_hard_assignment_accuracy(model, vocab, states, comps, device):
    """Evaluate hard-assignment correctness: does argmax find the right tokens?"""
    model.eval()
    records = {"state_changed": [], "comparison": []}
    
    with torch.no_grad():
        for q in states:
            if not q.is_changed: continue
            _, match_info = score_event_ha(model.event_state, vocab, q.event, q.names, device, mode="hard")
            for m in match_info:
                records["state_changed"].append(m)
        
        for c in comps:
            for ev in [c.event1, c.event2]:
                _, match_info = score_event_ha(model.event_cmp, vocab, ev, c.names, device, mode="hard")
                for m in match_info:
                    records["comparison"].append(m)
    
    summary = {}
    for kind, recs in records.items():
        if not recs: continue
        n = len(recs)
        cand_correct = sum(1 for r in recs if r.get("cand_correct", False)) / n
        other_correct = sum(1 for r in recs if r.get("other_correct", False)) / n
        both_correct = sum(1 for r in recs if r.get("both_correct", False)) / n
        summary[kind] = {"n": n, "cand_correct": cand_correct,
                         "other_correct": other_correct, "both_correct": both_correct}
    return summary, records


# ═════════════════════════════════════════════════════════════════════
# CharGRU equality pretraining (reuses research pattern)
# ═════════════════════════════════════════════════════════════════════

def unique_train_events(states, comps):
    """Unique changed-event texts + names for equality pretraining."""
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


def equality_loss_for_text(trunk, text, names, device, null_weight=1.0):
    """Cross-entropy loss for token-level equality labels."""
    _, char_forms = base.raw_tokenize(text)
    losses = []
    for ci, cand in enumerate(names):
        other = names[1 - ci]
        tok_chars = [base.name_to_char_ids(f) if f else [0] for f in char_forms]
        tcb = base.pad_char_batch(tok_chars, device)
        cc = base.pad_char_batch([base.name_to_char_ids(cand)], device)
        oc = base.pad_char_batch([base.name_to_char_ids(other)], device)
        
        tok_vecs = trunk.char_enc(tcb)
        cand_vec = trunk.char_enc(cc)
        other_vec = trunk.char_enc(oc)
        
        sc = (tok_vecs * cand_vec).sum(-1)
        so = (tok_vecs * other_vec).sum(-1)
        sn = trunk.neither_bias.expand(len(char_forms))
        
        logits = torch.stack([sc, so, sn], dim=-1)
        for i, f in enumerate(char_forms):
            if not f or f == "<hyp>":
                logits[i, 0] = -1e6; logits[i, 1] = -1e6
        
        cand_lc, other_lc = cand.lower(), other.lower()
        targets = []
        for f in char_forms:
            if f == cand_lc: targets.append(0)
            elif f == other_lc: targets.append(1)
            else: targets.append(2)
        targets = torch.tensor(targets, dtype=torch.long, device=device)
        
        ce = F.cross_entropy(logits, targets, reduction="none")
        name_mask = targets != 2
        null_mask = (targets == 2) & torch.tensor([bool(f) and f != "<hyp>" for f in char_forms], device=device)
        parts = []
        if name_mask.any(): parts.append(ce[name_mask].mean())
        if null_mask.any(): parts.append(null_weight * ce[null_mask].mean())
        if parts: losses.append(torch.stack(parts).mean())
    
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


def get_matcher_params(model):
    """Get CharGRU + neither_bias params from all unique trunks."""
    params = []; seen = set()
    for scorer in [model.event_state, model.event_cmp]:
        tr = scorer.trunk
        if id(tr) not in seen:
            seen.add(id(tr))
            for p in list(tr.char_enc.parameters()) + [tr.neither_bias]:
                if id(p) not in seen:
                    params.append(p); seen.add(id(p))
    return params


def pretrain_charenc(model, examples, device, epochs, lr, null_weight=1.0, print_every=10):
    """Pretrain the CharGRU with equality supervision."""
    params = get_matcher_params(model)
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(42)
    
    # Find unique trunks
    trunks = []; seen = set()
    for scorer in [model.event_state, model.event_cmp]:
        if id(scorer.trunk) not in seen:
            trunks.append(scorer.trunk); seen.add(id(scorer.trunk))
    
    hist = []; t0 = time.time()
    model.to(device)
    
    for ep in range(1, epochs + 1):
        model.train()
        exs = list(examples); rng.shuffle(exs)
        opt.zero_grad(set_to_none=True)
        losses = []
        for tr in trunks:
            for text, names in exs:
                losses.append(equality_loss_for_text(tr, text, names, device, null_weight))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        
        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            rec = {"epoch": ep, "loss": float(loss.detach().cpu()),
                   "elapsed": round(time.time() - t0, 1)}
            hist.append(rec)
            if print_every:
                print(json.dumps({"eq_pretrain": rec}), flush=True)
    
    return hist


def freeze_matcher(model):
    """Freeze CharGRU + neither_bias."""
    for p in get_matcher_params(model):
        p.requires_grad_(False)


# ═════════════════════════════════════════════════════════════════════
# Training with hard/soft/oracle routing
# ═════════════════════════════════════════════════════════════════════

def train_relational(model, vocab, train_states, train_comps, device, bridge_sign,
                     epochs, lr, wd, seed, cmp_w, state_w, static_w, mode, print_every=0):
    """Relational training using hard/soft/oracle scoring."""
    model.to(device)
    
    # Collect trainable params (CharGRU should already be frozen if freeze_matcher was called)
    seen = set(); all_params = []; clip_groups = []
    def add_mp(mod, _):
        gp = []
        for p in mod.parameters():
            if p.requires_grad:
                pid = id(p)
                if pid not in seen: seen.add(pid); all_params.append(p); gp.append(p)
        if gp: clip_groups.append(gp)
    add_mp(model.event_state, "es")
    if model.event_cmp is not model.event_state: add_mp(model.event_cmp, "ec")
    add_mp(model.static, "st")
    
    if not all_params:
        return {"error": "no trainable params", "elapsed_seconds": 0}
    
    opt = torch.optim.AdamW([{"params": all_params}], lr=lr, weight_decay=wd)
    rng = random.Random(seed)
    history = []; t0 = time.time()
    
    for ep in range(1, epochs + 1):
        model.train()
        states = list(train_states); comps = list(train_comps)
        rng.shuffle(states); rng.shuffle(comps)
        opt.zero_grad(set_to_none=True)
        losses = []
        
        # State loss
        for q in states:
            if q.is_changed:
                scores, _ = score_event_ha(model.event_state, vocab, q.event, q.names, device, mode)
            else:
                ss = []
                for c in q.names:
                    ss.append(score_static_ha(model.static, vocab, q.prefix,
                                             q.hypothesis_text_by_name[c], q.names, c, device, mode))
                scores = torch.stack(ss)
            
            target = base.get_target_idx(q)
            p = F.softmax(scores, dim=0)
            inv_target = q.inverted_target_by_name[q.names[1]]
            actual = target if bridge_sign == 1 else (1 - target if inv_target else target)
            bce_actual = -torch.log(p[actual].clamp(min=1e-9))
            bce_static = -torch.log(p[target].clamp(min=1e-9))
            
            if q.is_changed:
                losses.append(state_w * bce_actual)
            else:
                losses.append(static_w * bce_static)
        
        # Comparison loss
        for c in comps:
            s1, _ = score_event_ha(model.event_cmp, vocab, c.event1, c.names, device, mode)
            s2, _ = score_event_ha(model.event_cmp, vocab, c.event2, c.names, device, mode)
            p1, p2 = F.softmax(s1, dim=0), F.softmax(s2, dim=0)
            psame = (p1 * p2).sum().clamp(1e-6, 1 - 1e-6)
            y = torch.tensor(float(c.label), dtype=torch.float32, device=psame.device)
            bce_cmp = -(y * torch.log(psame) + (1 - y) * torch.log(1 - psame))
            losses.append(cmp_w * bce_cmp)
        
        total = torch.stack(losses).mean()
        total.backward()
        for gp in clip_groups:
            torch.nn.utils.clip_grad_norm_(gp, 5.0)
        opt.step()
        
        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            # Quick accuracy check
            model.eval()
            with torch.no_grad():
                cs, ns, cch, nch, cun, nun = 0, 0, 0, 0, 0, 0
                for q in states:
                    if q.is_changed:
                        sc2, _ = score_event_ha(model.event_state, vocab, q.event, q.names, device, mode)
                    else:
                        sc2 = torch.stack([score_static_ha(model.static, vocab, q.prefix,
                            q.hypothesis_text_by_name[cn], q.names, cn, device, mode) for cn in q.names])
                    target = base.get_target_idx(q)
                    inv_target = q.inverted_target_by_name[q.names[1]]
                    actual = target if bridge_sign == 1 else (1 - target if inv_target else target)
                    correct = int(sc2.argmax().item() == actual)
                    if q.is_changed:
                        cs += correct; ns += 1; cch += correct; nch += 1
                    else:
                        cs += correct; ns += 1; cun += correct; nun += 1
                
                cc_cmp, nc_cmp = 0, 0
                for c in comps:
                    s1, _ = score_event_ha(model.event_cmp, vocab, c.event1, c.names, device, mode)
                    s2, _ = score_event_ha(model.event_cmp, vocab, c.event2, c.names, device, mode)
                    p1, p2 = F.softmax(s1, dim=0), F.softmax(s2, dim=0)
                    psame = (p1 * p2).sum()
                    pred = int(psame >= 0.5)
                    cc_cmp += int(pred == c.label); nc_cmp += 1
            
            rec = {
                "epoch": ep, "loss": float(total.detach().cpu()),
                "train_state_acc": cs / ns if ns else 0,
                "train_changed_acc": cch / nch if nch else 0,
                "train_unchanged_acc": cun / nun if nun else 0,
                "train_cmp_acc": cc_cmp / nc_cmp if nc_cmp else 0,
            }
            history.append(rec)
            if print_every:
                print(json.dumps(rec), flush=True)
    
    return {"elapsed_seconds": time.time() - t0, "history": history}


# ═════════════════════════════════════════════════════════════════════
# Full evaluation pipeline
# ═════════════════════════════════════════════════════════════════════

def eval_predictions_ha(model, vocab, states, comps, device, condition, arm, seed, bridge_sign, mode):
    """Evaluate and collect per-row predictions."""
    model.eval()
    state_rows = []; comp_rows = []
    
    with torch.no_grad():
        for q in states:
            if q.is_changed:
                sc, mi = score_event_ha(model.event_state, vocab, q.event, q.names, device, mode)
            else:
                ss = []
                for cn in q.names:
                    ss.append(score_static_ha(model.static, vocab, q.prefix,
                                             q.hypothesis_text_by_name[cn], q.names, cn, device, mode))
                sc = torch.stack(ss); mi = []
            
            target = base.get_target_idx(q)
            inv_target = q.inverted_target_by_name[q.names[1]]
            for ci, cn in enumerate(q.names):
                state_rows.append({
                    "row_id": q.row_id, "suite": q.suite, "split": q.split, "arm": arm,
                    "condition": condition, "bridge_sign": bridge_sign, "seed": seed,
                    "candidate": cn, "candidate_index": ci,
                    "is_changed": q.is_changed, "names": list(q.names),
                    "score": float(sc[ci].detach().cpu()),
                    "target": int(target), "predicted": int(sc.argmax().item()),
                    "correct": int(sc.argmax().item() == target),
                    "inverted_target": bool(inv_target),
                    "mode": mode,
                })
        
        for c in comps:
            s1, _ = score_event_ha(model.event_cmp, vocab, c.event1, c.names, device, mode)
            s2, _ = score_event_ha(model.event_cmp, vocab, c.event2, c.names, device, mode)
            p1, p2 = F.softmax(s1, dim=0), F.softmax(s2, dim=0)
            psame = float((p1 * p2).sum().detach().cpu())
            for ci, cn in enumerate(c.names):
                comp_rows.append({
                    "row_id": c.row_id, "suite": c.suite, "split": c.split, "arm": arm,
                    "condition": condition, "bridge_sign": bridge_sign, "seed": seed,
                    "candidate": cn, "candidate_index": ci,
                    "names": list(c.names), "label": int(c.label),
                    "score1": float(s1[ci].detach().cpu()),
                    "score2": float(s2[ci].detach().cpu()),
                    "p_same": psame,
                    "predicted_same": int(psame >= 0.5),
                    "correct": int((psame >= 0.5) == c.label),
                    "mode": mode,
                })
    
    return state_rows, comp_rows


# ═════════════════════════════════════════════════════════════════════
# Main experimental loop
# ═════════════════════════════════════════════════════════════════════

def run_one(data_root, out, condition, arm, seed, bridge_sign, mode,
            eq_pretrain_epochs, eq_lr, eq_null_weight,
            epochs, lr, wd, emb_dim, hidden, char_dim, char_hidden,
            cmp_w, state_w, static_w, print_every, no_name_vocab,
            gpu, paired_models):
    """Run one (condition, bridge_sign, mode) cell."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
    
    ts, tc, es, ec, pe, counts = base.load_dataset(data_root, arm)
    
    # Build vocab
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
    
    model = copy.deepcopy(paired_models[condition])
    model.to(device)
    
    init_hash = hashlib.sha256()
    for p in model.parameters():
        init_hash.update(p.detach().cpu().numpy().tobytes())
    ihp = init_hash.hexdigest()[:16]
    print(f"Init hash {condition}/bs{'+' if bridge_sign==1 else '-'}1: {ihp}", flush=True)
    
    # Phase 1: Equality pretraining (if not oracle)
    eq_hist = None; ha_acc_before = None; ha_acc_after = None
    
    if mode != "oracle":
        examples = unique_train_events(ts, tc)
        
        # Measure hard-assignment accuracy BEFORE pretraining
        ha_acc_before, _ = eval_hard_assignment_accuracy(model, vocab, ts, tc, device)
        print(f"Hard-assignment before pretraining: {json.dumps(ha_acc_before)}", flush=True)
        
        if eq_pretrain_epochs > 0:
            print(f"Pretraining CharGRU for {eq_pretrain_epochs} epochs...", flush=True)
            eq_hist = pretrain_charenc(model, examples, device, eq_pretrain_epochs,
                                       eq_lr, eq_null_weight, print_every=max(1, eq_pretrain_epochs // 5))
        
        # Measure hard-assignment accuracy AFTER pretraining
        ha_acc_after_train, _ = eval_hard_assignment_accuracy(model, vocab, ts, tc, device)
        print(f"Hard-assignment after pretrain (train names): {json.dumps(ha_acc_after_train)}", flush=True)
        
        # Also check on eval names
        ha_acc_after_eval, _ = eval_hard_assignment_accuracy(model, vocab, es, ec, device)
        print(f"Hard-assignment after pretrain (eval names): {json.dumps(ha_acc_after_eval)}", flush=True)
        ha_acc_after = {"train": ha_acc_after_train, "eval": ha_acc_after_eval}
        
        # Freeze the CharGRU
        freeze_matcher(model)
        
        # Also freeze static scorer's CharGRU if present
        for p in list(model.static.trunk.char_enc.parameters()) + [model.static.trunk.neither_bias]:
            p.requires_grad_(False)
    
    # Phase 2: Relational training
    print(f"\n=== {mode} {condition}/bs={'+' if bridge_sign==1 else '-'}1/seed={seed} ===", flush=True)
    train_info = train_relational(model, vocab, ts, tc, device, bridge_sign,
                                   epochs, lr, wd, seed, cmp_w, state_w, static_w,
                                   mode, print_every)
    
    # Phase 3: Full evaluation
    with torch.no_grad():
        # Final train accuracy
        model.eval()
        cs, ns = 0, 0
        for q in ts:
            if q.is_changed:
                sc2, _ = score_event_ha(model.event_state, vocab, q.event, q.names, device, mode)
            else:
                sc2 = torch.stack([score_static_ha(model.static, vocab, q.prefix,
                    q.hypothesis_text_by_name[cn], q.names, cn, device, mode) for cn in q.names])
            target = base.get_target_idx(q)
            inv_target = q.inverted_target_by_name[q.names[1]]
            actual = target if bridge_sign == 1 else (1 - target if inv_target else target)
            cs += int(sc2.argmax().item() == actual); ns += 1
        
        cc_cmp, nc_cmp = 0, 0
        for c in tc:
            s1, _ = score_event_ha(model.event_cmp, vocab, c.event1, c.names, device, mode)
            s2, _ = score_event_ha(model.event_cmp, vocab, c.event2, c.names, device, mode)
            p1, p2 = F.softmax(s1, dim=0), F.softmax(s2, dim=0)
            psame = (p1 * p2).sum()
            cc_cmp += int((psame >= 0.5) == c.label); nc_cmp += 1
    
    final_train = {"train_state_acc": cs / ns if ns else 0,
                   "train_cmp_acc": cc_cmp / nc_cmp if nc_cmp else 0}
    
    # Eval predictions
    eval_sr, eval_cr = eval_predictions_ha(model, vocab, es, ec, device, condition, arm, seed, bridge_sign, mode)
    train_sr, train_cr = eval_predictions_ha(model, vocab, ts, tc, device, condition, arm, seed, bridge_sign, mode)
    
    # Also measure final hard-assignment accuracy on eval
    ha_acc_final = None
    if mode in ("hard", "soft"):
        ha_acc_final_eval, _ = eval_hard_assignment_accuracy(model, vocab, es, ec, device)
        ha_acc_final = ha_acc_final_eval
    
    # Matching accuracy from eval state rows (reuse base function)
    eval_match = base.matching_accuracy_from_state_rows(eval_sr)
    train_match = base.matching_accuracy_from_state_rows(train_sr)
    
    # Central evaluation metrics
    choices = base.state_choice_records(eval_sr)
    boths = base.pair_both_records(choices)
    central = base.compute_central(choices, boths, eval_cr, condition, arm, seed, bridge_sign)
    
    tag = f"{mode}_{condition}_bs{'+' if bridge_sign==1 else '-'}1_seed{seed}"
    rd = out / tag
    
    rec = {
        "condition": condition, "arm": arm, "seed": seed, "bridge_sign": bridge_sign,
        "mode": mode, "eq_pretrain_epochs": eq_pretrain_epochs,
        "epochs": epochs, "vocab_size": len(vocab.itos), "no_name_vocab": no_name_vocab,
        "init_hash_prefix": ihp, "counts": counts,
        "ha_accuracy_before_pretrain": ha_acc_before,
        "ha_accuracy_after_pretrain": ha_acc_after,
        "ha_accuracy_final": ha_acc_final,
        "equality_pretrain_history": eq_hist,
        "train_info": train_info,
        "final_train_metrics": final_train,
        "central_eval": central,
        "matching_accuracy": {"mode": mode, "eval": eval_match, "train": train_match},
    }
    
    write_json(rd / "result.json", rec)
    write_jsonl(rd / "state_predictions.jsonl", eval_sr)
    write_jsonl(rd / "comparison_predictions.jsonl", eval_cr)
    write_jsonl(rd / "train_state_predictions.jsonl", train_sr)
    
    return rec


def write_summary(out, results):
    lines = ["# research: Hard-assignment gauge-transport probe", "",
             "| mode | condition | bs | vocab | eq_pre | train_state | train_cmp | graph_same | unchanged | hh_closure | mixed | pair_both | ha_eval_both |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        ce = r.get("central_eval", {})
        ha = r.get("ha_accuracy_after_pretrain", {})
        ha_eval = ha.get("eval", {}) if isinstance(ha, dict) else {}
        ha_both = "n/a"
        for k in ("state_changed", "comparison"):
            if k in ha_eval:
                ha_both = f"{ha_eval[k].get('both_correct', 0):.3f}"
                break
        ft = r.get("final_train_metrics", {})
        lines.append(f"| {r.get('mode')} | {r.get('condition')} | {r.get('bridge_sign')} "
                     f"| {r.get('vocab_size')} | {r.get('eq_pretrain_epochs')} "
                     f"| {ft.get('train_state_acc',0):.3f} | {ft.get('train_cmp_acc',0):.3f} "
                     f"| {ce.get('graph_same',0):.3f} | {ce.get('unchanged',0):.3f} "
                     f"| {ce.get('hh_closure',0):.3f} | {ce.get('mixed_acc',0):.3f} "
                     f"| {ce.get('pair_both_graph_same',0):.3f} | {ha_both} |")
    
    write_json(out / "hard_assignment_summary.json", {"all_results": results})
    (out / "hard_assignment_summary.md").write_text("\n".join(lines) + "\n")


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
    ap.add_argument("--char-hidden", type=int, default=16)
    ap.add_argument("--cmp-weight", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=1.0)
    ap.add_argument("--static-weight", type=float, default=1.0)
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bridge-signs", nargs="+", type=int, choices=[1, -1], default=[1, -1])
    ap.add_argument("--no-name-vocab", action="store_true")
    ap.add_argument("--modes", nargs="+", default=["hard"], choices=["hard", "soft", "oracle"])
    ap.add_argument("--eq-pretrain-epochs", type=int, default=50)
    ap.add_argument("--eq-lr", type=float, default=5e-3)
    ap.add_argument("--eq-null-weight", type=float, default=1.0)
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
    
    print(f"research hard-assignment probe: modes={args.modes} conditions={args.conditions} "
          f"bridge_signs={args.bridge_signs} no_name_vocab={args.no_name_vocab} "
          f"vocab={len(vocab.itos)} eq_pre={args.eq_pretrain_epochs}", flush=True)
    
    results = []
    for mode in args.modes:
        for bs in args.bridge_signs:
            # Fresh paired models for each bridge sign
            paired = base.create_paired_span_models(
                len(vocab.itos), args.conditions, args.seed,
                args.emb_dim, args.hidden, args.char_dim, args.char_hidden)
            
            for cond in args.conditions:
                print(f"\n{'='*60}\n{mode} {cond}/bs={'+' if bs==1 else '-'}1\n{'='*60}", flush=True)
                rec = run_one(args.data_root, args.out, cond, args.arm, args.seed, bs, mode,
                              args.eq_pretrain_epochs, args.eq_lr, args.eq_null_weight,
                              args.epochs, args.lr, args.wd,
                              args.emb_dim, args.hidden, args.char_dim, args.char_hidden,
                              args.cmp_weight, args.state_weight, args.static_weight,
                              args.print_every, args.no_name_vocab, args.gpu, paired)
                results.append(rec)
                print(json.dumps({
                    "done": f"{mode}_{cond}_bs{'+' if bs==1 else '-'}1",
                    "train_state": rec.get("final_train_metrics", {}).get("train_state_acc"),
                    "train_cmp": rec.get("final_train_metrics", {}).get("train_cmp_acc"),
                    "graph_same": rec.get("central_eval", {}).get("graph_same"),
                    "unchanged": rec.get("central_eval", {}).get("unchanged"),
                    "hh_closure": rec.get("central_eval", {}).get("hh_closure"),
                    "mixed_acc": rec.get("central_eval", {}).get("mixed_acc"),
                }, indent=2), flush=True)
    
    write_summary(args.out, results)
    print(json.dumps({"status": "HARD_ASSIGNMENT_COMPLETE",
                      "summary": project_rel(args.out / "hard_assignment_summary.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
