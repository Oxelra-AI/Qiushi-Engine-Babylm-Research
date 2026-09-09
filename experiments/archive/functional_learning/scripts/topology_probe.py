#!/usr/bin/env python3
"""research functional_learning topology/gradient-path probe for the inherited gauge mechanism.

This local probe imports the representation_and_objectives controlled
raw-text relational harness but writes all new evidence under functional_learning.

Scientific purpose
------------------
Separate three information routes in the finite-supervision coordinate harness:
  1. representation formation by comparison-trained shared trunk features;
  2. graph/label-content mediated orientation through comparison edges;
  3. state-readout calibration from sparse h0/h2 anchors.

The two immediate interventions are:
  * detach_full_graph: state loss trains only the event-state head; the shared
    event trunk receives comparison gradients only. Because comparison labels are
    bridge-sign invariant, paired bs+1/bs-1 runs should have identical final
    event-trunk hashes if detach is implemented correctly.
  * disconn_delete / disconn_adversarial / disconn_shuffle: modify h1 incident
    comparison edges while keeping h0-h2 and h2-h3 correct, so h3 is the
    correctly connected control and h1 tests missing/wrong/noisy graph content.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

A01_SCRIPT_DIR = Path("experiments/archive/representation_and_objectives/training/scripts")
sys.path.insert(0, str(A01_SCRIPT_DIR))

import dualpos_learned_equality_gauge_probe as s299  # noqa: E402
import posalign_gauge_probe as p298  # noqa: E402
import raw_span_discovery_probe as base  # noqa: E402

DEFAULT_OUT = Path("experiments/archive/functional_learning/data/topology_probe")
TARGET_H1_EDGES = {
    frozenset(("h0_dax", "h1_mep")),
    frozenset(("h1_mep", "h2_norp")),
}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def module_hash(module: torch.nn.Module, n: int = 16) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in module.named_parameters():
            h.update(name.encode("utf-8"))
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:n]


def edge_key(c) -> frozenset:
    return frozenset((c.relation1, c.relation2))


def is_h1_target_edge(c) -> bool:
    return edge_key(c) in TARGET_H1_EDGES


def copy_comp_with_label(c, new_label: bool):
    c_new = copy.copy(c)
    c_new.label = bool(new_label)
    return c_new


def make_training_comparisons(tc, intervention: str, seed: int, shuffle_seed: int) -> Tuple[List, dict]:
    """Return modified training comparison objects and a provenance record."""
    target = [c for c in tc if is_h1_target_edge(c)]
    control = [c for c in tc if not is_h1_target_edge(c)]
    before_edges = Counter(f"{c.relation1}->{c.relation2}" for c in tc)
    before_labels = Counter(str(bool(c.label)) for c in tc)

    if intervention in {"full_graph", "detach_full_graph"}:
        out = list(tc)
        action = "unchanged"
    elif intervention == "disconn_delete":
        out = control
        action = "deleted_h1_incident_edges"
    elif intervention == "disconn_adversarial":
        out = [copy_comp_with_label(c, not c.label) if is_h1_target_edge(c) else c for c in tc]
        action = "inverted_h1_incident_edge_labels"
    elif intervention == "disconn_shuffle":
        rng = random.Random(seed * 1000003 + shuffle_seed + 404)
        target_labels = [bool(c.label) for c in target]
        rng.shuffle(target_labels)
        label_iter = iter(target_labels)
        out = [copy_comp_with_label(c, next(label_iter)) if is_h1_target_edge(c) else c for c in tc]
        action = "shuffled_h1_incident_edge_labels"
    else:
        raise ValueError(intervention)

    after_edges = Counter(f"{c.relation1}->{c.relation2}" for c in out)
    after_labels = Counter(str(bool(c.label)) for c in out)
    changed_labels = 0
    retained_target = 0
    correct_target_after = 0
    # For copied/inverted/shuffled rows, zip is valid because row order is preserved except delete.
    by_row = {getattr(c, "row_id", i): c for i, c in enumerate(tc)}
    for j, c in enumerate(out):
        orig = by_row.get(getattr(c, "row_id", j))
        if orig is not None and bool(orig.label) != bool(c.label):
            changed_labels += 1
        if is_h1_target_edge(c):
            retained_target += 1
            if orig is not None and bool(orig.label) == bool(c.label):
                correct_target_after += 1

    provenance = {
        "intervention": intervention,
        "action": action,
        "target_edges_unordered": [sorted(list(x)) for x in TARGET_H1_EDGES],
        "original_total": len(tc),
        "modified_total": len(out),
        "target_rows_original": len(target),
        "target_rows_retained": retained_target,
        "target_rows_with_original_label_after_modification": correct_target_after,
        "changed_labels_total": changed_labels,
        "shuffle_seed": shuffle_seed if intervention == "disconn_shuffle" else None,
        "original_edge_counts": dict(sorted(before_edges.items())),
        "modified_edge_counts": dict(sorted(after_edges.items())),
        "original_label_balance": dict(sorted(before_labels.items())),
        "modified_label_balance": dict(sorted(after_labels.items())),
    }
    return out, provenance


def score_event_pa_for_state(scorer, vocab, text: str, names: Tuple[str, str], device, hard: bool,
                             detach_trunk: bool):
    """Score changed-state query; optionally detach the event trunk before state head."""
    if not detach_trunk:
        return p298.score_event_pa(scorer, vocab, text, names, device, hard)

    tok_ids_str, char_forms, tcp, _, _, isp = p298.prepare_posalign_inputs(text, names[0], names[1], device)
    token_ids = vocab.encode(tok_ids_str)
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
    mask = torch.ones((1, len(token_ids)), dtype=torch.bool, device=device)

    scores = []
    match_info = []
    for ci, cand in enumerate(names):
        other = names[1 - ci]
        cc = torch.tensor(p298.name_to_padded_chars(cand), dtype=torch.long, device=device)
        oc = torch.tensor(p298.name_to_padded_chars(other), dtype=torch.long, device=device)
        h, _probs, assign = scorer.trunk.forward_matched(ids_t, mask, tcp, cc, oc, isp, hard)
        s = scorer.head(h.detach())
        scores.append(s.squeeze())

        cand_lc = cand.lower(); other_lc = other.lower()
        cand_pos = next((i for i, f in enumerate(char_forms) if f == cand_lc), -1)
        other_pos = next((i for i, f in enumerate(char_forms) if f == other_lc), -1)
        rec = {"cand_name": cand, "other_name": other}
        if hard:
            rec["assigned_cand"] = assign[0]
            rec["assigned_other"] = assign[1]
            rec["cand_correct"] = (assign[0] == cand_pos)
            rec["other_correct"] = (assign[1] == other_pos)
            rec["both_correct"] = rec["cand_correct"] and rec["other_correct"]
        match_info.append(rec)
    return torch.stack(scores), match_info


def changed_state_scores(model, vocab, q, device, hard: bool, detach_trunk: bool):
    return score_event_pa_for_state(model.event_state, vocab, q.event, q.names, device, hard, detach_trunk)[0]


def state_scores_for_training(model, vocab, q, device, hard: bool, detach_trunk: bool):
    if q.is_changed:
        return changed_state_scores(model, vocab, q, device, hard, detach_trunk)
    return torch.stack([
        p298.score_static_pa(model.static, vocab, q.prefix, q.hypothesis_text_by_name[cn], q.names, cn, device, hard)
        for cn in q.names
    ])


def verify_state_detach_gradient(model, vocab, ts, device, bridge_sign: int, hard: bool) -> dict:
    """Measure whether one detached state loss sends gradient to the event trunk."""
    model.zero_grad(set_to_none=True)
    q = next(q for q in ts if q.is_changed and q.is_direct_anchor)
    sc = state_scores_for_training(model, vocab, q, device, hard, detach_trunk=True)
    if bridge_sign == -1 and q.is_direct_anchor:
        sc = sc.flip(0)
    target = base.get_target_idx(q)
    loss = F.cross_entropy(sc.view(1, -1), torch.tensor([target], dtype=torch.long, device=device))
    loss.backward()
    trunk_max = 0.0
    trunk_nonzero = 0
    trunk_params = 0
    for p in model.event_state.trunk.parameters():
        trunk_params += 1
        if p.grad is not None:
            gm = float(p.grad.detach().abs().max().cpu())
            trunk_max = max(trunk_max, gm)
            trunk_nonzero += int(gm > 0.0)
    state_head_max = 0.0
    for p in model.event_state.head.parameters():
        if p.grad is not None:
            state_head_max = max(state_head_max, float(p.grad.detach().abs().max().cpu()))
    model.zero_grad(set_to_none=True)
    return {
        "checked_query_key": q.key,
        "loss": float(loss.detach().cpu()),
        "event_trunk_max_abs_grad_from_detached_state_loss": trunk_max,
        "event_trunk_params_with_grad": trunk_nonzero,
        "event_trunk_param_tensors": trunk_params,
        "event_state_head_max_abs_grad": state_head_max,
    }


def quick_train_metrics(model, vocab, states, comps_train, comps_original, device, bridge_sign: int,
                        hard: bool, detach_trunk_for_state_eval: bool) -> dict:
    model.eval()
    ns = cs = nch = cch = nun = cun = 0
    with torch.no_grad():
        for q in states:
            sc = state_scores_for_training(model, vocab, q, device, hard, detach_trunk_for_state_eval)
            if bridge_sign == -1 and q.is_direct_anchor:
                sc_eff = sc.flip(0)
            else:
                sc_eff = sc
            target = base.get_target_idx(q)
            ok = int(sc_eff.argmax().item() == target)
            ns += 1; cs += ok
            if q.is_changed:
                nch += 1; cch += ok
            else:
                nun += 1; cun += ok

        def comp_acc(comps):
            nc = cc = 0
            for c in comps:
                s1, _ = p298.score_event_pa(model.event_cmp, vocab, c.event1, c.names, device, hard)
                s2, _ = p298.score_event_pa(model.event_cmp, vocab, c.event2, c.names, device, hard)
                psame = (F.softmax(s1, dim=0) * F.softmax(s2, dim=0)).sum()
                cc += int(bool(psame >= 0.5) == bool(c.label)); nc += 1
            return cc / nc if nc else math.nan

    return {
        "train_state": cs / ns if ns else math.nan,
        "train_changed": cch / nch if nch else math.nan,
        "train_unchanged": cun / nun if nun else math.nan,
        "train_cmp_modified": comp_acc(comps_train),
        "train_cmp_original_labels": comp_acc(comps_original),
    }


def train_relational_topology(model, vocab, ts, tc_train, tc_original, device, bridge_sign: int,
                              epochs: int, lr: float, wd: float, seed: int, cmp_w: float,
                              state_w: float, static_w: float, print_every: int, hard: bool,
                              detach_state_trunk: bool) -> dict:
    model.to(device)
    seen = set(); all_params = []; clip_groups = []

    def add_params(params):
        gp = []
        for p in params:
            if p.requires_grad:
                pid = id(p)
                if pid not in seen:
                    seen.add(pid); all_params.append(p); gp.append(p)
        if gp:
            clip_groups.append(gp)

    if detach_state_trunk:
        # Keep trunk clipping independent from bridge-sign-dependent state-head
        # gradients. Otherwise torch.nn.utils.clip_grad_norm_ over a mixed
        # trunk+state-head group would scale comparison-trunk gradients by a
        # bridge-sign-dependent state-head norm, breaking the expected identical
        # DETACH trunk hash.
        add_params(model.event_state.trunk.parameters())
        add_params(model.event_state.head.parameters())
        if model.event_cmp is not model.event_state:
            if model.event_cmp.trunk is not model.event_state.trunk:
                add_params(model.event_cmp.trunk.parameters())
            add_params(model.event_cmp.head.parameters())
        add_params(model.static.parameters())
    else:
        add_params(model.event_state.parameters())
        if model.event_cmp is not model.event_state:
            add_params(model.event_cmp.parameters())
        add_params(model.static.parameters())
    if not all_params:
        return {"error": "no trainable params"}

    opt = torch.optim.AdamW([{"params": all_params}], lr=lr, weight_decay=wd)
    rng = random.Random(seed)
    history = []; t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        states = list(ts); comps = list(tc_train)
        rng.shuffle(states); rng.shuffle(comps)
        opt.zero_grad(set_to_none=True)
        losses = []

        for q in states:
            sc = state_scores_for_training(model, vocab, q, device, hard, detach_state_trunk)
            target = base.get_target_idx(q)
            if bridge_sign == -1 and q.is_direct_anchor:
                sc_eff = sc.flip(0)
            else:
                sc_eff = sc
            loss = F.cross_entropy(sc_eff.view(1, -1), torch.tensor([target], dtype=torch.long, device=device))
            losses.append((state_w if q.is_changed else static_w) * loss)

        for c in comps:
            s1, _ = p298.score_event_pa(model.event_cmp, vocab, c.event1, c.names, device, hard)
            s2, _ = p298.score_event_pa(model.event_cmp, vocab, c.event2, c.names, device, hard)
            p1, p2 = F.softmax(s1, dim=0), F.softmax(s2, dim=0)
            psame = (p1 * p2).sum().clamp(1e-6, 1 - 1e-6)
            y = torch.tensor(float(c.label), dtype=torch.float32, device=psame.device)
            losses.append(cmp_w * (-(y * torch.log(psame) + (1 - y) * torch.log(1 - psame))))

        total = torch.stack(losses).mean()
        total.backward()
        for gp in clip_groups:
            torch.nn.utils.clip_grad_norm_(gp, 5.0)
        opt.step()

        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            met = quick_train_metrics(model, vocab, ts, tc_train, tc_original, device, bridge_sign, hard, detach_state_trunk)
            rec = {"epoch": ep, "loss": float(total.detach().cpu()), **{k: float(v) for k, v in met.items()}}
            history.append(rec)
            if print_every:
                print(json.dumps(rec, sort_keys=True), flush=True)

    return {"elapsed_seconds": time.time() - t0, "history": history}


def comparison_edge_summary(comp_rows: List[dict]) -> dict:
    by_edge = defaultdict(list)
    for r in comp_rows:
        by_edge[f"{r.get('relation1')}->{r.get('relation2')}"].append(r)
    out = {}
    for edge, rows in sorted(by_edge.items()):
        acc = sum(int(r.get("correct", 0)) for r in rows) / len(rows) if rows else math.nan
        margins = [float(r.get("signed_margin", 0.0)) for r in rows]
        out[edge] = {
            "n": len(rows),
            "acc": acc,
            "mean_signed_margin": float(np.mean(margins)) if margins else math.nan,
        }
    return out


def graph_manifest(ts, tc_original, tc_train, es, ec, modification: dict) -> dict:
    def state_counts(rows):
        return {
            "total": len(rows),
            "by_relation": dict(sorted(Counter(q.relation for q in rows).items())),
            "n_changed": sum(1 for q in rows if q.is_changed),
            "n_direct_anchor": sum(1 for q in rows if q.is_direct_anchor),
            "h1h3_state_rows": sum(1 for q in rows if q.relation in {"h1_mep", "h3_ziv"}),
        }

    def comp_counts(rows):
        return {
            "total": len(rows),
            "edge_types": dict(sorted(Counter(f"{c.relation1}->{c.relation2}" for c in rows).items())),
            "label_balance": dict(sorted(Counter(str(bool(c.label)) for c in rows).items())),
        }

    return {
        "train_state": state_counts(ts),
        "train_comparison_original": comp_counts(tc_original),
        "train_comparison_modified": comp_counts(tc_train),
        "eval_state": state_counts(es),
        "eval_comparison": comp_counts(ec),
        "modification": modification,
    }


def run_one(args) -> dict:
    s299.patch_dual_matcher()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    ts, tc_original, es, ec, _pe, counts = base.load_dataset(args.data_root, args.arm)
    tc_train, modification = make_training_comparisons(tc_original, args.intervention, args.seed, args.shuffle_seed)
    detach_state_trunk = bool(args.intervention == "detach_full_graph" or args.detach_state_trunk)

    train_name_set = set()
    for q in ts:
        for n in q.names:
            train_name_set.add(n)
    for c in tc_original:
        for n in c.names:
            train_name_set.add(n)
    vocab = base.RawVocab()
    if args.no_name_vocab:
        base.collect_raw_vocab(vocab, ts, tc_original, exclude_names=train_name_set)
    else:
        base.collect_raw_vocab(vocab, ts, tc_original)

    if args.char_pair_mode == "none":
        char_ids = []
    elif args.char_pair_mode == "train_alphabet":
        char_ids = s299.char_ids_from_names(s299.names_from_training(ts, tc_original))
    else:
        char_ids = list(range(1, 27))

    paired = p298.create_paired_posalign(len(vocab.itos), [args.condition], args.seed,
                                         args.emb_dim, args.hidden, args.char_dim)
    model = copy.deepcopy(paired[args.condition]).to(device)

    init_hashes = {
        "model": module_hash(model, 16),
        "event_trunk": module_hash(model.event_state.trunk, 16),
        "event_state_head": module_hash(model.event_state.head, 16),
        "event_cmp_head": module_hash(model.event_cmp.head, 16),
        "static": module_hash(model.static, 16),
    }
    print(json.dumps({
        "run": "topology_probe",
        "intervention": args.intervention,
        "condition": args.condition,
        "bridge_sign": args.bridge_sign,
        "seed": args.seed,
        "gpu": args.gpu,
        "device": str(device),
        "detach_state_trunk": detach_state_trunk,
        "tc_original": len(tc_original),
        "tc_train": len(tc_train),
        "vocab": len(vocab.itos),
        "init_hashes": init_hashes,
    }, indent=2), flush=True)

    match_before_train, _ = p298.eval_matching_accuracy(model, vocab, ts, tc_original, device)
    match_before_eval, _ = p298.eval_matching_accuracy(model, vocab, es, ec, device)
    pre_hist = []
    if args.char_pair_mode != "none" and args.char_pair_epochs > 0:
        pre_hist = s299.pretrain_char_pairs(model, char_ids, device, args.char_pair_epochs,
                                            args.char_pair_lr, args.print_every)
    match_after_train, _ = p298.eval_matching_accuracy(model, vocab, ts, tc_original, device)
    match_after_eval, _ = p298.eval_matching_accuracy(model, vocab, es, ec, device)
    pretrain_hashes = {
        "model": module_hash(model, 16),
        "event_trunk": module_hash(model.event_state.trunk, 16),
        "event_state_head": module_hash(model.event_state.head, 16),
        "event_cmp_head": module_hash(model.event_cmp.head, 16),
        "static": module_hash(model.static, 16),
    }
    print(json.dumps({
        "matching": {
            "before_train": match_before_train,
            "before_eval": match_before_eval,
            "after_train": match_after_train,
            "after_eval": match_after_eval,
        },
        "pretrain_hashes": pretrain_hashes,
    }, indent=2), flush=True)

    p298.freeze_matcher(model)
    detach_grad_check = None
    if detach_state_trunk:
        detach_grad_check = verify_state_detach_gradient(model, vocab, ts, device, args.bridge_sign, hard=True)
        print(json.dumps({"detach_gradient_check": detach_grad_check}, indent=2), flush=True)

    train_info = train_relational_topology(
        model, vocab, ts, tc_train, tc_original, device, args.bridge_sign,
        args.epochs, args.lr, args.wd, args.seed, args.cmp_weight, args.state_weight,
        args.static_weight, args.print_every, hard=True, detach_state_trunk=detach_state_trunk)

    eval_sr, eval_cr = p298.eval_predictions(model, vocab, es, ec, device,
                                             args.condition, args.arm, args.seed,
                                             args.bridge_sign, hard=True)
    train_sr_orig, train_cr_orig = p298.eval_predictions(model, vocab, ts, tc_original, device,
                                                         args.condition, args.arm, args.seed,
                                                         args.bridge_sign, hard=True)
    _train_sr_mod, train_cr_mod = p298.eval_predictions(model, vocab, ts, tc_train, device,
                                                        args.condition, args.arm, args.seed,
                                                        args.bridge_sign, hard=True)
    eval_ma_final, _ = p298.eval_matching_accuracy(model, vocab, es, ec, device)

    choices = base.state_choice_records(eval_sr)
    boths = base.pair_both_records(choices)
    central = base.compute_central(choices, boths, eval_cr, args.condition, args.arm, args.seed,
                                   args.bridge_sign)

    final_hashes = {
        "model": module_hash(model, 16),
        "event_trunk": module_hash(model.event_state.trunk, 16),
        "event_state_head": module_hash(model.event_state.head, 16),
        "event_cmp_head": module_hash(model.event_cmp.head, 16),
        "static": module_hash(model.static, 16),
    }

    manifest = graph_manifest(ts, tc_original, tc_train, es, ec, modification)
    rec = {
        "intervention": args.intervention,
        "condition": args.condition,
        "arm": args.arm,
        "seed": args.seed,
        "bridge_sign": args.bridge_sign,
        "mode": "dual_position_learned_equality_hard_step004_topology",
        "detach_state_trunk": detach_state_trunk,
        "char_pair_mode": args.char_pair_mode,
        "char_pair_epochs": args.char_pair_epochs,
        "char_ids": list(map(int, char_ids)),
        "epochs": args.epochs,
        "vocab_size": len(vocab.itos),
        "no_name_vocab": args.no_name_vocab,
        "hashes": {"init": init_hashes, "after_pretrain": pretrain_hashes, "final": final_hashes},
        "matching_before": {"train": match_before_train, "eval": match_before_eval},
        "matching_after_pretrain": {"train": match_after_train, "eval": match_after_eval},
        "matching_final": eval_ma_final,
        "pretrain_history": pre_hist,
        "detach_gradient_check": detach_grad_check,
        "train_info": train_info,
        "central_eval": central,
        "comparison_edge_summary_eval_original": comparison_edge_summary(eval_cr),
        "comparison_edge_summary_train_original_labels": comparison_edge_summary(train_cr_orig),
        "comparison_edge_summary_train_modified_labels": comparison_edge_summary(train_cr_mod),
        "manifest": manifest,
        "counts": counts,
    }

    sign = "+1" if args.bridge_sign == 1 else "-1"
    tag = f"{args.intervention}_{args.condition}_bs{sign}_seed{args.seed}"
    rd = args.out / tag
    write_json(rd / "result.json", rec)
    write_json(rd / "manifest.json", manifest)
    write_jsonl(rd / "state_predictions.jsonl", eval_sr)
    write_jsonl(rd / "comparison_predictions_original_eval.jsonl", eval_cr)
    write_jsonl(rd / "train_state_predictions.jsonl", train_sr_orig)
    write_jsonl(rd / "train_comparison_predictions_original_labels.jsonl", train_cr_orig)
    write_jsonl(rd / "train_comparison_predictions_modified_labels.jsonl", train_cr_mod)

    last_hist = train_info.get("history", [{}])[-1] if train_info.get("history") else {}
    lines = [
        f"# research topology probe: {args.intervention} bs{sign}",
        "",
        f"- condition: `{args.condition}`",
        f"- seed: `{args.seed}`",
        f"- bridge_sign: `{args.bridge_sign}`",
        f"- detach_state_trunk: `{detach_state_trunk}`",
        f"- training comparisons: {len(tc_train)} / original {len(tc_original)}",
        f"- matching after equality pretrain eval: {match_after_eval:.6f}",
        f"- final event trunk hash: `{final_hashes['event_trunk']}`",
        f"- final event_state_head hash: `{final_hashes['event_state_head']}`",
        f"- final event_cmp_head hash: `{final_hashes['event_cmp_head']}`",
        "",
        "## Final train metrics",
        "",
        "```json",
        json.dumps(last_hist, indent=2, sort_keys=True),
        "```",
        "",
        "## Central eval",
        "",
        "```json",
        json.dumps(central, indent=2, sort_keys=True),
        "```",
        "",
        "## Modified comparison provenance",
        "",
        "```json",
        json.dumps(modification, indent=2, sort_keys=True),
        "```",
    ]
    if detach_grad_check is not None:
        lines.extend(["", "## Detach gradient check", "", "```json",
                      json.dumps(detach_grad_check, indent=2, sort_keys=True), "```"])
    (rd / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(args.out / f"{tag}_run_result.json", {
        "status": "TOPOLOGY_RUN_COMPLETE",
        "run_dir": str(rd),
        "summary": str(rd / "summary.md"),
        "central_eval": central,
        "final_hashes": final_hashes,
        "train_last": last_hist,
        "modification": modification,
    })

    print(json.dumps({
        "status": "TOPOLOGY_RUN_COMPLETE",
        "run_dir": str(rd),
        "summary": str(rd / "summary.md"),
        "central_eval": central,
        "final_event_trunk_hash": final_hashes["event_trunk"],
        "train_last": last_hist,
    }, indent=2, sort_keys=True), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervention", choices=[
        "full_graph", "detach_full_graph", "disconn_delete", "disconn_adversarial", "disconn_shuffle"],
        required=True)
    ap.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arm", default=base.DEFAULT_ARM)
    ap.add_argument("--condition", default="shared_trunk")
    ap.add_argument("--bridge-sign", type=int, choices=[1, -1], required=True)
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--char-pair-mode", choices=["none", "train_alphabet", "full_alphabet"], default="full_alphabet")
    ap.add_argument("--char-pair-epochs", type=int, default=60)
    ap.add_argument("--char-pair-lr", type=float, default=5e-2)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--emb-dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--char-dim", type=int, default=16)
    ap.add_argument("--cmp-weight", type=float, default=1.0)
    ap.add_argument("--state-weight", type=float, default=1.0)
    ap.add_argument("--static-weight", type=float, default=1.0)
    ap.add_argument("--print-every", type=int, default=45)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-name-vocab", action="store_true", default=True)
    ap.add_argument("--include-name-vocab", dest="no_name_vocab", action="store_false")
    ap.add_argument("--detach-state-trunk", action="store_true", help="debug/ablation: detach even outside detach_full_graph")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    run_one(args)


if __name__ == "__main__":
    main()
