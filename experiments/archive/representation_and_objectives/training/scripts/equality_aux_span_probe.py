#!/usr/bin/env python3
"""research: equality-auxiliary span-routing probe.

Scientific purpose
------------------
research no-name-vocabulary training showed that removing train-name word
embeddings does not by itself make the all-token CharGRU gate learn robust
candidate/other/neither routing. Hard exact raw-token localization, however,
restored the shared-trunk gauge fingerprint. This script tests the next causal
fork: whether explicit sparse equality/null routing evidence can establish a
reusable matcher before local relational losses absorb the problem.

The protected condition pretrains only matcher parameters (CharGRU + neither
bias) on token/query equality labels from training texts, freezes those matcher
parameters, and then runs the original relational training. It is not a natural
span-discovery result; it is a controlled intervention on the identity-routing
interface.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, random, sys, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Import the research raw-span harness after the research hard-oracle repair.
from experiments.archive.representation_and_objectives.ai_lab.scripts import raw_span_discovery_probe as base

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_OUT = PROJECT / "data/equality_aux_span"


def project_rel(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except Exception:
        return str(p)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def collect_train_names(states, comps) -> set:
    names = set()
    for q in states:
        for n in q.names:
            names.add(n)
    for c in comps:
        for n in c.names:
            names.add(n)
    return names


def build_vocab(states, comps, no_name_vocab: bool):
    vocab = base.RawVocab()
    if no_name_vocab:
        base.collect_raw_vocab(vocab, states, comps, exclude_names=collect_train_names(states, comps))
    else:
        base.collect_raw_vocab(vocab, states, comps)
    return vocab


def unique_train_event_examples(states, comps) -> List[Tuple[str, Tuple[str, str], str]]:
    """Return unique changed-event examples for equality pretraining.

    Each item is (kind, text, names). `kind` is used only for provenance.
    Static examples are intentionally omitted by default: the central bottleneck
    is changed-event and comparison routing, while static has a separate scorer
    and can be trained by the relational objective once name routing exists.
    """
    seen = set(); out = []
    for q in states:
        if q.is_changed:
            k = ("state", q.event, tuple(q.names))
            if k not in seen:
                seen.add(k); out.append(k)
    for c in comps:
        for tag, ev in [("cmp1", c.event1), ("cmp2", c.event2)]:
            k = (tag, ev, tuple(c.names))
            if k not in seen:
                seen.add(k); out.append(k)
    return out


def gate_logits_for_forms(trunk, char_forms: Sequence[str], cand: str, other: str, device):
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
    special = torch.tensor([not bool(f) or f == "<hyp>" for f in char_forms], dtype=torch.bool, device=device)
    logits[special, 0] = -1e6
    logits[special, 1] = -1e6
    targets = []
    cand_lc = cand.lower(); other_lc = other.lower()
    for f in char_forms:
        if f == cand_lc:
            targets.append(0)
        elif f == other_lc:
            targets.append(1)
        else:
            targets.append(2)
    return logits, torch.tensor(targets, dtype=torch.long, device=device), special


def equality_loss_for_text(trunk, text: str, names: Tuple[str, str], device,
                           null_weight: float = 1.0) -> torch.Tensor:
    _, char_forms = base.raw_tokenize(text)
    losses = []
    for cand in names:
        other = names[1] if cand == names[0] else names[0]
        logits, targets, special = gate_logits_for_forms(trunk, char_forms, cand, other, device)
        ce = F.cross_entropy(logits, targets, reduction="none")
        name_mask = targets != 2
        null_mask = (targets == 2) & (~special)
        # Balance the rare name-token routing and the abundant null rejection.
        parts = []
        if bool(name_mask.any()):
            parts.append(ce[name_mask].mean())
        if bool(null_mask.any()):
            parts.append(null_weight * ce[null_mask].mean())
        if parts:
            losses.append(torch.stack(parts).mean())
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


def unique_matcher_trunks(model, include_static: bool = False):
    trunks = []
    seen = set()
    for scorer in [model.event_state, model.event_cmp] + ([model.static] if include_static else []):
        tr = scorer.trunk
        if id(tr) not in seen:
            trunks.append(tr); seen.add(id(tr))
    return trunks


def matcher_parameters(model, include_static: bool = False):
    params = []
    seen = set()
    for tr in unique_matcher_trunks(model, include_static=include_static):
        for p in list(tr.char_enc.parameters()) + [tr.neither_bias]:
            if id(p) not in seen:
                params.append(p); seen.add(id(p))
    return params


def set_matcher_requires_grad(model, requires_grad: bool, include_static: bool = False) -> None:
    for p in matcher_parameters(model, include_static=include_static):
        p.requires_grad_(requires_grad)


def equality_eval(model, examples, device, include_static: bool = False, max_examples: int | None = None):
    model.eval()
    trunks = unique_matcher_trunks(model, include_static=include_static)
    rows = []
    with torch.no_grad():
        subset = examples[:max_examples] if max_examples else examples
        for tr_idx, tr in enumerate(trunks):
            for kind, text, names in subset:
                _, forms = base.raw_tokenize(text)
                for cand in names:
                    other = names[1] if cand == names[0] else names[0]
                    logits, targets, special = gate_logits_for_forms(tr, forms, cand, other, device)
                    probs = torch.softmax(logits, dim=-1).detach().cpu()
                    cand_lc = cand.lower(); other_lc = other.lower()
                    rec = {"trunk": tr_idx, "kind": kind, "cand": cand, "other": other}
                    for i, f in enumerate(forms):
                        if f == cand_lc:
                            rec.update({"cand_p_cand": float(probs[i,0]),
                                        "cand_p_other": float(probs[i,1]),
                                        "cand_p_neither": float(probs[i,2])})
                        elif f == other_lc:
                            rec.update({"other_p_cand": float(probs[i,0]),
                                        "other_p_other": float(probs[i,1]),
                                        "other_p_neither": float(probs[i,2])})
                    rows.append(rec)
    def avg(vals):
        vals = list(vals)
        return sum(vals)/len(vals) if vals else math.nan
    summary = {
        "n": len(rows),
        "cand_gt_other": avg(1.0 if r.get("cand_p_cand",0)>r.get("cand_p_other",0) else 0.0 for r in rows),
        "other_gt_cand": avg(1.0 if r.get("other_p_other",0)>r.get("other_p_cand",0) else 0.0 for r in rows),
        "cand_wins_gate": avg(1.0 if r.get("cand_p_cand",0)>max(r.get("cand_p_other",0),r.get("cand_p_neither",0)) else 0.0 for r in rows),
        "other_wins_gate": avg(1.0 if r.get("other_p_other",0)>max(r.get("other_p_cand",0),r.get("other_p_neither",0)) else 0.0 for r in rows),
        "both_wins_gate": avg(1.0 if (r.get("cand_p_cand",0)>max(r.get("cand_p_other",0),r.get("cand_p_neither",0)) and r.get("other_p_other",0)>max(r.get("other_p_cand",0),r.get("other_p_neither",0))) else 0.0 for r in rows),
        "cand_above_half": avg(1.0 if r.get("cand_p_cand",0)>0.5 else 0.0 for r in rows),
        "other_above_half": avg(1.0 if r.get("other_p_other",0)>0.5 else 0.0 for r in rows),
        "mean_cand_p_neither": avg(float(r.get("cand_p_neither",0)) for r in rows),
        "mean_other_p_neither": avg(float(r.get("other_p_neither",0)) for r in rows),
    }
    return summary


def pretrain_equality(model, examples, device, epochs: int, lr: float, null_weight: float,
                      include_static: bool, print_every: int = 0):
    params = matcher_parameters(model, include_static=include_static)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(12345)
    hist = []
    t0 = time.time()
    model.to(device)
    for ep in range(1, epochs+1):
        model.train()
        exs = list(examples)
        rng.shuffle(exs)
        opt.zero_grad(set_to_none=True)
        losses = []
        trunks = unique_matcher_trunks(model, include_static=include_static)
        for tr in trunks:
            for _, text, names in exs:
                losses.append(equality_loss_for_text(tr, text, names, device, null_weight=null_weight))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            ev = equality_eval(model, examples, device, include_static=include_static, max_examples=min(128, len(examples)))
            rec = {"epoch": ep, "loss": float(loss.detach().cpu()), **{k: float(v) for k, v in ev.items() if isinstance(v, (int,float))}}
            hist.append(rec)
            if print_every:
                print(json.dumps({"eq_pretrain": rec}, sort_keys=True), flush=True)
    return {"elapsed_seconds": time.time()-t0, "history": hist}


def run_one(args, condition: str, bridge_sign: int, paired_models):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    ts, tc, es, ec, pe, counts = base.load_dataset(args.data_root, args.arm)
    vocab = build_vocab(ts, tc, args.no_name_vocab)
    model = copy.deepcopy(paired_models[condition])
    model.to(device)
    examples = unique_train_event_examples(ts, tc)
    init_hash = hashlib.sha256()
    for p in model.parameters():
        init_hash.update(p.detach().cpu().numpy().tobytes())
    ihp = init_hash.hexdigest()[:16]

    pre_info = None
    pre_eval_before = equality_eval(model, examples, device, include_static=args.eq_include_static, max_examples=min(128, len(examples)))
    if args.eq_pretrain_epochs > 0:
        pre_info = pretrain_equality(model, examples, device, args.eq_pretrain_epochs,
                                     args.eq_lr, args.eq_null_weight,
                                     include_static=args.eq_include_static,
                                     print_every=args.eq_print_every)
    pre_eval_after = equality_eval(model, examples, device, include_static=args.eq_include_static, max_examples=min(128, len(examples)))
    if args.freeze_matcher:
        set_matcher_requires_grad(model, False, include_static=args.eq_include_static)

    train_info = base.train_one_span(model, vocab, ts, tc, device, bridge_sign,
                                     args.epochs, args.lr, args.wd, args.seed,
                                     args.cmp_weight, args.state_weight, args.static_weight,
                                     args.print_every, matcher_mode="learned")
    post_eval = equality_eval(model, examples, device, include_static=args.eq_include_static, max_examples=min(128, len(examples)))
    with torch.no_grad():
        ft = base.quick_train_span(model, vocab, ts, tc, device, bridge_sign, "learned")
    train_sr, train_cr = base.eval_predictions_span(model, vocab, ts, tc, device, condition, args.arm, args.seed, bridge_sign, "learned")
    eval_sr, eval_cr = base.eval_predictions_span(model, vocab, es, ec, device, condition, args.arm, args.seed, bridge_sign, "learned")
    eval_match = base.matching_accuracy_from_state_rows(eval_sr)
    train_match = base.matching_accuracy_from_state_rows(train_sr)
    choices = base.state_choice_records(eval_sr)
    boths = base.pair_both_records(choices)
    central = base.compute_central(choices, boths, eval_cr, condition, args.arm, args.seed, bridge_sign)

    tag = f"eqaux_{'freeze' if args.freeze_matcher else 'joint'}_{condition}_bs{'+' if bridge_sign==1 else '-'}1_seed{args.seed}"
    rd = args.out / tag
    rec = {
        "condition": condition,
        "arm": args.arm,
        "seed": args.seed,
        "bridge_sign": bridge_sign,
        "matcher": "learned_eq_aux",
        "eq_pretrain_epochs": args.eq_pretrain_epochs,
        "freeze_matcher": args.freeze_matcher,
        "eq_lr": args.eq_lr,
        "eq_null_weight": args.eq_null_weight,
        "eq_include_static": args.eq_include_static,
        "epochs": args.epochs,
        "vocab_size": len(vocab.itos),
        "no_name_vocab": args.no_name_vocab,
        "init_hash_prefix": ihp,
        "counts": counts,
        "equality_eval_before": pre_eval_before,
        "equality_pretrain_info": pre_info,
        "equality_eval_after_pretrain": pre_eval_after,
        "equality_eval_after_relation": post_eval,
        "train_info": train_info,
        "train_state_acc": ft.get("train_state_acc"),
        "train_cmp_acc": ft.get("train_cmp_acc"),
        "central_eval": central,
        "matching_accuracy": {"mode": "learned_eq_aux", "eval": eval_match, "train": train_match},
    }
    write_json(rd / "result.json", rec)
    write_jsonl(rd / "state_predictions.jsonl", eval_sr)
    write_jsonl(rd / "comparison_predictions.jsonl", eval_cr)
    return rec


def write_summary(out: Path, results: List[Dict[str, Any]]) -> None:
    lines = ["# research equality-auxiliary span-routing probe", "",
             "Protected equality means matcher CharGRU + neither bias were pretrained on token/query equality labels and frozen before relational training. This is a controlled routing intervention, not natural span discovery.", "",
             "| condition | bs | vocab | eq_pre | frozen | train_state | train_cmp | graph_same | unchanged | hh_closure | mixed | match_cand | match_other | eq both gate before | after pre | after rel |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        ce = r.get("central_eval", {})
        ma = r.get("matching_accuracy", {}).get("eval", {})
        eb = r.get("equality_eval_before", {})
        ea = r.get("equality_eval_after_pretrain", {})
        er = r.get("equality_eval_after_relation", {})
        lines.append("| {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            r.get("condition"), r.get("bridge_sign"), r.get("vocab_size"), r.get("eq_pretrain_epochs"), r.get("freeze_matcher"),
            float(r.get("train_state_acc", 0.0)), float(r.get("train_cmp_acc", 0.0)),
            float(ce.get("graph_same", 0.0)), float(ce.get("unchanged", 0.0)), float(ce.get("hh_closure", 0.0)), float(ce.get("mixed_acc", 0.0)),
            float(ma.get("cand_match_acc", 0.0)), float(ma.get("other_match_acc", 0.0)),
            float(eb.get("both_wins_gate", 0.0)), float(ea.get("both_wins_gate", 0.0)), float(er.get("both_wins_gate", 0.0))))
    write_json(out / "equality_aux_summary.json", {"all_results": results})
    (out / "equality_aux_summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arm", default=base.DEFAULT_ARM)
    ap.add_argument("--conditions", nargs="+", default=["shared_trunk"])
    ap.add_argument("--seed", type=int, default=29600)
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
    ap.add_argument("--eq-pretrain-epochs", type=int, default=25)
    ap.add_argument("--eq-lr", type=float, default=5e-3)
    ap.add_argument("--eq-null-weight", type=float, default=1.0)
    ap.add_argument("--eq-print-every", type=int, default=5)
    ap.add_argument("--eq-include-static", action="store_true")
    ap.add_argument("--freeze-matcher", action="store_true")
    args = ap.parse_args()

    ts, tc, _, _, _, _ = base.load_dataset(args.data_root, args.arm)
    vocab = build_vocab(ts, tc, args.no_name_vocab)
    print(f"research equality-aux span probe conditions={args.conditions} bridge_signs={args.bridge_signs} no_name_vocab={args.no_name_vocab} vocab={len(vocab.itos)} eq_pre={args.eq_pretrain_epochs} freeze={args.freeze_matcher}", flush=True)
    results = []
    for bs in args.bridge_signs:
        paired = base.create_paired_span_models(len(vocab.itos), args.conditions, args.seed,
                                                args.emb_dim, args.hidden, args.char_dim, args.char_hidden)
        for cond in args.conditions:
            print(f"\n=== eqaux {'freeze' if args.freeze_matcher else 'joint'} {cond}/bs={'+' if bs==1 else '-'}1 ===", flush=True)
            rec = run_one(args, cond, bs, paired)
            results.append(rec)
            print(json.dumps(rec, indent=2, sort_keys=True), flush=True)
    write_summary(args.out, results)
    print(json.dumps({"status": "EQUALITY_AUX_SPAN_COMPLETE",
                      "summary": project_rel(args.out / "equality_aux_summary.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
