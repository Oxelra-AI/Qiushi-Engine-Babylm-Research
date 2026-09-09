#!/usr/bin/env python3
"""research: dual-position learned-equality gauge probe.

This script is the relational continuation of the research CPU equality pilot.
It deliberately removes the zero-shot same-symbol prior in research's matcher by
using separate token-side and query-side character embeddings. Equality must be
installed from finite character-pair evidence before relational training.

Scientific use:
- If full-alphabet character-pair pretraining gives perfect held assignment and
  shared_trunk then transports the gauge, finite lower-level equality evidence is
  sufficient for the raw-text gauge mechanism.
- If train-alphabet pretraining fails exactly on held names containing unseen
  characters, the alphabet coverage boundary from the CPU pilot is confirmed in
  the downstream relational task.
- Matched untied controls keep the shared-representation condition separate.

This script should be launched only after the pending research tasks are collected
and, for causal sign-pair analysis, with paired positive/negative bridge cells
using the same code, same seed, and fresh untouched initialization.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, copy, hashlib, json, random, sys, time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(_public_path('experiments/archive/representation_and_objectives/training/scripts')))
import posalign_gauge_probe as p298  # noqa: E402
import raw_span_discovery_probe as base  # noqa: E402

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_OUT = PROJECT / "data/dualpos_learned_equality_gauge"


class DualPositionAlignedMatcher(nn.Module):
    """Position-sensitive matcher without shared token/query character priors."""
    def __init__(self, n_chars=28, char_dim=16, temperature=1.0):
        super().__init__()
        self.tok_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.qry_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.length_penalty = nn.Parameter(torch.tensor(-3.0))

    def batch_match_scores(self, tok_chars_batch: torch.Tensor, qry_chars: torch.Tensor) -> torch.Tensor:
        tok_emb = self.tok_emb(tok_chars_batch)
        qry_emb = self.qry_emb(qry_chars)
        sim = (tok_emb * qry_emb.unsqueeze(0)).sum(dim=-1)
        tok_mask = tok_chars_batch > 0
        qry_mask = qry_chars > 0
        both_valid = tok_mask & qry_mask.unsqueeze(0)
        tok_len = tok_mask.sum(dim=1)
        qry_len = qry_mask.sum()
        same_length = (tok_len == qry_len).float()
        match_prob = torch.sigmoid(sim / self.temperature.abs().clamp(min=0.1))
        valid_match = match_prob * both_valid.float() + (1.0 - both_valid.float())
        product = valid_match.prod(dim=1)
        len_factor = same_length + (1 - same_length) * torch.sigmoid(self.length_penalty)
        return product * len_factor

    def match_score(self, tok_chars: torch.Tensor, qry_chars: torch.Tensor) -> torch.Tensor:
        return self.batch_match_scores(tok_chars.unsqueeze(0), qry_chars)[0]


def patch_dual_matcher() -> None:
    # PosAlignTrunk resolves PositionAlignedMatcher from the p298 module globals
    # at construction time, so patch before create_paired_posalign is called.
    p298.PositionAlignedMatcher = DualPositionAlignedMatcher


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def all_matchers(model) -> List[DualPositionAlignedMatcher]:
    out = []
    seen = set()
    for scorer in [model.event_state, model.event_cmp, model.static]:
        m = scorer.trunk.matcher
        if id(m) not in seen:
            seen.add(id(m)); out.append(m)
    return out


def names_from_training(ts, tc) -> List[str]:
    ns = set()
    for q in ts:
        for n in q.names: ns.add(n.lower())
    for c in tc:
        for n in c.names: ns.add(n.lower())
    return sorted(ns)


def char_ids_from_names(names: Sequence[str]) -> List[int]:
    out = set()
    for n in names:
        for ch in n.lower():
            if "a" <= ch <= "z": out.add(ord(ch) - ord("a") + 1)
    return sorted(out)


def pretrain_char_pairs(model, char_ids: Sequence[int], device, epochs: int, lr: float, print_every: int):
    matchers = all_matchers(model)
    params = []
    for m in matchers:
        params.extend(list(m.parameters()))
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    pairs = [(a, b, float(a == b)) for a in char_ids for b in char_ids]
    rng = random.Random(29902)
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        rng.shuffle(pairs)
        opt.zero_grad(set_to_none=True)
        losses = []
        for m in matchers:
            for a, b, y in pairs:
                tok = torch.zeros((1, p298.MAX_NAME_LEN), dtype=torch.long, device=device)
                qry = torch.zeros(p298.MAX_NAME_LEN, dtype=torch.long, device=device)
                tok[0, 0] = int(a); qry[0] = int(b)
                s = m.batch_match_scores(tok, qry).clamp(1e-7, 1-1e-7)[0]
                target = torch.tensor(y, dtype=torch.float32, device=device)
                losses.append(F.binary_cross_entropy(s, target, reduction="mean"))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        if ep == 1 or ep == epochs or (print_every and ep % print_every == 0):
            with torch.no_grad():
                ok = 0; tot = 0
                for a, b, y in pairs:
                    tok = torch.zeros((1, p298.MAX_NAME_LEN), dtype=torch.long, device=device)
                    qry = torch.zeros(p298.MAX_NAME_LEN, dtype=torch.long, device=device)
                    tok[0, 0] = int(a); qry[0] = int(b)
                    # Check the first matcher; all are trained with the same evidence.
                    pred = bool(matchers[0].batch_match_scores(tok, qry)[0].item() >= 0.5)
                    ok += int(pred == bool(y)); tot += 1
            rec = {"epoch": ep, "loss": float(loss.detach().cpu()),
                   "pair_acc_first_matcher": ok/tot if tot else 0,
                   "elapsed": round(time.time() - t0, 2)}
            hist.append(rec)
            if print_every:
                print(json.dumps({"char_pair_pretrain": rec}), flush=True)
    return hist


def run_one(args, cond, bs, paired, vocab, ts, tc, es, ec, char_ids):
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    model = copy.deepcopy(paired[cond]).to(device)

    ih = hashlib.sha256()
    for p in model.parameters(): ih.update(p.detach().cpu().numpy().tobytes())
    ihp = ih.hexdigest()[:16]
    print(f"Init hash dualpos {cond}/bs{'+' if bs==1 else '-'}1: {ihp}", flush=True)

    match_before_train, _ = p298.eval_matching_accuracy(model, vocab, ts, tc, device)
    match_before_eval, _ = p298.eval_matching_accuracy(model, vocab, es, ec, device)
    pre_hist = []
    if args.char_pair_mode != "none" and args.char_pair_epochs > 0:
        pre_hist = pretrain_char_pairs(model, char_ids, device, args.char_pair_epochs, args.char_pair_lr, args.print_every)
    match_after_train, _ = p298.eval_matching_accuracy(model, vocab, ts, tc, device)
    match_after_eval, _ = p298.eval_matching_accuracy(model, vocab, es, ec, device)
    print(json.dumps({"matching": {"before_train": match_before_train, "before_eval": match_before_eval,
                                    "after_train": match_after_train, "after_eval": match_after_eval}}, indent=2), flush=True)
    p298.freeze_matcher(model)

    print(f"\n=== dualpos {args.char_pair_mode} {cond}/bs={'+' if bs==1 else '-'}1/seed={seed} ===", flush=True)
    train_info = p298.train_relational(model, vocab, ts, tc, device, bs, args.epochs, args.lr, args.wd,
                                       seed, args.cmp_weight, args.state_weight, args.static_weight,
                                       args.print_every, hard=True)
    eval_sr, eval_cr = p298.eval_predictions(model, vocab, es, ec, device, cond, args.arm, seed, bs, hard=True)
    train_sr, train_cr = p298.eval_predictions(model, vocab, ts, tc, device, cond, args.arm, seed, bs, hard=True)
    eval_ma_final, _ = p298.eval_matching_accuracy(model, vocab, es, ec, device)
    choices = base.state_choice_records(eval_sr)
    boths = base.pair_both_records(choices)
    central = base.compute_central(choices, boths, eval_cr, cond, args.arm, seed, bs)

    tag = f"dualpos_{args.char_pair_mode}_{cond}_bs{'+' if bs==1 else '-'}1_seed{seed}"
    rd = args.out / tag
    rec = {
        "condition": cond, "arm": args.arm, "seed": seed, "bridge_sign": bs,
        "mode": "dual_position_learned_equality_hard", "char_pair_mode": args.char_pair_mode,
        "char_pair_epochs": args.char_pair_epochs, "char_ids": list(map(int, char_ids)),
        "epochs": args.epochs, "vocab_size": len(vocab.itos), "no_name_vocab": args.no_name_vocab,
        "init_hash_prefix": ihp,
        "matching_before": {"train": match_before_train, "eval": match_before_eval},
        "matching_after_pretrain": {"train": match_after_train, "eval": match_after_eval},
        "matching_final": eval_ma_final,
        "pretrain_history": pre_hist,
        "train_info": train_info,
        "central_eval": central,
    }
    write_json(rd / "result.json", rec)
    p298.write_jsonl(rd / "state_predictions.jsonl", eval_sr)
    p298.write_jsonl(rd / "comparison_predictions.jsonl", eval_cr)
    p298.write_jsonl(rd / "train_state_predictions.jsonl", train_sr)
    p298.write_jsonl(rd / "train_comparison_predictions.jsonl", train_cr)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arm", default=base.DEFAULT_ARM)
    ap.add_argument("--conditions", nargs="+", default=["shared_trunk"])
    ap.add_argument("--bridge-signs", nargs="+", type=int, choices=[1, -1], default=[1, -1])
    ap.add_argument("--char-pair-mode", choices=["none", "train_alphabet", "full_alphabet"], default="full_alphabet")
    ap.add_argument("--char-pair-epochs", type=int, default=60)
    ap.add_argument("--char-pair-lr", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=29910)
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
    ap.add_argument("--no-name-vocab", action="store_true")
    args = ap.parse_args()

    patch_dual_matcher()
    ts, tc, es, ec, _, counts = base.load_dataset(args.data_root, args.arm)
    train_names = names_from_training(ts, tc)
    if args.char_pair_mode == "none":
        char_ids = []
    elif args.char_pair_mode == "train_alphabet":
        char_ids = char_ids_from_names(train_names)
    else:
        char_ids = list(range(1, 27))

    vocab = base.RawVocab()
    train_name_set = set()
    for q in ts:
        for n in q.names: train_name_set.add(n)
    for c in tc:
        for n in c.names: train_name_set.add(n)
    if args.no_name_vocab:
        base.collect_raw_vocab(vocab, ts, tc, exclude_names=train_name_set)
    else:
        base.collect_raw_vocab(vocab, ts, tc)

    print(f"research dualpos learned-equality gauge probe: conditions={args.conditions} bridge_signs={args.bridge_signs} "
          f"char_pair_mode={args.char_pair_mode} char_ids={char_ids} no_name_vocab={args.no_name_vocab} vocab={len(vocab.itos)}",
          flush=True)

    results = []
    for bs in args.bridge_signs:
        # Fresh untouched initialization per sign; causal paired analysis requires
        # rerunning both signs with the same seed and code.
        paired = p298.create_paired_posalign(len(vocab.itos), args.conditions, args.seed,
                                             args.emb_dim, args.hidden, args.char_dim)
        for cond in args.conditions:
            rec = run_one(args, cond, bs, paired, vocab, ts, tc, es, ec, char_ids)
            results.append(rec)
            ce = rec.get("central_eval", {})
            print(json.dumps({"done": f"dualpos_{args.char_pair_mode}_{cond}_bs{'+' if bs==1 else '-'}1",
                              "match_eval_after": rec.get("matching_after_pretrain", {}).get("eval"),
                              "graph_same": ce.get("graph_same"),
                              "unchanged": ce.get("unchanged"),
                              "hh_closure": ce.get("hh_closure"),
                              "mixed_acc": ce.get("mixed_acc"),
                              "train_state": rec.get("train_info", {}).get("history", [{}])[-1].get("train_state", None) if rec.get("train_info", {}).get("history") else None,
                              "train_cmp": rec.get("train_info", {}).get("history", [{}])[-1].get("train_cmp", None) if rec.get("train_info", {}).get("history") else None}, indent=2), flush=True)

    lines = ["# research dual-position learned-equality gauge probe", "",
             "| condition | bs | char mode | match after | train state | train cmp | graph_same | unchanged | hh_closure | mixed |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        ti = r.get("train_info", {}).get("history", [{}])[-1] if r.get("train_info", {}).get("history") else {}
        ce = r.get("central_eval", {})
        lines.append(f"| {r['condition']} | {r['bridge_sign']} | {r['char_pair_mode']} | {r.get('matching_after_pretrain',{}).get('eval',0):.3f} | {ti.get('train_state',0):.3f} | {ti.get('train_cmp',0):.3f} | {ce.get('graph_same',0):.3f} | {ce.get('unchanged',0):.3f} | {ce.get('hh_closure',0):.3f} | {ce.get('mixed_acc',0):.3f} |")
    write_json(args.out / "dualpos_gauge_summary.json", {"counts": counts, "results": results})
    (args.out / "dualpos_gauge_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "DUALPOS_LEARNED_EQUALITY_GAUGE_COMPLETE",
                      "summary": str(args.out / "dualpos_gauge_summary.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
