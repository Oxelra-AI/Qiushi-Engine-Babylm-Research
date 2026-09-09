#!/usr/bin/env python3
"""Step036b: Paired-context contrastive objective on natural-language rows.

Extends the paired-context loss from research template to multi-token natural rows.
For each pair (UPDATE + RETAIN), computes:

  d(ctx) = mean_log_P(new_state | ctx) - mean_log_P(source_state | ctx)

where mean_log_P uses per-token log probability averaged over the candidate span.

  L_paired = -log σ(d_update - d_retain)  [directly optimizes (U+R)/2]
  L_ce     = -(mean_log_P(new | update) + mean_log_P(source | retain)) / 2
  L        = L_ce + λ * L_paired

Four forward passes per pair (both candidates in both contexts) because candidate
token lengths may differ.

Evaluation reports (U+R)/2, (U-R)/2, absolute UPDATE/RETAIN correctness, and
joint correctness for train and held splits separately.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import copy
import json
import math
import pathlib
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))

from corruption_vs_loss_factorial import (
    locate_span_token_positions,
    load_private_model, freeze_to_private_adapters,
)
from multitoken_scorer import (
    read_jsonl, write_jsonl, build_candidate_full_text,
    validate_pairs as validate_a02_pairs,
)


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Pre-computation: 4 masked inputs per pair
# ---------------------------------------------------------------------------

def prepare_masked_input(row: Dict[str, Any], candidate: str, tokenizer,
                         seq_length: int, device: torch.device) -> Dict[str, Any]:
    """Build one masked input from a row and candidate string.

    Inserts the candidate at {STATE} in the use_sentence_frame, tokenizes with
    offset mapping to find the candidate span, replaces those positions with <mask>,
    and returns the masked input_ids, attention_mask, mask positions, and target
    token IDs.
    """
    full, cand_start, cand_end = build_candidate_full_text(row, candidate)
    enc = tokenizer(
        full, add_special_tokens=True, return_offsets_mapping=True,
        return_tensors="pt", max_length=seq_length, truncation=True,
        padding="max_length",
    )
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
    positions = locate_span_token_positions(offsets, cand_start, cand_end)
    if not positions:
        raise ValueError(
            f"Candidate span not found/truncated: candidate={candidate!r}, "
            f"pair_id={row.get('pair_id')}, packet={row.get('packet_type')}"
        )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    target_ids = input_ids[0, positions].clone()
    masked_ids = input_ids.clone()
    masked_ids[0, positions] = int(tokenizer.mask_token_id)
    return {
        "masked_ids": masked_ids,  # [1, L]
        "attention_mask": attention_mask,  # [1, L]
        "positions": positions,  # list of int
        "target_ids": target_ids,  # [n_tokens]
        "n_tokens": len(positions),
        "candidate": candidate,
    }


def prepare_natural_pair(update_row: Dict, retain_row: Dict, tokenizer,
                         seq_length: int, device: torch.device) -> Dict[str, Any]:
    """Pre-compute all 4 masked inputs for a natural-language pair."""
    new_state = str(update_row["answer_text"]).strip()
    source_state = str(update_row["foil_text"]).strip()

    inputs = {}
    for ctx_name, row in [("update", update_row), ("retain", retain_row)]:
        for cand_name, candidate in [("new", new_state), ("source", source_state)]:
            key = f"{ctx_name}_{cand_name}"
            inputs[key] = prepare_masked_input(row, candidate, tokenizer, seq_length, device)

    return {
        "pair_id": str(update_row["pair_id"]),
        "new_state": new_state,
        "source_state": source_state,
        "inputs": inputs,
    }


# ---------------------------------------------------------------------------
# Scoring (differentiable for training, detach for evaluation)
# ---------------------------------------------------------------------------

def score_pair_forward(model, pair_data: Dict[str, Any], no_grad: bool = False):
    """Forward all 4 inputs and compute d_update, d_retain, CE components.

    Returns differentiable tensors when no_grad=False (training).
    """
    scores = {}
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        for key, inp in pair_data["inputs"].items():
            logits = model(input_ids=inp["masked_ids"],
                           attention_mask=inp["attention_mask"]).logits[0]
            lp = F.log_softmax(logits[inp["positions"]], dim=-1)
            pos_idx = torch.arange(len(inp["positions"]), device=logits.device)
            token_lps = lp[pos_idx, inp["target_ids"]]
            scores[key] = token_lps.mean()  # mean per-token log prob

    d_update = scores["update_new"] - scores["update_source"]
    d_retain = scores["retain_new"] - scores["retain_source"]

    return {
        "d_update": d_update,
        "d_retain": d_retain,
        "score_update_new": scores["update_new"],
        "score_update_source": scores["update_source"],
        "score_retain_new": scores["retain_new"],
        "score_retain_source": scores["retain_source"],
    }


def compute_natural_pair_loss(model, pair_data: Dict, arm: str, lambda_paired: float):
    """Compute loss for one natural pair."""
    s = score_pair_forward(model, pair_data, no_grad=False)

    ce_update = -s["score_update_new"]   # new is correct in update
    ce_retain = -s["score_retain_source"]  # source is correct in retain
    ce_loss = (ce_update + ce_retain) / 2

    paired_loss = -F.logsigmoid(s["d_update"] - s["d_retain"])

    if arm == "ce_only":
        total = ce_loss
    elif arm == "paired_only":
        total = paired_loss
    elif arm == "combined":
        total = ce_loss + lambda_paired * paired_loss
    else:
        raise ValueError(f"Unknown arm: {arm}")

    return total, float(ce_loss.item()), float(paired_loss.item())


# ---------------------------------------------------------------------------
# Evaluation with decomposition
# ---------------------------------------------------------------------------

def evaluate_natural_pairs(model, pair_datas: Sequence[Dict], device, tag: str = "") -> Dict[str, Any]:
    """Evaluate all pairs with (U+R)/2 decomposition."""
    model.eval()
    results = []
    for pd in pair_datas:
        s = score_pair_forward(model, pd, no_grad=True)
        d_u = float(s["d_update"].item())
        d_r = float(s["d_retain"].item())

        # U = d_update, R = -d_retain (in our signed convention)
        U = d_u
        R = -d_r  # source_minus_new margin
        UR = U + R
        recip_dep = UR / 2
        shared_pref = (U - R) / 2

        results.append({
            "pair_id": pd["pair_id"],
            "U_new_minus_source": U,
            "R_source_minus_new": R,
            "UR_sum": UR,
            "recipient_dependent_half": recip_dep,
            "shared_preference_half": shared_pref,
            "d_update": d_u,
            "d_retain": d_r,
            "update_correct": U > 0,
            "retain_correct": R > 0,
            "joint_correct": U > 0 and R > 0,
        })

    n = len(results)
    mean_U = sum(r["U_new_minus_source"] for r in results) / max(1, n)
    mean_R = sum(r["R_source_minus_new"] for r in results) / max(1, n)

    return {
        "tag": tag,
        "n_pairs": n,
        "mean_U": mean_U,
        "mean_R": mean_R,
        "mean_UR": mean_U + mean_R,
        "recipient_dependent_half": (mean_U + mean_R) / 2,
        "shared_preference_half": (mean_U - mean_R) / 2,
        "n_update_correct": sum(1 for r in results if r["update_correct"]),
        "n_retain_correct": sum(1 for r in results if r["retain_correct"]),
        "n_joint_correct": sum(1 for r in results if r["joint_correct"]),
        "per_pair": results,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_arm_natural(arm_name: str, model_path: pathlib.Path, tokenizer,
                      train_pairs: Sequence[Dict], held_pairs: Sequence[Dict],
                      args, device: torch.device) -> Dict[str, Any]:
    seed_all(args.seed)
    model, load_info = load_private_model(model_path, args.private_bottleneck,
                                          args.private_scale, device)
    trainable = freeze_to_private_adapters(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    trajectory = []
    eval_epochs = set(range(0, args.epochs + 1, args.eval_every))
    eval_epochs.add(args.epochs)

    for epoch in range(args.epochs + 1):
        if epoch in eval_epochs:
            ev_train = evaluate_natural_pairs(model, train_pairs, device, f"train_e{epoch}")
            ev_held = evaluate_natural_pairs(model, held_pairs, device, f"held_e{epoch}")
            point = {
                "epoch": epoch,
                "train_U": ev_train["mean_U"],
                "train_R": ev_train["mean_R"],
                "train_recip_dep": ev_train["recipient_dependent_half"],
                "train_shared_pref": ev_train["shared_preference_half"],
                "train_joint": ev_train["n_joint_correct"],
                "train_n": ev_train["n_pairs"],
                "held_U": ev_held["mean_U"],
                "held_R": ev_held["mean_R"],
                "held_recip_dep": ev_held["recipient_dependent_half"],
                "held_shared_pref": ev_held["shared_preference_half"],
                "held_joint": ev_held["n_joint_correct"],
                "held_n": ev_held["n_pairs"],
            }
            trajectory.append(point)
            print(f"[{arm_name}] e{epoch:04d}"
                  f" | tr j={ev_train['n_joint_correct']}/{ev_train['n_pairs']}"
                  f" U={ev_train['mean_U']:+.3f} R={ev_train['mean_R']:+.3f}"
                  f" dep={ev_train['recipient_dependent_half']:+.4f}"
                  f" shared={ev_train['shared_preference_half']:+.4f}"
                  f" | he j={ev_held['n_joint_correct']}/{ev_held['n_pairs']}"
                  f" U={ev_held['mean_U']:+.3f} R={ev_held['mean_R']:+.3f}"
                  f" dep={ev_held['recipient_dependent_half']:+.4f}"
                  f" shared={ev_held['shared_preference_half']:+.4f}",
                  flush=True)

        if epoch == args.epochs:
            break

        # Training step: accumulate gradients over all pairs
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        total_ce = 0.0
        total_paired = 0.0

        indices = list(range(len(train_pairs)))
        random.shuffle(indices)

        for i in indices:
            loss, ce, paired = compute_natural_pair_loss(
                model, train_pairs[i], arm_name, args.lambda_paired
            )
            (loss / len(train_pairs)).backward()  # gradient accumulation
            total_loss += loss.item()
            total_ce += ce
            total_paired += paired

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

    # Final evaluation
    model.eval()
    final_train = evaluate_natural_pairs(model, train_pairs, device, "final_train")
    final_held = evaluate_natural_pairs(model, held_pairs, device, "final_held")

    return {
        "arm": arm_name,
        "trajectory": trajectory,
        "final_train": {k: v for k, v in final_train.items() if k != "per_pair"},
        "final_held": {k: v for k, v in final_held.items() if k != "per_pair"},
        "final_train_per_pair": final_train["per_pair"],
        "final_held_per_pair": final_held["per_pair"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step036b natural-row paired-context objective")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rows_path", type=str,
                        default="experiments/archive/relation_learning/data/contrastive_binding_validated_strict"
                                "accepted_contrastive_binding_training_rows_strict_pilot512.jsonl")
    parser.add_argument("--model_path", type=str,
                        default="models/frontier")
    parser.add_argument("--lambda_paired", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=43036)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--private_bottleneck", type=int, default=128)
    parser.add_argument("--private_scale", type=float, default=0.75)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_frac", type=float, default=0.73,
                        help="Fraction of pairs for training (~40/55)")
    parser.add_argument("--arms", nargs="+", default=["ce_only", "combined"],
                        help="Arms to run")
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    device = torch.device(args.device)
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and rows
    model_path = pathlib.Path(args.model_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)

    rows = read_jsonl(pathlib.Path(args.rows_path))
    print(f"Loaded {len(rows)} rows from {args.rows_path}", flush=True)

    # Validate pairs
    meta, by_pair = validate_a02_pairs(rows)
    print(f"Pairs: {meta['n_pairs']}, issues: {meta['n_pair_issues']}", flush=True)

    # Build sorted pair list
    pair_ids = sorted(pid for pid, d in by_pair.items() if set(d) == {"UPDATE", "RETAIN"})
    print(f"Valid pairs: {len(pair_ids)}", flush=True)

    # Split train/held
    seed_all(args.seed)
    random.shuffle(pair_ids)
    n_train = int(len(pair_ids) * args.train_frac)
    train_ids = set(pair_ids[:n_train])
    held_ids = set(pair_ids[n_train:])
    print(f"Train pairs: {len(train_ids)}, held pairs: {len(held_ids)}", flush=True)

    # Pre-compute masked inputs for all pairs
    print("Pre-computing masked inputs...", flush=True)
    train_pair_datas = []
    held_pair_datas = []
    errors = []
    for pid in pair_ids:
        d = by_pair[pid]
        update_row = d["UPDATE"]
        retain_row = d["RETAIN"]
        try:
            pd = prepare_natural_pair(update_row, retain_row, tokenizer, args.seq_length, device)
            if pid in train_ids:
                train_pair_datas.append(pd)
            else:
                held_pair_datas.append(pd)
        except Exception as e:
            errors.append({"pair_id": pid, "error": str(e)})
    print(f"Pre-computed: {len(train_pair_datas)} train, {len(held_pair_datas)} held, {len(errors)} errors",
          flush=True)
    if errors:
        for e in errors[:5]:
            print(f"  Error: {e}", flush=True)

    # Baseline evaluation
    seed_all(args.seed)
    model, _ = load_private_model(model_path, args.private_bottleneck, args.private_scale, device)
    model.eval()
    base_train = evaluate_natural_pairs(model, train_pair_datas, device, "baseline_train")
    base_held = evaluate_natural_pairs(model, held_pair_datas, device, "baseline_held")
    del model
    torch.cuda.empty_cache()

    print(f"\n=== BASELINE DECOMPOSITION (natural rows) ===", flush=True)
    for tag, ev in [("train", base_train), ("held", base_held)]:
        print(f"  {tag}: U={ev['mean_U']:+.4f} R={ev['mean_R']:+.4f} "
              f"(U+R)/2={ev['recipient_dependent_half']:+.4f} "
              f"(U-R)/2={ev['shared_preference_half']:+.4f} "
              f"joint={ev['n_joint_correct']}/{ev['n_pairs']}", flush=True)

    # Run arms
    all_results = {"baseline_train": {k: v for k, v in base_train.items() if k != "per_pair"},
                   "baseline_held": {k: v for k, v in base_held.items() if k != "per_pair"},
                   "arms": {}}

    for arm_name in args.arms:
        print(f"\n{'=' * 60}", flush=True)
        print(f"=== ARM: {arm_name} ===", flush=True)
        print(f"{'=' * 60}", flush=True)
        result = train_arm_natural(arm_name, model_path, tokenizer,
                                   train_pair_datas, held_pair_datas, args, device)
        all_results["arms"][arm_name] = result
        torch.cuda.empty_cache()

    # Save results
    summary = {
        "status": "STEP036B_NATURAL_PAIRED_CONTEXT",
        "n_train": len(train_pair_datas),
        "n_held": len(held_pair_datas),
        "n_errors": len(errors),
        "baseline_train_recip_dep": base_train["recipient_dependent_half"],
        "baseline_held_recip_dep": base_held["recipient_dependent_half"],
        "baseline_train_joint": f"{base_train['n_joint_correct']}/{base_train['n_pairs']}",
        "baseline_held_joint": f"{base_held['n_joint_correct']}/{base_held['n_pairs']}",
        "arms": {},
    }
    for name, r in all_results["arms"].items():
        ft = r["final_train"]
        fh = r["final_held"]
        summary["arms"][name] = {
            "final_train_joint": f"{ft['n_joint_correct']}/{ft['n_pairs']}",
            "final_held_joint": f"{fh['n_joint_correct']}/{fh['n_pairs']}",
            "final_train_recip_dep": ft["recipient_dependent_half"],
            "final_held_recip_dep": fh["recipient_dependent_half"],
            "final_train_shared_pref": ft["shared_preference_half"],
            "final_held_shared_pref": fh["shared_preference_half"],
            "final_train_U": ft["mean_U"],
            "final_train_R": ft["mean_R"],
            "final_held_U": fh["mean_U"],
            "final_held_R": fh["mean_R"],
        }

    with open(out / "natural_paired_context_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out / "natural_paired_context_full.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    for name, r in all_results["arms"].items():
        with open(out / f"trajectory_{name}.json", "w") as f:
            json.dump(r["trajectory"], f, indent=2)
        with open(out / f"final_train_pairs_{name}.json", "w") as f:
            json.dump(r.get("final_train_per_pair", []), f, indent=2)
        with open(out / f"final_held_pairs_{name}.json", "w") as f:
            json.dump(r.get("final_held_per_pair", []), f, indent=2)

    # Summary markdown
    md = ["# Step036b natural-row paired-context objective", "",
          f"Rows: {args.rows_path}", f"Train: {len(train_pair_datas)}, Held: {len(held_pair_datas)}", "",
          "## Baseline",
          f"- Train: (U+R)/2 = {base_train['recipient_dependent_half']:+.4f}, "
          f"joint = {base_train['n_joint_correct']}/{base_train['n_pairs']}",
          f"- Held: (U+R)/2 = {base_held['recipient_dependent_half']:+.4f}, "
          f"joint = {base_held['n_joint_correct']}/{base_held['n_pairs']}", "",
          "## Arms", "",
          "| Arm | Train joint | Held joint | Train (U+R)/2 | Held (U+R)/2 | Train (U-R)/2 | Held (U-R)/2 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name in args.arms:
        if name in summary["arms"]:
            a = summary["arms"][name]
            md.append(f"| {name} | {a['final_train_joint']} | {a['final_held_joint']} | "
                      f"{a['final_train_recip_dep']:+.4f} | {a['final_held_recip_dep']:+.4f} | "
                      f"{a['final_train_shared_pref']:+.4f} | {a['final_held_shared_pref']:+.4f} |")
    md.append("")
    with open(out / "natural_paired_context_summary.md", "w") as f:
        f.write("\n".join(md))

    print("\n" + "=" * 60, flush=True)
    print("FINAL SUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
