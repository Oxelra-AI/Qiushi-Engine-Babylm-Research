#!/usr/bin/env python3
"""Step035b: answer-only training preflight on natural-language contrast rows.

Purpose
-------
This is a pipeline and mechanism preflight, not strong training evidence. The
research pilot rows are known to contain semantic noise, but they have explicit
answer_text/foil_text/use_sentence_frame fields and pass the strict pair contract.
The question here is whether the multi-token scorer and private-adapter training
harness can install the UPDATE/RETAIN contrast on natural-language rows and whether
any movement transfers to held-out pilot pairs.

The learner receives only final-answer labels: all tokens of the final candidate
span in the use frame are replaced by <mask> and trained with CE. Source and update
sentences remain visible. Evaluation uses the symmetric multi-token scorer from
a02_multitoken_scorer.py.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import copy
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))

from corruption_vs_loss_factorial import (  # noqa: E402
    locate_span_token_positions,
    load_private_model,
    freeze_to_private_adapters,
)
from multitoken_scorer import (  # noqa: E402
    read_jsonl,
    write_jsonl,
    validate_pairs,
    build_candidate_full_text,
    score_row,
    summarize_pairs,
)


def seed_all(seed: int):
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def split_pair_ids(rows: Sequence[Dict[str, Any]], held_n: int, seed: int) -> Tuple[List[str], List[str]]:
    pair_ids = sorted({str(r["pair_id"]) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(pair_ids)
    held = sorted(pair_ids[:held_n])
    train = sorted(pair_ids[held_n:])
    return train, held


def subset_rows(rows: Sequence[Dict[str, Any]], keep_ids: Sequence[str]) -> List[Dict[str, Any]]:
    keep = set(keep_ids)
    return [r for r in rows if str(r["pair_id"]) in keep]


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def summarize_row_tokens(rows: Sequence[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    ans = [token_len(tokenizer, str(r["answer_text"])) for r in rows]
    foi = [token_len(tokenizer, str(r["foil_text"])) for r in rows]
    return {
        "n_rows": len(rows),
        "answer_token_lengths": dict(Counter(ans)),
        "foil_token_lengths": dict(Counter(foi)),
        "packet_type_counts": dict(Counter(str(r.get("packet_type")).upper() for r in rows)),
        "role_position_counts": dict(Counter(str(r.get("role_position_relation")) for r in rows)),
    }


class A02AnswerDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tokenizer, seq_length: int = 256):
        self.rows = list(rows)
        self.items: List[Dict[str, Any]] = []
        for row in self.rows:
            full, cand_start, cand_end = build_candidate_full_text(row, str(row["answer_text"]).strip())
            enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True,
                            return_tensors="pt", max_length=seq_length, truncation=True,
                            padding="max_length")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
            pos = locate_span_token_positions(offsets, cand_start, cand_end)
            if not pos:
                raise ValueError(f"Answer span missing/truncated for {row.get('pair_id')} {row.get('packet_type')}")
            labels = torch.full_like(input_ids, -100)
            masked_ids = input_ids.clone()
            for p in pos:
                labels[p] = input_ids[p]
                masked_ids[p] = int(tokenizer.mask_token_id)
            self.items.append({
                "input_ids": input_ids,
                "masked_input_ids": masked_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "n_answer_tokens": len(pos),
                "pair_id": row.get("pair_id"),
                "packet_type": row.get("packet_type"),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "masked_input_ids": torch.stack([b["masked_input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "n_answer_tokens": torch.tensor([int(b["n_answer_tokens"]) for b in batch], dtype=torch.long),
    }


def eval_rows(model, tokenizer, rows: Sequence[Dict[str, Any]], device: torch.device, seq_length: int,
              tag: str) -> Dict[str, Any]:
    model.eval()
    row_scores = []
    errors = []
    for i, r in enumerate(rows):
        try:
            row_scores.append(score_row(model, tokenizer, r, device, seq_length))
        except Exception as e:
            errors.append({"i": i, "pair_id": r.get("pair_id"), "packet_type": r.get("packet_type"), "error": repr(e)})
    pair_scores, pair_summary = summarize_pairs(row_scores)
    return {
        "tag": tag,
        "n_rows": len(rows),
        "n_scored_rows": len(row_scores),
        "n_errors": len(errors),
        "errors_first10": errors[:10],
        "pair_summary": pair_summary,
    }


def compact_pair_summary(ev: Dict[str, Any]) -> Dict[str, Any]:
    allp = ev["pair_summary"]["all_pairs"]
    by_role = ev["pair_summary"].get("by_role_position_relation", {})
    return {
        "tag": ev["tag"],
        "n_rows": ev["n_rows"],
        "n_pairs": allp.get("n_pairs", 0),
        "mean_U": allp.get("mean_U_mean"),
        "mean_R": allp.get("mean_R_mean"),
        "mean_U_plus_R": allp.get("mean_UR_mean"),
        "n_update_correct": allp.get("n_update_correct_mean"),
        "n_retain_correct": allp.get("n_retain_correct_mean"),
        "n_joint_correct": allp.get("n_joint_correct_mean"),
        "by_role": {
            k: {
                "n_pairs": v.get("n_pairs"),
                "mean_U_plus_R": v.get("mean_UR_mean"),
                "n_joint_correct": v.get("n_joint_correct_mean"),
            }
            for k, v in by_role.items()
        },
    }


def train(model, trainable, tokenizer, train_rows, held_rows, args, device) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    ds = A02AnswerDataset(train_rows, tokenizer, seq_length=args.seq_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed + 77))
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    trajectory: List[Dict[str, Any]] = []

    def record(epoch: int, train_loss: float | None = None):
        tr_ev = eval_rows(model, tokenizer, train_rows, device, args.seq_length, tag=f"train_e{epoch}")
        he_ev = eval_rows(model, tokenizer, held_rows, device, args.seq_length, tag=f"held_e{epoch}")
        entry = {
            "epoch": int(epoch),
            "train": compact_pair_summary(tr_ev),
            "held": compact_pair_summary(he_ev),
        }
        if train_loss is not None:
            entry["loss"] = float(train_loss)
        trajectory.append(entry)
        tr = entry["train"]; he = entry["held"]
        print(
            f"e{epoch:04d}" + (f" loss={train_loss:.4f}" if train_loss is not None else "") +
            f" | train joint={tr['n_joint_correct']}/{tr['n_pairs']} U={tr['mean_U']:+.3f} R={tr['mean_R']:+.3f} U+R={tr['mean_U_plus_R']:+.3f}"
            f" | held joint={he['n_joint_correct']}/{he['n_pairs']} U={he['mean_U']:+.3f} R={he['mean_R']:+.3f} U+R={he['mean_U_plus_R']:+.3f}",
            flush=True,
        )

    record(0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch in loader:
            seed_all(args.seed + epoch * 1009 + total_n)
            input_ids = batch["masked_input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            mask = labels != -100
            if not mask.any():
                continue
            loss = F.cross_entropy(outputs.logits[mask].view(-1, outputs.logits.size(-1)),
                                   labels[mask].view(-1), reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            n = int(mask.sum().item())
            total_loss += float(loss.detach().item()) * n
            total_n += n
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            record(epoch, total_loss / max(1, total_n))
    final_train_full = eval_rows(model, tokenizer, train_rows, device, args.seq_length, tag="final_train")
    final_held_full = eval_rows(model, tokenizer, held_rows, device, args.seq_length, tag="final_held")
    return trajectory, final_train_full, final_held_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="experiments/archive/relation_learning/data/contrastive_binding_validated_strict/accepted_contrastive_binding_training_rows_strict_pilot512.jsonl")
    ap.add_argument("--model-path", default="models/frontier")
    ap.add_argument("--out-dir", default="experiments/archive/functional_learning/data/revision_035b_answer_only_train_pilot")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--private-bottleneck", type=int, default=128)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--eval-every", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=43035)
    ap.add_argument("--held-pairs", type=int, default=15)
    args = ap.parse_args()

    seed_all(args.seed)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(pathlib.Path(args.rows))
    validation, by_pair = validate_pairs(rows)
    if validation["n_pair_issues"]:
        raise RuntimeError(f"Pair validation issues: {validation['pair_issues_first20']}")
    train_ids, held_ids = split_pair_ids(rows, args.held_pairs, args.seed)
    train_rows = subset_rows(rows, train_ids)
    held_rows = subset_rows(rows, held_ids)

    from transformers import AutoTokenizer
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    model, load_info = load_private_model(pathlib.Path(args.model_path), args.private_bottleneck,
                                          args.private_scale, device)
    trainable = freeze_to_private_adapters(model)
    print(f"Loaded model on {device}; train pairs={len(train_ids)} held pairs={len(held_ids)}", flush=True)

    baseline_train = eval_rows(model, tokenizer, train_rows, device, args.seq_length, tag="baseline_train")
    baseline_held = eval_rows(model, tokenizer, held_rows, device, args.seq_length, tag="baseline_held")
    print("Baseline:", json.dumps({
        "train": compact_pair_summary(baseline_train),
        "held": compact_pair_summary(baseline_held),
    }, ensure_ascii=False), flush=True)

    trajectory, final_train, final_held = train(model, trainable, tokenizer, train_rows, held_rows, args, device)

    write_jsonl(out_dir / "train_rows.jsonl", train_rows)
    write_jsonl(out_dir / "held_rows.jsonl", held_rows)
    summary = {
        "status": "STEP035B_A02_ANSWER_ONLY_TRAIN_PILOT",
        "purpose": "pipeline/mechanism preflight on noisy research pilot rows, not final training evidence",
        "rows_path": args.rows,
        "model_path": args.model_path,
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "train_pair_ids": train_ids,
        "held_pair_ids": held_ids,
        "validation": validation,
        "token_summary_train": summarize_row_tokens(train_rows, tokenizer),
        "token_summary_held": summarize_row_tokens(held_rows, tokenizer),
        "baseline_train": compact_pair_summary(baseline_train),
        "baseline_held": compact_pair_summary(baseline_held),
        "trajectory": trajectory,
        "final_train": compact_pair_summary(final_train),
        "final_held": compact_pair_summary(final_held),
        "load_info": {"missing": list(load_info.get("missing", []))[:20], "unexpected": list(load_info.get("unexpected", []))[:20]},
    }
    (out_dir / "answer_only_train_pilot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Step035b answer-only natural-row training pilot\n\n"]
    md.append("This is a scorer/training preflight on noisy 55-pair pilot rows, not evidence that these rows are ready for BabyLM training.\n\n")
    md.append(f"Train pairs: {len(train_ids)}; held pairs: {len(held_ids)}; epochs: {args.epochs}; lr: {args.lr}.\n\n")
    md.append("| Eval | joint | update | retain | mean U | mean R | mean U+R |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for name, ev in [("baseline train", compact_pair_summary(baseline_train)),
                     ("baseline held", compact_pair_summary(baseline_held)),
                     ("final train", compact_pair_summary(final_train)),
                     ("final held", compact_pair_summary(final_held))]:
        md.append(f"| {name} | {ev['n_joint_correct']}/{ev['n_pairs']} | {ev['n_update_correct']}/{ev['n_pairs']} | {ev['n_retain_correct']}/{ev['n_pairs']} | {ev['mean_U']:+.3f} | {ev['mean_R']:+.3f} | {ev['mean_U_plus_R']:+.3f} |\n")
    md.append("\n## Interpretation\n\n")
    md.append("The training masks only the final answer span and leaves source/update evidence visible. Movement on train rows primarily validates that the multi-token answer-loss path can shape the private adapter. Movement on held rows, if any, is only suggestive because the pilot rows contain semantic noise and are few. The decisive next experiment still needs cleaner accepted rows and an exposure-matched ALN-preserving continuation control.\n")
    (out_dir / "answer_only_train_pilot_summary.md").write_text("".join(md), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "out_dir": str(out_dir),
        "baseline_train": summary["baseline_train"],
        "baseline_held": summary["baseline_held"],
        "final_train": summary["final_train"],
        "final_held": summary["final_held"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
