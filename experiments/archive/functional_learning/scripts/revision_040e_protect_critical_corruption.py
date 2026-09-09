#!/usr/bin/env python3
"""Step040e: critical-region protection under background corruption.

Uses the repaired relation-first rows and the same model/scoring/training substrate
as Step040b, but excludes critical evidence word groups from 15% background
corruption: entity A/B surface forms, source values A/B, and the shared replacement
value wherever they appear outside the final answer span. This tests whether the
large clean-vs-corrupted gap is mainly caused by corrupting the few support tokens
needed for recipient selection.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import gc
import json
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import relation_first_repaired_harness as h  # noqa: E402


def is_word_start(token: str) -> bool:
    return token.startswith("Ġ") or token.startswith("▁")


def find_all_spans(text: str, needle: str, limit_end: int | None = None) -> List[Tuple[int, int]]:
    if not needle:
        return []
    hay = text if limit_end is None else text[:limit_end]
    spans = []
    start = 0
    while True:
        idx = hay.find(needle, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(needle)))
        start = idx + max(1, len(needle))
    return spans


class ProtectedDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tokenizer, seq_length: int):
        self.items = []
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(int(x) for x in tokenizer.all_special_ids)
        self._word_start_cache: Dict[int, bool] = {}
        for row in rows:
            full, answer_start, answer_end = h.full_text_and_span(row, str(row["answer_text"]))
            enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True,
                            return_tensors="pt", max_length=seq_length, truncation=True,
                            padding="max_length")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
            word_group = torch.full((seq_length,), -1, dtype=torch.long)
            gid = -1
            for i in range(seq_length):
                if int(attention_mask[i]) == 0:
                    continue
                tid = int(input_ids[i])
                if tid in self.special_ids:
                    continue
                if gid < 0 or self._word_start(tid) or i == 0:
                    gid += 1
                word_group[i] = gid
            ans_pos = h.locate_span_positions(offsets, answer_start, answer_end)
            if not ans_pos:
                raise ValueError(f"answer span unavailable: {row['pair_id']} {row['row_type']}")
            answer_gids = sorted({int(word_group[p].item()) for p in ans_pos if int(word_group[p].item()) >= 0})

            # Protect critical evidence outside final answer span. Limit search to the
            # context before the final candidate insertion for replacement/source evidence,
            # but entity names may occur in source, update, and final query.
            protect_intervals: List[Tuple[int, int, str]] = []
            evidence_strings = [
                (str(row.get("entity_a", "")), "entity_a"),
                (str(row.get("entity_b", "")), "entity_b"),
                (str(row.get("value_a", "")), "value_a"),
                (str(row.get("value_b", "")), "value_b"),
                (str(row.get("shared_new_value", "")), "shared_new"),
            ]
            for text, label in evidence_strings:
                for s, e in find_all_spans(full, text):
                    # Do not treat final answer span as a background support token;
                    # it is protected separately as the target. Protect final query
                    # entity names, because they identify which state entry is queried.
                    if s < answer_end and e > answer_start and label in {"value_a", "value_b", "shared_new"}:
                        continue
                    protect_intervals.append((s, e, label))
            protected_positions = set()
            protected_gids = set()
            protected_by_label = collections.Counter()
            for s, e, label in protect_intervals:
                for pos in h.locate_span_positions(offsets, s, e):
                    if 0 <= pos < seq_length and int(attention_mask[pos].item()) == 1:
                        protected_positions.add(int(pos))
                        g = int(word_group[pos].item())
                        if g >= 0:
                            protected_gids.add(g)
                            protected_by_label[label] += 1
            for g in answer_gids:
                protected_gids.add(int(g))
            self.items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "word_group": word_group,
                "answer_positions": torch.tensor([int(p) for p in ans_pos], dtype=torch.long),
                "answer_gids": torch.tensor(answer_gids, dtype=torch.long),
                "protected_gids": torch.tensor(sorted(protected_gids), dtype=torch.long),
                "protected_position_count": len(protected_positions),
                "protected_by_label": dict(protected_by_label),
                "row": row,
            })

    def _word_start(self, tid: int) -> bool:
        if tid not in self._word_start_cache:
            tok = str(self.tokenizer.convert_ids_to_tokens(int(tid)))
            self._word_start_cache[tid] = is_word_start(tok)
        return self._word_start_cache[tid]

    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx: int):
        return self.items[idx]


def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "word_group": torch.stack([b["word_group"] for b in batch]),
        "answer_positions": [b["answer_positions"] for b in batch],
        "answer_gids": [b["answer_gids"] for b in batch],
        "protected_gids": [b["protected_gids"] for b in batch],
        "protected_position_count": torch.tensor([int(b["protected_position_count"]) for b in batch], dtype=torch.long),
    }


def apply_corruption(input_ids, attention_mask, word_group, answer_positions, protected_gids, tokenizer, bg_mask_prob, gen):
    device = input_ids.device
    bsz, seq = input_ids.shape
    bg_selected = torch.zeros((bsz, seq), dtype=torch.bool, device=device)
    protected_selected_would = 0
    for b in range(bsz):
        max_gid = int(word_group[b].max().item())
        if max_gid < 0:
            continue
        group_mask = torch.rand(max_gid + 1, generator=gen, device=device) < bg_mask_prob
        # Count how many protected groups would have been selected before exclusion.
        pg = [int(x) for x in protected_gids[b].detach().cpu().tolist()]
        for g in pg:
            if 0 <= g <= max_gid and bool(group_mask[g].item()):
                protected_selected_would += int((word_group[b] == g).sum().item())
                group_mask[g] = False
        for g in range(max_gid + 1):
            if bool(group_mask[g].item()):
                bg_selected[b] |= (word_group[b] == g)
    masked = input_ids.clone()
    if bg_selected.any():
        replace_mask = (torch.rand(input_ids.shape, generator=gen, device=device) < 0.8) & bg_selected
        masked[replace_mask] = int(tokenizer.mask_token_id)
        rand_mask = (torch.rand(input_ids.shape, generator=gen, device=device) < 0.5) & bg_selected & ~replace_mask
        if int(rand_mask.sum().item()) > 0:
            rand_ids = torch.randint(0, int(tokenizer.vocab_size), (int(rand_mask.sum().item()),), generator=gen, device=device)
            masked[rand_mask] = rand_ids
    labels = torch.full_like(input_ids, -100)
    forced = torch.zeros((bsz, seq), dtype=torch.bool, device=device)
    for b, positions in enumerate(answer_positions):
        for p0 in positions.tolist():
            p = int(p0)
            if 0 <= p < seq and int(attention_mask[b, p].item()) == 1:
                labels[b, p] = input_ids[b, p]
                masked[b, p] = int(tokenizer.mask_token_id)
                forced[b, p] = True
    return masked, labels, {
        "bg_corrupted_positions": int(bg_selected.sum().item()),
        "bg_masked_positions": int(((masked == int(tokenizer.mask_token_id)) & bg_selected).sum().item()),
        "answer_label_positions": int((labels != -100).sum().item()),
        "protected_positions_would_have_been_selected": int(protected_selected_would),
        "answer_bg_overlap_positions": int((forced & bg_selected).sum().item()),
    }


def compact(s: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["n_pairs", "n_four_condition_success", "n_query_orientation_success", "n_query_orientations", "mean_U", "mean_R", "mean_beta_pair_average", "mean_abs_alpha_pair_average", "mean_min_four_signed_margin"]
    return {k: s.get(k) for k in keys}


def evaluate(model, tokenizer, rows, device, seq_length, tag):
    scored, errors = h.score_rows(model, tokenizer, rows, device, seq_length)
    if errors:
        raise RuntimeError(f"scoring errors {tag}: {errors[:5]}")
    return h.summarize_metrics(h.pair_metrics(scored), tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/archive/functional_learning/data/revision_040e_protect_critical_corruption")
    ap.add_argument("--source-dir", default="experiments/archive/functional_learning/data/relation_first_repaired")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=40040)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--bg-mask-prob", type=float, default=0.15)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = pathlib.Path(args.source_dir)
    if not (source_dir / "repaired_scoring_rows.jsonl").exists():
        h.construct_outputs(args.seed, source_dir)
    rows = h.read_jsonl(source_dir / "repaired_scoring_rows.jsonl")
    train_rows = [r for r in rows if r.get("split") == "train" and r.get("role") != "NEUTRAL"]
    eval_train_rows = [r for r in rows if r.get("split") == "train"]
    eval_held_rows = [r for r in rows if r.get("split") == "held"]
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(h.MODEL_PATH), local_files_only=True, use_fast=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    model, load_info = h.load_trusted_model(device, private_scale=args.private_scale)
    ident = h.model_identity(model, load_info)
    optimizer, trainable_info = h.freeze_to_private_optimizer(model, args.lr, args.weight_decay)
    ds = ProtectedDataset(train_rows, tokenizer, args.seq_length)
    validation = {
        "n_examples": len(ds),
        "mean_protected_positions": sum(int(x["protected_position_count"]) for x in ds.items) / max(1, len(ds)),
        "protected_position_quantiles": None,
    }
    vals = sorted(int(x["protected_position_count"]) for x in ds.items)
    def q(p):
        if not vals: return None
        idx = p * (len(vals)-1); lo=int(idx); hi=min(len(vals)-1, lo+1); return vals[lo]*(hi-idx)+vals[hi]*(idx-lo)
    validation["protected_position_quantiles"] = {"min": vals[0], "p10": q(0.1), "median": q(0.5), "p90": q(0.9), "max": vals[-1]} if vals else {}
    dl_gen = torch.Generator(); dl_gen.manual_seed(args.seed + 123)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=dl_gen)
    mask_gen = torch.Generator(device=device); mask_gen.manual_seed(args.seed + 999)
    trajectory = []
    cum = collections.defaultdict(int)
    t0 = time.time()
    def rec(epoch: int, loss: float | None):
        st = evaluate(model, tokenizer, eval_train_rows, device, args.seq_length, f"protect_train_e{epoch}")
        sh = evaluate(model, tokenizer, eval_held_rows, device, args.seq_length, f"protect_held_e{epoch}")
        trajectory.append({"epoch": epoch, "loss": loss, "train": st, "held": sh})
        print(f"[protect e{epoch:04d}]" + (f" loss={loss:.4f}" if loss is not None else "") +
              f" train four={st['n_four_condition_success']}/{st['n_pairs']} orient={st['n_query_orientation_success']}/{st['n_query_orientations']} min4={st['mean_min_four_signed_margin']:+.3f}" +
              f" | held four={sh['n_four_condition_success']}/{sh['n_pairs']} orient={sh['n_query_orientation_success']}/{sh['n_query_orientations']} min4={sh['mean_min_four_signed_margin']:+.3f}", flush=True)
    rec(0, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_num = 0.0; tok_den = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_group = batch["word_group"].to(device)
            answer_positions = [x.to(device) for x in batch["answer_positions"]]
            protected_gids = [x.to(device) for x in batch["protected_gids"]]
            masked, labels, st = apply_corruption(input_ids, attention_mask, word_group, answer_positions, protected_gids, tokenizer, args.bg_mask_prob, mask_gen)
            out = model(input_ids=masked, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            ntok = int((labels != -100).sum().item())
            loss_num += float(loss.detach().cpu()) * ntok; tok_den += ntok
            for k, v in st.items(): cum[k] += int(v)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            rec(epoch, loss_num / max(1, tok_den))
    summary = {
        "status": "STEP040E_PROTECT_CRITICAL_CORRUPTION",
        "contract": "15% background corruption excluding critical evidence word groups; answer-only labels; trusted coherent86 private adapters; repaired relation-first rows",
        "model_identity": ident,
        "trainable_info": trainable_info,
        "validation": validation,
        "epochs": args.epochs,
        "bg_mask_prob": args.bg_mask_prob,
        "cumulative_stats": dict(cum),
        "trajectory": trajectory,
        "final_train": trajectory[-1]["train"],
        "final_held": trajectory[-1]["held"],
        "elapsed_sec": time.time() - t0,
    }
    (out_dir / "protect_critical_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Step040e protected-critical corruption\n\n"]
    md.append("15% background corruption was applied as in Step040b, but word groups containing entity A/B, source values A/B, or the shared replacement occurrence were excluded from corruption. Only final answer spans were supervised.\n\n")
    md.append(f"- Final train: {compact(summary['final_train'])}\n")
    md.append(f"- Final held: {compact(summary['final_held'])}\n")
    md.append(f"- Cumulative stats: {summary['cumulative_stats']}\n")
    md.append(f"- Protected positions per example: {validation}\n")
    (out_dir / "protect_critical_summary.md").write_text("".join(md), encoding="utf-8")
    del model, optimizer
    gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({
        "status": summary["status"],
        "out_dir": h.rel(out_dir),
        "final_train": compact(summary["final_train"]),
        "final_held": compact(summary["final_held"]),
        "cumulative_stats": summary["cumulative_stats"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
