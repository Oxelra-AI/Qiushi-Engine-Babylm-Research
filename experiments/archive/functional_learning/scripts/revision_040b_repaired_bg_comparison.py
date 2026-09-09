#!/usr/bin/env python3
"""Step040b: matched background-objective comparison on repaired relation-first rows.

This uses the repaired research harness (trusted coherent86 loader, shared-new
recipient-only contrasts, explicit final-span scorer) and tests whether adding
background MLM loss changes acquisition when background input corruption is
matched.

Arms:
  corrupted_answer_only: background tokens are corrupted in the input, but only
    final answer span tokens are supervised.
  corrupted_answer_plus_bg: identical background corruption stream, final answer
    span supervised, and corrupted background positions supervised.

This is not a BabyLM continuation; it is a bounded acquisition/mechanism test.
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
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))

import relation_first_repaired_harness as h  # noqa: E402


def is_word_start(token: str) -> bool:
    return token.startswith("Ġ") or token.startswith("▁")


class BgRowDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tokenizer, seq_length: int):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(int(x) for x in tokenizer.all_special_ids)
        self.items: List[Dict[str, Any]] = []
        self._word_start_cache: Dict[int, bool] = {}
        for row in self.rows:
            full, start, end = h.full_text_and_span(row, str(row["answer_text"]))
            enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True,
                            return_tensors="pt", max_length=seq_length, truncation=True,
                            padding="max_length")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
            ans_pos = h.locate_span_positions(offsets, start, end)
            if not ans_pos:
                raise ValueError(f"answer span unavailable: {row['pair_id']} {row['row_type']}")
            covered_start = min(offsets[p][0] for p in ans_pos)
            covered_end = max(offsets[p][1] for p in ans_pos)
            if covered_start > start or covered_end < end:
                raise ValueError(f"answer span partial: {row['pair_id']} {row['row_type']} span={start,end} covered={covered_start,covered_end}")
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
            answer_gids = sorted({int(word_group[p].item()) for p in ans_pos if int(word_group[p].item()) >= 0})
            if not answer_gids:
                raise ValueError(f"answer has no word group: {row['pair_id']} {row['row_type']}")
            self.items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "word_group": word_group,
                "answer_positions": torch.tensor([int(p) for p in ans_pos], dtype=torch.long),
                "answer_gids": torch.tensor(answer_gids, dtype=torch.long),
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
    # Answer spans have variable length, so keep lists.
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "word_group": torch.stack([b["word_group"] for b in batch]),
        "answer_positions": [b["answer_positions"] for b in batch],
        "answer_gids": [b["answer_gids"] for b in batch],
    }


def apply_corruption(input_ids: torch.Tensor, attention_mask: torch.Tensor, word_group: torch.Tensor,
                     answer_positions: Sequence[torch.Tensor], answer_gids: Sequence[torch.Tensor], tokenizer,
                     bg_mask_prob: float, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    device = input_ids.device
    bsz, seq = input_ids.shape
    bg_selected = torch.zeros((bsz, seq), dtype=torch.bool, device=device)
    for b in range(bsz):
        max_gid = int(word_group[b].max().item())
        if max_gid < 0 or bg_mask_prob <= 0:
            continue
        group_mask = torch.rand(max_gid + 1, generator=gen, device=device) < bg_mask_prob
        # Exclude every word group touched by the final answer span.
        for g in answer_gids[b].tolist():
            if 0 <= int(g) <= max_gid:
                group_mask[int(g)] = False
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
    answer_labels = torch.full_like(input_ids, -100)
    bg_labels = torch.full_like(input_ids, -100)
    forced = torch.zeros((bsz, seq), dtype=torch.bool, device=device)
    for b, positions in enumerate(answer_positions):
        for pos0 in positions.tolist():
            pos = int(pos0)
            if 0 <= pos < seq and int(attention_mask[b, pos].item()) == 1:
                forced[b, pos] = True
                answer_labels[b, pos] = input_ids[b, pos]
                masked[b, pos] = int(tokenizer.mask_token_id)
    bg_labels[bg_selected] = input_ids[bg_selected]
    stats = {
        "bg_corrupted_positions": int(bg_selected.sum().item()),
        "answer_label_positions": int((answer_labels != -100).sum().item()),
        "forced_answer_mask_positions": int(forced.sum().item()),
        "bg_masked_positions": int(((masked == int(tokenizer.mask_token_id)) & bg_selected).sum().item()),
        "answer_bg_overlap_positions": int((forced & bg_selected).sum().item()),
    }
    return masked, answer_labels, bg_labels, stats


def ce_on_labels(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
    mask = labels != -100
    n = int(mask.sum().item())
    if n == 0:
        return torch.tensor(0.0, device=logits.device), 0
    return F.cross_entropy(logits[mask].view(-1, logits.size(-1)), labels[mask].view(-1), reduction="mean"), n


def compact(s: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["n_pairs", "n_four_condition_success", "n_query_orientation_success", "n_query_orientations", "mean_U", "mean_R", "mean_beta_pair_average", "mean_abs_alpha_pair_average", "mean_min_four_signed_margin"]
    return {k: s.get(k) for k in keys}


def run_arm(name: str, use_bg_loss: bool, rows: Sequence[Dict[str, Any]], tokenizer, args, device: torch.device) -> Dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, load_info = h.load_trusted_model(device, private_scale=args.private_scale)
    ident = h.model_identity(model, load_info)
    optimizer, trainable_info = h.freeze_to_private_optimizer(model, args.lr, args.weight_decay)
    train_rows = [r for r in rows if r.get("split") == "train" and r.get("role") != "NEUTRAL"]
    eval_train_rows = [r for r in rows if r.get("split") == "train"]
    eval_held_rows = [r for r in rows if r.get("split") == "held"]
    ds = BgRowDataset(train_rows, tokenizer, args.seq_length)
    dl_gen = torch.Generator()
    dl_gen.manual_seed(args.seed + 123)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=dl_gen)
    mask_gen = torch.Generator(device=device)
    mask_gen.manual_seed(args.seed + 999)  # identical corruption stream across arms
    trajectory: List[Dict[str, Any]] = []
    cum = collections.defaultdict(int)

    def evaluate(epoch: int, answer_loss: float | None, bg_loss: float | None):
        scored_train, err_train = h.score_rows(model, tokenizer, eval_train_rows, device, args.seq_length)
        scored_held, err_held = h.score_rows(model, tokenizer, eval_held_rows, device, args.seq_length)
        if err_train or err_held:
            raise RuntimeError(f"eval scoring errors: train={err_train[:5]} held={err_held[:5]}")
        st = h.summarize_metrics(h.pair_metrics(scored_train), f"{name}_train_e{epoch}")
        sh = h.summarize_metrics(h.pair_metrics(scored_held), f"{name}_held_e{epoch}")
        entry = {"epoch": epoch, "answer_loss": answer_loss, "bg_loss": bg_loss, "train": st, "held": sh}
        trajectory.append(entry)
        print(f"[{name} e{epoch:04d}]" + (f" ansL={answer_loss:.4f} bgL={bg_loss:.4f}" if answer_loss is not None else "") +
              f" train four={st['n_four_condition_success']}/{st['n_pairs']} orient={st['n_query_orientation_success']}/{st['n_query_orientations']} min4={st['mean_min_four_signed_margin']:+.3f}" +
              f" | held four={sh['n_four_condition_success']}/{sh['n_pairs']} orient={sh['n_query_orientation_success']}/{sh['n_query_orientations']} min4={sh['mean_min_four_signed_margin']:+.3f}", flush=True)

    evaluate(0, None, None)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        ans_num = bg_num = 0.0
        ans_den = bg_den = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_group = batch["word_group"].to(device)
            answer_positions = [x.to(device) for x in batch["answer_positions"]]
            answer_gids = [x.to(device) for x in batch["answer_gids"]]
            masked, answer_labels, bg_labels, st = apply_corruption(
                input_ids, attention_mask, word_group, answer_positions, answer_gids,
                tokenizer, args.bg_mask_prob, mask_gen)
            out = model(input_ids=masked, attention_mask=attention_mask)
            logits = out.logits
            ans_loss, n_ans = ce_on_labels(logits, answer_labels)
            bg_loss, n_bg = ce_on_labels(logits, bg_labels)
            loss = ans_loss + (bg_loss if use_bg_loss else torch.tensor(0.0, device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ans_num += float(ans_loss.detach().cpu()) * max(1, n_ans)
            bg_num += float(bg_loss.detach().cpu()) * max(1, n_bg)
            ans_den += n_ans
            bg_den += n_bg
            for k, v in st.items():
                cum[k] += int(v)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            evaluate(epoch, ans_num / max(1, ans_den), bg_num / max(1, bg_den) if bg_den else 0.0)
    result = {
        "arm": name,
        "use_bg_loss": use_bg_loss,
        "model_identity": ident,
        "trainable_info": trainable_info,
        "bg_mask_prob": args.bg_mask_prob,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_train_examples": len(ds),
        "cumulative_stats": dict(cum),
        "trajectory": trajectory,
        "final_train": trajectory[-1]["train"],
        "final_held": trajectory[-1]["held"],
        "elapsed_sec": time.time() - t0,
    }
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_markdown(out_dir: pathlib.Path, summary: Dict[str, Any]) -> None:
    lines = ["# Step040b matched background comparison on repaired relation-first rows\n\n"]
    lines.append("Both arms use the same repaired rows, trusted coherent86 loader, private-adapter-only optimizer, explicit final-span scoring, and identical background corruption stream. The only arm difference is whether corrupted background positions contribute MLM loss.\n\n")
    for arm in summary["arms"]:
        fh = compact(arm["final_held"])
        ft = compact(arm["final_train"])
        st = arm["cumulative_stats"]
        lines.append(f"## {arm['arm']}\n\n")
        lines.append(f"- Train final: {ft}\n")
        lines.append(f"- Held final: {fh}\n")
        lines.append(f"- Cumulative positions: answer={st.get('answer_label_positions')}, bg_corrupted={st.get('bg_corrupted_positions')}, answer/bg overlap={st.get('answer_bg_overlap_positions')}\n\n")
    lines.append("## Files\n\n")
    lines.append(f"- summary JSON: `{h.rel(out_dir / 'bg_comparison_summary.json')}`\n")
    (out_dir / "bg_comparison_summary.md").write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/archive/functional_learning/data/revision_040b_repaired_bg_comparison")
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
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(h.MODEL_PATH), local_files_only=True, use_fast=True)
    arms = [
        run_arm("corrupted_answer_only", False, rows, tokenizer, args, device),
        run_arm("corrupted_answer_plus_bg", True, rows, tokenizer, args, device),
    ]
    summary = {
        "status": "STEP040B_REPAIRED_BACKGROUND_COMPARISON",
        "source_rows": h.rel(source_dir / "repaired_scoring_rows.jsonl"),
        "contract": "shared-new recipient-only rows; trusted coherent86 private adapters; identical background corruption stream; explicit final-span simultaneous scorer",
        "arms": arms,
    }
    (out_dir / "bg_comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out_dir, summary)
    print(json.dumps({
        "status": summary["status"],
        "out_dir": h.rel(out_dir),
        "arms": {arm["arm"]: {"final_train": compact(arm["final_train"]), "final_held": compact(arm["final_held"]), "stats": arm["cumulative_stats"]} for arm in arms},
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
