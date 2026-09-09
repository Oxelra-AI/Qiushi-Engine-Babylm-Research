#!/usr/bin/env python3
from __future__ import annotations

"""BabyLM masked-LM trainer with a train-only cross-sentence counterfactual auxiliary.

The ordinary WWM stream is intentionally the same as babylm_masked_train_fullcycle:
official corpus selection, tokenization, masking, checkpoints, and HF evaluation
artifacts are preserved.  A separate counted auxiliary stream reads linked or
unlinked adjacent-sentence JSONL rows and adds a compatibility-ranking loss:

    score(s1, s2_dependent) > score(s1', s2_dependent)

where the score is computed from the final hidden representation at the designated
sentence-2 dependent token(s).  The auxiliary head is train-only and is saved
outside hf_model.
"""

import argparse
import itertools
import json
import math
import pathlib
import random
import re
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

# Import trusted full-cycle utilities from the same directory.
import babylm_masked_train_fullcycle as base

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?")


@dataclass
class AuxRow:
    row_id: int
    split: str
    source: str
    text: str
    text_perturbed: str
    text_random_control: str
    dep_span_s2: tuple[int, int]
    nondep_span_s2: tuple[int, int]
    edit_span_s1: tuple[int, int]
    s2_start_text: int
    s2_start_perturbed: int
    s2_start_random: int
    counted_words_pair: int
    counted_words_triple: int
    dependency_type: str


def first_word_span(text: str) -> tuple[int, int]:
    m = WORD_RE.search(text)
    if not m:
        return (0, min(1, len(text)))
    return (m.start(), m.end())


def last_word_span(text: str) -> tuple[int, int]:
    ms = list(WORD_RE.finditer(text))
    if not ms:
        return (0, min(1, len(text)))
    m = ms[-1]
    return (m.start(), m.end())


def load_aux_rows(path: pathlib.Path, split: str, max_rows: int = 0) -> tuple[list[AuxRow], dict]:
    rows: list[AuxRow] = []
    total_pair_words = 0
    total_triple_words = 0
    split_counts: dict[str, int] = {}
    dep_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            sp = str(obj.get("split", "train"))
            split_counts[sp] = split_counts.get(sp, 0) + 1
            if split != "all" and sp != split:
                continue
            dep = obj.get("linked_span", {}).get("s2_dependent")
            if dep and "dependent_span" in dep:
                a, b = dep["dependent_span"]
                dep_lower = dep.get("dependent_lower", "linked")
                dependency_type = obj.get("linked_span", {}).get("dependency_type", "linked")
            else:
                # Unlinked leakage-control rows: use initial token in s2 as a remote readout location.
                a, b = first_word_span(str(obj.get("s2", "")))
                dep_lower = "inferred_initial"
                dependency_type = "unlinked_or_inferred_initial"
            nondep_span = last_word_span(str(obj.get("s2", "")))
            edit_obj = obj.get("edit", {})
            if "span" in edit_obj:
                edit_span = (int(edit_obj["span"][0]), int(edit_obj["span"][1]))
            else:
                edit_span = first_word_span(str(obj.get("s1", "")))
            dep_counts[str(dep_lower)] = dep_counts.get(str(dep_lower), 0) + 1
            w = int(obj.get("words", len(str(obj["text"]).split())))
            wp = int(obj.get("words_perturbed", len(str(obj["text_perturbed"]).split())))
            wr = int(obj.get("words_random_control", len(str(obj.get("text_random_control", obj["text_perturbed"])).split())))
            row = AuxRow(
                row_id=int(obj.get("example_id", line_no - 1)),
                split=sp,
                source=str(obj.get("source", "aux_jsonl")),
                text=str(obj["text"]),
                text_perturbed=str(obj["text_perturbed"]),
                text_random_control=str(obj.get("text_random_control", obj["text_perturbed"])),
                dep_span_s2=(int(a), int(b)),
                nondep_span_s2=(int(nondep_span[0]), int(nondep_span[1])),
                edit_span_s1=(int(edit_span[0]), int(edit_span[1])),
                s2_start_text=int(obj.get("s2_start_char_text", 0)),
                s2_start_perturbed=int(obj.get("s2_start_char_perturbed", obj.get("s2_start_char_text", 0))),
                s2_start_random=int(obj.get("s2_start_char_random_control", obj.get("s2_start_char_perturbed", obj.get("s2_start_char_text", 0)))),
                counted_words_pair=w + wp,
                counted_words_triple=w + wp + wr,
                dependency_type=dependency_type,
            )
            rows.append(row)
            total_pair_words += row.counted_words_pair
            total_triple_words += row.counted_words_triple
            if max_rows and len(rows) >= max_rows:
                break
    meta = {
        "path": str(path),
        "split_requested": split,
        "rows_loaded": len(rows),
        "split_counts_all_seen": split_counts,
        "dependent_counts_loaded": dep_counts,
        "counted_words_pair_once": total_pair_words,
        "counted_words_triple_once": total_triple_words,
        "max_rows": max_rows,
    }
    if not rows:
        raise RuntimeError(f"no auxiliary rows loaded from {path} split={split}")
    return rows, meta


class CounterfactualDataset(Dataset):
    def __init__(self, rows: list[AuxRow], tokenizer, seq_length: int, readout_mode: str = "dependent"):
        if readout_mode not in {"dependent", "s2_nondependent", "s1_local"}:
            raise ValueError(f"unknown readout_mode {readout_mode}")
        self.rows = rows
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.readout_mode = readout_mode

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, text: str, s2_start: int, dep_span_s2: tuple[int, int], nondep_span_s2: tuple[int, int], edit_span_s1: tuple[int, int]) -> dict:
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.seq_length,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offsets = enc["offset_mapping"].squeeze(0)
        if self.readout_mode == "dependent":
            span_a, span_b = int(s2_start + dep_span_s2[0]), int(s2_start + dep_span_s2[1])
        elif self.readout_mode == "s2_nondependent":
            span_a, span_b = int(s2_start + nondep_span_s2[0]), int(s2_start + nondep_span_s2[1])
        else:
            span_a, span_b = int(edit_span_s1[0]), int(edit_span_s1[1])
        dep_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(input_ids.shape[0]):
            if attention_mask[i] == 0:
                continue
            a = int(offsets[i, 0].item()); b = int(offsets[i, 1].item())
            if b > span_a and a < span_b:
                dep_mask[i] = True
        # If tokenization/truncation lost the readout span, leave dep_mask empty; loss will skip.
        return {"input_ids": input_ids, "attention_mask": attention_mask, "dep_mask": dep_mask}

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        orig = self._encode(r.text, r.s2_start_text, r.dep_span_s2, r.nondep_span_s2, r.edit_span_s1)
        pert = self._encode(r.text_perturbed, r.s2_start_perturbed, r.dep_span_s2, r.nondep_span_s2, r.edit_span_s1)
        rand = self._encode(r.text_random_control, r.s2_start_random, r.dep_span_s2, r.nondep_span_s2, r.edit_span_s1)
        return {
            "orig": orig,
            "pert": pert,
            "rand": rand,
            "pair_words": torch.tensor(r.counted_words_pair, dtype=torch.long),
            "triple_words": torch.tensor(r.counted_words_triple, dtype=torch.long),
            "row_id": torch.tensor(r.row_id, dtype=torch.long),
        }


def collate_aux(batch: list[dict]) -> dict:
    out: dict = {}
    for name in ["orig", "pert", "rand"]:
        out[name] = {
            "input_ids": torch.stack([x[name]["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x[name]["attention_mask"] for x in batch]),
            "dep_mask": torch.stack([x[name]["dep_mask"] for x in batch]),
        }
    out["pair_words"] = torch.stack([x["pair_words"] for x in batch])
    out["triple_words"] = torch.stack([x["triple_words"] for x in batch])
    out["row_id"] = torch.stack([x["row_id"] for x in batch])
    return out


def score_from_hidden(hidden: torch.Tensor, dep_mask: torch.Tensor, head: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    counts = dep_mask.float().sum(dim=1)
    valid = counts > 0
    safe_counts = counts.clamp_min(1.0)
    rep = (hidden * dep_mask.float().unsqueeze(-1)).sum(dim=1) / safe_counts.unsqueeze(-1)
    score = head(rep).squeeze(-1)
    return score, valid


def aux_forward_loss(model, head, batch: dict, lambda_random: float = 0.0) -> tuple[torch.Tensor, dict]:
    device = next(model.parameters()).device
    # Concatenate variants for one efficient forward pass: [orig, pert, rand].
    names = ["orig", "pert", "rand"]
    input_ids = torch.cat([batch[n]["input_ids"].to(device, non_blocking=True) for n in names], dim=0)
    attention_mask = torch.cat([batch[n]["attention_mask"].to(device, non_blocking=True) for n in names], dim=0)
    dep_mask = torch.cat([batch[n]["dep_mask"].to(device, non_blocking=True) for n in names], dim=0)
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden = out.hidden_states[-1]
    score, valid = score_from_hidden(hidden, dep_mask, head)
    bsz = batch["pair_words"].shape[0]
    s_orig, s_pert, s_rand = score[:bsz], score[bsz:2*bsz], score[2*bsz:]
    v_orig, v_pert, v_rand = valid[:bsz], valid[bsz:2*bsz], valid[2*bsz:]
    pair_valid = v_orig & v_pert
    zero = hidden.new_tensor(0.0)
    if pair_valid.any():
        logits = torch.stack([s_orig[pair_valid], s_pert[pair_valid]], dim=1)
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
        loss_pair = F.cross_entropy(logits, target)
        acc_pair = (logits.argmax(dim=1) == 0).float().mean()
    else:
        loss_pair = zero
        acc_pair = zero
    loss = loss_pair
    acc_rand = zero
    rand_valid = v_orig & v_rand
    if lambda_random > 0 and rand_valid.any():
        logits_r = torch.stack([s_orig[rand_valid], s_rand[rand_valid]], dim=1)
        target_r = torch.zeros(logits_r.shape[0], dtype=torch.long, device=device)
        loss_rand = F.cross_entropy(logits_r, target_r)
        acc_rand = (logits_r.argmax(dim=1) == 0).float().mean()
        loss = loss + float(lambda_random) * loss_rand
    else:
        loss_rand = zero
    tele = {
        "cf_loss_pair": float(loss_pair.detach().cpu()),
        "cf_loss_random": float(loss_rand.detach().cpu()),
        "cf_acc_pair": float(acc_pair.detach().cpu()),
        "cf_acc_random": float(acc_rand.detach().cpu()),
        "cf_valid_pair": int(pair_valid.sum().detach().cpu()),
        "cf_valid_random": int(rand_valid.sum().detach().cpu()),
        "cf_batch_rows": int(bsz),
        "cf_pair_words": int(batch["pair_words"].sum().item()),
        "cf_triple_words": int(batch["triple_words"].sum().item()),
    }
    return loss, tele


@torch.no_grad()
def evaluate_aux(model, head, loader: DataLoader, max_batches: int, lambda_random: float) -> dict:
    model.eval(); head.eval()
    total_pair = total_rand = 0
    correct_pair = correct_rand = 0.0
    losses = []
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        loss, tele = aux_forward_loss(model, head, batch, lambda_random=lambda_random)
        losses.append(float(loss.detach().cpu()))
        total_pair += int(tele["cf_valid_pair"]); correct_pair += tele["cf_acc_pair"] * int(tele["cf_valid_pair"])
        total_rand += int(tele["cf_valid_random"]); correct_rand += tele["cf_acc_random"] * int(tele["cf_valid_random"])
    model.train(); head.train()
    return {
        "eval_batches": len(losses),
        "loss_mean": sum(losses)/len(losses) if losses else None,
        "pair_accuracy": correct_pair / total_pair if total_pair else None,
        "random_accuracy": correct_rand / total_rand if total_rand else None,
        "valid_pair": total_pair,
        "valid_random": total_rand,
    }


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Base trainer contract.
    p.add_argument("--dataset_id", default="BabyLM-community/BabyLM-2026-Strict-Small")
    p.add_argument("--dataset_revision", default="c92ab16b4f08858304b0815706065b3354d8fc0a")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_word_exposure", type=int, default=1_000_000, help="ordinary WWM word exposure")
    p.add_argument("--example_pool_words", type=int, default=10_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--words_per_example", type=int, default=160)
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--tokenizer_label", default="baseline16k")
    p.add_argument("--tokenization_summary_limit", type=int, default=0)
    p.add_argument("--mask_mode", choices=["token", "wwm"], default="wwm")
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--seq_len_schedule", default="")
    p.add_argument("--model_type", choices=["bert", "deberta_v2"], default="deberta_v2")
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    p.add_argument("--deberta_relative_attention", choices=["true", "false"], default="true")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=0)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extra_init_seed", type=int, default=-1)
    p.add_argument("--train_rng_seed", type=int, default=-1)
    p.add_argument("--log_every", type=int, default=50)
    # Counterfactual auxiliary.
    p.add_argument("--cf_aux_jsonl", default="")
    p.add_argument("--cf_aux_split", choices=["train", "heldout", "all"], default="train")
    p.add_argument("--cf_aux_max_rows", type=int, default=0)
    p.add_argument("--cf_readout_mode", choices=["dependent", "s2_nondependent", "s1_local"], default="dependent", help="Auxiliary readout location: true s2 dependent, matched nondependent s2 token, or local edited s1 span")
    p.add_argument("--cf_aux_batch_size", type=int, default=32)
    p.add_argument("--cf_aux_lambda", type=float, default=0.05)
    p.add_argument("--cf_random_lambda", type=float, default=0.0)
    p.add_argument("--cf_head_hidden", type=int, default=128)
    p.add_argument("--cf_every_n_steps", type=int, default=1)
    p.add_argument("--cf_eval_max_batches", type=int, default=32)
    p.add_argument("--cf_heldout_jsonl", default="", help="optional heldout aux JSONL; defaults to cf_aux_jsonl split=heldout")
    return p.parse_args()


def prepare_official_examples(args, out: pathlib.Path):
    pool_words = args.example_pool_words
    selected_words = args.max_word_exposure
    if pool_words < selected_words:
        pool_words = selected_words
    raw_dir, manifest_files = base.download_dataset(args, out)
    files = [raw_dir / n for n in base.TRAIN_FILES]
    total_words = sum(f["whitespace_words"] for f in manifest_files)
    if pool_words > total_words:
        pool_words = total_words
    pool_examples = list(base.iter_examples(files, pool_words, args.words_per_example))
    for i, ex in enumerate(pool_examples):
        ex.example_id = i
    pool_actual = sum(ex.words for ex in pool_examples)
    if pool_actual != pool_words:
        raise RuntimeError(f"pool word mismatch {pool_actual} vs {pool_words}")
    if selected_words > total_words * 10:
        raise RuntimeError(f"requested WWM exposure {selected_words} exceeds 10 official epochs ({total_words * 10})")
    examples: list[base.Example] = []
    actual_words = 0
    epoch = 0
    epoch_metadata = []
    while actual_words < selected_words:
        epoch_examples = list(pool_examples)
        shuffle_seed = args.seed + 1000003 * epoch
        random.Random(shuffle_seed).shuffle(epoch_examples)
        before = actual_words
        epoch_take_examples = 0
        for ex in epoch_examples:
            if actual_words >= selected_words:
                break
            source = f"epoch{epoch + 1}::{ex.source}"
            if actual_words + ex.words <= selected_words:
                examples.append(base.Example(ex.text, ex.words, example_id=ex.example_id, source=source))
                actual_words += ex.words; epoch_take_examples += 1
            else:
                take = selected_words - actual_words
                if take > 0:
                    examples.append(base.Example(" ".join(ex.text.split()[:take]), take, example_id=ex.example_id, source=source))
                    actual_words += take; epoch_take_examples += 1
                break
        epoch_metadata.append({"epoch_index": epoch + 1, "shuffle_seed": shuffle_seed, "words_added": actual_words - before, "examples_added": epoch_take_examples})
        epoch += 1
    if actual_words != selected_words:
        raise RuntimeError(f"word selection mismatch {actual_words} vs {selected_words}")
    selection_meta = {"data_source_type": "official_corpus_fullcycle_with_counterfactual_aux", "selection_epochs": epoch_metadata, "unique_official_pool_words": pool_words, "total_official_corpus_words": total_words}
    return examples, actual_words, pool_words, total_words, manifest_files, selection_meta


def main() -> None:
    args = build_args()
    if not args.cf_aux_jsonl:
        raise RuntimeError("--cf_aux_jsonl is required for counterfactual propagation trainer")
    start_time = time.time()
    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)

    examples, actual_words, pool_words, total_words, manifest_files, selection_meta = prepare_official_examples(args, out)
    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    (out / "example_order_manifest.json").write_text(json.dumps({
        "seed": args.seed, "example_pool_words_actual": pool_words, "selected_for_training_words": actual_words,
        "words_per_example": args.words_per_example, "num_consumed_examples": len(examples),
        "source_words_consumed": source_words, "consumed_example_ids_in_order": [ex.example_id for ex in examples],
        "mask_mode": args.mask_mode, "mask_prob": args.mask_prob, "tokenizer_label": args.tokenizer_label,
        "tokenizer_path": args.tokenizer_path, "tokenizer_vocab_size": len(tokenizer), **selection_meta,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    tokenization_summary = base.summarize_tokenization_coupling(examples, tokenizer, args.max_seq_length, args.tokenization_summary_limit)
    (out / "tokenization_coupling_summary.json").write_text(json.dumps(tokenization_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    wwm_dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    wwm_loader = DataLoader(wwm_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=2, pin_memory=torch.cuda.is_available())

    aux_rows, aux_meta = load_aux_rows(pathlib.Path(args.cf_aux_jsonl), args.cf_aux_split, args.cf_aux_max_rows)
    aux_dataset = CounterfactualDataset(aux_rows, tokenizer, args.max_seq_length, readout_mode=args.cf_readout_mode)
    aux_loader = DataLoader(aux_dataset, batch_size=args.cf_aux_batch_size, shuffle=True, collate_fn=collate_aux, num_workers=0, pin_memory=torch.cuda.is_available())
    aux_iter = itertools.cycle(aux_loader)
    heldout_path = pathlib.Path(args.cf_heldout_jsonl) if args.cf_heldout_jsonl else pathlib.Path(args.cf_aux_jsonl)
    held_rows, held_meta = load_aux_rows(heldout_path, "heldout", 0)
    held_loader = DataLoader(CounterfactualDataset(held_rows, tokenizer, args.max_seq_length, readout_mode=args.cf_readout_mode), batch_size=args.cf_aux_batch_size, shuffle=False, collate_fn=collate_aux, num_workers=0)

    if args.extra_init_seed >= 0:
        base.reset_all_rng(args.extra_init_seed)
    if args.max_position_embeddings < args.max_seq_length + 8:
        args.max_position_embeddings = args.max_seq_length + 8
    model = base.build_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        base.reset_all_rng(args.train_rng_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    cf_head = torch.nn.Sequential(torch.nn.Linear(args.hidden_size, args.cf_head_hidden), torch.nn.Tanh(), torch.nn.Linear(args.cf_head_hidden, 1)).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    input_embedding_params = model.get_input_embeddings().weight.numel()
    cf_head_params = sum(p.numel() for p in cf_head.parameters())
    optim_params = list(model.parameters()) + list(cf_head.parameters())
    optim = torch.optim.AdamW(optim_params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    total_steps = len(wwm_loader)
    schedule_total = args.lr_total_steps if args.lr_total_steps and args.lr_total_steps > 0 else total_steps
    if schedule_total < total_steps:
        raise RuntimeError(f"lr_total_steps {schedule_total} < actual steps {total_steps}")
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)
    seq_schedule: list[tuple[float, int]] = []
    if args.seq_len_schedule:
        for part in args.seq_len_schedule.split(","):
            t, L = part.split(":"); seq_schedule.append((float(t), int(L)))
        seq_schedule.sort()
    gen = torch.Generator(device=device); gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    model.train(); cf_head.train()
    cumulative_words = 0
    cumulative_aux_pair_words = 0
    cumulative_aux_triple_words = 0
    loss_values: list[float] = []
    mlm_loss_values: list[float] = []
    cf_loss_values: list[float] = []
    cf_pair_acc_weighted = 0.0; cf_pair_valid_total = 0
    masked_token_values: list[int] = []; seq_len_values: list[int] = []
    saved_checkpoints: list[dict] = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words and args.checkpoint_words > 0 else None
    log_path = out / "training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(wwm_loader, 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            frac = (step - 1) / max(1, schedule_total)
            cur_len = base.seq_length_for_progress(frac, seq_schedule, args.seq_length) if seq_schedule else args.seq_length
            cur_len = min(cur_len, args.max_seq_length)
            input_ids = input_ids[:, :cur_len].contiguous(); attention_mask = attention_mask[:, :cur_len].contiguous(); word_group = word_group[:, :cur_len].contiguous()
            masked_inputs, labels = base.apply_masking(input_ids, attention_mask, word_group, tokenizer, args.mask_mode, args.mask_prob, gen)
            optim.zero_grad(set_to_none=True)
            out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            mlm_loss = out_model.loss
            if mlm_loss is None:
                raise RuntimeError("model returned no MLM loss")
            cf_loss = mlm_loss.new_tensor(0.0)
            cf_tele = {"cf_loss_pair": 0.0, "cf_loss_random": 0.0, "cf_acc_pair": 0.0, "cf_acc_random": 0.0, "cf_valid_pair": 0, "cf_valid_random": 0, "cf_batch_rows": 0, "cf_pair_words": 0, "cf_triple_words": 0}
            if args.cf_aux_lambda > 0 and args.cf_every_n_steps > 0 and step % args.cf_every_n_steps == 0:
                aux_batch = next(aux_iter)
                cf_loss, cf_tele = aux_forward_loss(model, cf_head, aux_batch, lambda_random=args.cf_random_lambda)
                cumulative_aux_pair_words += int(cf_tele["cf_pair_words"])
                cumulative_aux_triple_words += int(cf_tele["cf_triple_words"])
                if int(cf_tele["cf_valid_pair"]):
                    cf_pair_acc_weighted += float(cf_tele["cf_acc_pair"]) * int(cf_tele["cf_valid_pair"])
                    cf_pair_valid_total += int(cf_tele["cf_valid_pair"])
            loss = mlm_loss + float(args.cf_aux_lambda) * cf_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
            optim.step(); sched.step()
            cumulative_words += words
            loss_float = float(loss.detach().cpu())
            loss_values.append(loss_float); mlm_loss_values.append(float(mlm_loss.detach().cpu())); cf_loss_values.append(float(cf_loss.detach().cpu()))
            n_pred = int((labels != -100).sum().item())
            masked_token_values.append(n_pred); seq_len_values.append(cur_len)
            rec = {"step": step, "loss": loss_float, "loss_mlm": mlm_loss_values[-1], "loss_cf": cf_loss_values[-1],
                   "lr": float(sched.get_last_lr()[0]), "batch_words": words, "cumulative_wwm_word_exposure": cumulative_words,
                   "cumulative_aux_pair_words": cumulative_aux_pair_words, "cumulative_aux_triple_words": cumulative_aux_triple_words,
                   "cumulative_counted_words_pair_rule": cumulative_words + cumulative_aux_pair_words,
                   "cumulative_counted_words_triple_rule": cumulative_words + cumulative_aux_triple_words,
                   "seq_len": cur_len, "masked_tokens": n_pred, "elapsed_sec": time.time() - start_time, **cf_tele}
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)
            while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                name = "chck_1M" if next_ckpt < 1_000_000 else (f"chck_{next_ckpt // 1_000_000}M" if next_ckpt % 1_000_000 == 0 else f"chck_{next_ckpt}w")
                cp = out / "hf_model" / name
                base.save_hf_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({"name": name, "target_wwm_word_exposure": next_ckpt, "actual_cumulative_wwm_word_exposure": cumulative_words, "path": str(cp)})
                print(json.dumps({"event": "checkpoint_saved", "name": name, "cum_wwm_words": cumulative_words}), flush=True)
                next_ckpt += args.checkpoint_words

    base.save_hf_checkpoint(model, tokenizer, out / "hf_model")
    if not saved_checkpoints:
        cp = out / "hf_model" / "chck_1M"; base.save_hf_checkpoint(model, tokenizer, cp)
        saved_checkpoints.append({"name": "chck_1M", "target_wwm_word_exposure": args.checkpoint_words, "actual_cumulative_wwm_word_exposure": cumulative_words, "path": str(cp)})
    torch.save(cf_head.state_dict(), out / "counterfactual_head_train_only.pt")
    (out / "counterfactual_aux_config.json").write_text(json.dumps({
        "cf_aux_jsonl": args.cf_aux_jsonl, "cf_aux_split": args.cf_aux_split, "cf_readout_mode": args.cf_readout_mode, "cf_aux_lambda": args.cf_aux_lambda,
        "cf_random_lambda": args.cf_random_lambda, "cf_aux_batch_size": args.cf_aux_batch_size, "cf_every_n_steps": args.cf_every_n_steps,
        "cf_head_hidden": args.cf_head_hidden, "note": "Counterfactual head is train-only and intentionally excluded from hf_model."
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    held_eval = evaluate_aux(model, cf_head, held_loader, args.cf_eval_max_batches, args.cf_random_lambda)
    metrics = {
        "variant": f"masked_{args.mask_mode}_counterfactual_propagation",
        "backend": "mlm", "model_family": model.__class__.__name__, "model_type": args.model_type,
        "parameter_count": param_count, "embedding_parameter_count": input_embedding_params,
        "non_embedding_parameter_count": param_count - input_embedding_params, "counterfactual_head_parameter_count": cf_head_params,
        "train_parameter_count_including_cf_head": param_count + cf_head_params,
        "vocab_size": len(tokenizer), "tokenizer_label": args.tokenizer_label, "tokenizer_path": args.tokenizer_path,
        "wwm_word_exposure": cumulative_words, "aux_pair_words_seen": cumulative_aux_pair_words, "aux_triple_words_seen": cumulative_aux_triple_words,
        "total_counted_words_pair_rule": cumulative_words + cumulative_aux_pair_words,
        "total_counted_words_triple_rule": cumulative_words + cumulative_aux_triple_words,
        "example_pool_words_actual": pool_words, "selected_for_training_words": actual_words,
        "loss_first": loss_values[0] if loss_values else None, "loss_last": loss_values[-1] if loss_values else None,
        "loss_mlm_first": mlm_loss_values[0] if mlm_loss_values else None, "loss_mlm_last": mlm_loss_values[-1] if mlm_loss_values else None,
        "loss_cf_first": cf_loss_values[0] if cf_loss_values else None, "loss_cf_last": cf_loss_values[-1] if cf_loss_values else None,
        "cf_train_pair_accuracy_weighted": cf_pair_acc_weighted / cf_pair_valid_total if cf_pair_valid_total else None,
        "cf_train_valid_pair_total": cf_pair_valid_total, "cf_heldout_eval": held_eval, "cf_readout_mode": args.cf_readout_mode,
        "lr_schedule_total_steps": schedule_total, "actual_training_steps": total_steps, "mask_mode": args.mask_mode, "mask_prob": args.mask_prob,
        "masked_tokens_total": sum(masked_token_values), "masked_tokens_mean_per_step": sum(masked_token_values)/len(masked_token_values) if masked_token_values else None,
        "masked_tokens_per_wwm_whitespace_word": sum(masked_token_values)/cumulative_words if cumulative_words else None,
        "unique_training_seq_lengths": sorted(set(seq_len_values)), "tokenization_coupling_summary": tokenization_summary,
        "seq_length": args.seq_length, "max_seq_length": args.max_seq_length, "max_position_embeddings": max(args.max_position_embeddings, args.max_seq_length + 8),
        "hidden_size": args.hidden_size, "n_layer": args.n_layer, "n_head": args.n_head, "ffn_mult": args.ffn_mult,
        "position_buckets": args.position_buckets, "max_relative_positions": args.max_relative_positions,
        "deberta_relative_attention": args.deberta_relative_attention, "deberta_pos_att_type": args.deberta_pos_att_type,
        "seed": args.seed, "extra_init_seed": args.extra_init_seed, "train_rng_seed": args.train_rng_seed,
        "saved_checkpoints": saved_checkpoints, "source_words_consumed": source_words,
        "counterfactual_aux_meta": aux_meta, "counterfactual_heldout_meta": held_meta, **selection_meta,
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "data_manifest.json").write_text(json.dumps({
        "dataset_id": args.dataset_id, "dataset_revision": args.dataset_revision, "files": manifest_files,
        "total_dataset_whitespace_words_counted": total_words, "selected_for_wwm_stream_whitespace_words": actual_words,
        "counterfactual_aux_jsonl": args.cf_aux_jsonl, "counterfactual_aux_readout_mode": args.cf_readout_mode, "counterfactual_aux_meta": aux_meta,
        "model_type": args.model_type, "tokenizer_label": args.tokenizer_label, "tokenizer_path": args.tokenizer_path,
        "tokenizer_vocab_size": len(tokenizer), "tokenization_coupling_summary_file": str(out / "tokenization_coupling_summary.json"), **selection_meta,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "param_count": param_count, "cf_head_params": cf_head_params,
                      "loss_first": metrics["loss_first"], "loss_last": metrics["loss_last"],
                      "wwm_word_exposure": cumulative_words, "aux_pair_words_seen": cumulative_aux_pair_words,
                      "heldout_pair_acc": held_eval.get("pair_accuracy"), "checkpoints": [c["name"] for c in saved_checkpoints]}), flush=True)


if __name__ == "__main__":
    main()
