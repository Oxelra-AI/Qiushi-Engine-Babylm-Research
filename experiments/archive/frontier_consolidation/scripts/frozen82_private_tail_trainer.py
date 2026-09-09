#!/usr/bin/env python3
"""research frozen-82M slow-path private-tail trainer.

This is a prepared trainer for the anchored follow-up.  It is not
launched automatically by this script file.  It starts from the verified above-41.8
scale1.75 chck_82M function, freezes the entire slow path, attaches a fresh
zero-output private residual, and trains only `.private_adapter.*` on sparse
source-conditioned/source-free auxiliary views plus deterministic neutrality KL on
already-charged main-stream rows.

Legal accounting:
  total_exposure = initial_consumed_words + tail_main_words + tail_aux_words
The default boundary is the exact research extraction from the original ladder:
  initial_consumed_words = 82,012,495
  skip_rows = 530,944
  max_tail_charged_words = 17,987,505

Why deterministic neutrality: in the from-scratch research trainer the neutral KL was
computed in train mode and therefore included dropout noise.  For a frozen score-
bearing slow function, the neutral term should penalize only the fresh private path's
functional deviation, not stochastic dropout differences.  This trainer temporarily
uses eval mode for the private-on/private-off neutral pair; aux CE still uses train
mode.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DebertaV2Config

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(SCRIPTS))
from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM  # noqa: E402

DEFAULT_ENDPOINT = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
DEFAULT_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
DEFAULT_AUX = _public_path('experiments/archive/frontier_consolidation/data/sparse_aux_pair_data/top20/sparse_aux_pair_data_top20.json')

DEFAULTS = dict(
    initial_consumed_words=82_012_495,
    full_cap_words=100_000_000,
    max_tail_charged_words=17_987_505,
    skip_rows=530_944,
    hidden_size=480,  # sanity only, inherited from endpoint config
    n_layer=8,
    batch_size=256,
    seq_length=256,
    aux_max_length=464,
    learning_rate=0.001,
    warmup_fraction=0.06,
    lr_total_steps=455,
    weight_decay=0.01,
    mask_prob=0.15,
    checkpoint_words=1_000_000,
    log_every=50,
    private_adapter_bottleneck=128,
    private_adapter_scale=1.0,
    aux_lambda=1.0,
    neutral_lambda=1.0,
    neutral_subsample=32,
    aux_pair_shuffle_seed=43022,
    train_rng_seed=43023,
    max_updates=0,
)


@dataclass
class Ex:
    text: str
    words: int
    source: str
    example_id: int
    row_index: int


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def reset_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def load_examples_tail(path: Path, skip_rows: int, max_main_words: int) -> list[Ex]:
    examples: list[Ex] = []
    selected = 0
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < skip_rows:
                continue
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch row={idx} example_id={obj.get('example_id')}")
            if selected + words > max_main_words:
                break
            examples.append(Ex(text=text, words=words, source=str(obj.get("source", "")),
                               example_id=int(obj.get("example_id", -1)), row_index=idx))
            selected += words
    return examples


class TailDataset(Dataset):
    def __init__(self, examples: list[Ex], tokenizer, seq_length: int, pair_eids: set[int]):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pair_eids = pair_eids
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start_cache: dict[int, bool] = {}

    def _word_start(self, tid: int) -> bool:
        v = self._word_start_cache.get(tid)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and is_word_start(str(s)))
            self._word_start_cache[tid] = v
        return v

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text, add_special_tokens=False, truncation=True,
            max_length=self.seq_length, padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        word_group = torch.full_like(input_ids, -1)
        gid = -1
        for i in range(input_ids.shape[0]):
            if attention_mask[i] == 0:
                continue
            tid = int(input_ids[i])
            if tid in self.special_ids:
                continue
            if gid < 0 or self._word_start(tid) or i == 0:
                gid += 1
            word_group[i] = gid
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_group": word_group,
            "words": ex.words,
            "example_id": ex.example_id,
            "row_index": ex.row_index,
            "is_pair": ex.example_id in self.pair_eids,
        }


def collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
        "example_id": torch.tensor([x["example_id"] for x in batch], dtype=torch.long),
        "row_index": torch.tensor([x["row_index"] for x in batch], dtype=torch.long),
        "is_pair": torch.tensor([x["is_pair"] for x in batch], dtype=torch.bool),
    }


def apply_wwm(input_ids, attention_mask, word_group, tokenizer, mask_prob: float, gen: torch.Generator):
    device = input_ids.device
    B, S = input_ids.shape
    labels = input_ids.clone()
    mask_token_id = int(tokenizer.mask_token_id)
    special = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special)
    select = torch.zeros_like(candidate)
    for b in range(B):
        groups = word_group[b]
        valid = torch.unique(groups[groups >= 0])
        if valid.numel() == 0:
            continue
        r = torch.rand(valid.numel(), generator=gen, device=device)
        chosen = valid[r < mask_prob]
        if chosen.numel() > 0:
            select[b] = torch.isin(groups, chosen) & candidate[b]
    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True
    labels[~select] = -100
    masked = input_ids.clone()
    r = torch.rand(B, S, generator=gen, device=device)
    masked[select & (r < 0.8)] = mask_token_id
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    if rand_tok.any():
        masked[rand_tok] = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),),
                                         generator=gen, device=device)
    return masked, labels


def load_aux_pair_data(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["pair_data"], raw["summary"]


@dataclass
class AuxUnit:
    source_ids: list[int]
    source_words: int
    rewrite_ids: list[int]
    rewrite_word_group: list[int]
    rewrite_word_groups_to_mask: set[int]
    rewrite_words: int


def collect_aux_units(word_group, labels, pair_batch_indices: list[int], pair_records: list[dict]) -> list[AuxUnit]:
    units: list[AuxUnit] = []
    for bi, rec in zip(pair_batch_indices, pair_records):
        masked_pos = labels[bi] != -100
        if not masked_pos.any():
            continue
        masked_full_wgs = set(int(x) for x in torch.unique(word_group[bi][masked_pos]).tolist() if x >= 0)
        if not masked_full_wgs:
            continue
        for pr in rec["pairs"]:
            fmap = {int(k): int(v) for k, v in pr["full_wg_to_rw_wg"].items()}
            rw_wgs = {rwg for fwg, rwg in fmap.items() if fwg in masked_full_wgs}
            if not rw_wgs:
                continue
            units.append(AuxUnit(
                source_ids=list(pr["source_ids"]),
                source_words=int(pr["source_words"]),
                rewrite_ids=list(pr["rw_ids"]),
                rewrite_word_group=list(pr["rw_word_group"]),
                rewrite_word_groups_to_mask=rw_wgs,
                rewrite_words=int(pr.get("rewrite_words", max(1, len(pr["rw_ids"]) // 2))),
            ))
    return units


def source_assignment(units: list[AuxUnit], mode: str, seed: int, step: int) -> list[tuple[list[int], int]]:
    if mode == "aligned":
        return [(u.source_ids, u.source_words) for u in units]
    if mode == "neutral_only":
        return []
    if len(units) < 2:
        return []
    rng = random.Random(seed + 1000003 * step)
    idxs = list(range(len(units)))
    rng.shuffle(idxs)
    src_from = idxs[1:] + idxs[:1]
    assign: list[tuple[list[int], int]] = [([], 0)] * len(units)
    for target_i, src_i in zip(idxs, src_from):
        assign[target_i] = (units[src_i].source_ids, units[src_i].source_words)
    return assign


def build_view_batches(units: list[AuxUnit], mode: str, seed: int, step: int,
                       cls_id: int, sep_id: int, mask_id: int, pad_id: int,
                       max_len: int, device: torch.device):
    if mode == "neutral_only" or not units:
        return None, 0, 0, 0
    assigned = source_assignment(units, mode, seed, step)
    if not assigned:
        return None, 0, 0, 0
    seqs: list[list[int]] = []
    labs: list[list[int]] = []
    charged_words = 0
    n_cond = 0
    n_free = 0
    for unit, (source_ids, source_words) in zip(units, assigned):
        masked_rw = list(unit.rewrite_ids)
        lab_rw = [-100] * len(unit.rewrite_ids)
        for j, g in enumerate(unit.rewrite_word_group):
            if g in unit.rewrite_word_groups_to_mask:
                lab_rw[j] = unit.rewrite_ids[j]
                masked_rw[j] = mask_id
        if all(x == -100 for x in lab_rw):
            continue
        cond_seq = [cls_id] + list(source_ids) + [sep_id] + masked_rw + [sep_id]
        cond_lab = [-100] + [-100] * len(source_ids) + [-100] + lab_rw + [-100]
        free_seq = [cls_id] + masked_rw + [sep_id]
        free_lab = [-100] + lab_rw + [-100]
        if len(cond_seq) > max_len or len(free_seq) > max_len:
            continue
        seqs.append(cond_seq); labs.append(cond_lab); n_cond += 1
        seqs.append(free_seq); labs.append(free_lab); n_free += 1
        charged_words += int(source_words) + 2 * int(unit.rewrite_words)
    if not seqs:
        return None, 0, 0, 0
    B = len(seqs)
    inp = torch.full((B, max_len), pad_id, dtype=torch.long)
    att = torch.zeros((B, max_len), dtype=torch.long)
    lab = torch.full((B, max_len), -100, dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        att[i, :len(s)] = 1
        lab[i, :len(l)] = torch.tensor(l, dtype=torch.long)
    return (inp.to(device), att.to(device), lab.to(device)), charged_words, n_cond, n_free


def lr_at_update(update_index0: int, schedule_total: int, warmup: int, peak_lr: float) -> float:
    if update_index0 < warmup:
        return peak_lr * update_index0 / max(1, warmup)
    p = (update_index0 - warmup) / max(1, schedule_total - warmup)
    p = min(max(p, 0.0), 1.0)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * p))


def load_frozen_private_model(endpoint: Path, args):
    cfg = DebertaV2Config.from_pretrained(str(endpoint), local_files_only=True)
    cfg.private_adapter_bottleneck = int(args.private_adapter_bottleneck)
    cfg.private_adapter_scale = float(args.private_adapter_scale)
    cfg.private_adapter_enabled = True
    model = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
    sd = load_file(str(endpoint / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    tied_missing = {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}
    bad_missing = [x for x in missing if "private_adapter" not in x and x not in tied_missing]
    if bad_missing or unexpected:
        raise RuntimeError(f"unexpected load mismatch: bad_missing={bad_missing[:10]} unexpected={unexpected[:10]}")
    model.tie_weights()
    model.register_for_auto_class("AutoModelForMaskedLM")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, missing, unexpected


def set_private_enabled(model, enabled: bool):
    model.set_private_enabled(enabled)


def save_checkpoint(model, tokenizer, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst), safe_serialization=True)
    tokenizer.save_pretrained(str(dst))
    src = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_private_modeling.py')
    d = dst / src.name
    if src.exists() and not d.exists():
        shutil.copy2(str(src), str(d))


def compute_neutral_loss(model, input_ids, attention_mask, neutral_subsample: int):
    n = min(int(neutral_subsample), input_ids.shape[0])
    if n <= 0:
        return None, 0.0
    # Deterministic pair of private-off/private-on logits: eval mode removes dropout
    # so neutral loss is zero at attachment and later measures private deviation only.
    was_training = model.training
    model.eval()
    ids = input_ids[:n]
    att = attention_mask[:n]
    set_private_enabled(model, False)
    with torch.no_grad():
        slow_logits = model(input_ids=ids, attention_mask=att).logits.detach()
    set_private_enabled(model, True)
    out = model(input_ids=ids, attention_mask=att)
    log_p_private = F.log_softmax(out.logits, dim=-1)
    p_slow = F.softmax(slow_logits, dim=-1)
    kl_tok = F.kl_div(log_p_private, p_slow, reduction="none").sum(-1)
    mask_float = att.float()
    loss = (kl_tok * mask_float).sum() / max(1.0, float(mask_float.sum()))
    if was_training:
        model.train()
    return loss, float(loss.detach().cpu())


def train(args):
    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hf_cache = out / "hf_cache"
    (hf_cache / "modules").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_MODULES_CACHE"] = str(hf_cache / "modules")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    endpoint = Path(args.endpoint)
    stream = Path(args.example_jsonl)
    aux_path = Path(args.aux_pair_data_path)

    print(json.dumps({"event": "start", "trainer": "FROZEN82_PRIVATE_TAIL",
                      "device": str(device), "mode": args.mode,
                      "initial_consumed_words": args.initial_consumed_words,
                      "max_tail_charged_words": args.max_tail_charged_words,
                      "skip_rows": args.skip_rows,
                      "deterministic_neutrality": True}), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(endpoint), local_files_only=True)
    mask_id = int(tokenizer.mask_token_id)
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        raise RuntimeError("tokenizer lacks BOS/EOS fallback for auxiliary special tokens")

    aux_data, aux_summary = load_aux_pair_data(aux_path)
    pair_eids = {int(k) for k in aux_data.keys()}
    examples = load_examples_tail(stream, int(args.skip_rows), int(args.max_tail_charged_words))
    main_words_available = sum(e.words for e in examples)
    n_pair_examples = sum(1 for e in examples if e.example_id in pair_eids)
    print(json.dumps({"event": "data_loaded", "tail_examples": len(examples),
                      "main_words_available": main_words_available,
                      "pair_rows_available": n_pair_examples,
                      "aux_summary_status": aux_summary.get("status")}), flush=True)

    dataset = TailDataset(examples, tokenizer, args.seq_length, pair_eids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0,
                        pin_memory=torch.cuda.is_available())

    reset_all(args.train_rng_seed)
    model, missing, unexpected = load_frozen_private_model(endpoint, args)
    model.to(device)
    if torch.cuda.is_available():
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    private_names = {n for n, _ in model.named_parameters() if ".private_adapter." in n}
    for n, p in model.named_parameters():
        p.requires_grad_(n in private_names)

    private_up_ids = {id(p) for n, p in model.named_parameters() if ".private_adapter.up." in n}
    private_normal = [p for n, p in model.named_parameters() if n in private_names and id(p) not in private_up_ids]
    private_zero_wd = [p for n, p in model.named_parameters() if n in private_names and id(p) in private_up_ids]
    optimizer = torch.optim.AdamW([
        {"params": private_normal, "weight_decay": args.weight_decay},
        {"params": private_zero_wd, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    private_params = sum(p.numel() for n, p in model.named_parameters() if n in private_names)
    print(json.dumps({"event": "model_loaded", "endpoint": rel(endpoint),
                      "total_params": total_params, "private_params": private_params,
                      "frozen_slow_params": total_params - private_params,
                      "missing_keys_count": len(missing), "unexpected_keys_count": len(unexpected)}), flush=True)

    schedule_total = int(args.lr_total_steps) if args.lr_total_steps > 0 else max(1, len(loader))
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed)

    config = {
        "status": "FROZEN82_PRIVATE_TAIL_CONFIG",
        "endpoint": rel(endpoint),
        "example_jsonl": rel(stream),
        "aux_pair_data_path": rel(aux_path),
        "mode": args.mode,
        "initial_consumed_words": args.initial_consumed_words,
        "full_cap_words": args.full_cap_words,
        "max_tail_charged_words": args.max_tail_charged_words,
        "skip_rows": args.skip_rows,
        "tail_start_row_index_0based": args.skip_rows,
        "deterministic_neutrality_eval_mode": True,
        "trainable": "private_adapter_only",
        "slow_path": "frozen_verified_scale1p75_chck_82M",
        "schedule_total": schedule_total,
        "warmup_steps": warmup,
        "total_params": total_params,
        "private_params": private_params,
        **{k: getattr(args, k) for k in DEFAULTS.keys()},
    }
    (out / "train_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    model.train()
    cum_main = 0
    cum_aux = 0
    updates = 0
    next_ckpt_tail = args.checkpoint_words if args.checkpoint_words > 0 else None
    losses_aux: list[float] = []
    losses_neutral: list[float] = []
    stopped_before_cap = False
    log_path = out / "training_log.jsonl"

    with log_path.open("w", encoding="utf-8") as logf:
        for loader_step, batch in enumerate(loader, 1):
            if args.max_updates and updates >= args.max_updates:
                print(json.dumps({"event": "stop_after_max_updates", "updates": updates}), flush=True)
                break
            words = int(batch["words"].sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            example_ids = batch["example_id"]
            is_pair = batch["is_pair"]

            masked_inputs, labels = apply_wwm(input_ids, attention_mask, word_group, tokenizer, args.mask_prob, gen)
            n_main_targets = int((labels != -100).sum().item())

            units: list[AuxUnit] = []
            if args.mode != "neutral_only":
                pidx = is_pair.nonzero(as_tuple=False).squeeze(-1).tolist()
                if isinstance(pidx, int):
                    pidx = [pidx]
                recs = []
                valid_indices = []
                for bi in pidx:
                    k = str(int(example_ids[bi].item()))
                    if k in aux_data:
                        valid_indices.append(bi)
                        recs.append(aux_data[k])
                if valid_indices:
                    units = collect_aux_units(word_group, labels, valid_indices, recs)

            built_aux, aux_words, n_cond, n_free = build_view_batches(
                units, args.mode, args.aux_pair_shuffle_seed, loader_step,
                cls_id, sep_id, mask_id, pad_id, args.aux_max_length, device)

            tail_next = cum_main + cum_aux + words + aux_words
            total_next = int(args.initial_consumed_words) + tail_next
            if tail_next > args.max_tail_charged_words or total_next > args.full_cap_words:
                stopped_before_cap = True
                print(json.dumps({"event": "stop_before_over_cap", "loader_step": loader_step,
                                  "tail_current_charged": cum_main + cum_aux,
                                  "next_main_words": words, "next_aux_words": aux_words,
                                  "initial_consumed_words": args.initial_consumed_words,
                                  "total_next": total_next,
                                  "full_cap_words": args.full_cap_words}), flush=True)
                break

            lr = lr_at_update(updates, schedule_total, warmup, args.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            aux_loss_value = 0.0
            n_aux_targets = 0
            # Aux CE in train mode; all slow tensors are frozen, so gradients enter private only.
            model.train()
            set_private_enabled(model, True)
            if built_aux is not None:
                aux_input, aux_attention, aux_labels = built_aux
                n_aux_targets = int((aux_labels != -100).sum().item())
                if n_aux_targets > 0:
                    aux_loss_sum_value = 0.0
                    mb = max(1, int(args.aux_micro_batch_size))
                    for s in range(0, aux_input.shape[0], mb):
                        e = min(s + mb, aux_input.shape[0])
                        out_aux = model(input_ids=aux_input[s:e], attention_mask=aux_attention[s:e])
                        vocab = out_aux.logits.shape[-1]
                        aux_loss_sum = F.cross_entropy(
                            out_aux.logits.reshape(-1, vocab), aux_labels[s:e].reshape(-1),
                            ignore_index=-100, reduction="sum")
                        aux_loss_sum_value += float(aux_loss_sum.detach().cpu())
                        (args.aux_lambda * aux_loss_sum / max(1, n_aux_targets)).backward()
                        del out_aux, aux_loss_sum
                    aux_loss_value = aux_loss_sum_value / max(1, n_aux_targets)
                    losses_aux.append(aux_loss_value)
                del aux_input, aux_attention, aux_labels

            neutral_loss_value = 0.0
            if args.neutral_lambda > 0:
                neutral_loss, neutral_loss_value = compute_neutral_loss(
                    model, masked_inputs, attention_mask, args.neutral_subsample)
                if neutral_loss is not None:
                    (args.neutral_lambda * neutral_loss).backward()
                    del neutral_loss
                losses_neutral.append(neutral_loss_value)

            private_params_list = private_normal + private_zero_wd
            torch.nn.utils.clip_grad_norm_(private_params_list, 1.0)
            optimizer.step()

            updates += 1
            cum_main += words
            cum_aux += aux_words
            tail_charged = cum_main + cum_aux
            total_consumed = int(args.initial_consumed_words) + tail_charged
            rec = {
                "update": updates,
                "loader_step": loader_step,
                "source_row_start": int(batch["row_index"][0].item()),
                "source_row_end": int(batch["row_index"][-1].item()),
                "lr": lr,
                "batch_words": words,
                "aux_words": aux_words,
                "tail_main_words": cum_main,
                "tail_aux_words": cum_aux,
                "tail_charged_words": tail_charged,
                "total_consumed_words": total_consumed,
                "masked_tokens_for_aux_selection": n_main_targets,
                "aux_targets": n_aux_targets,
                "aux_units": len(units),
                "aux_conditioned_views": n_cond,
                "aux_free_views": n_free,
                "aux_loss": aux_loss_value,
                "neutral_loss": neutral_loss_value,
                "private_rms_max": max(model.private_adapter_rms()) if hasattr(model, "private_adapter_rms") else None,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            logf.write(json.dumps(rec) + "\n")
            if updates == 1 or updates % args.log_every == 0:
                print(json.dumps({"event": "train", **rec}), flush=True)

            while next_ckpt_tail is not None and tail_charged >= next_ckpt_tail and next_ckpt_tail <= args.max_tail_charged_words:
                approx_total = int(args.initial_consumed_words) + next_ckpt_tail
                name = f"chck_{approx_total // 1_000_000}M" if approx_total % 1_000_000 == 0 else f"chck_total_{approx_total}w"
                save_checkpoint(model, tokenizer, out / "hf_model" / name)
                print(json.dumps({"event": "checkpoint", "name": name,
                                  "tail_charged_words": tail_charged,
                                  "total_consumed_words": total_consumed,
                                  "update": updates}), flush=True)
                next_ckpt_tail += args.checkpoint_words

    set_private_enabled(model, True)
    save_checkpoint(model, tokenizer, out / "hf_model" / "final")
    metrics = {
        "status": "FROZEN82_PRIVATE_TAIL",
        "mode": args.mode,
        "endpoint": rel(endpoint),
        "initial_consumed_words": args.initial_consumed_words,
        "skip_rows": args.skip_rows,
        "max_tail_charged_words": args.max_tail_charged_words,
        "full_cap_words": args.full_cap_words,
        "stopped_before_cap": stopped_before_cap,
        "updates": updates,
        "schedule_total": schedule_total,
        "tail_main_word_exposure": cum_main,
        "tail_aux_word_exposure": cum_aux,
        "tail_charged_words": cum_main + cum_aux,
        "total_consumed_words": int(args.initial_consumed_words) + cum_main + cum_aux,
        "total_params": total_params,
        "private_params": private_params,
        "frozen_slow_params": total_params - private_params,
        "first_aux_loss": losses_aux[0] if losses_aux else None,
        "final_aux_loss": losses_aux[-1] if losses_aux else None,
        "mean_aux_loss": sum(losses_aux) / len(losses_aux) if losses_aux else 0.0,
        "aux_loss_batches": len(losses_aux),
        "first_neutral_loss": losses_neutral[0] if losses_neutral else None,
        "final_neutral_loss": losses_neutral[-1] if losses_neutral else None,
        "mean_neutral_loss": sum(losses_neutral) / len(losses_neutral) if losses_neutral else 0.0,
        "deterministic_neutrality_eval_mode": True,
        "trainable": "private_adapter_only",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", **metrics}), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Frozen 82M slow path + fresh private-tail trainer")
    p.add_argument("--endpoint", default=str(DEFAULT_ENDPOINT))
    p.add_argument("--example_jsonl", default=str(DEFAULT_STREAM))
    p.add_argument("--aux_pair_data_path", default=str(DEFAULT_AUX))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["aligned", "shuffled", "neutral_only"], default="aligned")
    p.add_argument("--aux_micro_batch_size", type=int, default=8)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        elif isinstance(v, int):
            p.add_argument(f"--{k}", type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(f"--{k}", type=float, default=v)
        else:
            p.add_argument(f"--{k}", default=v)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
