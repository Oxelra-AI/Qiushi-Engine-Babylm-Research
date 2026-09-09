#!/usr/bin/env python3
"""research pathway-separated detached-private trainer.

ordinary MLM updates an adapter-off stock
path; the private residual trains from detached source-conditioned/source-free
states; detached stock logits on already-charged main batches keep the residual
neutral away from the selected edit signal.

Per batch:
  Phase A: Main MLM (adapter disabled)
    - Standard WWM on full legal row → main loss
    - Save detached stock logits for neutrality subsample (fp16)
    - Backward: stock parameters get gradients; adapter parameters do not
    - Clip stock grads, step stock optimizer

  Phase B: Auxiliary (adapter enabled, stock frozen) — pair rows only
    - For pair rows with masked rewrite word groups, build conditioned/free views
    - Forward in adapter-only microbatches → aux CE loss
    - Adapter parameters accumulate gradients

  Phase C: Neutrality (adapter enabled, stock frozen) — every batch
    - Forward neutrality subsample with adapter enabled, stock frozen
    - KL(P_stock || P_adapter) on non-padding positions
    - Adapter parameters accumulate gradients
    - Clip adapter grads, step adapter optimizer

  Unfreeze stock, proceed to next batch.

Charged-word accounting: charged = main row words + auxiliary input words.
Neutrality does NOT charge additional words — it uses already-charged main tokens.
"""
from __future__ import annotations

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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DebertaV2Config

STUDY = Path("experiments/archive/frontier_consolidation")
WORKSPACE = STUDY
SCRIPTS = WORKSPACE / "scripts"
sys.path.insert(0, str(SCRIPTS))
from adapter_modeling import AdapterDebertaV2ForMaskedLM  # noqa: E402

DEFAULTS = dict(
    hidden_size=480,
    n_layer=8,
    n_head=8,
    ffn_mult=4,
    seed=43,
    extra_init_seed=43022,
    train_rng_seed=43023,
    batch_size=256,
    seq_length=256,
    aux_max_length=464,
    learning_rate=0.001,
    warmup_fraction=0.06,
    weight_decay=0.01,
    mask_prob=0.15,
    max_word_exposure=20_000_000,
    lr_total_steps=2529,
    checkpoint_words=1_000_000,
    log_every=50,
    adapter_bottleneck=128,
    adapter_scale=1.0,
    aux_lambda=1.0,
    neutral_lambda=1.0,
    neutral_subsample=32,
    aux_pair_shuffle_seed=43022,
)


# ── Data loading (identical to research) ──────────────────────────────────────

@dataclass
class Ex:
    text: str
    words: int
    source: str
    example_id: int


def load_examples(path: str, max_main_words: int) -> list[Ex]:
    examples: list[Ex] = []
    selected = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch example_id={obj.get('example_id')}")
            if selected + words > max_main_words:
                break
            examples.append(Ex(text=text, words=words, source=str(obj.get("source", "")),
                               example_id=int(obj.get("example_id", -1))))
            selected += words
    return examples


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


class DualViewDataset(Dataset):
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
            "input_ids": input_ids, "attention_mask": attention_mask,
            "word_group": word_group, "words": ex.words,
            "example_id": ex.example_id, "is_pair": ex.example_id in self.pair_eids,
        }


def collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
        "example_id": torch.tensor([x["example_id"] for x in batch], dtype=torch.long),
        "is_pair": torch.tensor([x["is_pair"] for x in batch], dtype=torch.bool),
    }


def apply_wwm(input_ids, attention_mask, word_group, tokenizer, mask_prob, gen):
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


def load_aux_pair_data(path: str):
    raw = json.load(open(path, encoding="utf-8"))
    return raw["pair_data"], raw["summary"]


@dataclass
class AuxUnit:
    source_ids: list[int]
    source_words: int
    rewrite_ids: list[int]
    rewrite_word_group: list[int]
    rewrite_word_groups_to_mask: set[int]
    rewrite_words: int


def collect_aux_units(word_group, labels, pair_batch_indices, pair_records):
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


def source_assignment(units, mode, seed, step):
    if mode == "aligned":
        return [(u.source_ids, u.source_words) for u in units]
    if mode == "mlm_only":
        return []
    if len(units) < 2:
        return []
    rng = random.Random(seed + 1000003 * step)
    idxs = list(range(len(units)))
    rng.shuffle(idxs)
    src_from = idxs[1:] + idxs[:1]
    assign = [([], 0)] * len(units)
    for target_i, src_i in zip(idxs, src_from):
        assign[target_i] = (units[src_i].source_ids, units[src_i].source_words)
    return assign


def build_view_batches(units, mode, seed, step, cls_id, sep_id, mask_id, pad_id, max_len, device):
    if mode == "mlm_only" or not units:
        return None, 0, 0, 0
    assigned = source_assignment(units, mode, seed, step)
    if not assigned:
        return None, 0, 0, 0
    seqs, labs = [], []
    charged_words = 0
    n_cond = n_free = 0
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


# ── Model and training infrastructure ────────────────────────────────────────

def set_adapters_enabled(model, enabled: bool):
    """Toggle adapter contribution in all layers."""
    for layer in model.deberta.encoder.layer:
        layer.adapter.enabled = enabled


def build_model(args, tokenizer):
    max_pos = max(512, args.seq_length + 8, args.aux_max_length + 8)
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=args.hidden_size * args.ffn_mult,
        max_position_embeddings=max_pos,
        max_relative_positions=256,
        position_buckets=256,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_activation="gelu",
        adapter_enabled=True,
        adapter_scale=args.adapter_scale,
    )
    return AdapterDebertaV2ForMaskedLM(cfg)


def save_checkpoint(model, tokenizer, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst), safe_serialization=True)
    tokenizer.save_pretrained(str(dst))
    for src_name in ["adapter_modeling.py"]:
        src = SCRIPTS / src_name
        d = dst / src_name
        if src.exists() and not d.exists():
            shutil.copy2(str(src), str(d))


def lr_at_update(update_index0, schedule_total, warmup, peak_lr):
    if update_index0 < warmup:
        return peak_lr * update_index0 / max(1, warmup)
    p = (update_index0 - warmup) / max(1, schedule_total - warmup)
    p = min(max(p, 0.0), 1.0)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * p))


def reset_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Training loop ────────────────────────────────────────────────────────────

def train(args):
    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hf_cache = str(out / "hf_cache")
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_cache
    os.environ["HF_MODULES_CACHE"] = str(Path(hf_cache) / "modules")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "start", "device": str(device), "mode": args.mode,
                      "trainer": "DETACHED_PRIVATE",
                      "max_word_exposure_is_charged": True,
                      "neutral_lambda": args.neutral_lambda,
                      "neutral_subsample": args.neutral_subsample,
                      "lr_total_steps": args.lr_total_steps}), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    mask_id = int(tokenizer.mask_token_id)
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if cls_id is None or sep_id is None:
        raise RuntimeError("tokenizer lacks BOS/EOS fallback for auxiliary special tokens")

    aux_data, aux_summary = load_aux_pair_data(args.aux_pair_data_path)
    pair_eids = {int(k) for k in aux_data.keys()}
    print(json.dumps({"event": "aux_loaded", "pair_rows": len(pair_eids),
                      "summary_status": aux_summary.get("status")}), flush=True)

    examples = load_examples(args.example_jsonl, args.max_word_exposure)
    main_words_available = sum(e.words for e in examples)
    n_pair_examples = sum(1 for e in examples if e.example_id in pair_eids)
    print(json.dumps({"event": "data_loaded", "examples": len(examples),
                      "main_words_available": main_words_available,
                      "pair_examples_available": n_pair_examples}), flush=True)

    dataset = DualViewDataset(examples, tokenizer, args.seq_length, pair_eids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0,
                        pin_memory=torch.cuda.is_available())

    reset_all(args.seed)
    if args.extra_init_seed >= 0:
        reset_all(args.extra_init_seed)
    model = build_model(args, tokenizer)
    model.register_for_auto_class("AutoModelForMaskedLM")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    if args.train_rng_seed >= 0:
        reset_all(args.train_rng_seed)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    adapter_param_count = sum(p.numel() for n, p in model.named_parameters() if ".adapter." in n)
    adapter_names = {n for n, _ in model.named_parameters() if ".adapter." in n}

    # ── Separate optimizers: stock and adapter ──
    stock_params_list = [p for n, p in model.named_parameters() if ".adapter." not in n]
    adapter_up_ids = {id(p) for n, p in model.named_parameters() if ".adapter.up." in n}
    adapter_normal = [p for n, p in model.named_parameters()
                      if ".adapter." in n and id(p) not in adapter_up_ids]
    adapter_zero_wd = [p for n, p in model.named_parameters()
                       if ".adapter." in n and id(p) in adapter_up_ids]

    stock_optimizer = torch.optim.AdamW(
        [{"params": stock_params_list, "weight_decay": args.weight_decay}],
        lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6)
    adapter_optimizer = torch.optim.AdamW([
        {"params": adapter_normal, "weight_decay": args.weight_decay},
        {"params": adapter_zero_wd, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6)

    print(json.dumps({"event": "model_built", "total_params": total_params,
                      "adapter_params": adapter_param_count,
                      "stock_params": total_params - adapter_param_count,
                      "gradient_checkpointing": True,
                      "aux_micro_batch_size": args.aux_micro_batch_size,
                      "separated_optimizers": True}), flush=True)

    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else len(loader)
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    (out / "train_config.json").write_text(json.dumps({
        "status": "DETACHED_PRIVATE_CONFIG",
        "mode": args.mode,
        "aux_lambda": args.aux_lambda,
        "neutral_lambda": args.neutral_lambda,
        "neutral_subsample": args.neutral_subsample,
        "aux_gradient": "adapter_only",
        "neutral_gradient": "adapter_only",
        "main_gradient": "stock_only",
        "aux_views": "conditioned_plus_source_free",
        "pathway": "separated",
        "max_word_exposure_is_charged": True,
        "schedule_total": schedule_total,
        "warmup_steps": warmup,
        "total_params": total_params,
        "adapter_params": adapter_param_count,
        "stock_params": total_params - adapter_param_count,
        "gradient_checkpointing": True,
        "aux_micro_batch_size": args.aux_micro_batch_size,
        "main_words_available": main_words_available,
        "n_pair_examples_available": n_pair_examples,
        **{k: getattr(args, k) for k in DEFAULTS.keys()},
    }, indent=2), encoding="utf-8")

    model.train()
    cum_main = cum_aux = updates = 0
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    losses, aux_losses, neutral_losses = [], [], []
    stopped_before_cap = False
    log_path = out / "training_log.jsonl"

    with log_path.open("w", encoding="utf-8") as logf:
        for loader_step, batch in enumerate(loader, 1):
            words = int(batch["words"].sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            example_ids = batch["example_id"]
            is_pair = batch["is_pair"]
            B = input_ids.shape[0]

            masked_inputs, labels = apply_wwm(input_ids, attention_mask, word_group,
                                              tokenizer, args.mask_prob, gen)
            n_main_targets = int((labels != -100).sum().item())

            # Collect auxiliary units
            units: list[AuxUnit] = []
            if args.mode != "mlm_only":
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

            next_charged = cum_main + cum_aux + words + aux_words
            if next_charged > args.max_word_exposure:
                stopped_before_cap = True
                print(json.dumps({"event": "stop_before_over_cap", "loader_step": loader_step,
                                  "current_charged_words": cum_main + cum_aux,
                                  "next_main_words": words, "next_aux_words": aux_words,
                                  "max_word_exposure": args.max_word_exposure}), flush=True)
                break

            lr = lr_at_update(updates, schedule_total, warmup, args.learning_rate)
            for pg in stock_optimizer.param_groups:
                pg["lr"] = lr
            for pg in adapter_optimizer.param_groups:
                pg["lr"] = lr

            # ═══════════════════════════════════════════════════════════
            # Phase A: Main MLM — adapter OFF, stock gradients only
            # ═══════════════════════════════════════════════════════════
            set_adapters_enabled(model, False)
            stock_optimizer.zero_grad(set_to_none=True)
            adapter_optimizer.zero_grad(set_to_none=True)

            out_main = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            main_loss = out_main.loss
            loss_value = float(main_loss.detach().cpu())

            # Save detached stock logits for neutrality subsample (fp16 to save memory)
            n_neutral = min(args.neutral_subsample, B)
            neutral_stock_logits = out_main.logits[:n_neutral].detach().half()
            neutral_input_ids = masked_inputs[:n_neutral].clone()
            neutral_attention = attention_mask[:n_neutral].clone()

            main_loss.backward()
            del out_main, main_loss

            torch.nn.utils.clip_grad_norm_(stock_params_list, 1.0)
            stock_optimizer.step()
            stock_optimizer.zero_grad(set_to_none=True)
            # Ensure adapter grads from Phase A are zero (should be by construction)
            adapter_optimizer.zero_grad(set_to_none=True)

            # ═══════════════════════════════════════════════════════════
            # Phase B + C: Auxiliary + Neutrality — adapter ON, stock frozen
            # ═══════════════════════════════════════════════════════════
            set_adapters_enabled(model, True)

            # Freeze stock parameters
            stock_flags = []
            for name, param in model.named_parameters():
                stock_flags.append((param, param.requires_grad))
                if name not in adapter_names:
                    param.requires_grad_(False)

            # Disable gradient checkpointing for small adapter-only passes
            was_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
            if was_checkpointing:
                model.gradient_checkpointing_disable()

            aux_loss_value = 0.0
            n_aux_targets = 0
            neutral_loss_value = 0.0

            try:
                # ── Phase B: Auxiliary (pair rows only) ──
                if built_aux is not None:
                    aux_input, aux_attention, aux_labels = built_aux
                    n_aux_targets = int((aux_labels != -100).sum().item())
                    if n_aux_targets > 0:
                        aux_loss_sum_value = 0.0
                        mb = max(1, int(args.aux_micro_batch_size))
                        for s in range(0, aux_input.shape[0], mb):
                            e = min(s + mb, aux_input.shape[0])
                            out_aux = model(input_ids=aux_input[s:e],
                                            attention_mask=aux_attention[s:e])
                            vocab = out_aux.logits.shape[-1]
                            aux_loss_sum = F.cross_entropy(
                                out_aux.logits.reshape(-1, vocab),
                                aux_labels[s:e].reshape(-1),
                                ignore_index=-100, reduction="sum",
                            )
                            aux_loss_sum_value += float(aux_loss_sum.detach().cpu())
                            (args.aux_lambda * aux_loss_sum / max(1, n_aux_targets)).backward()
                            del out_aux, aux_loss_sum
                        aux_loss_value = aux_loss_sum_value / max(1, n_aux_targets)
                    del aux_input, aux_attention, aux_labels

                # ── Phase C: Neutrality (every batch) ──
                if args.neutral_lambda > 0 and n_neutral > 0:
                    neutral_out = model(input_ids=neutral_input_ids,
                                        attention_mask=neutral_attention)
                    adapter_logits = neutral_out.logits  # [N, S, V]

                    # KL(P_stock || P_adapter) on non-padding positions
                    log_p_adapter = F.log_softmax(adapter_logits, dim=-1)
                    p_stock = F.softmax(neutral_stock_logits.float(), dim=-1)

                    kl_per_token = F.kl_div(log_p_adapter, p_stock,
                                            reduction='none').sum(-1)  # [N, S]
                    mask_float = neutral_attention.float()
                    n_tokens = max(1.0, float(mask_float.sum()))
                    neutral_loss = (kl_per_token * mask_float).sum() / n_tokens

                    neutral_loss_value = float(neutral_loss.detach().cpu())
                    (args.neutral_lambda * neutral_loss).backward()
                    del neutral_out, adapter_logits, log_p_adapter, p_stock
                    del kl_per_token, neutral_loss

            finally:
                # Restore gradient checkpointing and stock requires_grad
                if was_checkpointing:
                    try:
                        model.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False})
                    except TypeError:
                        model.gradient_checkpointing_enable()
                for param, flag in stock_flags:
                    param.requires_grad_(flag)

            del neutral_stock_logits, neutral_input_ids, neutral_attention

            # Step adapter optimizer (accumulated aux + neutral grads)
            all_adapter_params = adapter_normal + adapter_zero_wd
            torch.nn.utils.clip_grad_norm_(all_adapter_params, 1.0)
            adapter_optimizer.step()
            adapter_optimizer.zero_grad(set_to_none=True)

            updates += 1
            cum_main += words
            cum_aux += aux_words
            charged = cum_main + cum_aux
            losses.append(loss_value)
            if aux_loss_value > 0:
                aux_losses.append(aux_loss_value)
            neutral_losses.append(neutral_loss_value)

            rec = {
                "update": updates,
                "loader_step": loader_step,
                "loss": loss_value,
                "aux_loss": aux_loss_value,
                "neutral_loss": neutral_loss_value,
                "lr": lr,
                "batch_words": words,
                "aux_words": aux_words,
                "cumulative_main_words": cum_main,
                "cumulative_aux_words": cum_aux,
                "cumulative_charged_words": charged,
                "masked_tokens": n_main_targets,
                "aux_targets": n_aux_targets,
                "aux_units": len(units),
                "aux_conditioned_views": n_cond,
                "aux_free_views": n_free,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            logf.write(json.dumps(rec) + "\n")
            if updates == 1 or updates % args.log_every == 0:
                print(json.dumps({"event": "train", **rec}), flush=True)

            while next_ckpt is not None and charged >= next_ckpt and next_ckpt <= args.max_word_exposure:
                name = "chck_1M" if next_ckpt < 1_000_000 else (
                    f"chck_{next_ckpt // 1_000_000}M" if next_ckpt % 1_000_000 == 0
                    else f"chck_{next_ckpt}w")
                save_checkpoint(model, tokenizer, out / "hf_model" / name)
                print(json.dumps({"event": "checkpoint", "name": name,
                                  "charged_words": charged, "main_words": cum_main,
                                  "update": updates}), flush=True)
                next_ckpt += args.checkpoint_words

    # Ensure adapters are enabled for final checkpoint (inference mode)
    set_adapters_enabled(model, True)
    save_checkpoint(model, tokenizer, out / "hf_model" / "final")

    metrics = {
        "status": "DETACHED_PRIVATE",
        "mode": args.mode,
        "aux_lambda": args.aux_lambda,
        "neutral_lambda": args.neutral_lambda,
        "neutral_subsample": args.neutral_subsample,
        "aux_gradient": "adapter_only",
        "neutral_gradient": "adapter_only",
        "main_gradient": "stock_only",
        "pathway": "separated",
        "aux_views": "conditioned_plus_source_free",
        "max_word_exposure_is_charged": True,
        "stopped_before_cap": stopped_before_cap,
        "updates": updates,
        "schedule_total": schedule_total,
        "total_main_word_exposure": cum_main,
        "total_aux_word_exposure": cum_aux,
        "total_charged_words": cum_main + cum_aux,
        "total_params": total_params,
        "adapter_params": adapter_param_count,
        "gradient_checkpointing": True,
        "aux_micro_batch_size": args.aux_micro_batch_size,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "mean_aux_loss": sum(aux_losses) / len(aux_losses) if aux_losses else 0.0,
        "aux_loss_batches": len(aux_losses),
        "mean_neutral_loss": sum(neutral_losses) / len(neutral_losses) if neutral_losses else 0.0,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **metrics}), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="research pathway-separated detached-private trainer")
    parser.add_argument("--example_jsonl", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--aux_pair_data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["aligned", "shuffled", "mlm_only"], default="aligned")
    parser.add_argument("--aux_micro_batch_size", type=int, default=8)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", action="store_true", default=v)
        elif isinstance(v, int):
            parser.add_argument(f"--{k}", type=int, default=v)
        elif isinstance(v, float):
            parser.add_argument(f"--{k}", type=float, default=v)
        else:
            parser.add_argument(f"--{k}", default=v)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
