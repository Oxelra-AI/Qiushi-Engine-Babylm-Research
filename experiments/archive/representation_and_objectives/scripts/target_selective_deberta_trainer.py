#!/usr/bin/env python3
"""research target-selective compact-view DeBERTa trainer.

This script reuses the corrected research compact-pair data/interface but changes only
which already-corrupted target positions contribute loss.  The input text, source/rewrite
visibility, WWM corruption schedule, filler dose, source dose, tokenizer, initialization,
optimizer, and LR schedule are otherwise unchanged.

Modes
-----
full
    identical to research own_visible full-loss training.
drop_abs_content
    keep the same input corruption but set labels for intrinsically source-absent compact-side
    content tokens to -100.
drop_copied_matched
    keep the same input corruption but set labels for an exactly matched number of compact-side
    copied tokens to -100, using a deterministic global selection over the realized stream.

The target-selection screen asks whether direct prediction of the rare source-absent compact
content tokens is causally special relative to removing the same number of copied-side labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

USER_ROOT = pathlib.Path(".").resolve()
SCRIPT_DIR = USER_ROOT / "experiments/archive" / 'representation_and_objectives' / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crossview_deberta_trainer_v2 as base  # noqa: E402


MODES = ("full", "drop_abs_content", "drop_copied_matched")


def target_key(example_index: int, pos: int, mode: str, seed: int) -> str:
    return f"research|mode={mode}|seed={seed}|ex={example_index}|pos={pos}"


def stable_u64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


class TargetSelectiveDataset(base.PairAwareDataset):
    def __init__(
        self,
        *args,
        target_mode: str = "full",
        drop_copied_positions: set[tuple[int, int]] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if target_mode not in MODES:
            raise ValueError(target_mode)
        self.target_mode = target_mode
        self.drop_copied_positions = drop_copied_positions or set()

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if self.target_mode == "full":
            item["dropped_abs_content"] = 0
            item["dropped_copied"] = 0
            item["kept_target_count"] = int((item["labels"] != -100).sum().item())
            return item

        labels = item["labels"].clone()
        ts = item["target_stratum"]
        target_mask = labels != -100
        if self.target_mode == "drop_abs_content":
            dm = target_mask & (ts == base.S_RW_ABS_CONTENT)
        elif self.target_mode == "drop_copied_matched":
            if self.drop_copied_positions:
                dm = torch.zeros_like(target_mask, dtype=torch.bool)
                for pos in range(labels.shape[0]):
                    if (idx, pos) in self.drop_copied_positions:
                        dm[pos] = True
                dm &= target_mask & (ts == base.S_RW_COPIED)
            else:
                dm = torch.zeros_like(target_mask, dtype=torch.bool)
        else:
            raise ValueError(self.target_mode)
        dropped_abs = int((dm & (ts == base.S_RW_ABS_CONTENT)).sum().item())
        dropped_copied = int((dm & (ts == base.S_RW_COPIED)).sum().item())
        labels[dm] = -100
        item["labels"] = labels
        item["dropped_abs_content"] = dropped_abs
        item["dropped_copied"] = dropped_copied
        item["kept_target_count"] = int((labels != -100).sum().item())
        return item


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [x[k] for x in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals)
        else:
            out[k] = torch.tensor(vals, dtype=torch.long)
    return out


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target_mode", required=True, choices=MODES)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--init_model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_word_exposure", type=int, default=20_000_000)
    p.add_argument("--checkpoint_words", type=int, default=10_000_000)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=2529)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--mask_seed", type=int, default=430230221)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=43022)
    p.add_argument("--train_rng_seed", type=int, default=43023)
    p.add_argument("--selection_seed", type=int, default=22543023)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--precompute_only", action="store_true")
    return p.parse_args()


def count_original_targets(dataset: base.PairAwareDataset) -> dict[str, int]:
    counts = {v: 0 for v in base.STRATA.values() if v != "pad"}
    total = 0
    for i in range(len(dataset)):
        item = dataset[i]
        labels = item["labels"]
        ts = item["target_stratum"]
        m = labels != -100
        total += int(m.sum().item())
        for sid, sn in base.STRATA.items():
            if sid == base.S_PAD:
                continue
            counts[sn] += int((m & (ts == sid)).sum().item())
    counts["all"] = total
    return counts


def select_copied_positions(dataset: base.PairAwareDataset, k: int, seed: int) -> set[tuple[int, int]]:
    scored: list[tuple[int, int, int]] = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels = item["labels"]
        ts = item["target_stratum"]
        m = (labels != -100) & (ts == base.S_RW_COPIED)
        for pos in torch.nonzero(m, as_tuple=False).view(-1).tolist():
            scored.append((stable_u64(target_key(i, int(pos), "drop_copied_matched", seed)), i, int(pos)))
    if k > len(scored):
        raise ValueError(f"Need {k} copied positions but only {len(scored)} available")
    scored.sort(key=lambda x: x[0])
    return {(i, pos) for _, i, pos in scored[:k]}


def build_datasets(args):
    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    pool = base.load_pool(args.data)
    own_words = sum(base.row_words(r, "own") for r in pool)
    wrong_words = sum(base.row_words(r, "wrong") for r in pool)
    if own_words != wrong_words:
        raise ValueError(f"Own/wrong words differ: {own_words} vs {wrong_words}")
    examples, aw, ep = base.build_stream(pool, args.max_word_exposure, "own", args.seed)
    full = base.PairAwareDataset(examples, tokenizer, args.max_seq_length, "own", False, args.mask_prob, args.mask_seed)
    counts = count_original_targets(full)
    drop_positions: set[tuple[int, int]] | None = None
    if args.target_mode == "drop_copied_matched":
        drop_positions = select_copied_positions(full, counts["rw_abs_content"], args.selection_seed)
    ds = TargetSelectiveDataset(
        examples,
        tokenizer,
        args.max_seq_length,
        "own",
        False,
        args.mask_prob,
        args.mask_seed,
        target_mode=args.target_mode,
        drop_copied_positions=drop_positions,
    )
    # Count realized kept/dropped targets after selection.
    kept_counts = {v: 0 for v in base.STRATA.values() if v != "pad"}
    dropped_abs = 0
    dropped_copied = 0
    kept_all = 0
    for i in range(len(ds)):
        item = ds[i]
        labels = item["labels"]
        ts = item["target_stratum"]
        m = labels != -100
        kept_all += int(m.sum().item())
        for sid, sn in base.STRATA.items():
            if sid == base.S_PAD:
                continue
            kept_counts[sn] += int((m & (ts == sid)).sum().item())
        dropped_abs += int(item["dropped_abs_content"])
        dropped_copied += int(item["dropped_copied"])
    kept_counts["all"] = kept_all
    selection_summary = {
        "target_mode": args.target_mode,
        "original_target_counts": counts,
        "kept_target_counts": kept_counts,
        "dropped_abs_content": dropped_abs,
        "dropped_copied": dropped_copied,
        "matched_drop_count": counts["rw_abs_content"],
        "copied_position_set_size": len(drop_positions or []),
        "word_exposure_realized": aw,
        "epochs_realized": ep,
        "examples": len(examples),
    }
    return tokenizer, ds, selection_summary


def run_smoke(model, dataset, device, out, args, selection_summary):
    batch = collate([dataset[i] for i in range(min(4, len(dataset)))])
    ids = batch["input_ids"].to(device)
    mids = batch["masked_input_ids"].to(device)
    labs = batch["labels"].to(device)
    att = batch["attention_mask"].to(device)
    pm = batch["pair_mask"].to(device)
    with torch.no_grad():
        stock = model(input_ids=ids, attention_mask=att)
        full = (att.unsqueeze(2) * att.unsqueeze(1)).long()
        cl, _ = base.custom_mlm_forward(model, ids, att, full)
        full_delta = float((stock.logits - cl).abs().max().item())
        pair_logits, _ = base.custom_mlm_forward(model, ids, att, pm)
        pair_delta = float((cl - pair_logits).abs().max().item())
    model.train()
    _, loss = base.custom_mlm_forward(model, mids, att, pm, labs)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    res = {
        "status": "TARGET_SELECTIVE_SMOKE",
        "mode": args.target_mode,
        "full_vis_delta": full_delta,
        "pair_mask_delta": pair_delta,
        "loss_initial_batch4": float(loss.detach().cpu()),
        "grad_tensors": len(grads),
        "gradients_finite": all(torch.isfinite(g).all().item() for g in grads),
        "masked_tokens_batch4_after_selection": int((labs != -100).sum().item()),
        "selection_summary": selection_summary,
        "pass": full_delta < 1e-4 and all(torch.isfinite(g).all().item() for g in grads) and int((labs != -100).sum().item()) > 0,
    }
    (out / f"smoke_{args.target_mode}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
    if not res["pass"]:
        raise SystemExit(1)


def main() -> None:
    args = build_args()
    t0 = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, dataset, selection_summary = build_datasets(args)
    (out / "target_selection_summary.json").write_text(json.dumps(selection_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "selection", **selection_summary}, indent=2, ensure_ascii=False), flush=True)
    if args.precompute_only:
        return

    if args.train_rng_seed >= 0:
        base.reset_seeds(args.train_rng_seed)
    model = base.load_or_build_model(args, tokenizer)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    device = torch.device(args.device)
    model.to(device)
    pc = sum(p.numel() for p in model.parameters())
    init_sha = base.sha256_file(pathlib.Path(args.init_model_path) / "model.safetensors")
    print(f"Model: {pc:,} params on {device}; init_sha={init_sha}", flush=True)

    if args.smoke:
        run_smoke(model, dataset, device, out, args, selection_summary)
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=torch.cuda.is_available())
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    st_total = args.lr_total_steps if args.lr_total_steps > 0 else len(loader)
    warmup = max(1, int(st_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, warmup, st_total)

    log_path = out / "training_log.jsonl"
    cum = 0
    loss_first = None
    loss_last = None
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    saved = []
    stratum_acc = {v: [0.0, 0] for v in base.STRATA.values() if v != "pad"}
    stratum_last = {v: None for v in base.STRATA.values() if v != "pad"}
    dropped_abs_total = 0
    dropped_copied_total = 0
    kept_total = 0

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            dropped_abs = int(batch.pop("dropped_abs_content").sum().item())
            dropped_copied = int(batch.pop("dropped_copied").sum().item())
            kept_count = int(batch.pop("kept_target_count").sum().item())
            ids = batch["masked_input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            att = batch["attention_mask"].to(device, non_blocking=True)
            pm = batch["pair_mask"].to(device, non_blocking=True)
            ts = batch["target_stratum"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            logits, loss = base.custom_mlm_forward(model, ids, att, pm, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            cum += words
            dropped_abs_total += dropped_abs
            dropped_copied_total += dropped_copied
            kept_total += kept_count
            lf = float(loss.detach().cpu())
            if loss_first is None:
                loss_first = lf
            loss_last = lf

            srec = {}
            with torch.no_grad():
                mask = labels != -100
                ptl = F.cross_entropy(
                    logits.view(-1, model.config.vocab_size),
                    labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view(labels.shape)
                for sid, sn in base.STRATA.items():
                    if sid == base.S_PAD:
                        continue
                    sm = mask & (ts == sid)
                    cnt = int(sm.sum().item())
                    if cnt > 0:
                        sl = float(ptl[sm].mean().item())
                        srec[sn] = round(sl, 5)
                        srec[f"{sn}_n"] = cnt
                        stratum_acc[sn][0] += sl * cnt
                        stratum_acc[sn][1] += cnt
                        stratum_last[sn] = sl
            nm = int((labels != -100).sum().item())
            emr = nm / max(1, int(att.sum().item()))
            rec = {
                "step": step,
                "loss": round(lf, 5),
                "lr": float(sched.get_last_lr()[0]),
                "words": words,
                "cum": cum,
                "masked_kept": nm,
                "dropped_abs_content": dropped_abs,
                "dropped_copied": dropped_copied,
                "emr_after_selection": round(emr, 4),
                **srec,
            }
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if step == 1 or step % args.log_every == 0 or step == len(loader):
                print(json.dumps({"e": "t", "mode": args.target_mode, "s": step, "l": round(lf, 4), "c": cum, "lr": float(sched.get_last_lr()[0]), "kept": nm, "drop_abs": dropped_abs, "drop_copied": dropped_copied, **{k: v for k, v in srec.items() if not k.endswith("_n")}}, ensure_ascii=False), flush=True)
            while next_ckpt and cum >= next_ckpt and next_ckpt <= args.max_word_exposure:
                nm_ck = f"chck_{next_ckpt // 1_000_000}M" if next_ckpt >= 1_000_000 else "chck_1M"
                cp = out / "hf_model" / nm_ck
                base.save_hf_checkpoint(model, tokenizer, cp)
                saved.append({"name": nm_ck, "cum": cum, "path": str(cp)})
                print(json.dumps({"e": "ckpt", "mode": args.target_mode, "n": nm_ck, "c": cum}), flush=True)
                next_ckpt += args.checkpoint_words

    base.save_hf_checkpoint(model, tokenizer, out / "hf_model")
    sm = {sn: round(a[0] / a[1], 5) if a[1] > 0 else None for sn, a in stratum_acc.items()}
    sn_ = {f"{sn}_n": a[1] for sn, a in stratum_acc.items()}
    slast = {f"{sn}_last": (round(v, 5) if v is not None else None) for sn, v in stratum_last.items()}
    summary = {
        "status": f"TARGET_SELECTIVE_{args.target_mode.upper()}_DONE",
        "target_mode": args.target_mode,
        "data": args.data,
        "init_model_path": args.init_model_path,
        "init_model_sha256": init_sha,
        "mask_schedule": "identity_attached_deterministic_wwm_v2_same_as_step221",
        "selection_seed": args.selection_seed,
        "target_selection_summary": selection_summary,
        "word_exposure": cum,
        "steps": len(loader),
        "lr_total_steps": st_total,
        "warmup_steps": warmup,
        "loss_first": round(loss_first, 5) if loss_first is not None else None,
        "loss_last": round(loss_last, 5) if loss_last is not None else None,
        "params": pc,
        "checkpoints": saved,
        "kept_target_total_seen": kept_total,
        "dropped_abs_content_seen": dropped_abs_total,
        "dropped_copied_seen": dropped_copied_total,
        "stratum_mean_loss_after_selection": sm,
        "stratum_counts_after_selection": sn_,
        "stratum_last_loss_after_selection": slast,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"e": "done", "mode": args.target_mode, "cum": cum, "loss_last": loss_last, "kept": kept_total, "drop_abs_total": dropped_abs_total, "drop_copied_total": dropped_copied_total, "stratum": sm, "t": round(time.time() - t0, 0)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
