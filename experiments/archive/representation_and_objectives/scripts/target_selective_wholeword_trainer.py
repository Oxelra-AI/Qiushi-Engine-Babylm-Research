#!/usr/bin/env python3
"""research whole-word copied-content target-selective DeBERTa trainer.

This is the stronger comparator requested after research.  It keeps the exact research
own-visible compact-pair interface: same data stream, same source+rewrite text, same
identity-attached deterministic WWM corruption, same tokenizer, same frozen init, same
optimizer/schedule.  It changes only the loss labels by setting to -100 all positions in
whole copied-content target-word groups selected by
`build_wholeword_copied_content_selection.py`.

The intended scientific contrast is:
    drop_abs_content (research)  vs  drop_copied_content_wholeword (this script)
where both remove 5,779 whole target groups / 7,649 BPE pieces under matched support,
position, epoch, and exposure features.  A survival of the source-absent fixed-event loss
effect under this comparator is stronger evidence for a target-type-specific noncopy
compact-content learning channel.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
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

DEFAULT_DATA = "experiments/archive/representation_and_objectives/data/crossview_data_v2/crossview_consolidated_v2.jsonl"
DEFAULT_TOKENIZER = "experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
DEFAULT_INIT = "experiments/archive/representation_and_objectives/training/runs/crossview_identical_init_seed43022/hf_model_init"
DEFAULT_SELECTION = "experiments/archive/representation_and_objectives/data/wholeword_copied_content_selection/drop_position_map.json"


class WholeWordCopiedDropDataset(base.PairAwareDataset):
    def __init__(self, *args, drop_position_map: dict[int, set[int]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_position_map = drop_position_map or {}

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        labels = item["labels"].clone()
        ts = item["target_stratum"]
        target_mask = labels != -100
        dm = torch.zeros_like(target_mask, dtype=torch.bool)
        if idx in self.drop_position_map:
            for pos in self.drop_position_map[idx]:
                if 0 <= int(pos) < labels.shape[0]:
                    dm[int(pos)] = True
        # Fail safe: the selection file was built from copied_content whole-word groups;
        # at training time the base dataset only distinguishes rw_copied from rw_abs_content.
        # The intersection preserves exact target/mask identity and prevents accidental source/filler drops.
        dm &= target_mask & (ts == base.S_RW_COPIED)
        dropped_copied = int(dm.sum().item())
        labels[dm] = -100
        item["labels"] = labels
        item["dropped_abs_content"] = 0
        item["dropped_copied_content_wholeword"] = dropped_copied
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
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
    p.add_argument("--init_model_path", default=DEFAULT_INIT)
    p.add_argument("--drop_position_map", default=DEFAULT_SELECTION)
    p.add_argument("--selection_audit", default="experiments/archive/representation_and_objectives/data/wholeword_copied_content_selection/wholeword_copied_content_selection_audit.json")
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
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--precompute_only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def load_position_map(path: str) -> dict[int, set[int]]:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return {int(k): {int(x) for x in v} for k, v in raw.items()}


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


def build_dataset(args) -> tuple[Any, WholeWordCopiedDropDataset, dict[str, Any]]:
    tok = base.make_portable_tokenizer(args.tokenizer_path)
    pool = base.load_pool(args.data)
    examples, aw, ep = base.build_stream(pool, args.max_word_exposure, "own", args.seed)
    if aw != args.max_word_exposure:
        raise RuntimeError(f"stream filled {aw}, expected {args.max_word_exposure}")
    full = base.PairAwareDataset(examples, tok, args.max_seq_length, "own", False, args.mask_prob, args.mask_seed)
    original_counts = count_original_targets(full)
    pos_map = load_position_map(args.drop_position_map)
    ds = WholeWordCopiedDropDataset(examples, tok, args.max_seq_length, "own", False, args.mask_prob, args.mask_seed, drop_position_map=pos_map)

    kept_counts = {v: 0 for v in base.STRATA.values() if v != "pad"}
    dropped = 0
    kept_all = 0
    examples_with_drop = 0
    position_map_total = sum(len(v) for v in pos_map.values())
    unexpected_positions = []
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
        dc = int(item["dropped_copied_content_wholeword"])
        dropped += dc
        if dc:
            examples_with_drop += 1
        # robust audit on requested positions: they must be copied targets in the original dataset.
        if i in pos_map:
            base_item = full[i]
            base_labels = base_item["labels"]
            base_ts = base_item["target_stratum"]
            for pos in pos_map[i]:
                if pos < 0 or pos >= base_labels.shape[0] or int(base_labels[pos]) == -100 or int(base_ts[pos]) != base.S_RW_COPIED:
                    if len(unexpected_positions) < 20:
                        unexpected_positions.append({"example_index": i, "pos": int(pos), "label": int(base_labels[pos]) if 0 <= pos < base_labels.shape[0] else None, "stratum": int(base_ts[pos]) if 0 <= pos < base_ts.shape[0] else None})
    kept_counts["all"] = kept_all
    selection_audit = None
    if args.selection_audit and pathlib.Path(args.selection_audit).exists():
        selection_audit = json.loads(pathlib.Path(args.selection_audit).read_text(encoding="utf-8"))
    summary = {
        "target_mode": "drop_copied_content_wholeword",
        "original_target_counts": original_counts,
        "kept_target_counts": kept_counts,
        "dropped_abs_content": 0,
        "dropped_copied_content_wholeword": dropped,
        "position_map_total_positions": int(position_map_total),
        "position_map_examples": len(pos_map),
        "examples_with_drop": examples_with_drop,
        "unexpected_position_count_first20_listed": len(unexpected_positions),
        "unexpected_positions_first20": unexpected_positions,
        "word_exposure_realized": aw,
        "epochs_realized": ep,
        "examples": len(examples),
        "selection_audit_core": {
            "status": selection_audit.get("status") if isinstance(selection_audit, dict) else None,
            "selected_groups": selection_audit.get("selected_copied_content_wholeword_groups", {}).get("n_groups") if isinstance(selection_audit, dict) else None,
            "selected_pieces": selection_audit.get("selected_copied_content_wholeword_groups", {}).get("piece_total") if isinstance(selection_audit, dict) else None,
            "absent_groups": selection_audit.get("absent_content_groups", {}).get("n_groups") if isinstance(selection_audit, dict) else None,
            "absent_pieces": selection_audit.get("absent_content_groups", {}).get("piece_total") if isinstance(selection_audit, dict) else None,
            "fraction_exact_bpe_len": selection_audit.get("match_diagnostics", {}).get("fraction_exact_bpe_len") if isinstance(selection_audit, dict) else None,
            "feature_deltas": selection_audit.get("selected_minus_abs_feature_deltas") if isinstance(selection_audit, dict) else None,
        },
    }
    if unexpected_positions:
        raise RuntimeError(f"position map contains positions that are not realized copied targets: {unexpected_positions[:3]}")
    return tok, ds, summary


def run_smoke(model, dataset, device, out: pathlib.Path, args, selection_summary):
    bs = min(4, len(dataset))
    batch = collate([dataset[i] for i in range(bs)])
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
        "status": "WHOLEWORD_COPIED_CONTENT_SMOKE",
        "mode": "drop_copied_content_wholeword",
        "full_vis_delta": full_delta,
        "pair_mask_delta": pair_delta,
        "loss_initial_batch4": float(loss.detach().cpu()),
        "grad_tensors": len(grads),
        "gradients_finite": all(torch.isfinite(g).all().item() for g in grads),
        "masked_tokens_batch4_after_selection": int((labs != -100).sum().item()),
        "selection_summary": selection_summary,
        "pass": full_delta < 1e-4 and all(torch.isfinite(g).all().item() for g in grads) and int((labs != -100).sum().item()) > 0,
    }
    (out / "smoke_drop_copied_content_wholeword.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
    if not res["pass"]:
        raise SystemExit(1)


def main() -> None:
    args = build_args()
    t0 = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok, dataset, selection_summary = build_dataset(args)
    (out / "target_selection_summary.json").write_text(json.dumps(selection_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "selection", **selection_summary}, indent=2, ensure_ascii=False), flush=True)
    if args.precompute_only:
        return

    if args.train_rng_seed >= 0:
        base.reset_seeds(args.train_rng_seed)
    model = base.load_or_build_model(args, tok)
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
    dropped_total = 0
    kept_total = 0

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            dropped_abs = int(batch.pop("dropped_abs_content").sum().item())
            dropped_copied = int(batch.pop("dropped_copied_content_wholeword").sum().item())
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
            dropped_total += dropped_copied
            kept_total += kept_count
            lf = float(loss.detach().cpu())
            if loss_first is None:
                loss_first = lf
            loss_last = lf

            srec = {}
            with torch.no_grad():
                mask = labels != -100
                ptl = F.cross_entropy(logits.view(-1, model.config.vocab_size), labels.view(-1), reduction="none", ignore_index=-100).view(labels.shape)
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
                "dropped_copied_content_wholeword": dropped_copied,
                "emr_after_selection": round(emr, 4),
                **srec,
            }
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if step == 1 or step % args.log_every == 0 or step == len(loader):
                print(json.dumps({"e": "t", "mode": "drop_copied_content_wholeword", "s": step, "l": round(lf, 4), "c": cum, "lr": float(sched.get_last_lr()[0]), "kept": nm, "drop_abs": dropped_abs, "drop_copied_whole": dropped_copied, **{k: v for k, v in srec.items() if not k.endswith("_n")}}, ensure_ascii=False), flush=True)
            while next_ckpt and cum >= next_ckpt and next_ckpt <= args.max_word_exposure:
                nm_ck = f"chck_{next_ckpt // 1_000_000}M" if next_ckpt >= 1_000_000 else "chck_1M"
                cp = out / "hf_model" / nm_ck
                base.save_hf_checkpoint(model, tok, cp)
                saved.append({"name": nm_ck, "cum": cum, "path": str(cp)})
                print(json.dumps({"e": "ckpt", "mode": "drop_copied_content_wholeword", "n": nm_ck, "c": cum}), flush=True)
                next_ckpt += args.checkpoint_words

    base.save_hf_checkpoint(model, tok, out / "hf_model")
    sm = {sn: round(a[0] / a[1], 5) if a[1] > 0 else None for sn, a in stratum_acc.items()}
    sn_ = {f"{sn}_n": a[1] for sn, a in stratum_acc.items()}
    slast = {f"{sn}_last": (round(v, 5) if v is not None else None) for sn, v in stratum_last.items()}
    summary = {
        "status": "TARGET_SELECTIVE_DROP_COPIED_CONTENT_WHOLEWORD_DONE",
        "target_mode": "drop_copied_content_wholeword",
        "data": args.data,
        "init_model_path": args.init_model_path,
        "init_model_sha256": init_sha,
        "mask_schedule": "identity_attached_deterministic_wwm_v2_same_as_step221_step225",
        "drop_position_map": args.drop_position_map,
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
        "dropped_copied_content_wholeword_seen": dropped_total,
        "stratum_mean_loss_after_selection": sm,
        "stratum_counts_after_selection": sn_,
        "stratum_last_loss_after_selection": slast,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"e": "done", "mode": "drop_copied_content_wholeword", "cum": cum, "loss_last": loss_last, "kept": kept_total, "drop_copied_whole_total": dropped_total, "stratum": sm, "t": round(time.time() - t0, 0)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
