#!/usr/bin/env python3
"""research stagewise Phase-2 trainer: DeBERTa-v2 12x384, LAMB, 40k shared tok.

Faithful sequence curriculum: three stages with dynamic batch sizes so padded
positions/update stay ~constant (512x64, 256x128, 128x256 = 32768), and counted
words are visible (low-truncation v2 chunks). LR is one cosine over total stage
updates; masking switches WWM->token at cumulative word exposure >= switch_words.
Model initialized once with --extra_init_seed; training RNG with --train_rng_seed.

Reuses tokenizer/model/masking/LAMB from phase2_sota_trainer.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(_public_path('experiments/archive/compact_experience/scripts')))
import phase2_sota_trainer as tr  # type: ignore


class MArgs:
    """Minimal args object for tr.build_model."""
    def __init__(self, a):
        self.model_type = a.model_type
        self.n_layer = a.n_layer; self.hidden_size = a.hidden_size; self.n_head = a.n_head
        self.ffn_mult = a.ffn_mult; self.intermediate_size = a.intermediate_size
        self.max_seq_length = a.max_seq_length
        self.max_position_embeddings = a.max_position_embeddings
        self.max_relative_positions = a.max_relative_positions
        self.position_buckets = a.position_buckets
        self.deberta_relative_attention = a.deberta_relative_attention
        self.deberta_pos_att_type = a.deberta_pos_att_type
        self.share_att_key = a.share_att_key
        self.position_biased_input = a.position_biased_input


def count_word_groups(word_group: torch.Tensor, attention_mask: torch.Tensor) -> int:
    # number of distinct word-start groups actually visible in the batch
    total = 0
    wg = word_group
    am = attention_mask.bool()
    for b in range(wg.shape[0]):
        row = wg[b][am[b]]
        row = row[row >= 0]
        if row.numel():
            total += int(row.max().item()) + 1
    return total


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_jsonl", required=True)
    p.add_argument("--stage2_jsonl", required=True)
    p.add_argument("--stage3_jsonl", required=True)
    p.add_argument("--stage1_seq", type=int, default=64)
    p.add_argument("--stage2_seq", type=int, default=128)
    p.add_argument("--stage3_seq", type=int, default=256)
    p.add_argument("--stage1_batch", type=int, default=512)
    p.add_argument("--stage2_batch", type=int, default=256)
    p.add_argument("--stage3_batch", type=int, default=128)
    p.add_argument("--stage1_words", type=int, default=30_000_000)
    p.add_argument("--stage2_words", type=int, default=30_000_000)
    p.add_argument("--stage3_words", type=int, default=40_000_000)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--tokenizer_label", default="sp40k_shared")
    # Architecture
    p.add_argument("--model_type", default="deberta_v2")
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--hidden_size", type=int, default=384)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=1280)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--max_relative_positions", type=int, default=-1)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--deberta_relative_attention", default="true")
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    p.add_argument("--share_att_key", default="true")
    p.add_argument("--position_biased_input", default="false")
    # Optim / masking
    p.add_argument("--optimizer", choices=["adam", "lamb"], default="lamb")
    p.add_argument("--learning_rate", type=float, default=0.007)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.06)
    p.add_argument("--mask_mode", default="wwm", choices=["wwm", "token"])
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--mask_switch_mode", default="token", choices=["wwm", "token", "none"])
    p.add_argument("--mask_switch_words", type=int, default=70_000_000)
    # RNG / logging
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=-1)
    p.add_argument("--train_rng_seed", type=int, default=-1)
    p.add_argument("--checkpoint_words", type=int, default=10_000_000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_words_override", type=int, default=0,
                   help="If >0, cap stage words for a smoke run (applied proportionally per stage).")
    return p.parse_args()


def main():
    args = build_args()
    t0 = time.time()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    tokenizer = tr.make_portable_tokenizer(args.tokenizer_path)
    print(json.dumps({"event": "tokenizer_loaded", "vocab_size": len(tokenizer),
                      "mask_token_id": tokenizer.mask_token_id, "pad_token_id": tokenizer.pad_token_id}), flush=True)

    stage_cfg = [
        {"stage": 1, "jsonl": args.stage1_jsonl, "seq": args.stage1_seq, "batch": args.stage1_batch, "words": args.stage1_words},
        {"stage": 2, "jsonl": args.stage2_jsonl, "seq": args.stage2_seq, "batch": args.stage2_batch, "words": args.stage2_words},
        {"stage": 3, "jsonl": args.stage3_jsonl, "seq": args.stage3_seq, "batch": args.stage3_batch, "words": args.stage3_words},
    ]
    if args.max_words_override > 0:
        for s in stage_cfg:
            s["words"] = min(s["words"], args.max_words_override)

    # Load stage datasets and precompute total updates for the LR schedule.
    datasets = []
    total_steps = 0
    for s in stage_cfg:
        exs = tr.load_examples_jsonl(Path(s["jsonl"]), s["words"])
        ds = tr.MaskedChunkDataset(exs, tokenizer, s["seq"])
        n_steps = (len(exs) + s["batch"] - 1) // s["batch"]
        datasets.append((s, ds, n_steps))
        total_steps += n_steps
        print(json.dumps({"event": "stage_loaded", "stage": s["stage"], "examples": len(exs),
                          "words": sum(e.words for e in exs), "seq": s["seq"], "batch": s["batch"],
                          "n_steps": n_steps}), flush=True)

    # Build model once with fixed initialization.
    margs = MArgs(args)
    if args.extra_init_seed >= 0:
        tr.reset_all_rng(args.extra_init_seed)
    model = tr.build_model(margs, tokenizer)
    param_count = sum(p.numel() for p in model.parameters())
    emb_params = model.get_input_embeddings().weight.numel()
    if args.train_rng_seed >= 0:
        tr.reset_all_rng(args.train_rng_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.optimizer == "lamb":
        optim = tr.make_lamb_optimizer(model, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    warmup = max(1, int(total_steps * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=total_steps)
    gen = torch.Generator(device=device); gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    print(json.dumps({"event": "training_start", "param_count": param_count,
                      "embedding_params": emb_params, "non_embedding_params": param_count - emb_params,
                      "vocab_size": len(tokenizer), "total_steps": total_steps, "warmup": warmup,
                      "optimizer": args.optimizer, "lr": args.learning_rate,
                      "mask_switch_words": args.mask_switch_words,
                      "extra_init_seed": args.extra_init_seed, "train_rng_seed": args.train_rng_seed}), flush=True)

    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    loss_values = []
    saved_checkpoints = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    stage_metrics = []
    global_step = 0
    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for s, ds, n_steps in datasets:
            loader = DataLoader(ds, batch_size=s["batch"], shuffle=False, collate_fn=tr.collate,
                                num_workers=0, pin_memory=torch.cuda.is_available())
            st_vis_tokens = 0; st_vis_groups = 0; st_masked = 0; st_words = 0; st_losses = []
            for batch in loader:
                global_step += 1
                words = int(batch["words"].sum().item())
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                word_group = batch["word_group"].to(device, non_blocking=True)

                mode = args.mask_mode
                if args.mask_switch_mode != "none" and cumulative_words >= args.mask_switch_words:
                    mode = args.mask_switch_mode

                masked_inputs, labels = tr.apply_masking(input_ids, attention_mask, word_group, tokenizer, mode, args.mask_prob, gen)
                optim.zero_grad(set_to_none=True)
                out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
                loss = out_model.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step()

                cumulative_words += words
                lf = float(loss.detach().cpu()); loss_values.append(lf); st_losses.append(lf)
                st_words += words
                st_vis_tokens += int(attention_mask.sum().item())
                st_vis_groups += count_word_groups(word_group, attention_mask)
                st_masked += int((labels != -100).sum().item())

                rec = {"step": global_step, "stage": s["stage"], "loss": lf,
                       "lr": float(sched.get_last_lr()[0]), "batch_words": words,
                       "cumulative_word_exposure": cumulative_words, "seq_len": s["seq"],
                       "mask_mode": mode, "elapsed_sec": round(time.time() - t0, 1)}
                logf.write(json.dumps(rec) + "\n")
                if global_step == 1 or global_step % args.log_every == 0:
                    print(json.dumps({"event": "train", **rec}), flush=True)

                while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= (args.stage1_words + args.stage2_words + args.stage3_words):
                    name = f"chck_{next_ckpt // 1_000_000}M"
                    cp = out / "hf_model" / name
                    tr.save_hf_checkpoint(model, tokenizer, cp)
                    saved_checkpoints.append({"name": name, "target_word_exposure": next_ckpt,
                                              "actual_cumulative_word_exposure": cumulative_words, "path": str(cp)})
                    print(json.dumps({"event": "checkpoint", "name": name, "words": cumulative_words}), flush=True)
                    next_ckpt += args.checkpoint_words
            stage_metrics.append({
                "stage": s["stage"], "seq": s["seq"], "batch": s["batch"],
                "words": st_words, "visible_tokens": st_vis_tokens, "visible_word_groups": st_vis_groups,
                "masked_tokens": st_masked, "loss_first": st_losses[0] if st_losses else None,
                "loss_last": st_losses[-1] if st_losses else None,
                "visible_tokens_per_word": round(st_vis_tokens / st_words, 4) if st_words else None,
                "visible_word_groups_per_word": round(st_vis_groups / st_words, 4) if st_words else None,
            })
            print(json.dumps({"event": "stage_done", **stage_metrics[-1]}), flush=True)

    tr.save_hf_checkpoint(model, tokenizer, out / "hf_model")
    metrics = {
        "variant": f"phase2_stagewise_{args.mask_mode}_to_{args.mask_switch_mode}",
        "backend": "mlm", "model_family": "DebertaV2ForMaskedLM", "model_type": args.model_type,
        "parameter_count": param_count, "embedding_parameter_count": emb_params,
        "non_embedding_parameter_count": param_count - emb_params, "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label, "tokenizer_path": args.tokenizer_path,
        "word_exposure": cumulative_words, "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None, "actual_training_steps": global_step,
        "total_scheduled_steps": total_steps, "mask_mode": args.mask_mode,
        "mask_switch_mode": args.mask_switch_mode, "mask_switch_words": args.mask_switch_words,
        "mask_prob": args.mask_prob, "optimizer": args.optimizer, "learning_rate": args.learning_rate,
        "warmup_fraction": args.warmup_fraction, "weight_decay": args.weight_decay,
        "n_layer": args.n_layer, "hidden_size": args.hidden_size, "n_head": args.n_head,
        "intermediate_size": args.intermediate_size, "share_att_key": args.share_att_key,
        "position_biased_input": args.position_biased_input, "seed": args.seed,
        "extra_init_seed": args.extra_init_seed, "train_rng_seed": args.train_rng_seed,
        "stage_config": stage_cfg, "stage_metrics": stage_metrics,
        "saved_checkpoints": saved_checkpoints, "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "param_count": param_count, "loss_first": metrics["loss_first"],
                      "loss_last": metrics["loss_last"], "word_exposure": cumulative_words,
                      "steps": global_step, "elapsed_sec": metrics["elapsed_sec"],
                      "checkpoints": [c["name"] for c in saved_checkpoints]}), flush=True)


if __name__ == "__main__":
    main()
