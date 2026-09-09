#!/usr/bin/env python3
"""research exact-swap conditional-innovation WWM trainer.

This is a main-workspace port of the exact-swap mechanism.  It is not
launched in research.  Its purpose is to make the lower-perturbation fallback route
inspectable and mechanically testable while the research probability-reallocation
80M screen is still running.

All model/training ingredients match the research legal16k compact-view reinvest
recipe unless explicitly stated.  The only treatment variable is the WWM selected
group set: for each batch, baseline 15% WWM is reproduced first, then at most one
source-absent rewrite innovation group per changed row is swapped in for an
already-selected same-token-length ordinary non-pair donor group.  Selected group
and token mass are exactly preserved per batch; source and copyable-pair masks are
not touched.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import importlib.util
import json
import pathlib
import random
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from exact_swap_wwm_core import (
    SelectionAudit,
    apply_inherited_tokenwise_corruption,
    count_selected_groups,
    group_positions,
    innovation_biased_selection,
)


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
BASE_TRAINER_PATH = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_METADATA = WORKSPACE / "analysis/innovation_metadata_train.jsonl"
DEFAULT_TRAIN_100M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
DEFAULT_TOKENIZER = WORKSPACE / "data/compliant_tokenizer"

EXPECTED_TRAIN_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_TOKENIZER_JSON_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
EXPECTED_METADATA_SHA = "351a95362db3b074f4bc5be6b3871d10861482371f380420f9f62b866ba74520"


def load_base_trainer():
    spec = importlib.util.spec_from_file_location("compact_experience_base_trainer", BASE_TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base trainer from {BASE_TRAINER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_base_trainer()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_innovation_metadata(path: pathlib.Path) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    metadata: dict[int, dict[str, Any]] = {}
    totals: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("schema") != "frontier_consolidation_innovation_wwm_train_v1":
                raise RuntimeError(f"unexpected metadata schema at line {line_no}: {obj.get('schema')}")
            eid = int(obj["example_id"])
            if eid in metadata:
                raise RuntimeError(f"duplicate metadata example_id {eid}")
            metadata[eid] = obj
            totals["records"] += 1
            totals["innovation_groups"] += len(obj.get("innovation_groups") or [])
            totals["copyable_groups"] += len(obj.get("copyable_group_ids") or [])
            totals["protected_source_groups"] += len(obj.get("protected_source_group_ids") or [])
            totals["protected_pair_groups"] += len(obj.get("protected_pair_group_ids") or [])
            totals["innovation_tokens"] += sum(int(t.get("token_count", 0)) for t in obj.get("innovation_groups") or [])
            totals["rows_with_innovation"] += int(bool(obj.get("innovation_groups")))
            totals["frozen_stream_occurrences_sum"] += int(obj.get("frozen_stream_occurrences", 0))
    return metadata, dict(totals)


class ExactSwapMaskedChunkDataset(Dataset):
    """Base masked chunk dataset plus example_id for exact-swap metadata lookup."""

    def __init__(self, examples, tokenizer, seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start: dict[int, bool] = {}

    def _word_start_flag(self, token_id: int) -> bool:
        val = self._word_start.get(token_id)
        if val is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            val = bool(s is not None and base.is_word_start(str(s)))
            self._word_start[token_id] = val
        return val

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        group = torch.full_like(input_ids, -1)
        gid = -1
        for i in range(input_ids.shape[0]):
            if attention_mask[i] == 0:
                continue
            tid = int(input_ids[i])
            if tid in self.special_ids:
                continue
            if gid < 0 or self._word_start_flag(tid) or i == 0:
                gid += 1
            group[i] = gid
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_group": group,
            "words": ex.words,
            "example_id": ex.example_id,
        }


def exact_swap_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
        "example_ids": torch.tensor([x["example_id"] for x in batch], dtype=torch.long),
    }


def _selected_group_ids(select_row: torch.Tensor, positions_by_gid: dict[int, tuple[int, ...]]) -> set[int]:
    return {gid for gid, positions in positions_by_gid.items() if bool(select_row[list(positions)].all())}


def summarize_exact_swap_audit(
    audit: SelectionAudit,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    row_metadata: list[dict[str, Any] | None],
    special_ids: list[int],
) -> dict[str, int]:
    device = input_ids.device
    special = torch.tensor(sorted(int(x) for x in special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special)
    stats: dict[str, int] = defaultdict(int)
    for k, v in audit.counters.items():
        # `baseline_selected_innovation_groups` is recomputed below by category;
        # keep the core-side counter under a separate name to avoid double counting.
        out_key = "counter_baseline_selected_innovation_groups" if k == "baseline_selected_innovation_groups" else k
        stats[out_key] += int(v)
    stats["baseline_selected_tokens"] += int(audit.baseline_select.sum().item())
    stats["biased_selected_tokens"] += int(audit.biased_select.sum().item())
    stats["selected_token_delta"] += int(audit.biased_select.sum().item() - audit.baseline_select.sum().item())
    base_groups = count_selected_groups(audit.baseline_select, word_group, candidate)
    biased_groups = count_selected_groups(audit.biased_select, word_group, candidate)
    stats["baseline_selected_groups"] += int(base_groups)
    stats["biased_selected_groups"] += int(biased_groups)
    stats["selected_group_delta"] += int(biased_groups - base_groups)
    stats["batch_mass_mismatch"] += int(base_groups != biased_groups or int(audit.baseline_select.sum()) != int(audit.biased_select.sum()))
    for b, meta in enumerate(row_metadata):
        if not meta:
            continue
        pos_by_gid = group_positions(word_group[b], candidate[b])
        base_sel = _selected_group_ids(audit.baseline_select[b], pos_by_gid)
        biased_sel = _selected_group_ids(audit.biased_select[b], pos_by_gid)
        innovation = {int(t["group_id"]) for t in meta.get("innovation_groups") or []}
        copyable = {int(x) for x in meta.get("copyable_group_ids") or []}
        source = {int(x) for x in meta.get("protected_source_group_ids") or []}
        protected_pair = {int(x) for x in meta.get("protected_pair_group_ids") or []}
        for name, group_set in [
            ("innovation", innovation),
            ("copyable", copyable),
            ("source", source),
            ("protected_pair", protected_pair),
        ]:
            if not group_set:
                continue
            stats[f"baseline_selected_{name}_groups"] += len(base_sel & group_set)
            stats[f"biased_selected_{name}_groups"] += len(biased_sel & group_set)
            stats[f"{name}_group_selection_delta"] += len(biased_sel & group_set) - len(base_sel & group_set)
    return dict(stats)


def apply_exact_swap_masking(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    example_ids: torch.Tensor,
    tokenizer,
    mask_prob: float,
    metadata: dict[int, dict[str, Any]],
    gen: torch.Generator,
    priority_salt: str,
    max_swaps_per_changed_row: int,
    return_events: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], list[dict[str, Any]]]:
    row_metadata = [metadata.get(int(eid.item())) for eid in example_ids]
    audit = innovation_biased_selection(
        input_ids=input_ids,
        attention_mask=attention_mask,
        word_group=word_group,
        special_ids=tokenizer.all_special_ids,
        mask_prob=mask_prob,
        generator=gen,
        row_metadata=row_metadata,
        priority_salt=priority_salt,
        max_swaps_per_changed_row=max_swaps_per_changed_row,
    )
    masked_inputs, labels, _ = apply_inherited_tokenwise_corruption(
        input_ids=input_ids,
        select=audit.biased_select,
        mask_token_id=int(tokenizer.mask_token_id),
        vocab_size=len(tokenizer),
        generator=gen,
    )
    stats = summarize_exact_swap_audit(
        audit, input_ids, attention_mask, word_group, row_metadata, list(tokenizer.all_special_ids)
    )
    return masked_inputs, labels, stats, (audit.events if return_events else [])


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact-swap conditional-innovation WWM trainer")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--example_jsonl", default=str(DEFAULT_TRAIN_100M))
    p.add_argument("--example_jsonl_label", default="compact_view_reinvest")
    p.add_argument("--tokenizer_path", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--tokenizer_label", default="legal16k")
    p.add_argument("--metadata_train", default=str(DEFAULT_METADATA))
    p.add_argument("--max_word_exposure", type=int, default=100_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--max_swaps_per_changed_row", type=int, default=1)
    p.add_argument("--priority_salt_prefix", default="")
    p.add_argument("--preflight_only", action="store_true")
    p.add_argument("--verify_hashes", action="store_true")
    # Model shape matching research legal16k reinvest.
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--seq_length", type=int, default=256)
    # Optimization matching research.
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=43022)
    p.add_argument("--train_rng_seed", type=int, default=43023)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = build_args()
    start = time.time()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.max_seq_length != 256 or args.seq_length != 256:
        raise RuntimeError("exact-swap metadata was built for fixed seq256")
    if abs(args.mask_prob - 0.15) > 1e-12:
        raise RuntimeError("exact-swap screen is only defined for fixed 15% WWM")

    train_path = pathlib.Path(args.example_jsonl)
    tokenizer_path = pathlib.Path(args.tokenizer_path)
    metadata_path = pathlib.Path(args.metadata_train)
    if args.verify_hashes:
        observed_train = sha256_file(train_path)
        observed_tok = sha256_file(tokenizer_path / "tokenizer.json")
        observed_meta = sha256_file(metadata_path)
        if observed_train != EXPECTED_TRAIN_SHA:
            raise RuntimeError(f"train SHA mismatch: {observed_train}")
        if observed_tok != EXPECTED_TOKENIZER_JSON_SHA:
            raise RuntimeError(f"tokenizer.json SHA mismatch: {observed_tok}")
        if observed_meta != EXPECTED_METADATA_SHA:
            raise RuntimeError(f"metadata SHA mismatch: {observed_meta}")

    metadata, metadata_totals = load_innovation_metadata(metadata_path)
    header = {
        "event": "exact_swap_metadata_loaded",
        "records": len(metadata),
        "metadata_totals": metadata_totals,
        "mask_prob": args.mask_prob,
        "max_swaps_per_changed_row": args.max_swaps_per_changed_row,
        "metadata_train": str(metadata_path),
        "time": now_utc(),
    }
    (out_dir / "metadata_header.json").write_text(json.dumps(header, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(header, ensure_ascii=False), flush=True)
    if args.preflight_only:
        result = {
            "event": "preflight_only_done",
            "expected_train_sha": EXPECTED_TRAIN_SHA,
            "expected_tokenizer_json_sha": EXPECTED_TOKENIZER_JSON_SHA,
            "expected_metadata_sha": EXPECTED_METADATA_SHA,
            "elapsed_sec": round(time.time() - start, 3),
            "time": now_utc(),
        }
        (out_dir / "preflight_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return

    tokenizer = base.make_portable_tokenizer(str(tokenizer_path))
    examples, jsonl_total_words, jsonl_total_rows, sample_rows = base.load_examples_jsonl(
        train_path, args.max_word_exposure
    )
    actual_words = sum(ex.words for ex in examples)
    if actual_words != args.max_word_exposure:
        raise RuntimeError(f"word mismatch {actual_words} vs {args.max_word_exposure}")

    dataset = ExactSwapMaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=exact_swap_collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    total_steps = len(loader)

    reset_all_rng(args.seed)
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = base.build_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)
    param_count = sum(p.numel() for p in model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)
    salt_prefix = args.priority_salt_prefix or f"seed={args.seed}:train_rng={args.train_rng_seed}"

    manifest = {
        "session": "frontier_consolidation",
        "step_label": "exact_swap_conditional_innovation_wwm",
        "description": "exact-swap source-absent rewrite innovation WWM on legal16k compact-view reinvest",
        "metadata_train": str(metadata_path),
        "metadata_totals": metadata_totals,
        "mask_prob": args.mask_prob,
        "max_swaps_per_changed_row": args.max_swaps_per_changed_row,
        "tokenizer_path": str(tokenizer_path),
        "example_jsonl": str(train_path),
        "max_word_exposure": args.max_word_exposure,
        "param_count": param_count,
        "total_steps": total_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "priority_salt_prefix": salt_prefix,
        "jsonl_total_words": jsonl_total_words,
        "jsonl_total_rows": jsonl_total_rows,
        "sample_rows_without_text": sample_rows,
        "created_utc": now_utc(),
    }
    (out_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    log_path = out_dir / "training_log.jsonl"
    event_path = out_dir / "swap_events_sample.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict[str, Any]] = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    cumulative_mask_stats: dict[str, int] = defaultdict(int)
    event_budget = 2000

    model.train()
    with log_path.open("w", encoding="utf-8") as logf, event_path.open("w", encoding="utf-8") as evf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            example_ids = batch.pop("example_ids")
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)

            salt = f"{salt_prefix}:step={step}"
            masked_inputs, labels, mask_stats, events = apply_exact_swap_masking(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_group=word_group,
                example_ids=example_ids,
                tokenizer=tokenizer,
                mask_prob=args.mask_prob,
                metadata=metadata,
                gen=gen,
                priority_salt=salt,
                max_swaps_per_changed_row=args.max_swaps_per_changed_row,
                return_events=(event_budget > 0),
            )
            if mask_stats.get("selected_token_delta", 0) != 0 or mask_stats.get("selected_group_delta", 0) != 0:
                raise RuntimeError(f"exact-swap mass invariant failed at step {step}: {mask_stats}")
            if mask_stats.get("forced_copyable", 0) != 0:
                raise RuntimeError(f"copyable target forced at step {step}: {mask_stats}")
            if mask_stats.get("donor_protected_pair", 0) != 0 or mask_stats.get("donor_protected_source", 0) != 0:
                raise RuntimeError(f"protected-pair/source donor at step {step}: {mask_stats}")
            for k, v in mask_stats.items():
                cumulative_mask_stats[k] += int(v)
            if event_budget > 0:
                for ev in events[:event_budget]:
                    evf.write(json.dumps({"step": step, **ev}, ensure_ascii=False) + "\n")
                    event_budget -= 1
                    if event_budget <= 0:
                        break
                evf.flush()

            optim.zero_grad(set_to_none=True)
            out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            loss = out_model.loss
            if loss is None:
                raise RuntimeError("model returned no loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            cumulative_words += words
            loss_float = float(loss.detach().cpu())
            loss_values.append(loss_float)
            n_pred = int((labels != -100).sum().item())
            n_candidate = int(attention_mask.bool().sum().item())
            effective_mask_rate = n_pred / max(1, n_candidate)

            if step % args.log_every == 0 or step == 1:
                avg_loss = sum(loss_values[-args.log_every:]) / len(loss_values[-args.log_every:])
                entry = {
                    "step": step,
                    "loss": round(loss_float, 6),
                    "avg_loss": round(avg_loss, 6),
                    "words": cumulative_words,
                    "lr": round(sched.get_last_lr()[0], 8),
                    "mask_rate": round(effective_mask_rate, 4),
                    "n_pred": n_pred,
                    "successful_swaps_cumulative": cumulative_mask_stats.get("successful_swaps", 0),
                    "changed_row_exposures_cumulative": cumulative_mask_stats.get("changed_row_exposures", 0),
                }
                logf.write(json.dumps(entry) + "\n")
                logf.flush()
                print(json.dumps({"event": "log", **entry, "time": now_utc()}), flush=True)

            if next_ckpt is not None and cumulative_words >= next_ckpt:
                ckpt_name = f"chck_{cumulative_words // 1_000_000}M"
                ckpt_dir = out_dir / "hf_model" / ckpt_name
                base.save_hf_checkpoint(model, tokenizer, ckpt_dir)
                ckpt_info = {
                    "name": ckpt_name,
                    "step": step,
                    "words": cumulative_words,
                    "loss": round(loss_float, 6),
                    "path": str(ckpt_dir),
                    "mask_rate": round(effective_mask_rate, 4),
                    "successful_swaps_cumulative": cumulative_mask_stats.get("successful_swaps", 0),
                }
                saved_checkpoints.append(ckpt_info)
                print(json.dumps({"event": "checkpoint", **ckpt_info, "time": now_utc()}), flush=True)
                next_ckpt += args.checkpoint_words

    elapsed = time.time() - start
    final_loss = loss_values[-1] if loss_values else 0.0
    avg_last_50 = sum(loss_values[-50:]) / max(1, len(loss_values[-50:]))
    sci_metrics = {
        "session": "frontier_consolidation",
        "step_label": "exact_swap_conditional_innovation_wwm",
        "description": manifest["description"],
        "mask_prob": args.mask_prob,
        "max_swaps_per_changed_row": args.max_swaps_per_changed_row,
        "final_loss": round(final_loss, 6),
        "avg_last_50_loss": round(avg_last_50, 6),
        "total_steps": total_steps,
        "param_count": param_count,
        "total_words": cumulative_words,
        "checkpoints": [c["name"] for c in saved_checkpoints],
        "cumulative_mask_stats": dict(cumulative_mask_stats),
        "elapsed_sec": round(elapsed, 1),
        "finished_utc": now_utc(),
    }
    (out_dir / "scientific_metrics.json").write_text(json.dumps(sci_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "done",
        "final_loss": sci_metrics["final_loss"],
        "total_steps": total_steps,
        "total_words": cumulative_words,
        "param_count": param_count,
        "elapsed_sec": sci_metrics["elapsed_sec"],
        "successful_swaps": cumulative_mask_stats.get("successful_swaps", 0),
    }), flush=True)


if __name__ == "__main__":
    main()
