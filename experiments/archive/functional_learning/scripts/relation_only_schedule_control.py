#!/usr/bin/env python3
"""research: compressed relation-only acquisition control on the bridge schedule.

Scientific question
-------------------
The corrected legal bridge showed that answer-allocation inside ordinary continuation
mostly removes a shared replacement preference and gives only weak correct-source vs
wrong-source separation.  This script asks whether the same relation rows, grouped by
the same 354 word-paced bridge macro-updates and trained under the same LR schedule,
can acquire the contextual state-selection operation when ordinary-loss updates are
removed.

This is not a new BabyLM candidate by itself.  It is a control that distinguishes:
  * failure caused already by compressed relation optimization opportunities/schedule;
  * failure that appears only when relation credit is mixed with ordinary continuation;
  * a trajectory where source retrieval appears but update/retain balance fails.

The script preserves the original bridge stream for batching: each macro-update is
formed by reading the interspersed legal overlay until the same words-per-update
threshold is reached, then only the relation rows from that macro are used for the
answer-span objective.  Ordinary rows are not forwarded and do not contribute loss.
Both training-map and held-map counterfactual behavior are scored from saved
checkpoints, with the same simultaneous full-span scoring used in Steps 041/047.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import gc
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import corrected_bridge_trainer as tr  # noqa: E402
import bridge_eval as beval  # noqa: E402
import threeway_and_reassignment_probe as probe  # noqa: E402
import saved_state_replicate_neutral_threeentity as repl  # noqa: E402

DEFAULT_OVERLAY = _public_path('experiments/archive/functional_learning/data/bridge_schedule_plan/overlay_tail_relation_e080_interspersed_wordpaced.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/relation_only_schedule354')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def finite_mean(xs: Iterable[float]) -> float:
    vals = []
    for x in xs:
        try:
            xf = float(x)
        except Exception:
            continue
        if math.isfinite(xf):
            vals.append(xf)
    return sum(vals) / len(vals) if vals else float("nan")


def write_jsonl(path: pathlib.Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def bridge_macro_batches(overlay: Sequence[Dict[str, Any]], words_per_update: int, max_updates: int) -> List[Dict[str, Any]]:
    """Reconstruct the exact research word-paced macro-update partition."""
    cursor = 0
    batches: List[Dict[str, Any]] = []
    for update_i in range(int(max_updates)):
        rows: List[Dict[str, Any]] = []
        words = 0
        while cursor < len(overlay) and words < int(words_per_update):
            row = overlay[cursor]
            rows.append(row)
            words += int(row.get("words", len(str(row.get("text", "")).split())))
            cursor += 1
        if not rows:
            break
        rel_rows = [r for r in rows if r.get("bridge_kind") == "relation_answer_packet"]
        batches.append({
            "update_index0": update_i,
            "update": update_i + 1,
            "rows": rows,
            "relation_rows": rel_rows,
            "words": words,
            "relation_words": sum(int(r.get("words", 0)) for r in rel_rows),
            "ordinary_words": words - sum(int(r.get("words", 0)) for r in rel_rows),
            "n_rows": len(rows),
            "n_relation_rows": len(rel_rows),
            "n_ordinary_rows": len(rows) - len(rel_rows),
        })
        if cursor >= len(overlay):
            break
    return batches


def one_relation_step(model, opt, rows: Sequence[Dict[str, Any]], tokenizer, wgb, device, args, update_i: int) -> Dict[str, Any]:
    examples, prep_stats = tr.prepare_targets_for_macro(
        list(rows), tokenizer, int(args.seq_length), wgb, "answer_allocation", int(args.train_seed), float(args.mask_prob)
    )
    rel_n = int(prep_stats["relation_target_tokens"])
    if rel_n <= 0:
        raise RuntimeError({"error": "no relation targets", "update": update_i + 1, "prep_stats": prep_stats})
    if int(prep_stats.get("ordinary_rows", 0)) != 0 or int(prep_stats.get("ordinary_target_tokens", 0)) != 0:
        raise RuntimeError({"error": "ordinary rows leaked into relation-only objective", "update": update_i + 1, "prep_stats": prep_stats})

    opt.zero_grad(set_to_none=True)
    rel_sum_total = 0.0
    rel_seen = 0
    for mb_start in range(0, len(examples), int(args.micro_batch)):
        mb = examples[mb_start:mb_start + int(args.micro_batch)]
        input_ids = torch.stack([x["input_ids"] for x in mb]).to(device)
        attention_mask = torch.stack([x["attention_mask"] for x in mb]).to(device)
        labels = torch.stack([x["labels"] for x in mb]).to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        vocab = out.logits.shape[-1]
        ce = F.cross_entropy(out.logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100, reduction="none").view_as(labels)
        active = labels != -100
        rel_sum = ce[active].sum() if active.any() else torch.tensor(0.0, device=device)
        loss = rel_sum / float(rel_n)
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"bad loss at update {update_i + 1}, mb_start {mb_start}")
        loss.backward()
        rel_sum_total += float(rel_sum.detach().cpu())
        rel_seen += int(active.sum().detach().cpu())
        del input_ids, attention_mask, labels, out, ce, active, rel_sum, loss
    if rel_seen != rel_n:
        raise RuntimeError(f"macro target count mismatch update {update_i + 1}: prepared={rel_n}, seen={rel_seen}")
    grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.max_grad_norm)).detach().cpu())
    opt.step()
    return {
        "relation_target_tokens": rel_n,
        "answer_span_rows": int(prep_stats["answer_span_rows"]),
        "relation_zero_label_rows": int(prep_stats["relation_zero_label_rows"]),
        "relation_loss": rel_sum_total / float(rel_n),
        "grad_norm_preclip": grad_norm,
    }


def score_pair_set(model, tokenizer, pairs: Sequence[Dict[str, Any]], device, seq_length: int, label: str, raw_dir: pathlib.Path) -> Dict[str, Any]:
    pair_results = []
    for i, p in enumerate(pairs):
        pair_results.append(probe.threeway_score_pair(model, tokenizer, p, device, seq_length))
        if (i + 1) % 30 == 0:
            print(json.dumps({"event": "pairset_threeway_progress", "label": label, "n": i + 1, "total": len(pairs)}), flush=True)
    tw_summary, tw_records = repl.analyze_threeway_extended(pair_results, pairs, label)
    reassign_summary, reassign_records, reassign_n = repl.score_reassignment_held(model, tokenizer, pairs, pair_results, device, seq_length)
    write_jsonl(raw_dir / f"{label}_threeway_records.jsonl", tw_records)
    write_jsonl(raw_dir / f"{label}_reassignment_records.jsonl", reassign_records)
    return {
        "threeway": tw_summary,
        "margin_summary": beval.extra_margin_summary(tw_records),
        "reassignment": reassign_summary,
        "reassignment_valid_n": reassign_n,
    }


def score_checkpoint(ckpt_path: pathlib.Path, label: str, tokenizer, train_pairs, held_pairs, three_cases, device, seq_length: int, out_dir: pathlib.Path, include_three_entity: bool = True) -> Dict[str, Any]:
    print(json.dumps({"event": "score_checkpoint_start", "label": label, "path": rel(ckpt_path)}), flush=True)
    t0 = time.time()
    if ckpt_path == beval.PARENT_PATH:
        model, ident = beval.load_parent(device, 0.75)
    else:
        model, ident = beval.strict_load_checkpoint(ckpt_path, device, 0.75)
    raw_dir = out_dir / "raw_records"
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_res = score_pair_set(model, tokenizer, train_pairs, device, seq_length, f"{label}_train", raw_dir)
    held_res = score_pair_set(model, tokenizer, held_pairs, device, seq_length, f"{label}_held", raw_dir)
    three_summary: Dict[str, Any] = {}
    three_records: List[Dict[str, Any]] = []
    if include_three_entity:
        three_scored = [repl.score_three_entity_case(model, tokenizer, c, device, seq_length) for c in three_cases]
        three_summary, three_records = repl.analyze_three_entity(three_scored, three_cases, f"{label}_three_entity")
        write_jsonl(raw_dir / f"{label}_three_entity_records.jsonl", three_records)
    rec = {
        "label": label,
        "path": rel(ckpt_path),
        "model_identity": ident,
        "train_relation": train_res,
        "held_relation": held_res,
        "held_three_entity": three_summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(json.dumps({
        "event": "score_checkpoint_done",
        "label": label,
        "train_neutral_full": train_res["threeway"].get("neutral_both_full_source"),
        "train_retain_full": train_res["threeway"].get("retain_both_full_source"),
        "held_neutral_full": held_res["threeway"].get("neutral_both_full_source"),
        "held_retain_full": held_res["threeway"].get("retain_both_full_source"),
        "held_reassign_both": held_res["reassignment"].get("n_swap_both_follow"),
        "held_three_entity": three_summary.get("update_c_both_query_indexed"),
    }), flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def train(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "hf_cache"
    os.environ["HF_HOME"] = str(cache.resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    (cache / "modules").mkdir(parents=True, exist_ok=True)

    tr.reset_all(int(args.train_seed))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    overlay_path = pathlib.Path(args.overlay_jsonl)
    overlay = tr.load_jsonl(overlay_path)
    batches = bridge_macro_batches(overlay, int(args.words_per_update), int(args.max_updates))
    empty_relation_updates = [b["update"] for b in batches if not b["relation_rows"]]
    if empty_relation_updates and not args.allow_empty_relation_updates:
        raise RuntimeError({"empty_relation_updates": empty_relation_updates[:20], "n_empty": len(empty_relation_updates)})

    model, missing, unexpected = tr.load_model(device, float(args.private_scale))
    ident = tr.model_identity(model)
    if ident["class"] != "FrozenSlowPrivateDebertaV2ForMaskedLM" or int(ident["private_adapter_params"]) != 995584:
        raise RuntimeError(f"Bad model identity: {ident}")
    tokenizer = AutoTokenizer.from_pretrained(str(tr.PARENT_PATH), local_files_only=True, use_fast=True)
    wgb = tr.WordGroupBuilder(tokenizer)
    opt, opt_info = tr.freeze_to_private_optimizer(model, float(args.lr), float(args.weight_decay))
    if torch.cuda.is_available():
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    batch_summary = {
        "status": "RELATION_ONLY_SCHEDULE_CONFIG",
        "scientific_question": "Can the same bridge relation rows acquire contextual selection under the same 354 macro-update grouping/LR schedule when ordinary-loss updates are removed?",
        "overlay_jsonl": rel(overlay_path),
        "out_dir": rel(out_dir),
        "parent_path": rel(tr.PARENT_PATH),
        "schedule_definition": "research word-paced batches over the full interspersed overlay; only relation_answer_packet rows from each original macro-update are forwarded and optimized.",
        "ordinary_loss": "absent; ordinary rows define the macro partition only and are not forwarded",
        "objective": "answer-span cross entropy mean over relation rows within each original macro-update",
        "n_original_overlay_rows": len(overlay),
        "n_batches": len(batches),
        "empty_relation_updates": empty_relation_updates,
        "total_words_in_original_batches": sum(int(b["words"]) for b in batches),
        "total_relation_rows": sum(int(b["n_relation_rows"]) for b in batches),
        "total_relation_words": sum(int(b["relation_words"]) for b in batches),
        "total_ordinary_words_partition_only": sum(int(b["ordinary_words"]) for b in batches),
        "words_per_update": int(args.words_per_update),
        "max_updates": int(args.max_updates),
        "train_seed": int(args.train_seed),
        "lr_peak": float(args.lr),
        "schedule_total": int(args.schedule_total),
        "schedule_offset": int(args.schedule_offset),
        "warmup": int(args.warmup),
        "private_scale": float(args.private_scale),
        "model_identity": ident,
        "optimizer": {"trainable_tensors": opt_info["trainable_tensors"], "trainable_params": opt_info["trainable_params"]},
    }
    (out_dir / "relation_only_config.json").write_text(json.dumps(batch_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "config", **batch_summary}, ensure_ascii=False), flush=True)

    model.train()
    logs: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    cum_original_words = 0
    cum_relation_words = 0
    cum_relation_rows = 0
    cum_relation_targets = 0
    t0 = time.time()
    for b in batches:
        update_i = int(b["update_index0"])
        schedule_idx = int(args.schedule_offset) + update_i
        lr = tr.lr_at_update(schedule_idx, int(args.schedule_total), int(args.warmup), float(args.lr))
        for pg in opt.param_groups:
            pg["lr"] = lr
        if not b["relation_rows"]:
            log = {
                "update": b["update"],
                "update_index0": update_i,
                "schedule_idx": schedule_idx,
                "lr": lr,
                "original_rows": int(b["n_rows"]),
                "original_words": int(b["words"]),
                "relation_rows": 0,
                "relation_words": 0,
                "ordinary_rows_partition_only": int(b["n_ordinary_rows"]),
                "ordinary_words_partition_only": int(b["ordinary_words"]),
                "relation_target_tokens": 0,
                "relation_loss": None,
                "grad_norm_preclip": None,
                "skipped_empty_relation": True,
                "elapsed_sec": round(time.time() - t0, 1),
            }
        else:
            step_stats = one_relation_step(model, opt, b["relation_rows"], tokenizer, wgb, device, args, update_i)
            log = {
                "update": b["update"],
                "update_index0": update_i,
                "schedule_idx": schedule_idx,
                "lr": lr,
                "original_rows": int(b["n_rows"]),
                "original_words": int(b["words"]),
                "relation_rows": int(b["n_relation_rows"]),
                "relation_words": int(b["relation_words"]),
                "ordinary_rows_partition_only": int(b["n_ordinary_rows"]),
                "ordinary_words_partition_only": int(b["ordinary_words"]),
                **step_stats,
                "skipped_empty_relation": False,
                "elapsed_sec": round(time.time() - t0, 1),
            }
        logs.append(log)
        cum_original_words += int(b["words"])
        cum_relation_words += int(b["relation_words"])
        cum_relation_rows += int(b["n_relation_rows"])
        cum_relation_targets += int(log.get("relation_target_tokens") or 0)
        if b["update"] == 1 or b["update"] % int(args.log_every) == 0:
            print(json.dumps({"event": "update", **log}, ensure_ascii=False), flush=True)
        if (b["update"] % int(args.checkpoint_every) == 0) or b["update"] == len(batches) or b["update"] == int(args.max_updates):
            ckpt_name = f"update_{b['update']:04d}"
            ckpt_dir = out_dir / "checkpoints" / ckpt_name
            metadata = {
                "arm": "relation_only_answer_schedule354",
                "update": int(b["update"]),
                "schedule_idx": schedule_idx,
                "cum_original_words_partitioned": cum_original_words,
                "cum_relation_words_optimized": cum_relation_words,
                "cum_relation_rows_optimized": cum_relation_rows,
                "cum_relation_targets": cum_relation_targets,
                "ordinary_loss_updates": 0,
                "ordinary_words_partition_only_seen": cum_original_words - cum_relation_words,
                "relation_loss_last_update": log.get("relation_loss"),
                "train_seed": int(args.train_seed),
                "private_scale": float(args.private_scale),
                "model_identity": ident,
                "scientific_status": "compressed relation-only acquisition control; not a legal BabyLM continuation candidate",
            }
            tr.save_checkpoint(model, tokenizer, ckpt_dir, metadata, float(args.private_scale))
            checkpoints.append({"update": int(b["update"]), "path": rel(ckpt_dir), "relation_loss": log.get("relation_loss")})
            print(json.dumps({"event": "checkpoint", "update": int(b["update"]), "path": rel(ckpt_dir)}, ensure_ascii=False), flush=True)

    with (out_dir / "update_log.jsonl").open("w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
    train_summary = {
        "status": "RELATION_ONLY_SCHEDULE_TRAIN_DONE",
        "out_dir": rel(out_dir),
        "overlay_jsonl": rel(overlay_path),
        "completed_updates": len(logs),
        "empty_relation_updates": empty_relation_updates,
        "total_original_words_partitioned": cum_original_words,
        "total_relation_words_optimized": cum_relation_words,
        "total_relation_rows_optimized": cum_relation_rows,
        "total_relation_targets": cum_relation_targets,
        "final_update": logs[-1] if logs else None,
        "checkpoints": checkpoints,
        "elapsed_sec": round(time.time() - t0, 1),
        "model_identity": ident,
        "scientific_status": "This control tests compressed acquisition opportunity/schedule separately from ordinary-loss interference; it is not a broad-competence model candidate.",
    }
    (out_dir / "relation_only_train_summary.json").write_text(json.dumps(train_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(train_summary, indent=2, ensure_ascii=False), flush=True)

    if args.eval_after:
        model.cpu()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        eval_device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        pairs, train_pairs, held_pairs, _train_rows, _held_rows = repl.fixed_repaired_pairs(int(args.construction_seed))
        three_cases = repl.build_three_entity_cases(pairs, train_pairs, held_pairs, int(args.construction_seed), max_cases=12)
        eval_results: List[Dict[str, Any]] = []
        if args.include_parent_eval:
            eval_results.append(score_checkpoint(beval.PARENT_PATH, "coherent86_parent", tokenizer, train_pairs, held_pairs, three_cases, eval_device, int(args.seq_length), out_dir))
            (out_dir / "relation_only_eval_partial.json").write_text(json.dumps({"results": eval_results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        ckpt_map = {pathlib.Path(c["path"]).name: ROOT / c["path"] for c in checkpoints}
        for spec in [s.strip() for s in args.eval_checkpoints.split(",") if s.strip()]:
            name = spec if spec.startswith("update_") else f"update_{int(spec):04d}"
            if name not in ckpt_map:
                raise FileNotFoundError({"requested_eval_checkpoint": name, "available": sorted(ckpt_map)})
            eval_results.append(score_checkpoint(ckpt_map[name], name, tokenizer, train_pairs, held_pairs, three_cases, eval_device, int(args.seq_length), out_dir))
            (out_dir / "relation_only_eval_partial.json").write_text(json.dumps({"results": eval_results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        eval_summary = {
            "status": "RELATION_ONLY_SCHEDULE_EVAL_DONE",
            "scientific_question": batch_summary["scientific_question"],
            "train_summary": rel(out_dir / "relation_only_train_summary.json"),
            "results": eval_results,
            "full_source_rule": "correct source must outrank wrong source and shared replacement; no ordering between wrong source and replacement is required",
        }
        (out_dir / "relation_only_eval_summary.json").write_text(json.dumps(eval_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(eval_summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overlay-jsonl", default=str(DEFAULT_OVERLAY))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--max-updates", type=int, default=354)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--schedule-total", type=int, default=455)
    ap.add_argument("--schedule-offset", type=int, default=101)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--train-seed", type=int, default=47047)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--construction-seed", type=int, default=40040)
    ap.add_argument("--allow-empty-relation-updates", action="store_true")
    ap.add_argument("--eval-after", action="store_true")
    ap.add_argument("--include-parent-eval", action="store_true")
    ap.add_argument("--eval-checkpoints", default="150,354")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
