#!/usr/bin/env python3
"""research: real-stream learner comparison for structural compact policy.

This script compares two objectives on the SAME packed legal-stream prefix:

1. inherited_wwm: standard row-keyed 15% whole-word masking on all ordinary rows;
2. correspondence_focus: ordinary WWM on non-Qwen rows, but for qwen_pair_packed
   rows uses source-visible view-focused targets on the second-view segment of
   every source/view pair. This keeps ordinary learning elsewhere while transporting
   the successful research learning condition (source prefix visible, credit on view
   words) into realistic packed stream order and competing ordinary experience.

The stream is built by research structural policy materialization. Its compacted rows
are *unverified structural candidates*, not semantic faithful labels. Therefore a
positive result would support this practical mixed policy only, not a clean compaction
principle. The early prefix run is designed to cheaply determine whether focused
correspondence credit survives realistic packing before spending full-tail compute.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import corrected_bridge_trainer as bridge  # noqa: E402
import concentrated_compact_learning_test as s57  # noqa: E402

DEFAULT_TAIL = _public_path('experiments/archive/functional_learning/data/structural_policy_materialization/compact_structural_reinvest/compact_structural_reinvest_reference_tail_wordpaced_segments.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/real_stream_policy_comparison')
DEFAULT_LABELS = _public_path('experiments/archive/functional_learning/data/selective_reviewed_labels/selective_semantic_labels.jsonl')
DEFAULT_AUX_TASKS = _public_path('experiments/archive/functional_learning/data/allocation_design_v2/aux_support_tasks_repaired.jsonl')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_seed(*parts: Any) -> int:
    h = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def wc(text: str) -> int:
    return len((text or "").strip().split())


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def row_key(row: Dict[str, Any]) -> str:
    return bridge.row_key(row)


def content_word_spans(text: str) -> List[Tuple[int, int, str]]:
    return s57.content_word_spans(text)


def locate_positions(offsets: torch.Tensor, start: int, end: int) -> List[int]:
    pos: List[int] = []
    for i in range(int(offsets.shape[0])):
        s, e = int(offsets[i][0]), int(offsets[i][1])
        if e <= s:
            continue
        if s < end and e > start:
            pos.append(i)
    return pos


def lr_at_update(schedule_idx: int, schedule_total: int, warmup: int, peak: float) -> float:
    return bridge.lr_at_update(schedule_idx, schedule_total, warmup, peak)


def load_prefix(tail_jsonl: pathlib.Path, max_updates: int, words_per_update: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total_words = 0
    max_words = int(max_updates) * int(words_per_update)
    qwen_rows = 0
    compact_rows = 0
    compact_occ = 0
    segment_count = 0
    by_source = Counter()
    with tail_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if total_words >= max_words:
                break
            r = json.loads(line)
            w = int(r.get("words", wc(r.get("text", ""))))
            rows.append(r)
            total_words += w
            by_source[str(r.get("source", ""))] += 1
            if r.get("source") == "qwen_pair_packed":
                qwen_rows += 1
                segment_count += len(r.get("qwen_pair_segments") or [])
            if r.get("compact_modified_pair_ids"):
                compact_rows += 1
                compact_occ += len(r.get("compact_modified_pair_ids") or [])
    return rows, {
        "tail_jsonl": rel(tail_jsonl),
        "requested_max_updates": int(max_updates),
        "words_per_update": int(words_per_update),
        "prefix_rows": len(rows),
        "prefix_words": total_words,
        "target_prefix_words": max_words,
        "qwen_rows": qwen_rows,
        "qwen_pair_segments": segment_count,
        "compact_modified_rows": compact_rows,
        "compact_modified_pair_occurrences": compact_occ,
        "source_rows": dict(by_source),
    }


def apply_view_focus_row(row: Dict[str, Any], tok: Dict[str, Any], tokenizer, seed: int,
                         focus_prob: float, max_focus_groups_per_row: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Mask content-word groups in qwen second-view segments only.

    Source spans remain visible. Candidate groups are restricted to content words in
    the view_text field of each qwen_pair_segment and located by character offsets
    in the packed row. If no group is sampled, one group is chosen to prevent qwen
    correspondence rows from receiving zero view credit.
    """
    input_ids: torch.Tensor = tok["input_ids"]
    offsets: torch.Tensor = tok["offsets"]
    attention_mask: torch.Tensor = tok["attention_mask"]
    masked = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    groups: List[Dict[str, Any]] = []
    text = str(row.get("text", ""))
    for seg_i, seg in enumerate(row.get("qwen_pair_segments") or []):
        view_text = str(seg.get("view_text", ""))
        vstart = int(seg.get("view_start", -1))
        vend = int(seg.get("view_end", -1))
        if vstart < 0 or vend <= vstart or vend > len(text):
            continue
        # Use supplied view_text for spans, but char offsets are in the packed row.
        # A mismatch indicates metadata corruption; skip rather than fabricate labels.
        if text[vstart:vend] != view_text:
            continue
        for a, b, word in content_word_spans(view_text):
            pos = locate_positions(offsets, vstart + a, vstart + b)
            pos = [p for p in pos if int(attention_mask[p]) == 1]
            if pos:
                groups.append({
                    "segment_index": seg_i,
                    "pair_id": seg.get("pair_id"),
                    "view_kind": seg.get("view_kind"),
                    "candidate_kind": seg.get("candidate_kind"),
                    "word": word,
                    "positions": pos,
                })
    rng = random.Random(int(seed))
    if len(groups) > int(max_focus_groups_per_row):
        # Stable row-local cap to avoid letting dense qwen packed rows dominate the
        # macro update. This preserves source-visible view credit while keeping the
        # objective in the same scale as ordinary WWM.
        groups = rng.sample(groups, int(max_focus_groups_per_row))
    chosen = [g for g in groups if rng.random() < float(focus_prob)]
    if not chosen and groups:
        chosen = [rng.choice(groups)]
    target_tokens = 0
    kind_counts = Counter()
    pair_ids = set()
    for g in chosen:
        kind_counts[str(g.get("candidate_kind", "unknown"))] += 1
        if g.get("pair_id"):
            pair_ids.add(str(g["pair_id"]))
        for p in g["positions"]:
            if int(attention_mask[p]) == 1:
                labels[p] = input_ids[p]
                masked[p] = int(tokenizer.mask_token_id)
                target_tokens += 1
    return masked, labels, {
        "n_candidate_groups": len(groups),
        "n_selected_groups": len(chosen),
        "n_target_tokens": target_tokens,
        "zero_label_row": target_tokens == 0,
        "selected_candidate_kind_counts": dict(kind_counts),
        "selected_pair_ids": sorted(pair_ids),
    }


def prepare_macro(rows: List[Dict[str, Any]], tokenizer, seq_length: int, wgb, objective: str, train_seed: int,
                  mask_prob: float, focus_prob: float, max_focus_groups_per_row: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    examples = []
    stats: Dict[str, Any] = {
        "rows": len(rows),
        "words": 0,
        "ordinary_wwm_rows": 0,
        "qwen_focus_rows": 0,
        "qwen_wwm_rows": 0,
        "ordinary_target_tokens": 0,
        "focus_target_tokens": 0,
        "zero_label_rows": 0,
        "qwen_rows": 0,
        "compact_modified_rows": 0,
        "focus_selected_groups": 0,
        "focus_candidate_groups": 0,
        "focus_selected_candidate_kind_counts": Counter(),
        "focus_selected_pairs": set(),
    }
    for presentation_i, row in enumerate(rows):
        tok = bridge.tokenize_row(row, tokenizer, seq_length, wgb)
        words = int(row.get("words", wc(row.get("text", ""))))
        stats["words"] += words
        is_qwen = row.get("source") == "qwen_pair_packed" and bool(row.get("qwen_pair_segments"))
        if is_qwen:
            stats["qwen_rows"] += 1
        if row.get("compact_modified_pair_ids"):
            stats["compact_modified_rows"] += 1

        if objective == "correspondence_focus" and is_qwen:
            seed = stable_seed("view-focus", int(train_seed), row_key(row))
            masked, labels, mstats = apply_view_focus_row(row, tok, tokenizer, seed, focus_prob, max_focus_groups_per_row)
            component = "focus"
            stats["qwen_focus_rows"] += 1
            stats["focus_target_tokens"] += int(mstats["n_target_tokens"])
            stats["focus_selected_groups"] += int(mstats["n_selected_groups"])
            stats["focus_candidate_groups"] += int(mstats["n_candidate_groups"])
            stats["focus_selected_candidate_kind_counts"].update(mstats.get("selected_candidate_kind_counts") or {})
            stats["focus_selected_pairs"].update(mstats.get("selected_pair_ids") or [])
        else:
            seed = stable_seed("wwm", "ordinary_stream", int(train_seed), row_key(row))
            masked, labels, mstats = bridge.apply_wwm_row(tok["input_ids"], tok["attention_mask"], tok["word_group"], tokenizer, seed, mask_prob)
            component = "ordinary"
            stats["ordinary_target_tokens"] += int(mstats["n_target_tokens"])
            if is_qwen:
                stats["qwen_wwm_rows"] += 1
            else:
                stats["ordinary_wwm_rows"] += 1
        if int((labels != -100).sum().item()) == 0:
            stats["zero_label_rows"] += 1
        examples.append({
            "input_ids": masked,
            "attention_mask": tok["attention_mask"],
            "labels": labels,
            "component": component,
            "row_key": row_key(row),
            "words": words,
            "presentation_i": presentation_i,
            "source": row.get("source", ""),
        })
    stats["focus_selected_candidate_kind_counts"] = dict(stats["focus_selected_candidate_kind_counts"])
    stats["focus_selected_pair_count"] = len(stats["focus_selected_pairs"])
    stats.pop("focus_selected_pairs", None)
    return examples, stats


def save_checkpoint(model, tokenizer, ckpt_dir: pathlib.Path, metadata: Dict[str, Any], private_scale: float) -> None:
    bridge.save_checkpoint(model, tokenizer, ckpt_dir, metadata, private_scale)


def train_one(objective: str, rows: List[Dict[str, Any]], prefix_info: Dict[str, Any], args: argparse.Namespace, device: torch.device,
              tokenizer, parent_surface_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    out_dir = pathlib.Path(args.out_dir) / objective
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(stable_seed("real-stream", objective, int(args.train_seed)))

    model, missing, unexpected = bridge.load_model(device, args.private_scale)
    ident = bridge.model_identity(model)
    if ident.get("class") != "FrozenSlowPrivateDebertaV2ForMaskedLM" or int(ident.get("private_adapter_params", 0)) != 995584:
        raise RuntimeError(f"Bad model identity: {ident}")
    opt, opt_info = bridge.freeze_to_private_optimizer(model, float(args.lr), float(args.weight_decay))
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    wgb = bridge.WordGroupBuilder(tokenizer)
    config = {
        "status": "REAL_STREAM_CONFIG",
        "created_utc": now(),
        "objective": objective,
        "tail_jsonl": rel(args.tail_jsonl),
        "prefix_info": prefix_info,
        "parent_path": rel(bridge.PARENT_PATH),
        "train_seed": int(args.train_seed),
        "max_updates": int(args.max_updates),
        "words_per_update": int(args.words_per_update),
        "schedule_total": int(args.schedule_total),
        "schedule_offset": int(args.schedule_offset),
        "warmup": int(args.warmup),
        "lr_peak": float(args.lr),
        "mask_prob": float(args.mask_prob),
        "focus_prob": float(args.focus_prob),
        "max_focus_groups_per_row": int(args.max_focus_groups_per_row),
        "seq_length": int(args.seq_length),
        "objective_definition": "WWM everywhere" if objective == "inherited_wwm" else "WWM on non-qwen ordinary rows; qwen_pair_packed rows use source-visible targets on content words in second-view segments",
        "model_identity": ident,
        "optimizer": {"trainable_tensors": opt_info["trainable_tensors"], "trainable_params": opt_info["trainable_params"]},
    }
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "train_start", "objective": objective, "rows": len(rows), "prefix_words": prefix_info.get("prefix_words"), "device": str(device), "identity": ident}, ensure_ascii=False), flush=True)

    cursor = 0
    cum_words = 0
    logs: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    t0 = time.time()
    model.train()
    for update_i in range(int(args.max_updates)):
        macro_rows: List[Dict[str, Any]] = []
        words = 0
        while cursor < len(rows) and words < int(args.words_per_update):
            r = rows[cursor]
            macro_rows.append(r)
            words += int(r.get("words", wc(r.get("text", ""))))
            cursor += 1
        if not macro_rows:
            print(json.dumps({"event": "data_exhausted", "objective": objective, "update": update_i}), flush=True)
            break
        schedule_idx = int(args.schedule_offset) + update_i
        lr = lr_at_update(schedule_idx, int(args.schedule_total), int(args.warmup), float(args.lr))
        for pg in opt.param_groups:
            pg["lr"] = lr
        examples, prep = prepare_macro(macro_rows, tokenizer, int(args.seq_length), wgb, objective, int(args.train_seed), float(args.mask_prob), float(args.focus_prob), int(args.max_focus_groups_per_row))
        n_targets = int(prep["ordinary_target_tokens"]) + int(prep["focus_target_tokens"])
        if n_targets <= 0:
            raise RuntimeError(f"No targets at update {update_i}: {prep}")
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(examples), int(args.micro_batch)):
            mb = examples[start:start + int(args.micro_batch)]
            input_ids = torch.stack([x["input_ids"] for x in mb]).to(device)
            attention_mask = torch.stack([x["attention_mask"] for x in mb]).to(device)
            labels = torch.stack([x["labels"] for x in mb]).to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            vocab = out.logits.shape[-1]
            ce = F.cross_entropy(out.logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100, reduction="none").view_as(labels)
            mask = labels != -100
            mb_loss_sum = ce[mask].sum() if mask.any() else torch.tensor(0.0, device=device)
            if mask.any():
                # Macro-update normalization: each microbatch contributes its token-loss
                # sum divided by the full macro target count, matching research's corrected
                # denominator style and avoiding microbatch-mean bias.
                loss = mb_loss_sum / float(n_targets)
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"bad loss update {update_i} mb {start}")
                loss.backward()
                loss_sum += float(mb_loss_sum.detach().cpu())
                seen += int(mask.sum().detach().cpu())
            del input_ids, attention_mask, labels, out, ce, mask, mb_loss_sum
        if seen != n_targets:
            raise RuntimeError(f"target count mismatch update {update_i}: prep={n_targets} seen={seen}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.max_grad_norm)).detach().cpu())
        opt.step()
        cum_words += words
        log = {
            "update": update_i + 1,
            "schedule_idx": schedule_idx,
            "lr": lr,
            "rows": len(macro_rows),
            "words": words,
            "cum_words": cum_words,
            "targets": n_targets,
            "ordinary_target_tokens": int(prep["ordinary_target_tokens"]),
            "focus_target_tokens": int(prep["focus_target_tokens"]),
            "focus_target_fraction": float(prep["focus_target_tokens"] / max(1, n_targets)),
            "qwen_rows": int(prep["qwen_rows"]),
            "qwen_focus_rows": int(prep["qwen_focus_rows"]),
            "qwen_wwm_rows": int(prep["qwen_wwm_rows"]),
            "ordinary_wwm_rows": int(prep["ordinary_wwm_rows"]),
            "compact_modified_rows": int(prep["compact_modified_rows"]),
            "zero_label_rows": int(prep["zero_label_rows"]),
            "focus_selected_groups": int(prep["focus_selected_groups"]),
            "focus_candidate_groups": int(prep["focus_candidate_groups"]),
            "focus_selected_pair_count": int(prep["focus_selected_pair_count"]),
            "focus_selected_candidate_kind_counts": prep.get("focus_selected_candidate_kind_counts", {}),
            "loss": loss_sum / max(1, n_targets),
            "grad_norm_preclip": grad_norm,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        logs.append(log)
        if update_i == 0 or (update_i + 1) % int(args.log_every) == 0:
            print(json.dumps({"event": "update", "objective": objective, **log}, ensure_ascii=False), flush=True)
        if ((update_i + 1) % int(args.checkpoint_every) == 0 or update_i + 1 == int(args.max_updates) or cursor >= len(rows)):
            ckpt_name = f"update_{update_i + 1:04d}"
            ckpt_dir = out_dir / "checkpoints" / ckpt_name
            metadata = {
                "objective": objective,
                "update": update_i + 1,
                "cum_words": cum_words,
                "train_seed": int(args.train_seed),
                "prefix_info": prefix_info,
                "final_update": log,
                "model_identity": ident,
                "private_scale": float(args.private_scale),
            }
            save_checkpoint(model, tokenizer, ckpt_dir, metadata, float(args.private_scale))
            checkpoints.append({"update": update_i + 1, "path": rel(ckpt_dir), "loss": log["loss"]})
            print(json.dumps({"event": "checkpoint", "objective": objective, "update": update_i + 1, "path": rel(ckpt_dir)}, ensure_ascii=False), flush=True)
        if cursor >= len(rows):
            print(json.dumps({"event": "prefix_exhausted", "objective": objective, "update": update_i + 1}, ensure_ascii=False), flush=True)
            break
    write_jsonl(out_dir / "update_log.jsonl", logs)
    summary = {
        "status": "REAL_STREAM_TRAIN_DONE",
        "objective": objective,
        "out_dir": rel(out_dir),
        "tail_jsonl": rel(args.tail_jsonl),
        "prefix_info": prefix_info,
        "completed_updates": len(logs),
        "total_words_consumed": cum_words,
        "total_targets": sum(int(x["targets"]) for x in logs),
        "total_focus_targets": sum(int(x["focus_target_tokens"]) for x in logs),
        "total_ordinary_targets": sum(int(x["ordinary_target_tokens"]) for x in logs),
        "mean_focus_target_fraction": sum(float(x["focus_target_fraction"]) for x in logs) / max(1, len(logs)),
        "final_update": logs[-1] if logs else None,
        "checkpoints": checkpoints,
        "elapsed_sec": round(time.time() - t0, 1),
        "model_identity": ident,
        "scientific_status": "early real-stream policy test; not a BabyLM endpoint; compare common-target and Cheap7 before scaling",
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=DEFAULT_TAIL)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--objectives", nargs="+", default=["inherited_wwm", "correspondence_focus"], choices=["inherited_wwm", "correspondence_focus"])
    ap.add_argument("--max-updates", type=int, default=80)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--schedule-total", type=int, default=455)
    ap.add_argument("--schedule-offset", type=int, default=101)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--focus-prob", type=float, default=0.35)
    ap.add_argument("--max-focus-groups-per-row", type=int, default=16)
    ap.add_argument("--checkpoint-every", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--train-seed", type=int, default=62062)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str((args.out_dir / "hf_cache").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((args.out_dir / "hf_cache/transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((args.out_dir / "hf_cache/modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    (args.out_dir / "hf_cache/modules").mkdir(parents=True, exist_ok=True)

    rows, prefix_info = load_prefix(args.tail_jsonl, int(args.max_updates), int(args.words_per_update))
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    plan = {
        "status": "REAL_STREAM_PLAN",
        "created_utc": now(),
        "tail_jsonl": rel(args.tail_jsonl),
        "out_dir": rel(args.out_dir),
        "objectives": list(args.objectives),
        "prefix_info": prefix_info,
        "objective_comparison": {
            "inherited_wwm": "row-keyed 15% WWM on all rows, including compacted qwen rows",
            "correspondence_focus": "same packed rows; WWM on non-qwen rows; source-visible view-focused targets on qwen second views",
        },
        "policy_scope": "structural nonexact compaction candidate; not semantically certified faithful; exact source returns and demonstrated failures excluded",
        "dry_run": bool(args.dry_run),
    }
    (args.out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        # Also prepare one macro for each objective to verify target availability without loading the model.
        wgb = bridge.WordGroupBuilder(tokenizer)
        macro_rows: List[Dict[str, Any]] = []
        words = 0
        for r in rows:
            if words >= int(args.words_per_update):
                break
            macro_rows.append(r)
            words += int(r.get("words", wc(r.get("text", ""))))
        prep_preview = {}
        for objective in args.objectives:
            _ex, st = prepare_macro(macro_rows, tokenizer, int(args.seq_length), wgb, objective, int(args.train_seed), float(args.mask_prob), float(args.focus_prob), int(args.max_focus_groups_per_row))
            prep_preview[objective] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in st.items()}
        (args.out_dir / "dry_run_prep_preview.json").write_text(json.dumps(prep_preview, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "REAL_STREAM_DRY_RUN_DONE", "prep_preview": prep_preview}, indent=2, ensure_ascii=False), flush=True)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    summaries = []
    for objective in args.objectives:
        summaries.append(train_one(objective, rows, prefix_info, args, device, tokenizer))
    final = {
        "status": "REAL_STREAM_COMPARISON_DONE",
        "created_utc": now(),
        "plan": rel(args.out_dir / "plan.json"),
        "summaries": summaries,
        "scientific_status": "training complete; evaluate checkpoints with common-target probe and Cheap7 before interpreting practical value",
    }
    (args.out_dir / "comparison_train_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
