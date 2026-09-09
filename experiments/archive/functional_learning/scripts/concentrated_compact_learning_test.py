#!/usr/bin/env python3
"""research: concentrated learner test for reviewed selective compact views.

This is a bounded diagnostic, not a legal full-tail BabyLM endpoint.  It asks whether
individually reviewed transformed second views teach the same source-supported content
as the inherited rewrites under paired initialization and training conditions.

Primary comparison on the same reviewed faithful source IDs:
  current_equal_epoch   : source + inherited current rewrite, E epochs
  compact_equal_epoch   : source + reviewed compact rewrite, E epochs
  compact_wordmatched   : source + reviewed compact rewrite, enough epochs to match
                          the current arm's charged source+view words

Training uses the coherent86 parent, trusted private-adapter loading, frozen non-private
parameters, and labels only on selected content words inside the second view while the
original source remains visible.  Evaluation masks matched content words in both the
inherited current view and the compact view, with and without the original source
prefix.  Thus the result separates surface-specific acquisition, cross-view transfer,
source-conditioned help, shorter expression, and recurrence/word-budget effects.
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
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPTS))
import corrected_bridge_trainer as bridge  # noqa: E402

DEFAULT_LABELS = _public_path('experiments/archive/functional_learning/data/selective_reviewed_labels/selective_semantic_labels.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/concentrated_compact_learning')

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "to", "of", "in", "on", "for", "with", "as", "at", "by", "from", "into", "about", "over", "under",
    "after", "before", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
    "have", "has", "had", "will", "would", "can", "could", "may", "might", "must", "should", "shall",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "their", "our", "not", "no", "so", "there", "here", "what", "which", "who", "when", "where", "why", "how",
    "current", "state", "information", "according", "context", "also", "than", "into", "onto", "between",
}


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
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


def stable_seed(*parts: Any) -> int:
    txt = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return int.from_bytes(hashlib.blake2b(txt.encode("utf-8"), digest_size=8).digest(), "little") & ((1 << 63) - 1)


def wc(text: str) -> int:
    return len((text or "").strip().split())


def content_word_spans(text: str) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    for m in WORD_RE.finditer(text or ""):
        w = m.group(0)
        wl = w.lower().strip("'\u2019")
        if len(wl) < 4 or wl in STOP or wl.isdigit():
            continue
        out.append((m.start(), m.end(), w))
    return out


def locate_positions(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> List[int]:
    pos = []
    for i, (a, b) in enumerate(offsets):
        a = int(a); b = int(b)
        if b <= a:
            continue
        if a < end and b > start:
            pos.append(i)
    return pos


def tokenize_eval_view(tokenizer, source: str, view: str, with_source: bool, max_length: int) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    if with_source:
        prefix = source.strip() + " "
        text = prefix + view.strip()
        view_offset = len(prefix)
    else:
        text = view.strip()
        view_offset = 0
    enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=int(max_length), return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    att = [1] * len(ids)
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    # Offsets are relative to the full text, so view-local spans need view_offset added by caller.
    return ids, att, offsets


def build_examples(records: List[Dict[str, Any]], tokenizer, max_length: int, view_kind: str,
                   max_train_targets_per_row: int, max_eval_targets_per_view: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    skips = Counter()
    for r in records:
        source = str(r.get("original", "")).strip()
        current = str(r.get("current_rewrite", "")).strip()
        compact = str(r.get("compact_rewrite", "")).strip()
        view = current if view_kind == "current" else compact
        if not source or not view or view == "KEEP_CURRENT":
            skips["empty_or_keep_current"] += 1
            continue
        prefix = source + " "
        full_text = prefix + view
        view_offset = len(prefix)
        enc = tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=int(max_length), return_offsets_mapping=True)
        ids = [int(x) for x in enc["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        if len(ids) >= int(max_length) and enc.get("overflowing_tokens"):
            skips["truncated"] += 1
        train_groups = []
        eval_groups = []
        spans = content_word_spans(view)
        # Training target candidates: content words inside the second view.
        for a, b, word in spans:
            pos = locate_positions(offsets, view_offset + a, view_offset + b)
            if not pos:
                continue
            train_groups.append({"word": word, "view_start": a, "view_end": b, "positions": pos})
        if not train_groups:
            skips["no_train_targets"] += 1
            continue
        # Deterministic subset for evaluation to keep the probe bounded and comparable.
        rng = random.Random(stable_seed("eval-targets", r["pair_id"], view_kind))
        candidates = list(train_groups)
        if len(candidates) > int(max_eval_targets_per_view):
            candidates = rng.sample(candidates, int(max_eval_targets_per_view))
            candidates.sort(key=lambda x: x["view_start"])
        eval_groups = candidates
        if len(train_groups) > int(max_train_targets_per_row):
            train_groups = rng.sample(train_groups, int(max_train_targets_per_row))
            train_groups.sort(key=lambda x: x["view_start"])
        examples.append({
            "pair_id": r["pair_id"],
            "label": r.get("label"),
            "source_corpus": r.get("source_corpus", r.get("source", "")),
            "source_text": source,
            "current_rewrite": current,
            "compact_rewrite": compact,
            "view_kind": view_kind,
            "view_text": view,
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "train_groups": train_groups,
            "eval_groups": eval_groups,
            "source_words": wc(source),
            "view_words": wc(view),
            "row_words": wc(source) + wc(view),
            "current_words": wc(current),
            "compact_words": wc(compact),
        })
    stats = {
        "view_kind": view_kind,
        "n_examples": len(examples),
        "skips": dict(skips),
        "source_words_per_epoch": sum(x["source_words"] for x in examples),
        "view_words_per_epoch": sum(x["view_words"] for x in examples),
        "row_words_per_epoch": sum(x["row_words"] for x in examples),
        "target_groups_per_epoch": sum(len(x["train_groups"]) for x in examples),
    }
    return examples, stats


def query_gpu_memory() -> List[Dict[str, Any]]:
    import subprocess
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 4:
            rows.append({"index": int(parts[0]), "name": parts[1], "memory_used_mb": int(parts[2]), "memory_total_mb": int(parts[3])})
    return rows


def wait_for_gpu(max_used_mb: int, interval_sec: float, max_wait_sec: float, out_dir: pathlib.Path) -> int:
    start = time.time(); attempt = 0
    log_path = out_dir / "gpu_wait_log.jsonl"
    while True:
        attempt += 1
        rows = query_gpu_memory()
        idle = [r for r in rows if r["memory_used_mb"] <= int(max_used_mb)]
        idle.sort(key=lambda r: (r["memory_used_mb"], r["index"]))
        selected = idle[0] if idle else None
        event = {"event": "gpu_wait_check", "attempt": attempt, "elapsed_sec": round(time.time() - start, 1), "selected": selected, "gpus": rows}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if selected:
            return int(selected["index"])
        if time.time() - start > float(max_wait_sec):
            raise TimeoutError(f"No idle GPU after {max_wait_sec}s")
        time.sleep(float(interval_sec))


def collate_train(batch: List[Dict[str, Any]], tokenizer, epoch: int, schedule_key: str, seed: int,
                  mask_prob: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    max_len = max(len(x["input_ids"]) for x in batch)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    target_groups = 0
    target_tokens = 0
    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        ids[i, :L] = torch.tensor(ex["input_ids"], dtype=torch.long)
        att[i, :L] = 1
        groups = list(ex["train_groups"])
        rng = random.Random(stable_seed("train-mask", seed, schedule_key, epoch, ex["pair_id"], ex["view_kind"]))
        chosen = [g for g in groups if rng.random() < float(mask_prob)]
        if not chosen and groups:
            chosen = [rng.choice(groups)]
        for g in chosen:
            target_groups += 1
            for p in g["positions"]:
                if p < L:
                    labels[i, p] = ids[i, p]
                    ids[i, p] = int(tokenizer.mask_token_id)
                    target_tokens += 1
    return ids.to(device), att.to(device), labels.to(device), {"target_groups": target_groups, "target_tokens": target_tokens}


def finite_mean(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def score_mask_tasks(model, tokenizer, tasks: List[Dict[str, Any]], device: torch.device, batch_size: int) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    mask_id = int(tokenizer.mask_token_id)
    for start in range(0, len(tasks), int(batch_size)):
        batch = tasks[start:start + int(batch_size)]
        max_len = max(len(t["input_ids"]) for t in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        att = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        orig = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        for i, t in enumerate(batch):
            L = len(t["input_ids"])
            arr = torch.tensor(t["input_ids"], dtype=torch.long, device=device)
            ids[i, :L] = arr
            orig[i, :L] = arr
            att[i, :L] = 1
            for p in t["positions"]:
                if p < L:
                    ids[i, p] = mask_id
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        for i, t in enumerate(batch):
            vals = []
            for p in t["positions"]:
                if p >= logits.shape[1]:
                    continue
                lp = torch.log_softmax(logits[i, p], dim=-1)
                vals.append(-float(lp[int(orig[i, p])].detach().cpu()))
            rr = {k: v for k, v in t.items() if k not in {"input_ids", "positions"}}
            rr["nll"] = sum(vals) / len(vals) if vals else float("nan")
            rr["n_tokens"] = len(vals)
            out_rows.append(rr)
    return out_rows


def build_eval_tasks(records: List[Dict[str, Any]], tokenizer, max_length: int, max_targets_per_view: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    skips = Counter()
    for r in records:
        source = str(r.get("original", "")).strip()
        for view_kind, view in [("current", str(r.get("current_rewrite", "")).strip()), ("compact", str(r.get("compact_rewrite", "")).strip())]:
            if not source or not view or view == "KEEP_CURRENT":
                skips[f"{view_kind}_empty_or_keep"] += 1
                continue
            groups = content_word_spans(view)
            rng = random.Random(stable_seed("eval-build", r["pair_id"], view_kind))
            if len(groups) > int(max_targets_per_view):
                groups = rng.sample(groups, int(max_targets_per_view))
                groups.sort(key=lambda x: x[0])
            for with_source in [True, False]:
                if with_source:
                    prefix = source + " "
                    text = prefix + view
                    view_offset = len(prefix)
                    condition = "with_source"
                else:
                    text = view
                    view_offset = 0
                    condition = "view_only"
                enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=int(max_length), return_offsets_mapping=True)
                ids = [int(x) for x in enc["input_ids"]]
                offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
                for gi, (a, b, word) in enumerate(groups):
                    pos = locate_positions(offsets, view_offset + a, view_offset + b)
                    if not pos:
                        skips[f"{view_kind}_{condition}_no_pos"] += 1
                        continue
                    tasks.append({
                        "pair_id": r["pair_id"],
                        "semantic_label": r.get("label"),
                        "eval_view_kind": view_kind,
                        "condition": condition,
                        "target_index": gi,
                        "target_word": word,
                        "input_ids": ids,
                        "positions": pos,
                    })
    return tasks, {"n_tasks": len(tasks), "skips": dict(skips)}


def summarize_eval(score_rows: List[Dict[str, Any]], parent_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    parent_by_key = {}
    if parent_rows is not None:
        for r in parent_rows:
            key = (r["pair_id"], r["eval_view_kind"], r["condition"], r["target_index"])
            parent_by_key[key] = r
    by_view_cond: Dict[str, Dict[str, Any]] = {}
    for view in sorted(set(r["eval_view_kind"] for r in score_rows)):
        for cond in ["with_source", "view_only"]:
            g = [r for r in score_rows if r["eval_view_kind"] == view and r["condition"] == cond]
            if not g:
                continue
            by_view_cond[f"{view}/{cond}"] = {
                "n_targets": len(g),
                "n_pairs": len(set(r["pair_id"] for r in g)),
                "mean_nll": finite_mean(r["nll"] for r in g),
                "median_nll": statistics.median([r["nll"] for r in g if math.isfinite(r["nll"])]) if g else None,
                "mean_delta_nll_vs_parent": finite_mean(r["nll"] - parent_by_key.get((r["pair_id"], r["eval_view_kind"], r["condition"], r["target_index"]), {"nll": r["nll"]})["nll"] for r in g) if parent_rows is not None else None,
            }
    # Pair/target source-help: view_only - with_source.
    paired: Dict[Tuple[str, str, int], Dict[str, float]] = defaultdict(dict)
    for r in score_rows:
        paired[(r["pair_id"], r["eval_view_kind"], int(r["target_index"]))][r["condition"]] = float(r["nll"])
    source_help = []
    for (pid, view, ti), vals in paired.items():
        if "with_source" in vals and "view_only" in vals:
            source_help.append({"pair_id": pid, "eval_view_kind": view, "target_index": ti, "source_help": vals["view_only"] - vals["with_source"]})
    by_view_help = {}
    for view in sorted(set(r["eval_view_kind"] for r in source_help)):
        g = [r for r in source_help if r["eval_view_kind"] == view]
        by_view_help[view] = {
            "n_targets": len(g),
            "n_pairs": len(set(r["pair_id"] for r in g)),
            "mean_source_help": finite_mean(r["source_help"] for r in g),
            "median_source_help": statistics.median([r["source_help"] for r in g if math.isfinite(r["source_help"])]) if g else None,
        }
    return {"by_view_condition": by_view_cond, "by_view_source_help": by_view_help}


def train_one_arm(arm: Dict[str, Any], examples: List[Dict[str, Any]], eval_tasks: List[Dict[str, Any]], parent_scores: List[Dict[str, Any]],
                  tokenizer, device: torch.device, args: argparse.Namespace, out_dir: pathlib.Path) -> Dict[str, Any]:
    print(json.dumps({"event": "arm_load_model", "arm": arm["name"], "epochs": arm["epochs"]}), flush=True)
    model, missing, unexpected = bridge.load_model(device, private_scale=float(args.private_scale))
    ident = bridge.model_identity(model)
    if ident.get("class") != "FrozenSlowPrivateDebertaV2ForMaskedLM" or int(ident.get("private_adapter_params", 0)) != 995584:
        raise RuntimeError(f"bad model identity: {ident}")
    opt, opt_info = bridge.freeze_to_private_optimizer(model, float(args.lr), float(args.weight_decay))
    model.train()
    logs = []
    schedule_key = str(arm.get("schedule_key", arm["name"]))
    rng = random.Random(stable_seed("arm-order", int(args.seed), schedule_key))
    t0 = time.time()
    for epoch in range(int(arm["epochs"])):
        order = list(range(len(examples)))
        rng.shuffle(order)
        epoch_loss_sum = 0.0
        epoch_batches = 0
        epoch_targets = 0
        for start in range(0, len(order), int(args.batch_size)):
            batch = [examples[i] for i in order[start:start + int(args.batch_size)]]
            ids, att, labels, st = collate_train(batch, tokenizer, epoch, schedule_key, int(args.seed), float(args.mask_prob), device)
            if int((labels != -100).sum().item()) == 0:
                continue
            out = model(input_ids=ids, attention_mask=att)
            vocab = out.logits.shape[-1]
            loss = F.cross_entropy(out.logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100, reduction="mean")
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"bad loss in {arm['name']} epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.max_grad_norm)).detach().cpu())
            opt.step()
            epoch_loss_sum += float(loss.detach().cpu())
            epoch_batches += 1
            epoch_targets += int(st["target_tokens"])
            del ids, att, labels, out, loss
        log = {
            "epoch": epoch + 1,
            "mean_loss": epoch_loss_sum / max(1, epoch_batches),
            "batches": epoch_batches,
            "target_tokens": epoch_targets,
            "charged_row_words_cum": (epoch + 1) * int(arm["row_words_per_epoch"]),
            "view_words_cum": (epoch + 1) * int(arm["view_words_per_epoch"]),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        logs.append(log)
        if epoch == 0 or (epoch + 1) % int(args.log_every) == 0 or epoch + 1 == int(arm["epochs"]):
            print(json.dumps({"event": "train_epoch", "arm": arm["name"], **log}), flush=True)
    # Score and save.
    model.eval()
    scores = score_mask_tasks(model, tokenizer, eval_tasks, device, int(args.eval_batch_size))
    arm_dir = out_dir / arm["name"]
    arm_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(arm_dir / "eval_scores.jsonl", scores)
    write_jsonl(arm_dir / "train_log.jsonl", logs)
    eval_summary = summarize_eval(scores, parent_scores)
    final_ckpt = arm_dir / "checkpoint_final"
    bridge.save_checkpoint(model, tokenizer, final_ckpt, {"step": "concentrated_compact_learning", "arm": arm, "model_identity": ident}, float(args.private_scale))
    summary = {
        "arm": arm,
        "model_identity": ident,
        "optimizer": opt_info,
        "completed_epochs": int(arm["epochs"]),
        "final_train_log": logs[-1] if logs else None,
        "eval_summary": eval_summary,
        "checkpoint_final": rel(final_ckpt),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (arm_dir / "arm_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "arm_done", "arm": arm["name"], "summary": rel(arm_dir / "arm_summary.json")}), flush=True)
    return summary


def build_plan(records: List[Dict[str, Any]], tokenizer, args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    labels = Counter(r.get("label") for r in records)
    faithful = [r for r in records if r.get("label") == "faithful_shortening"]
    if int(args.max_pairs) > 0:
        faithful = faithful[: int(args.max_pairs)]
    current_ex, current_stats = build_examples(faithful, tokenizer, int(args.max_length), "current", int(args.max_train_targets_per_row), int(args.max_eval_targets_per_view))
    compact_ex, compact_stats = build_examples(faithful, tokenizer, int(args.max_length), "compact", int(args.max_train_targets_per_row), int(args.max_eval_targets_per_view))
    # Keep only pair IDs valid in both views.
    common = sorted(set(x["pair_id"] for x in current_ex) & set(x["pair_id"] for x in compact_ex))
    current_ex = [x for x in current_ex if x["pair_id"] in common]
    compact_ex = [x for x in compact_ex if x["pair_id"] in common]
    faithful_common_records = [r for r in faithful if r["pair_id"] in set(common)]
    eval_tasks, eval_stats = build_eval_tasks(faithful_common_records, tokenizer, int(args.max_length), int(args.max_eval_targets_per_view))
    current_row_words = sum(x["row_words"] for x in current_ex)
    compact_row_words = sum(x["row_words"] for x in compact_ex)
    compact_wordmatched_epochs = int(math.ceil(int(args.base_epochs) * current_row_words / max(1, compact_row_words)))
    arms = [
        {"name": "current_equal_epoch", "train_view": "current", "schedule_key": "current", "epochs": int(args.base_epochs), **{k: current_stats[k] for k in ["source_words_per_epoch", "view_words_per_epoch", "row_words_per_epoch", "target_groups_per_epoch"]}},
        {"name": "compact_equal_epoch", "train_view": "compact", "schedule_key": "compact", "epochs": int(args.base_epochs), **{k: compact_stats[k] for k in ["source_words_per_epoch", "view_words_per_epoch", "row_words_per_epoch", "target_groups_per_epoch"]}},
        {"name": "compact_wordmatched", "train_view": "compact", "schedule_key": "compact", "epochs": compact_wordmatched_epochs, **{k: compact_stats[k] for k in ["source_words_per_epoch", "view_words_per_epoch", "row_words_per_epoch", "target_groups_per_epoch"]}},
    ]
    plan = {
        "status": "CONCENTRATED_COMPACT_PLAN",
        "created_utc": now(),
        "scientific_purpose": "Bounded learner test on individually reviewed faithful shortening rows; compares same source IDs under inherited vs transformed second views before full-tail investment.",
        "reviewed_label_counts": dict(labels),
        "n_faithful_reviewed": len(faithful),
        "n_common_train_pairs": len(common),
        "common_pair_ids": common,
        "current_view_stats": current_stats,
        "compact_view_stats": compact_stats,
        "current_row_words_per_epoch_common": current_row_words,
        "compact_row_words_per_epoch_common": compact_row_words,
        "realized_saved_words_per_epoch_common": current_row_words - compact_row_words,
        "view_saved_words_per_epoch_common": sum(x["current_words"] - x["compact_words"] for x in compact_ex),
        "eval_stats": eval_stats,
        "arms": arms,
        "interpretation_boundaries": [
            "This is concentrated learning evidence, not a legal BabyLM endpoint.",
            "Equal-epoch compact improvement tests shorter transformed exposure at identical presentations/update count.",
            "Compact word-matched improvement tests spending the word savings as extra recurrence on the same source IDs, not new support.",
            "Cross-view scores separate exact surface adaptation from source-supported reusable content.",
        ],
    }
    return plan, current_ex, compact_ex, eval_tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=pathlib.Path, default=DEFAULT_LABELS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--base-epochs", type=int, default=80)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--mask-prob", type=float, default=0.35)
    ap.add_argument("--max-train-targets-per-row", type=int, default=16)
    ap.add_argument("--max-eval-targets-per-view", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=57057)
    ap.add_argument("--gpu", default="auto", help="GPU index or auto")
    ap.add_argument("--idle-memory-mb", type=int, default=2500)
    ap.add_argument("--gpu-wait-sec", type=float, default=10800.0)
    ap.add_argument("--gpu-check-interval-sec", type=float, default=60.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str((out_dir / "hf_cache/hf_home").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((out_dir / "hf_cache/transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((out_dir / "hf_cache/modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    (out_dir / "hf_cache/modules").mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.labels)
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    if tokenizer.mask_token_id is None:
        raise RuntimeError("tokenizer has no mask token")
    plan, current_ex, compact_ex, eval_tasks = build_plan(records, tokenizer, args)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(out_dir / "current_training_examples.jsonl", [{k: v for k, v in x.items() if k not in {"input_ids", "attention_mask"}} for x in current_ex])
    write_jsonl(out_dir / "compact_training_examples.jsonl", [{k: v for k, v in x.items() if k not in {"input_ids", "attention_mask"}} for x in compact_ex])
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    if args.gpu == "auto":
        gpu_idx = wait_for_gpu(int(args.idle_memory_mb), float(args.gpu_check_interval_sec), float(args.gpu_wait_sec), out_dir)
    else:
        gpu_idx = int(args.gpu)
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "device_selected", "device": str(device), "gpu": gpu_idx}), flush=True)

    # Parent baseline once.
    parent_model, missing, unexpected = bridge.load_model(device, private_scale=float(args.private_scale))
    ident = bridge.model_identity(parent_model)
    if ident.get("class") != "FrozenSlowPrivateDebertaV2ForMaskedLM" or int(ident.get("private_adapter_params", 0)) != 995584:
        raise RuntimeError(f"bad parent identity: {ident}")
    parent_model.eval()
    parent_scores = score_mask_tasks(parent_model, tokenizer, eval_tasks, device, int(args.eval_batch_size))
    write_jsonl(out_dir / "parent_eval_scores.jsonl", parent_scores)
    parent_summary = summarize_eval(parent_scores, None)
    (out_dir / "parent_eval_summary.json").write_text(json.dumps(parent_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del parent_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    arm_summaries = []
    examples_by_view = {"current": current_ex, "compact": compact_ex}
    for arm in plan["arms"]:
        ex = examples_by_view[arm["train_view"]]
        arm_summary = train_one_arm(arm, ex, eval_tasks, parent_scores, tokenizer, device, args, out_dir)
        arm_summaries.append(arm_summary)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final = {
        "status": "CONCENTRATED_COMPACT_LEARNING_DONE",
        "created_utc": now(),
        "plan": rel(out_dir / "plan.json"),
        "parent_eval_summary": parent_summary,
        "arm_summaries": arm_summaries,
        "scientific_status": "bounded concentrated learner diagnostic; use to decide whether reviewed transformed views can teach before legal full-tail training",
    }
    (out_dir / "summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
