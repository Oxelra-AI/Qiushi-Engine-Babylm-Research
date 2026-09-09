#!/usr/bin/env python3
"""research: fixed-target evidence-availability readout for Qwen pair states.

The dense-mask result showed that clue-suppressed input geometry is sufficient to
produce source-responsive movement, but the same geometry damages wrong-source,
view-only, CDI, and broad language behavior.  This readout holds the target word
fixed while changing only the evidence state around it:

  * pair_correct_source: correct source sentence + current view
  * pair_wrong_source: different source sentence + same current view
  * view_only: current view alone
  * full_row_original: original packed row with all source/view pairs
  * full_row_this_source_erased: same packed row with this source span blanked
  * full_row_all_sources_erased: same packed row with every source span blanked

For each endpoint it records target NLL, target rank, and full-vocabulary
KL(teacher || endpoint) at the same masked target positions, using coherent86
and the sparse-focus endpoint as reference teachers.  The same script can score
held-out post-prefix Qwen rows now and later include clean preservation endpoints
as soon as their checkpoints are available.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import os
import pathlib
import random
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
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import corrected_bridge_trainer as bridge  # noqa: E402
import concentrated_compact_learning_test as s57  # noqa: E402
import qwen_view_surface_probe as qview  # noqa: E402

DEFAULT_TAIL = _public_path('experiments/archive/functional_learning/data/unchanged_pair_segments_tail/reference_tail_unchanged_qwen_segments.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/evidence_availability_readout')
WORDS_PER_UPDATE = 39_533
PREFIX_UPDATES = 80
PREFIX_WORD_LIMIT = WORDS_PER_UPDATE * PREFIX_UPDATES

BASE_MODEL_SPECS: Dict[str, Optional[pathlib.Path]] = {
    "coherent86": None,
    "sparse_focus_seed62064": _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/correspondence_focus_weighted/checkpoints/update_0080'),
    "densemask_sparselabel_seed62064": _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/checkpoints/update_0080'),
    "dense_focus_seed62064": _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/checkpoints/update_0080'),
    "dense_focus_seed62065": _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/checkpoints/update_0080'),
    "pres_lambda1_trainmode_confounded": _public_path('experiments/archive/functional_learning/data/preservation_lambda1p0_seed62064/checkpoints/update_0080'),
    "clean_pres_lambda1_eval_full80": _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_eval_full80/checkpoints/update_0080'),
    "clean_pres_lambda1_train_full80": _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_train_full80/checkpoints/update_0080'),
}

DEFAULT_CONDITIONS = [
    "pair_correct_source",
    "pair_wrong_source",
    "view_only",
    "full_row_original",
    "full_row_this_source_erased",
    "full_row_all_sources_erased",
]


def rel(p: pathlib.Path | str | None) -> str | None:
    if p is None:
        return None
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wc(text: str) -> int:
    return len((text or "").strip().split())


def stable_seed(*parts: Any) -> int:
    h = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def finite_vals(xs: Iterable[Any]) -> List[float]:
    vals: List[float] = []
    for x in xs:
        if x is None:
            continue
        try:
            xf = float(x)
        except Exception:
            continue
        if math.isfinite(xf):
            vals.append(xf)
    return vals


def mean(xs: Iterable[Any]) -> Optional[float]:
    vals = finite_vals(xs)
    return sum(vals) / len(vals) if vals else None


def median(xs: Iterable[Any]) -> Optional[float]:
    vals = finite_vals(xs)
    return statistics.median(vals) if vals else None


def sem(xs: Iterable[Any]) -> Optional[float]:
    vals = finite_vals(xs)
    if len(vals) <= 1:
        return None
    return statistics.stdev(vals) / math.sqrt(len(vals))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def setup_cache(out_dir: pathlib.Path) -> None:
    hf = out_dir / "hf_cache"
    os.environ["HF_HOME"] = str(hf.resolve())
    os.environ["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    os.environ["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for sub in ["hub", "datasets", "transformers", "modules"]:
        (hf / sub).mkdir(parents=True, exist_ok=True)


def locate_positions(offsets: List[Tuple[int, int]], start: int, end: int) -> List[int]:
    pos: List[int] = []
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        if a < end and b > start:
            pos.append(i)
    return pos


def erase_spans_preserve_offsets(text: str, spans: Iterable[Tuple[int, int]]) -> str:
    chars = list(text)
    n = len(chars)
    for s, e in spans:
        ss = max(0, int(s))
        ee = min(n, int(e))
        if ee <= ss:
            continue
        for i in range(ss, ee):
            # Preserve offsets while removing lexical evidence.
            chars[i] = " "
    return "".join(chars)


def context_contains_target(text: str, target_start: int, target_end: int, target_word: str) -> bool:
    if not target_word:
        return False
    chars = list(text)
    for i in range(max(0, target_start), min(len(chars), target_end)):
        chars[i] = " "
    return target_word.lower() in "".join(chars).lower()


def visible_source_contains_target(text: str, source_spans: Iterable[Tuple[int, int]], target_word: str) -> bool:
    if not target_word:
        return False
    t = target_word.lower()
    for s, e in source_spans:
        frag = text[max(0, int(s)):max(0, int(e))]
        if t in frag.lower():
            return True
    return False


def iter_qwen_segments(tail_jsonl: pathlib.Path, scope: str, max_updates: int, words_per_update: int):
    limit = int(max_updates) * int(words_per_update)
    total_words = 0
    row_idx = 0
    with tail_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_words = int(row.get("words", wc(row.get("text", ""))))
            before_prefix = total_words < limit
            include = False
            if scope == "prefix":
                include = before_prefix
            elif scope == "post_prefix":
                include = not before_prefix
            elif scope == "all":
                include = True
            else:
                raise ValueError(f"unknown scope {scope!r}")
            if include and row.get("source") == "qwen_pair_packed" and row.get("qwen_pair_segments"):
                row_text = str(row.get("text", ""))
                all_source_spans = []
                for seg in row.get("qwen_pair_segments") or []:
                    try:
                        all_source_spans.append((int(seg.get("source_start")), int(seg.get("source_end"))))
                    except Exception:
                        pass
                for seg_i, seg in enumerate(row.get("qwen_pair_segments") or []):
                    source = str(seg.get("source_text", "")).strip()
                    view = str(seg.get("view_text", "")).strip()
                    if not source or not view:
                        continue
                    groups = s57.content_word_spans(view)
                    if not groups:
                        continue
                    yield {
                        "row_idx": row_idx,
                        "row_words_before": total_words,
                        "row_words_after": total_words + row_words,
                        "row_words": row_words,
                        "row_text": row_text,
                        "segment_index": seg_i,
                        "pair_id": str(seg.get("pair_id")),
                        "candidate_kind": str(seg.get("candidate_kind", "unknown")),
                        "view_kind": str(seg.get("view_kind", "unknown")),
                        "source_text": source,
                        "view_text": view,
                        "source_start": int(seg.get("source_start")),
                        "source_end": int(seg.get("source_end")),
                        "view_start": int(seg.get("view_start")),
                        "view_end": int(seg.get("view_end")),
                        "source_words": int(seg.get("source_words", wc(source))),
                        "view_words": int(seg.get("view_words", wc(view))),
                        "groups": groups,
                        "all_source_spans": all_source_spans,
                    }
            total_words += row_words
            row_idx += 1


def build_segment_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_records = list(iter_qwen_segments(args.tail_jsonl, args.scope, args.max_updates, args.words_per_update))
    rng = random.Random(int(args.sample_seed))
    sampled = list(all_records)
    if int(args.max_segments) > 0 and len(sampled) > int(args.max_segments):
        sampled = rng.sample(sampled, int(args.max_segments))
        sampled.sort(key=lambda r: (int(r["row_idx"]), int(r["segment_index"]), str(r["pair_id"])))
    return sampled, {
        "scope": args.scope,
        "total_segments_available_in_scope": len(all_records),
        "segments_sampled": len(sampled),
        "max_updates": int(args.max_updates),
        "words_per_update": int(args.words_per_update),
        "prefix_word_limit": int(args.max_updates) * int(args.words_per_update),
        "sample_seed": int(args.sample_seed),
        "row_idx_min": min((int(r["row_idx"]) for r in sampled), default=None),
        "row_idx_max": max((int(r["row_idx"]) for r in sampled), default=None),
        "pair_count_sampled": len(set(str(r["pair_id"]) for r in sampled)),
        "candidate_kind_counts": dict(Counter(str(r.get("candidate_kind")) for r in sampled)),
        "view_kind_counts": dict(Counter(str(r.get("view_kind")) for r in sampled)),
    }


def choose_wrong_source(records: List[Dict[str, Any]], rec: Dict[str, Any], target_word: str, sample_seed: int) -> Tuple[str, str, str]:
    target_l = (target_word or "").lower()
    candidates: List[int] = []
    loose: List[int] = []
    for j, r in enumerate(records):
        if r["pair_id"] == rec["pair_id"]:
            continue
        loose.append(j)
        if target_l and target_l not in r["source_text"].lower() and target_l not in r["view_text"].lower():
            candidates.append(j)
    pool = candidates or loose
    if not pool:
        return "", "", "no_wrong_source_available"
    rng = random.Random(stable_seed(bytes((115, 116, 101, 112, 48, 57, 48, 45, 119, 114, 111, 110, 103, 45, 115, 111, 117, 114, 99, 101)).decode('utf-8'), int(sample_seed), rec["row_idx"], rec["segment_index"], rec["pair_id"], target_word))
    j = pool[rng.randrange(len(pool))]
    reason = "target_absent_wrong_source" if candidates else "fallback_different_pair"
    return records[j]["source_text"], records[j]["pair_id"], reason


def make_task(tokenizer, rec: Dict[str, Any], group: Tuple[int, int, str], target_group_index: int,
              condition: str, wrong_source: str, wrong_pair: str, wrong_reason: str,
              max_length: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    a, b, word = int(group[0]), int(group[1]), str(group[2])
    row_text = str(rec["row_text"])
    source_spans_visible: List[Tuple[int, int]] = []
    source_status = "none"
    source_visible_nominal = False
    other_sources_visible = False
    target_start = 0
    target_end = 0

    if condition == "pair_correct_source":
        prefix = rec["source_text"] + " "
        text = prefix + rec["view_text"]
        target_start = len(prefix) + a
        target_end = len(prefix) + b
        source_spans_visible = [(0, len(rec["source_text"]))]
        source_status = "correct_pair_source"
        source_visible_nominal = True
    elif condition == "pair_wrong_source":
        if not wrong_source:
            return None, "wrong_source_missing"
        prefix = wrong_source + " "
        text = prefix + rec["view_text"]
        target_start = len(prefix) + a
        target_end = len(prefix) + b
        source_spans_visible = [(0, len(wrong_source))]
        source_status = wrong_reason
        source_visible_nominal = True
    elif condition == "view_only":
        text = rec["view_text"]
        target_start = a
        target_end = b
        source_status = "view_only"
    elif condition == "full_row_original":
        text = row_text
        target_start = int(rec["view_start"]) + a
        target_end = int(rec["view_start"]) + b
        source_spans_visible = list(rec.get("all_source_spans") or [])
        source_status = "full_row_original"
        source_visible_nominal = True
        other_sources_visible = True
    elif condition == "full_row_this_source_erased":
        text = erase_spans_preserve_offsets(row_text, [(int(rec["source_start"]), int(rec["source_end"]))])
        target_start = int(rec["view_start"]) + a
        target_end = int(rec["view_start"]) + b
        source_spans_visible = [(s, e) for (s, e) in rec.get("all_source_spans", []) if not (int(s) == int(rec["source_start"]) and int(e) == int(rec["source_end"]))]
        source_status = "this_source_erased_other_sources_visible"
        source_visible_nominal = False
        other_sources_visible = bool(source_spans_visible)
    elif condition == "full_row_all_sources_erased":
        text = erase_spans_preserve_offsets(row_text, rec.get("all_source_spans") or [])
        target_start = int(rec["view_start"]) + a
        target_end = int(rec["view_start"]) + b
        source_status = "all_sources_erased"
        source_visible_nominal = False
        other_sources_visible = False
    else:
        return None, f"unknown_condition:{condition}"

    if target_start < 0 or target_end <= target_start or target_end > len(text):
        return None, "target_char_span_out_of_bounds"
    enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=int(max_length), return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(x), int(y)) for x, y in enc["offset_mapping"]]
    positions = locate_positions(offsets, target_start, target_end)
    if not positions:
        return None, f"{condition}_target_truncated_or_unmapped"
    base_id = f"scope{rec.get('row_idx')}_r{int(rec['row_idx']):07d}_s{int(rec['segment_index']):02d}_g{int(target_group_index):02d}"
    task = {
        "task_id": f"{base_id}_{condition}",
        "base_task_id": base_id,
        "condition": condition,
        "row_idx": int(rec["row_idx"]),
        "row_words_before": int(rec.get("row_words_before", -1)),
        "row_words_after": int(rec.get("row_words_after", -1)),
        "segment_index": int(rec["segment_index"]),
        "pair_id": rec["pair_id"],
        "candidate_kind": rec["candidate_kind"],
        "view_kind": rec["view_kind"],
        "target_group_index": int(target_group_index),
        "target_word": word,
        "target_char_start": int(target_start),
        "target_char_end": int(target_end),
        "input_len": len(ids),
        "positions": positions,
        "n_positions": len(positions),
        "input_ids": ids,
        "source_status": source_status,
        "wrong_pair_id": wrong_pair if condition == "pair_wrong_source" else None,
        "wrong_source_reason": wrong_reason if condition == "pair_wrong_source" else None,
        "source_visible_nominal": bool(source_visible_nominal),
        "other_sources_visible": bool(other_sources_visible),
        "source_region_contains_target_string": bool(visible_source_contains_target(text, source_spans_visible, word)),
        "target_string_in_context_excluding_target": bool(context_contains_target(text, target_start, target_end, word)),
        "text_words": wc(text),
        "text_chars": len(text),
        "text_sha256_16": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "text_preview": text[:220].replace("\n", "\\n"),
    }
    return task, None


def build_tasks(records: List[Dict[str, Any]], tokenizer, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_tasks: List[Dict[str, Any]] = []
    skips = Counter()
    target_stats = Counter()
    conditions = list(args.conditions)
    # Wrong-source pool should cover the full sampled scope, not only the final kept targets.
    for rec in records:
        groups = list(rec["groups"])
        grng = random.Random(stable_seed(bytes((115, 116, 101, 112, 48, 57, 48, 45, 116, 97, 114, 103, 101, 116, 115)).decode('utf-8'), int(args.sample_seed), rec["row_idx"], rec["segment_index"], rec["pair_id"]))
        if len(groups) > int(args.max_targets_per_segment):
            groups = grng.sample(groups, int(args.max_targets_per_segment))
            groups.sort(key=lambda x: int(x[0]))
        for gi, group in enumerate(groups):
            word = str(group[2])
            wrong_source, wrong_pair, wrong_reason = choose_wrong_source(records, rec, word, int(args.sample_seed))
            target_stats["base_targets_seen"] += 1
            for cond in conditions:
                task, err = make_task(tokenizer, rec, group, gi, cond, wrong_source, wrong_pair, wrong_reason, int(args.max_length))
                if task is None:
                    skips[str(err)] += 1
                else:
                    candidate_tasks.append(task)
    by_base: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for t in candidate_tasks:
        by_base[str(t["base_task_id"])][str(t["condition"])] = t
    kept: List[Dict[str, Any]] = []
    dropped_missing = Counter()
    for base_id, conds in by_base.items():
        missing = [c for c in conditions if c not in conds]
        if missing:
            for c in missing:
                dropped_missing[c] += 1
            continue
        for c in conditions:
            kept.append(conds[c])
    kept.sort(key=lambda t: (int(t["row_idx"]), int(t["segment_index"]), int(t["target_group_index"]), conditions.index(t["condition"])))
    return kept, {
        "conditions_required": conditions,
        "candidate_tasks_before_complete_filter": len(candidate_tasks),
        "tasks_after_complete_filter": len(kept),
        "complete_base_targets": len(set(t["base_task_id"] for t in kept)),
        "condition_counts": dict(Counter(t["condition"] for t in kept)),
        "skips_before_complete_filter": dict(skips),
        "base_targets_seen": int(target_stats["base_targets_seen"]),
        "dropped_base_targets_missing_condition_counts": dict(dropped_missing),
        "target_string_in_context_excluding_target_fraction_by_condition": {
            c: mean(1.0 if t.get("target_string_in_context_excluding_target") else 0.0 for t in kept if t["condition"] == c)
            for c in conditions
        },
        "source_region_contains_target_string_fraction_by_condition": {
            c: mean(1.0 if t.get("source_region_contains_target_string") else 0.0 for t in kept if t["condition"] == c)
            for c in conditions
        },
        "mean_input_len_by_condition": {c: mean(t.get("input_len") for t in kept if t["condition"] == c) for c in conditions},
    }


def task_for_json(t: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in t.items() if k not in {"input_ids", "positions"}}


def load_models(model_specs: Dict[str, Optional[pathlib.Path]], device: torch.device, private_scale: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    models: Dict[str, Any] = {}
    infos: Dict[str, Any] = {}
    for name, path in model_specs.items():
        if path is not None and not pathlib.Path(path).exists():
            infos[name] = {"status": "missing_checkpoint", "path": rel(path)}
            continue
        model, info = qview.load_scoring_model(path, device, private_scale)
        models[name] = model
        infos[name] = {"status": "loaded", **info}
        print(json.dumps({"event": "loaded_model", "model": name, "path": rel(path) if path else rel(bridge.PARENT_PATH), "identity": info.get("identity")}, ensure_ascii=False), flush=True)
    return models, infos


def score_all_models(models: Dict[str, Any], tasks: List[Dict[str, Any]], device: torch.device,
                     tokenizer, batch_size: int, kl_temperature: float) -> Dict[str, List[Dict[str, Any]]]:
    if "coherent86" not in models:
        raise RuntimeError("coherent86 reference must be loaded")
    if "sparse_focus_seed62064" not in models:
        raise RuntimeError("sparse_focus_seed62064 reference must be loaded")
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    mask_id = int(tokenizer.mask_token_id)
    out_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in models}
    model_names = list(models.keys())
    for start in range(0, len(tasks), int(batch_size)):
        batch = tasks[start:start + int(batch_size)]
        max_len = max(len(t["input_ids"]) for t in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        att = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        orig = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        flat_b: List[int] = []
        flat_p: List[int] = []
        flat_labels: List[int] = []
        flat_task_local: List[int] = []
        for i, t in enumerate(batch):
            L = len(t["input_ids"])
            arr = torch.tensor(t["input_ids"], dtype=torch.long, device=device)
            ids[i, :L] = arr
            orig[i, :L] = arr
            att[i, :L] = 1
            for p in t["positions"]:
                pp = int(p)
                if pp < L:
                    flat_b.append(i)
                    flat_p.append(pp)
                    flat_labels.append(int(arr[pp].detach().cpu()))
                    flat_task_local.append(i)
                    ids[i, pp] = mask_id
        if not flat_labels:
            continue
        b_ix = torch.tensor(flat_b, dtype=torch.long, device=device)
        p_ix = torch.tensor(flat_p, dtype=torch.long, device=device)
        labels = torch.tensor(flat_labels, dtype=torch.long, device=device)
        logits_by_model: Dict[str, torch.Tensor] = {}
        logp_by_model: Dict[str, torch.Tensor] = {}
        rank_by_model: Dict[str, torch.Tensor] = {}
        nll_by_model: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name in model_names:
                logits_full = models[name](input_ids=ids, attention_mask=att).logits.float()
                logits = logits_full[b_ix, p_ix, :]
                logits_by_model[name] = logits
                lp = F.log_softmax(logits / float(kl_temperature), dim=-1) if float(kl_temperature) != 1.0 else F.log_softmax(logits, dim=-1)
                logp_by_model[name] = lp
                raw_lp = F.log_softmax(logits, dim=-1)
                nll_by_model[name] = -raw_lp.gather(1, labels.view(-1, 1)).squeeze(1)
                lab_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
                rank_by_model[name] = (logits > lab_logits.view(-1, 1)).sum(dim=1).float() + 1.0
                del logits_full
            parent_lp = logp_by_model["coherent86"]
            parent_p = torch.exp(parent_lp)
            sparse_lp = logp_by_model["sparse_focus_seed62064"]
            sparse_p = torch.exp(sparse_lp)
            # Per-position KL(reference || model).  Temperature-scaled KL is reported only
            # as a distribution comparison, not as a training loss.
            kl_parent: Dict[str, torch.Tensor] = {}
            kl_sparse: Dict[str, torch.Tensor] = {}
            for name in model_names:
                klp = (parent_p * (parent_lp - logp_by_model[name])).sum(dim=-1)
                kls = (sparse_p * (sparse_lp - logp_by_model[name])).sum(dim=-1)
                if float(kl_temperature) != 1.0:
                    klp = klp * (float(kl_temperature) ** 2)
                    kls = kls * (float(kl_temperature) ** 2)
                kl_parent[name] = klp
                kl_sparse[name] = kls
        # Aggregate target-subtoken positions back to fixed target words.
        for name in model_names:
            sums: Dict[int, Dict[str, float]] = defaultdict(lambda: {"n": 0.0, "nll": 0.0, "rank": 0.0, "kl_parent": 0.0, "kl_sparse": 0.0})
            nll_cpu = nll_by_model[name].detach().cpu().tolist()
            rank_cpu = rank_by_model[name].detach().cpu().tolist()
            klp_cpu = kl_parent[name].detach().cpu().tolist()
            kls_cpu = kl_sparse[name].detach().cpu().tolist()
            label_cpu = labels.detach().cpu().tolist()
            for j, task_i in enumerate(flat_task_local):
                acc = sums[int(task_i)]
                acc["n"] += 1.0
                acc["nll"] += float(nll_cpu[j])
                acc["rank"] += float(rank_cpu[j])
                acc["kl_parent"] += float(klp_cpu[j])
                acc["kl_sparse"] += float(kls_cpu[j])
            for i, t in enumerate(batch):
                acc = sums.get(i)
                if not acc or acc["n"] <= 0:
                    continue
                base = task_for_json(t)
                denom = acc["n"]
                base.update({
                    "model": name,
                    "n_target_subtokens_scored": int(denom),
                    "nll": acc["nll"] / denom,
                    "rank": acc["rank"] / denom,
                    "kl_coherent86_to_model": acc["kl_parent"] / denom,
                    "kl_sparse_to_model": acc["kl_sparse"] / denom,
                })
                out_rows[name].append(base)
        del ids, att, orig, b_ix, p_ix, labels
        for d in [logits_by_model, logp_by_model, rank_by_model, nll_by_model]:
            d.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps({"event": "batch_scored", "start": start, "end": start + len(batch), "tasks_total": len(tasks)}, ensure_ascii=False), flush=True)
    return out_rows


def summarize_model(rows: List[Dict[str, Any]], parent_rows: List[Dict[str, Any]], sparse_rows: List[Dict[str, Any]], conditions: List[str]) -> Dict[str, Any]:
    parent_by_task = {str(r["task_id"]): r for r in parent_rows}
    sparse_by_task = {str(r["task_id"]): r for r in sparse_rows}
    by_condition: Dict[str, Any] = {}
    for cond in conditions:
        g = [r for r in rows if r["condition"] == cond]
        by_condition[cond] = {
            "n_tasks": len(g),
            "n_base_targets": len(set(str(r["base_task_id"]) for r in g)),
            "mean_nll": mean(r.get("nll") for r in g),
            "median_nll": median(r.get("nll") for r in g),
            "mean_rank": mean(r.get("rank") for r in g),
            "median_rank": median(r.get("rank") for r in g),
            "mean_kl_coherent86_to_model": mean(r.get("kl_coherent86_to_model") for r in g),
            "mean_kl_sparse_to_model": mean(r.get("kl_sparse_to_model") for r in g),
            "mean_delta_nll_vs_coherent86": mean(float(r["nll"]) - float(parent_by_task[r["task_id"]]["nll"]) for r in g if r["task_id"] in parent_by_task),
            "mean_delta_rank_vs_coherent86": mean(float(r["rank"]) - float(parent_by_task[r["task_id"]]["rank"]) for r in g if r["task_id"] in parent_by_task),
            "mean_delta_nll_vs_sparse": mean(float(r["nll"]) - float(sparse_by_task[r["task_id"]]["nll"]) for r in g if r["task_id"] in sparse_by_task),
            "mean_delta_rank_vs_sparse": mean(float(r["rank"]) - float(sparse_by_task[r["task_id"]]["rank"]) for r in g if r["task_id"] in sparse_by_task),
            "target_string_in_context_fraction": mean(1.0 if r.get("target_string_in_context_excluding_target") else 0.0 for r in g),
            "source_region_contains_target_fraction": mean(1.0 if r.get("source_region_contains_target_string") else 0.0 for r in g),
            "mean_input_len": mean(r.get("input_len") for r in g),
        }
    def by_base(rs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for r in rs:
            out[str(r["base_task_id"])][str(r["condition"])] = r
        return out
    bb = by_base(rows)
    pbb = by_base(parent_rows)
    sbb = by_base(sparse_rows)
    interaction_rows: List[Dict[str, Any]] = []
    for base_id, conds in bb.items():
        if not all(c in conds for c in conditions):
            continue
        rr: Dict[str, Any] = {
            "base_task_id": base_id,
            "pair_id": conds[conditions[0]]["pair_id"],
            "target_word": conds[conditions[0]]["target_word"],
            "row_idx": conds[conditions[0]]["row_idx"],
            "segment_index": conds[conditions[0]]["segment_index"],
        }
        def val(c: str, key: str) -> float:
            return float(conds[c][key])
        def ref_val(ref: Dict[str, Dict[str, Dict[str, Any]]], c: str, key: str) -> Optional[float]:
            if base_id in ref and c in ref[base_id]:
                return float(ref[base_id][c][key])
            return None
        if all(c in conds for c in ["pair_correct_source", "pair_wrong_source", "view_only"]):
            rr["specific_source_penalty_nll_wrong_minus_correct"] = val("pair_wrong_source", "nll") - val("pair_correct_source", "nll")
            rr["view_absence_penalty_nll_view_minus_correct"] = val("view_only", "nll") - val("pair_correct_source", "nll")
            rr["generic_source_penalty_nll_view_minus_wrong"] = val("view_only", "nll") - val("pair_wrong_source", "nll")
            rr["specific_source_penalty_rank_wrong_minus_correct"] = val("pair_wrong_source", "rank") - val("pair_correct_source", "rank")
            rr["view_absence_penalty_rank_view_minus_correct"] = val("view_only", "rank") - val("pair_correct_source", "rank")
        if all(c in conds for c in ["full_row_original", "full_row_this_source_erased", "full_row_all_sources_erased"]):
            rr["this_source_erasure_cost_nll"] = val("full_row_this_source_erased", "nll") - val("full_row_original", "nll")
            rr["all_sources_erasure_cost_nll"] = val("full_row_all_sources_erased", "nll") - val("full_row_original", "nll")
            rr["this_source_erasure_cost_rank"] = val("full_row_this_source_erased", "rank") - val("full_row_original", "rank")
            rr["all_sources_erasure_cost_rank"] = val("full_row_all_sources_erased", "rank") - val("full_row_original", "rank")
        if all(c in conds for c in ["full_row_original", "pair_correct_source"]):
            rr["full_row_minus_pair_correct_nll"] = val("full_row_original", "nll") - val("pair_correct_source", "nll")
            rr["full_row_minus_pair_correct_rank"] = val("full_row_original", "rank") - val("pair_correct_source", "rank")
        # Record deltas in interaction quantities relative to coherent86 and sparse.
        for prefix, ref in [("delta_vs_coherent86", pbb), ("delta_vs_sparse", sbb)]:
            if base_id not in ref:
                continue
            rconds = ref[base_id]
            def rval(c: str, key: str) -> Optional[float]:
                if c in rconds:
                    return float(rconds[c][key])
                return None
            if all(c in rconds for c in ["pair_correct_source", "pair_wrong_source", "view_only"]):
                ref_specific = rval("pair_wrong_source", "nll") - rval("pair_correct_source", "nll")
                ref_view = rval("view_only", "nll") - rval("pair_correct_source", "nll")
                ref_specific_rank = rval("pair_wrong_source", "rank") - rval("pair_correct_source", "rank")
                ref_view_rank = rval("view_only", "rank") - rval("pair_correct_source", "rank")
                rr[f"{prefix}_specific_source_penalty_nll"] = rr.get("specific_source_penalty_nll_wrong_minus_correct") - ref_specific
                rr[f"{prefix}_view_absence_penalty_nll"] = rr.get("view_absence_penalty_nll_view_minus_correct") - ref_view
                rr[f"{prefix}_specific_source_penalty_rank"] = rr.get("specific_source_penalty_rank_wrong_minus_correct") - ref_specific_rank
                rr[f"{prefix}_view_absence_penalty_rank"] = rr.get("view_absence_penalty_rank_view_minus_correct") - ref_view_rank
            if all(c in rconds for c in ["full_row_original", "full_row_this_source_erased", "full_row_all_sources_erased"]):
                ref_this = rval("full_row_this_source_erased", "nll") - rval("full_row_original", "nll")
                ref_all = rval("full_row_all_sources_erased", "nll") - rval("full_row_original", "nll")
                ref_this_rank = rval("full_row_this_source_erased", "rank") - rval("full_row_original", "rank")
                ref_all_rank = rval("full_row_all_sources_erased", "rank") - rval("full_row_original", "rank")
                rr[f"{prefix}_this_source_erasure_cost_nll"] = rr.get("this_source_erasure_cost_nll") - ref_this
                rr[f"{prefix}_all_sources_erasure_cost_nll"] = rr.get("all_sources_erasure_cost_nll") - ref_all
                rr[f"{prefix}_this_source_erasure_cost_rank"] = rr.get("this_source_erasure_cost_rank") - ref_this_rank
                rr[f"{prefix}_all_sources_erasure_cost_rank"] = rr.get("all_sources_erasure_cost_rank") - ref_all_rank
        interaction_rows.append(rr)
    interaction_summary: Dict[str, Any] = {"n_complete_base_targets": len(interaction_rows)}
    for key in [
        "specific_source_penalty_nll_wrong_minus_correct",
        "view_absence_penalty_nll_view_minus_correct",
        "generic_source_penalty_nll_view_minus_wrong",
        "specific_source_penalty_rank_wrong_minus_correct",
        "view_absence_penalty_rank_view_minus_correct",
        "this_source_erasure_cost_nll",
        "all_sources_erasure_cost_nll",
        "this_source_erasure_cost_rank",
        "all_sources_erasure_cost_rank",
        "full_row_minus_pair_correct_nll",
        "full_row_minus_pair_correct_rank",
        "delta_vs_coherent86_specific_source_penalty_nll",
        "delta_vs_coherent86_view_absence_penalty_nll",
        "delta_vs_coherent86_this_source_erasure_cost_nll",
        "delta_vs_coherent86_all_sources_erasure_cost_nll",
        "delta_vs_sparse_specific_source_penalty_nll",
        "delta_vs_sparse_view_absence_penalty_nll",
        "delta_vs_sparse_this_source_erasure_cost_nll",
        "delta_vs_sparse_all_sources_erasure_cost_nll",
    ]:
        vals = [r.get(key) for r in interaction_rows if r.get(key) is not None]
        interaction_summary[key] = {
            "mean": mean(vals),
            "median": median(vals),
            "sem": sem(vals),
            "positive_fraction": mean(1.0 if float(v) > 0 else 0.0 for v in vals),
        }
    return {
        "by_condition": by_condition,
        "interactions": interaction_summary,
        "interaction_rows_head": interaction_rows[:20],
    }


def comparison_against_baseline(summaries: Dict[str, Any], model: str, baseline: str) -> Dict[str, Any]:
    if model not in summaries or baseline not in summaries:
        return {}
    out: Dict[str, Any] = {}
    sm = summaries[model]["interactions"]
    sb = summaries[baseline]["interactions"]
    for key in [
        "specific_source_penalty_nll_wrong_minus_correct",
        "view_absence_penalty_nll_view_minus_correct",
        "this_source_erasure_cost_nll",
        "all_sources_erasure_cost_nll",
        "specific_source_penalty_rank_wrong_minus_correct",
        "view_absence_penalty_rank_view_minus_correct",
        "this_source_erasure_cost_rank",
        "all_sources_erasure_cost_rank",
    ]:
        a = sm.get(key, {}).get("mean")
        b = sb.get(key, {}).get("mean")
        out[f"{key}_mean_delta"] = (float(a) - float(b)) if a is not None and b is not None else None
    for cond in summaries[model]["by_condition"]:
        cm = summaries[model]["by_condition"][cond]
        cb = summaries[baseline]["by_condition"].get(cond, {})
        out[f"{cond}_mean_nll_delta"] = (float(cm["mean_nll"]) - float(cb["mean_nll"])) if cm.get("mean_nll") is not None and cb.get("mean_nll") is not None else None
        out[f"{cond}_mean_rank_delta"] = (float(cm["mean_rank"]) - float(cb["mean_rank"])) if cm.get("mean_rank") is not None and cb.get("mean_rank") is not None else None
    return out


def write_markdown(path: pathlib.Path, report: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research evidence-availability readout\n\n")
    lines.append(f"Status: `{report['status']}`\n\n")
    lines.append("This readout holds Qwen current-view target words fixed while changing the evidence state around the target. It is not an official BabyLM score; it is a mechanism/evidence instrument for the preservation route.\n\n")
    lines.append("## Task construction\n")
    lines.append(json.dumps(report["segment_stats"], indent=2, ensure_ascii=False) + "\n\n")
    lines.append(json.dumps(report["task_stats"], indent=2, ensure_ascii=False) + "\n\n")
    lines.append("## Model summaries\n")
    for name, summ in report["model_summaries"].items():
        lines.append(f"### {name}\n")
        inter = summ["interactions"]
        lines.append(f"- complete base targets: `{inter.get('n_complete_base_targets')}`\n")
        for key in [
            "specific_source_penalty_nll_wrong_minus_correct",
            "view_absence_penalty_nll_view_minus_correct",
            "this_source_erasure_cost_nll",
            "all_sources_erasure_cost_nll",
            "delta_vs_coherent86_specific_source_penalty_nll",
            "delta_vs_coherent86_view_absence_penalty_nll",
            "delta_vs_coherent86_this_source_erasure_cost_nll",
            "delta_vs_coherent86_all_sources_erasure_cost_nll",
        ]:
            val = inter.get(key, {}).get("mean") if isinstance(inter.get(key), dict) else inter.get(key)
            lines.append(f"- {key}: `{val}`\n")
        for cond in report["conditions"]:
            c = summ["by_condition"].get(cond, {})
            lines.append(f"  - {cond}: NLL `{c.get('mean_nll')}`, rank `{c.get('mean_rank')}`, ΔNLL vs coherent86 `{c.get('mean_delta_nll_vs_coherent86')}`, KL coherent86→model `{c.get('mean_kl_coherent86_to_model')}`\n")
        lines.append("\n")
    lines.append("## Dense-mask comparisons\n")
    lines.append(json.dumps(report.get("key_comparisons", {}), indent=2, ensure_ascii=False) + "\n\n")
    lines.append("## Interpretation scope\n")
    for k, v in report.get("interpretation", {}).items():
        lines.append(f"- **{k}**: {v}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=DEFAULT_TAIL)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--scope", choices=["prefix", "post_prefix", "all"], default="post_prefix")
    ap.add_argument("--max-updates", type=int, default=PREFIX_UPDATES)
    ap.add_argument("--words-per-update", type=int, default=WORDS_PER_UPDATE)
    ap.add_argument("--max-segments", type=int, default=96)
    ap.add_argument("--max-targets-per-segment", type=int, default=2)
    ap.add_argument("--sample-seed", type=int, default=90090)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--kl-temperature", type=float, default=1.0)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=DEFAULT_CONDITIONS)
    ap.add_argument("--models", nargs="+", default=["coherent86", "sparse_focus_seed62064", "densemask_sparselabel_seed62064", "dense_focus_seed62064", "dense_focus_seed62065", "pres_lambda1_trainmode_confounded"], choices=list(BASE_MODEL_SPECS.keys()))
    ap.add_argument("--extra-model", action="append", default=[], help="name=path checkpoint to score in addition to built-in model names")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_cache(args.out_dir)
    torch.set_num_threads(max(1, min(12, int(os.environ.get("QIUSHI_TORCH_THREADS", "8")))))
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    elif args.device == "auto" and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    segments, segment_stats = build_segment_records(args)
    tasks, task_stats = build_tasks(segments, tokenizer, args)
    write_jsonl(args.out_dir / "probe_tasks.jsonl", (task_for_json(t) for t in tasks))

    specs = dict(BASE_MODEL_SPECS)
    for spec in args.extra_model:
        if "=" not in spec:
            raise ValueError(f"--extra-model must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        specs[name] = pathlib.Path(path)
    requested = list(dict.fromkeys(["coherent86", "sparse_focus_seed62064"] + list(args.models) + [spec.split("=", 1)[0] for spec in args.extra_model]))
    selected_specs = {name: specs[name] for name in requested if name in specs}
    plan = {
        "status": "EVIDENCE_AVAILABILITY_READOUT_PLAN",
        "created_utc": now(),
        "tail_jsonl": rel(args.tail_jsonl),
        "out_dir": rel(args.out_dir),
        "device": str(device),
        "parameters": {
            "scope": args.scope,
            "max_updates": int(args.max_updates),
            "words_per_update": int(args.words_per_update),
            "max_segments": int(args.max_segments),
            "max_targets_per_segment": int(args.max_targets_per_segment),
            "sample_seed": int(args.sample_seed),
            "max_length": int(args.max_length),
            "conditions": list(args.conditions),
            "batch_size": int(args.batch_size),
            "kl_temperature": float(args.kl_temperature),
        },
        "segment_stats": segment_stats,
        "task_stats": task_stats,
        "model_specs": {name: rel(path) if path is not None else rel(bridge.PARENT_PATH) for name, path in selected_specs.items()},
        "scientific_use": "Fixed-target readout of correct evidence, wrong evidence, evidence absence, and packed-row source erasure for the dense-mask preservation route.",
    }
    (args.out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if not tasks:
        raise RuntimeError("no complete fixed-target tasks constructed")

    models, load_infos = load_models(selected_specs, device, float(args.private_scale))
    score_rows = score_all_models(models, tasks, device, tokenizer, int(args.batch_size), float(args.kl_temperature))
    score_paths: Dict[str, str] = {}
    for name, rows in score_rows.items():
        p = args.out_dir / f"scores_{name}.jsonl"
        write_jsonl(p, rows)
        score_paths[name] = rel(p)

    parent_rows = score_rows.get("coherent86", [])
    sparse_rows = score_rows.get("sparse_focus_seed62064", [])
    summaries: Dict[str, Any] = {}
    for name, rows in score_rows.items():
        summaries[name] = summarize_model(rows, parent_rows, sparse_rows, list(args.conditions))

    key_comparisons: Dict[str, Any] = {}
    for name in summaries:
        if name not in {"coherent86", "sparse_focus_seed62064"}:
            key_comparisons[f"{name}_minus_sparse"] = comparison_against_baseline(summaries, name, "sparse_focus_seed62064")
            key_comparisons[f"{name}_minus_densemask"] = comparison_against_baseline(summaries, name, "densemask_sparselabel_seed62064")
    if "dense_focus_seed62064" in summaries and "dense_focus_seed62065" in summaries:
        # Agreement on interaction rows for the two dense seeds, using row-head only in summaries;
        # full score files contain all rows for exact follow-up.
        pass

    report = {
        "status": "EVIDENCE_AVAILABILITY_READOUT_DONE",
        "created_utc": now(),
        "plan": rel(args.out_dir / "plan.json"),
        "conditions": list(args.conditions),
        "segment_stats": segment_stats,
        "task_stats": task_stats,
        "score_paths": score_paths,
        "load_infos": load_infos,
        "model_summaries": summaries,
        "key_comparisons": key_comparisons,
        "interpretation": {
            "scope": "The readout is fixed-target Qwen pair evidence-state scoring. It is not a BabyLM leaderboard component and should be combined with broad scores before judging a candidate.",
            "held_out_default": "Default scope is post_prefix, so targets are outside the 80-update Qwen prefix used by the dense-mask training arms; use scope=prefix when direct trained-material acquisition is the question.",
            "source_erasure": "Source-erased full-row states blank character spans to preserve target offsets while removing lexical source evidence. Other current-view text may still contain target words, and leakage flags are recorded.",
            "preservation_use": "A useful preservation endpoint should reduce view-only, wrong-source, source-erased, and CDI/broad-language damage without merely erasing correct-source/Entity movement.",
        },
    }
    out_json = args.out_dir / "evidence_availability_readout.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.out_dir / "evidence_availability_readout.md", report)
    print(json.dumps({"status": report["status"], "out_json": rel(out_json), "out_md": rel(args.out_dir / 'evidence_availability_readout.md'), "models": list(score_rows.keys()), "tasks": len(tasks)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
