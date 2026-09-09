#!/usr/bin/env python3
"""research: Qwen source-alignment x view-clue factorial probe.

Scientific purpose
------------------
Dense unchanged-Qwen focus may work because it masks many second-view content words,
removing local completion clues while preserving the paired source.  This script
constructs an evaluation-only factorial that separates two factors on the same target
words from the unchanged Qwen-pair tail:

  source alignment: aligned source vs length-matched shuffled source vs no source
  view clues:       retained local view content vs suppressed view content

For each selected target content group, retained masks only the target group; suppressed
masks every detected content-word group in the second view and scores only the target
group.  A useful source-to-view route should give a larger aligned-vs-shuffled advantage
when local content clues are suppressed, reproducibly across dense seeds and more than
for parent/sparse/ordinary controls.

This is a mechanism readout, not BabyLM official evaluation.
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
import pathlib
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import coherent86_continuation_trainer as base_loader  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402
import concentrated_compact_learning_test as s57  # noqa: E402

DEFAULT_TAIL = _public_path('experiments/archive/functional_learning/data/unchanged_pair_segments_tail/reference_tail_unchanged_qwen_segments.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/qwen_alignment_clue_factorial')

SOURCE_TYPES = ["aligned_source", "shuffled_source", "no_source"]
CLUE_TYPES = ["retained_view_clues", "suppressed_view_clues"]


def rel(p: pathlib.Path | str | None) -> str | None:
    if p is None:
        return None
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_seed(*parts: Any) -> int:
    h = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def wc(text: str) -> int:
    return len((text or "").strip().split())


def finite(vals: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for x in vals:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def mean(vals: Iterable[Any]) -> Optional[float]:
    xs = finite(vals)
    return sum(xs) / len(xs) if xs else None


def median(vals: Iterable[Any]) -> Optional[float]:
    xs = finite(vals)
    return statistics.median(xs) if xs else None


def trimmed_mean(vals: Iterable[Any], frac: float = 0.1) -> Optional[float]:
    xs = sorted(finite(vals))
    if not xs:
        return None
    k = int(len(xs) * frac)
    if 2 * k >= len(xs):
        return sum(xs) / len(xs)
    ys = xs[k:len(xs)-k]
    return sum(ys) / len(ys)


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def content_spans(text: str) -> List[Tuple[int, int, str]]:
    return list(s57.content_word_spans(text))


WORDLIKE_RE = re.compile(r"[a-z0-9]+(?:['’.-][a-z0-9]+)*")


def normalize_for_copy(text: str) -> str:
    return " ".join(WORDLIKE_RE.findall((text or "").lower().replace("’", "'")))


def target_copy_features(aligned_source: str, target_word: str) -> Dict[str, Any]:
    """Classify whether the target surface is literally available in the aligned source.

    This is intentionally a surface diagnostic, not a semantic-correction test.  Exact
    copied targets can often be solved by lexical reuse from the source; source-absent
    targets require paraphrase, inference, or local completion.  The larger factorial
    should report both groups because shuffled-source controls confound semantic
    correspondence with lexical/topic availability.
    """
    source_norm = normalize_for_copy(aligned_source)
    target_norm = normalize_for_copy(target_word)
    source_tokens = set(source_norm.split())
    exact = bool(target_norm) and target_norm in source_tokens
    substring = bool(target_norm) and target_norm in source_norm
    if exact:
        copy_class = "exact_source_copy"
        copy_binary = "copied"
    elif substring:
        copy_class = "source_substring_only"
        copy_binary = "copied"
    else:
        copy_class = "source_absent"
        copy_binary = "not_copied"
    return {
        "aligned_source_target_copy_class": copy_class,
        "aligned_source_target_copy_binary": copy_binary,
        "target_norm_for_copy": target_norm,
        "target_in_aligned_source_exact": exact,
        "target_in_aligned_source_substring": substring,
    }


def locate_positions(offsets: List[Tuple[int, int]], start: int, end: int) -> List[int]:
    pos: List[int] = []
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        if a < end and b > start:
            pos.append(i)
    return pos


def iter_prefix_segments(tail_jsonl: pathlib.Path, max_updates: int, words_per_update: int):
    limit = int(max_updates) * int(words_per_update)
    total_words = 0
    row_idx = 0
    with tail_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if total_words >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            words = int(row.get("words", wc(row.get("text", ""))))
            if row.get("source") == "qwen_pair_packed" and row.get("qwen_pair_segments"):
                for seg_i, seg in enumerate(row.get("qwen_pair_segments") or []):
                    yield row_idx, seg_i, row, seg
            total_words += words
            row_idx += 1


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def build_segment_records(args: argparse.Namespace, tokenizer) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    stats = Counter()
    for row_idx, seg_i, _row, seg in iter_prefix_segments(args.tail_jsonl, int(args.max_updates), int(args.words_per_update)):
        source = str(seg.get("source_text", "")).strip()
        view = str(seg.get("view_text", "")).strip()
        if not source or not view:
            stats["empty_source_or_view"] += 1
            continue
        groups = content_spans(view)
        if not groups:
            stats["no_content_groups"] += 1
            continue
        records.append({
            "row_idx": int(row_idx),
            "segment_index": int(seg_i),
            "pair_id": str(seg.get("pair_id")),
            "candidate_kind": str(seg.get("candidate_kind", "unknown")),
            "view_kind": str(seg.get("view_kind", "unknown")),
            "source_text": source,
            "view_text": view,
            "source_words": int(seg.get("source_words", wc(source))),
            "view_words": int(seg.get("view_words", wc(view))),
            "source_token_count": token_count(tokenizer, source),
            "view_token_count": token_count(tokenizer, view),
            "groups": groups,
        })
        stats["segments_available"] += 1
    rng = random.Random(int(args.sample_seed))
    if int(args.max_segments) > 0 and len(records) > int(args.max_segments):
        records = rng.sample(records, int(args.max_segments))
        records.sort(key=lambda r: (r["row_idx"], r["segment_index"], r["pair_id"]))
        stats["segments_sampled"] = len(records)
    else:
        stats["segments_sampled"] = len(records)
    stats["mean_source_tokens_sampled"] = mean(r["source_token_count"] for r in records)
    stats["mean_view_tokens_sampled"] = mean(r["view_token_count"] for r in records)
    return records, dict(stats)


def choose_shuffled_source(records: List[Dict[str, Any]], rec_i: int, target_word: str, sample_seed: int) -> Tuple[Optional[Dict[str, Any]], str]:
    current = records[rec_i]
    target_l = str(target_word).lower()
    candidates: List[Tuple[int, int, int]] = []
    fallback: List[Tuple[int, int, int]] = []
    for j, r in enumerate(records):
        if j == rec_i or r["pair_id"] == current["pair_id"]:
            continue
        dist = abs(int(r["source_token_count"]) - int(current["source_token_count"]))
        word_dist = abs(int(r["source_words"]) - int(current["source_words"]))
        tup = (dist, word_dist, j)
        fallback.append(tup)
        if target_l and target_l not in r["source_text"].lower() and target_l not in r["view_text"].lower():
            candidates.append(tup)
    pool = candidates or fallback
    if not pool:
        return None, "no_shuffled_source_available"
    pool.sort()
    # Choose deterministically among the closest few rather than always the single nearest.
    best_dist = pool[0][0]
    close = [x for x in pool if x[0] <= best_dist + int(max(1, current["source_token_count"] * 0.05))]
    # Bound tie set to avoid systematic reuse of the same source.
    close = close[: min(len(close), 64)]
    rng = random.Random(stable_seed("aligned-shuffled-source", int(sample_seed), current["row_idx"], current["segment_index"], current["pair_id"], target_word))
    _, _, j = close[rng.randrange(len(close))]
    reason = "target_absent_length_matched" if candidates else "fallback_length_matched"
    return records[j], reason


def build_one_condition(tokenizer, rec: Dict[str, Any], group_i: int, group: Tuple[int, int, str],
                        source_type: str, clue_type: str, source_rec: Optional[Dict[str, Any]],
                        source_choice_reason: str, max_length: int) -> Optional[Dict[str, Any]]:
    a, b, word = group
    if source_type == "aligned_source":
        prefix_source = rec["source_text"]
        source_pair_id = rec["pair_id"]
        source_status = "aligned"
    elif source_type == "shuffled_source":
        if source_rec is None:
            return None
        prefix_source = source_rec["source_text"]
        source_pair_id = source_rec["pair_id"]
        source_status = source_choice_reason
    elif source_type == "no_source":
        prefix_source = ""
        source_pair_id = ""
        source_status = "none"
    else:
        raise ValueError(source_type)
    if prefix_source:
        text = prefix_source + " " + rec["view_text"]
        view_offset = len(prefix_source) + 1
    else:
        text = rec["view_text"]
        view_offset = 0
    copy_features = target_copy_features(str(rec["source_text"]), str(word))
    enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=int(max_length), return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(x), int(y)) for x, y in enc["offset_mapping"]]
    target_positions = locate_positions(offsets, view_offset + int(a), view_offset + int(b))
    if not target_positions:
        return None
    mask_positions = set(target_positions)
    if clue_type == "suppressed_view_clues":
        for ga, gb, _gw in rec["groups"]:
            for p in locate_positions(offsets, view_offset + int(ga), view_offset + int(gb)):
                mask_positions.add(p)
    elif clue_type != "retained_view_clues":
        raise ValueError(clue_type)
    mask_positions = sorted(p for p in mask_positions if 0 <= p < len(ids))
    target_positions = sorted(p for p in target_positions if 0 <= p < len(ids))
    if not target_positions:
        return None
    return {
        "base_task_id": f"r{rec['row_idx']:06d}_s{rec['segment_index']:02d}_g{group_i:02d}",
        "task_id": f"r{rec['row_idx']:06d}_s{rec['segment_index']:02d}_g{group_i:02d}_{source_type}_{clue_type}",
        "row_idx": rec["row_idx"],
        "segment_index": rec["segment_index"],
        "pair_id": rec["pair_id"],
        "candidate_kind": rec["candidate_kind"],
        "view_kind": rec["view_kind"],
        "target_index": int(group_i),
        "target_word": str(word),
        "aligned_source_target_copy_class": copy_features["aligned_source_target_copy_class"],
        "aligned_source_target_copy_binary": copy_features["aligned_source_target_copy_binary"],
        "target_norm_for_copy": copy_features["target_norm_for_copy"],
        "target_in_aligned_source_exact": copy_features["target_in_aligned_source_exact"],
        "target_in_aligned_source_substring": copy_features["target_in_aligned_source_substring"],
        "target_char_span": [int(a), int(b)],
        "source_type": source_type,
        "clue_type": clue_type,
        "source_pair_id_used": source_pair_id,
        "source_status": source_status,
        "source_token_delta_vs_aligned": None if source_type == "no_source" else int(token_count(tokenizer, prefix_source)) - int(rec["source_token_count"]),
        "input_ids": ids,
        "target_positions": target_positions,
        "mask_positions": mask_positions,
        "n_target_positions": len(target_positions),
        "n_mask_positions": len(mask_positions),
        "source_words": rec["source_words"],
        "view_words": rec["view_words"],
        "n_content_groups_in_view": len(rec["groups"]),
        "view_text": rec["view_text"] if len(rec["view_text"]) <= 240 else rec["view_text"][:237] + "...",
        "source_text_preview": prefix_source if len(prefix_source) <= 240 else prefix_source[:237] + "...",
    }


def build_tasks(records: List[Dict[str, Any]], tokenizer, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    skips = Counter()
    source_reasons = Counter()
    source_token_deltas: List[int] = []
    for rec_i, rec in enumerate(records):
        groups = list(rec["groups"])
        rng = random.Random(stable_seed("factorial-targets", int(args.sample_seed), rec["row_idx"], rec["segment_index"], rec["pair_id"]))
        if len(groups) > int(args.max_targets_per_segment):
            groups = rng.sample(groups, int(args.max_targets_per_segment))
            groups.sort(key=lambda x: x[0])
        for gi, group in enumerate(groups):
            shuffled_rec, reason = choose_shuffled_source(records, rec_i, group[2], int(args.sample_seed))
            source_reasons[reason] += 1
            for source_type in SOURCE_TYPES:
                for clue_type in CLUE_TYPES:
                    src_rec = shuffled_rec if source_type == "shuffled_source" else None
                    task = build_one_condition(tokenizer, rec, gi, group, source_type, clue_type, src_rec, reason, int(args.max_length))
                    if task is None:
                        skips[f"{source_type}_{clue_type}_no_task"] += 1
                        continue
                    if task["source_token_delta_vs_aligned"] is not None and source_type == "shuffled_source":
                        source_token_deltas.append(int(task["source_token_delta_vs_aligned"]))
                    tasks.append(task)
    base_copy_class = {t["base_task_id"]: t.get("aligned_source_target_copy_class", "unknown") for t in tasks}
    base_copy_binary = {t["base_task_id"]: t.get("aligned_source_target_copy_binary", "unknown") for t in tasks}
    meta = {
        "n_tasks": len(tasks),
        "n_base_targets": len(set(t["base_task_id"] for t in tasks)),
        "condition_counts": dict(Counter(f"{t['source_type']}|{t['clue_type']}" for t in tasks)),
        "base_target_copy_class_counts": dict(Counter(base_copy_class.values())),
        "base_target_copy_binary_counts": dict(Counter(base_copy_binary.values())),
        "source_choice_reasons": dict(source_reasons),
        "skips": dict(skips),
        "shuffled_source_token_delta": {
            "n": len(source_token_deltas),
            "mean": mean(source_token_deltas),
            "median": median(source_token_deltas),
            "max_abs": max([abs(x) for x in source_token_deltas], default=None),
        },
    }
    return tasks, meta


def load_scoring_model(model_path: Optional[pathlib.Path], device: torch.device, private_scale: float):
    endpoint = pathlib.Path(model_path) if model_path is not None else bridge.PARENT_PATH
    model, missing, unexpected = base_loader.load_model(endpoint, device, 128, float(private_scale))
    try:
        for layer in model.deberta.encoder.layer:
            layer.private_adapter.scale = float(private_scale)
            layer.private_adapter.enabled = True
    except Exception:
        pass
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(True)
    model.eval()
    ident = bridge.model_identity(model)
    return model, {"endpoint": rel(endpoint), "missing": list(missing), "unexpected": list(unexpected), "identity": ident}


def score_tasks(model, tokenizer, tasks: List[Dict[str, Any]], device: torch.device, batch_size: int) -> List[Dict[str, Any]]:
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    mask_id = int(tokenizer.mask_token_id)
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(tasks), int(batch_size)):
        batch = tasks[start:start + int(batch_size)]
        max_len = max(len(t["input_ids"]) for t in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        att = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        orig = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        for i, t in enumerate(batch):
            arr = torch.tensor(t["input_ids"], dtype=torch.long, device=device)
            L = len(t["input_ids"])
            ids[i, :L] = arr
            orig[i, :L] = arr
            att[i, :L] = 1
            for p in t["mask_positions"]:
                if 0 <= int(p) < L:
                    ids[i, int(p)] = mask_id
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        for i, t in enumerate(batch):
            nlls: List[float] = []
            for p in t["target_positions"]:
                p = int(p)
                if p >= logits.shape[1]:
                    continue
                lp = torch.log_softmax(logits[i, p], dim=-1)
                nlls.append(-float(lp[int(orig[i, p])].detach().cpu()))
            rr = {k: v for k, v in t.items() if k not in {"input_ids", "target_positions", "mask_positions"}}
            rr["nll"] = sum(nlls) / len(nlls) if nlls else float("nan")
            rr["n_token_scores"] = len(nlls)
            rows.append(rr)
        del ids, att, orig, logits
    return rows


def summarize_scores(rows: List[Dict[str, Any]], parent_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    parent_by_task = {r["task_id"]: r for r in parent_rows or []}

    def condition_summary(g: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n": len(g),
            "n_base_targets": len(set(r["base_task_id"] for r in g)),
            "copy_binary_counts": dict(Counter(str(r.get("aligned_source_target_copy_binary", "unknown")) for r in g)),
            "copy_class_counts": dict(Counter(str(r.get("aligned_source_target_copy_class", "unknown")) for r in g)),
            "mean_nll": mean(r["nll"] for r in g),
            "median_nll": median(r["nll"] for r in g),
            "mean_delta_nll_vs_parent": mean(float(r["nll"]) - float(parent_by_task[r["task_id"]]["nll"]) for r in g if r["task_id"] in parent_by_task) if parent_rows is not None else None,
        }

    by_cond: Dict[str, Any] = {}
    for source_type in SOURCE_TYPES:
        for clue_type in CLUE_TYPES:
            key = f"{source_type}|{clue_type}"
            g = [r for r in rows if r["source_type"] == source_type and r["clue_type"] == clue_type]
            by_cond[key] = condition_summary(g)

    by_base: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_base[str(r["base_task_id"])][(str(r["source_type"]), str(r["clue_type"]))] = r
    parent_by_base: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = defaultdict(dict)
    for r in parent_rows or []:
        parent_by_base[str(r["base_task_id"])][(str(r["source_type"]), str(r["clue_type"]))] = r

    inter_rows: List[Dict[str, Any]] = []
    needed = [("aligned_source", c) for c in CLUE_TYPES] + [("shuffled_source", c) for c in CLUE_TYPES]
    for base_id, vals in by_base.items():
        if not all(k in vals for k in needed):
            continue
        d: Dict[str, float] = {}
        row: Dict[str, Any] = {
            "base_task_id": base_id,
            "pair_id": vals[("aligned_source", "retained_view_clues")]["pair_id"],
            "target_word": vals[("aligned_source", "retained_view_clues")]["target_word"],
            "aligned_source_target_copy_class": vals[("aligned_source", "retained_view_clues")].get("aligned_source_target_copy_class", "unknown"),
            "aligned_source_target_copy_binary": vals[("aligned_source", "retained_view_clues")].get("aligned_source_target_copy_binary", "unknown"),
            "target_norm_for_copy": vals[("aligned_source", "retained_view_clues")].get("target_norm_for_copy", ""),
            "target_in_aligned_source_exact": vals[("aligned_source", "retained_view_clues")].get("target_in_aligned_source_exact"),
            "target_in_aligned_source_substring": vals[("aligned_source", "retained_view_clues")].get("target_in_aligned_source_substring"),
        }
        for clue_type in CLUE_TYPES:
            aligned = float(vals[("aligned_source", clue_type)]["nll"])
            shuffled = float(vals[("shuffled_source", clue_type)]["nll"])
            no_src = vals.get(("no_source", clue_type))
            row[f"nll_aligned_{clue_type}"] = aligned
            row[f"nll_shuffled_{clue_type}"] = shuffled
            row[f"nll_no_source_{clue_type}"] = float(no_src["nll"]) if no_src is not None else None
            d[clue_type] = shuffled - aligned  # positive means aligned source helps vs shuffled.
            row[f"alignment_advantage_{clue_type}"] = d[clue_type]
            if no_src is not None:
                row[f"source_presence_advantage_aligned_{clue_type}"] = float(no_src["nll"]) - aligned
                row[f"source_presence_advantage_shuffled_{clue_type}"] = float(no_src["nll"]) - shuffled
        row["alignment_advantage_retained"] = d["retained_view_clues"]
        row["alignment_advantage_suppressed"] = d["suppressed_view_clues"]
        row["clue_suppression_interaction"] = d["suppressed_view_clues"] - d["retained_view_clues"]
        if parent_rows is not None and base_id in parent_by_base and all(k in parent_by_base[base_id] for k in needed):
            pvals = parent_by_base[base_id]
            pd: Dict[str, float] = {}
            for clue_type in CLUE_TYPES:
                paligned = float(pvals[("aligned_source", clue_type)]["nll"])
                pshuffled = float(pvals[("shuffled_source", clue_type)]["nll"])
                pno = pvals.get(("no_source", clue_type))
                pd[clue_type] = pshuffled - paligned
                row[f"delta_nll_aligned_{clue_type}_vs_parent"] = row[f"nll_aligned_{clue_type}"] - paligned
                row[f"delta_nll_shuffled_{clue_type}_vs_parent"] = row[f"nll_shuffled_{clue_type}"] - pshuffled
                if pno is not None and row.get(f"nll_no_source_{clue_type}") is not None:
                    row[f"delta_nll_no_source_{clue_type}_vs_parent"] = float(row[f"nll_no_source_{clue_type}"]) - float(pno["nll"])
                row[f"delta_alignment_advantage_{clue_type}_vs_parent"] = row[f"alignment_advantage_{clue_type}"] - pd[clue_type]
                if pno is not None and row.get(f"source_presence_advantage_aligned_{clue_type}") is not None:
                    p_source_aligned = float(pno["nll"]) - paligned
                    p_source_shuffled = float(pno["nll"]) - pshuffled
                    row[f"delta_source_presence_advantage_aligned_{clue_type}_vs_parent"] = row[f"source_presence_advantage_aligned_{clue_type}"] - p_source_aligned
                    row[f"delta_source_presence_advantage_shuffled_{clue_type}_vs_parent"] = row[f"source_presence_advantage_shuffled_{clue_type}"] - p_source_shuffled
            row["delta_alignment_advantage_retained_vs_parent"] = row["delta_alignment_advantage_retained_view_clues_vs_parent"]
            row["delta_alignment_advantage_suppressed_vs_parent"] = row["delta_alignment_advantage_suppressed_view_clues_vs_parent"]
            row["delta_clue_suppression_interaction_vs_parent"] = row["clue_suppression_interaction"] - (pd["suppressed_view_clues"] - pd["retained_view_clues"])
        inter_rows.append(row)

    def aggregate_interactions(g: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n_complete_base_targets": len(g),
            "copy_binary_counts": dict(Counter(str(r.get("aligned_source_target_copy_binary", "unknown")) for r in g)),
            "copy_class_counts": dict(Counter(str(r.get("aligned_source_target_copy_class", "unknown")) for r in g)),
            "mean_nll_aligned_retained": mean(r.get("nll_aligned_retained_view_clues") for r in g),
            "mean_nll_aligned_suppressed": mean(r.get("nll_aligned_suppressed_view_clues") for r in g),
            "mean_delta_nll_aligned_retained_vs_parent": mean(r.get("delta_nll_aligned_retained_view_clues_vs_parent") for r in g),
            "mean_delta_nll_aligned_suppressed_vs_parent": mean(r.get("delta_nll_aligned_suppressed_view_clues_vs_parent") for r in g),
            "mean_alignment_advantage_retained": mean(r["alignment_advantage_retained"] for r in g),
            "median_alignment_advantage_retained": median(r["alignment_advantage_retained"] for r in g),
            "trimmed_mean_alignment_advantage_retained": trimmed_mean(r["alignment_advantage_retained"] for r in g),
            "mean_alignment_advantage_suppressed": mean(r["alignment_advantage_suppressed"] for r in g),
            "median_alignment_advantage_suppressed": median(r["alignment_advantage_suppressed"] for r in g),
            "trimmed_mean_alignment_advantage_suppressed": trimmed_mean(r["alignment_advantage_suppressed"] for r in g),
            "mean_delta_alignment_advantage_retained_vs_parent": mean(r.get("delta_alignment_advantage_retained_vs_parent") for r in g),
            "mean_delta_alignment_advantage_suppressed_vs_parent": mean(r.get("delta_alignment_advantage_suppressed_vs_parent") for r in g),
            "mean_clue_suppression_interaction": mean(r["clue_suppression_interaction"] for r in g),
            "median_clue_suppression_interaction": median(r["clue_suppression_interaction"] for r in g),
            "trimmed_mean_clue_suppression_interaction": trimmed_mean(r["clue_suppression_interaction"] for r in g),
            "share_interaction_positive": (sum(1 for r in g if r["clue_suppression_interaction"] > 0) / len(g)) if g else None,
            "mean_delta_clue_suppression_interaction_vs_parent": mean(r.get("delta_clue_suppression_interaction_vs_parent") for r in g),
            "median_delta_clue_suppression_interaction_vs_parent": median(r.get("delta_clue_suppression_interaction_vs_parent") for r in g),
            "share_delta_interaction_positive_vs_parent": (sum(1 for r in g if r.get("delta_clue_suppression_interaction_vs_parent") is not None and r["delta_clue_suppression_interaction_vs_parent"] > 0) / len(g)) if g else None,
            "mean_source_presence_advantage_aligned_retained": mean(r.get("source_presence_advantage_aligned_retained_view_clues") for r in g),
            "mean_source_presence_advantage_aligned_suppressed": mean(r.get("source_presence_advantage_aligned_suppressed_view_clues") for r in g),
            "mean_source_presence_advantage_shuffled_retained": mean(r.get("source_presence_advantage_shuffled_retained_view_clues") for r in g),
            "mean_source_presence_advantage_shuffled_suppressed": mean(r.get("source_presence_advantage_shuffled_suppressed_view_clues") for r in g),
            "mean_delta_source_presence_advantage_aligned_retained_vs_parent": mean(r.get("delta_source_presence_advantage_aligned_retained_view_clues_vs_parent") for r in g),
            "mean_delta_source_presence_advantage_aligned_suppressed_vs_parent": mean(r.get("delta_source_presence_advantage_aligned_suppressed_view_clues_vs_parent") for r in g),
        }

    summary = {
        "by_condition": by_cond,
        "alignment_clue_interaction": aggregate_interactions(inter_rows),
        "by_target_copy_binary": {
            key: aggregate_interactions([r for r in inter_rows if str(r.get("aligned_source_target_copy_binary", "unknown")) == key])
            for key in sorted(set(str(r.get("aligned_source_target_copy_binary", "unknown")) for r in inter_rows))
        },
        "by_target_copy_class": {
            key: aggregate_interactions([r for r in inter_rows if str(r.get("aligned_source_target_copy_class", "unknown")) == key])
            for key in sorted(set(str(r.get("aligned_source_target_copy_class", "unknown")) for r in inter_rows))
        },
        "interpretation_note": "Read interaction together with absolute aligned-source NLL and the retained/suppressed alignment advantages. Shuffled-source controls are target-string-absent length matches, not semantic counterfactuals; copied and noncopied targets can behave differently.",
    }
    return {"summary": summary, "interaction_rows": inter_rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=DEFAULT_TAIL)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--max-updates", type=int, default=80)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--max-segments", type=int, default=1200)
    ap.add_argument("--max-targets-per-segment", type=int, default=4)
    ap.add_argument("--sample-seed", type=int, default=74074)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--extra-model", action="append", default=[], help="name=path checkpoint to score")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    if tokenizer.mask_token_id is None:
        raise RuntimeError("Tokenizer has no [MASK] token")
    records, segment_stats = build_segment_records(args, tokenizer)
    tasks, task_stats = build_tasks(records, tokenizer, args)
    write_jsonl(args.out_dir / "factorial_tasks.jsonl", ({k: v for k, v in t.items() if k not in {"input_ids", "target_positions", "mask_positions"}} for t in tasks))
    plan = {
        "status": "QWEN_ALIGNMENT_CLUE_FACTORIAL_PLAN",
        "created_utc": now_utc(),
        "tail_jsonl": rel(args.tail_jsonl),
        "out_dir": rel(args.out_dir),
        "parameters": {
            "max_updates": int(args.max_updates),
            "words_per_update": int(args.words_per_update),
            "max_segments": int(args.max_segments),
            "max_targets_per_segment": int(args.max_targets_per_segment),
            "sample_seed": int(args.sample_seed),
            "max_length": int(args.max_length),
        },
        "segment_stats": segment_stats,
        "task_stats": task_stats,
        "source_types": SOURCE_TYPES,
        "clue_types": CLUE_TYPES,
        "scientific_use": "Evaluation-only test of whether dense focus increases correct-source advantage especially when second-view content clues are suppressed. Compare parent, ordinary WWM, sparse focus, dense seed62064, and replicated dense seed62065 when available.",
        "interpretation_boundaries": [
            "Shuffled source is length-matched from another pair, usually target-string absent, but it is not a semantic counterfactual source.",
            "Suppressed-view scoring masks all detected content groups and scores only the selected target group, approximating the dense effective-input geometry.",
            "The result is mechanistic evidence about trained unchanged Qwen material, not official BabyLM competence.",
        ],
    }
    (args.out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")
    model_specs: List[Tuple[str, Optional[pathlib.Path]]] = [("parent", None)]
    for spec in args.extra_model:
        if "=" not in spec:
            raise ValueError(f"--extra-model must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        model_specs.append((name, pathlib.Path(path)))

    score_paths: Dict[str, str] = {}
    load_infos: Dict[str, Any] = {}
    summaries: Dict[str, Any] = {}
    parent_rows: Optional[List[Dict[str, Any]]] = None
    for name, path in model_specs:
        print(json.dumps({"event": "load_model", "model": name, "path": rel(path) if path else rel(bridge.PARENT_PATH), "device": str(device)}), flush=True)
        model, load_info = load_scoring_model(path, device, float(args.private_scale))
        load_infos[name] = load_info
        rows = score_tasks(model, tokenizer, tasks, device, int(args.batch_size))
        for r in rows:
            r["model"] = name
        out_path = args.out_dir / f"scores_{name}.jsonl"
        write_jsonl(out_path, rows)
        score_paths[name] = rel(out_path)
        if name == "parent":
            parent_rows = rows
        summ = summarize_scores(rows, parent_rows)
        summaries[name] = summ["summary"]
        write_jsonl(args.out_dir / f"interaction_rows_{name}.jsonl", summ["interaction_rows"])
        print(json.dumps({"event": "model_done", "model": name, "interaction": summ["summary"].get("alignment_clue_interaction")}, ensure_ascii=False), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    final = {
        "status": "QWEN_ALIGNMENT_CLUE_FACTORIAL_DONE",
        "created_utc": now_utc(),
        "plan": rel(args.out_dir / "plan.json"),
        "factorial_tasks": rel(args.out_dir / "factorial_tasks.jsonl"),
        "score_paths": score_paths,
        "load_infos": load_infos,
        "model_summaries": summaries,
        "scientific_status": "mechanistic trained-material readout; interpret alongside official/fast benchmark and common-target results",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
