#!/usr/bin/env python3
"""research: source-visible surface probe for trained qwen source/current-rewrite pairs.

The research unchanged-tail objective trains on the inherited qwen source/current pairs
inside the legal stream.  This probe measures the material actually trained: masked
content words in the current-rewrite second views, either with the source sentence
visible or with the view alone.  It reports NLL changes against coherent86 on a fixed,
deterministically sampled subset of pair occurrences from the same prefix.

This is not a semantic-transfer result by itself.  It separates ordinary broad scores
from direct acquisition/reconstruction of the paired source/view material.
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
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/qwen_view_surface_probe')


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


def wc(text: str) -> int:
    return len((text or "").strip().split())


def finite_mean(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def finite_median(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.median(vals) if vals else None


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def content_word_spans(text: str) -> List[Tuple[int, int, str]]:
    return s57.content_word_spans(text)


def locate_positions(offsets: List[Tuple[int, int]], start: int, end: int) -> List[int]:
    pos: List[int] = []
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        if s < end and e > start:
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


def build_tasks(tail_jsonl: pathlib.Path, tokenizer, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    segment_records = []
    stats = Counter()
    for row_idx, seg_i, row, seg in iter_prefix_segments(tail_jsonl, int(args.max_updates), int(args.words_per_update)):
        source = str(seg.get("source_text", "")).strip()
        view = str(seg.get("view_text", "")).strip()
        if not source or not view:
            stats["empty_source_or_view"] += 1
            continue
        groups = content_word_spans(view)
        if not groups:
            stats["no_content_groups"] += 1
            continue
        segment_records.append({
            "row_idx": row_idx,
            "segment_index": seg_i,
            "pair_id": seg.get("pair_id"),
            "candidate_kind": seg.get("candidate_kind", "unknown"),
            "view_kind": seg.get("view_kind", "unknown"),
            "source_text": source,
            "view_text": view,
            "source_words": int(seg.get("source_words", wc(source))),
            "view_words": int(seg.get("view_words", wc(view))),
            "groups": groups,
        })
        stats["segments_available"] += 1
    rng = random.Random(int(args.sample_seed))
    if int(args.max_segments) > 0 and len(segment_records) > int(args.max_segments):
        segment_records = rng.sample(segment_records, int(args.max_segments))
        segment_records.sort(key=lambda x: (x["row_idx"], x["segment_index"], str(x["pair_id"])))
        stats["segments_sampled"] = len(segment_records)
    else:
        stats["segments_sampled"] = len(segment_records)
    tasks: List[Dict[str, Any]] = []
    skips = Counter()
    for rec in segment_records:
        groups = list(rec["groups"])
        grng = random.Random(stable_seed("qwen-view-probe-groups", int(args.sample_seed), rec["row_idx"], rec["segment_index"], rec["pair_id"]))
        if len(groups) > int(args.max_targets_per_segment):
            groups = grng.sample(groups, int(args.max_targets_per_segment))
            groups.sort(key=lambda x: x[0])
        for condition in ["with_source", "view_only"]:
            if condition == "with_source":
                prefix = rec["source_text"] + " "
                text = prefix + rec["view_text"]
                view_offset = len(prefix)
            else:
                text = rec["view_text"]
                view_offset = 0
            enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=int(args.max_length), return_offsets_mapping=True)
            ids = [int(x) for x in enc["input_ids"]]
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            for gi, (a, b, word) in enumerate(groups):
                pos = locate_positions(offsets, view_offset + a, view_offset + b)
                if not pos:
                    skips[f"{condition}_no_pos"] += 1
                    continue
                tasks.append({
                    "task_id": f"r{rec['row_idx']:06d}_s{rec['segment_index']:02d}_g{gi:02d}_{condition}",
                    "row_idx": rec["row_idx"],
                    "segment_index": rec["segment_index"],
                    "pair_id": rec["pair_id"],
                    "candidate_kind": rec["candidate_kind"],
                    "view_kind": rec["view_kind"],
                    "condition": condition,
                    "target_index": gi,
                    "target_word": word,
                    "input_ids": ids,
                    "positions": pos,
                    "source_words": rec["source_words"],
                    "view_words": rec["view_words"],
                })
    return tasks, {"segment_stats": dict(stats), "skips": dict(skips), "n_tasks": len(tasks), "n_pairs": len(set(str(t.get("pair_id")) for t in tasks))}


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


def score_tasks(model, tasks: List[Dict[str, Any]], device: torch.device, tokenizer, batch_size: int) -> List[Dict[str, Any]]:
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


def summarize_scores(score_rows: List[Dict[str, Any]], parent_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    parent_by_key = {}
    if parent_rows is not None:
        for r in parent_rows:
            key = (r["task_id"], r["condition"])
            parent_by_key[key] = r
    by_condition = {}
    for cond in sorted(set(r["condition"] for r in score_rows)):
        g = [r for r in score_rows if r["condition"] == cond]
        by_condition[cond] = {
            "n_targets": len(g),
            "n_pairs": len(set(r["pair_id"] for r in g)),
            "mean_nll": finite_mean(r["nll"] for r in g),
            "median_nll": finite_median(r["nll"] for r in g),
            "mean_delta_nll_vs_parent": finite_mean(r["nll"] - parent_by_key.get((r["task_id"], r["condition"]), {"nll": r["nll"]})["nll"] for r in g) if parent_rows is not None else None,
        }
    paired: Dict[Tuple[str, int, int], Dict[str, float]] = defaultdict(dict)
    for r in score_rows:
        paired[(str(r["pair_id"]), int(r["row_idx"]), int(r["target_index"]))][r["condition"]] = float(r["nll"])
    source_help = []
    for (pid, row_idx, ti), vals in paired.items():
        if "with_source" in vals and "view_only" in vals:
            source_help.append({"pair_id": pid, "row_idx": row_idx, "target_index": ti, "source_help": vals["view_only"] - vals["with_source"]})
    if parent_rows is not None:
        ppaired: Dict[Tuple[str, int, int], Dict[str, float]] = defaultdict(dict)
        for r in parent_rows:
            ppaired[(str(r["pair_id"]), int(r["row_idx"]), int(r["target_index"]))][r["condition"]] = float(r["nll"])
        for rr in source_help:
            pv = ppaired.get((str(rr["pair_id"]), int(rr["row_idx"]), int(rr["target_index"])), {})
            rr["delta_source_help_vs_parent"] = rr["source_help"] - (pv["view_only"] - pv["with_source"]) if "with_source" in pv and "view_only" in pv else None
    return {
        "by_condition": by_condition,
        "source_help": {
            "n_targets": len(source_help),
            "n_pairs": len(set(r["pair_id"] for r in source_help)),
            "mean_source_help": finite_mean(r["source_help"] for r in source_help),
            "median_source_help": finite_median(r["source_help"] for r in source_help),
            "mean_delta_source_help_vs_parent": finite_mean(r.get("delta_source_help_vs_parent") for r in source_help) if parent_rows is not None else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=DEFAULT_TAIL)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--max-updates", type=int, default=80)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--max-segments", type=int, default=1200)
    ap.add_argument("--max-targets-per-segment", type=int, default=4)
    ap.add_argument("--sample-seed", type=int, default=64064)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--extra-model", action="append", default=[], help="name=path checkpoint to score")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    tasks, task_stats = build_tasks(args.tail_jsonl, tokenizer, args)
    write_jsonl(args.out_dir / "probe_tasks.jsonl", ({k: v for k, v in t.items() if k != "input_ids"} for t in tasks))
    plan = {
        "status": "QWEN_VIEW_SURFACE_PROBE_PLAN",
        "created_utc": now(),
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
        "task_stats": task_stats,
        "scientific_status": "direct trained-material readout for qwen source/current pairs; not semantic transfer by itself",
    }
    (args.out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)

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
    model_summaries: Dict[str, Any] = {}
    load_infos: Dict[str, Any] = {}
    score_paths: Dict[str, str] = {}
    parent_scores: Optional[List[Dict[str, Any]]] = None
    for name, path in model_specs:
        print(json.dumps({"event": "load_model", "model": name, "path": rel(path) if path else rel(bridge.PARENT_PATH)}), flush=True)
        model, load_info = load_scoring_model(path, device, float(args.private_scale))
        load_infos[name] = load_info
        scores = score_tasks(model, tasks, device, tokenizer, int(args.batch_size))
        if name == "parent":
            parent_scores = scores
        out_path = args.out_dir / f"scores_{name}.jsonl"
        write_jsonl(out_path, scores)
        score_paths[name] = rel(out_path)
        model_summaries[name] = summarize_scores(scores, parent_scores)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    final = {
        "status": "QWEN_VIEW_SURFACE_PROBE_DONE",
        "created_utc": now(),
        "plan": rel(args.out_dir / "plan.json"),
        "task_stats": task_stats,
        "score_paths": score_paths,
        "load_infos": load_infos,
        "model_summaries": model_summaries,
        "scientific_status": "Use with common-target and Cheap7: this readout measures direct qwen view acquisition/source-help on the trained unchanged-pair material, not broad BabyLM capability.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
