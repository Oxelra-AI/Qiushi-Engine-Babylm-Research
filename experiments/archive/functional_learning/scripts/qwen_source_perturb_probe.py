#!/usr/bin/env python3
"""research: source-specificity readout for unchanged Qwen pair training.

The research unchanged-tail focus run improved reconstruction NLL on Qwen second-view
content words, but the improvement was almost identical with and without the source
visible.  This probe separates generic source/context help from source-specific use:
for the same masked target word in a Qwen second view, score the model with
(1) the correct source, (2) a deterministic wrong source from another pair that does
not contain the target word when possible, and (3) the view alone.

If weighted source-visible training strengthens reusable source-conditioned behavior,
the correct-vs-wrong source advantage should increase relative to the parent and to
ordinary WWM.  If it mainly teaches second-view surface reconstruction, NLL can improve
without increasing this advantage.
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

import corrected_bridge_trainer as bridge  # noqa: E402
import concentrated_compact_learning_test as s57  # noqa: E402
import qwen_view_surface_probe as qprobe  # noqa: E402

DEFAULT_TAIL = _public_path('experiments/archive/functional_learning/data/unchanged_pair_segments_tail/reference_tail_unchanged_qwen_segments.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/qwen_source_perturb_probe')


def rel(p: pathlib.Path | str | None) -> str | None:
    if p is None:
        return None
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


def finite(vals: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for x in vals:
        if x is None:
            continue
        try:
            xf = float(x)
        except Exception:
            continue
        if math.isfinite(xf):
            out.append(xf)
    return out


def mean(vals: Iterable[Any]) -> Optional[float]:
    xs = finite(vals)
    return sum(xs) / len(xs) if xs else None


def median(vals: Iterable[Any]) -> Optional[float]:
    xs = finite(vals)
    return statistics.median(xs) if xs else None


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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


def build_segment_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    stats = Counter()
    for row_idx, seg_i, _row, seg in iter_prefix_segments(args.tail_jsonl, args.max_updates, args.words_per_update):
        source = str(seg.get("source_text", "")).strip()
        view = str(seg.get("view_text", "")).strip()
        if not source or not view:
            stats["empty_source_or_view"] += 1
            continue
        groups = s57.content_word_spans(view)
        if not groups:
            stats["no_content_groups"] += 1
            continue
        records.append({
            "row_idx": row_idx,
            "segment_index": seg_i,
            "pair_id": str(seg.get("pair_id")),
            "candidate_kind": str(seg.get("candidate_kind", "unknown")),
            "view_kind": str(seg.get("view_kind", "unknown")),
            "source_text": source,
            "view_text": view,
            "source_words": int(seg.get("source_words", wc(source))),
            "view_words": int(seg.get("view_words", wc(view))),
            "groups": groups,
        })
        stats["segments_available"] += 1
    rng = random.Random(int(args.sample_seed))
    if int(args.max_segments) > 0 and len(records) > int(args.max_segments):
        records = rng.sample(records, int(args.max_segments))
        records.sort(key=lambda x: (x["row_idx"], x["segment_index"], x["pair_id"]))
        stats["segments_sampled"] = len(records)
    else:
        stats["segments_sampled"] = len(records)
    return records, dict(stats)


def choose_wrong_source(records: List[Dict[str, Any]], rec_i: int, target_word: str, sample_seed: int) -> Tuple[str, str, str]:
    target_l = target_word.lower()
    cur = records[rec_i]
    candidates = []
    loose = []
    for j, r in enumerate(records):
        if j == rec_i or r["pair_id"] == cur["pair_id"]:
            continue
        loose.append(j)
        if target_l and target_l not in r["source_text"].lower() and target_l not in r["view_text"].lower():
            candidates.append(j)
    pool = candidates or loose
    if not pool:
        return "", "", "no_wrong_source_available"
    rng = random.Random(stable_seed("wrong-source", int(sample_seed), cur["row_idx"], cur["segment_index"], cur["pair_id"], target_word))
    j = pool[rng.randrange(len(pool))]
    reason = "target_absent_wrong_source" if candidates else "fallback_different_pair"
    return records[j]["source_text"], records[j]["pair_id"], reason


def build_tasks(records: List[Dict[str, Any]], tokenizer, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    skips = Counter()
    wrong_reasons = Counter()
    for rec_i, rec in enumerate(records):
        groups = list(rec["groups"])
        grng = random.Random(stable_seed("perturb-targets", int(args.sample_seed), rec["row_idx"], rec["segment_index"], rec["pair_id"]))
        if len(groups) > int(args.max_targets_per_segment):
            groups = grng.sample(groups, int(args.max_targets_per_segment))
            groups.sort(key=lambda x: x[0])
        for gi, (a, b, word) in enumerate(groups):
            wrong_source, wrong_pair, wrong_reason = choose_wrong_source(records, rec_i, word, int(args.sample_seed))
            wrong_reasons[wrong_reason] += 1
            conditions = [
                ("correct_source", rec["source_text"], rec["pair_id"], "correct"),
                ("wrong_source", wrong_source, wrong_pair, wrong_reason),
                ("view_only", "", "", "none"),
            ]
            for condition, source_text, source_pair_id, source_status in conditions:
                if condition == "wrong_source" and not source_text:
                    skips["wrong_source_missing"] += 1
                    continue
                if condition == "view_only":
                    text = rec["view_text"]
                    view_offset = 0
                else:
                    prefix = source_text + " "
                    text = prefix + rec["view_text"]
                    view_offset = len(prefix)
                enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=int(args.max_length), return_offsets_mapping=True)
                ids = [int(x) for x in enc["input_ids"]]
                offsets = [(int(x), int(y)) for x, y in enc["offset_mapping"]]
                pos = locate_positions(offsets, view_offset + a, view_offset + b)
                if not pos:
                    skips[f"{condition}_no_pos"] += 1
                    continue
                tasks.append({
                    "task_id": f"r{rec['row_idx']:06d}_s{rec['segment_index']:02d}_g{gi:02d}_{condition}",
                    "base_task_id": f"r{rec['row_idx']:06d}_s{rec['segment_index']:02d}_g{gi:02d}",
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
                    "n_positions": len(pos),
                    "source_words": rec["source_words"],
                    "view_words": rec["view_words"],
                    "source_pair_id_used": source_pair_id,
                    "source_status": source_status,
                    "wrong_source_has_target_string": bool(condition == "wrong_source" and word.lower() in source_text.lower()),
                })
    pair_keys = set((t["base_task_id"], t["pair_id"], t["target_word"]) for t in tasks)
    return tasks, {
        "skips": dict(skips),
        "wrong_source_reasons": dict(wrong_reasons),
        "n_tasks": len(tasks),
        "n_base_targets": len(set(t["base_task_id"] for t in tasks)),
        "n_pair_target_keys": len(pair_keys),
        "condition_counts": dict(Counter(t["condition"] for t in tasks)),
        "wrong_source_with_target_string": int(sum(1 for t in tasks if t.get("wrong_source_has_target_string"))),
    }


def paired_by_base(score_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in score_rows:
        out[str(r["base_task_id"])][str(r["condition"])] = r
    return out


def summarize(score_rows: List[Dict[str, Any]], parent_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    by_cond: Dict[str, Any] = {}
    parent_by_task = {r["task_id"]: r for r in parent_rows or []}
    for cond in sorted(set(r["condition"] for r in score_rows)):
        g = [r for r in score_rows if r["condition"] == cond]
        by_cond[cond] = {
            "n_targets": len(g),
            "n_pairs": len(set(r["pair_id"] for r in g)),
            "mean_nll": mean(r["nll"] for r in g),
            "median_nll": median(r["nll"] for r in g),
            "mean_delta_nll_vs_parent": mean(float(r["nll"]) - float(parent_by_task[r["task_id"]]["nll"]) for r in g if r["task_id"] in parent_by_task) if parent_rows is not None else None,
        }
    paired = paired_by_base(score_rows)
    parent_paired = paired_by_base(parent_rows or []) if parent_rows is not None else {}
    rows = []
    for base_id, vals in paired.items():
        if not all(c in vals for c in ["correct_source", "wrong_source", "view_only"]):
            continue
        correct = float(vals["correct_source"]["nll"])
        wrong = float(vals["wrong_source"]["nll"])
        view = float(vals["view_only"]["nll"])
        rr = {
            "base_task_id": base_id,
            "pair_id": vals["correct_source"]["pair_id"],
            "target_word": vals["correct_source"]["target_word"],
            "specific_source_advantage": wrong - correct,
            "generic_source_advantage": view - wrong,
            "total_source_advantage": view - correct,
            "correct_source_nll": correct,
            "wrong_source_nll": wrong,
            "view_only_nll": view,
        }
        if parent_rows is not None and base_id in parent_paired and all(c in parent_paired[base_id] for c in ["correct_source", "wrong_source", "view_only"]):
            pc = float(parent_paired[base_id]["correct_source"]["nll"])
            pw = float(parent_paired[base_id]["wrong_source"]["nll"])
            pv = float(parent_paired[base_id]["view_only"]["nll"])
            rr["delta_specific_source_advantage_vs_parent"] = rr["specific_source_advantage"] - (pw - pc)
            rr["delta_generic_source_advantage_vs_parent"] = rr["generic_source_advantage"] - (pv - pw)
            rr["delta_total_source_advantage_vs_parent"] = rr["total_source_advantage"] - (pv - pc)
            rr["delta_correct_source_nll_vs_parent"] = correct - pc
            rr["delta_wrong_source_nll_vs_parent"] = wrong - pw
            rr["delta_view_only_nll_vs_parent"] = view - pv
        rows.append(rr)
    paired_summary = {
        "n_complete_triplets": len(rows),
        "n_pairs": len(set(r["pair_id"] for r in rows)),
        "mean_specific_source_advantage": mean(r["specific_source_advantage"] for r in rows),
        "median_specific_source_advantage": median(r["specific_source_advantage"] for r in rows),
        "mean_generic_source_advantage": mean(r["generic_source_advantage"] for r in rows),
        "mean_total_source_advantage": mean(r["total_source_advantage"] for r in rows),
    }
    if parent_rows is not None:
        paired_summary.update({
            "mean_delta_specific_source_advantage_vs_parent": mean(r.get("delta_specific_source_advantage_vs_parent") for r in rows),
            "median_delta_specific_source_advantage_vs_parent": median(r.get("delta_specific_source_advantage_vs_parent") for r in rows),
            "share_delta_specific_positive_vs_parent": (sum(1 for r in rows if r.get("delta_specific_source_advantage_vs_parent") is not None and r["delta_specific_source_advantage_vs_parent"] > 0) / len(rows)) if rows else None,
            "mean_delta_total_source_advantage_vs_parent": mean(r.get("delta_total_source_advantage_vs_parent") for r in rows),
            "mean_delta_correct_source_nll_vs_parent": mean(r.get("delta_correct_source_nll_vs_parent") for r in rows),
            "mean_delta_wrong_source_nll_vs_parent": mean(r.get("delta_wrong_source_nll_vs_parent") for r in rows),
            "mean_delta_view_only_nll_vs_parent": mean(r.get("delta_view_only_nll_vs_parent") for r in rows),
        })
    return {"by_condition": by_cond, "source_specificity": paired_summary}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=DEFAULT_TAIL)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--max-updates", type=int, default=80)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--max-segments", type=int, default=1200)
    ap.add_argument("--max-targets-per-segment", type=int, default=4)
    ap.add_argument("--sample-seed", type=int, default=67067)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--extra-model", action="append", default=[], help="name=path checkpoint to score")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    records, segment_stats = build_segment_records(args)
    tasks, task_stats = build_tasks(records, tokenizer, args)
    write_jsonl(args.out_dir / "probe_tasks.jsonl", ({k: v for k, v in t.items() if k not in {"input_ids", "positions"}} for t in tasks))
    plan = {
        "status": "QWEN_SOURCE_PERTURB_PROBE_PLAN",
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
        "segment_stats": segment_stats,
        "task_stats": task_stats,
        "scientific_use": "Compare correct-source, wrong-source, and view-only masked reconstruction on Qwen second views to test whether training increased source-specific use rather than only second-view surface fit.",
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

    summaries: Dict[str, Any] = {}
    score_paths: Dict[str, str] = {}
    load_infos: Dict[str, Any] = {}
    parent_scores: Optional[List[Dict[str, Any]]] = None
    for name, path in model_specs:
        print(json.dumps({"event": "load_model", "model": name, "path": rel(path) if path else rel(bridge.PARENT_PATH)}), flush=True)
        model, load_info = qprobe.load_scoring_model(path, device, float(args.private_scale))
        load_infos[name] = load_info
        scores = qprobe.score_tasks(model, tasks, device, tokenizer, int(args.batch_size))
        out_path = args.out_dir / f"scores_{name}.jsonl"
        write_jsonl(out_path, scores)
        score_paths[name] = rel(out_path)
        if name == "parent":
            parent_scores = scores
        summaries[name] = summarize(scores, parent_scores)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    final = {
        "status": "QWEN_SOURCE_PERTURB_PROBE_DONE",
        "created_utc": now(),
        "plan": rel(args.out_dir / "plan.json"),
        "score_paths": score_paths,
        "load_infos": load_infos,
        "model_summaries": summaries,
        "interpretation_scope": "This is a controlled readout on sampled unchanged Qwen second-view targets. It measures source-specific conditioning on trained material; it is not broad BabyLM competence and should be interpreted with Cheap7/common-target results.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
