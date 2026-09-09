#!/usr/bin/env python3
"""research: conditional innovation probe for compact source/rewrite pairs.

CPU-only. No model update, no official evaluation, no corpus/tokenizer change, no H100 work.

The goal is to measure whether the validated compact-view mechanism contains an
unsaturated quantity beyond copyable tokens and already-solved pair identity:
when a rewrite word is masked, does the true source span help predict it more than
(1) masking the source span or (2) replacing the source span with a same-row decoy
source from another pair?  Results are stratified into rewrite groups whose normalized
word is present in the source (copyable) versus absent (innovation).
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import math
import pathlib
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
COLLATE_SCRIPT = WORKSPACE / "scripts/consistency_collate_smoke.py"
OUT_DIR = WORKSPACE / "data/conditional_innovation_probe"

CHECKPOINTS = {
    "tokenmean_20M": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_20M",
    "tokenmean_80M": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M",
    "tokenmean_100M": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M",
    "clean_80M": WORKSPACE / "training/runs/complianttok_cleanqwen_seed43022_80M/hf_model/chck_80M",
}

RELATION_WORDS = {
    "in", "on", "under", "over", "above", "below", "behind", "beside", "near", "inside", "outside",
    "into", "onto", "between", "through", "across", "around", "from", "to", "with", "without", "before",
    "after", "during", "while", "when", "because", "therefore", "then", "if", "unless", "although",
    "but", "not", "no", "never", "only", "all", "some", "every", "any", "more", "less", "same", "different",
    "cause", "causes", "caused", "make", "makes", "made", "move", "moves", "moved", "put", "puts", "placed",
    "go", "goes", "went", "fall", "falls", "fell", "open", "opens", "closed", "break", "breaks", "broke",
}


@dataclass(frozen=True)
class Target:
    target_id: int
    example_id: int
    global_row_1based: int
    pair_id: str
    category: str
    cue_class: str
    norm: str
    source_norm_hit: bool
    target_positions: tuple[int, ...]
    source_positions: tuple[int, ...]
    decoy_source_positions: tuple[int, ...]
    gold_ids: tuple[int, ...]
    row_words: int


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


collate_mod = load_module("consistency_collate_step073", COLLATE_SCRIPT)


def norm_text(text: str) -> str:
    text = text.lower().replace("Ġ", " ").replace("▁", " ")
    pieces = re.findall(r"[a-z0-9]+", text)
    return " ".join(pieces)


def token_norm(tokenizer, ids: list[int]) -> str:
    if not ids:
        return ""
    return norm_text(tokenizer.decode([int(x) for x in ids], clean_up_tokenization_spaces=False))


def ordered_positions(ranges: list[list[int]]) -> list[int]:
    out: list[int] = []
    for a, b in ranges:
        out.extend(range(int(a), int(b)))
    return out


def group_positions_in_ranges(word_group: torch.Tensor, ranges: list[list[int]]) -> list[tuple[int, list[int]]]:
    by_gid: dict[int, list[int]] = defaultdict(list)
    first: dict[int, int] = {}
    for pos in ordered_positions(ranges):
        gid = int(word_group[pos].item())
        if gid < 0:
            continue
        by_gid[gid].append(pos)
        first.setdefault(gid, pos)
    return [(gid, by_gid[gid]) for gid in sorted(by_gid, key=lambda g: first[g])]


def choose_example_ids(span_by_ex: dict[int, dict[str, Any]], sample_rows: int, seed: int) -> list[int]:
    ids = sorted(int(k) for k in span_by_ex.keys())
    if sample_rows <= 0 or sample_rows >= len(ids):
        return ids
    # Interleave stride and random sampling so broad corpus positions and rare row shapes appear.
    stride_n = sample_rows // 2
    stride_ids = [ids[round(i * (len(ids) - 1) / max(1, stride_n - 1))] for i in range(stride_n)] if stride_n else []
    rng = random.Random(seed)
    rest = [x for x in ids if x not in set(stride_ids)]
    rng.shuffle(rest)
    return sorted(stride_ids + rest[:sample_rows - len(stride_ids)])


def read_examples_by_example_id(path: pathlib.Path, example_ids: list[int]) -> list[Any]:
    wanted = set(int(x) for x in example_ids)
    found: dict[int, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for row_1, line in enumerate(f, 1):
            obj = json.loads(line)
            exid = int(obj.get("example_id", row_1 - 1))
            if exid not in wanted or exid in found:
                continue
            if str(obj.get("source", "")) != collate_mod.CHANGED_SOURCE:
                continue
            text = str(obj["text"])
            found[exid] = collate_mod.StreamExample(
                text=text,
                words=int(obj.get("words", len(text.split()))),
                example_id=exid,
                source=str(obj.get("source", "")),
                global_row_1based=row_1,
            )
            if len(found) >= len(wanted):
                break
    examples = [found[eid] for eid in example_ids if eid in found]
    examples.sort(key=lambda x: x.global_row_1based)
    return examples


def build_targets(examples: list[Any], tokenizer, span_by_ex: dict[int, dict[str, Any]], max_per_bucket: int, seed: int) -> tuple[list[Target], dict[str, Any]]:
    ds = collate_mod.ConsistencySpanDataset(examples, tokenizer, span_by_ex, collate_mod.SEQ_LEN)
    candidates: dict[str, list[Target]] = {"innovation": [], "copyable": []}
    counts = Counter()
    target_id = 0
    for idx in range(len(ds)):
        item = ds[idx]
        if item.get("aux_error"):
            counts[f"aux_error_{item['aux_error']}"] += 1
            continue
        pairs = item.get("aux_pairs") or []
        if len(pairs) < 2:
            counts["rows_with_lt2_pairs"] += 1
            continue
        input_ids: torch.Tensor = item["input_ids"]
        word_group: torch.Tensor = item["word_group"]
        for p_i, p in enumerate(pairs):
            src_groups = group_positions_in_ranges(word_group, p["source_ranges"])
            rew_groups = group_positions_in_ranges(word_group, p["rewrite_ranges"])
            src_norms = set()
            src_positions = tuple(ordered_positions(p["source_ranges"]))
            if not src_positions:
                counts["pair_no_source_positions"] += 1
                continue
            for _, posns in src_groups:
                n = token_norm(tokenizer, [int(input_ids[q].item()) for q in posns])
                if n:
                    src_norms.add(n)
            decoy_pair = pairs[(p_i + 1) % len(pairs)]
            if decoy_pair.get("pair_id") == p.get("pair_id") and len(pairs) > 1:
                decoy_pair = pairs[(p_i + 2) % len(pairs)]
            decoy_source_positions = tuple(ordered_positions(decoy_pair["source_ranges"]))
            if not decoy_source_positions:
                counts["pair_no_decoy_source_positions"] += 1
                continue
            for _, posns in rew_groups:
                ids = [int(input_ids[q].item()) for q in posns]
                if any(x in tokenizer.all_special_ids for x in ids):
                    continue
                n = token_norm(tokenizer, ids)
                if not n or len(n) <= 1:
                    counts["skip_empty_or_single"] += 1
                    continue
                source_hit = n in src_norms
                category = "copyable" if source_hit else "innovation"
                first_word = n.split()[0] if n.split() else n
                cue_class = "relation_cue" if first_word in RELATION_WORDS else "content_or_other"
                candidates[category].append(Target(
                    target_id=target_id,
                    example_id=int(item["example_id"]),
                    global_row_1based=int(item["global_row_1based"]),
                    pair_id=str(p.get("pair_id", "")),
                    category=category,
                    cue_class=cue_class,
                    norm=n,
                    source_norm_hit=source_hit,
                    target_positions=tuple(int(q) for q in posns),
                    source_positions=tuple(int(q) for q in src_positions),
                    decoy_source_positions=tuple(int(q) for q in decoy_source_positions),
                    gold_ids=tuple(ids),
                    row_words=int(item["words"]),
                ))
                target_id += 1
    rng = random.Random(seed)
    selected: list[Target] = []
    for bucket in ["innovation", "copyable"]:
        arr = candidates[bucket]
        rng.shuffle(arr)
        selected.extend(arr[:max_per_bucket])
    selected.sort(key=lambda t: (t.global_row_1based, t.pair_id, t.target_positions))
    # Reassign compact consecutive target ids for output records.
    selected2 = []
    for i, t in enumerate(selected):
        selected2.append(Target(i, t.example_id, t.global_row_1based, t.pair_id, t.category, t.cue_class, t.norm,
                                t.source_norm_hit, t.target_positions, t.source_positions, t.decoy_source_positions,
                                t.gold_ids, t.row_words))
    build_summary = {
        "candidate_counts": {k: len(v) for k, v in candidates.items()},
        "selected_counts": dict(Counter(t.category for t in selected2)),
        "selected_cue_counts": dict(Counter(f"{t.category}:{t.cue_class}" for t in selected2)),
        "skips": dict(counts),
        "examples": len(examples),
    }
    return selected2, build_summary


def make_scenario_input(base_ids: torch.Tensor, target: Target, mask_id: int, scenario: str) -> torch.Tensor:
    ids = base_ids.clone()
    if scenario == "true_source":
        pass
    elif scenario == "source_masked":
        for p in target.source_positions:
            ids[p] = mask_id
    elif scenario == "same_row_decoy_source":
        decoy_vals = [int(base_ids[p].item()) for p in target.decoy_source_positions]
        for k, p in enumerate(target.source_positions):
            ids[p] = decoy_vals[k] if k < len(decoy_vals) else mask_id
    else:
        raise ValueError(scenario)
    for p in target.target_positions:
        ids[p] = mask_id
    return ids


def make_eval_examples(examples: list[Any], tokenizer, span_by_ex: dict[int, dict[str, Any]], targets: list[Target], scenarios: list[str]) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    ds = collate_mod.ConsistencySpanDataset(examples, tokenizer, span_by_ex, collate_mod.SEQ_LEN)
    row_by_global: dict[int, dict[str, Any]] = {}
    for idx in range(len(ds)):
        item = ds[idx]
        row_by_global[int(item["global_row_1based"])] = item
    mask_id = int(tokenizer.mask_token_id)
    base_ids_by_global = {gr: item["input_ids"] for gr, item in row_by_global.items()}
    attn_by_global = {gr: item["attention_mask"] for gr, item in row_by_global.items()}
    rows = []
    for t in targets:
        base_ids = base_ids_by_global[t.global_row_1based]
        for sc in scenarios:
            rows.append({
                "target_id": t.target_id,
                "scenario": sc,
                "input_ids": make_scenario_input(base_ids, t, mask_id, sc),
                "attention_mask": attn_by_global[t.global_row_1based],
                "target_positions": t.target_positions,
                "gold_ids": t.gold_ids,
                "category": t.category,
                "cue_class": t.cue_class,
            })
    return rows, base_ids_by_global, attn_by_global


def summarize(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "std": None, "p05": None, "p95": None}
    s = sorted(vals)
    def pct(q: float) -> float:
        if len(s) == 1:
            return float(s[0])
        pos = q * (len(s) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return float(s[lo])
        return float(s[lo] * (hi - pos) + s[hi] * (pos - lo))
    return {
        "n": len(vals),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "std": float(statistics.pstdev(vals)),
        "p05": pct(0.05),
        "p95": pct(0.95),
    }


def evaluate_checkpoint(label: str, ckpt: pathlib.Path, eval_rows: list[dict[str, Any]], batch_size: int, torch_threads: int) -> dict[str, Any]:
    if not ckpt.exists():
        return {"label": label, "status": "missing", "path": str(ckpt)}
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt))
    model.eval(); model.to("cpu")
    losses_by_target_scenario: dict[tuple[int, str], float] = {}
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for start in range(0, len(eval_rows), batch_size):
            batch = eval_rows[start:start + batch_size]
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attn = torch.stack([x["attention_mask"] for x in batch])
            logits = model(input_ids=input_ids, attention_mask=attn, return_dict=True).logits
            for bi, item in enumerate(batch):
                pos = torch.tensor(item["target_positions"], dtype=torch.long)
                gold = torch.tensor(item["gold_ids"], dtype=torch.long)
                l = ce(logits[bi, pos, :], gold).mean().item()
                losses_by_target_scenario[(int(item["target_id"]), str(item["scenario"]))] = float(l)
            del logits
    del model
    grouped: dict[str, list[float]] = defaultdict(list)
    target_meta: dict[int, dict[str, Any]] = {}
    for item in eval_rows:
        tid = int(item["target_id"])
        target_meta.setdefault(tid, {"category": item["category"], "cue_class": item["cue_class"]})
    per_target_rows = []
    for tid, meta in target_meta.items():
        true = losses_by_target_scenario.get((tid, "true_source"))
        smask = losses_by_target_scenario.get((tid, "source_masked"))
        decoy = losses_by_target_scenario.get((tid, "same_row_decoy_source"))
        if true is None or smask is None or decoy is None:
            continue
        source_help = smask - true
        true_over_decoy = decoy - true
        generic_source_help = smask - decoy
        keys = ["all", meta["category"], meta["cue_class"], f"{meta['category']}:{meta['cue_class']}"]
        for key in keys:
            grouped[f"{key}|true_loss"].append(true)
            grouped[f"{key}|source_masked_loss"].append(smask)
            grouped[f"{key}|decoy_loss"].append(decoy)
            grouped[f"{key}|source_help"].append(source_help)
            grouped[f"{key}|true_over_decoy"].append(true_over_decoy)
            grouped[f"{key}|generic_source_help"].append(generic_source_help)
        per_target_rows.append({
            "target_id": tid,
            "category": meta["category"],
            "cue_class": meta["cue_class"],
            "true_loss": true,
            "source_masked_loss": smask,
            "same_row_decoy_source_loss": decoy,
            "source_help": source_help,
            "true_over_decoy": true_over_decoy,
            "generic_source_help": generic_source_help,
        })
    summary = {k: summarize(v) for k, v in sorted(grouped.items())}
    return {
        "label": label,
        "status": "ok",
        "path": str(ckpt),
        "n_eval_sequences": len(eval_rows),
        "n_targets": len(target_meta),
        "summary": summary,
        "per_target_rows": per_target_rows,
    }


def compact_checkpoint_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        if r.get("status") != "ok":
            continue
        s = r["summary"]
        for key in ["all", "innovation", "copyable", "innovation:relation_cue", "innovation:content_or_other"]:
            rows.append({
                "checkpoint": r["label"],
                "group": key,
                "n": s.get(f"{key}|true_loss", {}).get("n"),
                "true_loss": s.get(f"{key}|true_loss", {}).get("mean"),
                "source_help": s.get(f"{key}|source_help", {}).get("mean"),
                "true_over_decoy": s.get(f"{key}|true_over_decoy", {}).get("mean"),
                "generic_source_help": s.get(f"{key}|generic_source_help", {}).get("mean"),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--sample-rows", type=int, default=512)
    ap.add_argument("--sample-seed", type=int, default=73073)
    ap.add_argument("--max-per-bucket", type=int, default=300)
    ap.add_argument("--target-seed", type=int, default=73173)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--checkpoints", default="tokenmean_20M,tokenmean_80M,tokenmean_100M,clean_80M")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_sha = collate_mod.sha256_file(collate_mod.TRAIN_100M)
    tok_sha = collate_mod.sha256_file(collate_mod.TOKENIZER_DIR / "tokenizer.json")
    if train_sha != collate_mod.EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch: {train_sha}")
    if tok_sha != collate_mod.EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    span_by_ex = collate_mod.load_span_map(collate_mod.SPAN_JSONL)
    example_ids = choose_example_ids(span_by_ex, args.sample_rows, args.sample_seed)
    examples = read_examples_by_example_id(collate_mod.TRAIN_100M, example_ids)
    tokenizer = AutoTokenizer.from_pretrained(str(collate_mod.TOKENIZER_DIR), use_fast=True)
    targets, build_summary = build_targets(examples, tokenizer, span_by_ex, args.max_per_bucket, args.target_seed)
    scenarios = ["true_source", "source_masked", "same_row_decoy_source"]
    eval_rows, _, _ = make_eval_examples(examples, tokenizer, span_by_ex, targets, scenarios)
    ckpt_labels = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    results = []
    for label in ckpt_labels:
        if label not in CHECKPOINTS:
            raise ValueError(f"unknown checkpoint {label}")
        print(json.dumps({"event": "checkpoint_start", "label": label, "targets": len(targets), "eval_sequences": len(eval_rows), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
        r = evaluate_checkpoint(label, CHECKPOINTS[label], eval_rows, args.batch_size, args.torch_threads)
        # Drop bulky per-target rows from main JSON into separate file.
        rows = r.pop("per_target_rows", []) if r.get("status") == "ok" else []
        if rows:
            row_path = out_dir / f"per_target_{label}.jsonl"
            with row_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            r["per_target_jsonl"] = str(row_path)
        results.append(r)
        print(json.dumps({"event": "checkpoint_done", "label": label, "status": r.get("status")}), flush=True)
    table = compact_checkpoint_table(results)
    summary = {
        "status": "CONDITIONAL_INNOVATION_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Measure true-source conditional help on masked rewrite groups, separating copyable words from source-absent innovation words and same-row decoy identity.",
        "inputs": {
            "train_100m": str(collate_mod.TRAIN_100M),
            "train_sha256": train_sha,
            "tokenizer": str(collate_mod.TOKENIZER_DIR),
            "tokenizer_sha256": tok_sha,
            "span_map": str(collate_mod.SPAN_JSONL),
            "span_examples": len(span_by_ex),
            "sample_rows_requested": args.sample_rows,
            "sample_rows_loaded": len(examples),
            "max_per_bucket": args.max_per_bucket,
            "selected_targets": len(targets),
            "scenarios": scenarios,
            "checkpoints": ckpt_labels,
            "batch_size": args.batch_size,
        },
        "target_build": build_summary,
        "checkpoint_results": results,
        "compact_table": table,
        "scientific_reading": [
            "source_help = loss(source masked) - loss(true source); positive means the real source span helps predict the masked rewrite group.",
            "true_over_decoy = loss(same-row decoy source) - loss(true source); positive means the help is source-specific rather than generic same-row pair identity.",
            "If innovation groups have positive true_over_decoy and remain high-loss at mature checkpoints, conditional innovation is a credible unsaturated route; if only copyable groups show source help, a training objective would mostly teach copying.",
            "This is not an official score result. It informs the choice between a small single-variable objective and a contextual-computation test.",
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "conditional_innovation_probe.json"
    out_md = out_dir / "conditional_innovation_probe.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — conditional innovation in compact source/rewrite pairs",
        "",
        "CPU-only; no model update, no official evaluation, no corpus/tokenizer change, no H100 work.",
        "",
        "## What was measured",
        "For masked rewrite word groups, compare three contexts: true source span visible, source span masked, and source span replaced by a same-row decoy source from another pair. Groups are split into `copyable` if the normalized rewrite group occurs in the source span and `innovation` otherwise.",
        "",
        f"- changed rows loaded: `{len(examples)}` of `{len(span_by_ex)}`",
        f"- selected targets: `{len(targets)}`; counts: `{build_summary['selected_counts']}`; cue counts: `{build_summary['selected_cue_counts']}`",
        f"- train SHA: `{train_sha}`",
        f"- tokenizer SHA: `{tok_sha}`",
        "",
        "## Compact table",
        "| checkpoint | group | n | true loss | source help | true over decoy | masked over decoy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        def fmt(x):
            return "" if x is None else f"{float(x):.4f}"
        lines.append(f"| {row['checkpoint']} | {row['group']} | {row['n']} | {fmt(row['true_loss'])} | {fmt(row['source_help'])} | {fmt(row['true_over_decoy'])} | {fmt(row['generic_source_help'])} |")
    lines += ["", "## Reading"]
    for item in summary["scientific_reading"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"Full JSON: `{out_json}`")
    lines.append(f"Per-target rows: `{out_dir}/per_target_<checkpoint>.jsonl`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "targets": len(targets),
        "checkpoint_status": {r.get("label"): r.get("status") for r in results},
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
