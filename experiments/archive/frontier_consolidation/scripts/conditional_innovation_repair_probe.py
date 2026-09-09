#!/usr/bin/env python3
"""research repaired conditional-innovation probe.

CPU-only. No model update/evaluation, no corpus/tokenizer change, no H100 work.

Repairs the first research probe along independent review lines:
- stricter target classes: source-absent and unique in the whole row outside the target;
- coarse content-like versus relation/function-like split;
- same-row and cross-row decoys filled to the true source span length without adding extra [MASK]
  tokens when the decoy source is shorter;
- row-bootstrap intervals for the key conditional-help metrics.
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
COND_SCRIPT = WORKSPACE / "scripts/conditional_innovation_probe.py"
OUT_DIR = WORKSPACE / "data/conditional_innovation_repair_probe"
CHECKPOINTS = {
    "tokenmean_80M": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M",
    "tokenmean_100M": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M",
    "clean_80M": WORKSPACE / "training/runs/complianttok_cleanqwen_seed43022_80M/hf_model/chck_80M",
}

STOPWORDS = set("""
a an the and or but if then when while because so for to of in on at by from with without into onto through over under above below after before during as is are was were be been being has have had do does did it its they them their he she his her we our you your this that these those there here not no only all some any every more less same different like than can could should would may might must will shall just also very
""".split())


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cond = load_module("conditional_import_for_repair", COND_SCRIPT)


@dataclass(frozen=True)
class Target:
    target_id: int
    example_id: int
    global_row_1based: int
    pair_id: str
    bucket: str
    norm: str
    first_word: str
    target_positions: tuple[int, ...]
    source_positions: tuple[int, ...]
    same_row_decoy_positions: tuple[int, ...]
    cross_row_decoy_global: int
    cross_row_decoy_positions: tuple[int, ...]
    gold_ids: tuple[int, ...]
    row_group_count: int
    row_norm_count: int
    source_norm_hit: bool


def is_content_like(norm: str) -> bool:
    words = norm.split()
    if not words:
        return False
    w = words[0]
    if w in STOPWORDS or w in cond.RELATION_WORDS:
        return False
    if len(w) < 3:
        return False
    return bool(re.search(r"[a-z]", w))


def fill_to_length(values: list[int], n: int, mask_id: int) -> list[int]:
    if n <= 0:
        return []
    if not values:
        return [mask_id] * n
    return [int(values[i % len(values)]) for i in range(n)]


def all_group_infos(item: dict[str, Any], tokenizer) -> list[dict[str, Any]]:
    input_ids = item["input_ids"]
    word_group = item["word_group"]
    by_gid: dict[int, list[int]] = defaultdict(list)
    first: dict[int, int] = {}
    for pos in range(int(item["attention_mask"].sum().item())):
        gid = int(word_group[pos].item())
        if gid < 0:
            continue
        by_gid[gid].append(pos)
        first.setdefault(gid, pos)
    infos = []
    for gid in sorted(by_gid, key=lambda g: first[g]):
        posns = by_gid[gid]
        ids = [int(input_ids[p].item()) for p in posns]
        n = cond.token_norm(tokenizer, ids)
        if n:
            infos.append({"gid": gid, "positions": tuple(posns), "norm": n, "ids": tuple(ids)})
    return infos


def choose_example_ids(span_by_ex: dict[int, dict[str, Any]], sample_rows: int, seed: int) -> list[int]:
    ids = sorted(span_by_ex)
    if sample_rows <= 0 or sample_rows >= len(ids):
        return ids
    rng = random.Random(seed)
    # 2/3 stride, 1/3 random to cover corpus range while not missing uncommon rows.
    stride_n = int(sample_rows * 2 / 3)
    stride = [ids[round(i * (len(ids) - 1) / max(1, stride_n - 1))] for i in range(stride_n)]
    rem = [x for x in ids if x not in set(stride)]
    rng.shuffle(rem)
    return sorted(stride + rem[: sample_rows - len(stride)])


def build_targets(examples: list[Any], tokenizer, span_by_ex: dict[int, dict[str, Any]], max_per_bucket: int, seed: int) -> tuple[list[Target], dict[str, Any]]:
    ds = cond.collate_mod.ConsistencySpanDataset(examples, tokenizer, span_by_ex, cond.collate_mod.SEQ_LEN)
    row_items = [ds[i] for i in range(len(ds))]
    row_by_global = {int(x["global_row_1based"]): x for x in row_items}
    # all pair source positions for cross-row decoys
    source_position_bank: list[tuple[int, tuple[int, ...]]] = []
    for item in row_items:
        for p in item.get("aux_pairs") or []:
            pos = tuple(cond.ordered_positions(p["source_ranges"]))
            if pos:
                source_position_bank.append((int(item["global_row_1based"]), pos))
    buckets: dict[str, list[Target]] = {"copyable": [], "unique_content_innovation": [], "unique_relation_or_function_innovation": []}
    counts = Counter()
    target_id = 0
    rng = random.Random(seed)
    for item in row_items:
        if item.get("aux_error"):
            counts[f"aux_error_{item['aux_error']}"] += 1
            continue
        pairs = item.get("aux_pairs") or []
        if len(pairs) < 2:
            counts["rows_lt2_pairs"] += 1
            continue
        input_ids = item["input_ids"]
        group_infos = all_group_infos(item, tokenizer)
        row_norm_positions = defaultdict(list)
        for gi in group_infos:
            row_norm_positions[gi["norm"]].append(gi["positions"])
        for pi, p in enumerate(pairs):
            src_groups = cond.group_positions_in_ranges(item["word_group"], p["source_ranges"])
            rew_groups = cond.group_positions_in_ranges(item["word_group"], p["rewrite_ranges"])
            src_norms = set()
            for _, posns in src_groups:
                n = cond.token_norm(tokenizer, [int(input_ids[q].item()) for q in posns])
                if n:
                    src_norms.add(n)
            src_pos = tuple(cond.ordered_positions(p["source_ranges"]))
            if not src_pos:
                continue
            same_decoy_pair = pairs[(pi + 1) % len(pairs)]
            same_decoy_pos = tuple(cond.ordered_positions(same_decoy_pair["source_ranges"]))
            if not same_decoy_pos:
                continue
            # nearest-length cross-row decoy, deterministic among a small random subset for speed.
            candidates = [x for x in source_position_bank if x[0] != int(item["global_row_1based"])]
            # sample 64 candidates, choose closest length
            sample = rng.sample(candidates, min(64, len(candidates))) if candidates else []
            cross_gr, cross_pos = min(sample, key=lambda x: abs(len(x[1]) - len(src_pos))) if sample else (int(item["global_row_1based"]), same_decoy_pos)
            for _, posns in rew_groups:
                ids = [int(input_ids[q].item()) for q in posns]
                n = cond.token_norm(tokenizer, ids)
                if not n or len(n) <= 1:
                    counts["skip_empty_or_single"] += 1
                    continue
                source_hit = n in src_norms
                other_occurs = any(tuple(o) != tuple(posns) for o in row_norm_positions.get(n, []))
                first = n.split()[0] if n.split() else n
                if source_hit:
                    bucket = "copyable"
                elif not other_occurs and is_content_like(n):
                    bucket = "unique_content_innovation"
                elif not other_occurs:
                    bucket = "unique_relation_or_function_innovation"
                else:
                    counts["loose_innovation_or_duplicate_not_selected"] += 1
                    continue
                buckets[bucket].append(Target(
                    target_id=target_id,
                    example_id=int(item["example_id"]),
                    global_row_1based=int(item["global_row_1based"]),
                    pair_id=str(p.get("pair_id", "")),
                    bucket=bucket,
                    norm=n,
                    first_word=first,
                    target_positions=tuple(int(q) for q in posns),
                    source_positions=tuple(int(q) for q in src_pos),
                    same_row_decoy_positions=tuple(int(q) for q in same_decoy_pos),
                    cross_row_decoy_global=int(cross_gr),
                    cross_row_decoy_positions=tuple(int(q) for q in cross_pos),
                    gold_ids=tuple(ids),
                    row_group_count=len(group_infos),
                    row_norm_count=len(row_norm_positions.get(n, [])),
                    source_norm_hit=source_hit,
                ))
                target_id += 1
    selected: list[Target] = []
    for b, arr in buckets.items():
        rng.shuffle(arr)
        selected.extend(arr[:max_per_bucket])
    selected.sort(key=lambda t: (t.global_row_1based, t.pair_id, t.target_positions))
    selected = [Target(i, t.example_id, t.global_row_1based, t.pair_id, t.bucket, t.norm, t.first_word, t.target_positions, t.source_positions, t.same_row_decoy_positions, t.cross_row_decoy_global, t.cross_row_decoy_positions, t.gold_ids, t.row_group_count, t.row_norm_count, t.source_norm_hit) for i, t in enumerate(selected)]
    return selected, {
        "candidate_counts": {k: len(v) for k, v in buckets.items()},
        "selected_counts": dict(Counter(t.bucket for t in selected)),
        "skips": dict(counts),
        "n_source_position_bank": len(source_position_bank),
    }


def make_rows(examples: list[Any], tokenizer, span_by_ex: dict[int, dict[str, Any]], targets: list[Target]) -> list[dict[str, Any]]:
    ds = cond.collate_mod.ConsistencySpanDataset(examples, tokenizer, span_by_ex, cond.collate_mod.SEQ_LEN)
    item_by_global = {int(ds[i]["global_row_1based"]): ds[i] for i in range(len(ds))}
    base_ids = {gr: item["input_ids"] for gr, item in item_by_global.items()}
    attn = {gr: item["attention_mask"] for gr, item in item_by_global.items()}
    mask_id = int(tokenizer.mask_token_id)
    rows = []
    scenarios = ["true_source", "source_masked", "same_row_decoy_filled", "cross_row_decoy_filled"]
    for t in targets:
        ids0 = base_ids[t.global_row_1based]
        for sc in scenarios:
            ids = ids0.clone()
            if sc == "source_masked":
                for p in t.source_positions:
                    ids[p] = mask_id
            elif sc == "same_row_decoy_filled":
                vals = [int(ids0[p].item()) for p in t.same_row_decoy_positions]
                for p, v in zip(t.source_positions, fill_to_length(vals, len(t.source_positions), mask_id)):
                    ids[p] = v
            elif sc == "cross_row_decoy_filled":
                cross_ids = base_ids.get(t.cross_row_decoy_global, ids0)
                vals = [int(cross_ids[p].item()) for p in t.cross_row_decoy_positions]
                for p, v in zip(t.source_positions, fill_to_length(vals, len(t.source_positions), mask_id)):
                    ids[p] = v
            elif sc == "true_source":
                pass
            else:
                raise ValueError(sc)
            for p in t.target_positions:
                ids[p] = mask_id
            rows.append({
                "target_id": t.target_id,
                "scenario": sc,
                "bucket": t.bucket,
                "global_row_1based": t.global_row_1based,
                "input_ids": ids,
                "attention_mask": attn[t.global_row_1based],
                "target_positions": t.target_positions,
                "gold_ids": t.gold_ids,
                "norm": t.norm,
            })
    return rows


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
        return float(s[lo]) if lo == hi else float(s[lo] * (hi - pos) + s[hi] * (pos - lo))
    return {"n": len(vals), "mean": float(statistics.mean(vals)), "median": float(statistics.median(vals)), "std": float(statistics.pstdev(vals)), "p05": pct(0.05), "p95": pct(0.95)}


def bootstrap_mean(rows: list[dict[str, Any]], metric: str, iters: int, seed: int) -> dict[str, Any]:
    by_row = defaultdict(list)
    for r in rows:
        by_row[int(r["global_row_1based"])].append(float(r[metric]))
    keys = list(by_row)
    if not keys:
        return {"mean": None, "ci05": None, "ci95": None, "rows": 0}
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        sample_keys = [rng.choice(keys) for _ in keys]
        xs = [x for k in sample_keys for x in by_row[k]]
        vals.append(statistics.mean(xs))
    vals.sort()
    return {"mean": statistics.mean([float(r[metric]) for r in rows]), "ci05": vals[int(0.05 * (len(vals)-1))], "ci95": vals[int(0.95 * (len(vals)-1))], "rows": len(keys)}


def eval_checkpoint(label: str, ckpt: pathlib.Path, rows: list[dict[str, Any]], batch_size: int, threads: int, boot_iters: int, seed: int) -> dict[str, Any]:
    if not ckpt.exists():
        return {"label": label, "status": "missing", "path": str(ckpt)}
    if threads > 0:
        torch.set_num_threads(threads)
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt))
    model.eval(); model.to("cpu")
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    loss = {}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start+batch_size]
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attn = torch.stack([x["attention_mask"] for x in batch])
            logits = model(input_ids=input_ids, attention_mask=attn, return_dict=True).logits
            for bi, item in enumerate(batch):
                pos = torch.tensor(item["target_positions"], dtype=torch.long)
                gold = torch.tensor(item["gold_ids"], dtype=torch.long)
                loss[(int(item["target_id"]), str(item["scenario"]))] = float(ce(logits[bi, pos, :], gold).mean().item())
            del logits
    del model
    meta = {}
    for item in rows:
        tid = int(item["target_id"])
        meta.setdefault(tid, {"bucket": item["bucket"], "global_row_1based": int(item["global_row_1based"]), "norm": item["norm"]})
    per = []
    for tid, m in meta.items():
        true = loss.get((tid, "true_source")); masked = loss.get((tid, "source_masked"))
        same = loss.get((tid, "same_row_decoy_filled")); cross = loss.get((tid, "cross_row_decoy_filled"))
        if None in (true, masked, same, cross):
            continue
        per.append({
            "target_id": tid,
            "bucket": m["bucket"],
            "global_row_1based": m["global_row_1based"],
            "norm": m["norm"],
            "true_loss": true,
            "source_help": masked - true,
            "same_decoy_advantage": same - true,
            "cross_decoy_advantage": cross - true,
            "masked_minus_same_decoy": masked - same,
            "masked_minus_cross_decoy": masked - cross,
        })
    summary = {}
    for bucket in ["all", "copyable", "unique_content_innovation", "unique_relation_or_function_innovation"]:
        sub = per if bucket == "all" else [r for r in per if r["bucket"] == bucket]
        for metric in ["true_loss", "source_help", "same_decoy_advantage", "cross_decoy_advantage", "masked_minus_same_decoy", "masked_minus_cross_decoy"]:
            summary[f"{bucket}|{metric}"] = summarize([r[metric] for r in sub])
        summary[f"{bucket}|bootstrap_source_help"] = bootstrap_mean(sub, "source_help", boot_iters, seed + len(summary)) if sub else {"mean": None}
        summary[f"{bucket}|bootstrap_same_decoy_advantage"] = bootstrap_mean(sub, "same_decoy_advantage", boot_iters, seed + len(summary) + 1) if sub else {"mean": None}
    return {"label": label, "status": "ok", "path": str(ckpt), "n_targets": len(meta), "summary": summary, "per_target_rows": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--sample-rows", type=int, default=768)
    ap.add_argument("--max-per-bucket", type=int, default=220)
    ap.add_argument("--sample-seed", type=int, default=73373)
    ap.add_argument("--target-seed", type=int, default=73473)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--checkpoints", default="tokenmean_80M,tokenmean_100M,clean_80M")
    ap.add_argument("--bootstrap-iters", type=int, default=300)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_sha = cond.collate_mod.sha256_file(cond.collate_mod.TRAIN_100M)
    tok_sha = cond.collate_mod.sha256_file(cond.collate_mod.TOKENIZER_DIR / "tokenizer.json")
    if train_sha != cond.collate_mod.EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch {train_sha}")
    if tok_sha != cond.collate_mod.EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch {tok_sha}")
    span_by_ex = cond.collate_mod.load_span_map(cond.collate_mod.SPAN_JSONL)
    example_ids = choose_example_ids(span_by_ex, args.sample_rows, args.sample_seed)
    examples = cond.read_examples_by_example_id(cond.collate_mod.TRAIN_100M, example_ids)
    tokenizer = AutoTokenizer.from_pretrained(str(cond.collate_mod.TOKENIZER_DIR), use_fast=True)
    targets, build = build_targets(examples, tokenizer, span_by_ex, args.max_per_bucket, args.target_seed)
    rows = make_rows(examples, tokenizer, span_by_ex, targets)
    results = []
    for label in [x.strip() for x in args.checkpoints.split(",") if x.strip()]:
        if label not in CHECKPOINTS:
            raise ValueError(label)
        print(json.dumps({"event": "checkpoint_start", "label": label, "targets": len(targets), "eval_sequences": len(rows), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
        r = eval_checkpoint(label, CHECKPOINTS[label], rows, args.batch_size, args.torch_threads, args.bootstrap_iters, args.target_seed)
        per = r.pop("per_target_rows", []) if r.get("status") == "ok" else []
        if per:
            p = out_dir / f"per_target_{label}.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for row in per:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            r["per_target_jsonl"] = str(p)
        results.append(r)
        print(json.dumps({"event": "checkpoint_done", "label": label, "status": r.get("status")}), flush=True)
    summary = {
        "status": "CONDITIONAL_INNOVATION_REPAIR_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Stricter leakage-aware conditional-source probe using unique source-absent targets and no extra mask tokens in decoy source fills.",
        "inputs": {"train_sha256": train_sha, "tokenizer_sha256": tok_sha, "sample_rows_loaded": len(examples), "selected_targets": len(targets), "target_build": build, "checkpoints": [x.strip() for x in args.checkpoints.split(",") if x.strip()]},
        "checkpoint_results": results,
        "scientific_reading": [
            "Unique innovation buckets exclude target norms seen in the paired source or elsewhere in the same row; this reduces exact-form leakage but does not prove semantic novelty.",
            "Filled decoys remove the previous extra-[MASK] artifact for short decoys, though any source substitution remains a grammatical corruption control rather than a perfect semantic negative.",
            "If unique_content_innovation keeps positive source_help and same_decoy_advantage with bootstrap intervals away from zero, the conditional-source signal is not only copyability or duplicated function-word entropy.",
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "conditional_innovation_repair_probe.json"
    out_md = out_dir / "conditional_innovation_repair_probe.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research — repaired conditional-innovation probe", "", "CPU-only; no model update/evaluation, no corpus/tokenizer change, no H100 work.", "", summary["purpose"], "", f"- changed rows loaded: `{len(examples)}`", f"- selected targets: `{len(targets)}`; counts: `{build['selected_counts']}`", f"- train SHA: `{train_sha}`", f"- tokenizer SHA: `{tok_sha}`", "", "## Results", "| checkpoint | bucket | n | true loss | source help | same-row decoy advantage | cross-row decoy advantage | masked-same | source help bootstrap 5-95 | same decoy bootstrap 5-95 |", "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"| {r.get('label')} | status {r.get('status')} | | | | | | | | |")
            continue
        for bucket in ["all", "copyable", "unique_content_innovation", "unique_relation_or_function_innovation"]:
            s = r["summary"]
            def mean(metric):
                return (s.get(f"{bucket}|{metric}") or {}).get("mean")
            def nval():
                return (s.get(f"{bucket}|true_loss") or {}).get("n")
            def fmt(x):
                return "" if x is None else f"{float(x):.4f}"
            b1 = s.get(f"{bucket}|bootstrap_source_help") or {}
            b2 = s.get(f"{bucket}|bootstrap_same_decoy_advantage") or {}
            ci1 = "" if b1.get("ci05") is None else f"[{b1['ci05']:.4f},{b1['ci95']:.4f}]"
            ci2 = "" if b2.get("ci05") is None else f"[{b2['ci05']:.4f},{b2['ci95']:.4f}]"
            lines.append(f"| {r['label']} | {bucket} | {nval()} | {fmt(mean('true_loss'))} | {fmt(mean('source_help'))} | {fmt(mean('same_decoy_advantage'))} | {fmt(mean('cross_decoy_advantage'))} | {fmt(mean('masked_minus_same_decoy'))} | {ci1} | {ci2} |")
    lines += ["", "## Reading"]
    for x in summary["scientific_reading"]:
        lines.append(f"- {x}")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "targets": len(targets), "checkpoint_status": {r.get("label"): r.get("status") for r in results}, "elapsed_sec": summary["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
