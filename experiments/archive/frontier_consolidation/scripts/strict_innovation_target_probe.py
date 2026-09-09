#!/usr/bin/env python3
"""research: target-prediction readout for strict row-unique content innovations.

CPU-only. This probe tests the following criterion: after the
research strict p_strict/p_copy run produces 70M/80M checkpoints, compare broad
BabyLM transfer with direct prediction of the same repaired row-unique
conditional-innovation targets. If broad transfer weakens and these targets do
not improve, do not carry the innovation-masking family forward merely because a
mechanically cleaner exact-swap variant exists.

The target population matches research strict map / repaired research logic:
  - changed-row compact source/rewrite pair with both spans visible;
  - rewrite word-group normalized form absent from its paired source;
  - normalized form absent from every other word group in the row;
  - content-like under the research stopword/relation filter.

For each selected target we score masked rewrite-token loss under true source,
masked source, same-row source-decoy fill, and cross-row source-decoy fill.
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
import math
import pathlib
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
POOL_10M = WS / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
TRAIN_100M = WS / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
SPAN_JSONL = WS / "data/pair_span_map/pair_span_map.jsonl"
TOKENIZER_DIR = WS / "data/compliant_tokenizer"
MAP_JSON = WS / "data/strict_innovation_group_map/strict_innovation_group_map.json"
COLLATE_SCRIPT = WS / "scripts/consistency_collate_smoke.py"
OUT_DIR_DEFAULT = WS / "data/strict_innovation_target_probe"

EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TRAIN_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_TOK_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
SEQ_LEN = 256
SCENARIOS = ["true_source", "source_masked", "same_row_decoy_filled", "cross_row_decoy_filled"]

CHECKPOINTS = {
    "tokenmean_70M": WS / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_70M",
    "tokenmean_80M": WS / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M",
    "tokenmean_100M": WS / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M",
    "clean_80M": WS / "training/runs/complianttok_cleanqwen_seed43022_80M/hf_model/chck_80M",
    "reference_70M": WS / "training/runs/strict_content_innovation_wwm_reinvest_seed43022_80M/hf_model/chck_70M",
    "reference_80M": WS / "training/runs/strict_content_innovation_wwm_reinvest_seed43022_80M/hf_model/chck_80M",
}

RELATION_WORDS = {
    "in", "on", "under", "over", "above", "below", "behind", "beside", "near", "inside", "outside",
    "into", "onto", "between", "through", "across", "around", "from", "to", "with", "without", "before",
    "after", "during", "while", "when", "because", "therefore", "then", "if", "unless", "although",
    "but", "not", "no", "never", "only", "all", "some", "every", "any", "more", "less", "same", "different",
    "cause", "causes", "caused", "make", "makes", "made", "move", "moves", "moved", "put", "puts", "placed",
    "go", "goes", "went", "fall", "falls", "fell", "open", "opens", "closed", "break", "breaks", "broke",
}
STOPWORDS = set("""
a an the and or but if then when while because so for to of in on at by from with without into onto through over under above below after before during as is are was were be been being has have had do does did it its they them their he she his her we our you your this that these those there here not no only all some any every more less same different like than can could should would may might must will shall just also very
""".split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


collate_mod = load_module("consistency_collate_for_step077", COLLATE_SCRIPT)


def norm_text(text: str) -> str:
    text = text.lower().replace("Ġ", " ").replace("▁", " ")
    pieces = re.findall(r"[a-z0-9]+", text)
    return " ".join(pieces)


def token_norm(tokenizer, ids: list[int]) -> str:
    if not ids:
        return ""
    return norm_text(tokenizer.decode([int(x) for x in ids], clean_up_tokenization_spaces=False))


def is_content_like(norm: str) -> bool:
    parts = norm.split()
    if not parts:
        return False
    w = parts[0]
    if w in STOPWORDS or w in RELATION_WORDS:
        return False
    if len(w) < 3:
        return False
    return bool(re.search(r"[a-z]", w))


def ordered_positions(ranges: list[list[int]]) -> list[int]:
    out: list[int] = []
    for a, b in ranges:
        out.extend(range(int(a), int(b)))
    return out


def fill_to_length(values: list[int], n: int, mask_id: int) -> list[int]:
    if n <= 0:
        return []
    if not values:
        return [mask_id] * n
    return [int(values[i % len(values)]) for i in range(n)]


def group_positions_in_ranges(word_group: torch.Tensor, ranges: list[list[int]]) -> list[tuple[int, list[int]]]:
    by_gid: dict[int, list[int]] = defaultdict(list)
    first: dict[int, int] = {}
    for pos in ordered_positions(ranges):
        if 0 <= pos < int(word_group.numel()):
            gid = int(word_group[pos].item())
            if gid >= 0:
                by_gid[gid].append(pos)
                first.setdefault(gid, pos)
    return [(gid, by_gid[gid]) for gid in sorted(by_gid, key=lambda g: first[g])]


@dataclass(frozen=True)
class Target:
    target_id: int
    example_id: int
    global_row_1based: int
    pair_id: str
    gid: int
    norm: str
    first_word: str
    target_positions: tuple[int, ...]
    source_positions: tuple[int, ...]
    same_row_decoy_positions: tuple[int, ...]
    cross_row_decoy_global: int
    cross_row_decoy_positions: tuple[int, ...]
    gold_ids: tuple[int, ...]
    n_tokens: int
    row_group_count: int


def choose_example_ids(all_ids: list[int], sample_rows: int, seed: int) -> list[int]:
    ids = sorted(int(x) for x in all_ids)
    if sample_rows <= 0 or sample_rows >= len(ids):
        return ids
    rng = random.Random(seed)
    stride_n = int(sample_rows * 2 / 3)
    stride = [ids[round(i * (len(ids) - 1) / max(1, stride_n - 1))] for i in range(stride_n)] if stride_n else []
    rem = [x for x in ids if x not in set(stride)]
    rng.shuffle(rem)
    return sorted(stride + rem[: sample_rows - len(stride)])


def read_examples(path: pathlib.Path, example_ids: list[int]) -> list[Any]:
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
            found[exid] = collate_mod.StreamExample(text=text, words=int(obj.get("words", len(text.split()))), example_id=exid, source=str(obj.get("source", "")), global_row_1based=row_1)
            if len(found) >= len(wanted):
                break
    out = [found[eid] for eid in example_ids if eid in found]
    out.sort(key=lambda x: x.global_row_1based)
    return out


def build_targets(examples: list[Any], tokenizer, span_by_ex: dict[int, dict[str, Any]], strict_map: dict[str, Any], max_targets: int, seed: int) -> tuple[list[Target], dict[str, Any]]:
    ds = collate_mod.ConsistencySpanDataset(examples, tokenizer, span_by_ex, SEQ_LEN)
    row_items = [ds[i] for i in range(len(ds))]
    row_by_global = {int(x["global_row_1based"]): x for x in row_items}
    source_position_bank: list[tuple[int, tuple[int, ...]]] = []
    for item in row_items:
        for p in item.get("aux_pairs") or []:
            pos = tuple(ordered_positions(p["source_ranges"]))
            if pos:
                source_position_bank.append((int(item["global_row_1based"]), pos))
    rng = random.Random(seed)
    candidates: list[Target] = []
    counts = Counter()
    for item in row_items:
        eid = int(item["example_id"])
        map_rec = strict_map.get(str(eid)) or strict_map.get(eid)
        if not map_rec:
            counts["no_strict_map_record"] += 1
            continue
        allowed = set(int(x) for x in map_rec.get("strict_content_innovation_gids", []))
        if not allowed:
            counts["row_no_strict_gids"] += 1
            continue
        if item.get("aux_error"):
            counts[f"aux_error_{item['aux_error']}"] += 1
            continue
        pairs = item.get("aux_pairs") or []
        if len(pairs) < 2:
            counts["rows_lt2_pairs"] += 1
            continue
        input_ids: torch.Tensor = item["input_ids"]
        word_group: torch.Tensor = item["word_group"]
        # Row-normalized group occurrence counts for an independent assertion that the selected group is row-unique.
        row_gid_to_norm: dict[int, str] = {}
        row_norm_counts: Counter[str] = Counter()
        by_gid: dict[int, list[int]] = defaultdict(list)
        for pos in range(int(item["attention_mask"].sum().item())):
            gid = int(word_group[pos].item())
            if gid >= 0:
                by_gid[gid].append(pos)
        for gid, posns in by_gid.items():
            n = token_norm(tokenizer, [int(input_ids[p].item()) for p in posns])
            if n:
                row_gid_to_norm[int(gid)] = n
                row_norm_counts[n] += 1
        for pi, p in enumerate(pairs):
            src_groups = group_positions_in_ranges(word_group, p["source_ranges"])
            rew_groups = group_positions_in_ranges(word_group, p["rewrite_ranges"])
            src_norms = set()
            for _, posns in src_groups:
                n = token_norm(tokenizer, [int(input_ids[q].item()) for q in posns])
                if n:
                    src_norms.add(n)
            src_pos = tuple(ordered_positions(p["source_ranges"]))
            if not src_pos:
                counts["pair_no_source"] += 1
                continue
            same_decoy_pair = pairs[(pi + 1) % len(pairs)]
            if same_decoy_pair.get("pair_id") == p.get("pair_id") and len(pairs) > 1:
                same_decoy_pair = pairs[(pi + 2) % len(pairs)]
            same_decoy_pos = tuple(ordered_positions(same_decoy_pair["source_ranges"]))
            if not same_decoy_pos:
                counts["pair_no_same_decoy"] += 1
                continue
            cross_candidates = [x for x in source_position_bank if x[0] != int(item["global_row_1based"])]
            sample = rng.sample(cross_candidates, min(64, len(cross_candidates))) if cross_candidates else []
            cross_gr, cross_pos = min(sample, key=lambda x: abs(len(x[1]) - len(src_pos))) if sample else (int(item["global_row_1based"]), same_decoy_pos)
            for gid, posns in rew_groups:
                gid = int(gid)
                if gid not in allowed:
                    continue
                ids = [int(input_ids[q].item()) for q in posns]
                if any(x in tokenizer.all_special_ids for x in ids):
                    counts["skip_special_in_target"] += 1
                    continue
                n = token_norm(tokenizer, ids)
                if not n or len(n) <= 1:
                    counts["skip_empty_or_single"] += 1
                    continue
                if n in src_norms:
                    counts["unexpected_source_hit"] += 1
                    continue
                if row_norm_counts.get(n, 0) != 1:
                    counts["unexpected_not_row_unique"] += 1
                    continue
                if not is_content_like(n):
                    counts["unexpected_not_content_like"] += 1
                    continue
                candidates.append(Target(
                    target_id=-1,
                    example_id=eid,
                    global_row_1based=int(item["global_row_1based"]),
                    pair_id=str(p.get("pair_id", "")),
                    gid=gid,
                    norm=n,
                    first_word=n.split()[0] if n.split() else n,
                    target_positions=tuple(int(q) for q in posns),
                    source_positions=tuple(int(q) for q in src_pos),
                    same_row_decoy_positions=tuple(int(q) for q in same_decoy_pos),
                    cross_row_decoy_global=int(cross_gr),
                    cross_row_decoy_positions=tuple(int(q) for q in cross_pos),
                    gold_ids=tuple(ids),
                    n_tokens=len(ids),
                    row_group_count=max([int(x.item()) for x in word_group if int(x.item()) >= 0], default=-1) + 1,
                ))
    rng.shuffle(candidates)
    if max_targets > 0:
        candidates = candidates[:max_targets]
    candidates.sort(key=lambda t: (t.global_row_1based, t.pair_id, t.target_positions, t.gid))
    targets = [Target(i, t.example_id, t.global_row_1based, t.pair_id, t.gid, t.norm, t.first_word, t.target_positions, t.source_positions, t.same_row_decoy_positions, t.cross_row_decoy_global, t.cross_row_decoy_positions, t.gold_ids, t.n_tokens, t.row_group_count) for i, t in enumerate(candidates)]
    return targets, {
        "n_candidates_before_cap": len(candidates) if max_targets <= 0 else None,
        "selected_targets": len(targets),
        "selected_rows": len(set(t.global_row_1based for t in targets)),
        "selected_examples": len(set(t.example_id for t in targets)),
        "selected_token_labels": int(sum(t.n_tokens for t in targets)),
        "first_word_top20": Counter(t.first_word for t in targets).most_common(20),
        "skips_or_assertions": dict(counts),
    }


def make_eval_rows(examples: list[Any], tokenizer, span_by_ex: dict[int, dict[str, Any]], targets: list[Target]) -> list[dict[str, Any]]:
    ds = collate_mod.ConsistencySpanDataset(examples, tokenizer, span_by_ex, SEQ_LEN)
    item_by_global = {int(ds[i]["global_row_1based"]): ds[i] for i in range(len(ds))}
    base_ids = {gr: item["input_ids"] for gr, item in item_by_global.items()}
    attn = {gr: item["attention_mask"] for gr, item in item_by_global.items()}
    mask_id = int(tokenizer.mask_token_id)
    rows: list[dict[str, Any]] = []
    for t in targets:
        ids0 = base_ids[t.global_row_1based]
        for sc in SCENARIOS:
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
                "input_ids": ids,
                "attention_mask": attn[t.global_row_1based],
                "target_positions": t.target_positions,
                "gold_ids": t.gold_ids,
                "global_row_1based": t.global_row_1based,
                "example_id": t.example_id,
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
    if not keys or iters <= 0:
        vals = [float(r[metric]) for r in rows]
        return {"mean": statistics.mean(vals) if vals else None, "ci05": None, "ci95": None, "rows": len(keys)}
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        sample_keys = [rng.choice(keys) for _ in keys]
        xs = [x for k in sample_keys for x in by_row[k]]
        vals.append(statistics.mean(xs))
    vals.sort()
    return {"mean": statistics.mean([float(r[metric]) for r in rows]), "ci05": vals[int(0.05 * (len(vals) - 1))], "ci95": vals[int(0.95 * (len(vals) - 1))], "rows": len(keys)}


def eval_checkpoint(label: str, ckpt: pathlib.Path, rows: list[dict[str, Any]], batch_size: int, threads: int, boot_iters: int, seed: int, out_dir: pathlib.Path) -> dict[str, Any]:
    if not ckpt.exists():
        return {"label": label, "status": "missing", "path": str(ckpt)}
    if threads > 0:
        torch.set_num_threads(threads)
    t0 = time.time()
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt))
    model.eval(); model.to("cpu")
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    loss: dict[tuple[int, str], float] = {}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
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
        meta.setdefault(tid, {"global_row_1based": int(item["global_row_1based"]), "example_id": int(item["example_id"]), "norm": str(item["norm"])})
    per = []
    for tid, m in meta.items():
        true = loss.get((tid, "true_source")); masked = loss.get((tid, "source_masked"))
        same = loss.get((tid, "same_row_decoy_filled")); cross = loss.get((tid, "cross_row_decoy_filled"))
        if None in (true, masked, same, cross):
            continue
        per.append({
            "target_id": tid,
            "example_id": m["example_id"],
            "global_row_1based": m["global_row_1based"],
            "norm": m["norm"],
            "true_loss": true,
            "source_help": masked - true,
            "same_decoy_advantage": same - true,
            "cross_decoy_advantage": cross - true,
            "masked_minus_same_decoy": masked - same,
            "masked_minus_cross_decoy": masked - cross,
        })
    metrics = ["true_loss", "source_help", "same_decoy_advantage", "cross_decoy_advantage", "masked_minus_same_decoy", "masked_minus_cross_decoy"]
    summary = {m: summarize([r[m] for r in per]) for m in metrics}
    summary["bootstrap_source_help"] = bootstrap_mean(per, "source_help", boot_iters, seed + 11)
    summary["bootstrap_same_decoy_advantage"] = bootstrap_mean(per, "same_decoy_advantage", boot_iters, seed + 17)
    per_path = out_dir / f"per_target_{label}.jsonl"
    with per_path.open("w", encoding="utf-8") as f:
        for r in per:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"label": label, "status": "ok", "path": str(ckpt), "n_targets": len(per), "summary": summary, "per_target_jsonl": str(per_path), "elapsed_sec": round(time.time() - t0, 3)}


def delta_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r.get("label"): r for r in results if r.get("status") == "ok"}
    pairs = [("reference_70M", "tokenmean_70M"), ("reference_80M", "tokenmean_80M"), ("tokenmean_80M", "clean_80M"), ("tokenmean_100M", "tokenmean_80M")]
    out = {}
    for a, b in pairs:
        if a not in by or b not in by:
            continue
        sa = by[a]["summary"]; sb = by[b]["summary"]
        d = {}
        for metric in ["true_loss", "source_help", "same_decoy_advantage", "cross_decoy_advantage", "masked_minus_same_decoy"]:
            va = (sa.get(metric) or {}).get("mean")
            vb = (sb.get(metric) or {}).get("mean")
            d[f"delta_{a}_minus_{b}_{metric}"] = None if va is None or vb is None else float(va) - float(vb)
        out[f"{a}_minus_{b}"] = d
    return out


def fmt(x: Any) -> str:
    try:
        y = float(x)
    except Exception:
        return "NA"
    return f"{y:.4f}" if math.isfinite(y) else "NA"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--sample-rows", type=int, default=768)
    ap.add_argument("--max-targets", type=int, default=320)
    ap.add_argument("--sample-seed", type=int, default=77773)
    ap.add_argument("--target-seed", type=int, default=77873)
    ap.add_argument("--checkpoints", default="tokenmean_70M,tokenmean_80M")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--bootstrap-iters", type=int, default=200)
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_sha = sha256_file(POOL_10M)
    train_sha = sha256_file(TRAIN_100M)
    tok_sha = sha256_file(TOKENIZER_DIR / "tokenizer.json")
    if pool_sha != EXPECTED_POOL_SHA:
        raise RuntimeError(f"pool SHA mismatch {pool_sha}")
    if train_sha != EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch {train_sha}")
    if tok_sha != EXPECTED_TOK_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch {tok_sha}")
    strict_map = json.loads(MAP_JSON.read_text(encoding="utf-8"))
    span_by_ex = collate_mod.load_span_map(SPAN_JSONL)
    valid_ids = [int(k) for k, v in strict_map.items() if v.get("strict_content_innovation_gids")]
    example_ids = choose_example_ids(valid_ids, args.sample_rows, args.sample_seed)
    examples = read_examples(POOL_10M, example_ids)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    targets, build = build_targets(examples, tokenizer, span_by_ex, strict_map, args.max_targets, args.target_seed)
    target_manifest = out_dir / "strict_innovation_target_manifest.jsonl"
    with target_manifest.open("w", encoding="utf-8") as f:
        for t in targets:
            d = asdict(t)
            # tuples are JSON-encoded as arrays by json.dumps.
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    rows = make_eval_rows(examples, tokenizer, span_by_ex, targets)
    labels = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    results: list[dict[str, Any]] = []
    if not args.manifest_only:
        for label in labels:
            if label not in CHECKPOINTS:
                raise ValueError(f"unknown checkpoint label {label}")
            print(json.dumps({"event": "checkpoint_start", "label": label, "path": str(CHECKPOINTS[label]), "targets": len(targets), "eval_sequences": len(rows), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
            r = eval_checkpoint(label, CHECKPOINTS[label], rows, args.batch_size, args.torch_threads, args.bootstrap_iters, args.target_seed, out_dir)
            results.append(r)
            print(json.dumps({"event": "checkpoint_done", "label": label, "status": r.get("status"), "elapsed_sec": r.get("elapsed_sec")}), flush=True)
    summary = {
        "status": "STRICT_INNOVATION_TARGET_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Fixed CPU readout of repaired row-unique content-innovation target prediction for token-mean and, once available, research checkpoints.",
        "inputs": {
            "pool_10m_sha256": pool_sha,
            "train_100m_sha256": train_sha,
            "tokenizer_sha256": tok_sha,
            "strict_map": str(MAP_JSON),
            "sample_rows_requested": args.sample_rows,
            "examples_loaded": len(examples),
            "valid_strict_rows_total": len(valid_ids),
            "build": build,
            "target_manifest": str(target_manifest),
            "checkpoints_requested": labels,
        },
        "checkpoint_results": results,
        "comparisons": delta_summary(results),
        "scientific_reading": [
            "This readout is not a BabyLM score; it measures whether the exact target class selected for innovation-biased masking becomes easier under true source context.",
            "For research, a useful mechanism signature is lower true_loss and/or higher source_help on these strict targets versus tokenmean at the same exposure, interpreted together with the official-compatible cheap-column trajectory.",
            "If broad transfer weakens and this target readout does not improve, the innovation-masking family should not be carried forward just because exact-swap is mechanically cleaner.",
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "strict_innovation_target_probe.json"
    out_md = out_dir / "strict_innovation_target_probe.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research strict row-unique innovation target probe",
        "",
        summary["purpose"],
        "",
        f"- examples loaded: `{len(examples)}` / requested `{args.sample_rows}`",
        f"- selected targets: `{len(targets)}`; selected rows: `{build['selected_rows']}`; token labels: `{build['selected_token_labels']}`",
        f"- target manifest: `{target_manifest}`",
        f"- train SHA: `{train_sha}`",
        f"- tokenizer SHA: `{tok_sha}`",
        "",
        "## Results",
        "| checkpoint | status | n | true loss | source help | same-row decoy advantage | cross-row decoy advantage | masked-same | source help bootstrap 5-95 | same decoy bootstrap 5-95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"| {r.get('label')} | {r.get('status')} | | | | | | | | |")
            continue
        s = r["summary"]
        b1 = s.get("bootstrap_source_help") or {}; b2 = s.get("bootstrap_same_decoy_advantage") or {}
        ci1 = "" if b1.get("ci05") is None else f"[{b1['ci05']:.4f},{b1['ci95']:.4f}]"
        ci2 = "" if b2.get("ci05") is None else f"[{b2['ci05']:.4f},{b2['ci95']:.4f}]"
        lines.append(f"| {r['label']} | ok | {(s.get('true_loss') or {}).get('n')} | {fmt((s.get('true_loss') or {}).get('mean'))} | {fmt((s.get('source_help') or {}).get('mean'))} | {fmt((s.get('same_decoy_advantage') or {}).get('mean'))} | {fmt((s.get('cross_decoy_advantage') or {}).get('mean'))} | {fmt((s.get('masked_minus_same_decoy') or {}).get('mean'))} | {ci1} | {ci2} |")
    if summary["comparisons"]:
        lines += ["", "## Direct comparisons"]
        for name, vals in summary["comparisons"].items():
            lines.append(f"- `{name}`: " + ", ".join(f"{k}={fmt(v)}" for k, v in vals.items()))
    lines += ["", "## Reading"]
    for x in summary["scientific_reading"]:
        lines.append(f"- {x}")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "targets": len(targets), "checkpoint_status": {r.get("label"): r.get("status") for r in results}, "elapsed_sec": summary["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
