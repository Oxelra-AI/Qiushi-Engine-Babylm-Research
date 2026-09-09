#!/usr/bin/env python3
"""research: direct within-context copy score for existing DeBERTa arms.

Scientific purpose
------------------
Entity depth results suggest a fixed-budget trade-off: exact in-window repetition may
train an MLM to use a copied span in the same context, while varied source-conditioned
restatement cannot be solved by exact copying. This script directly measures whether
existing arms differ in a repeated-versus-unrepeated masked-token benefit on the same
probe contexts.

The score is paired: each item has a repeated condition in which an unmasked copy of
the target token/span appears elsewhere in the same <=256-token window, and an
unrepeated condition where that other occurrence is replaced by length-matched random
non-special tokens. Copy gain = NLL(unrepeated) - NLL(repeated); larger means the
model gains more from exact in-context recurrence. The same records are scored for
VIEW, CLEAN, and REPEAT in both existing seeds and late checkpoints.
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
import pathlib
import random
import re
import statistics
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'relation_learning'
OUT_DEFAULT = WS / "data" / "copy_score_probe"
frontier_consolidation_RUNS = ROOT / "experiments/archive" / 'frontier_consolidation' / "training" / "runs"
PAIR_PATH = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "dose_distribution_select" / "selected_matched_max_pairs.jsonl"
POOL_DIR = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "dose_2p64x_rowholdout_pools"
TRAIN_STREAMS = {
    "view": POOL_DIR / "compact_view_dose2p64x_10M.jsonl",
    "repeat": POOL_DIR / "compact_repeat_dose2p64x_10M.jsonl",
}
ROW_META = {
    "view": POOL_DIR / "compact_view_dose2p64x_changed_block_rows_meta.jsonl",
    "repeat": POOL_DIR / "compact_repeat_dose2p64x_changed_block_rows_meta.jsonl",
}
ARM_CONFIGS = {
    "D_V_43022": frontier_consolidation_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_C_43022": frontier_consolidation_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_R_43022": frontier_consolidation_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_V_43122": frontier_consolidation_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_C_43122": frontier_consolidation_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_R_43122": frontier_consolidation_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122",
}
CKS = ["chck_80M", "chck_90M", "chck_100M"]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: pathlib.Path, limit: int | None = None):
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
                n += 1
                if limit is not None and n >= limit:
                    return


def wc(text: str) -> int:
    return len(text.split())


def repeat_source_words(source_text: str, n_words: int, salt: str) -> str:
    toks = source_text.split()
    if not toks or n_words <= 0:
        return ""
    h = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16)
    start = h % len(toks)
    rot = toks[start:] + toks[:start]
    out: list[str] = []
    while len(out) < n_words:
        out.extend(rot[: n_words - len(out)])
    return " ".join(out)


def safe_vocab_ids(tokenizer) -> list[int]:
    special = set(int(x) for x in tokenizer.all_special_ids)
    ids = []
    vocab_size = len(tokenizer)
    for i in range(vocab_size):
        if i in special:
            continue
        tok = tokenizer.convert_ids_to_tokens(i)
        if tok is None:
            continue
        s = str(tok)
        if not s or s.startswith("[") or s.startswith("<"):
            continue
        ids.append(i)
    if len(ids) < 1000:
        raise RuntimeError(f"too few safe vocab ids: {len(ids)}")
    return ids


def rand_ids(rng: random.Random, vocab_ids: list[int], n: int, forbid: set[int] | None = None) -> list[int]:
    forbid = forbid or set()
    out = []
    while len(out) < n:
        x = rng.choice(vocab_ids)
        if x not in forbid:
            out.append(x)
    return out


def wrap_body(tokenizer, body: list[int], mask_body_pos: int, target_id: int, max_len: int) -> dict[str, Any] | None:
    start_tok = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    end_tok = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    mask = tokenizer.mask_token_id
    if start_tok is None or end_tok is None or mask is None:
        raise RuntimeError("tokenizer lacks start/end/mask token id")
    if len(body) + 2 > max_len:
        return None
    ids = [int(start_tok)] + list(map(int, body)) + [int(end_tok)]
    pos = int(mask_body_pos) + 1
    if pos <= 0 or pos >= len(ids) - 1:
        raise RuntimeError("bad mask position")
    ids[pos] = int(mask)
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "position": pos, "target_id": int(target_id)}


def find_subseq(hay: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(hay):
        return None
    first = needle[0]
    L = len(needle)
    for i, x in enumerate(hay[: len(hay) - L + 1]):
        if x == first and hay[i : i + L] == needle:
            return i
    return None


def build_random_records(tokenizer, vocab_ids: list[int], rng: random.Random, n_per_span: int, span_lengths: list[int], max_len: int) -> list[dict[str, Any]]:
    records = []
    pair_i = 0
    for span_len in span_lengths:
        for j in range(n_per_span):
            prefix_len = rng.randint(12, 36)
            middle_len = rng.randint(8, 24)
            suffix_len = rng.randint(12, 36)
            span = rand_ids(rng, vocab_ids, span_len)
            replacement = rand_ids(rng, vocab_ids, span_len, forbid=set(span))
            prefix = rand_ids(rng, vocab_ids, prefix_len)
            middle = rand_ids(rng, vocab_ids, middle_len)
            suffix = rand_ids(rng, vocab_ids, suffix_len)
            mask_second = (j % 2 == 0)
            k = rng.randrange(span_len)
            if mask_second:
                body_rep = prefix + span + middle + span + suffix
                body_unr = prefix + replacement + middle + span + suffix
                mask_pos = len(prefix) + span_len + len(middle) + k
            else:
                body_rep = prefix + span + middle + span + suffix
                body_unr = prefix + span + middle + replacement + suffix
                mask_pos = len(prefix) + k
            target = span[k]
            for cond, body in [("repeated", body_rep), ("unrepeated", body_unr)]:
                rec = wrap_body(tokenizer, body, mask_pos, target, max_len=max_len)
                if rec is None:
                    continue
                rec.update({
                    "probe_family": "random_token_span",
                    "pair_id": f"rand:{span_len}:{pair_i}",
                    "condition": cond,
                    "span_len": span_len,
                    "mask_occurrence": "second" if mask_second else "first",
                    "source_pair_id": "",
                    "body_len": len(body),
                })
                records.append(rec)
            pair_i += 1
    return records


def load_pairs(limit_scan: int | None = None) -> list[dict[str, Any]]:
    out = []
    for obj in read_jsonl(PAIR_PATH, limit_scan):
        out.append(obj)
    return out


def build_packet_records(tokenizer, vocab_ids: list[int], rng: random.Random, n_per_span: int, span_lengths: list[int], max_len: int, scan_pairs: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = load_pairs(scan_pairs)
    records: list[dict[str, Any]] = []
    stats = defaultdict(int)
    made_by_span = {L: 0 for L in span_lengths}
    for obj in pairs:
        if all(made_by_span[L] >= n_per_span for L in span_lengths):
            break
        src_text = " ".join(str(obj["source_text"]).split())
        rw_words = int(obj["rewrite_words"])
        pid = str(obj["pair_id"])
        rep_text = repeat_source_words(src_text, rw_words, pid)
        src_ids = tokenizer(src_text, add_special_tokens=False)["input_ids"]
        rep_ids = tokenizer(rep_text, add_special_tokens=False)["input_ids"]
        if not src_ids or not rep_ids:
            stats["empty"] += 1
            continue
        if len(src_ids) + len(rep_ids) + 2 > max_len:
            stats["too_long"] += 1
            continue
        for span_len in span_lengths:
            if made_by_span[span_len] >= n_per_span:
                continue
            candidates = []
            for j in range(0, len(rep_ids) - span_len + 1):
                needle = rep_ids[j : j + span_len]
                si = find_subseq(src_ids, needle)
                if si is not None:
                    candidates.append((si, j))
            if not candidates:
                stats[f"no_match_span_{span_len}"] += 1
                continue
            si, j = rng.choice(candidates)
            k = rng.randrange(span_len)
            target = rep_ids[j + k]
            src_control = list(src_ids)
            src_control[si : si + span_len] = rand_ids(rng, vocab_ids, span_len, forbid=set(rep_ids[j : j + span_len]))
            body_rep = src_ids + rep_ids
            body_unr = src_control + rep_ids
            mask_pos = len(src_ids) + j + k
            for cond, body in [("repeated", body_rep), ("unrepeated", body_unr)]:
                rec = wrap_body(tokenizer, body, mask_pos, target, max_len=max_len)
                if rec is None:
                    stats["wrap_too_long"] += 1
                    continue
                rec.update({
                    "probe_family": "actual_packet_source_repeat",
                    "pair_id": f"packet:{span_len}:{made_by_span[span_len]}:{pid}",
                    "condition": cond,
                    "span_len": span_len,
                    "mask_occurrence": "companion",
                    "source_pair_id": pid,
                    "body_len": len(body),
                    "source_token_len": len(src_ids),
                    "companion_token_len": len(rep_ids),
                })
                records.append(rec)
            made_by_span[span_len] += 1
    stats.update({f"made_span_{L}": made_by_span[L] for L in span_lengths})
    return records, dict(stats)


def verify_training_design(tokenizer, max_rows: int = 32) -> dict[str, Any]:
    pair_map = {str(p["pair_id"]): p for p in load_pairs(None)}
    checks: dict[str, Any] = {
        "pair_path": rel(PAIR_PATH),
        "view_stream": rel(TRAIN_STREAMS["view"]),
        "repeat_stream": rel(TRAIN_STREAMS["repeat"]),
        "view_row_meta": rel(ROW_META["view"]),
        "repeat_row_meta": rel(ROW_META["repeat"]),
    }
    rows = {k: list(read_jsonl(v, max_rows)) for k, v in TRAIN_STREAMS.items()}
    metas = {k: list(read_jsonl(v, max_rows)) for k, v in ROW_META.items()}
    design_rows = []
    exact_view_rows = 0
    exact_repeat_rows = 0
    all_pair_local = True
    token_lens = []
    for i in range(max_rows):
        if i >= len(rows["view"]) or i >= len(rows["repeat"]):
            break
        vrow = rows["view"][i]
        rrow = rows["repeat"][i]
        meta = metas["view"][i]
        pid_list = [str(x) for x in meta.get("pair_ids", [])]
        v_segments = []
        r_segments = []
        for pid in pid_list:
            p = pair_map[pid]
            src = " ".join(str(p["source_text"]).split())
            rew = " ".join(str(p["rewrite_text"]).split())
            rep = repeat_source_words(src, int(p["rewrite_words"]), pid)
            v_segments.append(f"{src} {rew}".strip())
            r_segments.append(f"{src} {rep}".strip())
            if not str(vrow["text"]).find(src) >= 0 or not str(rrow["text"]).find(src) >= 0:
                all_pair_local = False
        v_rebuilt = " ".join(v_segments)
        r_rebuilt = " ".join(r_segments)
        v_ok = (" ".join(str(vrow["text"]).split()) == v_rebuilt)
        r_ok = (" ".join(str(rrow["text"]).split()) == r_rebuilt)
        exact_view_rows += int(v_ok)
        exact_repeat_rows += int(r_ok)
        v_tok = len(tokenizer(str(vrow["text"]), add_special_tokens=True)["input_ids"])
        r_tok = len(tokenizer(str(rrow["text"]), add_special_tokens=True)["input_ids"])
        token_lens.append((v_tok, r_tok))
        design_rows.append({
            "row_index": i,
            "pair_count": len(pid_list),
            "words": int(vrow["words"]),
            "view_rebuild_exact": int(v_ok),
            "repeat_rebuild_exact": int(r_ok),
            "view_token_len_with_special": v_tok,
            "repeat_token_len_with_special": r_tok,
            "view_fits_256": int(v_tok <= 256),
            "repeat_fits_256": int(r_tok <= 256),
            "example_view_text_prefix": str(vrow["text"])[:180],
            "example_repeat_text_prefix": str(rrow["text"])[:180],
        })
    checks.update({
        "rows_checked": len(design_rows),
        "view_rows_rebuilt_exact": exact_view_rows,
        "repeat_rows_rebuilt_exact": exact_repeat_rows,
        "all_checked_pairs_local_source_then_companion": all_pair_local,
        "checked_view_token_len_max_with_special": max([x[0] for x in token_lens], default=0),
        "checked_repeat_token_len_max_with_special": max([x[1] for x in token_lens], default=0),
        "checked_view_rows_fitting_256": sum(1 for x, _ in token_lens if x <= 256),
        "checked_repeat_rows_fitting_256": sum(1 for _, y in token_lens if y <= 256),
        "row_examples": design_rows[:5],
    })
    return checks


@torch.no_grad()
def score_records(model, records: list[dict[str, Any]], device: torch.device, pad_id: int, batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        max_len = max(len(r["input_ids"]) for r in batch)
        ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
        att = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, r in enumerate(batch):
            L = len(r["input_ids"])
            ids[i, :L] = torch.tensor(r["input_ids"], dtype=torch.long)
            att[i, :L] = torch.tensor(r["attention_mask"], dtype=torch.long)
        ids = ids.to(device)
        att = att.to(device)
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        for i, r in enumerate(batch):
            lp = float(logp[i, int(r["position"]), int(r["target_id"])].detach().cpu())
            meta = {k: v for k, v in r.items() if k not in {"input_ids", "attention_mask", "position", "target_id"}}
            meta.update({"nll": -lp, "logprob": lp, "target_id": int(r["target_id"]), "seq_len": len(r["input_ids"])})
            rows.append(meta)
    return rows


def pair_copy_rows(scored: list[dict[str, Any]], arm: str, ck: str) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in scored:
        key = (r["probe_family"], r["pair_id"], int(r["span_len"]))
        by_pair[key][r["condition"]] = r
    out = []
    for (fam, pid, span_len), d in sorted(by_pair.items()):
        if "repeated" not in d or "unrepeated" not in d:
            continue
        rr = d["repeated"]
        ur = d["unrepeated"]
        out.append({
            "arm": arm,
            "checkpoint": ck,
            "seed": arm.split("_")[-1],
            "role": arm.split("_")[1],
            "probe_family": fam,
            "pair_id": pid,
            "span_len": span_len,
            "mask_occurrence": rr.get("mask_occurrence", ""),
            "source_pair_id": rr.get("source_pair_id", ""),
            "body_len": rr.get("body_len", ""),
            "seq_len": rr.get("seq_len", ""),
            "repeated_nll": rr["nll"],
            "unrepeated_nll": ur["nll"],
            "copy_gain": ur["nll"] - rr["nll"],
        })
    return out


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return float(statistics.mean(xs)) if xs else float("nan")


def pstdev(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for fam in [r["probe_family"], "ALL"]:
            for span in [str(r["span_len"]), "ALL"]:
                groups[(r["arm"], r["checkpoint"], fam, span)].append(r)
    out = []
    for (arm, ck, fam, span), vals in sorted(groups.items()):
        gains = [float(v["copy_gain"]) for v in vals]
        out.append({
            "arm": arm,
            "checkpoint": ck,
            "seed": arm.split("_")[-1],
            "role": arm.split("_")[1],
            "probe_family": fam,
            "span_len": span,
            "n": len(vals),
            "mean_repeated_nll": mean([float(v["repeated_nll"]) for v in vals]),
            "mean_unrepeated_nll": mean([float(v["unrepeated_nll"]) for v in vals]),
            "mean_copy_gain": mean(gains),
            "sd_copy_gain": pstdev(gains),
            "se_copy_gain": pstdev(gains) / math.sqrt(len(gains)) if len(gains) > 1 else float("nan"),
        })
    return out


def contrasts(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idx = {(r["arm"], r["checkpoint"], r["probe_family"], r["span_len"]): r for r in summary}
    out = []
    for seed in ["43022", "43122"]:
        arms = {role: f"D_{role}_{seed}" for role in ["V", "C", "R"]}
        for ck in CKS:
            combos = sorted({(r["probe_family"], r["span_len"]) for r in summary if r["checkpoint"] == ck and r["seed"] == seed})
            for fam, span in combos:
                for name, a, b in [("RminusC", "R", "C"), ("RminusV", "R", "V"), ("CminusV", "C", "V"), ("VminusC", "V", "C"), ("VminusR", "V", "R")]:
                    ka = (arms[a], ck, fam, span)
                    kb = (arms[b], ck, fam, span)
                    if ka not in idx or kb not in idx:
                        continue
                    ra, rb = idx[ka], idx[kb]
                    out.append({
                        "seed": seed,
                        "checkpoint": ck,
                        "probe_family": fam,
                        "span_len": span,
                        "contrast": name,
                        "delta_copy_gain_a_minus_b": float(ra["mean_copy_gain"]) - float(rb["mean_copy_gain"]),
                        "copy_gain_a": float(ra["mean_copy_gain"]),
                        "copy_gain_b": float(rb["mean_copy_gain"]),
                        "delta_repeated_nll_a_minus_b": float(ra["mean_repeated_nll"]) - float(rb["mean_repeated_nll"]),
                        "delta_unrepeated_nll_a_minus_b": float(ra["mean_unrepeated_nll"]) - float(rb["mean_unrepeated_nll"]),
                        "n": int(ra["n"]),
                    })
    return out


def late_summary(con_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in con_rows:
        groups[(r["seed"], r["probe_family"], r["span_len"], r["contrast"])].append(r)
    out = []
    for (seed, fam, span, contrast), vals in sorted(groups.items()):
        out.append({
            "seed": seed,
            "probe_family": fam,
            "span_len": span,
            "contrast": contrast,
            "n_checkpoints": len(vals),
            "late_mean_delta_copy_gain_a_minus_b": mean([float(v["delta_copy_gain_a_minus_b"]) for v in vals]),
            "late_mean_copy_gain_a": mean([float(v["copy_gain_a"]) for v in vals]),
            "late_mean_copy_gain_b": mean([float(v["copy_gain_b"]) for v in vals]),
            "late_mean_delta_repeated_nll_a_minus_b": mean([float(v["delta_repeated_nll_a_minus_b"]) for v in vals]),
            "late_mean_delta_unrepeated_nll_a_minus_b": mean([float(v["delta_unrepeated_nll_a_minus_b"]) for v in vals]),
        })
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    preferred = [
        "arm", "checkpoint", "seed", "role", "probe_family", "span_len", "contrast", "n", "n_checkpoints",
        "mean_copy_gain", "sd_copy_gain", "se_copy_gain", "delta_copy_gain_a_minus_b", "late_mean_delta_copy_gain_a_minus_b",
        "copy_gain_a", "copy_gain_b", "late_mean_copy_gain_a", "late_mean_copy_gain_b",
        "mean_repeated_nll", "mean_unrepeated_nll", "delta_repeated_nll_a_minus_b", "delta_unrepeated_nll_a_minus_b",
        "pair_id", "mask_occurrence", "source_pair_id", "body_len", "seq_len", "repeated_nll", "unrepeated_nll", "copy_gain",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARM_CONFIGS))
    ap.add_argument("--checkpoints", nargs="+", default=CKS)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--n-random-per-span", type=int, default=600)
    ap.add_argument("--n-packet-per-span", type=int, default=800)
    ap.add_argument("--span-lengths", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--scan-pairs", type=int, default=6000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=908008)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        for ck in args.checkpoints:
            mp = ARM_CONFIGS[arm] / "hf_model" / ck
            if not mp.exists():
                raise FileNotFoundError(mp)

    tok_root = ARM_CONFIGS[args.arms[0]] / "hf_model"
    tokenizer = AutoTokenizer.from_pretrained(str(tok_root), use_fast=True)
    vocab_ids = safe_vocab_ids(tokenizer)
    rng = random.Random(args.seed)
    random_records = build_random_records(tokenizer, vocab_ids, rng, args.n_random_per_span, args.span_lengths, args.max_len)
    packet_records, packet_stats = build_packet_records(tokenizer, vocab_ids, rng, args.n_packet_per_span, args.span_lengths, args.max_len, args.scan_pairs)
    records = random_records + packet_records
    design = verify_training_design(tokenizer)
    plan = {
        "status": "COPY_SCORE_PLAN",
        "created_utc": now(),
        "arms": args.arms,
        "checkpoints": args.checkpoints,
        "span_lengths": args.span_lengths,
        "n_records": len(records),
        "n_pairs": len(records) // 2,
        "record_counts": dict(sorted({fam: sum(1 for r in records if r["probe_family"] == fam) for fam in {r["probe_family"] for r in records}}.items())),
        "packet_record_stats": packet_stats,
        "scoring": "paired masked-token NLL, copy_gain = NLL(unrepeated) - NLL(repeated)",
        "training_design_check": design,
    }
    (out_dir / "copy_score_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    print(json.dumps({**plan, "device": str(device), "started_utc": now()}, indent=2, ensure_ascii=False), flush=True)

    all_pair_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for arm in args.arms:
        for ck in args.checkpoints:
            t0 = time.time()
            model_path = ARM_CONFIGS[arm] / "hf_model" / ck
            print(f"[LOAD] {arm} {ck} {model_path}", flush=True)
            model = AutoModelForMaskedLM.from_pretrained(str(model_path), torch_dtype=torch.float32)
            model.eval().to(device)
            scored = score_records(model, records, device, pad_id, args.batch_size)
            pair_rows = pair_copy_rows(scored, arm, ck)
            write_csv(out_dir / f"copy_score_pair_rows_{arm}_{ck}.csv", pair_rows)
            all_pair_rows.extend(pair_rows)
            meta_rows.append({"arm": arm, "checkpoint": ck, "scored_records": len(scored), "pair_rows": len(pair_rows), "elapsed_sec": round(time.time() - t0, 2)})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[DONE] {arm} {ck} pair_rows={len(pair_rows)} elapsed={time.time()-t0:.1f}s", flush=True)

    summary_rows = summarize(all_pair_rows)
    con_rows = contrasts(summary_rows)
    late_rows = late_summary(con_rows)
    write_csv(out_dir / "copy_score_pair_rows.csv", all_pair_rows)
    write_csv(out_dir / "copy_score_summary.csv", summary_rows)
    write_csv(out_dir / "copy_score_contrasts.csv", con_rows)
    write_csv(out_dir / "copy_score_late_summary.csv", late_rows)
    write_csv(out_dir / "copy_score_meta.csv", meta_rows)
    result = {
        "status": "COPY_SCORE_DONE",
        "finished_utc": now(),
        "device": str(device),
        "arms": args.arms,
        "checkpoints": args.checkpoints,
        "span_lengths": args.span_lengths,
        "n_pair_rows": len(all_pair_rows),
        "files": {
            "plan": rel(out_dir / "copy_score_plan.json"),
            "pair_rows": rel(out_dir / "copy_score_pair_rows.csv"),
            "summary": rel(out_dir / "copy_score_summary.csv"),
            "contrasts": rel(out_dir / "copy_score_contrasts.csv"),
            "late_summary": rel(out_dir / "copy_score_late_summary.csv"),
            "meta": rel(out_dir / "copy_score_meta.csv"),
        },
        "interpretation": "Larger copy_gain means lower NLL when an exact unmasked copy of the target/span exists elsewhere in the same MLM window. The proposed copy account predicts REPEAT > CLEAN >= VIEW, seed-stably, especially on actual_packet_source_repeat.",
    }
    (out_dir / "copy_score_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nKEY LATE COPY-GAIN CONTRASTS")
    for r in late_rows:
        if r["probe_family"] in {"ALL", "random_token_span", "actual_packet_source_repeat"} and r["span_len"] == "ALL" and r["contrast"] in {"RminusC", "RminusV", "CminusV"}:
            print(f"seed={r['seed']} {r['probe_family']} {r['contrast']} dCopyGain={r['late_mean_delta_copy_gain_a_minus_b']:+.4f} gain_a={r['late_mean_copy_gain_a']:+.4f} gain_b={r['late_mean_copy_gain_b']:+.4f}")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
