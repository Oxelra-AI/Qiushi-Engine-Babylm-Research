#!/usr/bin/env python3
"""research: fixed semantic denoising probe on existing compact/repeat/adjbreak trajectories.

Purpose
-------
Anchor the next target-selection intervention to what the already successful compact-view
trajectory actually changed, rather than using early MLM loss of newly constructed arms.

The probe samples fixed compact-side whole-word denoising targets from the 12,155 source+compact
pairs, categorized as:
  * retained_content: compact word is content-like and appears in the source (lexical retained proposition anchor)
  * source_absent_content: compact word is content-like and absent from the source (abstractive/reformulation content)
  * function_other: other compact-side words

For each target event, the input is identical across models: source_text + compact_rewrite with
that compact-side word's BPE pieces replaced by <mask>. Cross-entropy is measured only on the
masked pieces. This is not a held-out corpus in the training-data sense; it is a fixed held-out
mask/event probe sampled after training, meant to compare trajectory geometry and category timing.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
A02 = pathlib.Path("experiments/archive/frontier_consolidation")
CSG = ROOT / "data/marginal_corpora_v2/marginal_corpora_csg_v2.jsonl"
OUT_DIR = ROOT / "data/existing_trajectory_denoising_probe"

RUNS = {
    "compact": A02 / "training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model",
    "repeat": ROOT / "training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model",
    "adjbreak": ROOT / "training/runs/gc_adjbreak_reinvest_16k_seed43022_r2/hf_model",
}
CHECKPOINTS = ["chck_20M", "chck_60M", "chck_100M"]
TOKENIZER = ROOT / "training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
SEQ_LEN = 256

FUNC_WORDS = frozenset("a an the this that these those my your his her its our their "
    "is am are was were be been being have has had do does did will would shall should "
    "can could may might must need ought to of in on at by for with from into through "
    "during before after above below between under over about against along across "
    "around behind beside beyond down near off since toward upon within without "
    "and but or nor so yet both either neither not no nor if then else than as "
    "which who whom whose what when where how while until because although though "
    "even also just only still already very much more most less least too quite "
    "really rather than such same other another each every all some any few many "
    "no more several enough another each own same different many little much few "
    "i me we us you he him she her it they them myself yourself himself herself "
    "itself ourselves themselves one ones there here up out off away again back "
    "now then so however therefore thus hence moreover furthermore nevertheless "
    "meanwhile otherwise instead indeed certainly perhaps maybe probably "
    "been being getting going doing having making taking".split())


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def is_content(w: str) -> bool:
    n = norm(w)
    return n not in FUNC_WORDS and len(n) > 1


def stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(float(x) for x in xs)
    n = len(s)
    def q(p: float):
        if n == 1: return s[0]
        return s[min(n-1, max(0, int(round(p*(n-1)))))]
    return {"n": n, "mean": round(statistics.mean(s), 6), "median": round(statistics.median(s), 6),
            "p10": round(q(0.10), 6), "p25": round(q(0.25), 6), "p75": round(q(0.75), 6),
            "p90": round(q(0.90), 6), "min": round(s[0], 6), "max": round(s[-1], 6)}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pairs(limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with open(CSG, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def token_count(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def target_span(tok, source_text: str, side_words: list[str], word_i: int) -> tuple[int, int] | None:
    before_side = " ".join(side_words[:word_i])
    through_side = " ".join(side_words[:word_i+1])
    prefix_before = source_text if not before_side else source_text + " " + before_side
    prefix_after = source_text + " " + through_side
    st = token_count(tok, prefix_before)
    en = token_count(tok, prefix_after)
    if st < en and st < SEQ_LEN:
        return st, min(en, SEQ_LEN)
    return None


def build_events(tok, per_category: int, seed: int = 22443023) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {"retained_content": [], "source_absent_content": [], "function_other": []}
    for r in read_pairs():
        src = r["source_text"]
        side = r["C_text"]
        source_norms = set(norm(w) for w in src.split() if norm(w))
        side_words = side.split()
        full_ids = tok(src + " " + side, add_special_tokens=False, truncation=True, max_length=SEQ_LEN)["input_ids"]
        if len(full_ids) < 4:
            continue
        for wi, w in enumerate(side_words):
            nw = norm(w)
            if not nw:
                continue
            span = target_span(tok, src, side_words, wi)
            if span is None or span[0] >= span[1] or span[1] > len(full_ids):
                continue
            if is_content(w) and nw in source_norms:
                cat = "retained_content"
            elif is_content(w) and nw not in source_norms:
                cat = "source_absent_content"
            else:
                cat = "function_other"
            # Drop one-piece punctuation-ish and too-long spans only lightly; keep real target distribution.
            buckets[cat].append({
                "pair_id": r["pair_id"], "source_text": src, "side_text": side,
                "word": w, "word_index": wi, "category": cat,
                "span": [span[0], span[1]], "n_pieces": span[1]-span[0],
                "input_ids": full_ids,
            })
    selected = []
    counts_before = {k: len(v) for k, v in buckets.items()}
    for cat, evs in buckets.items():
        rng.shuffle(evs)
        # Restrict per pair modestly to avoid over-representing long rows.
        by_pair = collections.Counter()
        keep = []
        for e in evs:
            if by_pair[e["pair_id"]] >= 2:
                continue
            keep.append(e)
            by_pair[e["pair_id"]] += 1
            if len(keep) >= per_category:
                break
        selected.extend(keep)
    rng.shuffle(selected)
    meta = {"counts_before_pair_cap": counts_before,
            "selected_counts": dict(collections.Counter(e["category"] for e in selected)),
            "per_category_requested": per_category,
            "seed": seed,
            "n_events": len(selected)}
    return selected, meta


def make_batch(events: list[dict[str, Any]], mask_id: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
    max_len = max(len(e["input_ids"]) for e in events)
    max_len = min(max_len, SEQ_LEN)
    input_batch = torch.full((len(events), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(events), max_len), -100, dtype=torch.long)
    cats = []
    npieces = []
    for i, e in enumerate(events):
        ids = list(e["input_ids"][:max_len])
        st, en = e["span"]
        en = min(en, max_len)
        input_batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        if st < en:
            labels[i, st:en] = torch.tensor(ids[st:en], dtype=torch.long)
            input_batch[i, st:en] = mask_id
        cats.append(e["category"])
        npieces.append(en-st)
    return input_batch, labels, cats, npieces


def eval_model(model_path: pathlib.Path, events: list[dict[str, Any]], tok, device: str, batch_size: int) -> dict[str, Any]:
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    losses_by_cat = collections.defaultdict(list)
    pieces_by_cat = collections.Counter()
    total_loss_sum_by_cat = collections.Counter()
    total_pieces_by_cat = collections.Counter()
    with torch.no_grad():
        for i in range(0, len(events), batch_size):
            batch = events[i:i+batch_size]
            input_ids, labels, cats, npieces = make_batch(batch, tok.mask_token_id, tok.pad_token_id)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attn = (input_ids != tok.pad_token_id).long().to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            vocab = logits.shape[-1]
            per_tok = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), reduction="none").view(labels.shape)
            mask = labels != -100
            for b, cat in enumerate(cats):
                vals = per_tok[b][mask[b]]
                if vals.numel() == 0:
                    continue
                mean_loss = float(vals.mean().detach().cpu())
                loss_sum = float(vals.sum().detach().cpu())
                losses_by_cat[cat].append(mean_loss)
                pieces_by_cat[cat] += int(vals.numel())
                total_loss_sum_by_cat[cat] += loss_sum
                total_pieces_by_cat[cat] += int(vals.numel())
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    out = {}
    for cat in ["retained_content", "source_absent_content", "function_other"]:
        out[cat] = {
            "event_loss": stats(losses_by_cat[cat]),
            "piece_count": int(pieces_by_cat[cat]),
            "piece_weighted_loss": round(total_loss_sum_by_cat[cat] / total_pieces_by_cat[cat], 6) if total_pieces_by_cat[cat] else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_category", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "existing_trajectory_semantic_denoising_probe.json"
    if out_json.exists() and not args.force:
        print(out_json)
        return
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    events, event_meta = build_events(tok, per_category=args.per_category)
    event_path = OUT_DIR / "probe_events.jsonl"
    with open(event_path, "w", encoding="utf-8") as f:
        for e in events:
            r = {k: v for k, v in e.items() if k != "input_ids"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    results: dict[str, Any] = {}
    for arm, root in RUNS.items():
        results[arm] = {}
        for ck in CHECKPOINTS:
            path = root / ck
            if not path.exists():
                results[arm][ck] = {"missing": str(path)}
                continue
            print(f"evaluating {arm}/{ck} on {args.device}", flush=True)
            results[arm][ck] = eval_model(path, events, tok, args.device, args.batch_size)
    # Deltas: compact minus repeat/adjbreak in loss (negative = compact better).
    deltas: dict[str, Any] = {}
    for ck in CHECKPOINTS:
        deltas[ck] = {}
        for ref in ["repeat", "adjbreak"]:
            key = f"compact_minus_{ref}"
            deltas[ck][key] = {}
            for cat in ["retained_content", "source_absent_content", "function_other"]:
                c = results.get("compact", {}).get(ck, {}).get(cat, {}).get("piece_weighted_loss")
                r = results.get(ref, {}).get(ck, {}).get(cat, {}).get("piece_weighted_loss")
                deltas[ck][key][cat] = round(c - r, 6) if c is not None and r is not None else None
    # Temporal changes within each arm.
    temporal: dict[str, Any] = {}
    for arm in RUNS:
        temporal[arm] = {}
        for cat in ["retained_content", "source_absent_content", "function_other"]:
            l20 = results[arm]["chck_20M"][cat]["piece_weighted_loss"]
            l60 = results[arm]["chck_60M"][cat]["piece_weighted_loss"]
            l100 = results[arm]["chck_100M"][cat]["piece_weighted_loss"]
            temporal[arm][cat] = {
                "loss_20M": l20, "loss_60M": l60, "loss_100M": l100,
                "delta_60_minus_20": round(l60-l20, 6),
                "delta_100_minus_60": round(l100-l60, 6),
                "delta_100_minus_20": round(l100-l20, 6),
            }
    payload = {
        "status": "EXISTING_TRAJECTORY_SEMANTIC_DENOISING_PROBE",
        "meaning": "Evaluation-only fixed-mask probe over existing compact/repeat/adjbreak checkpoints. Negative compact-minus-reference loss means compact predicts that target category better under identical source+compact input.",
        "inputs": {
            "csg": str(CSG), "csg_sha256": sha256_file(CSG),
            "tokenizer": str(TOKENIZER), "tokenizer_len": len(tok),
            "runs": {k: str(v) for k, v in RUNS.items()},
            "checkpoints": CHECKPOINTS, "seq_len": SEQ_LEN,
            "per_category": args.per_category, "batch_size": args.batch_size,
            "device": args.device,
        },
        "event_meta": event_meta,
        "event_file": str(event_path),
        "results": results,
        "deltas": deltas,
        "temporal": temporal,
        "elapsed_sec": round(time.time()-t0, 1),
        "scientific_cautions": [
            "Targets are sampled from the compact pair corpus, so this is a held-out mask/event probe, not a held-out text corpus.",
            "This probe does not by itself prove downstream utility; it anchors target-selection design to existing trajectory category timing.",
            "A small or late compact advantage warns against selecting Stage-1 arms by early MLM loss alone."
        ],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    # Compact console summary.
    print(json.dumps({"status": payload["status"], "out": str(out_json), "event_meta": event_meta,
                      "deltas": deltas, "elapsed_sec": payload["elapsed_sec"]}, indent=2))


if __name__ == "__main__":
    main()
