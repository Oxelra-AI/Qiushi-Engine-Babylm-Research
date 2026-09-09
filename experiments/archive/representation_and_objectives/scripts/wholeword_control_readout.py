#!/usr/bin/env python3
"""research readout: whole-word copied-content control vs research arms.

Evaluates the fixed compact-side denoising events (research) on:
  full             = research own_visible full loss
  drop_abs         = research drop_abs_content
  drop_copied_tok  = research drop_copied_matched (token-piece matched, partial-group)
  drop_copied_word = research drop_copied_content_wholeword (whole-word, matched piece mass)

The central comparison is drop_abs vs drop_copied_word: if removing source-absent labels still
worsens later source-absent denoising relative to removing whole matched copied-content groups,
the target-type-specific source-absent channel survives the stronger control.
Uses pair-cluster bootstrap on piece-weighted per-event losses.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import statistics
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
EVENT_FILE = ROOT / "data/existing_trajectory_denoising_probe/probe_events.jsonl"
TOKENIZER = ROOT / "training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
OUT_DIR = ROOT / "data/wholeword_control_readout"
SEQ_LEN = 256

ARMS = {
    "full": ROOT / "training/runs/crossview_own_visible_20M",
    "drop_abs": ROOT / "training/runs/target_selective_drop_abs_content_20M",
    "drop_copied_tok": ROOT / "training/runs/target_selective_drop_copied_matched_20M",
    "drop_copied_word": ROOT / "training/runs/target_selective_drop_copied_content_wholeword_20M",
}
CHECKPOINTS = ["chck_10M", "chck_20M"]
CATEGORIES = ["retained_content", "source_absent_content", "function_other"]


def read_events() -> list[dict[str, Any]]:
    events = []
    with EVENT_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def make_batch(events, tok):
    encs, max_len = [], 0
    for e in events:
        ids = tok(e["source_text"] + " " + e["side_text"], add_special_tokens=False, truncation=True, max_length=SEQ_LEN)["input_ids"]
        encs.append(ids)
        max_len = max(max_len, len(ids))
    input_batch = torch.full((len(events), max_len), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(events), max_len), -100, dtype=torch.long)
    cats = []
    for i, (e, ids) in enumerate(zip(events, encs)):
        st, en = e["span"]
        en = min(en, len(ids), max_len)
        input_batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        if st < en:
            labels[i, st:en] = torch.tensor(ids[st:en], dtype=torch.long)
            input_batch[i, st:en] = tok.mask_token_id
        cats.append(e["category"])
    return input_batch, labels, cats


def eval_events(model_path, events, tok, device, batch_size):
    """Return per-event mean loss and per-event piece count/sum by (category, event_index)."""
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    per_event = []  # list of dict: cat, event_index, loss_sum, pieces
    with torch.no_grad():
        for i in range(0, len(events), batch_size):
            batch = events[i : i + batch_size]
            input_ids, labels, cats = make_batch(batch, tok)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attn = (input_ids != tok.pad_token_id).long().to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            vocab = logits.shape[-1]
            per_tok = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), reduction="none", ignore_index=-100).view(labels.shape)
            mask = labels != -100
            for b, cat in enumerate(cats):
                vals = per_tok[b][mask[b]]
                if vals.numel() == 0:
                    continue
                per_event.append({
                    "cat": cat,
                    "event_index": i + b,
                    "pair_id": batch[b].get("pair_id"),
                    "loss_sum": float(vals.sum().detach().cpu()),
                    "pieces": int(vals.numel()),
                })
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return per_event


def piece_weighted(records):
    s = sum(r["loss_sum"] for r in records)
    p = sum(r["pieces"] for r in records)
    return s / p if p else None


def cluster_bootstrap(deltas_by_cluster: dict[Any, list[tuple[float, int]]], n_boot: int, seed: int):
    """deltas_by_cluster: cluster -> list of (loss_sum_delta, pieces). Piece-weighted bootstrap over clusters."""
    clusters = list(deltas_by_cluster.keys())
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        num = 0.0
        den = 0
        for _ in range(len(clusters)):
            c = clusters[rng.randrange(len(clusters))]
            for ds, p in deltas_by_cluster[c]:
                num += ds
                den += p
        boot.append(num / den if den else 0.0)
    boot.sort()
    n = len(boot)
    def q(pp):
        return boot[min(n - 1, max(0, int(round(pp * (n - 1)))))]
    return {
        "median": round(statistics.median(boot), 6),
        "p025": round(q(0.025), 6),
        "p05": round(q(0.05), 6),
        "p95": round(q(0.95), 6),
        "p975": round(q(0.975), 6),
        "fraction_gt0": round(sum(1 for x in boot if x > 0) / n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=226)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "wholeword_control_readout.json"
    if out_json.exists() and not args.force:
        print(out_json)
        return
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    events = read_events()
    event_counts = dict(collections.Counter(e["category"] for e in events))

    missing = []
    per_arm = {}
    for arm, root in ARMS.items():
        per_arm[arm] = {}
        for ck in CHECKPOINTS:
            mp = root / "hf_model" / ck
            if not mp.exists():
                missing.append(str(mp))
                per_arm[arm][ck] = None
                continue
            print(f"eval {arm}/{ck}", flush=True)
            per_arm[arm][ck] = eval_events(mp, events, tok, args.device, args.batch_size)

    # piece-weighted category losses per arm/ckpt
    pw = {}
    for arm in ARMS:
        pw[arm] = {}
        for ck in CHECKPOINTS:
            recs = per_arm[arm][ck]
            if recs is None:
                pw[arm][ck] = None
                continue
            pw[arm][ck] = {cat: round(piece_weighted([r for r in recs if r["cat"] == cat]) or 0.0, 6) for cat in CATEGORIES}

    # Save event-level losses for downstream exposure split and other paired analyses.
    event_losses_path = OUT_DIR / "event_losses.jsonl"
    with event_losses_path.open("w", encoding="utf-8") as f:
        by_arm_ck = {}
        for arm in ARMS:
            by_arm_ck[arm] = {}
            for ck in CHECKPOINTS:
                recs = per_arm[arm][ck]
                by_arm_ck[arm][ck] = {r["event_index"]: r for r in recs} if recs is not None else {}
        for i, e in enumerate(events):
            row = {
                "event_index": i,
                "pair_id": e.get("pair_id"),
                "category": e.get("category"),
                "word_index": e.get("word_index"),
                "n_pieces": e.get("n_pieces"),
                "word": e.get("word"),
            }
            for ck in CHECKPOINTS:
                row[ck] = {}
                for arm in ARMS:
                    r = by_arm_ck[arm][ck].get(i)
                    if r is not None:
                        row[ck][arm] = {
                            "loss_sum": r["loss_sum"],
                            "pieces": r["pieces"],
                            "loss_mean": r["loss_sum"] / max(1, r["pieces"]),
                        }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # central paired deltas with pair-cluster bootstrap for the key contrasts
    contrasts = [
        ("drop_abs_minus_drop_copied_word", "drop_abs", "drop_copied_word"),
        ("drop_abs_minus_drop_copied_tok", "drop_abs", "drop_copied_tok"),
        ("drop_abs_minus_full", "drop_abs", "full"),
        ("drop_copied_word_minus_full", "drop_copied_word", "full"),
        ("drop_copied_tok_minus_full", "drop_copied_tok", "full"),
    ]
    boot_out = {}
    for ck in CHECKPOINTS:
        boot_out[ck] = {}
        for name, a_arm, b_arm in contrasts:
            recs_a = per_arm[a_arm][ck]
            recs_b = per_arm[b_arm][ck]
            if recs_a is None or recs_b is None:
                boot_out[ck][name] = None
                continue
            # index by (cat, event_index)
            b_by_key = {(r["cat"], r["event_index"]): r for r in recs_b}
            boot_out[ck][name] = {}
            for cat in CATEGORIES:
                by_cluster: dict[Any, list[tuple[float, int]]] = collections.defaultdict(list)
                num = 0.0
                den = 0
                n_ev = 0
                for r in recs_a:
                    if r["cat"] != cat:
                        continue
                    key = (r["cat"], r["event_index"])
                    rb = b_by_key.get(key)
                    if rb is None or rb["pieces"] != r["pieces"]:
                        continue
                    delta_sum = r["loss_sum"] - rb["loss_sum"]
                    num += delta_sum
                    den += r["pieces"]
                    n_ev += 1
                    cluster = r.get("pair_id") or f"ev{r['event_index']}"
                    by_cluster[cluster].append((delta_sum, r["pieces"]))
                pwd = num / den if den else None
                boot = cluster_bootstrap(by_cluster, args.n_boot, args.seed) if by_cluster else None
                boot_out[ck][name][cat] = {
                    "n_events": n_ev,
                    "n_clusters": len(by_cluster),
                    "piece_weighted_delta": round(pwd, 6) if pwd is not None else None,
                    "piece_cluster_bootstrap": boot,
                }

    payload = {
        "status": "WHOLEWORD_CONTROL_READOUT",
        "meaning": "Positive drop_abs_minus_drop_copied_word means removing source-absent content labels hurt later source-absent denoising more than removing whole matched copied-content target groups. This is the stronger word-level control after research.",
        "inputs": {
            "event_file": str(EVENT_FILE),
            "events": len(events),
            "event_counts": event_counts,
            "tokenizer": str(TOKENIZER),
            "arms": {k: str(v) for k, v in ARMS.items()},
            "checkpoints": CHECKPOINTS,
            "n_boot": args.n_boot,
            "device": args.device,
        },
        "missing": missing,
        "event_losses_jsonl": str(event_losses_path),
        "piece_weighted_category_loss": pw,
        "contrast_bootstrap": boot_out,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out_json), "missing": missing, "pw": pw, "contrast_bootstrap": boot_out, "elapsed_sec": payload["elapsed_sec"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
