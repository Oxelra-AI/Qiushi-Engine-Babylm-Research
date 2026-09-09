#!/usr/bin/env python3
"""Paired event-level bootstrap for research target-selective screen."""
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
OUT_DIR = ROOT / "data/target_selective_event_bootstrap"
ARMS = {
    "full": ROOT / "training/runs/crossview_own_visible_20M/hf_model",
    "drop_abs": ROOT / "training/runs/target_selective_drop_abs_content_20M/hf_model",
    "drop_copied": ROOT / "training/runs/target_selective_drop_copied_matched_20M/hf_model",
}
CHECKPOINTS = ["chck_10M", "chck_20M"]
CATEGORIES = ["retained_content", "source_absent_content", "function_other"]
SEQ_LEN = 256


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(float(x) for x in xs)
    n = len(s)
    def q(p: float) -> float:
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {"p025": round(q(0.025), 6), "p05": round(q(0.05), 6), "median": round(q(0.5), 6), "p95": round(q(0.95), 6), "p975": round(q(0.975), 6)}


def read_events() -> list[dict[str, Any]]:
    evs = []
    with EVENT_FILE.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                e = json.loads(line)
                e["event_index"] = i
                evs.append(e)
    return evs


def make_batch(events: list[dict[str, Any]], tok):
    encs = []
    max_len = 0
    for e in events:
        ids = tok(e["source_text"] + " " + e["side_text"], add_special_tokens=False, truncation=True, max_length=SEQ_LEN)["input_ids"]
        encs.append(ids)
        max_len = max(max_len, len(ids))
    x = torch.full((len(events), max_len), tok.pad_token_id, dtype=torch.long)
    y = torch.full((len(events), max_len), -100, dtype=torch.long)
    for i, (e, ids) in enumerate(zip(events, encs)):
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        st, en = e["span"]
        en = min(en, len(ids), max_len)
        if st < en:
            y[i, st:en] = torch.tensor(ids[st:en], dtype=torch.long)
            x[i, st:en] = tok.mask_token_id
    return x, y


def eval_arm_ck(model_path: pathlib.Path, events: list[dict[str, Any]], tok, device: str, batch_size: int) -> dict[int, dict[str, Any]]:
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    out: dict[int, dict[str, Any]] = {}
    with torch.no_grad():
        for i in range(0, len(events), batch_size):
            batch = events[i:i+batch_size]
            x, y = make_batch(batch, tok)
            x = x.to(device)
            y = y.to(device)
            att = (x != tok.pad_token_id).long().to(device)
            logits = model(input_ids=x, attention_mask=att).logits
            ptl = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), reduction="none", ignore_index=-100).view(y.shape)
            m = y != -100
            for b, e in enumerate(batch):
                vals = ptl[b][m[b]]
                out[int(e["event_index"])] = {
                    "loss_mean": float(vals.mean().detach().cpu()),
                    "loss_sum": float(vals.sum().detach().cpu()),
                    "pieces": int(vals.numel()),
                }
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def weighted_delta(records: list[dict[str, Any]], lhs: str, rhs: str) -> float:
    num = sum(r[lhs]["loss_sum"] - r[rhs]["loss_sum"] for r in records)
    den = sum(r[lhs]["pieces"] for r in records)
    return num / den if den else float("nan")


def mean_event_delta(records: list[dict[str, Any]], lhs: str, rhs: str) -> float:
    return statistics.mean(r[lhs]["loss_mean"] - r[rhs]["loss_mean"] for r in records)


def bootstrap(records: list[dict[str, Any]], lhs: str, rhs: str, n_boot: int, seed: int) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_pair[r["pair_id"]].append(r)
    pairs = sorted(by_pair)
    rng = random.Random(seed)
    ev_vals = []
    wt_vals = []
    for _ in range(n_boot):
        sample_records = []
        for _j in range(len(pairs)):
            sample_records.extend(by_pair[rng.choice(pairs)])
        ev_vals.append(mean_event_delta(sample_records, lhs, rhs))
        wt_vals.append(weighted_delta(sample_records, lhs, rhs))
    orig_ev = mean_event_delta(records, lhs, rhs)
    orig_wt = weighted_delta(records, lhs, rhs)
    return {
        "n_events": len(records),
        "n_pair_clusters": len(pairs),
        "event_mean_delta": round(orig_ev, 6),
        "piece_weighted_delta": round(orig_wt, 6),
        "event_cluster_bootstrap": quantiles(ev_vals),
        "piece_cluster_bootstrap": quantiles(wt_vals),
        "p_event_delta_gt0": round(sum(v > 0 for v in ev_vals) / len(ev_vals), 4),
        "p_piece_delta_gt0": round(sum(v > 0 for v in wt_vals) / len(wt_vals), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "target_selective_event_bootstrap.json"
    out_jsonl = OUT_DIR / "event_losses.jsonl"
    if out_json.exists() and not args.force:
        print(out_json)
        return
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    events = read_events()
    losses: dict[str, dict[str, dict[int, dict[str, Any]]]] = {arm: {} for arm in ARMS}
    for arm, root in ARMS.items():
        for ck in CHECKPOINTS:
            mp = root / ck
            if not mp.exists():
                raise FileNotFoundError(mp)
            print(f"eval {arm}/{ck}", flush=True)
            losses[arm][ck] = eval_arm_ck(mp, events, tok, args.device, args.batch_size)
    # Save compact event records.
    with out_jsonl.open("w", encoding="utf-8") as f:
        for e in events:
            rec = {"event_index": e["event_index"], "pair_id": e["pair_id"], "category": e["category"], "word": e.get("word"), "n_pieces": e.get("n_pieces")}
            for ck in CHECKPOINTS:
                rec[ck] = {arm: losses[arm][ck][e["event_index"]] for arm in ARMS}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    records_by_ck_cat: dict[str, dict[str, list[dict[str, Any]]]] = {ck: {cat: [] for cat in CATEGORIES} for ck in CHECKPOINTS}
    for e in events:
        for ck in CHECKPOINTS:
            r = {"pair_id": e["pair_id"], "category": e["category"]}
            for arm in ARMS:
                r[arm] = losses[arm][ck][e["event_index"]]
            records_by_ck_cat[ck][e["category"]].append(r)
    comparisons = {
        "drop_abs_minus_full": ("drop_abs", "full"),
        "drop_copied_minus_full": ("drop_copied", "full"),
        "drop_abs_minus_drop_copied": ("drop_abs", "drop_copied"),
    }
    summary: dict[str, Any] = {}
    for ck in CHECKPOINTS:
        summary[ck] = {}
        for cat in CATEGORIES:
            summary[ck][cat] = {}
            recs = records_by_ck_cat[ck][cat]
            for name, (lhs, rhs) in comparisons.items():
                summary[ck][cat][name] = bootstrap(recs, lhs, rhs, args.n_boot, seed=225000 + 1000*CHECKPOINTS.index(ck) + 17*CATEGORIES.index(cat) + len(name))
    payload = {
        "status": "TARGET_SELECTIVE_EVENT_BOOTSTRAP",
        "meaning": "Paired fixed-event loss deltas with pair-cluster bootstrap. Positive drop_abs_minus_drop_copied means removing source-absent compact-content labels hurt more than removing the same number of copied compact-side labels.",
        "inputs": {"events": len(events), "event_file": str(EVENT_FILE), "tokenizer": str(TOKENIZER), "arms": {k: str(v) for k, v in ARMS.items()}, "checkpoints": CHECKPOINTS, "n_boot": args.n_boot, "device": args.device, "batch_size": args.batch_size},
        "event_losses_jsonl": str(out_jsonl),
        "summary": summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Print only main comparison for readability.
    main_view = {ck: {cat: summary[ck][cat]["drop_abs_minus_drop_copied"] for cat in CATEGORIES} for ck in CHECKPOINTS}
    print(json.dumps({"status": payload["status"], "out": str(out_json), "main_drop_abs_minus_drop_copied": main_view, "elapsed_sec": payload["elapsed_sec"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
