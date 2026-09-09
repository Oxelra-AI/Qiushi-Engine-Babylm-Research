#!/usr/bin/env python3
"""research readout for target-selective compact-view intervention.

Reads the already-trained research full own-visible 20M arm plus research target-selective
arms, evaluates all checkpoints on the fixed compact-side semantic denoising events from
research, and compares training logs/target accounting.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
EVENT_FILE = ROOT / "data/existing_trajectory_denoising_probe/probe_events.jsonl"
TOKENIZER = ROOT / "training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
OUT_DIR = ROOT / "data/target_selective_readout"
SEQ_LEN = 256

DEFAULT_ARMS = {
    "full_step221_own_visible": ROOT / "training/runs/crossview_own_visible_20M",
    "drop_abs_content": ROOT / "training/runs/target_selective_drop_abs_content_20M",
    "drop_copied_matched": ROOT / "training/runs/target_selective_drop_copied_matched_20M",
}
CHECKPOINTS = ["chck_10M", "chck_20M"]
CATEGORIES = ["retained_content", "source_absent_content", "function_other"]


def stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(float(x) for x in xs)
    n = len(s)
    def q(p: float):
        if n == 1:
            return s[0]
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {
        "n": n,
        "mean": round(statistics.mean(s), 6),
        "median": round(statistics.median(s), 6),
        "p10": round(q(0.10), 6),
        "p25": round(q(0.25), 6),
        "p75": round(q(0.75), 6),
        "p90": round(q(0.90), 6),
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
    }


def read_events(limit: int | None = None) -> list[dict[str, Any]]:
    events = []
    with EVENT_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                events.append(e)
                if limit and len(events) >= limit:
                    break
    return events


def make_batch(events: list[dict[str, Any]], tok) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    encs = []
    max_len = 0
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


def eval_model(model_path: pathlib.Path, events: list[dict[str, Any]], tok, device: str, batch_size: int) -> dict[str, Any]:
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    losses_by_cat = collections.defaultdict(list)
    loss_sum_by_cat = collections.Counter()
    pieces_by_cat = collections.Counter()
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
                losses_by_cat[cat].append(float(vals.mean().detach().cpu()))
                loss_sum_by_cat[cat] += float(vals.sum().detach().cpu())
                pieces_by_cat[cat] += int(vals.numel())
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    out = {}
    for cat in CATEGORIES:
        out[cat] = {
            "event_loss": stats(losses_by_cat[cat]),
            "piece_count": int(pieces_by_cat[cat]),
            "piece_weighted_loss": round(loss_sum_by_cat[cat] / pieces_by_cat[cat], 6) if pieces_by_cat[cat] else None,
        }
    return out


def read_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def training_log_summary(run_root: pathlib.Path) -> dict[str, Any]:
    p = run_root / "training_log.jsonl"
    if not p.exists():
        return {"missing": str(p)}
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return {"rows": 0}
    # last window carries the final local training behavior; use up to last 50 steps.
    tail = rows[-min(50, len(rows)):]
    keys = ["loss", "filler", "source", "rw_copied", "rw_abs_content", "rw_abs_other"]
    summary: dict[str, Any] = {"rows": len(rows), "first": rows[0], "last": rows[-1]}
    for k in keys:
        vals = [r[k] for r in tail if k in r]
        if vals:
            summary[f"tail50_{k}_mean"] = round(statistics.mean(vals), 6)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--limit_events", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "target_selective_readout.json"
    if out_json.exists() and not args.force:
        print(out_json)
        return
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    events = read_events(args.limit_events or None)
    event_counts = dict(collections.Counter(e["category"] for e in events))
    results: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    train_logs: dict[str, Any] = {}
    missing = []
    for arm, root in DEFAULT_ARMS.items():
        metrics[arm] = read_json(root / "scientific_metrics.json")
        train_logs[arm] = training_log_summary(root)
        results[arm] = {}
        for ck in CHECKPOINTS:
            mp = root / "hf_model" / ck
            if not mp.exists():
                results[arm][ck] = {"missing": str(mp)}
                missing.append(str(mp))
                continue
            print(f"evaluating {arm}/{ck}", flush=True)
            results[arm][ck] = eval_model(mp, events, tok, args.device, args.batch_size)
    deltas: dict[str, Any] = {}
    for ck in CHECKPOINTS:
        deltas[ck] = {}
        full = results.get("full_step221_own_visible", {}).get(ck, {})
        for arm in ["drop_abs_content", "drop_copied_matched"]:
            deltas[ck][f"{arm}_minus_full"] = {}
            cur = results.get(arm, {}).get(ck, {})
            for cat in CATEGORIES:
                a = cur.get(cat, {}).get("piece_weighted_loss")
                f = full.get(cat, {}).get("piece_weighted_loss")
                deltas[ck][f"{arm}_minus_full"][cat] = round(a - f, 6) if a is not None and f is not None else None
        # positive value means removing absent-content labels leaves worse denoising than removing matched copied labels.
        deltas[ck]["drop_abs_minus_drop_copied"] = {}
        for cat in CATEGORIES:
            a = results.get("drop_abs_content", {}).get(ck, {}).get(cat, {}).get("piece_weighted_loss")
            c = results.get("drop_copied_matched", {}).get(ck, {}).get(cat, {}).get("piece_weighted_loss")
            deltas[ck]["drop_abs_minus_drop_copied"][cat] = round(a - c, 6) if a is not None and c is not None else None
    payload = {
        "status": "TARGET_SELECTIVE_READOUT",
        "meaning": "Fixed-event compact-side semantic denoising readout. Positive drop_abs_minus_drop_copied means source-absent target loss was more useful than an equal count of copied-side target loss for that evaluated category.",
        "inputs": {
            "event_file": str(EVENT_FILE),
            "tokenizer": str(TOKENIZER),
            "tokenizer_len": len(tok),
            "events": len(events),
            "event_counts": event_counts,
            "arms": {k: str(v) for k, v in DEFAULT_ARMS.items()},
            "checkpoints": CHECKPOINTS,
            "device": args.device,
            "batch_size": args.batch_size,
        },
        "missing": missing,
        "training_metrics": metrics,
        "training_log_summary": train_logs,
        "probe_results": results,
        "deltas": deltas,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out_json), "missing": missing, "deltas": deltas, "elapsed_sec": payload["elapsed_sec"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
