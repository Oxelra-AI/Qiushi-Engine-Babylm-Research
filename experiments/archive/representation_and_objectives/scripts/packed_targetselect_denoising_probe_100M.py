#!/usr/bin/env python3
"""research: local compact-side denoising readout for 100M packed target-selective endpoints.

This complements the official-compatible endpoint signature.  It asks whether the
100M source-absent target-label deletion specifically raises compact-side
source-absent content loss, relative to the whole-word copied-content deletion,
on both the fixed in-training compact probe and source-disjoint held-out compact
pairs.

Positive `drop_abs_minus_drop_copied_word` on `source_absent_content` means the
source-absent labels were more useful for that compact-content prediction
category than the matched copied-content labels.  Retained content and function
words should not move in the same positive direction if the effect is specific.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import statistics
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

USER_ROOT = pathlib.Path(".").resolve()
ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
SCRIPT_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crossview_deberta_trainer_v2 as base  # noqa: E402

SEQ_LEN = 256
TOKENIZER = ROOT / "training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
TRAIN_FIXED_EVENTS = ROOT / "data/existing_trajectory_denoising_probe/probe_events.jsonl"
SOURCE_DISJOINT_EVENTS = ROOT / "data/source_disjoint_target_probe/source_disjoint_probe_events.jsonl"
DEFAULT_OUT = ROOT / "data/packed_targetselect_denoising_probe_100M"

ARMS = {
    "full_compact_100M": pathlib.Path("experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M"),
    "drop_abs_100M": ROOT / "training/runs/packed_drop_abs_content_100M_fast/hf_model/chck_100M",
    "drop_copied_word_100M": ROOT / "training/runs/packed_drop_copied_content_wholeword_100M_fast/hf_model/chck_100M",
}
CONTRASTS = [
    ("drop_abs_minus_drop_copied_word", "drop_abs_100M", "drop_copied_word_100M"),
    ("drop_abs_minus_full", "drop_abs_100M", "full_compact_100M"),
    ("drop_copied_word_minus_full", "drop_copied_word_100M", "full_compact_100M"),
]
CATEGORIES = ["retained_content", "source_absent_content", "function_other"]


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def q_stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(float(x) for x in xs)
    n = len(s)

    def q(p: float) -> float:
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "n": n,
        "mean": round(statistics.mean(s), 6),
        "median": round(statistics.median(s), 6),
        "p05": round(q(0.05), 6),
        "p25": round(q(0.25), 6),
        "p75": round(q(0.75), 6),
        "p95": round(q(0.95), 6),
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
    }


def stable_u64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def load_train_fixed_events() -> list[dict[str, Any]]:
    events = []
    with TRAIN_FIXED_EVENTS.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            d["eval_set"] = "train_fixed_probe"
            d["event_uid"] = f"train_fixed_probe|{i}|{d.get('pair_id')}|wi{d.get('word_index')}|{d.get('category')}"
            events.append(d)
    return events


def load_source_disjoint_events(args: argparse.Namespace) -> list[dict[str, Any]]:
    events = []
    keep_sets = set(args.heldout_sets.split(",")) if args.heldout_sets else None
    with SOURCE_DISJOINT_EVENTS.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            if keep_sets is not None and str(d.get("eval_set")) not in keep_sets:
                continue
            d.setdefault("event_uid", f"heldout|{i}|{d.get('pair_id')}|wi{d.get('word_index')}|{d.get('category')}")
            events.append(d)
    return events


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_set_cat: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for e in events:
        by_set_cat[(str(e.get("eval_set", "unknown")), str(e.get("category", "unknown")))].append(e)
    out: dict[str, Any] = {
        "n_events": len(events),
        "n_pairs": len(set(str(e.get("pair_id")) for e in events)),
        "eval_sets": sorted(set(str(e.get("eval_set", "unknown")) for e in events)),
        "category_counts": dict(collections.Counter(str(e.get("category", "unknown")) for e in events)),
        "by_eval_set_category": {},
    }
    for (es, cat), rows in sorted(by_set_cat.items()):
        out["by_eval_set_category"].setdefault(es, {})[cat] = {
            "n_events": len(rows),
            "n_pairs": len(set(str(e.get("pair_id")) for e in rows)),
            "piece_total": int(sum(int(e.get("n_pieces", 0)) for e in rows)),
            "n_pieces": q_stats([float(e.get("n_pieces", 0)) for e in rows]),
        }
    return out


def prepare_event(e: dict[str, Any], tok) -> dict[str, Any] | None:
    src = str(e.get("source_text", ""))
    side = str(e.get("side_text", e.get("rewrite_text", "")))
    if not src or not side:
        return None
    enc = tok(src + " " + side, add_special_tokens=False, truncation=True, max_length=SEQ_LEN)
    ids = list(enc["input_ids"])
    span = e.get("span")
    if not isinstance(span, list) or len(span) != 2:
        return None
    st, en = int(span[0]), int(span[1])
    en = min(en, len(ids), SEQ_LEN)
    if st < 0 or st >= en or st >= len(ids):
        return None
    out = dict(e)
    out["input_ids"] = ids
    out["span"] = [st, en]
    out["n_pieces_realized"] = en - st
    return out


def make_batch(events: list[dict[str, Any]], tok) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    max_len = min(SEQ_LEN, max(len(e["input_ids"]) for e in events))
    input_batch = torch.full((len(events), max_len), int(tok.pad_token_id), dtype=torch.long)
    labels = torch.full((len(events), max_len), -100, dtype=torch.long)
    kept = []
    for i, e in enumerate(events):
        ids = list(e["input_ids"][:max_len])
        st, en = int(e["span"][0]), int(e["span"][1])
        en = min(en, len(ids), max_len)
        if st >= en:
            continue
        input_batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, st:en] = torch.tensor(ids[st:en], dtype=torch.long)
        input_batch[i, st:en] = int(tok.mask_token_id)
        kept.append(e)
    return input_batch, labels, events


def eval_model_events(model_path: pathlib.Path, events: list[dict[str, Any]], tok, device: str, batch_size: int) -> list[dict[str, Any]]:
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    per_event: list[dict[str, Any]] = []
    with torch.no_grad():
        for i in range(0, len(events), batch_size):
            raw = events[i : i + batch_size]
            input_ids, labels, _ = make_batch(raw, tok)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attn = (input_ids != int(tok.pad_token_id)).long().to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            vocab = logits.shape[-1]
            per_tok = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), reduction="none", ignore_index=-100).reshape(labels.shape)
            mask = labels != -100
            for b, e in enumerate(raw):
                vals = per_tok[b][mask[b]]
                if vals.numel() == 0:
                    continue
                per_event.append({
                    "event_uid": str(e["event_uid"]),
                    "eval_set": str(e.get("eval_set", "unknown")),
                    "category": str(e.get("category", "unknown")),
                    "pair_id": str(e.get("pair_id", e.get("event_uid"))),
                    "loss_sum": float(vals.sum().detach().cpu()),
                    "pieces": int(vals.numel()),
                })
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return per_event


def piece_weighted(records: list[dict[str, Any]], value_key: str = "loss_sum") -> float | None:
    den = sum(int(r["pieces"]) for r in records)
    if den == 0:
        return None
    return sum(float(r[value_key]) for r in records) / den


def cluster_bootstrap(delta_records: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any] | None:
    by_cluster: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
    for r in delta_records:
        by_cluster[str(r.get("pair_id") or r.get("event_uid"))].append((float(r["delta_sum"]), int(r["pieces"])))
    clusters = list(by_cluster)
    if not clusters:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        num = 0.0
        den = 0
        for _ in range(len(clusters)):
            c = clusters[rng.randrange(len(clusters))]
            for ds, p in by_cluster[c]:
                num += ds
                den += p
        vals.append(num / den if den else 0.0)
    vals.sort()
    n = len(vals)

    def q(p: float) -> float:
        return vals[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "median": round(statistics.median(vals), 6),
        "p025": round(q(0.025), 6),
        "p05": round(q(0.05), 6),
        "p95": round(q(0.95), 6),
        "p975": round(q(0.975), 6),
        "fraction_gt0": round(sum(1 for x in vals if x > 0) / n, 4),
    }


def summarize_losses(per_arm: dict[str, list[dict[str, Any]]], events: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    eval_sets = sorted(set(str(e.get("eval_set", "unknown")) for e in events))
    out: dict[str, Any] = {"piece_weighted_category_loss": {}, "contrasts": {}, "decision_fields": {}}
    for arm, recs in per_arm.items():
        by_key: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
        for r in recs:
            by_key[(str(r["eval_set"]), str(r["category"]))].append(r)
        out["piece_weighted_category_loss"][arm] = {}
        for es in eval_sets:
            out["piece_weighted_category_loss"][arm][es] = {}
            for cat in CATEGORIES:
                rr = by_key.get((es, cat), [])
                v = piece_weighted(rr)
                out["piece_weighted_category_loss"][arm][es][cat] = {
                    "n_events": len(rr),
                    "n_pairs": len(set(str(x.get("pair_id")) for x in rr)),
                    "piece_weighted_loss": round(v, 6) if v is not None else None,
                }
    for cname, a, b in CONTRASTS:
        a_by = {r["event_uid"]: r for r in per_arm.get(a, [])}
        b_by = {r["event_uid"]: r for r in per_arm.get(b, [])}
        out["contrasts"][cname] = {}
        for es in eval_sets:
            out["contrasts"][cname][es] = {}
            for cat in CATEGORIES:
                delta_records = []
                for uid, ar in a_by.items():
                    br = b_by.get(uid)
                    if br is None:
                        continue
                    if str(ar["eval_set"]) != es or str(ar["category"]) != cat:
                        continue
                    if int(ar["pieces"]) != int(br["pieces"]):
                        continue
                    delta_records.append({
                        "event_uid": uid,
                        "pair_id": ar.get("pair_id"),
                        "delta_sum": float(ar["loss_sum"]) - float(br["loss_sum"]),
                        "pieces": int(ar["pieces"]),
                    })
                v = piece_weighted([{"loss_sum": r["delta_sum"], "pieces": r["pieces"]} for r in delta_records])
                out["contrasts"][cname][es][cat] = {
                    "n_events": len(delta_records),
                    "n_pairs": len(set(str(r.get("pair_id")) for r in delta_records)),
                    "piece_weighted_delta": round(v, 6) if v is not None else None,
                    "pair_cluster_bootstrap": cluster_bootstrap(delta_records, args.n_boot, args.seed + stable_u64(cname + es + cat) % 1_000_000),
                }
    for es in eval_sets:
        for cname in ["drop_abs_minus_drop_copied_word", "drop_abs_minus_full"]:
            key = f"{es}/{cname}/source_absent_content"
            out["decision_fields"][key] = out["contrasts"].get(cname, {}).get(es, {}).get("source_absent_content")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--heldout_sets", default="source_disjoint_quality,doc_disjoint_quality,doc_disjoint_all_accepted")
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=23043023)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "packed_targetselect_denoising_probe_100M.json"
    per_event_jsonl = out_dir / "packed_targetselect_denoising_event_losses_100M.jsonl"
    if out_json.exists() and not args.force and not args.preflight_only:
        print(out_json)
        return

    model_status = {}
    for name, path in ARMS.items():
        model_status[name] = {
            "path": str(path),
            "exists": path.exists(),
            "config_exists": (path / "config.json").exists(),
            "model_safetensors_exists": (path / "model.safetensors").exists(),
        }
    waiting = [k for k, v in model_status.items() if not v["exists"]]

    raw_events = load_train_fixed_events() + load_source_disjoint_events(args)
    tok = base.make_portable_tokenizer(str(TOKENIZER))
    events = []
    skipped = collections.Counter()
    for e in raw_events:
        pe = prepare_event(e, tok)
        if pe is None:
            skipped["invalid_event"] += 1
        else:
            events.append(pe)
    preflight = {
        "status": "PACKED_TARGETSELECT_DENOISING_100M_PREFLIGHT",
        "meaning": "Prepared fixed in-training and source-disjoint compact-side target events for the 100M packed target-selective endpoint local readout.",
        "inputs": {
            "tokenizer": str(TOKENIZER),
            "train_fixed_events": str(TRAIN_FIXED_EVENTS),
            "train_fixed_events_sha256": sha256_file(TRAIN_FIXED_EVENTS),
            "source_disjoint_events": str(SOURCE_DISJOINT_EVENTS),
            "source_disjoint_events_sha256": sha256_file(SOURCE_DISJOINT_EVENTS),
            "heldout_sets": args.heldout_sets,
        },
        "model_status": model_status,
        "waiting_for_models": waiting,
        "event_summary": summarize_events(events),
        "skipped": dict(skipped),
        "elapsed_preflight_sec": round(time.time() - t0, 1),
    }
    (out_dir / "packed_targetselect_denoising_probe_100M_preflight.json").write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.preflight_only or waiting:
        status = "PACKED_TARGETSELECT_DENOISING_100M_WAITING_FOR_MODELS" if waiting else preflight["status"]
        payload = dict(preflight)
        payload["status"] = status
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": status, "out": str(out_json), "waiting_for_models": waiting, "event_summary": payload["event_summary"]}, indent=2, ensure_ascii=False), flush=True)
        return

    per_arm = {}
    for name, path in ARMS.items():
        print(json.dumps({"event": "eval_start", "arm": name, "model": str(path), "n_events": len(events), "device": args.device}), flush=True)
        per_arm[name] = eval_model_events(path, events, tok, args.device, args.batch_size)
        print(json.dumps({"event": "eval_done", "arm": name, "records": len(per_arm[name])}), flush=True)

    by_arm = {arm: {r["event_uid"]: r for r in recs} for arm, recs in per_arm.items()}
    with per_event_jsonl.open("w", encoding="utf-8") as f:
        for e in events:
            row = {
                "event_uid": e["event_uid"],
                "eval_set": e.get("eval_set"),
                "pair_id": e.get("pair_id"),
                "category": e.get("category"),
                "word": e.get("word"),
                "word_index": e.get("word_index"),
                "n_pieces": e.get("n_pieces"),
                "n_pieces_realized": e.get("n_pieces_realized"),
            }
            row["arms"] = {}
            for arm, idx in by_arm.items():
                r = idx.get(e["event_uid"])
                if r is not None:
                    row["arms"][arm] = {"loss_sum": r["loss_sum"], "pieces": r["pieces"], "loss_mean": r["loss_sum"] / max(1, r["pieces"])}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_losses(per_arm, events, args)
    payload = dict(preflight)
    payload.update({
        "status": "PACKED_TARGETSELECT_DENOISING_100M_DONE",
        "meaning": "Local fixed-event compact-side denoising readout for 100M packed source-absent versus copied-content target deletion endpoints; interpret together with official Supplement/EWoK transition signature.",
        "device": args.device,
        "arms": {k: str(v) for k, v in ARMS.items()},
        "event_losses_jsonl": str(per_event_jsonl),
        "eval_summary": summary,
        "elapsed_total_sec": round(time.time() - t0, 1),
    })
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out_json), "decision_fields": summary["decision_fields"], "elapsed_sec": payload["elapsed_total_sec"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
