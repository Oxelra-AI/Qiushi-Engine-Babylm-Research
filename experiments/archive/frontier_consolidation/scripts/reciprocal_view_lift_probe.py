#!/usr/bin/env python3
"""research: reciprocal cross-view conditioning probe.

Forward-only, benchmark-independent probe for the mechanism suggested by the
causal-transfer result: bidirectional MLM can condition masked tokens in each
view on the other view, while a decoder-only causal objective only lets the
later segment use the earlier segment.

The script samples legal source/compact pairs and compares token NLL with a
paired context against NLL with the target side alone. Positive lift means the
opposite view makes the target token easier. It can also construct an exact
repeat control by replacing the compact view with the first N source words.

This is a probe/scaffold: full scientific use should run on enough pairs and
predeclared checkpoints after the pending DeBERTa trajectory comparison is
finished. The default dry-run performs only token/target accounting.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import time
from typing import Any, Iterable

import torch

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_PAIRS = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
DEFAULT_CHCK82 = ROOT / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_piece(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def wc(text: str) -> int:
    return len(text.split())


def read_pairs(path: pathlib.Path, max_scan: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_scan is not None and i >= max_scan:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("source_text") and r.get("rewrite_text"):
                rows.append(r)
    return rows


def make_repeat_text(source_text: str, target_words: int) -> str:
    words = source_text.split()
    if target_words <= 0:
        target_words = max(1, len(words) // 2)
    return " ".join(words[: min(target_words, len(words))])


def get_special_id(tokenizer, candidates: Iterable[str], fallback: int | None = None) -> int | None:
    for attr in candidates:
        val = getattr(tokenizer, attr, None)
        if val is not None:
            return int(val)
    return fallback


def encode_no_special(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def token_text(tokenizer, tid: int) -> str:
    try:
        return tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
    except Exception:
        try:
            return str(tokenizer.convert_ids_to_tokens(int(tid)))
        except Exception:
            return str(tid)


def candidate_positions(tokenizer, ids: list[int], max_candidates: int, rng: random.Random) -> list[int]:
    cand: list[int] = []
    specials = set(getattr(tokenizer, "all_special_ids", []) or [])
    for i, tid in enumerate(ids):
        if int(tid) in specials:
            continue
        txt = token_text(tokenizer, int(tid))
        if any(ch.isalnum() for ch in txt):
            cand.append(i)
    if len(cand) <= max_candidates:
        return cand
    return sorted(rng.sample(cand, max_candidates))


def build_context(bos: int | None, eos: int | None, a_ids: list[int], b_ids: list[int] | None = None) -> tuple[list[int], int, int | None]:
    ids: list[int] = []
    if bos is not None:
        ids.append(bos)
    a_start = len(ids)
    ids.extend(a_ids)
    if eos is not None:
        ids.append(eos)
    b_start = None
    if b_ids is not None:
        b_start = len(ids)
        ids.extend(b_ids)
        if eos is not None:
            ids.append(eos)
    return ids, a_start, b_start


def nll_mlm(model, tokenizer, device: torch.device, ids: list[int], pos: int, orig_id: int, mask_id: int) -> float:
    x = torch.tensor([ids], dtype=torch.long, device=device)
    x[0, pos] = int(mask_id)
    with torch.no_grad():
        logits = model(input_ids=x).logits[0, pos].float()
        lp = torch.log_softmax(logits, dim=-1)[int(orig_id)]
    return float(-lp.detach().cpu())


def nll_causal(model, device: torch.device, ids: list[int], pos: int, orig_id: int) -> float:
    if pos <= 0:
        return float("nan")
    x = torch.tensor([ids[: pos + 1]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=x).logits[0, pos - 1].float()
        lp = torch.log_softmax(logits, dim=-1)[int(orig_id)]
    return float(-lp.detach().cpu())


def mean(xs: list[float]) -> float | None:
    ys = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not ys:
        return None
    return float(sum(ys) / len(ys))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_records": len(records), "groups": {}}
    keys = ["model_type", "pair_type", "order", "target_segment", "copy_by_id", "copy_by_text"]
    for keyset in [
        ("model_type", "pair_type", "target_segment"),
        ("model_type", "pair_type", "order", "target_segment"),
        ("model_type", "pair_type", "target_segment", "copy_by_id"),
        ("model_type", "pair_type", "target_segment", "copy_by_text"),
    ]:
        bucket: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
        for r in records:
            bucket[tuple(r.get(k) for k in keyset)].append(r)
        label = "/".join(keyset)
        out["groups"][label] = []
        for vals, rs in sorted(bucket.items(), key=lambda kv: str(kv[0])):
            lifts = [float(r["lift_nll_sideonly_minus_paired"]) for r in rs if math.isfinite(float(r.get("lift_nll_sideonly_minus_paired", float("nan"))))]
            out["groups"][label].append({
                "key": dict(zip(keyset, vals)),
                "n": len(rs),
                "mean_lift": mean(lifts),
                "median_lift": float(statistics.median(lifts)) if lifts else None,
                "positive_lift_frac": float(sum(1 for x in lifts if x > 0) / len(lifts)) if lifts else None,
                "mean_paired_nll": mean([float(r["paired_nll"]) for r in rs if math.isfinite(float(r.get("paired_nll", float("nan"))))]),
                "mean_sideonly_nll": mean([float(r["sideonly_nll"]) for r in rs if math.isfinite(float(r.get("sideonly_nll", float("nan"))))]),
            })
    return out


def write_report(result: dict[str, Any], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "reciprocal_view_lift_probe.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research reciprocal view-lift probe", ""]
    lines.append(f"Status: `{result.get('status')}`")
    lines.append(f"Model type: `{result.get('model_type')}`")
    lines.append(f"Pairs file: `{result.get('pairs_path')}`")
    lines.append(f"Sampled pairs: {result.get('sampled_pairs')} / available {result.get('available_pairs')}")
    if result.get("checkpoint"):
        lines.append(f"Checkpoint: `{result.get('checkpoint')}`")
    if result.get("checkpoint_model_sha256"):
        lines.append(f"Model SHA256: `{result.get('checkpoint_model_sha256')}`")
    lines.append("")
    if result.get("dry_run"):
        lines.append("Dry run only: model was not loaded; target accounting is construction evidence, not model evidence.")
    lines.append("")
    lines.append("## Summary groups")
    for label, rows in result.get("summary", {}).get("groups", {}).items():
        lines.append("")
        lines.append(f"### {label}")
        lines.append("| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in rows[:80]:
            key = ", ".join(f"{k}={v}" for k, v in row["key"].items())
            def fmt(x):
                return "" if x is None else f"{float(x):.6f}"
            lines.append(f"| {key} | {row['n']} | {fmt(row.get('mean_lift'))} | {fmt(row.get('median_lift'))} | {fmt(row.get('positive_lift_frac'))} | {fmt(row.get('mean_paired_nll'))} | {fmt(row.get('mean_sideonly_nll'))} |")
    lines.append("")
    lines.append("## Interpretation note")
    lines.append("For MLM, positive lift in both source and rewrite target segments is evidence of reciprocal cross-view conditioning. For causal models, lift should appear only for the second segment of a particular order because future tokens are masked by the objective. A full mechanistic conclusion requires enough sampled pairs and comparison to repeat controls; tiny smoke runs only validate the probe.")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    (out_dir / "reciprocal_view_lift_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHCK82))
    ap.add_argument("--model-type", choices=["mlm", "causal"], default="mlm")
    ap.add_argument("--model-label", default="chck82")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-scan", type=int, default=0, help="limit pair file scan; 0 means all")
    ap.add_argument("--max-pairs", type=int, default=32)
    ap.add_argument("--targets-per-side", type=int, default=6)
    ap.add_argument("--seed", type=int, default=178013)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    pairs_path = pathlib.Path(args.pairs)
    checkpoint = pathlib.Path(args.checkpoint)
    rows = read_pairs(pairs_path, None if args.max_scan <= 0 else args.max_scan)
    rng = random.Random(args.seed)
    if len(rows) > args.max_pairs:
        sampled = rng.sample(rows, args.max_pairs)
    else:
        sampled = rows

    # Set cache before importing transformers so custom-code dynamic modules have a writable home.
    cache = out_dir / ".hf_cache"
    os.environ["HF_HOME"] = str(cache)
    os.environ["TRANSFORMERS_CACHE"] = str(cache)
    os.environ["HF_MODULES_CACHE"] = str(cache / "modules")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    bos = get_special_id(tokenizer, ["bos_token_id", "cls_token_id"], None)
    eos = get_special_id(tokenizer, ["eos_token_id", "sep_token_id"], None)
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if args.model_type == "mlm" and mask_id is None:
        raise RuntimeError("MLM probe requires tokenizer.mask_token_id")

    construction_rows: list[dict[str, Any]] = []
    skipped = collections.Counter()
    targets: list[dict[str, Any]] = []

    for pair_idx, r in enumerate(sampled):
        source = r["source_text"]
        rewrite = r["rewrite_text"]
        repeat = make_repeat_text(source, int(r.get("rewrite_words") or wc(rewrite)))
        for pair_type, b_text in [("compact", rewrite), ("repeat", repeat)]:
            a_ids = encode_no_special(tokenizer, source)
            b_ids = encode_no_special(tokenizer, b_text)
            if not a_ids or not b_ids:
                skipped["empty_side"] += 1
                continue
            ids_pair, a_start, b_start = build_context(bos, eos, a_ids, b_ids)
            if len(ids_pair) > args.max_length:
                skipped["pair_too_long"] += 1
                continue
            a_pos = candidate_positions(tokenizer, a_ids, args.targets_per_side, rng)
            b_pos = candidate_positions(tokenizer, b_ids, args.targets_per_side, rng)
            if not a_pos or not b_pos:
                skipped["no_candidate_tokens"] += 1
                continue
            construction_rows.append({
                "pair_idx": pair_idx,
                "pair_id": r.get("pair_id"),
                "pair_type": pair_type,
                "source_words": wc(source),
                "other_words": wc(b_text),
                "context_len": len(ids_pair),
                "n_source_targets": len(a_pos),
                "n_other_targets": len(b_pos),
            })
            for side_name, side_ids, other_ids, rel_positions, start in [
                ("first", a_ids, b_ids, a_pos, a_start),
                ("second", b_ids, a_ids, b_pos, b_start),
            ]:
                other_text_norm = {norm_piece(token_text(tokenizer, t)) for t in other_ids}
                other_id_set = set(map(int, other_ids))
                for rel in rel_positions:
                    tid = int(side_ids[rel])
                    ttxt = token_text(tokenizer, tid)
                    targets.append({
                        "pair_idx": pair_idx,
                        "pair_id": r.get("pair_id"),
                        "pair_type": pair_type,
                        "side": side_name,
                        "rel_pos": rel,
                        "abs_pos": int(start + rel),
                        "orig_id": tid,
                        "token_text": ttxt,
                        "copy_by_id": tid in other_id_set,
                        "copy_by_text": norm_piece(ttxt) in other_text_norm and bool(norm_piece(ttxt)),
                    })

    if args.dry_run:
        result = {
            "status": "RECIPROCAL_VIEW_LIFT_DRY_RUN",
            "dry_run": True,
            "model_type": args.model_type,
            "model_label": args.model_label,
            "pairs_path": str(pairs_path),
            "pairs_sha256": sha256_file(pairs_path),
            "checkpoint": str(checkpoint),
            "available_pairs": len(rows),
            "sampled_pairs": len(sampled),
            "constructed_pair_contexts": len(construction_rows),
            "target_count": len(targets),
            "skipped": dict(skipped),
            "sample_construction": construction_rows[:10],
            "sample_targets": targets[:20],
            "summary": summarize([]),
            "elapsed_sec": round(time.time() - t0, 3),
        }
        write_report(result, out_dir)
        print(json.dumps({"status": result["status"], "out_dir": str(out_dir), "target_count": len(targets)}, indent=2), flush=True)
        return

    device = torch.device(args.device)
    if args.model_type == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(str(checkpoint), trust_remote_code=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), trust_remote_code=True).to(device)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())

    records: list[dict[str, Any]] = []
    for pair_idx, r in enumerate(sampled):
        source = r["source_text"]
        rewrite = r["rewrite_text"]
        repeat = make_repeat_text(source, int(r.get("rewrite_words") or wc(rewrite)))
        for pair_type, b_text in [("compact", rewrite), ("repeat", repeat)]:
            # For MLM, order is irrelevant in principle but the absolute side placement is still recorded.
            # For causal, score both directions to expose objective asymmetry directly.
            order_specs = [("source_to_other", source, b_text, "source", "other")]
            if args.model_type == "causal":
                order_specs.append(("other_to_source", b_text, source, "other", "source"))
            for order, text_a, text_b, name_a, name_b in order_specs:
                a_ids = encode_no_special(tokenizer, text_a)
                b_ids = encode_no_special(tokenizer, text_b)
                if not a_ids or not b_ids:
                    continue
                ids_pair, a_start, b_start = build_context(bos, eos, a_ids, b_ids)
                ids_aonly, aonly_start, _ = build_context(bos, eos, a_ids, None)
                ids_bonly, bonly_start, _ = build_context(bos, eos, b_ids, None)
                if len(ids_pair) > args.max_length:
                    continue
                for target_segment, side_ids, other_ids, rel_positions, full_start, only_ids, only_start in [
                    (name_a, a_ids, b_ids, candidate_positions(tokenizer, a_ids, args.targets_per_side, rng), a_start, ids_aonly, aonly_start),
                    (name_b, b_ids, a_ids, candidate_positions(tokenizer, b_ids, args.targets_per_side, rng), b_start, ids_bonly, bonly_start),
                ]:
                    if full_start is None:
                        continue
                    other_norm = {norm_piece(token_text(tokenizer, t)) for t in other_ids}
                    other_id_set = set(map(int, other_ids))
                    for rel in rel_positions:
                        tid = int(side_ids[rel])
                        full_pos = int(full_start + rel)
                        only_pos = int(only_start + rel)
                        if args.model_type == "mlm":
                            paired_nll = nll_mlm(model, tokenizer, device, ids_pair, full_pos, tid, int(mask_id))
                            sideonly_nll = nll_mlm(model, tokenizer, device, only_ids, only_pos, tid, int(mask_id))
                        else:
                            paired_nll = nll_causal(model, device, ids_pair, full_pos, tid)
                            sideonly_nll = nll_causal(model, device, only_ids, only_pos, tid)
                        lift = sideonly_nll - paired_nll if math.isfinite(paired_nll) and math.isfinite(sideonly_nll) else float("nan")
                        ttxt = token_text(tokenizer, tid)
                        records.append({
                            "model_type": args.model_type,
                            "model_label": args.model_label,
                            "pair_idx": pair_idx,
                            "pair_id": r.get("pair_id"),
                            "pair_type": pair_type,
                            "order": order,
                            "target_segment": target_segment,
                            "orig_id": tid,
                            "token_text": ttxt,
                            "copy_by_id": tid in other_id_set,
                            "copy_by_text": norm_piece(ttxt) in other_norm and bool(norm_piece(ttxt)),
                            "paired_nll": paired_nll,
                            "sideonly_nll": sideonly_nll,
                            "lift_nll_sideonly_minus_paired": lift,
                        })

    model_sha = None
    sf = checkpoint / "model.safetensors"
    if sf.exists():
        model_sha = sha256_file(sf)
    result = {
        "status": "RECIPROCAL_VIEW_LIFT_PROBE",
        "dry_run": False,
        "model_type": args.model_type,
        "model_label": args.model_label,
        "pairs_path": str(pairs_path),
        "pairs_sha256": sha256_file(pairs_path),
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": model_sha,
        "param_count": param_count,
        "available_pairs": len(rows),
        "sampled_pairs": len(sampled),
        "target_records": len(records),
        "args": vars(args),
        "records_path": str(out_dir / "reciprocal_view_lift_records.jsonl"),
        "summary": summarize(records),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    with (out_dir / "reciprocal_view_lift_records.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_report(result, out_dir)
    print(json.dumps({"status": result["status"], "out_dir": str(out_dir), "target_records": len(records), "elapsed_sec": result["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
