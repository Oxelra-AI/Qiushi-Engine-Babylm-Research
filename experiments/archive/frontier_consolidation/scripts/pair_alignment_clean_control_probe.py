#!/usr/bin/env python3
"""research: compare changed-block source/rewrite geometry in clean and reinvest models.

The source/rewrite pairs exist only in the compact-view-reinvest corpus, but their
texts are ordinary strings that can be fed to any same-tokenizer checkpoint.  This
CPU-only probe measures whether the matched clean-Qwen control, which did not train
on the compact FineWeb paired rows, nevertheless represents source and compact view
as close.  The contrast helps determine whether the strong pair geometry seen in
the reinvest model is a trivial semantic property of pretrained text or an acquired
feature of the reinvest training stream.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import pathlib
import sys
import time


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
BASE_PROBE = WORKSPACE / "scripts/pair_alignment_probe.py"
OUT_DIR = WORKSPACE / "data/pair_alignment_clean_control_probe"


def load_base_probe():
    spec = importlib.util.spec_from_file_location("pair_alignment_probe_base", BASE_PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_PROBE}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-examples", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=6901)
    ap.add_argument("--layers", default="0,4,-1")
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    t0 = time.time()
    mod = load_base_probe()
    import torch
    from transformers import AutoTokenizer
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_sha = mod.sha256_file(mod.TRAIN_100M)
    tok_sha = mod.sha256_file(mod.TOKENIZER_DIR / "tokenizer.json")
    if train_sha != mod.EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch: {train_sha}")
    if tok_sha != mod.EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    span_by_ex = mod.load_span_map(mod.SPAN_JSONL)
    all_changed = mod.load_changed_examples(mod.TRAIN_100M, span_by_ex)
    sampled = mod.select_examples(all_changed, args.sample_examples, args.seed)
    sample_pair_records = sum(1 for ex in sampled for p in (ex.span.get("pairs") or []) if p.get("visibility") == "both_visible")
    tokenizer = AutoTokenizer.from_pretrained(str(mod.TOKENIZER_DIR), use_fast=True)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    checkpoints = [
        {
            "name": "clean_20M",
            "family": "clean_control",
            "path": WORKSPACE / "training/runs/complianttok_cleanqwen_seed43022_80M/hf_model/chck_20M",
            "score_context": "matched research-tokenizer clean-Qwen control, 20M",
        },
        {
            "name": "clean_80M",
            "family": "clean_control",
            "path": WORKSPACE / "training/runs/complianttok_cleanqwen_seed43022_80M/hf_model/chck_80M",
            "score_context": "matched research-tokenizer clean-Qwen control, 80M",
        },
        {
            "name": "reinvest_20M",
            "family": "tokenmean_reinvest",
            "path": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_20M",
            "score_context": "token-mean legal compact-view reinvest, 20M",
        },
        {
            "name": "reinvest_80M",
            "family": "tokenmean_reinvest",
            "path": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M",
            "score_context": "token-mean legal compact-view reinvest, 80M",
        },
    ]
    results = []
    for ckpt in checkpoints:
        results.append(mod.analyze_checkpoint(ckpt, tokenizer, sampled, args.batch_size, layers, args.seed, args.torch_threads))
    by = {r.get("name"): r for r in results if r.get("status") == "ok"}
    contrasts = []
    for layer in ["embedding", "layer_4", "last"]:
        row = {"layer": layer}
        for exp in [20, 80]:
            c = by.get(f"clean_{exp}M", {}).get("layers", {}).get(layer)
            rv = by.get(f"reinvest_{exp}M", {}).get("layers", {}).get(layer)
            if c and rv:
                row[f"reinvest_minus_clean_{exp}M_margin"] = float(rv["centered_actual_minus_shuffled_mean"]) - float(c["centered_actual_minus_shuffled_mean"])
                row[f"reinvest_minus_clean_{exp}M_top1"] = float(rv["retrieval_centered"]["top1"]) - float(c["retrieval_centered"]["top1"])
        contrasts.append(row)
    interpretation = []
    last_c80 = by.get("clean_80M", {}).get("layers", {}).get("last")
    last_r80 = by.get("reinvest_80M", {}).get("layers", {}).get("last")
    if last_c80 and last_r80:
        dc = float(last_r80["centered_actual_minus_shuffled_mean"]) - float(last_c80["centered_actual_minus_shuffled_mean"])
        dt = float(last_r80["retrieval_centered"]["top1"]) - float(last_c80["retrieval_centered"]["top1"])
        interpretation.append(f"At 80M last layer, reinvest-minus-clean paired-geometry margin delta is {dc:+.4f} and retrieval-top1 delta is {dt:+.4f} on the same held-out changed-block pair texts. This quantifies whether pair co-training adds geometry beyond generic same-tokenizer language learning.")
    interpretation.append("Because the clean model never trained on the compact FineWeb source/rewrite paired rows, this is a route-design probe for shared-subspace consistency, not an official score measurement and not a contamination or compliance test.")
    summary = {
        "status": "PAIR_ALIGNMENT_CLEAN_CONTROL_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "CPU-only clean-vs-reinvest representation comparison on changed-block source/rewrite pair texts.",
        "inputs": {
            "train_sha256": train_sha,
            "tokenizer_sha256": tok_sha,
            "span_jsonl": str(mod.SPAN_JSONL),
            "span_examples": len(span_by_ex),
            "sample_examples": len(sampled),
            "sample_pair_records_both_visible": sample_pair_records,
            "layers": layers,
            "batch_size": args.batch_size,
        },
        "checkpoint_results": results,
        "contrasts": contrasts,
        "scientific_interpretation": interpretation,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "pair_alignment_clean_control_probe.json"
    out_md = out_dir / "pair_alignment_clean_control_probe.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research pair alignment clean-control probe",
        "",
        "CPU-only comparison on changed-block pair texts. No training, official evaluation, corpus change, or route selection.",
        "",
        f"Sampled `{len(sampled)}` changed examples and `{sample_pair_records}` both-visible pair records.",
        "",
        "## Last-layer centered geometry",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"- {r['name']}: status `{r.get('status')}`")
            continue
        last = r["layers"]["last"]
        lines.append(f"- {r['name']}: margin={last['centered_actual_minus_shuffled_mean']:.4f}, actual={last['centered_actual_cos']['mean']:.4f}, shuffled={last['centered_shuffled_cos']['mean']:.4f}, same-row-other={last['centered_same_row_other_cos']['mean']:.4f}, top1={last['retrieval_centered']['top1']:.4f}")
    lines += ["", "## Contrasts"]
    for c in contrasts:
        lines.append(f"- {c}")
    lines += ["", "## Interpretation"]
    for item in interpretation:
        lines.append(f"- {item}")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "elapsed_sec": summary["elapsed_sec"], "checkpoint_status": {r.get("name"): r.get("status") for r in results}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
