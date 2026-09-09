#!/usr/bin/env python3
"""research: layerwise contextual-computation probe on conditional-innovation targets.

CPU-only. No model update, no official evaluation, no corpus/tokenizer change, no H100 work.

This complements `conditional_innovation_probe.py`.  If source-specific
conditional help on innovation/relation targets appears only in late layers and the
true-context loss is still falling sharply at the final layer, a deeper or iterative
contextual-computation route is more plausible.  If the source-specific help is
already present and late-layer gains are modest while loss remains high, the stronger
route is a targeted changed-span/innovation learning signal rather than more compute.
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
import statistics
import sys
import time
from collections import defaultdict
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
COND_SCRIPT = WORKSPACE / "scripts/conditional_innovation_probe.py"
OUT_DIR = WORKSPACE / "data/contextual_computation_probe"
CHECKPOINTS = {
    "tokenmean_80M": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M",
    "tokenmean_100M": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M",
    "clean_80M": WORKSPACE / "training/runs/complianttok_cleanqwen_seed43022_80M/hf_model/chck_80M",
}


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cond = load_module("conditional_innovation_import", COND_SCRIPT)


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
        if lo == hi:
            return float(s[lo])
        return float(s[lo] * (hi - pos) + s[hi] * (pos - lo))
    return {"n": len(vals), "mean": float(statistics.mean(vals)), "median": float(statistics.median(vals)), "std": float(statistics.pstdev(vals)), "p05": pct(0.05), "p95": pct(0.95)}


def evaluate_layerwise(label: str, ckpt: pathlib.Path, eval_rows: list[dict[str, Any]], batch_size: int, torch_threads: int) -> dict[str, Any]:
    if not ckpt.exists():
        return {"label": label, "status": "missing", "path": str(ckpt)}
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt))
    model.eval(); model.to("cpu")
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    # (target_id, scenario, layer) -> loss
    losses: dict[tuple[int, str, int], float] = {}
    n_layers_seen = None
    with torch.no_grad():
        for start in range(0, len(eval_rows), batch_size):
            batch = eval_rows[start:start + batch_size]
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attn = torch.stack([x["attention_mask"] for x in batch])
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
            hs = list(out.hidden_states)
            n_layers_seen = len(hs) - 1
            # DeBERTaV2ForMaskedLM exposes `cls`; this is the trained MLM head used on final states.
            for li, h in enumerate(hs):
                logits = model.cls(h)
                for bi, item in enumerate(batch):
                    pos = torch.tensor(item["target_positions"], dtype=torch.long)
                    gold = torch.tensor(item["gold_ids"], dtype=torch.long)
                    l = ce(logits[bi, pos, :], gold).mean().item()
                    losses[(int(item["target_id"]), str(item["scenario"]), li)] = float(l)
                del logits
            del out
    del model
    # target metadata
    target_meta: dict[int, dict[str, str]] = {}
    for item in eval_rows:
        target_meta.setdefault(int(item["target_id"]), {"category": str(item["category"]), "cue_class": str(item["cue_class"])})
    grouped: dict[str, list[float]] = defaultdict(list)
    per_layer_rows = []
    n_layers = int(n_layers_seen or 0)
    for layer in range(n_layers + 1):
        for tid, meta in target_meta.items():
            true = losses.get((tid, "true_source", layer))
            smask = losses.get((tid, "source_masked", layer))
            decoy = losses.get((tid, "same_row_decoy_source", layer))
            if true is None or smask is None or decoy is None:
                continue
            vals = {
                "true_loss": true,
                "source_help": smask - true,
                "true_over_decoy": decoy - true,
                "generic_source_help": smask - decoy,
            }
            groups = ["all", meta["category"], meta["cue_class"], f"{meta['category']}:{meta['cue_class']}"]
            for g in groups:
                for k, v in vals.items():
                    grouped[f"L{layer}|{g}|{k}"].append(v)
        for g in ["all", "innovation", "copyable", "innovation:relation_cue", "innovation:content_or_other"]:
            rec = {"layer": layer, "group": g}
            for k in ["true_loss", "source_help", "true_over_decoy", "generic_source_help"]:
                rec[k] = summarize(grouped.get(f"L{layer}|{g}|{k}", []))
            per_layer_rows.append(rec)
    # Derived late-gain summaries.
    derived = []
    final = n_layers
    for g in ["all", "innovation", "copyable", "innovation:relation_cue", "innovation:content_or_other"]:
        def m(layer: int, metric: str):
            return summarize(grouped.get(f"L{layer}|{g}|{metric}", [])).get("mean")
        row = {"group": g, "final_layer": final}
        if final >= 2:
            row["true_loss_drop_last2"] = (m(final - 2, "true_loss") - m(final, "true_loss")) if m(final - 2, "true_loss") is not None and m(final, "true_loss") is not None else None
            row["source_help_gain_last2"] = (m(final, "source_help") - m(final - 2, "source_help")) if m(final, "source_help") is not None and m(final - 2, "source_help") is not None else None
            row["true_over_decoy_gain_last2"] = (m(final, "true_over_decoy") - m(final - 2, "true_over_decoy")) if m(final, "true_over_decoy") is not None and m(final - 2, "true_over_decoy") is not None else None
        if final >= 4:
            row["true_loss_drop_last4"] = (m(final - 4, "true_loss") - m(final, "true_loss")) if m(final - 4, "true_loss") is not None and m(final, "true_loss") is not None else None
            row["source_help_gain_last4"] = (m(final, "source_help") - m(final - 4, "source_help")) if m(final, "source_help") is not None and m(final - 4, "source_help") is not None else None
            row["true_over_decoy_gain_last4"] = (m(final, "true_over_decoy") - m(final - 4, "true_over_decoy")) if m(final, "true_over_decoy") is not None and m(final - 4, "true_over_decoy") is not None else None
        derived.append(row)
    return {
        "label": label,
        "status": "ok",
        "path": str(ckpt),
        "n_targets": len(target_meta),
        "n_eval_sequences": len(eval_rows),
        "n_encoder_layers": n_layers,
        "per_layer_rows": per_layer_rows,
        "derived_late_gains": derived,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--sample-rows", type=int, default=384)
    ap.add_argument("--max-per-bucket", type=int, default=220)
    ap.add_argument("--sample-seed", type=int, default=73073)
    ap.add_argument("--target-seed", type=int, default=73173)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--checkpoints", default="tokenmean_80M,clean_80M")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_sha = cond.collate_mod.sha256_file(cond.collate_mod.TRAIN_100M)
    tok_sha = cond.collate_mod.sha256_file(cond.collate_mod.TOKENIZER_DIR / "tokenizer.json")
    if train_sha != cond.collate_mod.EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch: {train_sha}")
    if tok_sha != cond.collate_mod.EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    span_by_ex = cond.collate_mod.load_span_map(cond.collate_mod.SPAN_JSONL)
    example_ids = cond.choose_example_ids(span_by_ex, args.sample_rows, args.sample_seed)
    examples = cond.read_examples_by_example_id(cond.collate_mod.TRAIN_100M, example_ids)
    tokenizer = AutoTokenizer.from_pretrained(str(cond.collate_mod.TOKENIZER_DIR), use_fast=True)
    targets, build_summary = cond.build_targets(examples, tokenizer, span_by_ex, args.max_per_bucket, args.target_seed)
    eval_rows, _, _ = cond.make_eval_examples(examples, tokenizer, span_by_ex, targets, ["true_source", "source_masked", "same_row_decoy_source"])
    results = []
    for label in [x.strip() for x in args.checkpoints.split(",") if x.strip()]:
        if label not in CHECKPOINTS:
            raise ValueError(label)
        print(json.dumps({"event": "checkpoint_start", "label": label, "targets": len(targets), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
        r = evaluate_layerwise(label, CHECKPOINTS[label], eval_rows, args.batch_size, args.torch_threads)
        results.append(r)
        print(json.dumps({"event": "checkpoint_done", "label": label, "status": r.get("status")}), flush=True)
    summary = {
        "status": "CONTEXTUAL_COMPUTATION_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Layerwise emergence of conditional source help on copyable versus innovation rewrite targets, used to compare targeted innovation-learning with deeper/iterative contextual-computation hypotheses.",
        "inputs": {
            "train_sha256": train_sha,
            "tokenizer_sha256": tok_sha,
            "sample_rows_loaded": len(examples),
            "targets": len(targets),
            "target_build": build_summary,
            "checkpoints": [x.strip() for x in args.checkpoints.split(",") if x.strip()],
            "batch_size": args.batch_size,
        },
        "checkpoint_results": results,
        "scientific_reading": [
            "A large positive final-layer true_over_decoy for innovation groups confirms source-specific conditional information beyond generic row identity.",
            "If last-two-layer source-help and true-over-decoy gains are still large, more contextual computation/depth may be a high-leverage separate route; if they are small, the existing 8-layer model already computes the signal and the remaining high innovation loss is better attacked by training signal allocation on changed spans.",
            "Intermediate-layer MLM-head losses are not official scores; compare patterns, not absolute endpoint values.",
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "contextual_computation_probe.json"
    out_md = out_dir / "contextual_computation_probe.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — layerwise contextual-computation probe",
        "",
        "CPU-only; no model update, no official evaluation, no corpus/tokenizer change, no H100 work.",
        "",
        summary["purpose"],
        "",
        f"- targets: `{len(targets)}`; changed rows loaded: `{len(examples)}`; counts: `{build_summary['selected_counts']}`",
        f"- train SHA: `{train_sha}`",
        f"- tokenizer SHA: `{tok_sha}`",
        "",
        "## Final-layer and late-layer summaries",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"### {r.get('label')}: `{r.get('status')}`")
            continue
        final = int(r["n_encoder_layers"])
        lines.append(f"### {r['label']} (encoder layers={final})")
        lines.append("| group | final true loss | final source help | final true over decoy | true loss drop last2 | source-help gain last2 | true-over-decoy gain last2 | true loss drop last4 | source-help gain last4 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        per = {(x["layer"], x["group"]): x for x in r["per_layer_rows"]}
        der = {x["group"]: x for x in r["derived_late_gains"]}
        for g in ["all", "innovation", "copyable", "innovation:relation_cue", "innovation:content_or_other"]:
            rec = per.get((final, g), {})
            d = der.get(g, {})
            def mean_metric(metric: str):
                obj = rec.get(metric) or {}
                return obj.get("mean")
            def fmt(x):
                return "" if x is None else f"{float(x):.4f}"
            lines.append(f"| {g} | {fmt(mean_metric('true_loss'))} | {fmt(mean_metric('source_help'))} | {fmt(mean_metric('true_over_decoy'))} | {fmt(d.get('true_loss_drop_last2'))} | {fmt(d.get('source_help_gain_last2'))} | {fmt(d.get('true_over_decoy_gain_last2'))} | {fmt(d.get('true_loss_drop_last4'))} | {fmt(d.get('source_help_gain_last4'))} |")
    lines += ["", "## Reading"]
    for item in summary["scientific_reading"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"Full JSON: `{out_json}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "checkpoint_status": {r.get("label"): r.get("status") for r in results}, "elapsed_sec": summary["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
