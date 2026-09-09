#!/usr/bin/env python3
"""research: Parameterized source-absent edit-state private-readout probe.

Replicates the research source-absent edit-state discriminator probe on arbitrary
checkpoints, supporting trust_remote_code for adapter models (chck_82M etc.).

Scientific purpose: test whether source-absent edit-state information remains
decodable at a later checkpoint. A positive private readout establishes only
decodability, not mechanism or transfer; the harder source-removal survival test
(research-style) is needed before any trainer is built.

This script:
  1. Dynamically loads the research module to reuse its item-building and probe code.
  2. Builds the SAME 3072 source-absent edit items (deterministic from seed 8122,
     legal pair data, legal tokenizer).
  3. Loads any checkpoint with optional trust_remote_code.
  4. Runs the same 3-context forward (base, true, decoy) and detached private
     residual readout probe.
  5. Compares against the saved research baseline results.

No pretraining, official evaluation, endpoint scoring, corpus modification, or
model parameter update is performed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# ── HF cache isolation: BEFORE any transformers/torch import ──
STUDY = Path("experiments/archive/frontier_consolidation")
WORKSPACE = STUDY
CACHE_DIR = WORKSPACE / "data/edit_state_probe_hf_cache"
for _env in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"):
    _p = str(CACHE_DIR / _env.lower())
    os.environ[_env] = _p
    Path(_p).mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer


# ── research baseline numbers from the canonical source-absent run ──
BASELINE = {
    "raw_true_vs_decoy_nll_advantage": -0.3121858850837877,
    "private_true_vs_decoy_heldout_advantage": 0.8364909677886369,
    "private_true_vs_decoy_high_error_advantage": 1.2396961331691432,
    "n_items": 3072,
    "n_train": 2153,
    "n_test": 919,
    "baseline_file": str(WORKSPACE / "data/edit_state_token_value_source_absent/"
                         "edit_token_value_private_readouts.json"),
}


def load_step122_module():
    """Import the original research module to reuse item-building and probe code."""
    script_path = WORKSPACE / "scripts/token_value_private_readouts.py"
    if not script_path.exists():
        raise FileNotFoundError(f"research script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("mod", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so @dataclass can resolve __module__
    sys.modules["mod"] = mod
    # The module uses os.environ.setdefault so our presets are safe
    spec.loader.exec_module(mod)
    return mod


def build_items_args() -> SimpleNamespace:
    """Return an args namespace matching the research source-absent edit run."""
    return SimpleNamespace(
        mode="edit",
        seed=8122,
        max_items=3072,
        min_items=1500,
        max_length=224,
        max_targets_per_pair=1,
        low_overlap_focus=False,
        content_targets_only=False,
        edit_require_target_id_absent_source=True,
        train_frac=0.70,
        # Probe hyperparameters (same as research defaults)
        probe_dim=64,
        probe_epochs=24,
        probe_batch=512,
        probe_eval_batch=1024,
        probe_lr=2e-3,
        probe_wd=1e-4,
        probe_dropout=0.05,
        delta_l2=1e-5,
        grad_clip=2.0,
        high_error_quantile=0.50,
        # Discourse-specific (not used but may be referenced)
        min_middle_words=4,
        max_middle_words=38,
        max_per_row=3,
        target_mix="balanced",
        # Extra
        forward_batch=128,
        group_min_n=20,
        raw_advantage_threshold=0.02,
        private_advantage_threshold=0.02,
        private_high_error_threshold=0.02,
        device="auto",
        probe_device="same",
        torch_threads=8,
    )


def main():
    ap = argparse.ArgumentParser(description="research: parameterized edit-state probe")
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint directory")
    ap.add_argument("--model-label", required=True, help="Label for the model being probed")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="Enable trust_remote_code for adapter/custom models")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (or -1 for CPU)")
    ap.add_argument("--forward-batch", type=int, default=128)
    ap.add_argument("--probe-epochs", type=int, default=24)
    ap.add_argument("--max-items", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=8122)
    ap.add_argument("--smoke-items", type=int, default=0,
                    help="If >0, truncate items to this count for CPU smoke testing")
    ap.add_argument("--torch-threads", type=int, default=8)
    cli = ap.parse_args()

    t0 = time.time()
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cli.torch_threads)

    ckpt_path = Path(cli.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── Device ──
    if cli.gpu >= 0 and torch.cuda.is_available():
        device = f"cuda:{cli.gpu}"
        torch.cuda.set_device(cli.gpu)
    else:
        device = "cpu"
    print(json.dumps({"event": "config", "checkpoint": str(ckpt_path),
                       "model_label": cli.model_label,
                       "trust_remote_code": cli.trust_remote_code,
                       "device": device, "smoke_items": cli.smoke_items}), flush=True)

    # ── Load research module ──
    mod = load_step122_module()
    print(json.dumps({"event": "module_loaded", "path": str(mod.__file__)}), flush=True)

    # ── SHA integrity checks ──
    pool_sha = mod.sha256_file(mod.POOL_10M)
    tok_sha = mod.sha256_file(mod.TOKENIZER_DIR / "tokenizer.json")
    if pool_sha != mod.EXPECTED_POOL_SHA:
        raise RuntimeError(f"Pool SHA mismatch: {pool_sha}")
    if tok_sha != mod.EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"Tokenizer SHA mismatch: {tok_sha}")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(str(mod.TOKENIZER_DIR), use_fast=True)
    if tokenizer.mask_token is None:
        raise RuntimeError("Tokenizer has no mask token")

    # ── Build items (deterministic, same as research source-absent) ──
    item_args = build_items_args()
    item_args.max_items = cli.max_items
    item_args.seed = cli.seed
    items, build_info = mod.build_edit_items(tokenizer, item_args)
    contexts = ["base", "true", "decoy"]
    print(json.dumps({"event": "items_built", "n_items": len(items),
                       "n_pairs": build_info.get("unique_pairs"),
                       "n_rows": build_info.get("unique_rows")}), flush=True)

    if cli.smoke_items > 0:
        items = items[:cli.smoke_items]
        print(json.dumps({"event": "smoke_truncated", "n_items": len(items)}), flush=True)

    if len(items) < 10:
        raise RuntimeError(f"Too few items: {len(items)}")

    # ── Verify item reproducibility against research ──
    csv = Path(BASELINE["baseline_file"]).parent / "edit_token_value_examples.csv"
    item_match = None
    if csv.exists():
        import csv
        with csv.open("r") as f:
            reader = csv.DictReader(f)
            ids = [row["item_id"] for row in reader]
        our_ids = [it.item_id for it in items]
        # Compare the subset we have (may be truncated for smoke)
        n_compare = min(len(our_ids), len(ids))
        item_match = our_ids[:n_compare] == ids[:n_compare]
        print(json.dumps({"event": "item_reproducibility",
                           "items_compared": n_compare,
                           "all_match": item_match}), flush=True)

    # ── Split (same as research) ──
    train_idx, test_idx = mod.split_items(items, item_args.train_frac, item_args.seed + 7)

    # ── Load model ──
    print(json.dumps({"event": "model_loading", "checkpoint": str(ckpt_path),
                       "trust_remote_code": cli.trust_remote_code}), flush=True)
    model = AutoModelForMaskedLM.from_pretrained(
        str(ckpt_path), trust_remote_code=cli.trust_remote_code)
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    model_class_name = type(model).__name__
    print(json.dumps({"event": "model_loaded", "n_params": n_params,
                       "model_class": model_class_name,
                       "device": device}), flush=True)

    # ── Forward pass through all contexts ──
    raw = {"targets": [it.target_id for it in items]}
    for c in contexts:
        print(json.dumps({"event": "forward", "context": c, "n_items": len(items)}), flush=True)
        raw[c] = mod.run_context_forward(model, tokenizer, items, c, cli.forward_batch, device)
    del model
    if "cuda" in device:
        torch.cuda.empty_cache()

    # ── Raw token-value summary ──
    raw_summary = mod.summarize_raw(raw, contexts)

    # ── Group metrics ──
    group_fields = ["target_kind", "overlap_bin", "changed_frac_bin"]
    group_rows = mod.group_raw_metrics(items, raw, contexts, group_fields, min_n=5)

    # ── Private readout probe ──
    probe_dev = device
    private = mod.private_readout_analysis(raw, contexts, train_idx, test_idx, item_args, probe_dev)

    # ── Route readout (same logic as research) ──
    false_keys = [k for k in contexts if k not in {"base", "true"}]
    raw_adv = []
    priv_adv = []
    high_adv = []
    for fk in false_keys:
        rv = raw_summary.get(f"true_vs_{fk}", {}).get("mean")
        pv = private.get(f"private_true_advantage_vs_{fk}_test_positive_good", {}).get("mean")
        hv = private.get(f"private_true_advantage_vs_{fk}_high_error_positive_good", {}).get("mean")
        if rv is not None:
            raw_adv.append(rv)
        if pv is not None:
            priv_adv.append(pv)
        if hv is not None:
            high_adv.append(hv)

    route_readout = {
        "raw_true_structure_has_token_value": bool(raw_adv and min(raw_adv) > 0.02),
        "private_true_structure_explains_residual_errors": bool(
            priv_adv and min(priv_adv) > 0.02 and (not high_adv or min(high_adv) > 0.02)),
        "raw_true_advantage_min_nll": min(raw_adv) if raw_adv else None,
        "private_true_advantage_min_nll": min(priv_adv) if priv_adv else None,
        "private_true_high_error_advantage_min_nll": min(high_adv) if high_adv else None,
    }

    # ── Comparison with research baseline ──
    comparison = {
        "raw_true_vs_decoy": BASELINE["raw_true_vs_decoy_nll_advantage"],
        "private_true_vs_decoy_heldout": BASELINE["private_true_vs_decoy_heldout_advantage"],
        "private_true_vs_decoy_high_error": BASELINE["private_true_vs_decoy_high_error_advantage"],
        "current_raw_true_vs_decoy": raw_adv[0] if raw_adv else None,
        "current_private_true_vs_decoy_heldout": priv_adv[0] if priv_adv else None,
        "current_private_true_vs_decoy_high_error": high_adv[0] if high_adv else None,
    }
    if comparison["current_raw_true_vs_decoy"] is not None:
        comparison["delta_raw"] = comparison["current_raw_true_vs_decoy"] - comparison["raw_true_vs_decoy"]
    if comparison["current_private_true_vs_decoy_heldout"] is not None:
        comparison["delta_private_heldout"] = comparison["current_private_true_vs_decoy_heldout"] - comparison["private_true_vs_decoy_heldout"]
    if comparison["current_private_true_vs_decoy_high_error"] is not None:
        comparison["delta_private_high_error"] = comparison["current_private_true_vs_decoy_high_error"] - comparison["private_true_vs_decoy_high_error"]

    # ── Write outputs ──
    elapsed = round(time.time() - t0, 3)
    result = {
        "status": "PARAMETERIZED_EDIT_STATE_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_label": cli.model_label,
        "checkpoint": str(ckpt_path),
        "trust_remote_code": cli.trust_remote_code,
        "model_class": model_class_name,
        "n_model_params": n_params,
        "device": device,
        "item_reproducibility_vs_step35": item_match,
        "sample": {
            "n_items": len(items),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        },
        "build_info": build_info,
        "raw_frozen_token_value": raw_summary,
        "group_raw_token_value": group_rows,
        "private_residual_readout": private,
        "route_readout": route_readout,
        "comparison_vs_step35": comparison,
        "baseline": BASELINE,
        "elapsed_sec": elapsed,
    }

    out_json = out_dir / "edit_state_probe.json"
    out_md = out_dir / "edit_state_probe.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── Write CSV ──
    mod.write_examples_csv(out_dir / "edit_token_value_examples.csv", items, raw, contexts)
    mod.write_group_csv(out_dir / "edit_token_value_group_summary.csv", group_rows)

    # ── Write markdown summary ──
    lines = [
        f"# research — source-absent edit-state probe: {cli.model_label}",
        "",
        f"Checkpoint: `{ckpt_path}`  ",
        f"Model class: `{result.get('model_class', 'unknown')}` ({n_params:,} params)  ",
        f"Trust remote code: `{cli.trust_remote_code}`  ",
        f"Items: {len(items)} (train {len(train_idx)}, test {len(test_idx)})  ",
        f"Item reproducibility vs research: `{item_match}`",
        "",
        "## Raw frozen token value",
        "Positive true-vs-decoy = lower NLL with true source → model uses edit structure.",
        "",
        "| comparison | mean NLL advantage | 5% boot | 95% boot |",
        "|---|---:|---:|---:|",
    ]
    for fk in false_keys:
        comp = raw_summary.get(f"true_vs_{fk}", {})
        lines.append(f"| true vs {fk} | {comp.get('mean')} | {comp.get('ci05')} | {comp.get('ci95')} |")

    lines += [
        "",
        "| context | mean NLL | top1 | mean rank |",
        "|---|---:|---:|---:|",
    ]
    for c in contexts:
        rs = raw_summary[c]
        lines.append(f"| {c} | {rs['nll']['mean']} | {rs['top1']} | {rs['rank']['mean']} |")

    lines += [
        "",
        "## Detached private readout",
        "Positive true-vs-decoy = true-structure hidden state explains residual errors better.",
        "",
        "| comparison | held-out NLL advantage | 5% boot | 95% boot | high-error advantage | high 5% | high 95% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fk in false_keys:
        comp = private.get(f"private_true_advantage_vs_{fk}_test_positive_good", {})
        high = private.get(f"private_true_advantage_vs_{fk}_high_error_positive_good", {})
        lines.append(f"| true vs {fk} | {comp.get('mean')} | {comp.get('ci05')} | {comp.get('ci95')} | {high.get('mean')} | {high.get('ci05')} | {high.get('ci95')} |")

    lines += [
        "",
        "## Comparison vs research baseline",
        "",
        "| metric | research | current | delta |",
        "|---|---:|---:|---:|",
        f"| raw true-vs-decoy NLL | {comparison['raw_true_vs_decoy']:.4f} | {comparison.get('current_raw_true_vs_decoy', 'N/A')} | {comparison.get('delta_raw', 'N/A')} |",
        f"| private true-vs-decoy held-out | {comparison['private_true_vs_decoy_heldout']:.4f} | {comparison.get('current_private_true_vs_decoy_heldout', 'N/A')} | {comparison.get('delta_private_heldout', 'N/A')} |",
        f"| private true-vs-decoy high-error | {comparison['private_true_vs_decoy_high_error']:.4f} | {comparison.get('current_private_true_vs_decoy_high_error', 'N/A')} | {comparison.get('delta_private_high_error', 'N/A')} |",
        "",
        "## Route readout",
        f"- raw_true_structure_has_token_value: `{route_readout['raw_true_structure_has_token_value']}`",
        f"- private_true_structure_explains_residual_errors: `{route_readout['private_true_structure_explains_residual_errors']}`",
        "",
        "## Interpretation",
        "A positive private readout establishes ONLY decodability of edit-state information.",
        "It does NOT establish that this information can survive source removal (research-style test),",
        "that it relates to EWoK/Entity relation/state competence, or that it justifies a new trainer.",
        "",
        f"JSON: `{out_json}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── Print summary ──
    print(json.dumps({
        "status": result["status"],
        "model_label": cli.model_label,
        "n_items": len(items),
        "item_reproducibility_vs_step35": item_match,
        "raw_true_vs_decoy": comparison.get("current_raw_true_vs_decoy"),
        "private_true_vs_decoy_heldout": comparison.get("current_private_true_vs_decoy_heldout"),
        "private_true_vs_decoy_high_error": comparison.get("current_private_true_vs_decoy_high_error"),
        "delta_private_heldout_vs_step35": comparison.get("delta_private_heldout"),
        "route_readout": route_readout,
        "out_json": str(out_json),
        "out_md": str(out_md),
        "elapsed_sec": elapsed,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
