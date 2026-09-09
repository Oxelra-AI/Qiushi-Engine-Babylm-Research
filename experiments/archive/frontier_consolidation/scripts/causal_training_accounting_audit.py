#!/usr/bin/env python3
"""research: CPU audit of causal GPT compact/repeat training accounting.

This is not a competence evaluator. It records what the completed training arms establish
before selected official-compatible cheap7 results arrive: legal-word accounting, tokenizer/
pool identity, effective-token asymmetry, checkpoint availability, and training-loss direction.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from statistics import mean
from typing import Any

ENDPOINTS = ["chck_20M", "chck_50M", "chck_70M", "chck_82M", "chck_90M", "chck_100M"]


def parse_arm(text: str) -> tuple[str, pathlib.Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use LABEL=RUN_DIR")
    label, path = text.split("=", 1)
    return label, pathlib.Path(path)


def load_manifest(run_dir: pathlib.Path) -> dict[str, Any]:
    p = run_dir / "training_manifest.json"
    if not p.exists():
        raise FileNotFoundError(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["_manifest_path"] = str(p)
    return data


def summarize_arm(label: str, run_dir: pathlib.Path, manifest: dict[str, Any]) -> dict[str, Any]:
    ckpts = {c["name"]: c for c in manifest.get("checkpoints", [])}
    selected = []
    for ep in ENDPOINTS:
        rec = ckpts.get(ep)
        model_path = run_dir / "hf_model" / ep / "model.safetensors"
        selected.append({
            "endpoint": ep,
            "exists_in_manifest": rec is not None,
            "model_exists": model_path.exists(),
            "words": rec.get("words") if rec else None,
            "step": rec.get("step") if rec else None,
            "loss": rec.get("loss") if rec else None,
            "model_sha256": rec.get("model_sha256") if rec else None,
        })
    train = manifest.get("training", {})
    losses = [r["loss"] for r in selected if r.get("loss") is not None]
    return {
        "label": label,
        "run_dir": str(run_dir),
        "manifest_path": manifest["_manifest_path"],
        "status": manifest.get("status"),
        "pool_sha256": manifest.get("pool_sha256"),
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "model": manifest.get("model", {}),
        "training": train,
        "selected_checkpoints": selected,
        "selected_all_present": all(r["exists_in_manifest"] and r["model_exists"] for r in selected),
        "selected_loss_mean": mean(losses) if losses else None,
        "selected_loss_best_endpoint": min((r for r in selected if r.get("loss") is not None), key=lambda r: r["loss"])["endpoint"] if losses else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", type=parse_arm, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arms: dict[str, Any] = {}
    for label, run_dir in args.arm:
        arms[label] = summarize_arm(label, run_dir, load_manifest(run_dir))

    comparisons: dict[str, Any] = {}
    if "compact" in arms and "repeat" in arms:
        c = arms["compact"]["training"]
        r = arms["repeat"]["training"]
        comparisons["compact_minus_repeat"] = {
            "final_loss": c.get("final_loss") - r.get("final_loss") if c.get("final_loss") is not None and r.get("final_loss") is not None else None,
            "raw_tokens_per_epoch_with_eos": c.get("raw_tokens_per_epoch_with_eos") - r.get("raw_tokens_per_epoch_with_eos") if c.get("raw_tokens_per_epoch_with_eos") is not None and r.get("raw_tokens_per_epoch_with_eos") is not None else None,
            "active_tokens_per_epoch": c.get("active_tokens_per_epoch") - r.get("active_tokens_per_epoch") if c.get("active_tokens_per_epoch") is not None and r.get("active_tokens_per_epoch") is not None else None,
            "active_token_relative_to_repeat": (c.get("active_tokens_per_epoch") - r.get("active_tokens_per_epoch")) / r.get("active_tokens_per_epoch") if c.get("active_tokens_per_epoch") is not None and r.get("active_tokens_per_epoch") else None,
            "dropped_tail_tokens_per_epoch": c.get("dropped_tail_tokens_per_epoch") - r.get("dropped_tail_tokens_per_epoch") if c.get("dropped_tail_tokens_per_epoch") is not None and r.get("dropped_tail_tokens_per_epoch") is not None else None,
            "charged_words_per_active_token": c.get("charged_words_per_active_token") - r.get("charged_words_per_active_token") if c.get("charged_words_per_active_token") is not None and r.get("charged_words_per_active_token") is not None else None,
        }
        per_ep = []
        csel = {x["endpoint"]: x for x in arms["compact"]["selected_checkpoints"]}
        rsel = {x["endpoint"]: x for x in arms["repeat"]["selected_checkpoints"]}
        for ep in ENDPOINTS:
            if ep in csel and ep in rsel:
                per_ep.append({
                    "endpoint": ep,
                    "compact_words_minus_repeat_words": (csel[ep].get("words") or 0) - (rsel[ep].get("words") or 0),
                    "compact_step_minus_repeat_step": (csel[ep].get("step") or 0) - (rsel[ep].get("step") or 0),
                    "compact_loss_minus_repeat_loss": csel[ep].get("loss") - rsel[ep].get("loss") if csel[ep].get("loss") is not None and rsel[ep].get("loss") is not None else None,
                })
        comparisons["selected_checkpoint_deltas"] = per_ep

    result = {
        "status": "CAUSAL_TRAINING_ACCOUNTING_AUDIT",
        "endpoints_for_selected_eval": ENDPOINTS,
        "arms": arms,
        "comparisons": comparisons,
        "interpretation": [
            "Both arms must be judged by official-compatible selected cheap7, not by training loss.",
            "Compact has a small active-token advantage under the neutral tokenizer; causal compact evidence must exceed and broaden beyond this implementation asymmetry.",
            "Repeat's lower training loss, if present, is corpus-fit evidence only and may move opposite to official competence."
        ],
    }
    out_json = args.out_dir / "causal_training_accounting_audit.json"
    out_md = args.out_dir / "causal_training_accounting_audit.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = ["# research causal GPT training accounting audit\n\n"]
    for label, rec in arms.items():
        tr = rec["training"]
        md.append(f"## {label}\n\n")
        md.append(f"Status: `{rec['status']}`; params: {rec['model'].get('params')}; final loss: {tr.get('final_loss')}.\n\n")
        md.append(f"Pool SHA: `{rec.get('pool_sha256')}`; tokenizer SHA: `{rec.get('tokenizer_sha256')}`.\n\n")
        md.append(f"Legal charged words: {tr.get('legal_charged_words')}; steps: {tr.get('total_steps')}; epochs: {tr.get('total_epochs')}; active tokens/epoch: {tr.get('active_tokens_per_epoch')}; dropped tail tokens/epoch: {tr.get('dropped_tail_tokens_per_epoch')}.\n\n")
        md.append("| endpoint | words | step | loss | model exists |\n|---|---:|---:|---:|---|\n")
        for row in rec["selected_checkpoints"]:
            loss = row.get("loss")
            md.append(f"| {row['endpoint']} | {row.get('words')} | {row.get('step')} | {loss if loss is not None else ''} | {row.get('model_exists')} |\n")
        md.append("\n")
    if comparisons.get("compact_minus_repeat"):
        cmr = comparisons["compact_minus_repeat"]
        md.append("## compact - repeat accounting deltas\n\n")
        for k, v in cmr.items():
            md.append(f"- {k}: {v}\n")
        md.append("\nSelected checkpoint deltas (loss remains non-decisive):\n\n")
        md.append("| endpoint | Δwords | Δstep | Δloss |\n|---|---:|---:|---:|\n")
        for row in comparisons["selected_checkpoint_deltas"]:
            md.append(f"| {row['endpoint']} | {row['compact_words_minus_repeat_words']} | {row['compact_step_minus_repeat_step']} | {row['compact_loss_minus_repeat_loss']} |\n")
    md.append("\nOfficial-compatible selected cheap7 results are required for the architecture-transfer conclusion.\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
