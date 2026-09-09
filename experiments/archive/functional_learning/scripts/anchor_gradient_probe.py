#!/usr/bin/env python3
"""Small gradient probe for research anchoring alternatives.

At the exact coherent86 parent parameters and a fixed first suffix microbatch, compare the
private-adapter gradient norms induced by:
  * the ordinary WWM main loss;
  * research carrier-off neutral KL;
  * research frozen parent-on neutral KL.

This does not evaluate model quality.  It verifies the mechanistic distinction in the
preservation reference: carrier anchoring already exerts a nonzero restoring force at the
parent, whereas parent-on anchoring is centered on the inherited useful function and has
zero loss/gradient at initialization.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))
import coherent86_continuation_trainer as prev  # noqa: E402

DEFAULT_PARENT = prev.DEFAULT_PARENT
DEFAULT_STREAM = prev.DEFAULT_STREAM
OUT_DIR = _public_path('experiments/archive/functional_learning/data/anchor_gradient_probe')


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def setup_env(out_dir: Path) -> None:
    hf = out_dir / "hf_cache"
    for sub in ["", "hub", "datasets", "transformers", "modules"]:
        (hf / sub).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str((hf).resolve())
    os.environ["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    os.environ["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def private_params(model):
    return [p for n, p in model.named_parameters() if ".private_adapter." in n]


def set_trainable_private(model) -> list[torch.nn.Parameter]:
    params = []
    for n, p in model.named_parameters():
        trainable = ".private_adapter." in n
        p.requires_grad_(trainable)
        if trainable:
            params.append(p)
    return params


def zero_private(params) -> None:
    for p in params:
        p.grad = None


def grad_norm(params) -> dict[str, float]:
    total_sq = 0.0
    max_norm = 0.0
    n_tensors = 0
    n_with_grad = 0
    for p in params:
        n_tensors += 1
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        gn = float(g.norm().cpu())
        max_norm = max(max_norm, gn)
        total_sq += gn * gn
        n_with_grad += 1
    return {"l2": math.sqrt(total_sq), "max_tensor_l2": max_norm, "n_tensors": n_tensors, "n_tensors_with_grad": n_with_grad}


def first_batch(tokenizer, stream: Path, skip_rows: int, tail_words: int, micro_batch: int, seq_length: int, device: torch.device):
    examples = prev.base.load_examples_tail(stream, skip_rows, tail_words)
    ds = prev.base.TailDataset(examples, tokenizer, seq_length)
    loader = DataLoader(ds, batch_size=micro_batch, shuffle=False, collate_fn=prev.base.collate, num_workers=0)
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    word_group = batch["word_group"].to(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(43023)
    masked, labels, _sel, _groups = prev.base.apply_wwm(input_ids, attention_mask, word_group, tokenizer, 0.15, gen)
    return {
        "masked_inputs": masked,
        "labels": labels,
        "attention_mask": attention_mask,
        "row_start": int(batch["row_index"][0].item()),
        "row_end": int(batch["row_index"][-1].item()),
        "words": int(batch["words"].sum().item()),
        "targets": int((labels != -100).sum().item()),
    }


def main_loss(model, b) -> torch.Tensor:
    model.train()
    model.set_private_enabled(True)
    out = model(input_ids=b["masked_inputs"], attention_mask=b["attention_mask"])
    return F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), b["labels"].reshape(-1), ignore_index=-100, reduction="mean")


def carrier_kl_loss(model, b) -> torch.Tensor:
    was_training = bool(model.training)
    flags = prev.private_flags(model)
    try:
        model.eval()
        model.set_private_enabled(False)
        with torch.no_grad():
            target_logits = model(input_ids=b["masked_inputs"], attention_mask=b["attention_mask"]).logits.detach()
        model.set_private_enabled(True)
        out = model(input_ids=b["masked_inputs"], attention_mask=b["attention_mask"])
        log_p = F.log_softmax(out.logits, dim=-1)
        p_target = F.softmax(target_logits, dim=-1)
        kl = F.kl_div(log_p, p_target, reduction="none").sum(-1)
        mask = b["attention_mask"].float()
        return (kl * mask).sum() / mask.sum().clamp_min(1.0)
    finally:
        prev.restore_private_flags(model, flags)
        if was_training:
            model.train()
        else:
            model.eval()


def parent_kl_loss(model, teacher, b) -> torch.Tensor:
    was_training = bool(model.training)
    flags = prev.private_flags(model)
    try:
        model.eval()
        model.set_private_enabled(True)
        teacher.eval()
        teacher.set_private_enabled(True)
        with torch.no_grad():
            target_logits = teacher(input_ids=b["masked_inputs"], attention_mask=b["attention_mask"]).logits.detach()
        out = model(input_ids=b["masked_inputs"], attention_mask=b["attention_mask"])
        log_p = F.log_softmax(out.logits, dim=-1)
        p_target = F.softmax(target_logits, dim=-1)
        kl = F.kl_div(log_p, p_target, reduction="none").sum(-1)
        mask = b["attention_mask"].float()
        return (kl * mask).sum() / mask.sum().clamp_min(1.0)
    finally:
        prev.restore_private_flags(model, flags)
        if was_training:
            model.train()
        else:
            model.eval()


def run_one(name: str, loss_fn, model, params, out: dict[str, Any]) -> None:
    zero_private(params)
    loss = loss_fn()
    loss.backward()
    out[name] = {"loss": float(loss.detach().cpu()), "grad_norm": grad_norm(params)}
    zero_private(params)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default=str(DEFAULT_PARENT))
    ap.add_argument("--stream", default=str(DEFAULT_STREAM))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip_rows", type=int, default=556791)
    ap.add_argument("--probe_tail_words", type=int, default=60000)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--seq_length", type=int, default=256)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    setup_env(out_dir)
    device = torch.device(args.device)
    from transformers import AutoTokenizer
    parent_path = Path(args.parent)
    tokenizer = AutoTokenizer.from_pretrained(str(parent_path), local_files_only=True)
    model, missing, unexpected = prev.load_model(parent_path, device, 128, 0.75)
    teacher, t_missing, t_unexpected = prev.load_model(parent_path, device, 128, 0.75)
    teacher.eval(); teacher.set_private_enabled(True)
    for p in teacher.parameters():
        p.requires_grad_(False)
    params = set_trainable_private(model)
    batch = first_batch(tokenizer, Path(args.stream), args.skip_rows, args.probe_tail_words, args.micro_batch, args.seq_length, device)
    result: dict[str, Any] = {
        "status": "ANCHOR_GRADIENT_PROBE",
        "parent": rel(parent_path),
        "stream": rel(Path(args.stream)),
        "device": str(device),
        "load_missing": missing,
        "load_unexpected": unexpected,
        "teacher_load_missing": t_missing,
        "teacher_load_unexpected": t_unexpected,
        "batch": {k: v for k, v in batch.items() if k not in {"masked_inputs", "labels", "attention_mask"}},
        "losses": {},
    }
    run_one("wwm_main", lambda: main_loss(model, batch), model, params, result["losses"])
    run_one("carrier_off_kl", lambda: carrier_kl_loss(model, batch), model, params, result["losses"])
    run_one("parent_on_kl", lambda: parent_kl_loss(model, teacher, batch), model, params, result["losses"])
    out_json = out_dir / "anchor_gradient_probe.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# research anchor gradient probe",
        "",
        f"Parent: `{rel(parent_path)}`",
        f"Batch rows {result['batch']['row_start']}–{result['batch']['row_end']}, words {result['batch']['words']}, targets {result['batch']['targets']}.",
        "",
        "| loss | value | private grad L2 | max tensor grad L2 | tensors with grad |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, rec in result["losses"].items():
        g = rec["grad_norm"]
        lines.append(f"| {name} | {rec['loss']:.8g} | {g['l2']:.8g} | {g['max_tensor_l2']:.8g} | {g['n_tensors_with_grad']} |")
    out_md = out_dir / "anchor_gradient_probe.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md), "losses": result["losses"]}), flush=True)


if __name__ == "__main__":
    main()
