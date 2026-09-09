#!/usr/bin/env python3
"""research parallel single-component official-like scorer.

Scores exactly one independent BabyLM cheap-surface component for one repaired
checkpoint.  It reuses the research official-like masked-LM scorer and the research
Reading subword fix, but avoids the endpoint-level serial loop so BLiMP,
Supplement, EWoK, Entity, COMPS, GlobalPIQA_parallel, GlobalPIQA_nonparallel, and
Reading components can be evaluated independently.

The script records CUDA_VISIBLE_DEVICES and a best-effort nvidia-smi mapping for
its own PID so the launched background job can be checked against the physical GPU
assignment.  It does not change model weights.
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
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import torch

ROOT = _public_path('.')
PATH = _public_path('experiments/archive/relation_learning/scripts/no_boundary_cheap7_diagnostic.py')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/parallel_components')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def import_step105(out_dir: pathlib.Path):
    # research chooses writable HF/module caches before importing Transformers.
    old_argv = list(sys.argv)
    sys.argv = [str(PATH), "--out-dir", str(out_dir)]
    try:
        spec = importlib.util.spec_from_file_location("single_component_for_step127", PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {PATH}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["single_component_for_step127"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = old_argv


def patch_reading_probability(mod) -> None:
    """Patch research Reading to official get_p2_mlm-style non-recursive continuation."""

    def _score_first_id(sentence: str, token_text: str, model, tokenizer, device: torch.device,
                        add_special_tokens: bool, num_mask_tokens: int = 3) -> float:
        text = "".join([sentence, "".join([tokenizer.mask_token for _ in range(num_mask_tokens)])])
        inpts = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
        if int(inpts.input_ids[:, -1].item()) == int(tokenizer.mask_token_id):
            position = -num_mask_tokens
        else:
            position = -(num_mask_tokens + 1)
        with torch.no_grad():
            outputs = model(**inpts)
            logits = mod.get_logits(outputs)[:, position, :].cpu()
        ids = tokenizer(token_text, add_special_tokens=False)["input_ids"]
        if not ids:
            return float("nan")
        return float(torch.softmax(logits[0], dim=-1)[int(ids[0])].item())

    def p_mlm_next_fixed(sentence: str, word: str, model, tokenizer, device: torch.device,
                         add_special_tokens: bool, num_mask_tokens: int = 3) -> tuple[float, int]:
        target = tokenizer(word, add_special_tokens=False)["input_ids"]
        if not target:
            return float("nan"), 0
        text = "".join([sentence, "".join([tokenizer.mask_token for _ in range(num_mask_tokens)])])
        inpts = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
        if int(inpts.input_ids[:, -1].item()) == int(tokenizer.mask_token_id):
            position = -num_mask_tokens
        else:
            position = -(num_mask_tokens + 1)
        with torch.no_grad():
            outputs = model(**inpts)
            logits = mod.get_logits(outputs)[:, position, :].cpu()
        out_p: list[float] = [float(torch.softmax(logits[0], dim=-1)[int(target[0])].item())]
        if len(target) == 1:
            return out_p[0], 0
        next_sentence = sentence + tokenizer.decode(int(target[0]))
        for tok in target[1:]:
            t = tokenizer.decode(int(tok))
            out_p.append(_score_first_id(next_sentence, t, model, tokenizer, device, add_special_tokens, num_mask_tokens))
            next_sentence = next_sentence + t
        prod = 1.0
        for p in out_p:
            if not math.isfinite(p):
                return float("nan"), 1
            prod *= float(p)
        return float(prod), 1

    mod.p_mlm_next = p_mlm_next_fixed


def gpu_process_snapshot(pid: int) -> dict[str, Any]:
    out: dict[str, Any] = {"pid": int(pid), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    try:
        gpu_proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=10,
        )
        out["gpus"] = gpu_proc.stdout.strip().splitlines() if gpu_proc.returncode == 0 else []
        if gpu_proc.returncode != 0:
            out["gpu_query_stderr"] = gpu_proc.stderr[-500:]
    except Exception as e:
        out["gpu_query_error"] = repr(e)
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory,process_name", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=10,
        )
        lines = apps.stdout.strip().splitlines() if apps.returncode == 0 else []
        out["compute_apps"] = lines
        hits = []
        for line in lines:
            parts = [x.strip() for x in line.split(",")]
            if parts and parts[0] == str(pid):
                hits.append(parts)
        out["this_pid_compute_apps"] = hits
        # Map UUID to physical index when possible.
        uuid_to_index: dict[str, str] = {}
        for line in out.get("gpus", []):
            parts = [x.strip() for x in str(line).split(",")]
            if len(parts) >= 2:
                uuid_to_index[parts[1]] = parts[0]
        out["this_pid_physical_gpu_indices"] = sorted({uuid_to_index.get(h[1], h[1]) for h in hits if len(h) >= 2})
        if apps.returncode != 0:
            out["apps_query_stderr"] = apps.stderr[-500:]
    except Exception as e:
        out["apps_query_error"] = repr(e)
    try:
        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["torch_visible_device_count"] = int(torch.cuda.device_count())
            out["torch_current_device"] = int(torch.cuda.current_device())
            out["torch_current_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception as e:
        out["torch_query_error"] = repr(e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--column", required=True, choices=["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"])
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--input-form", default="with_special", choices=["with_special", "no_special"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--non-causal-batch-size", type=int, default=128)
    ap.add_argument("--max-items-per-file", type=int, default=0)
    ap.add_argument("--max-reading-rows", type=int, default=0)
    ap.add_argument("--no-predictions", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out_root = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    ckpt = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    if not (ckpt / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint config not found: {ckpt / 'config.json'}")
    out_root.mkdir(parents=True, exist_ok=True)

    mod = import_step105(out_root)
    patch_reading_probability(mod)
    mod.setup_cache(out_root)
    torch.manual_seed(0)

    endpoint_out = out_root / args.label / args.input_form
    endpoint_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = mod.load_tokenizer(ckpt)
    mod.ensure_pad(tokenizer)
    model = mod.AutoModelForMaskedLM.from_pretrained(str(ckpt), trust_remote_code=True)
    model.to(device)
    model.eval()
    if torch.cuda.is_available() and not args.cpu:
        # Force CUDA context and memory accounting before nvidia-smi snapshot.
        torch.cuda.synchronize()
    ident = mod.model_identity(model, ckpt)
    launch_snapshot = gpu_process_snapshot(os.getpid())
    (out_root / args.label / "model_identity.json").write_text(json.dumps(ident, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_root / args.label / f"gpu_snapshot_{args.column}.json").write_text(json.dumps(launch_snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "component_start", "label": args.label, "column": args.column, "form": args.input_form, "device": str(device), "gpu_snapshot": launch_snapshot, "utc": now()}), flush=True)

    t0 = time.time()
    if args.column == "Reading":
        record = mod.score_reading(model, tokenizer, args.input_form == "with_special", device, endpoint_out / "Reading", max_rows=int(args.max_reading_rows))
        score = float(record["score"])
    else:
        specs = {spec["column"]: spec for spec in mod.ZERO_SHOT_TASKS}
        zspec = specs[args.column]
        record = mod.score_task(model, tokenizer, zspec, args.input_form == "with_special", device, endpoint_out,
                                max_items_per_file=int(args.max_items_per_file),
                                batch_size=int(args.batch_size),
                                non_causal_batch_size=int(args.non_causal_batch_size),
                                save_predictions=not bool(args.no_predictions))
        score = float(record["score"])
    elapsed = round(time.time() - t0, 3)
    payload = {
        "status": "SINGLE_COMPONENT_DONE",
        "created_utc": now(),
        "label": args.label,
        "checkpoint": rel(ckpt),
        "column": args.column,
        "input_form": args.input_form,
        "score": score,
        "record": record,
        "model_identity": ident,
        "gpu_snapshot": launch_snapshot,
        "elapsed_sec": elapsed,
        "script": rel(_public_path('experiments/archive/relation_learning/scripts/score_component_fixed.py')),
        "note": "Single independent component scored with the research Reading fix and research official-like masked-LM surface.",
    }
    out_json = out_root / args.label / args.input_form / f"{args.column}_component_payload.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "label": args.label, "column": args.column, "score": score, "elapsed_sec": elapsed, "out_json": rel(out_json), "gpu_indices": launch_snapshot.get("this_pid_physical_gpu_indices")}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
