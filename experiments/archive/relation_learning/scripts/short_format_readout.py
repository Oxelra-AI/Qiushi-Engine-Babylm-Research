#!/usr/bin/env python3
"""research: post-hoc short-format no-gradient readout.

The corrected format trainer records private/slow KL on a fixed coherent-row readout.
The first coherent-special endpoint showed large short-evaluation column movement while
that coherent KL stayed small.  This script builds a matched short-row version of the
same no-gradient readout content and measures private-on versus private-off KL on both
forms for completed endpoints.

No training is performed here.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
HELD_MANIFEST = _public_path('experiments/archive/relation_learning/data/held_coherent_sets/manifest.json')
ISOLATED_STREAM = _public_path('experiments/archive/relation_learning/data/isolated_replay_streams/isolated_all_replay_3992800w.jsonl')
OUT = _public_path('experiments/archive/relation_learning/data/short_format_readout')

DEFAULT_ENDPOINTS = {
    "coherent86_alpha075": "models/frontier",
    "coherent_special_98097_scale1p0": "experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special/seed98097/checkpoint",
    "coherent_special_98097_alpha075": "experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special/seed98097/alpha_0.75",
    "coherent_special_98098_scale1p0": "experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special/seed98098/checkpoint",
    "coherent_special_98098_alpha075": "experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special/seed98098/alpha_0.75",
}


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_rows(path: pathlib.Path, max_rows: int = 0) -> list[dict[str, Any]]:
    rows = []
    for obj in iter_jsonl(path):
        rows.append(obj)
        if max_rows and len(rows) >= max_rows:
            break
    return rows


def build_matched_short_readout(force: bool = False) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_json(HELD_MANIFEST)
    coherent_path = ROOT / manifest["sets"]["readout"]["path"]
    short_path = _public_path('experiments/archive/relation_learning/data/short_format_readout/held_short_isolated_from_coherent_readout_ids.jsonl')
    coh_rows = load_rows(coherent_path)
    ids = [obj.get("example_id") for obj in coh_rows]
    id_set = set(ids)
    if force or not short_path.exists():
        n = 0
        with short_path.open("w", encoding="utf-8") as out:
            for obj in iter_jsonl(ISOLATED_STREAM):
                if obj.get("example_id") in id_set:
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    n += 1
        if n <= 0:
            raise RuntimeError("no isolated rows matched coherent readout example_ids")
    short_rows = load_rows(short_path)
    rec = {
        "coherent_readout_path": rel(coherent_path),
        "short_isolated_readout_path": rel(short_path),
        "coherent_rows": len(coh_rows),
        "coherent_words": sum(int(o.get("words", len(str(o.get("text", "")).split()))) for o in coh_rows),
        "short_rows": len(short_rows),
        "short_words": sum(int(o.get("words", len(str(o.get("text", "")).split()))) for o in short_rows),
        "matched_example_ids": len(id_set),
        "first_ids": ids[:5],
        "last_ids": ids[-5:],
    }
    (_public_path('experiments/archive/relation_learning/data/short_format_readout/held_short_readout_manifest.json')).write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return rec


def set_private_enabled(model, enabled: bool) -> None:
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(bool(enabled))
    elif hasattr(model, "config") and hasattr(model.config, "private_adapter_enabled"):
        model.config.private_adapter_enabled = bool(enabled)


def collate_encoded(tokenizer, encoded: list[dict[str, torch.Tensor]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(e["input_ids"].numel()) for e in encoded)
    pad = int(tokenizer.pad_token_id)
    ids = torch.full((len(encoded), max_len), pad, dtype=torch.long, device=device)
    att = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        v = e["input_ids"].view(-1).to(device)
        a = e["attention_mask"].view(-1).to(device)
        ids[i, : v.numel()] = v
        att[i, : a.numel()] = a
    return ids, att


@torch.no_grad()
def measure_kl(model, tokenizer, rows: list[dict[str, Any]], device: torch.device, batch_size: int, add_special_tokens: bool) -> dict[str, Any]:
    model.eval()
    total_kl = 0.0
    total_tokens = 0
    total_abs = 0.0
    total_logit_tokens = 0
    max_kl = 0.0
    n_batches = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start+batch_size]
        encs = [tokenizer(str(o.get("text", "")), truncation=True, max_length=256, add_special_tokens=add_special_tokens, return_tensors="pt") for o in batch_rows]
        ids, att = collate_encoded(tokenizer, encs, device)
        set_private_enabled(model, False)
        slow = model(input_ids=ids, attention_mask=att).logits.detach()
        set_private_enabled(model, True)
        private = model(input_ids=ids, attention_mask=att).logits.detach()
        log_p = F.log_softmax(private, dim=-1)
        p_slow = F.softmax(slow, dim=-1)
        kl_tok = F.kl_div(log_p, p_slow, reduction="none").sum(-1)
        mask = att.bool()
        total_kl += float(kl_tok[mask].sum().cpu())
        total_tokens += int(mask.sum().item())
        if mask.any():
            max_kl = max(max_kl, float(kl_tok[mask].max().cpu()))
        # Bounded auxiliary magnitude: mean absolute difference of the top-256 vocabulary slice.
        sl = slice(0, min(256, slow.shape[-1]))
        total_abs += float((private[..., sl] - slow[..., sl]).abs()[mask].sum().cpu())
        total_logit_tokens += int(mask.sum().item()) * (sl.stop - sl.start)
        n_batches += 1
        del ids, att, slow, private, log_p, p_slow, kl_tok
    return {
        "rows": len(rows),
        "batches": n_batches,
        "tokens_with_attention": total_tokens,
        "kl_mean_per_attention_token": total_kl / max(1, total_tokens),
        "kl_max_token": max_kl,
        "mean_abs_logit_diff_first256": total_abs / max(1, total_logit_tokens),
        "add_special_tokens": add_special_tokens,
    }


def load_model_and_tokenizer(endpoint: pathlib.Path, device: torch.device, cache_root: pathlib.Path):
    os.environ["HF_HOME"] = str(cache_root / "hf_home")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
    os.environ["HF_MODULES_CACHE"] = str(cache_root / "modules")
    for p in [cache_root / "hf_home", cache_root / "transformers", cache_root / "modules"]:
        p.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(endpoint), trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(str(endpoint), trust_remote_code=True, local_files_only=True)
    model.to(device)
    model.eval()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--endpoints-json", default="")
    ap.add_argument("--force-build", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    matched = build_matched_short_readout(force=args.force_build)
    coherent_rows = load_rows(ROOT / matched["coherent_readout_path"])
    short_rows = load_rows(ROOT / matched["short_isolated_readout_path"])
    endpoints = DEFAULT_ENDPOINTS
    if args.endpoints_json:
        endpoints = read_json(pathlib.Path(args.endpoints_json))
    results: dict[str, Any] = {}
    for label, path_str in endpoints.items():
        endpoint = ROOT / path_str
        if not (endpoint / "model.safetensors").exists():
            results[label] = {"status": "missing_model", "endpoint": rel(endpoint)}
            continue
        cache_root = out / "hf_cache" / label
        t0 = time.time()
        model, tok = load_model_and_tokenizer(endpoint, device, cache_root)
        rec = {
            "status": "measured",
            "endpoint": rel(endpoint),
            "model_class": type(model).__name__,
            "private_scale": float(getattr(model.config, "private_adapter_scale", -1.0)) if hasattr(model, "config") else None,
            "coherent_row_readout_add_special": measure_kl(model, tok, coherent_rows, device, args.batch_size, add_special_tokens=True),
            "short_isolated_readout_add_special": measure_kl(model, tok, short_rows, device, args.batch_size, add_special_tokens=True),
            "coherent_row_readout_no_special": measure_kl(model, tok, coherent_rows, device, args.batch_size, add_special_tokens=False),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        results[label] = rec
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {
        "status": "SHORT_FORMAT_READOUT_DONE",
        "created_utc": now(),
        "matched_readout": matched,
        "device": str(device),
        "batch_size": args.batch_size,
        "results": results,
        "interpretation_note": "A small coherent-row KL does not imply small private/slow divergence on short isolated rows with special tokens. Compare short_isolated_readout_add_special against coherent_row_readout_add_special for the same endpoint.",
    }
    out_json = out / "short_format_readout.json"
    out_md = out / "short_format_readout.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research short-format no-gradient readout",
        "",
        f"Matched coherent readout rows: {matched['coherent_rows']} rows / {matched['coherent_words']} words.",
        f"Matched isolated readout rows: {matched['short_rows']} rows / {matched['short_words']} words.",
        "",
        "| endpoint | scale | coherent+special KL | short-isolated+special KL | coherent no-special KL | short/coherent ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, r in results.items():
        if r.get("status") != "measured":
            lines.append(f"| {label} | NA | NA | NA | NA | NA |")
            continue
        c = r["coherent_row_readout_add_special"]["kl_mean_per_attention_token"]
        s = r["short_isolated_readout_add_special"]["kl_mean_per_attention_token"]
        n = r["coherent_row_readout_no_special"]["kl_mean_per_attention_token"]
        ratio = s / c if c else None
        lines.append(f"| {label} | {r.get('private_scale')} | {c:.8f} | {s:.8f} | {n:.8f} | {ratio:.3f} |")
    lines += ["", f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
