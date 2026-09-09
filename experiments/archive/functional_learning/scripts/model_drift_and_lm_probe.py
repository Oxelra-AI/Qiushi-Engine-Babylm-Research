#!/usr/bin/env python3
"""research fixed-probe analysis for coherent86 continuation checkpoints.

This script is intended to be run after research continuation checkpoints exist.  It
separates score movement from function movement by measuring, on fixed WWM-masked
probe batches:
  * unweighted MLM CE for private-ON and private-OFF paths;
  * KL(parent private-ON || candidate private-ON);
  * KL(carrier/private-OFF || candidate private-ON);
  * per-layer private-adapter parameter displacement from the exact coherent86 parent.

The probes use fixed row ranges and fixed mask RNG so model comparisons see identical
inputs and masks.  The default parent is the exact coherent86 alpha0.75 endpoint.
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
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
A02 = _public_path('experiments/archive/frontier_consolidation')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(A02_SCRIPTS))

from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM  # noqa: E402
import frozen82_fastpath_replay_trainer as base  # noqa: E402

DEFAULT_PARENT = _public_path('models/frontier')
DEFAULT_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/model_drift_probe')


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def setup_env(out_dir: Path) -> None:
    hf = out_dir / "hf_cache"
    for sub in ["", "hub", "datasets", "transformers", "modules"]:
        (hf / sub).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf.resolve())
    os.environ["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    os.environ["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_model(path: Path, device: torch.device):
    from transformers import DebertaV2Config
    cfg = DebertaV2Config.from_pretrained(str(path), local_files_only=True)
    model = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
    sd = load_file(str(path / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if "private_adapter" not in k and k not in {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}]
    if bad_missing or unexpected:
        raise RuntimeError(f"load mismatch for {path}: bad_missing={bad_missing[:8]} unexpected={unexpected[:8]}")
    if any("private_adapter" in k for k in missing):
        raise RuntimeError(f"private adapter keys missing for {path}: {[k for k in missing if 'private_adapter' in k][:8]}")
    model.tie_weights()
    model.to(device)
    model.eval()
    return model


def set_private(model, enabled: bool) -> None:
    model.set_private_enabled(enabled)


def ce_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    vocab = logits.shape[-1]
    loss_sum = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100, reduction="sum")
    n = int((labels != -100).sum().item())
    return float(loss_sum.detach().cpu()), n


def kl_attn(p_logits: torch.Tensor, q_logits: torch.Tensor, attention_mask: torch.Tensor) -> tuple[float, int]:
    p = F.softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    log_p = F.log_softmax(p_logits, dim=-1)
    kl_tok = (p * (log_p - log_q)).sum(-1)
    mask = attention_mask.float()
    return float((kl_tok * mask).sum().detach().cpu()), int(mask.sum().item())


def private_layer_from_key(k: str) -> str:
    parts = k.split(".")
    if "layer" in parts:
        i = parts.index("layer")
        if i + 1 < len(parts):
            return f"L{parts[i+1]}"
    return "other"


def flatten_keys(sd: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    if not keys:
        return torch.zeros(1)
    return torch.cat([sd[k].float().reshape(-1).cpu() for k in keys])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na == 0.0 or nb == 0.0:
        return None
    return float(torch.dot(a, b).item() / (na * nb))


def param_stats(parent_path: Path, model_path: Path) -> dict[str, Any]:
    psd = load_file(str(parent_path / "model.safetensors"), device="cpu")
    csd = load_file(str(model_path / "model.safetensors"), device="cpu")
    pkeys = sorted(k for k in psd if ".private_adapter." in k)
    ckeys = sorted(k for k in csd if ".private_adapter." in k)
    if pkeys != ckeys:
        raise RuntimeError(f"private key mismatch parent={len(pkeys)} candidate={len(ckeys)}")
    by_layer: dict[str, list[str]] = {}
    for k in pkeys:
        by_layer.setdefault(private_layer_from_key(k), []).append(k)
    out: dict[str, Any] = {}
    all_p = flatten_keys(psd, pkeys)
    all_c = flatten_keys(csd, pkeys)
    all_d = all_c - all_p
    out["all"] = {
        "parent_norm": float(all_p.norm().item()),
        "candidate_norm": float(all_c.norm().item()),
        "delta_norm": float(all_d.norm().item()),
        "cos_candidate_parent": cosine(all_c, all_p),
        "cos_delta_parent": cosine(all_d, all_p),
        "relative_delta_to_parent_norm": float(all_d.norm().item() / max(1e-12, all_p.norm().item())),
    }
    for layer, keys in sorted(by_layer.items()):
        p = flatten_keys(psd, keys)
        c = flatten_keys(csd, keys)
        d = c - p
        out[layer] = {
            "parent_norm": float(p.norm().item()),
            "candidate_norm": float(c.norm().item()),
            "delta_norm": float(d.norm().item()),
            "cos_candidate_parent": cosine(c, p),
            "cos_delta_parent": cosine(d, p),
            "relative_delta_to_parent_norm": float(d.norm().item() / max(1e-12, p.norm().item())),
        }
    return out


def parse_model_arg(x: str) -> tuple[str, Path]:
    if "=" not in x:
        p = Path(x)
        return p.name, p
    tag, path = x.split("=", 1)
    return tag, Path(path)


def probe_section(parent, model, tokenizer, stream: Path, skip_rows: int, probe_words: int,
                  batch_size: int, seq_length: int, mask_seed: int, device: torch.device) -> dict[str, Any]:
    examples = base.load_examples_tail(stream, skip_rows, probe_words)
    dataset = base.TailDataset(examples, tokenizer, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=base.collate, num_workers=0)
    gen = torch.Generator(device=device)
    gen.manual_seed(mask_seed)

    totals = {
        "targets": 0,
        "attn_tokens": 0,
        "ce_current_on_sum": 0.0,
        "ce_current_off_sum": 0.0,
        "ce_parent_on_sum": 0.0,
        "kl_parent_on_to_current_on_sum": 0.0,
        "kl_carrier_off_to_current_on_sum": 0.0,
    }
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_group = batch["word_group"].to(device)
            masked, labels, _sel, _sg = base.apply_wwm(input_ids, attention_mask, word_group, tokenizer, 0.15, gen)

            set_private(parent, True)
            parent_on = parent(input_ids=masked, attention_mask=attention_mask).logits
            set_private(model, True)
            current_on = model(input_ids=masked, attention_mask=attention_mask).logits
            set_private(model, False)
            current_off = model(input_ids=masked, attention_mask=attention_mask).logits
            set_private(model, True)

            s, n = ce_from_logits(current_on, labels)
            totals["ce_current_on_sum"] += s
            totals["targets"] += n
            s_off, _ = ce_from_logits(current_off, labels)
            totals["ce_current_off_sum"] += s_off
            s_parent, _ = ce_from_logits(parent_on, labels)
            totals["ce_parent_on_sum"] += s_parent
            k1, an = kl_attn(parent_on, current_on, attention_mask)
            k2, _ = kl_attn(current_off, current_on, attention_mask)
            totals["kl_parent_on_to_current_on_sum"] += k1
            totals["kl_carrier_off_to_current_on_sum"] += k2
            totals["attn_tokens"] += an
            del input_ids, attention_mask, word_group, masked, labels, parent_on, current_on, current_off
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    n = max(1, totals["targets"])
    an = max(1, totals["attn_tokens"])
    return {
        "skip_rows": skip_rows,
        "probe_words_requested": probe_words,
        "examples": len(examples),
        "actual_words": sum(e.words for e in examples),
        "first_row": examples[0].row_index if examples else None,
        "last_row": examples[-1].row_index if examples else None,
        "targets": totals["targets"],
        "attn_tokens": totals["attn_tokens"],
        "ce_current_on": totals["ce_current_on_sum"] / n,
        "ce_current_off": totals["ce_current_off_sum"] / n,
        "ce_parent_on": totals["ce_parent_on_sum"] / n,
        "delta_ce_current_minus_parent": (totals["ce_current_on_sum"] - totals["ce_parent_on_sum"]) / n,
        "kl_parent_on_to_current_on": totals["kl_parent_on_to_current_on_sum"] / an,
        "kl_carrier_off_to_current_on": totals["kl_carrier_off_to_current_on_sum"] / an,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default=str(DEFAULT_PARENT))
    ap.add_argument("--model", action="append", required=True, help="tag=path; repeatable")
    ap.add_argument("--stream", default=str(DEFAULT_STREAM))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_length", type=int, default=256)
    ap.add_argument("--probe_words", type=int, default=120000)
    ap.add_argument("--mask_seed", type=int, default=927027)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_env(out_dir)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    parent_path = Path(args.parent)
    tokenizer = AutoTokenizer.from_pretrained(str(parent_path), local_files_only=True)
    parent = load_model(parent_path, device)
    sections = {
        "coherent86_training_tail_prefix": 530944,
        "new_continuation_tail_prefix": 556791,
        "late_stream_prefix": 620000,
    }

    results: dict[str, Any] = {
        "status": "MODEL_DRIFT_PROBE_COMPLETE",
        "created_unix": time.time(),
        "parent": rel(parent_path),
        "stream": rel(Path(args.stream)),
        "probe_words_per_section": args.probe_words,
        "mask_seed": args.mask_seed,
        "models": {},
    }

    for m_arg in args.model:
        tag, path = parse_model_arg(m_arg)
        print(json.dumps({"event": "start_model", "tag": tag, "path": rel(path)}), flush=True)
        model = load_model(path, device)
        rec: dict[str, Any] = {"path": rel(path), "param_stats": param_stats(parent_path, path), "sections": {}}
        for sec, skip in sections.items():
            rec["sections"][sec] = probe_section(parent, model, tokenizer, Path(args.stream), skip, int(args.probe_words),
                                                   int(args.batch_size), int(args.seq_length), int(args.mask_seed), device)
            print(json.dumps({"event": "section", "tag": tag, "section": sec, **rec["sections"][sec]}), flush=True)
        results["models"][tag] = rec
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_json = out_dir / "model_drift_probe.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# research model drift and fixed-probe LM analysis", "", f"Parent: `{rel(parent_path)}`", ""]
    for tag, rec in results["models"].items():
        lines += [f"## {tag}", "", f"Path: `{rec['path']}`", "", "### Adapter displacement", ""]
        st = rec["param_stats"]["all"]
        lines.append(f"All private-adapter delta norm `{st['delta_norm']:.6g}`, relative `{st['relative_delta_to_parent_norm']:.6g}`, cos(delta,parent) `{st['cos_delta_parent']}`.")
        lines += ["", "### Fixed WWM probes", "", "| section | CE current on | CE parent on | ΔCE current-parent | KL parent→current | KL carrier-off→current | targets |", "|---|---:|---:|---:|---:|---:|---:|"]
        for sec, s in rec["sections"].items():
            lines.append(f"| {sec} | {s['ce_current_on']:.4f} | {s['ce_parent_on']:.4f} | {s['delta_ce_current_minus_parent']:+.4f} | {s['kl_parent_on_to_current_on']:.6f} | {s['kl_carrier_off_to_current_on']:.6f} | {s['targets']} |")
        lines.append("")
    out_md = out_dir / "model_drift_probe.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": results["status"], "out_json": rel(out_json), "out_md": rel(out_md)}), flush=True)


if __name__ == "__main__":
    main()
