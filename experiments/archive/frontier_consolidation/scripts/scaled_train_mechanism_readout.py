#!/usr/bin/env python3
"""Mechanism readout for research train-time scaled adapter arms."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoTokenizer, DebertaV2ForMaskedLM
from safetensors.torch import load_file

USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPT_DIR = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adapter_scaled_modeling import AdapterDebertaV2ForMaskedLM

OUT = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_mechanism')
reference_20M = _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_20M')
LIVE = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022/hf_model/chck_20M')
RUNS = {
    "scale1_train_eval1": LIVE,
    "scale1p75_train": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M20M_seed43022/hf_model/chck_20M'),
    "scale2p00_train": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale2p00_h100M20M_seed43022/hf_model/chck_20M'),
}
SENTS = [
    "The cat sat on the mat and looked out the window.",
    "Scientists discovered a new species of fish in the deep ocean.",
    "She gave him the book that she had borrowed from the library.",
    "After the rain stopped, the children went outside to play in the puddles.",
    "The key that opened the cabinet was lying under the red cup.",
    "If the block pushes the ball, the ball rolls across the table.",
]


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().flatten(); bf = b.detach().float().flatten()
    denom = float(af.norm() * bf.norm())
    return float(torch.dot(af, bf) / denom) if denom else float("nan")


def load_custom(path: Path, device: torch.device):
    model = AdapterDebertaV2ForMaskedLM.from_pretrained(path).to(device).eval()
    return model


def compare_stock(model, ref_path: Path) -> dict:
    ref = DebertaV2ForMaskedLM.from_pretrained(ref_path).eval()
    by_name = dict(ref.named_parameters())
    common = []
    sum_diff2 = 0.0
    sum_ref2 = 0.0
    sum_model2 = 0.0
    dot = 0.0
    per = []
    for n, p in model.named_parameters():
        if ".adapter." in n:
            continue
        q = by_name.get(n)
        if q is None:
            continue
        a = p.detach().float().cpu().flatten()
        b = q.detach().float().cpu().flatten()
        diff = a - b
        diff2 = float(torch.dot(diff, diff))
        ref2 = float(torch.dot(b, b))
        model2 = float(torch.dot(a, a))
        d = float(torch.dot(a, b))
        sum_diff2 += diff2; sum_ref2 += ref2; sum_model2 += model2; dot += d
        per.append({"name": n, "diff_l2": math.sqrt(diff2), "ref_l2": math.sqrt(ref2), "rel_l2": math.sqrt(diff2 / ref2) if ref2 else None, "cosine": d / math.sqrt(model2 * ref2) if model2 and ref2 else None, "diff2": diff2})
        common.append(n)
    per.sort(key=lambda x: x["diff2"], reverse=True)
    return {
        "n_common_tensors": len(common),
        "whole_vector_cosine": dot / math.sqrt(sum_model2 * sum_ref2) if sum_model2 and sum_ref2 else None,
        "relative_l2_to_reference": math.sqrt(sum_diff2 / sum_ref2) if sum_ref2 else None,
        "diff_l2_total": math.sqrt(sum_diff2),
        "top10_diff_mass": per[:10],
    }


def compare_two_custom(a_model, b_model) -> dict:
    b = {n: p for n, p in b_model.named_parameters() if ".adapter." not in n}
    sum_diff2 = sum_ref2 = sum_model2 = dot = 0.0
    per = []
    for n, p in a_model.named_parameters():
        if ".adapter." in n or n not in b:
            continue
        av = p.detach().float().cpu().flatten()
        bv = b[n].detach().float().cpu().flatten()
        diff = av - bv
        diff2 = float(torch.dot(diff, diff)); ref2 = float(torch.dot(bv, bv)); model2 = float(torch.dot(av, av)); d = float(torch.dot(av, bv))
        sum_diff2 += diff2; sum_ref2 += ref2; sum_model2 += model2; dot += d
        per.append({"name": n, "diff_l2": math.sqrt(diff2), "ref_l2": math.sqrt(ref2), "rel_l2": math.sqrt(diff2 / ref2) if ref2 else None, "cosine": d / math.sqrt(model2 * ref2) if model2 and ref2 else None, "diff2": diff2})
    per.sort(key=lambda x: x["diff2"], reverse=True)
    return {"n_common_tensors": len(per), "whole_vector_cosine": dot / math.sqrt(sum_model2 * sum_ref2), "relative_l2_to_reference": math.sqrt(sum_diff2 / sum_ref2), "diff_l2_total": math.sqrt(sum_diff2), "top10_diff_mass": per[:10]}


def model_summary(label: str, path: Path, device: torch.device):
    tok = AutoTokenizer.from_pretrained(path)
    model = load_custom(path, device)
    enc = tok(SENTS, return_tensors="pt", padding=True, truncation=True, max_length=80).to(device)
    with torch.no_grad():
        out = model(**enc)
    stats = []
    for n, p in model.named_parameters():
        if ".adapter." in n:
            stats.append({"name": n, "norm": float(p.detach().float().norm().cpu()), "rms": float(p.detach().float().square().mean().sqrt().cpu())})
    rec = {
        "label": label,
        "path": str(path),
        "adapter_scale_config": float(getattr(model.config, "adapter_scale", 1.0)),
        "logits_rms": float(out.logits.detach().float().square().mean().sqrt().cpu()),
        "adapter_rms": model.adapter_rms(),
        "adapter_rms_mean": float(mean(model.adapter_rms())),
        "adapter_rms_max": float(max(model.adapter_rms())),
        "adapter_up_norm_sum": float(sum(x["norm"] for x in stats if ".adapter.up." in x["name"])),
        "adapter_down_norm_sum": float(sum(x["norm"] for x in stats if ".adapter.down." in x["name"])),
        "adapter_ln_norm_sum": float(sum(x["norm"] for x in stats if ".adapter.layer_norm." in x["name"])),
        "adapter_param_stats": stats,
        "stock_vs_step35_20M": compare_stock(model, reference_20M),
    }
    return model, rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {}
    recs = {}
    for label, path in RUNS.items():
        m, rec = model_summary(label, path, device)
        models[label] = m
        recs[label] = rec
    recs["scale1p75_train"]["stock_vs_step103_scale1_train"] = compare_two_custom(models["scale1p75_train"], models["scale1_train_eval1"])
    recs["scale2p00_train"]["stock_vs_step103_scale1_train"] = compare_two_custom(models["scale2p00_train"], models["scale1_train_eval1"])
    recs["scale2p00_train"]["stock_vs_step104_scale1p75_train"] = compare_two_custom(models["scale2p00_train"], models["scale1p75_train"])
    out = {"status": "SCALED_TRAIN_MECHANISM", "records": recs}
    out_json = _public_path('experiments/archive/frontier_consolidation/data/scaled_train_mechanism/scaled_train_mechanism.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/scaled_train_mechanism/scaled_train_mechanism.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research scaled-train mechanism readout", "", "| run | cfg scale | adapter RMS mean/max | up norm sum | down norm sum | stock relL2 vs research | stock cosine vs research |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k, r in recs.items():
        cmp = r["stock_vs_step35_20M"]
        lines.append(f"| {k} | {r['adapter_scale_config']:.2f} | {r['adapter_rms_mean']:.6f}/{r['adapter_rms_max']:.6f} | {r['adapter_up_norm_sum']:.3f} | {r['adapter_down_norm_sum']:.3f} | {cmp['relative_l2_to_reference']:.4f} | {cmp['whole_vector_cosine']:.5f} |")
    lines += ["", "## Pairwise stock displacement"]
    for k in ["scale1p75_train", "scale2p00_train"]:
        cmp = recs[k]["stock_vs_step103_scale1_train"]
        lines.append(f"- {k} vs research scale1 stock: relL2 {cmp['relative_l2_to_reference']:.4f}, cosine {cmp['whole_vector_cosine']:.5f}.")
    cmp = recs["scale2p00_train"]["stock_vs_step104_scale1p75_train"]
    lines.append(f"- scale2.00 vs scale1.75 stock: relL2 {cmp['relative_l2_to_reference']:.4f}, cosine {cmp['whole_vector_cosine']:.5f}.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
