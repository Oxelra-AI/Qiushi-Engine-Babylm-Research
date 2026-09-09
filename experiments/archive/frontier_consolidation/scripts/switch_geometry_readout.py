#!/usr/bin/env python3
"""research: hidden-matrix geometry for failed Muon→AdamW switch screens.

This script reads saved checkpoints only. It asks whether the two switch arms
lose, retain, or partially retain Muon-induced hidden spectral breadth after
late AdamW consolidation, and measures post-switch matrix drift. The behavioral
result is in actual_switch_eval_merged; this is mechanistic context.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import gc
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
TOK = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
OUT = _public_path('experiments/archive/frontier_consolidation/data/switch_geometry')

MODELS = {
    "exact_init": None,
    "reference_80M": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M'),
    "continuous_muon_80M": _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_80M/hf_model/chck_80M'),
    "switch20_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/muon20toadamw_seed43022_80M/hf_model/chck_20M'),
    "switch20_80M": _public_path('experiments/archive/frontier_consolidation/training/runs/muon20toadamw_seed43022_80M/hf_model/chck_80M'),
    "switch40_40M": _public_path('experiments/archive/frontier_consolidation/training/runs/muon40toadamw_seed43022_80M/hf_model/chck_40M'),
    "switch40_80M": _public_path('experiments/archive/frontier_consolidation/training/runs/muon40toadamw_seed43022_80M/hf_model/chck_80M'),
}

DRIFT_PAIRS = [
    ("switch20_20M", "switch20_80M"),
    ("switch40_40M", "switch40_80M"),
    ("reference_80M", "switch20_80M"),
    ("reference_80M", "switch40_80M"),
    ("continuous_muon_80M", "switch20_80M"),
    ("continuous_muon_80M", "switch40_80M"),
]


def is_core(name: str) -> bool:
    return name.startswith("deberta.encoder.layer.") and name.endswith((
        ".attention.self.query_proj.weight",
        ".attention.self.key_proj.weight",
        ".attention.self.value_proj.weight",
        ".attention.output.dense.weight",
        ".intermediate.dense.weight",
        ".output.dense.weight",
    ))


def family(name: str) -> str:
    if ".attention.self." in name:
        return "attention_qkv"
    if ".attention.output.dense." in name:
        return "attention_output"
    if ".intermediate.dense." in name:
        return "ffn_in"
    if ".output.dense." in name:
        return "ffn_out"
    return "other"


def exact_init_model():
    tok = AutoTokenizer.from_pretrained(TOK, use_fast=True)
    torch.manual_seed(43)
    torch.manual_seed(43022)
    cfg = DebertaV2Config(
        vocab_size=len(tok), hidden_size=480, num_hidden_layers=8,
        num_attention_heads=8, intermediate_size=1920,
        max_position_embeddings=512, max_relative_positions=256,
        position_buckets=256, relative_attention=True,
        pos_att_type=["p2c", "c2p"], hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )
    return DebertaV2ForMaskedLM(cfg)


def load_core(label: str, path: Path | None) -> dict[str, torch.Tensor]:
    if label == "exact_init":
        model = exact_init_model()
    else:
        assert path is not None
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError(path / "model.safetensors")
        model = AutoModelForMaskedLM.from_pretrained(path)
    recs = {}
    for name, p in model.named_parameters():
        if is_core(name):
            recs[name] = p.detach().float().cpu().clone()
    if len(recs) != 48:
        raise RuntimeError(f"{label}: expected 48 matrices, found {len(recs)}")
    del model
    gc.collect()
    return recs


def stats(w: torch.Tensor) -> dict[str, float]:
    s = torch.linalg.svdvals(w)
    s2 = s.square()
    total = float(s2.sum())
    p = s2 / max(total, 1e-30)
    ent = float(-(p * (p + 1e-30).log()).sum())
    return {
        "frob": float(w.norm()),
        "stable_rank": total / max(float(s2[0]), 1e-30),
        "entropy_rank": math.exp(ent),
        "top1_energy": float(p[0]),
        "top8_energy": float(p[:8].sum()),
    }


def subspace_angle(a: torch.Tensor, b: torch.Tensor, k: int = 8) -> tuple[float, float]:
    ua, _, va = torch.linalg.svd(a, full_matrices=False)
    ub, _, vb = torch.linalg.svd(b, full_matrices=False)
    k = min(k, ua.shape[1], ub.shape[1], va.shape[0], vb.shape[0])
    def mean_angle(x, y):
        c = torch.linalg.svdvals(x.T @ y).clamp(0, 1)
        return float(torch.rad2deg(torch.acos(c)).mean())
    return mean_angle(ua[:, :k], ub[:, :k]), mean_angle(va[:k, :].T, vb[:k, :].T)


def summarize(vals):
    return {"mean": mean(vals), "median": median(vals), "min": min(vals), "max": max(vals)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    weights = {label: load_core(label, path) for label, path in MODELS.items()}
    init = weights["exact_init"]

    raw_stats = {}
    summary = {}
    for label, mats in weights.items():
        recs = {}
        for name, w in mats.items():
            recs[name] = {"family": family(name), **stats(w)}
        raw_stats[label] = recs
        summary[label] = {}
        for key in ["frob", "stable_rank", "entropy_rank", "top1_energy", "top8_energy"]:
            summary[label][key] = summarize([r[key] for r in recs.values()])

    vs_init = {}
    for label, mats in weights.items():
        if label == "exact_init":
            continue
        rows = []
        for name, w in mats.items():
            wi = init[name]
            st = raw_stats[label][name]
            si = raw_stats["exact_init"][name]
            rows.append({
                "family": family(name),
                "frob_ratio_vs_init": float(w.norm()) / max(float(wi.norm()), 1e-30),
                "stable_rank_delta_vs_init": st["stable_rank"] - si["stable_rank"],
                "entropy_rank_delta_vs_init": st["entropy_rank"] - si["entropy_rank"],
                "top8_delta_vs_init": st["top8_energy"] - si["top8_energy"],
            })
        vs_init[label] = {
            "frob_ratio_vs_init_mean": mean(r["frob_ratio_vs_init"] for r in rows),
            "stable_rank_delta_vs_init_mean": mean(r["stable_rank_delta_vs_init"] for r in rows),
            "entropy_rank_delta_vs_init_mean": mean(r["entropy_rank_delta_vs_init"] for r in rows),
            "top8_delta_vs_init_mean": mean(r["top8_delta_vs_init"] for r in rows),
        }

    drift = {}
    for a, b in DRIFT_PAIRS:
        rows = []
        for name in weights[a]:
            wa, wb, wi = weights[a][name], weights[b][name], init[name]
            la, ra = subspace_angle(wa, wb, 8)
            rows.append({
                "family": family(name),
                "rel_frob_start": float((wb - wa).norm()) / max(float(wa.norm()), 1e-30),
                "rel_frob_init": float((wb - wa).norm()) / max(float(wi.norm()), 1e-30),
                "flat_cosine": float(torch.dot(wa.flatten(), wb.flatten()) / (wa.norm() * wb.norm() + 1e-30)),
                "left_angle8": la,
                "right_angle8": ra,
            })
        drift[f"{a}_to_{b}"] = {
            "rel_frob_start": summarize([r["rel_frob_start"] for r in rows]),
            "rel_frob_init": summarize([r["rel_frob_init"] for r in rows]),
            "flat_cosine": summarize([r["flat_cosine"] for r in rows]),
            "left_angle8": summarize([r["left_angle8"] for r in rows]),
            "right_angle8": summarize([r["right_angle8"] for r in rows]),
        }

    out = {"status": "SWITCH_GEOMETRY_READOUT", "summary": summary, "vs_init": vs_init, "drift": drift}
    (_public_path('experiments/archive/frontier_consolidation/data/switch_geometry/switch_geometry_readout.json')).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# research switch-arm hidden-matrix geometry",
        "",
        "Static spectra and selected drift for switch arms after mature evaluation failure.",
        "",
        "## Static spectra",
        "",
        "| model | Frobenius/init | stable rank | entropy rank | top8 energy | stable-rank Δ/init | top8 Δ/init |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ["reference_80M", "continuous_muon_80M", "switch20_20M", "switch20_80M", "switch40_40M", "switch40_80M"]:
        s = summary[label]
        v = vs_init[label]
        lines.append(f"| {label} | {v['frob_ratio_vs_init_mean']:.4f} | {s['stable_rank']['mean']:.2f} | {s['entropy_rank']['mean']:.2f} | {s['top8_energy']['mean']:.4f} | {v['stable_rank_delta_vs_init_mean']:+.2f} | {v['top8_delta_vs_init_mean']:+.4f} |")
    lines += ["", "## Drift", "", "| interval | rel Frobenius/start | rel Frobenius/init | flat cosine | left angle8 | right angle8 |", "|---|---:|---:|---:|---:|---:|"]
    for key, d in drift.items():
        lines.append(f"| {key} | {d['rel_frob_start']['mean']:.4f} | {d['rel_frob_init']['mean']:.4f} | {d['flat_cosine']['mean']:.4f} | {d['left_angle8']['mean']:.2f} | {d['right_angle8']['mean']:.2f} |")
    (_public_path('research/documents/frontier_consolidation/data/switch_geometry/switch_geometry_readout.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "out": str(OUT),
        "vs_init": vs_init,
        "drift": {k: {kk: vv["mean"] for kk, vv in d.items()} for k, d in drift.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
