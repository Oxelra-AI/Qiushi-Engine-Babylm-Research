#!/usr/bin/env python3
"""Compare hidden-weight geometry at exact init, research 20M, and Muon 20M.

Measures whether Muon training broadens hidden-matrix spectra and whether the
shared numeric weight_decay=0.01 causes material norm shrinkage because Muon LR
is 8-12x the AdamW LR.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
from pathlib import Path
from statistics import mean, median

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
TOK = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
MODELS = {
    "reference_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_20M'),
    "muon008_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_seed43022_20M/hf_model/chck_20M'),
    "muon012_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr012_seed43022_20M/hf_model/chck_20M'),
}
OUT = _public_path('experiments/archive/frontier_consolidation/data/muon_weight_geometry')


def is_core(name: str) -> bool:
    if not name.startswith("deberta.encoder.layer.") or not name.endswith(".weight"):
        return False
    return name.endswith((
        ".attention.self.query_proj.weight", ".attention.self.key_proj.weight",
        ".attention.self.value_proj.weight", ".attention.output.dense.weight",
        ".intermediate.dense.weight", ".output.dense.weight"))


def family(name: str) -> str:
    if ".attention.self." in name: return "attention_qkv"
    if ".attention.output.dense." in name: return "attention_output"
    if ".intermediate.dense." in name: return "ffn_in"
    if ".output.dense." in name: return "ffn_out"
    return "other"


def exact_init():
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
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    return DebertaV2ForMaskedLM(cfg)


def matrix_stats(w: torch.Tensor) -> dict:
    x = w.detach().float().cpu()
    s = torch.linalg.svdvals(x)
    s2 = s.square()
    total = float(s2.sum())
    p = s2 / max(total, 1e-30)
    entropy = float(-(p * (p + 1e-30).log()).sum())
    return {
        "frob_norm": float(x.norm()),
        "spectral_norm": float(s[0]),
        "stable_rank": total / max(float(s2[0]), 1e-30),
        "entropy_rank": math.exp(entropy),
        "top1_energy": float(p[0]),
        "top8_energy": float(p[:8].sum()),
        "condition_nonzero": float(s[0] / max(float(s[-1]), 1e-30)),
    }


def summarize(records: dict[str, dict]) -> dict:
    out = {}
    for key in ["frob_norm", "spectral_norm", "stable_rank", "entropy_rank", "top1_energy", "top8_energy"]:
        vals = [r[key] for r in records.values()]
        out[key] = {"mean": mean(vals), "median": median(vals), "min": min(vals), "max": max(vals)}
    fams = sorted({r["family"] for r in records.values()})
    out["families"] = {}
    for f in fams:
        rs = [r for r in records.values() if r["family"] == f]
        out["families"][f] = {key: mean(r[key] for r in rs) for key in ["frob_norm","stable_rank","entropy_rank","top1_energy","top8_energy"]}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = {"exact_init": exact_init()}
    for label, path in MODELS.items():
        models[label] = AutoModelForMaskedLM.from_pretrained(path)

    result = {"status": "MUON_WEIGHT_GEOMETRY", "models": {}, "pairwise": {}}
    raw = {}
    for label, model in models.items():
        recs = {}
        for name, p in model.named_parameters():
            if is_core(name):
                recs[name] = {"family": family(name), **matrix_stats(p)}
        assert len(recs) == 48, (label, len(recs))
        raw[label] = recs
        result["models"][label] = summarize(recs)

    for a in ["reference_20M", "muon008_20M", "muon012_20M"]:
        deltas = {}
        for name in raw[a]:
            init = raw["exact_init"][name]
            cur = raw[a][name]
            deltas[name] = {
                "family": cur["family"],
                "frob_ratio_vs_init": cur["frob_norm"] / init["frob_norm"],
                "stable_rank_delta_vs_init": cur["stable_rank"] - init["stable_rank"],
                "entropy_rank_delta_vs_init": cur["entropy_rank"] - init["entropy_rank"],
                "top8_delta_vs_init": cur["top8_energy"] - init["top8_energy"],
            }
        result["pairwise"][a] = {
            "frob_ratio_vs_init_mean": mean(x["frob_ratio_vs_init"] for x in deltas.values()),
            "frob_ratio_vs_init_median": median(x["frob_ratio_vs_init"] for x in deltas.values()),
            "stable_rank_delta_vs_init_mean": mean(x["stable_rank_delta_vs_init"] for x in deltas.values()),
            "entropy_rank_delta_vs_init_mean": mean(x["entropy_rank_delta_vs_init"] for x in deltas.values()),
            "top8_delta_vs_init_mean": mean(x["top8_delta_vs_init"] for x in deltas.values()),
        }

    out_json = _public_path('experiments/archive/frontier_consolidation/data/muon_weight_geometry/muon_weight_geometry.json')
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = ["# research Muon hidden-weight geometry", "", "48 attention/FFN matrices; exact init reconstructed from research seeds/config.", "",
             "| model | Frobenius/init | stable-rank Δ/init | entropy-rank Δ/init | top8-energy Δ/init |",
             "|---|---:|---:|---:|---:|"]
    for label, r in result["pairwise"].items():
        lines.append(f"| {label} | {r['frob_ratio_vs_init_mean']:.4f} | {r['stable_rank_delta_vs_init_mean']:+.2f} | {r['entropy_rank_delta_vs_init_mean']:+.2f} | {r['top8_delta_vs_init_mean']:+.4f} |")
    lines += ["", "## Absolute means", "", "| model | stable rank | entropy rank | top1 energy | top8 energy | Frobenius norm |", "|---|---:|---:|---:|---:|---:|"]
    for label, s in result["models"].items():
        lines.append(f"| {label} | {s['stable_rank']['mean']:.2f} | {s['entropy_rank']['mean']:.2f} | {s['top1_energy']['mean']:.4f} | {s['top8_energy']['mean']:.4f} | {s['frob_norm']['mean']:.2f} |")
    (_public_path('research/documents/frontier_consolidation/data/muon_weight_geometry/muon_weight_geometry.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "pairwise": result["pairwise"], "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
