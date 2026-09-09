#!/usr/bin/env python3
"""research: hidden-matrix temporal drift for continuous matched-decay Muon.

This probe complements static spectra. It compares research legal reinvest and
continuous matched-decay Muon at 20M/70M/80M using only saved checkpoints. For
each of the 48 attention/FFN matrices it measures normalized Frobenius
displacement and principal-angle movement of leading singular subspaces.
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
OUT = _public_path('experiments/archive/frontier_consolidation/data/continuous_muon_drift')

PATHS = {
    "exact_init": None,
    "reference_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_20M'),
    "reference_70M": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_70M'),
    "reference_80M": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M'),
    "muon_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_20M/hf_model/chck_20M'),
    "muon_70M": _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_80M/hf_model/chck_70M'),
    "muon_80M": _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_80M/hf_model/chck_80M'),
}

PAIRS = [
    ("reference_20M", "reference_70M"),
    ("reference_70M", "reference_80M"),
    ("reference_20M", "reference_80M"),
    ("muon_20M", "muon_70M"),
    ("muon_70M", "muon_80M"),
    ("muon_20M", "muon_80M"),
]
MATCHED_DIFFS = [
    ("20M", "muon_20M", "reference_20M"),
    ("70M", "muon_70M", "reference_70M"),
    ("80M", "muon_80M", "reference_80M"),
]


def is_core(name: str) -> bool:
    if not name.startswith("deberta.encoder.layer.") or not name.endswith(".weight"):
        return False
    return name.endswith((
        ".attention.self.query_proj.weight",
        ".attention.self.key_proj.weight",
        ".attention.self.value_proj.weight",
        ".attention.output.dense.weight",
        ".intermediate.dense.weight",
        ".output.dense.weight",
    ))


def fam(name: str) -> str:
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
    out = {}
    for name, p in model.named_parameters():
        if is_core(name):
            out[name] = p.detach().float().cpu().clone()
    if len(out) != 48:
        raise RuntimeError(f"{label}: expected 48 matrices, found {len(out)}")
    del model
    gc.collect()
    return out


def stable_stats(w: torch.Tensor) -> dict[str, float]:
    s = torch.linalg.svdvals(w)
    s2 = s.square()
    total = float(s2.sum())
    p = s2 / max(total, 1e-30)
    return {
        "frob": float(w.norm()),
        "stable_rank": total / max(float(s2[0]), 1e-30),
        "top8_energy": float(p[:8].sum()),
    }


def subspace_summary(a: torch.Tensor, b: torch.Tensor, k: int) -> dict[str, float]:
    # full_matrices=False keeps this bounded. Sign is irrelevant because singular
    # subspaces are compared through canonical correlations.
    ua, sa, va = torch.linalg.svd(a, full_matrices=False)
    ub, sb, vb = torch.linalg.svd(b, full_matrices=False)
    k_eff = min(k, ua.shape[1], ub.shape[1], va.shape[0], vb.shape[0])
    def angles(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
        c = torch.linalg.svdvals(x.T @ y).clamp(0, 1)
        ang = torch.rad2deg(torch.acos(c)).float()
        return float(ang.mean()), float(torch.quantile(ang, 0.5)), float(ang.max())
    left_mean, left_med, left_max = angles(ua[:, :k_eff], ub[:, :k_eff])
    # va returned as Vh; rows are right singular vectors.
    right_mean, right_med, right_max = angles(va[:k_eff, :].T, vb[:k_eff, :].T)
    return {
        "k": k_eff,
        "left_angle_mean_deg": left_mean,
        "left_angle_median_deg": left_med,
        "left_angle_max_deg": left_max,
        "right_angle_mean_deg": right_mean,
        "right_angle_median_deg": right_med,
        "right_angle_max_deg": right_max,
    }


def summarize(vals: list[float]) -> dict[str, float]:
    return {"mean": mean(vals), "median": median(vals), "min": min(vals), "max": max(vals)}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "rel_frob_to_start", "rel_frob_to_end", "rel_frob_to_init",
        "cosine_flat", "left_angle8_mean_deg", "right_angle8_mean_deg",
        "left_angle32_mean_deg", "right_angle32_mean_deg",
    ]
    out = {k: summarize([float(r[k]) for r in records]) for k in keys}
    out["families"] = {}
    for f in sorted({r["family"] for r in records}):
        rs = [r for r in records if r["family"] == f]
        out["families"][f] = {k: mean(float(r[k]) for r in rs) for k in keys}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    weights = {label: load_core(label, path) for label, path in PATHS.items()}
    init = weights["exact_init"]

    static = {}
    for label, mats in weights.items():
        static[label] = {name: {"family": fam(name), **stable_stats(w)} for name, w in mats.items()}

    pair_rows = {}
    for a, b in PAIRS:
        rows = []
        for name in weights[a]:
            wa, wb, wi = weights[a][name], weights[b][name], init[name]
            diff = wb - wa
            flat_a, flat_b = wa.flatten(), wb.flatten()
            cos = float(torch.dot(flat_a, flat_b) / (flat_a.norm() * flat_b.norm() + 1e-30))
            s8 = subspace_summary(wa, wb, 8)
            s32 = subspace_summary(wa, wb, 32)
            rows.append({
                "name": name,
                "family": fam(name),
                "rel_frob_to_start": float(diff.norm() / max(float(wa.norm()), 1e-30)),
                "rel_frob_to_end": float(diff.norm() / max(float(wb.norm()), 1e-30)),
                "rel_frob_to_init": float(diff.norm() / max(float(wi.norm()), 1e-30)),
                "cosine_flat": cos,
                "left_angle8_mean_deg": s8["left_angle_mean_deg"],
                "right_angle8_mean_deg": s8["right_angle_mean_deg"],
                "left_angle32_mean_deg": s32["left_angle_mean_deg"],
                "right_angle32_mean_deg": s32["right_angle_mean_deg"],
            })
        pair_rows[f"{a}_to_{b}"] = {"summary": summarize_records(rows), "rows": rows}

    matched = {}
    for ck, a, b in MATCHED_DIFFS:
        rows = []
        for name in weights[a]:
            wa, wb, wi = weights[a][name], weights[b][name], init[name]
            diff = wa - wb
            rows.append({
                "name": name,
                "family": fam(name),
                "muon_minus_step35_rel_to_init": float(diff.norm() / max(float(wi.norm()), 1e-30)),
                "muon_stable_rank_minus_step35": static[a][name]["stable_rank"] - static[b][name]["stable_rank"],
                "muon_top8_minus_step35": static[a][name]["top8_energy"] - static[b][name]["top8_energy"],
            })
        matched[ck] = {
            "summary": {
                "muon_minus_step35_rel_to_init": summarize([r["muon_minus_step35_rel_to_init"] for r in rows]),
                "muon_stable_rank_minus_step35": summarize([r["muon_stable_rank_minus_step35"] for r in rows]),
                "muon_top8_minus_step35": summarize([r["muon_top8_minus_step35"] for r in rows]),
            },
            "rows": rows,
        }

    result = {"status": "CONTINUOUS_MUON_DRIFT_PROBE", "pair_drift": pair_rows, "matched_step35_muon_difference": matched}
    (_public_path('experiments/archive/frontier_consolidation/data/continuous_muon_drift/continuous_muon_drift_probe.json')).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# research continuous Muon hidden-matrix temporal drift",
        "",
        "All values are means over the 48 attention/FFN matrices unless otherwise noted.",
        "",
        "## Consecutive / interval drift",
        "",
        "| interval | rel Frobenius/start | rel Frobenius/init | flat cosine | left angle k=8 | right angle k=8 | left angle k=32 | right angle k=32 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, obj in pair_rows.items():
        s = obj["summary"]
        lines.append(
            f"| {key} | {s['rel_frob_to_start']['mean']:.4f} | {s['rel_frob_to_init']['mean']:.4f} | {s['cosine_flat']['mean']:.4f} | {s['left_angle8_mean_deg']['mean']:.2f} | {s['right_angle8_mean_deg']['mean']:.2f} | {s['left_angle32_mean_deg']['mean']:.2f} | {s['right_angle32_mean_deg']['mean']:.2f} |"
        )
    lines += ["", "## Muon minus research at same checkpoint", "", "| ckpt | rel difference/init | stable-rank Δ | top8-energy Δ |", "|---|---:|---:|---:|"]
    for ck, obj in matched.items():
        s = obj["summary"]
        lines.append(f"| {ck} | {s['muon_minus_step35_rel_to_init']['mean']:.4f} | {s['muon_stable_rank_minus_step35']['mean']:+.2f} | {s['muon_top8_minus_step35']['mean']:+.4f} |")
    (_public_path('research/documents/frontier_consolidation/data/continuous_muon_drift/continuous_muon_drift_probe.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")

    compact = {
        "status": result["status"],
        "out": str(OUT),
        "interval_summary": {k: {
            "rel_frob_to_start_mean": v["summary"]["rel_frob_to_start"]["mean"],
            "rel_frob_to_init_mean": v["summary"]["rel_frob_to_init"]["mean"],
            "cosine_flat_mean": v["summary"]["cosine_flat"]["mean"],
            "left_angle8_mean": v["summary"]["left_angle8_mean_deg"]["mean"],
            "right_angle8_mean": v["summary"]["right_angle8_mean_deg"]["mean"],
        } for k, v in pair_rows.items()},
        "matched_diff_summary": {k: {
            "rel_diff_to_init_mean": v["summary"]["muon_minus_step35_rel_to_init"]["mean"],
            "stable_rank_delta_mean": v["summary"]["muon_stable_rank_minus_step35"]["mean"],
            "top8_delta_mean": v["summary"]["muon_top8_minus_step35"]["mean"],
        } for k, v in matched.items()},
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
