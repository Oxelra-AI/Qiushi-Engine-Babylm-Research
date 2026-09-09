#!/usr/bin/env python3
"""research: low-cost mechanism readout for mature matched-decay Muon.

This script does not run BabyLM evaluation. It reads completed training artifacts and
checkpoint weights to compare the Muon-hidden / AdamW-interface 80M run against
the research legal compact-view reference. It asks whether the intended hidden
matrix spectral broadening and loss dynamics persist into mature exposure, so
that pending official-compatible scores can be interpreted mechanistically.
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
OUT = _public_path('experiments/archive/frontier_consolidation/data/muon_mature_mechanism')

RUNS = {
    "research": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2'),
    "muon_wdmatch_80run": _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_80M'),
    "muon_wdmatch_20run": _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_20M'),
}

CHECKPOINT_MODELS = {
    "exact_init": None,
    "reference_20M": RUNS["research"] / "hf_model/chck_20M",
    "reference_70M": RUNS["research"] / "hf_model/chck_70M",
    "reference_80M": RUNS["research"] / "hf_model/chck_80M",
    "muon_wdmatch_20M": RUNS["muon_wdmatch_20run"] / "hf_model/chck_20M",
    "muon_wdmatch_70M": RUNS["muon_wdmatch_80run"] / "hf_model/chck_70M",
    "muon_wdmatch_80M": RUNS["muon_wdmatch_80run"] / "hf_model/chck_80M",
}

MILESTONES = [1_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000, 60_000_000, 70_000_000, 80_000_000]


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


def matrix_stats(w: torch.Tensor) -> dict[str, float]:
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
    }


def summarize_recs(recs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ["frob_norm", "spectral_norm", "stable_rank", "entropy_rank", "top1_energy", "top8_energy"]:
        vals = [float(r[key]) for r in recs.values()]
        out[key] = {"mean": mean(vals), "median": median(vals), "min": min(vals), "max": max(vals)}
    out["families"] = {}
    for fam in sorted({str(r["family"]) for r in recs.values()}):
        rs = [r for r in recs.values() if r["family"] == fam]
        out["families"][fam] = {key: mean(float(r[key]) for r in rs) for key in ["frob_norm", "stable_rank", "entropy_rank", "top1_energy", "top8_energy"]}
    return out


def load_core_records(label: str, path: Path | None) -> dict[str, dict[str, Any]]:
    if label == "exact_init":
        model = exact_init_model()
    else:
        assert path is not None
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError(path / "model.safetensors")
        model = AutoModelForMaskedLM.from_pretrained(path)
    recs: dict[str, dict[str, Any]] = {}
    for name, p in model.named_parameters():
        if is_core(name):
            recs[name] = {"family": family(name), **matrix_stats(p)}
    if len(recs) != 48:
        raise RuntimeError(f"{label}: expected 48 core matrices, found {len(recs)}")
    del model
    gc.collect()
    return recs


def parse_log(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def nearest_by_words(rows: list[dict[str, Any]], target: int) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(int(r["cumulative_word_exposure"]) - target))


def log_summary() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, run in RUNS.items():
        log_path = run / "training_log.jsonl"
        metrics_path = run / "scientific_metrics.json"
        rows = parse_log(log_path) if log_path.exists() else []
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        per_milestone = {}
        for m in MILESTONES:
            r = nearest_by_words(rows, m)
            if r is not None:
                per_milestone[str(m)] = {
                    "step": int(r["step"]),
                    "words": int(r["cumulative_word_exposure"]),
                    "loss": float(r["loss"]),
                    "logged_group0_lr": float(r["lr"]),
                    "mask_rate": float(r.get("effective_mask_rate", float("nan"))),
                }
        out[name] = {
            "run": str(run),
            "n_steps": len(rows),
            "metrics_word_exposure": metrics.get("word_exposure"),
            "loss_first": metrics.get("loss_first", rows[0]["loss"] if rows else None),
            "loss_last": metrics.get("loss_last", rows[-1]["loss"] if rows else None),
            "per_milestone": per_milestone,
        }
    # direct deltas where comparable
    deltas = {}
    research = out.get("research", {}).get("per_milestone", {})
    mu80 = out.get("muon_wdmatch_80run", {}).get("per_milestone", {})
    for m in MILESTONES:
        key = str(m)
        if key in research and key in mu80:
            deltas[key] = {
                "loss_muon_minus_step35": mu80[key]["loss"] - research[key]["loss"],
                "words_muon": mu80[key]["words"],
                "words_step35": research[key]["words"],
                "muon_logged_group0_lr": mu80[key]["logged_group0_lr"],
                "logged_lr": research[key]["logged_group0_lr"],
            }
    out["deltas_muon80run_minus_step35"] = deltas
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    logs = log_summary()

    raw = {}
    result: dict[str, Any] = {
        "status": "MUON_MATURE_MECHANISM_SUMMARY",
        "notes": [
            "This is mechanism context only; official-compatible 70M/80M scores decide continuation.",
            "In Muon runs the logged lr is group 0 (Muon hidden matrices); AdamW interface lr is lower by the base-lr ratio 0.001/0.008 under the same scheduler multiplier.",
        ],
        "loss_dynamics": logs,
        "models": {},
        "pairwise_vs_init": {},
        "pairwise_vs_step35_same_ckpt": {},
    }

    for label, path in CHECKPOINT_MODELS.items():
        recs = load_core_records(label, path)
        raw[label] = recs
        result["models"][label] = summarize_recs(recs)

    init = raw["exact_init"]
    for label, recs in raw.items():
        if label == "exact_init":
            continue
        deltas = []
        for n, cur in recs.items():
            base = init[n]
            deltas.append({
                "family": cur["family"],
                "frob_ratio_vs_init": float(cur["frob_norm"]) / float(base["frob_norm"]),
                "stable_rank_delta_vs_init": float(cur["stable_rank"]) - float(base["stable_rank"]),
                "entropy_rank_delta_vs_init": float(cur["entropy_rank"]) - float(base["entropy_rank"]),
                "top8_delta_vs_init": float(cur["top8_energy"]) - float(base["top8_energy"]),
            })
        result["pairwise_vs_init"][label] = {
            "frob_ratio_vs_init_mean": mean(x["frob_ratio_vs_init"] for x in deltas),
            "frob_ratio_vs_init_median": median(x["frob_ratio_vs_init"] for x in deltas),
            "stable_rank_delta_vs_init_mean": mean(x["stable_rank_delta_vs_init"] for x in deltas),
            "entropy_rank_delta_vs_init_mean": mean(x["entropy_rank_delta_vs_init"] for x in deltas),
            "top8_delta_vs_init_mean": mean(x["top8_delta_vs_init"] for x in deltas),
        }

    for ck in ["20M", "70M", "80M"]:
        a = f"muon_wdmatch_{ck}"
        b = f"step35_{ck}"
        if a in raw and b in raw:
            result["pairwise_vs_step35_same_ckpt"][ck] = {
                "stable_rank_mean_delta": result["models"][a]["stable_rank"]["mean"] - result["models"][b]["stable_rank"]["mean"],
                "entropy_rank_mean_delta": result["models"][a]["entropy_rank"]["mean"] - result["models"][b]["entropy_rank"]["mean"],
                "top8_energy_mean_delta": result["models"][a]["top8_energy"]["mean"] - result["models"][b]["top8_energy"]["mean"],
                "frob_norm_mean_delta": result["models"][a]["frob_norm"]["mean"] - result["models"][b]["frob_norm"]["mean"],
            }

    (_public_path('experiments/archive/frontier_consolidation/data/muon_mature_mechanism/muon_mature_mechanism_summary.json')).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# research mature matched-decay Muon mechanism summary",
        "",
        "This readout uses training logs and hidden-matrix checkpoints only; it is not a BabyLM evaluation result.",
        "",
        "## Training-loss milestones",
        "",
        "| words | research loss | Muon 80-run loss | Δ loss | Muon logged group0 lr | research lr |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for m in MILESTONES:
        key = str(m)
        d = logs.get("deltas_muon80run_minus_step35", {}).get(key)
        if not d:
            continue
        s = logs["research"]["per_milestone"][key]
        u = logs["muon_wdmatch_80run"]["per_milestone"][key]
        lines.append(f"| {m:,} | {s['loss']:.4f} | {u['loss']:.4f} | {d['loss_muon_minus_step35']:+.4f} | {u['logged_group0_lr']:.6g} | {s['logged_group0_lr']:.6g} |")

    lines += [
        "",
        "## Hidden attention/FFN matrix spectra",
        "",
        "| model | Frobenius/init | stable rank | entropy rank | top8 energy | stable-rank Δ vs init | top8 Δ vs init |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ["reference_20M", "muon_wdmatch_20M", "reference_70M", "muon_wdmatch_70M", "reference_80M", "muon_wdmatch_80M"]:
        s = result["models"][label]
        p = result["pairwise_vs_init"][label]
        lines.append(
            f"| {label} | {p['frob_ratio_vs_init_mean']:.4f} | {s['stable_rank']['mean']:.2f} | {s['entropy_rank']['mean']:.2f} | {s['top8_energy']['mean']:.4f} | {p['stable_rank_delta_vs_init_mean']:+.2f} | {p['top8_delta_vs_init_mean']:+.4f} |"
        )

    lines += [
        "",
        "## Muon minus research at matched exposure",
        "",
        "| ckpt | stable-rank Δ | entropy-rank Δ | top8-energy Δ | Frobenius-norm Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for ck, d in result["pairwise_vs_step35_same_ckpt"].items():
        lines.append(f"| {ck} | {d['stable_rank_mean_delta']:+.2f} | {d['entropy_rank_mean_delta']:+.2f} | {d['top8_energy_mean_delta']:+.4f} | {d['frob_norm_mean_delta']:+.2f} |")

    (_public_path('research/documents/frontier_consolidation/data/muon_mature_mechanism/muon_mature_mechanism_summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "loss_last_step35": logs["research"]["loss_last"],
        "loss_last_muon80run": logs["muon_wdmatch_80run"]["loss_last"],
        "matched_ckpt_geometry_deltas": result["pairwise_vs_step35_same_ckpt"],
        "out": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
