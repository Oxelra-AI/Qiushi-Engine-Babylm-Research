#!/usr/bin/env python3
"""research post-run mechanism summary.

After the wd-matched 20M arm finishes and evaluation exists, summarize:
- training integrity and loss trajectory against research/research;
- optimizer shrinkage settings from grouping reports;
- hidden-matrix weight spectra including research;
- cheap7 movement relative to research and research.
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

USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')

RUNS = {
    "reference_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2'),
    "muon008_wd01_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_seed43022_20M'),
    "muon012_wd01_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr012_seed43022_20M'),
    "muon008_wd00125_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_20M'),
}
TOK = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
OUT = _public_path('experiments/archive/frontier_consolidation/data/muon_postrun_mechanism')

EVALS = {
    "reference_20M": _public_path('experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json'),
    "muon008_wd01_20M": _public_path('experiments/archive/frontier_consolidation/data/muon_20M_eval/per_target/muon_lr008_20M.json'),
    "muon012_wd01_20M": _public_path('experiments/archive/frontier_consolidation/data/muon_20M_eval/per_target/muon_lr012_20M.json'),
    "muon008_wd00125_20M": _public_path('experiments/archive/frontier_consolidation/data/muon_wdmatched_20M_eval/per_target/muon_lr008_wd00125_20M.json'),
}
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]


def is_core(name: str) -> bool:
    return name.startswith("deberta.encoder.layer.") and name.endswith(".weight") and name.endswith((
        ".attention.self.query_proj.weight", ".attention.self.key_proj.weight", ".attention.self.value_proj.weight",
        ".attention.output.dense.weight", ".intermediate.dense.weight", ".output.dense.weight"))


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
    s2 = s.square(); total = float(s2.sum())
    p = s2 / max(total, 1e-30)
    ent = float(-(p * (p + 1e-30).log()).sum())
    return {
        "frob_norm": float(x.norm()),
        "spectral_norm": float(s[0]),
        "stable_rank": total / max(float(s2[0]), 1e-30),
        "entropy_rank": math.exp(ent),
        "top1_energy": float(p[0]),
        "top8_energy": float(p[:8].sum()),
    }


def summarize_matrices(model):
    recs = {}
    for n, p in model.named_parameters():
        if is_core(n):
            recs[n] = {"family": family(n), **matrix_stats(p)}
    if len(recs) != 48:
        raise RuntimeError(f"expected 48 core matrices, got {len(recs)}")
    out = {}
    for k in ["frob_norm", "spectral_norm", "stable_rank", "entropy_rank", "top1_energy", "top8_energy"]:
        vals = [r[k] for r in recs.values()]
        out[k] = {"mean": mean(vals), "median": median(vals), "min": min(vals), "max": max(vals)}
    out["families"] = {}
    for fam in sorted({r["family"] for r in recs.values()}):
        rs = [r for r in recs.values() if r["family"] == fam]
        out["families"][fam] = {k: mean(r[k] for r in rs) for k in ["frob_norm","stable_rank","entropy_rank","top8_energy"]}
    return out, recs


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def read_log_summary(run_dir: Path):
    p = run_dir / "training_log.jsonl"
    if not p.exists(): return None
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    sample_steps = [1, 50, 100, 200, 300, 400, 506]
    sampled = {str(s): rows[s-1] for s in sample_steps if len(rows) >= s}
    return {
        "n": len(rows),
        "first_loss": rows[0].get("loss") if rows else None,
        "last_loss": rows[-1].get("loss") if rows else None,
        "last_words": rows[-1].get("cumulative_word_exposure") if rows else None,
        "sampled": sampled,
    }


def extract_scores(payload):
    if payload is None: return None
    tasks = payload.get("tasks", {})
    out = {}
    for c in ZERO:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if rec.get("score") is not None else None
    gp = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c, {})
        if rec.get("score") is not None: gp.append(float(rec["score"]))
    out["GlobalPIQA"] = mean(gp) if len(gp) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def cheap7(scores):
    if scores is None: return None
    vals = [scores.get(c) for c in CHEAP]
    return None if any(v is None for v in vals) else mean(float(v) for v in vals)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"status": "POSTRUN_MECHANISM_SUMMARY", "runs": {}, "geometry": {}, "pairwise_vs_init": {}, "scores": {}}

    # run/loss/grouping integrity
    for label, run in RUNS.items():
        result["runs"][label] = {
            "run_dir": str(run),
            "metrics": read_json(run / "scientific_metrics.json"),
            "grouping": read_json(run / "grouping_report.json"),
            "log": read_log_summary(run),
            "checkpoint_exists": (run / "hf_model/chck_20M/model.safetensors").exists(),
        }

    # geometry
    models = {"exact_init": exact_init()}
    for label, run in RUNS.items():
        ck = run / "hf_model/chck_20M"
        if (ck / "model.safetensors").exists():
            models[label] = AutoModelForMaskedLM.from_pretrained(ck)
    raw = {}
    for label, model in models.items():
        summ, rec = summarize_matrices(model)
        result["geometry"][label] = summ
        raw[label] = rec
    for label in [x for x in raw if x != "exact_init"]:
        deltas = []
        for n, cur in raw[label].items():
            init = raw["exact_init"][n]
            deltas.append({
                "frob_ratio": cur["frob_norm"] / init["frob_norm"],
                "stable_rank_delta": cur["stable_rank"] - init["stable_rank"],
                "entropy_rank_delta": cur["entropy_rank"] - init["entropy_rank"],
                "top8_delta": cur["top8_energy"] - init["top8_energy"],
            })
        result["pairwise_vs_init"][label] = {k: mean(d[k] for d in deltas) for k in deltas[0]}

    # scores
    base_scores = None; base_c7 = None
    for label, p in EVALS.items():
        sc = extract_scores(read_json(p))
        c7 = cheap7(sc)
        if label == "reference_20M":
            base_scores, base_c7 = sc, c7
        result["scores"][label] = {"scores": sc, "cheap7": c7}
    if base_scores:
        for label, rec in result["scores"].items():
            sc = rec["scores"]
            if sc:
                rec["deltas_vs_step35"] = {c: sc[c] - base_scores[c] for c in CHEAP}
                rec["deltas_vs_step35"]["cheap7"] = rec["cheap7"] - base_c7

    out_json = _public_path('experiments/archive/frontier_consolidation/data/muon_postrun_mechanism/muon_postrun_mechanism_summary.json')
    out_json.write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# research Muon post-run mechanism summary", ""]
    lines += ["## Scores", "", "| run | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δcheap7 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, rec in result["scores"].items():
        sc = rec.get("scores") or {}; c7 = rec.get("cheap7")
        d = (rec.get("deltas_vs_step35") or {}).get("cheap7")
        def f(c): return f"{sc[c]:.2f}" if sc.get(c) is not None else ""
        lines.append(f"| {label} | {f('BLiMP')} | {f('Supplement')} | {f('EWoK')} | {f('Entity')} | {f('COMPS')} | {f('GlobalPIQA')} | {f('Reading')} | {c7:.4f} | {d:+.4f} |" if c7 is not None and d is not None else f"| {label} | {f('BLiMP')} | {f('Supplement')} | {f('EWoK')} | {f('Entity')} | {f('COMPS')} | {f('GlobalPIQA')} | {f('Reading')} | | |")
    lines += ["", "## Hidden-matrix geometry", "", "| run | frob/init | stable rank | entropy rank | top8 energy |", "|---|---:|---:|---:|---:|"]
    for label, geom in result["geometry"].items():
        pair = result["pairwise_vs_init"].get(label, {})
        frob = pair.get("frob_ratio")
        lines.append(f"| {label} | {frob:.4f} | {geom['stable_rank']['mean']:.2f} | {geom['entropy_rank']['mean']:.2f} | {geom['top8_energy']['mean']:.4f} |" if frob else f"| {label} | 1.0000 | {geom['stable_rank']['mean']:.2f} | {geom['entropy_rank']['mean']:.2f} | {geom['top8_energy']['mean']:.4f} |")
    (_public_path('research/documents/frontier_consolidation/data/muon_postrun_mechanism/muon_postrun_mechanism_summary.md')).write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "out": str(OUT), "scores": result["scores"], "pairwise_vs_init": result["pairwise_vs_init"]}, indent=2))


if __name__ == "__main__":
    main()
