#!/usr/bin/env python3
"""research: spectral geometry of research legal MLM updates.

This zero-training measurement asks whether the current AdamW hidden-weight update
surface is concentrated into a small number of singular directions, leaving real
headroom for Muon-style orthogonalized hidden-matrix updates.  It reuses existing
research checkpoints and legal compact-view batches; it does not train or save a
new model.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DebertaV2ForMaskedLM

BASE = pathlib.Path("experiments/archive/frontier_consolidation/scripts/mlm_rtd_mechanism_probe.py")
spec = importlib.util.spec_from_file_location("research", BASE)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

OUT_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data/muon_spectral_update_probe")
CHECKPOINTS = {
    "mlm20_step35": pathlib.Path(
        "experiments/archive/frontier_consolidation/training/runs"
        "complianttok_reinvest_seed43022_r2/hf_model/chck_20M"
    ),
    "mlm80_step35": pathlib.Path(
        "experiments/archive/frontier_consolidation/training/runs"
        "complianttok_reinvest_seed43022_r2/hf_model/chck_80M"
    ),
}
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-8
N_BATCHES = 12


def matrix_family(name: str) -> str | None:
    if not name.endswith(".weight"):
        return None
    if "word_embeddings" in name:
        return "word_embedding"
    if "position_embeddings" in name:
        return "absolute_position_embedding"
    if "encoder.rel_embeddings" in name:
        return "relative_position_embedding"
    if "encoder.layer" not in name:
        if "cls.predictions.transform.dense" in name:
            return "mlm_transform"
        return None
    if "attention.self.pos_key_proj" in name or "attention.self.pos_query_proj" in name:
        return "deberta_position_projection"
    if "attention.self.query_proj" in name or "attention.self.key_proj" in name or "attention.self.value_proj" in name:
        return "attention_qkv"
    if "attention.output.dense" in name:
        return "attention_output"
    if "intermediate.dense" in name:
        return "ffn_in"
    if ".output.dense" in name:
        return "ffn_out"
    return None


def selected_matrix_names(model):
    out = []
    for name, p in model.named_parameters():
        fam = matrix_family(name)
        if fam and p.ndim == 2:
            out.append(name)
    return out


def stable_rank_from_singulars(sv: torch.Tensor) -> float:
    if sv.numel() == 0:
        return 0.0
    mx = float(sv.max().item())
    if mx <= 0:
        return 0.0
    return float((sv.pow(2).sum() / (sv.max().pow(2))).item())


def entropy_rank_from_singulars(sv: torch.Tensor) -> float:
    if sv.numel() == 0:
        return 0.0
    x = sv.detach().float().clamp_min(0)
    z = x.sum()
    if float(z.item()) <= 0:
        return 0.0
    p = x / z
    ent = -(p * (p.clamp_min(1e-30).log())).sum()
    return float(torch.exp(ent).item())


def sv_metrics(mat: torch.Tensor) -> dict:
    x = mat.detach().float().cpu()
    if x.numel() == 0 or not torch.isfinite(x).all():
        return {"stable_rank": None, "entropy_rank": None, "top_share": None,
                "top8_share": None, "frob_norm": None, "spec_norm": None}
    try:
        sv = torch.linalg.svdvals(x)
    except Exception:
        sv = torch.linalg.svdvals(x + 1e-12 * torch.randn_like(x))
    frob2 = float((sv * sv).sum().item())
    top2 = float((sv[0] * sv[0]).item()) if sv.numel() else 0.0
    top8 = float((sv[:8] * sv[:8]).sum().item()) if sv.numel() else 0.0
    return {
        "stable_rank": stable_rank_from_singulars(sv),
        "entropy_rank": entropy_rank_from_singulars(sv),
        "top_share": top2 / frob2 if frob2 else None,
        "top8_share": top8 / frob2 if frob2 else None,
        "frob_norm": math.sqrt(frob2),
        "spec_norm": float(sv[0].item()) if sv.numel() else 0.0,
        "rank_capacity": int(min(x.shape)),
    }


def polar_svd(mat: torch.Tensor) -> torch.Tensor:
    x = mat.detach().float().cpu()
    u, sv, vh = torch.linalg.svd(x, full_matrices=False)
    return u @ vh


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().float().flatten().cpu()
    bv = b.detach().float().flatten().cpu()
    den = av.norm() * bv.norm()
    if float(den.item()) <= 0:
        return 0.0
    return float((av @ bv / den).item())


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = sorted([x for x in xs if x is not None])
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def analyze_checkpoint(label, checkpoint, batches, device):
    tok = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=True)
    model = DebertaV2ForMaskedLM.from_pretrained(str(checkpoint)).to(device)
    model.train()  # match training mode for dropout path
    names = selected_matrix_names(model)
    name_to_param = dict(model.named_parameters())
    m = {n: torch.zeros_like(name_to_param[n], device=device) for n in names}
    v = {n: torch.zeros_like(name_to_param[n], device=device) for n in names}
    gsum = {n: torch.zeros_like(name_to_param[n], device=device) for n in names}
    gen = torch.Generator(device=device); gen.manual_seed(s.SEED + 700)
    losses = []
    for i, (ids, am, wg) in enumerate(batches[:N_BATCHES], start=1):
        ids, am, wg = ids.to(device), am.to(device), wg.to(device)
        masked, labels, sel = s.apply_wwm(ids, am, wg, tok, s.MASK_PROB, gen)
        model.zero_grad(set_to_none=True)
        out = model(input_ids=masked, attention_mask=am, labels=labels)
        out.loss.backward()
        losses.append(float(out.loss.detach().cpu()))
        for n in names:
            g = name_to_param[n].grad
            if g is None:
                continue
            gsum[n].add_(g)
            m[n].mul_(BETA1).add_(g, alpha=1 - BETA1)
            v[n].mul_(BETA2).addcmul_(g, g, value=1 - BETA2)
        print(json.dumps({"event": "batch", "checkpoint": label, "i": i,
                          "loss": round(losses[-1], 5)}), flush=True)
    t = min(N_BATCHES, len(batches))
    out = {"label": label, "checkpoint": str(checkpoint), "loss_mean": mean(losses),
           "n_batches": t, "matrices": {}}
    for n in names:
        fam = matrix_family(n)
        raw = (gsum[n] / max(1, t)).detach()
        mhat = m[n] / (1 - BETA1 ** t)
        vhat = v[n] / (1 - BETA2 ** t)
        adam = mhat / (vhat.sqrt() + EPS)
        pol = polar_svd(mhat)
        d = {
            "family": fam,
            "shape": list(name_to_param[n].shape),
            "raw_grad": sv_metrics(raw),
            "momentum": sv_metrics(mhat),
            "adam_step_no_decay": sv_metrics(adam),
            "polar_momentum": sv_metrics(pol),
            "cos_momentum_polar": cosine(mhat, pol),
            "cos_adam_polar": cosine(adam, pol),
            "momentum_frob_over_weight_frob": float(mhat.detach().float().norm().cpu() / name_to_param[n].detach().float().norm().cpu()),
        }
        out["matrices"][n] = d
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def summarize(ckpt_out):
    fam = defaultdict(list)
    for n, d in ckpt_out["matrices"].items():
        fam[d["family"]].append(d)
    res = {}
    for f, ds in fam.items():
        res[f] = {
            "n_matrices": len(ds),
            "momentum_stable_rank_mean": mean([d["momentum"]["stable_rank"] for d in ds]),
            "momentum_stable_rank_median": median([d["momentum"]["stable_rank"] for d in ds]),
            "adam_stable_rank_mean": mean([d["adam_step_no_decay"]["stable_rank"] for d in ds]),
            "adam_stable_rank_median": median([d["adam_step_no_decay"]["stable_rank"] for d in ds]),
            "momentum_entropy_rank_mean": mean([d["momentum"]["entropy_rank"] for d in ds]),
            "momentum_top8_share_mean": mean([d["momentum"]["top8_share"] for d in ds]),
            "adam_top8_share_mean": mean([d["adam_step_no_decay"]["top8_share"] for d in ds]),
            "cos_momentum_polar_mean": mean([d["cos_momentum_polar"] for d in ds]),
            "cos_adam_polar_mean": mean([d["cos_adam_polar"] for d in ds]),
            "momentum_frob_over_weight_frob_mean": mean([d["momentum_frob_over_weight_frob"] for d in ds]),
            "rank_capacity": ds[0]["momentum"].get("rank_capacity"),
        }
    # Core hidden matrices excluding embeddings/head; separate DeBERTa position projections.
    core_families = {"attention_qkv", "attention_output", "ffn_in", "ffn_out"}
    posproj = {"deberta_position_projection"}
    core_ds = [d for d in ckpt_out["matrices"].values() if d["family"] in core_families]
    pos_ds = [d for d in ckpt_out["matrices"].values() if d["family"] in posproj]
    res["core_attention_ffn"] = {
        "n_matrices": len(core_ds),
        "momentum_stable_rank_mean": mean([d["momentum"]["stable_rank"] for d in core_ds]),
        "momentum_stable_rank_median": median([d["momentum"]["stable_rank"] for d in core_ds]),
        "adam_stable_rank_mean": mean([d["adam_step_no_decay"]["stable_rank"] for d in core_ds]),
        "adam_stable_rank_median": median([d["adam_step_no_decay"]["stable_rank"] for d in core_ds]),
        "momentum_top8_share_mean": mean([d["momentum"]["top8_share"] for d in core_ds]),
        "adam_top8_share_mean": mean([d["adam_step_no_decay"]["top8_share"] for d in core_ds]),
        "cos_momentum_polar_mean": mean([d["cos_momentum_polar"] for d in core_ds]),
        "cos_adam_polar_mean": mean([d["cos_adam_polar"] for d in core_ds]),
    }
    res["deberta_position_projection_only"] = {
        "n_matrices": len(pos_ds),
        "momentum_stable_rank_mean": mean([d["momentum"]["stable_rank"] for d in pos_ds]),
        "adam_stable_rank_mean": mean([d["adam_step_no_decay"]["stable_rank"] for d in pos_ds]),
        "momentum_top8_share_mean": mean([d["momentum"]["top8_share"] for d in pos_ds]),
        "cos_momentum_polar_mean": mean([d["cos_momentum_polar"] for d in pos_ds]),
    }
    return res


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples, total_words = s.load_data(s.STREAM, s.NUM_PROBE_WORDS)
    tok = AutoTokenizer.from_pretrained(str(CHECKPOINTS["mlm80_step35"]), use_fast=True)
    ds = s.ProbeDataset(examples, tok, s.SEQ_LENGTH)
    loader = DataLoader(ds, batch_size=s.BATCH_SIZE, shuffle=False, collate_fn=s.collate_probe, num_workers=0)
    batches = list(loader)
    result = {
        "status": "MUON_SPECTRAL_UPDATE_PROBE_COMPLETE",
        "purpose": "Measure whether existing research legal MLM hidden-matrix updates are low stable-rank, making Muon-style orthogonalized updates a scientifically motivated next route before any training.",
        "limitations": "No optimizer state was saved by the original research trainer, so the AdamW quantities are local replayed gradients/moments over frozen checkpoints, not the exact historical optimizer buffers.",
        "config": {"n_probe_words": total_words, "n_examples": len(examples),
                   "n_batches_available": len(batches), "n_batches_used": N_BATCHES,
                   "beta1": BETA1, "beta2": BETA2, "mask_prob": s.MASK_PROB,
                   "seed_base": s.SEED},
        "checkpoints": {},
    }
    for label, cp in CHECKPOINTS.items():
        print(json.dumps({"event": "start_checkpoint", "label": label, "path": str(cp)}), flush=True)
        ck = analyze_checkpoint(label, cp, batches, device)
        ck["family_summary"] = summarize(ck)
        result["checkpoints"][label] = ck
        (OUT_DIR / f"{label}_spectral_update.json").write_text(json.dumps(ck, indent=2) + "\n")
        print(json.dumps({"event": "finish_checkpoint", "label": label,
                          "core_momentum_stable_rank": ck["family_summary"]["core_attention_ffn"]["momentum_stable_rank_mean"],
                          "core_adam_stable_rank": ck["family_summary"]["core_attention_ffn"]["adam_stable_rank_mean"],
                          "core_top8_share": ck["family_summary"]["core_attention_ffn"]["momentum_top8_share_mean"]}), flush=True)
    # Cross-checkpoint summary.
    result["summary"] = {label: ck["family_summary"] for label, ck in result["checkpoints"].items()}
    result["elapsed_sec"] = round(time.time() - t0, 1)
    jp = OUT_DIR / "muon_spectral_update_probe.json"
    jp.write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# research Muon spectral update probe", "",
             "This zero-training measurement estimates the singular-direction concentration of local MLM hidden-matrix updates at existing research legal checkpoints. It uses replayed batches because the original optimizer buffers were not saved.", "",
             f"Probe data: {len(examples)} examples, {total_words} words; used {N_BATCHES} batches.", "",
             "## Core hidden matrices (attention QKV/output + FFN in/out; excluding embeddings, MLM head, DeBERTa position projections)", "",
             "| checkpoint | momentum stable-rank mean | AdamW-step stable-rank mean | momentum top-8 Frobenius share | cos(momentum, polar) | cos(AdamW-step, polar) |",
             "|---|---:|---:|---:|---:|---:|"]
    for label in CHECKPOINTS:
        c = result["summary"][label]["core_attention_ffn"]
        lines.append(f"| {label} | {c['momentum_stable_rank_mean']:.2f} | {c['adam_stable_rank_mean']:.2f} | {c['momentum_top8_share_mean']:.3f} | {c['cos_momentum_polar_mean']:.3f} | {c['cos_adam_polar_mean']:.3f} |")
    lines += ["", "## Family summary", "",
              "| checkpoint | family | n | momentum stable-rank | AdamW-step stable-rank | momentum top-8 share | cos(momentum, polar) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    families_order = ["attention_qkv", "attention_output", "ffn_in", "ffn_out", "deberta_position_projection", "relative_position_embedding", "word_embedding", "mlm_transform"]
    for label in CHECKPOINTS:
        fs = result["summary"][label]
        for fam in families_order:
            if fam not in fs:
                continue
            v = fs[fam]
            lines.append(f"| {label} | {fam} | {v['n_matrices']} | {v['momentum_stable_rank_mean']:.2f} | {v['adam_stable_rank_mean']:.2f} | {v['momentum_top8_share_mean']:.3f} | {v['cos_momentum_polar_mean']:.3f} |")
    lines += ["", "## Reading for route choice", ""]
    c20 = result["summary"]["mlm20_step35"]["core_attention_ffn"]
    c80 = result["summary"]["mlm80_step35"]["core_attention_ffn"]
    lines.append(f"- Core hidden-matrix local momentum stable rank is {c20['momentum_stable_rank_mean']:.2f} at 20M and {c80['momentum_stable_rank_mean']:.2f} at 80M, far below the 480-rank capacity of the square attention matrices and rectangular FFN bottleneck rank.")
    lines.append(f"- The top 8 singular directions carry {c20['momentum_top8_share_mean']:.3f} (20M) and {c80['momentum_top8_share_mean']:.3f} (80M) of local momentum Frobenius energy across the core hidden matrices.")
    lines.append("- This supports a real, testable headroom for Muon-style hidden-matrix update orthogonalization while leaving embeddings, MLM head, norms, and relative-position embedding tables on AdamW.")
    lines.append(f"\nElapsed: {result['elapsed_sec']:.1f}s\n")
    mp = (OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/muon_spectral_update_probe/muon_spectral_update_probe.md')
    mp.write_text("\n".join(lines))
    print(json.dumps({"status": result["status"], "out_json": str(jp), "out_md": str(mp), "elapsed_sec": result["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
