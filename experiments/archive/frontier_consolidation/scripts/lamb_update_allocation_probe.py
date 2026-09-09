#!/usr/bin/env python3
"""research: LAMB update-allocation probe on the research legal substrate.

The research LAMB runs are intended to test a whole-system optimizer mechanism:
preserve Adam-like per-tensor update directions while changing layer/tensor update
magnitudes by trust ratios. This script measures that mechanism directly on a
fixed masked batch, without training.

For each model checkpoint it computes an instantaneous first-step gradient on the
same WWM batch and reports:
  * per-tensor trust ratio ||w|| / ||Adam-like update|| with clamp
  * update-norm multiplier relative to research AdamW lr=0.001
  * cosine between pure Adam direction and Adam+weight-decay direction
  * full-model cosine between LAMB-scaled and AdamW-uniform update allocations
  * family/layer summaries, including whether embeddings/relative positions/head
    receive extreme trust-ratio scaling.

Important limitation: optimizer moment buffers are not stored in the HF checkpoints,
so this is a local update-allocation readout, not a reconstruction of the exact
historical LAMB states. It is still the right cheap test for the intended mechanism
before any mature GPU continuation.
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
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
BASE_TRAINER = _public_path('experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py')
CORPUS = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
TOK = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
OUT = _public_path('experiments/archive/frontier_consolidation/data/lamb_update_allocation_probe')

MODEL_PATHS = {
    "reference_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_20M'),
    "lamb005_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_20M/hf_model/chck_20M'),
    "lamb007_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr007_seed43022_20M/hf_model/chck_20M'),
    "lamb005_14M": _public_path('experiments/archive/frontier_consolidation/training/runs/lamb_lr005_seed43022_partial14M_shadow/hf_model/chck_14M'),
}

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def load_base_module():
    spec = importlib.util.spec_from_file_location("masking_curriculum_trainer_probe", str(BASE_TRAINER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def reset_all_rng(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exact_init_model(tokenizer):
    reset_all_rng(43)
    reset_all_rng(43022)
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer), hidden_size=480, num_hidden_layers=8,
        num_attention_heads=8, intermediate_size=1920,
        max_position_embeddings=512, max_relative_positions=256,
        position_buckets=256, relative_attention=True,
        pos_att_type=["p2c", "c2p"], hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return DebertaV2ForMaskedLM(cfg)


def load_first_examples(n_examples: int, base_mod):
    examples = []
    with CORPUS.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch at row {i}")
            examples.append(base_mod.Example(
                text=text,
                words=words,
                example_id=int(obj.get("example_id", i)),
                source=str(obj.get("source", "example_jsonl")),
            ))
            if len(examples) >= n_examples:
                break
    if len(examples) < n_examples:
        raise RuntimeError(f"only loaded {len(examples)} examples")
    return examples


def make_fixed_batch(base_mod, tokenizer, batch_size: int, device: torch.device):
    examples = load_first_examples(batch_size, base_mod)
    dataset = base_mod.MaskedChunkDataset(examples, tokenizer, seq_length=256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=base_mod.collate, num_workers=0)
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    word_group = batch["word_group"].to(device)
    state = base_mod.MaskingCurriculumState(
        curriculum="wwm_fixed",
        mask_prob_start=0.15,
        mask_prob_end=0.15,
        switch_frac=0.7,
        amlm_window=10,
        amlm_lambda=0.2,
    )
    state.initialize(vocab_size=len(tokenizer), total_steps=506)
    state.current_step = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(43023)
    masked_inputs, labels = base_mod.apply_masking_curriculum(
        input_ids, attention_mask, word_group, tokenizer, state, gen
    )
    return {"input_ids": masked_inputs, "attention_mask": attention_mask, "labels": labels,
            "masked_tokens": int((labels != -100).sum().item()),
            "candidate_tokens": int(attention_mask.sum().item())}


def family(name: str) -> str:
    if "word_embeddings" in name:
        return "word_embeddings"
    if "rel_embeddings" in name or "relative" in name:
        return "relative_position"
    if name.startswith("deberta.encoder.layer."):
        parts = name.split(".")
        layer = parts[3] if len(parts) > 3 else "?"
        if ".attention.self.query_proj." in name or ".attention.self.key_proj." in name or ".attention.self.value_proj." in name:
            return f"L{layer}_attention_qkv"
        if ".attention.output.dense." in name:
            return f"L{layer}_attention_out"
        if ".intermediate.dense." in name:
            return f"L{layer}_ffn_in"
        if ".output.dense." in name:
            return f"L{layer}_ffn_out"
        if "LayerNorm" in name or name.endswith("bias"):
            return f"L{layer}_norm_bias"
        return f"L{layer}_other"
    if name.startswith("cls.") or "lm_predictions" in name:
        return "mlm_head"
    if name.endswith("bias") or "LayerNorm" in name or ".layer_norm" in name:
        return "other_norm_bias"
    return "other"


def broad_family(fam: str) -> str:
    if fam.startswith("L"):
        if "attention_qkv" in fam:
            return "encoder_attention_qkv"
        if "attention_out" in fam:
            return "encoder_attention_out"
        if "ffn_in" in fam:
            return "encoder_ffn_in"
        if "ffn_out" in fam:
            return "encoder_ffn_out"
        if "norm_bias" in fam:
            return "encoder_norm_bias"
        return "encoder_other"
    return fam


def layer_index(name: str) -> int | None:
    if name.startswith("deberta.encoder.layer."):
        try:
            return int(name.split(".")[3])
        except Exception:
            return None
    return None


def is_core_hidden_matrix(name: str, p: torch.Tensor) -> bool:
    if p.ndim != 2:
        return False
    return name.startswith("deberta.encoder.layer.") and name.endswith((
        ".attention.self.query_proj.weight", ".attention.self.key_proj.weight",
        ".attention.self.value_proj.weight", ".attention.output.dense.weight",
        ".intermediate.dense.weight", ".output.dense.weight"))


def safe_cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    an = torch.linalg.vector_norm(af)
    bn = torch.linalg.vector_norm(bf)
    if float(an) == 0.0 or float(bn) == 0.0:
        return None
    return float(torch.dot(af, bf) / (an * bn))


def stable_rank_stats(w: torch.Tensor) -> dict | None:
    if w.ndim != 2:
        return None
    x = w.detach().float().cpu()
    s = torch.linalg.svdvals(x)
    s2 = s.square()
    total = float(s2.sum())
    if total <= 0:
        return None
    p = s2 / total
    ent = float(-(p * (p + 1e-30).log()).sum())
    return {
        "stable_rank": total / max(float(s2[0]), 1e-30),
        "entropy_rank": math.exp(ent),
        "top8_energy": float(p[:8].sum()),
    }


@dataclass
class Accum:
    n: int = 0
    params: int = 0
    w_norms: list = None
    adam_norms: list = None
    base_norms: list = None
    trust_ratios: list = None
    clipped_count: int = 0
    multipliers: list = None
    decay_frac: list = None
    pure_to_base_cos: list = None
    weighted_lamb_sq: float = 0.0
    weighted_adam_sq: float = 0.0
    weighted_dot: float = 0.0
    stepnorm_lamb: float = 0.0
    stepnorm_adamw: float = 0.0

    def __post_init__(self):
        self.w_norms = []
        self.adam_norms = []
        self.base_norms = []
        self.trust_ratios = []
        self.multipliers = []
        self.decay_frac = []
        self.pure_to_base_cos = []

    def add(self, rec: dict):
        self.n += 1
        self.params += rec["numel"]
        self.w_norms.append(rec["w_norm"])
        self.adam_norms.append(rec["adam_norm"])
        self.base_norms.append(rec["base_update_norm"])
        self.trust_ratios.append(rec["trust_ratio"])
        self.multipliers.append(rec["lamb_vs_step35_adamw_stepnorm_multiplier"])
        self.decay_frac.append(rec["wd_over_adam_norm"])
        if rec["pure_adam_to_base_update_cos"] is not None:
            self.pure_to_base_cos.append(rec["pure_adam_to_base_update_cos"])
        if rec["trust_ratio_clipped"]:
            self.clipped_count += 1
        # For full-vector cosine between LAMB allocation and uniform AdamW allocation.
        bn2 = rec["base_update_norm"] ** 2
        tr = rec["trust_ratio"]
        self.weighted_lamb_sq += (tr ** 2) * bn2
        self.weighted_adam_sq += bn2
        self.weighted_dot += tr * bn2
        self.stepnorm_lamb += rec["lamb_step_norm_sq"]
        self.stepnorm_adamw += rec["adamw_step_norm_sq"]

    def summary(self) -> dict:
        def sm(vals):
            if not vals:
                return None
            return {
                "mean": float(mean(vals)),
                "median": float(median(vals)),
                "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90)),
                "min": float(min(vals)),
                "max": float(max(vals)),
            }
        denom = math.sqrt(max(self.weighted_lamb_sq, 1e-300) * max(self.weighted_adam_sq, 1e-300))
        full_cos = self.weighted_dot / denom if denom > 0 else None
        return {
            "n_tensors": self.n,
            "n_parameters": self.params,
            "trust_ratio": sm(self.trust_ratios),
            "lamb_vs_step35_adamw_stepnorm_multiplier": sm(self.multipliers),
            "wd_over_adam_norm": sm(self.decay_frac),
            "pure_adam_to_base_update_cos": sm(self.pure_to_base_cos),
            "trust_ratio_clipped_count": self.clipped_count,
            "full_vector_cos_lamb_allocation_vs_uniform_adamw": full_cos,
            "full_lamb_step_norm_over_step35_adamw_step_norm": math.sqrt(self.stepnorm_lamb / max(self.stepnorm_adamw, 1e-300)),
        }


def probe_model(label: str, model, batch: dict, device: torch.device, lamb_lr: float, wd: float,
                eps: float, clamp: float, lr: float, dropout_seed: int) -> dict:
    model.to(device)
    model.train()
    reset_all_rng(dropout_seed)
    model.zero_grad(set_to_none=True)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    loss = out.loss
    if loss is None:
        raise RuntimeError(f"{label}: no loss")
    loss.backward()
    total_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())

    tensor_records = {}
    family_acc: dict[str, Accum] = {}
    broad_acc: dict[str, Accum] = {}
    layer_acc: dict[str, Accum] = {}
    all_acc = Accum()
    core_spectral = []

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        w = p.detach().float()
        adam = g / (g.abs() + eps)  # first-step bias-corrected Adam approximation after global clip
        wd_vec = w * wd
        base_update = adam + wd_vec
        w_norm = float(torch.linalg.vector_norm(w).detach().cpu())
        adam_norm = float(torch.linalg.vector_norm(adam).detach().cpu())
        wd_norm = float(torch.linalg.vector_norm(wd_vec).detach().cpu())
        base_norm = float(torch.linalg.vector_norm(base_update).detach().cpu())
        if w_norm > 0.0 and base_norm > 0.0:
            raw_trust = w_norm / base_norm
            trust = min(raw_trust, clamp) if clamp > 0 else raw_trust
        else:
            raw_trust = 1.0
            trust = 1.0
        multiplier = (lamb_lr * trust) / lr
        pure_cos = safe_cos(adam, base_update)
        rec = {
            "name": name,
            "family": family(name),
            "broad_family": broad_family(family(name)),
            "layer": layer_index(name),
            "shape": list(p.shape),
            "numel": int(p.numel()),
            "w_norm": w_norm,
            "adam_norm": adam_norm,
            "wd_norm": wd_norm,
            "base_update_norm": base_norm,
            "wd_over_adam_norm": wd_norm / max(adam_norm, 1e-30),
            "raw_trust_ratio": raw_trust,
            "trust_ratio": trust,
            "trust_ratio_clipped": bool(clamp > 0 and raw_trust > clamp),
            "lamb_vs_step35_adamw_stepnorm_multiplier": multiplier,
            "pure_adam_to_base_update_cos": pure_cos,
            "lamb_step_norm_sq": (lamb_lr * trust * base_norm) ** 2,
            "adamw_step_norm_sq": (lr * base_norm) ** 2,
            "is_core_hidden_matrix": is_core_hidden_matrix(name, p),
        }
        sp = stable_rank_stats(w) if rec["is_core_hidden_matrix"] else None
        if sp:
            rec.update(sp)
            core_spectral.append(sp)
        tensor_records[name] = rec
        all_acc.add(rec)
        family_acc.setdefault(rec["family"], Accum()).add(rec)
        broad_acc.setdefault(rec["broad_family"], Accum()).add(rec)
        if rec["layer"] is not None:
            layer_acc.setdefault(f"L{rec['layer']}", Accum()).add(rec)

    def top_by(field: str, reverse=True, k=20):
        vals = sorted(tensor_records.values(), key=lambda r: r[field], reverse=reverse)
        return [{x: r[x] for x in ["name", "family", "shape", "numel", field, "trust_ratio", "lamb_vs_step35_adamw_stepnorm_multiplier", "wd_over_adam_norm"] if x in r} for r in vals[:k]]

    result = {
        "label": label,
        "loss_on_fixed_batch": float(loss.detach().cpu()),
        "preclip_total_grad_norm": total_grad_norm,
        "batch": {"masked_tokens": batch["masked_tokens"], "candidate_tokens": batch["candidate_tokens"]},
        "assumed_lamb_lr": lamb_lr,
        "assumed_weight_decay": wd,
        "assumed_eps": eps,
        "assumed_clamp": clamp,
        "reference_step35_adamw_lr": lr,
        "all_tensors": all_acc.summary(),
        "by_broad_family": {k: v.summary() for k, v in sorted(broad_acc.items())},
        "by_layer": {k: v.summary() for k, v in sorted(layer_acc.items())},
        "top_trust_ratios": top_by("trust_ratio", True, 20),
        "lowest_trust_ratios": top_by("trust_ratio", False, 20),
        "top_stepnorm_multipliers": top_by("lamb_vs_step35_adamw_stepnorm_multiplier", True, 20),
        "core_hidden_spectral_means": {
            "stable_rank": float(mean([x["stable_rank"] for x in core_spectral])) if core_spectral else None,
            "entropy_rank": float(mean([x["entropy_rank"] for x in core_spectral])) if core_spectral else None,
            "top8_energy": float(mean([x["top8_energy"] for x in core_spectral])) if core_spectral else None,
        },
    }
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def make_markdown(results: dict) -> str:
    lines = [
        "# research LAMB update-allocation probe",
        "",
        "Fixed first research WWM batch; local first-step gradient readout. Optimizer moment buffers are not available in HF checkpoints, so this measures the intended update-allocation mechanism rather than exact historical moments.",
        "",
        "## All-tensor summary",
        "",
        "| model | loss | grad_norm | trust p50 | trust p10-p90 | clipped | full-vector cos vs uniform AdamW | LAMB/AdamW full step norm | multiplier p50 | multiplier p10-p90 | WD/Adam p50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in results["models"].items():
        a = r["all_tensors"]
        tr = a["trust_ratio"]
        mult = a["lamb_vs_step35_adamw_stepnorm_multiplier"]
        wd = a["wd_over_adam_norm"]
        lines.append(
            f"| {label} | {r['loss_on_fixed_batch']:.4f} | {r['preclip_total_grad_norm']:.4f} | "
            f"{tr['median']:.4f} | {tr['p10']:.4f}-{tr['p90']:.4f} | {a['trust_ratio_clipped_count']} | "
            f"{a['full_vector_cos_lamb_allocation_vs_uniform_adamw']:.4f} | {a['full_lamb_step_norm_over_step35_adamw_step_norm']:.2f} | "
            f"{mult['median']:.2f} | {mult['p10']:.2f}-{mult['p90']:.2f} | {wd['median']:.4f} |"
        )
    lines += ["", "## Broad family trust ratios and step multipliers", ""]
    for label, r in results["models"].items():
        lines += [f"### {label}", "", "| family | tensors | trust median | trust p10-p90 | multiplier median | multiplier p10-p90 | full-vector cos | full stepnorm ratio |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for fam, s in r["by_broad_family"].items():
            tr = s["trust_ratio"]
            mult = s["lamb_vs_step35_adamw_stepnorm_multiplier"]
            lines.append(f"| {fam} | {s['n_tensors']} | {tr['median']:.4f} | {tr['p10']:.4f}-{tr['p90']:.4f} | {mult['median']:.2f} | {mult['p10']:.2f}-{mult['p90']:.2f} | {s['full_vector_cos_lamb_allocation_vs_uniform_adamw']:.4f} | {s['full_lamb_step_norm_over_step35_adamw_step_norm']:.2f} |")
        lines.append("")
    lines += ["## Core hidden spectra", "", "| model | stable rank | entropy rank | top8 energy |", "|---|---:|---:|---:|"]
    for label, r in results["models"].items():
        sp = r["core_hidden_spectral_means"]
        lines.append(f"| {label} | {sp['stable_rank']:.2f} | {sp['entropy_rank']:.2f} | {sp['top8_energy']:.4f} |")
    lines += ["", "## Interpretation notes", "", "A high per-tensor direction cosine with a low full-vector cosine means LAMB preserves local Adam-like directions while reallocating update magnitude across tensors/layers. Extreme trust-ratio clipping or very large interface multipliers would weaken the intended mechanism and make a mature continuation less scientifically clean."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="exact_init,reference_20M,lamb005_20M,lamb007_20M",
                    help="Comma labels from exact_init plus MODEL_PATHS keys")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lamb_lr", type=float, default=0.005,
                    help="Nominal LAMB LR used for multiplier; label lamb007 overrides to 0.007")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--clamp", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--dropout_seed", type=int, default=123456)
    args = ap.parse_args()

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    base_mod = load_base_module()
    tokenizer = AutoTokenizer.from_pretrained(TOK, use_fast=True)
    batch = make_fixed_batch(base_mod, tokenizer, args.batch_size, device)

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    results = {
        "status": "LAMB_UPDATE_ALLOCATION_PROBE",
        "batch_size": args.batch_size,
        "device": str(device),
        "labels_requested": labels,
        "models": {},
        "skipped": {},
        "paths": {"corpus": str(CORPUS), "tokenizer": str(TOK), "base_trainer": str(BASE_TRAINER)},
    }

    for label in labels:
        if label == "exact_init":
            model = exact_init_model(tokenizer)
            lr = 0.005
        else:
            path = MODEL_PATHS.get(label)
            if path is None:
                results["skipped"][label] = "unknown label"
                continue
            if not (path / "model.safetensors").exists():
                results["skipped"][label] = f"missing {path}/model.safetensors"
                continue
            model = AutoModelForMaskedLM.from_pretrained(path)
            lr = 0.007 if "007" in label else args.lamb_lr
        results["models"][label] = probe_model(
            label, model, batch, device, lamb_lr=lr, wd=args.weight_decay,
            eps=args.eps, clamp=args.clamp, lr=args.lr,
            dropout_seed=args.dropout_seed,
        )

    out_json = _public_path('experiments/archive/frontier_consolidation/data/lamb_update_allocation_probe/lamb_update_allocation_probe.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/lamb_update_allocation_probe/lamb_update_allocation_probe.md')
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(make_markdown(results), encoding="utf-8")
    print(json.dumps({
        "status": results["status"],
        "models": list(results["models"].keys()),
        "skipped": results["skipped"],
        "out": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
