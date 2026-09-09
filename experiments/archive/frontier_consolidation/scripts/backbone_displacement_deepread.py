#!/usr/bin/env python3
"""Norm-weighted backbone displacement readout for research adapter result.

The earlier mechanism script reported an unweighted mean over parameter tensors. This script
computes whole-vector cosine/relative L2 and per-group/per-tensor contributors so the adapter
co-adaptation interpretation is not based on small tensors.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, os, random, sys
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM

USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/backbone_displacement_deepread')
HF_CACHE = _public_path('experiments/archive/frontier_consolidation/data/backbone_displacement_deepread/hf_modules_cache')
os.environ["HF_MODULES_CACHE"] = str(HF_CACHE)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PATHS = {
    "reference_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_20M'),
    "reference_21M": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_21M'),
    "reference_50M": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_50M'),
    "live128_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022/hf_model/chck_20M'),
    "disabled128_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_disabled_h100M20M_seed43022/hf_model/chck_20M'),
}
TOKENIZER = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')


def reset_all_rng(s: int):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_exact_init_stock():
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    cfg = DebertaV2Config(
        vocab_size=len(tok), hidden_size=480, num_hidden_layers=8, num_attention_heads=8,
        intermediate_size=480 * 4, max_position_embeddings=512, max_relative_positions=256,
        position_buckets=256, relative_attention=True, pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )
    reset_all_rng(43)
    reset_all_rng(43022)
    return DebertaV2ForMaskedLM(cfg).eval()


def load_model(label: str):
    if label == "exact_init_stock":
        return build_exact_init_stock()
    path = PATHS[label]
    if label.startswith("research"):
        # Avoid HF dynamic-module cache writes by importing the local class directly.
        sys.path.insert(0, str(_public_path('experiments/archive/frontier_consolidation/scripts')))
        from adapter_modeling import AdapterDebertaV2ForMaskedLM
        return AdapterDebertaV2ForMaskedLM.from_pretrained(path).eval()
    return DebertaV2ForMaskedLM.from_pretrained(path).eval()


def stock_params(model):
    return {n: p.detach().float().cpu() for n, p in model.named_parameters() if ".adapter." not in n}


def group_name(n: str) -> str:
    if n.startswith("deberta.embeddings"):
        return "embeddings"
    if n.startswith("deberta.encoder.layer."):
        parts = n.split(".")
        layer = parts[3]
        if ".attention." in n:
            sub = "attention"
        elif ".intermediate." in n or ".output." in n:
            sub = "ffn_output"
        else:
            sub = "encoder_other"
        return f"layer{layer}_{sub}"
    if n.startswith("cls."):
        return "mlm_head"
    return "other"


def compare(a_label: str, b_label: str, cache: dict) -> dict:
    if a_label not in cache:
        cache[a_label] = stock_params(load_model(a_label))
    if b_label not in cache:
        cache[b_label] = stock_params(load_model(b_label))
    A, B = cache[a_label], cache[b_label]
    names = [n for n in A if n in B]
    missing_a = [n for n in B if n not in A]
    missing_b = [n for n in A if n not in B]
    dot = 0.0; na2 = 0.0; nb2 = 0.0; diff2 = 0.0
    groups = {}
    tensors = []
    for n in names:
        a = A[n].flatten(); b = B[n].flatten(); d = a - b
        dd = float(torch.dot(d, d)); aa = float(torch.dot(a, a)); bb = float(torch.dot(b, b)); ab = float(torch.dot(a, b))
        dot += ab; na2 += aa; nb2 += bb; diff2 += dd
        g = group_name(n)
        gr = groups.setdefault(g, {"diff2": 0.0, "ref2": 0.0, "a2": 0.0, "dot": 0.0, "n_tensors": 0, "n_params": 0})
        gr["diff2"] += dd; gr["ref2"] += bb; gr["a2"] += aa; gr["dot"] += ab; gr["n_tensors"] += 1; gr["n_params"] += int(a.numel())
        refn = bb ** 0.5
        an = aa ** 0.5
        tensors.append({
            "name": n, "group": g, "n_params": int(a.numel()),
            "l2": dd ** 0.5, "ref_norm": refn, "a_norm": an,
            "rel_l2_to_ref": (dd ** 0.5 / refn) if refn > 0 else None,
            "cosine": (ab / ((aa * bb) ** 0.5)) if aa > 0 and bb > 0 else None,
            "diff2_fraction": 0.0,
        })
    for t in tensors:
        t["diff2_fraction"] = (t["l2"] ** 2 / diff2) if diff2 > 0 else 0.0
    for g, gr in groups.items():
        gr["rel_l2_to_ref"] = (gr["diff2"] ** 0.5 / gr["ref2"] ** 0.5) if gr["ref2"] > 0 else None
        gr["cosine"] = gr["dot"] / ((gr["a2"] * gr["ref2"]) ** 0.5) if gr["a2"] > 0 and gr["ref2"] > 0 else None
        gr["diff2_fraction"] = gr["diff2"] / diff2 if diff2 > 0 else 0.0
    return {
        "a": a_label, "b_reference": b_label, "n_common_tensors": len(names),
        "missing_from_a": missing_a[:10], "missing_from_b": missing_b[:10],
        "total_params_common": int(sum(A[n].numel() for n in names)),
        "whole_vector_cosine": dot / ((na2 * nb2) ** 0.5) if na2 > 0 and nb2 > 0 else None,
        "whole_vector_rel_l2_to_ref": (diff2 ** 0.5 / nb2 ** 0.5) if nb2 > 0 else None,
        "whole_vector_l2": diff2 ** 0.5,
        "reference_norm": nb2 ** 0.5,
        "a_norm": na2 ** 0.5,
        "groups": dict(sorted(groups.items(), key=lambda kv: kv[1]["diff2_fraction"], reverse=True)),
        "top_tensors_by_diff2": sorted(tensors, key=lambda x: x["diff2_fraction"], reverse=True)[:30],
        "top_tensors_by_rel_l2": sorted([t for t in tensors if t["rel_l2_to_ref"] is not None], key=lambda x: x["rel_l2_to_ref"], reverse=True)[:30],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    cache = {}
    comparisons = [
        ("live128_20M", "disabled128_20M"),
        ("disabled128_20M", "reference_20M"),
        ("reference_20M", "exact_init_stock"),
        ("reference_21M", "reference_20M"),
        ("reference_50M", "reference_20M"),
    ]
    out = {"status": "BACKBONE_DISPLACEMENT_DEEPREAD", "comparisons": []}
    for a, b in comparisons:
        out["comparisons"].append(compare(a, b, cache))
        print("compared", a, "vs", b, flush=True)
    (_public_path('experiments/archive/frontier_consolidation/data/backbone_displacement_deepread/backbone_displacement_deepread.json')).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    lines = ["# research backbone displacement deep read", "", "This replaces the earlier unweighted mean-over-tensors displacement with whole-vector and group-weighted measurements.", "", "| comparison | whole-vector cosine | rel L2 to reference | L2 | ref norm | top diff group | top group diff fraction |", "|---|---:|---:|---:|---:|---|---:|"]
    for c in out["comparisons"]:
        topg = next(iter(c["groups"].items())) if c["groups"] else ("", {"diff2_fraction": 0})
        lines.append(f"| {c['a']} vs {c['b_reference']} | {c['whole_vector_cosine']:.8f} | {c['whole_vector_rel_l2_to_ref']:.8f} | {c['whole_vector_l2']:.4f} | {c['reference_norm']:.4f} | {topg[0]} | {topg[1]['diff2_fraction']:.4f} |")
    lines += ["", "## Top tensors for live vs disabled by squared-difference fraction", "", "| tensor | group | frac | rel L2 | cosine |", "|---|---|---:|---:|---:|"]
    c0 = out["comparisons"][0]
    for t in c0["top_tensors_by_diff2"][:20]:
        lines.append(f"| `{t['name']}` | {t['group']} | {t['diff2_fraction']:.4f} | {t['rel_l2_to_ref']:.4f} | {t['cosine']:.6f} |")
    lines += ["", "## Top tensors for live vs disabled by relative L2", "", "| tensor | group | rel L2 | diff fraction | cosine |", "|---|---|---:|---:|---:|"]
    for t in c0["top_tensors_by_rel_l2"][:20]:
        lines.append(f"| `{t['name']}` | {t['group']} | {t['rel_l2_to_ref']:.4f} | {t['diff2_fraction']:.6f} | {t['cosine']:.6f} |")
    (_public_path('research/documents/frontier_consolidation/data/backbone_displacement_deepread/backbone_displacement_deepread.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(_public_path('experiments/archive/frontier_consolidation/data/backbone_displacement_deepread/backbone_displacement_deepread.json')), "out_md": str(_public_path('research/documents/frontier_consolidation/data/backbone_displacement_deepread/backbone_displacement_deepread.md'))}, indent=2), flush=True)

if __name__ == "__main__":
    main()
