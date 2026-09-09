#!/usr/bin/env python3
"""research: CPU mechanism readout for scale1.75 adapter 50M maturation.

Compares the exact reproduced 20M prefix and 50M checkpoint against matched research
20M/50M stock checkpoints. This is intentionally independent of score evaluation and
uses safetensors directly to avoid GPU/HF dynamic-module cache use.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

import torch
from safetensors.torch import load_file

USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_50m_mechanism')
PATHS = {
    "reference_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_20M/model.safetensors'),
    "reference_50M": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_50M/model.safetensors'),
    "scale1p75_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M20M_seed43022/hf_model/chck_20M/model.safetensors'),
    "scale1p75_50M": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M50M_seed43022/hf_model/chck_50M/model.safetensors'),
}


def group_name(n: str) -> str:
    if n.startswith("deberta.embeddings"):
        return "embeddings"
    if n.startswith("deberta.encoder.layer."):
        parts = n.split(".")
        layer = parts[3]
        if ".adapter." in n:
            return f"layer{layer}_adapter"
        if ".attention." in n:
            return f"layer{layer}_attention"
        if ".intermediate." in n or ".output." in n:
            return f"layer{layer}_ffn_output"
        return f"layer{layer}_encoder_other"
    if n.startswith("cls."):
        return "mlm_head"
    return "other"


def stock_items(state: Dict[str, torch.Tensor]):
    for n, t in state.items():
        if ".adapter." not in n:
            yield n, t.detach().float().cpu()


def adapter_items(state: Dict[str, torch.Tensor]):
    for n, t in state.items():
        if ".adapter." in n:
            yield n, t.detach().float().cpu()


def vector_compare(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], *, stock_only: bool = True) -> Dict[str, Any]:
    items = stock_items(a) if stock_only else a.items()
    bmap = {n: t.detach().float().cpu() for n, t in (stock_items(b) if stock_only else b.items())}
    dot = a2 = b2 = d2 = 0.0
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"diff2": 0.0, "a2": 0.0, "b2": 0.0, "dot": 0.0, "n_tensors": 0, "n_params": 0})
    tensors = []
    common = []
    missing = []
    for n, ta in items:
        tb = bmap.get(n)
        if tb is None:
            missing.append(n)
            continue
        common.append(n)
        av = ta.flatten(); bv = tb.flatten(); dv = av - bv
        aa = float(torch.dot(av, av)); bb = float(torch.dot(bv, bv)); ab = float(torch.dot(av, bv)); dd = float(torch.dot(dv, dv))
        a2 += aa; b2 += bb; dot += ab; d2 += dd
        g = group_name(n)
        gr = groups[g]
        gr["diff2"] += dd; gr["a2"] += aa; gr["b2"] += bb; gr["dot"] += ab; gr["n_tensors"] += 1; gr["n_params"] += int(ta.numel())
        tensors.append({"name": n, "group": g, "diff2": dd, "rel_l2_to_ref": math.sqrt(dd / bb) if bb > 0 else None, "cosine": ab / math.sqrt(aa * bb) if aa > 0 and bb > 0 else None, "n_params": int(ta.numel())})
    for gr in groups.values():
        gr["rel_l2_to_ref"] = math.sqrt(gr["diff2"] / gr["b2"]) if gr["b2"] > 0 else None
        gr["cosine"] = gr["dot"] / math.sqrt(gr["a2"] * gr["b2"]) if gr["a2"] > 0 and gr["b2"] > 0 else None
        gr["diff2_fraction"] = gr["diff2"] / d2 if d2 > 0 else 0.0
    for t in tensors:
        t["diff2_fraction"] = t["diff2"] / d2 if d2 > 0 else 0.0
    return {
        "n_common": len(common),
        "n_missing_from_a_side": len(missing),
        "missing_from_a_side_first10": missing[:10],
        "whole_vector_cosine": dot / math.sqrt(a2 * b2) if a2 > 0 and b2 > 0 else None,
        "whole_vector_rel_l2_to_ref": math.sqrt(d2 / b2) if b2 > 0 else None,
        "whole_vector_l2": math.sqrt(d2),
        "a_norm": math.sqrt(a2),
        "ref_norm": math.sqrt(b2),
        "groups": dict(sorted(groups.items(), key=lambda kv: kv[1]["diff2_fraction"], reverse=True)),
        "top_tensors_by_diff2": sorted(tensors, key=lambda x: x["diff2_fraction"], reverse=True)[:20],
    }


def update_alignment(a50: Dict[str, torch.Tensor], a20: Dict[str, torch.Tensor], b50: Dict[str, torch.Tensor], b20: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    # Compare stock update vectors: (adapter50-adapter20) against (reference_50-reference_20).
    names = sorted(set(n for n, _ in stock_items(a50)) & set(n for n, _ in stock_items(a20)) & set(n for n, _ in stock_items(b50)) & set(n for n, _ in stock_items(b20)))
    dot = da2 = db2 = diff2 = 0.0
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"dot": 0.0, "a_update2": 0.0, "b_update2": 0.0, "diff2": 0.0, "n_params": 0, "n_tensors": 0})
    per = []
    for n in names:
        ua = (a50[n].detach().float().cpu() - a20[n].detach().float().cpu()).flatten()
        ub = (b50[n].detach().float().cpu() - b20[n].detach().float().cpu()).flatten()
        aa = float(torch.dot(ua, ua)); bb = float(torch.dot(ub, ub)); ab = float(torch.dot(ua, ub)); dd = float(torch.dot(ua - ub, ua - ub))
        dot += ab; da2 += aa; db2 += bb; diff2 += dd
        g = group_name(n)
        gr = groups[g]
        gr["dot"] += ab; gr["a_update2"] += aa; gr["b_update2"] += bb; gr["diff2"] += dd; gr["n_params"] += int(ua.numel()); gr["n_tensors"] += 1
        per.append({"name": n, "group": g, "adapter_update2": aa, "update2": bb, "dot": ab, "diff2": dd, "cosine": ab / math.sqrt(aa * bb) if aa > 0 and bb > 0 else None})
    for gr in groups.values():
        gr["cosine"] = gr["dot"] / math.sqrt(gr["a_update2"] * gr["b_update2"]) if gr["a_update2"] > 0 and gr["b_update2"] > 0 else None
        gr["adapter_update_rel_norm_vs_step35"] = math.sqrt(gr["a_update2"] / gr["b_update2"]) if gr["b_update2"] > 0 else None
        gr["diff2_fraction"] = gr["diff2"] / diff2 if diff2 > 0 else 0.0
    for p in per:
        p["diff2_fraction"] = p["diff2"] / diff2 if diff2 > 0 else 0.0
    return {
        "n_tensors": len(names),
        "cosine_between_stock_updates": dot / math.sqrt(da2 * db2) if da2 > 0 and db2 > 0 else None,
        "adapter_update_norm": math.sqrt(da2),
        "update_norm": math.sqrt(db2),
        "adapter_update_rel_norm_vs_step35": math.sqrt(da2 / db2) if db2 > 0 else None,
        "update_difference_l2": math.sqrt(diff2),
        "groups": dict(sorted(groups.items(), key=lambda kv: kv[1]["diff2_fraction"], reverse=True)),
        "top_tensors_by_update_difference": sorted(per, key=lambda x: x["diff2_fraction"], reverse=True)[:20],
    }


def adapter_summary(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    stats = []
    by_kind: Dict[str, Dict[str, float]] = defaultdict(lambda: {"norm_sum": 0.0, "sq_sum": 0.0, "numel": 0, "n_tensors": 0})
    by_layer: Dict[str, Dict[str, float]] = defaultdict(lambda: {"norm_sum": 0.0, "sq_sum": 0.0, "numel": 0, "n_tensors": 0})
    for n, t in adapter_items(state):
        v = t.flatten()
        norm = float(v.norm())
        sq = float(torch.dot(v, v))
        kind = "other"
        if ".adapter.up." in n:
            kind = "up"
        elif ".adapter.down." in n:
            kind = "down"
        elif ".adapter.layer_norm." in n:
            kind = "layer_norm"
        layer = group_name(n)
        for bucket in [by_kind[kind], by_layer[layer]]:
            bucket["norm_sum"] += norm; bucket["sq_sum"] += sq; bucket["numel"] += int(v.numel()); bucket["n_tensors"] += 1
        stats.append({"name": n, "kind": kind, "layer": layer, "norm": norm, "rms": math.sqrt(sq / int(v.numel())) if v.numel() else 0.0, "numel": int(v.numel())})
    def finish(d: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        return {k: {**v, "rms": math.sqrt(v["sq_sum"] / v["numel"]) if v["numel"] else 0.0} for k, v in sorted(d.items())}
    return {
        "n_adapter_tensors": len(stats),
        "total_adapter_params": sum(s["numel"] for s in stats),
        "rms_all": math.sqrt(sum(s["rms"] ** 2 * s["numel"] for s in stats) / sum(s["numel"] for s in stats)) if stats else 0.0,
        "by_kind": finish(by_kind),
        "by_layer": finish(by_layer),
        "top_tensors_by_norm": sorted(stats, key=lambda x: x["norm"], reverse=True)[:20],
    }


def stable_rank_and_top8(t: torch.Tensor) -> Tuple[float, float]:
    if t.ndim != 2 or min(t.shape) == 0:
        return float("nan"), float("nan")
    # CPU SVD on 480/1920-sized matrices is cheap enough for this readout.
    s = torch.linalg.svdvals(t.detach().float().cpu())
    e = s.square()
    total = float(e.sum())
    sr = total / float(e.max()) if total > 0 else float("nan")
    top8 = float(e[:8].sum() / total) if total > 0 else float("nan")
    return sr, top8


def is_core_matrix_name(n: str) -> bool:
    if n.endswith(".weight") and n.startswith("deberta.encoder.layer.") and ".adapter." not in n and len(getattr(n, "shape", [])) == 2:
        return True
    return False


def core_matrix_spectra(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    recs = []
    for n, t in stock_items(state):
        if not (n.endswith(".weight") and n.startswith("deberta.encoder.layer.") and t.ndim == 2):
            continue
        sr, top8 = stable_rank_and_top8(t)
        recs.append({"name": n, "group": group_name(n), "shape": list(t.shape), "stable_rank": sr, "top8_energy": top8})
    return {
        "n_matrices": len(recs),
        "stable_rank_mean": float(mean(r["stable_rank"] for r in recs)) if recs else None,
        "top8_energy_mean": float(mean(r["top8_energy"] for r in recs)) if recs else None,
        "by_group": {g: {"n": len(rs), "stable_rank_mean": float(mean(r["stable_rank"] for r in rs)), "top8_energy_mean": float(mean(r["top8_energy"] for r in rs))} for g, rs in _group_records(recs).items()},
    }


def _group_records(recs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in recs:
        out[r["group"]].append(r)
    return dict(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    states = {}
    for label, path in PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        states[label] = load_file(str(path), device="cpu")
        print(f"loaded {label}: {path}", flush=True)

    comparisons = {
        "scale1p75_20M_stock_vs_step35_20M": vector_compare(states["scale1p75_20M"], states["reference_20M"], stock_only=True),
        "scale1p75_50M_stock_vs_step35_50M": vector_compare(states["scale1p75_50M"], states["reference_50M"], stock_only=True),
        "scale1p75_stock_50M_vs_20M": vector_compare(states["scale1p75_50M"], states["scale1p75_20M"], stock_only=True),
        "stock_50M_vs_20M": vector_compare(states["reference_50M"], states["reference_20M"], stock_only=True),
    }
    stock_update = update_alignment(states["scale1p75_50M"], states["scale1p75_20M"], states["reference_50M"], states["reference_20M"])
    adapters = {
        "scale1p75_20M": adapter_summary(states["scale1p75_20M"]),
        "scale1p75_50M": adapter_summary(states["scale1p75_50M"]),
    }
    spectra = {k: core_matrix_spectra(v) for k, v in states.items()}
    out = {
        "status": "SCALE1P75_50M_MECHANISM",
        "paths": {k: str(v) for k, v in PATHS.items()},
        "comparisons": comparisons,
        "stock_update_alignment_20M_to_50M": stock_update,
        "adapter_parameter_summary": adapters,
        "core_matrix_spectra": spectra,
    }
    out_json = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_50m_mechanism/scale1p75_50m_mechanism.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/scale1p75_50m_mechanism/scale1p75_50m_mechanism.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research scale1.75 50M mechanism readout",
        "",
        "## Stock displacement",
        "",
        "| comparison | cosine | rel L2 to reference | L2 | top diff group | top group fraction |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for key, cmp in comparisons.items():
        topg = next(iter(cmp["groups"].items())) if cmp["groups"] else ("", {"diff2_fraction": 0.0})
        lines.append(f"| {key} | {cmp['whole_vector_cosine']:.6f} | {cmp['whole_vector_rel_l2_to_ref']:.6f} | {cmp['whole_vector_l2']:.3f} | {topg[0]} | {topg[1]['diff2_fraction']:.4f} |")
    lines += [
        "",
        "## Stock update alignment from 20M to 50M",
        f"- cosine between scale1.75 stock update and research stock update: {stock_update['cosine_between_stock_updates']:.6f}",
        f"- scale1.75 stock update norm / research stock update norm: {stock_update['adapter_update_rel_norm_vs_step35']:.6f}",
        f"- scale1.75 update norm: {stock_update['adapter_update_norm']:.3f}; research update norm: {stock_update['update_norm']:.3f}",
        "",
        "## Adapter parameter growth",
        "",
        "| checkpoint | adapter RMS all | up RMS | down RMS | layer-norm RMS | up norm sum | down norm sum |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, rec in adapters.items():
        bk = rec["by_kind"]
        lines.append(f"| {key} | {rec['rms_all']:.6f} | {bk.get('up', {}).get('rms', 0.0):.6f} | {bk.get('down', {}).get('rms', 0.0):.6f} | {bk.get('layer_norm', {}).get('rms', 0.0):.6f} | {bk.get('up', {}).get('norm_sum', 0.0):.3f} | {bk.get('down', {}).get('norm_sum', 0.0):.3f} |")
    lines += [
        "",
        "## Core stock-matrix spectra",
        "",
        "| checkpoint | n matrices | stable-rank mean | top8-energy mean |",
        "|---|---:|---:|---:|",
    ]
    for key, rec in spectra.items():
        lines.append(f"| {key} | {rec['n_matrices']} | {rec['stable_rank_mean']:.3f} | {rec['top8_energy_mean']:.4f} |")
    lines += [
        "",
        "## Top 20M-to-50M update-difference tensors",
        "",
        "| tensor | group | diff fraction | update cosine |",
        "|---|---|---:|---:|",
    ]
    for t in stock_update["top_tensors_by_update_difference"][:15]:
        cos = t["cosine"]
        lines.append(f"| `{t['name']}` | {t['group']} | {t['diff2_fraction']:.4f} | {cos:.5f} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md), "stock_update_cosine": stock_update["cosine_between_stock_updates"], "stock_update_rel_norm": stock_update["adapter_update_rel_norm_vs_step35"], "scale50_stock_relL2_vs_step35_50M": comparisons["scale1p75_50M_stock_vs_step35_50M"]["whole_vector_rel_l2_to_ref"]}, indent=2), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
