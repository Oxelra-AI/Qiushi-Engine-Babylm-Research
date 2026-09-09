#!/usr/bin/env python3
"""Mechanism readout for adapter128 scale1.75 mature 70M/80M checkpoints.

Reads safetensors directly. If the adapter checkpoints are not present yet, writes a
pending record instead of failing, so the script can be prepared/run before the
managed training task finishes.
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
from typing import Any, Dict, Iterable, Optional

import torch
from safetensors.torch import load_file

USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_mature_mechanism')
RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model')
ADAPTER_RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model')
EXPOSURES = ["50M", "70M", "80M"]


def model_path(root: Path, exp: str) -> Path:
    return root / f"chck_{exp}" / "model.safetensors"


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
        return f"layer{layer}_other"
    if n.startswith("cls."):
        return "mlm_head"
    return "other"


def stock_names(state: Dict[str, torch.Tensor]):
    return [n for n in state if ".adapter." not in n]


def compare(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], stock_only: bool = True) -> Dict[str, Any]:
    names = sorted(set(a) & set(b))
    if stock_only:
        names = [n for n in names if ".adapter." not in n]
    dot = a2 = b2 = d2 = 0.0
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"dot": 0.0, "a2": 0.0, "b2": 0.0, "d2": 0.0, "n_tensors": 0, "n_params": 0})
    for n in names:
        av = a[n].detach().float().cpu().flatten(); bv = b[n].detach().float().cpu().flatten(); dv = av - bv
        aa = float(torch.dot(av, av)); bb = float(torch.dot(bv, bv)); ab = float(torch.dot(av, bv)); dd = float(torch.dot(dv, dv))
        dot += ab; a2 += aa; b2 += bb; d2 += dd
        g = group_name(n); gr = groups[g]
        gr["dot"] += ab; gr["a2"] += aa; gr["b2"] += bb; gr["d2"] += dd; gr["n_tensors"] += 1; gr["n_params"] += int(av.numel())
    for gr in groups.values():
        gr["cosine"] = gr["dot"] / math.sqrt(gr["a2"] * gr["b2"]) if gr["a2"] and gr["b2"] else None
        gr["rel_l2_to_ref"] = math.sqrt(gr["d2"] / gr["b2"]) if gr["b2"] else None
        gr["diff2_fraction"] = gr["d2"] / d2 if d2 else 0.0
    return {
        "n_tensors": len(names),
        "cosine": dot / math.sqrt(a2 * b2) if a2 and b2 else None,
        "rel_l2_to_ref": math.sqrt(d2 / b2) if b2 else None,
        "l2": math.sqrt(d2),
        "a_norm": math.sqrt(a2),
        "ref_norm": math.sqrt(b2),
        "groups": dict(sorted(groups.items(), key=lambda kv: kv[1]["diff2_fraction"], reverse=True)),
    }


def update_alignment(a_hi: Dict[str, torch.Tensor], a_lo: Dict[str, torch.Tensor], b_hi: Dict[str, torch.Tensor], b_lo: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    names = sorted(set(stock_names(a_hi)) & set(stock_names(a_lo)) & set(stock_names(b_hi)) & set(stock_names(b_lo)))
    dot = a2 = b2 = d2 = 0.0
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"dot": 0.0, "a2": 0.0, "b2": 0.0, "d2": 0.0, "n_tensors": 0, "n_params": 0})
    for n in names:
        ua = (a_hi[n].detach().float().cpu() - a_lo[n].detach().float().cpu()).flatten()
        ub = (b_hi[n].detach().float().cpu() - b_lo[n].detach().float().cpu()).flatten()
        aa = float(torch.dot(ua, ua)); bb = float(torch.dot(ub, ub)); ab = float(torch.dot(ua, ub)); dd = float(torch.dot(ua - ub, ua - ub))
        dot += ab; a2 += aa; b2 += bb; d2 += dd
        g = group_name(n); gr = groups[g]
        gr["dot"] += ab; gr["a2"] += aa; gr["b2"] += bb; gr["d2"] += dd; gr["n_tensors"] += 1; gr["n_params"] += int(ua.numel())
    for gr in groups.values():
        gr["cosine"] = gr["dot"] / math.sqrt(gr["a2"] * gr["b2"]) if gr["a2"] and gr["b2"] else None
        gr["adapter_update_rel_norm_vs_step35"] = math.sqrt(gr["a2"] / gr["b2"]) if gr["b2"] else None
        gr["diff2_fraction"] = gr["d2"] / d2 if d2 else 0.0
    return {
        "n_tensors": len(names),
        "cosine": dot / math.sqrt(a2 * b2) if a2 and b2 else None,
        "adapter_update_norm": math.sqrt(a2),
        "update_norm": math.sqrt(b2),
        "adapter_update_rel_norm_vs_step35": math.sqrt(a2 / b2) if b2 else None,
        "update_difference_l2": math.sqrt(d2),
        "groups": dict(sorted(groups.items(), key=lambda kv: kv[1]["diff2_fraction"], reverse=True)),
    }


def adapter_summary(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    by_kind: Dict[str, Dict[str, float]] = defaultdict(lambda: {"sq": 0.0, "numel": 0, "norm_sum": 0.0, "n_tensors": 0})
    by_layer: Dict[str, Dict[str, float]] = defaultdict(lambda: {"sq": 0.0, "numel": 0, "norm_sum": 0.0, "n_tensors": 0})
    for n, t in state.items():
        if ".adapter." not in n:
            continue
        v = t.detach().float().cpu().flatten(); sq = float(torch.dot(v, v)); norm = math.sqrt(sq); numel = int(v.numel())
        if ".adapter.up." in n:
            kind = "up"
        elif ".adapter.down." in n:
            kind = "down"
        elif ".adapter.layer_norm." in n:
            kind = "layer_norm"
        else:
            kind = "other"
        for bucket in (by_kind[kind], by_layer[group_name(n)]):
            bucket["sq"] += sq; bucket["numel"] += numel; bucket["norm_sum"] += norm; bucket["n_tensors"] += 1
    def finish(d: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        out = {}
        for k, v in d.items():
            out[k] = {**v, "rms": math.sqrt(v["sq"] / v["numel"]) if v["numel"] else 0.0}
        return dict(sorted(out.items()))
    total_sq = sum(v["sq"] for v in by_kind.values()); total_numel = sum(v["numel"] for v in by_kind.values())
    return {"total_params": total_numel, "rms_all": math.sqrt(total_sq / total_numel) if total_numel else 0.0, "by_kind": finish(by_kind), "by_layer": finish(by_layer)}


def matrix_spectra(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    recs = []
    for n, t in state.items():
        if ".adapter." in n or not n.startswith("deberta.encoder.layer.") or not n.endswith(".weight") or t.ndim != 2:
            continue
        s = torch.linalg.svdvals(t.detach().float().cpu())
        e = s.square(); total = float(e.sum())
        recs.append({"name": n, "group": group_name(n), "stable_rank": total / float(e.max()) if total else None, "top8_energy": float(e[:8].sum() / total) if total else None})
    return {"n_matrices": len(recs), "stable_rank_mean": float(mean(r["stable_rank"] for r in recs)) if recs else None, "top8_energy_mean": float(mean(r["top8_energy"] for r in recs)) if recs else None}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {f"step35_{e}": model_path(RUN, e) for e in EXPOSURES}
    paths.update({f"scale1p75_{e}": model_path(ADAPTER_RUN, e) for e in EXPOSURES})
    missing = {k: str(p) for k, p in paths.items() if not p.exists()}
    states: Dict[str, Dict[str, torch.Tensor]] = {}
    if missing:
        out = {"status": "SCALE1P75_MATURE_MECHANISM_PENDING", "missing": missing, "paths": {k: str(v) for k, v in paths.items()}}
    else:
        for k, p in paths.items():
            states[k] = load_file(str(p), device="cpu")
            print(f"loaded {k}", flush=True)
        comparisons = {}
        for e in EXPOSURES:
            comparisons[f"scale1p75_{e}_stock_vs_step35_{e}"] = compare(states[f"scale1p75_{e}"], states[f"step35_{e}"], stock_only=True)
        updates = {}
        for lo, hi in [("50M", "70M"), ("70M", "80M"), ("50M", "80M")]:
            updates[f"stock_update_{lo}_to_{hi}"] = update_alignment(states[f"scale1p75_{hi}"], states[f"scale1p75_{lo}"], states[f"step35_{hi}"], states[f"step35_{lo}"])
        adapters = {e: adapter_summary(states[f"scale1p75_{e}"]) for e in EXPOSURES}
        spectra = {k: matrix_spectra(v) for k, v in states.items()}
        out = {"status": "SCALE1P75_MATURE_MECHANISM", "missing": {}, "paths": {k: str(v) for k, v in paths.items()}, "comparisons": comparisons, "stock_update_alignments": updates, "adapter_summary": adapters, "core_matrix_spectra": spectra}
    out_json = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_mature_mechanism/scale1p75_mature_mechanism.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/scale1p75_mature_mechanism/scale1p75_mature_mechanism.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research scale1.75 mature mechanism", "", f"Status: `{out['status']}`"]
    if out.get("missing"):
        lines += ["", "Missing model files:"] + [f"- {k}: `{v}`" for k, v in out["missing"].items()]
    else:
        lines += ["", "## Stock displacement vs matched research", "", "| exposure | cosine | rel L2 | top diff group | top group fraction |", "|---|---:|---:|---|---:|"]
        for e in EXPOSURES:
            c = out["comparisons"][f"scale1p75_{e}_stock_vs_step35_{e}"]
            topg = next(iter(c["groups"].items())) if c["groups"] else ("", {"diff2_fraction": 0})
            lines.append(f"| {e} | {c['cosine']:.6f} | {c['rel_l2_to_ref']:.6f} | {topg[0]} | {topg[1]['diff2_fraction']:.4f} |")
        lines += ["", "## Stock update alignment", "", "| interval | update cosine | adapter/research update norm |", "|---|---:|---:|"]
        for k, u in out["stock_update_alignments"].items():
            lines.append(f"| {k} | {u['cosine']:.6f} | {u['adapter_update_rel_norm_vs_step35']:.6f} |")
        lines += ["", "## Adapter parameter RMS", "", "| exposure | all RMS | up RMS | down RMS | layer-norm RMS |", "|---|---:|---:|---:|---:|"]
        for e, a in out["adapter_summary"].items():
            bk = a["by_kind"]
            lines.append(f"| {e} | {a['rms_all']:.6f} | {bk.get('up', {}).get('rms', 0.0):.6f} | {bk.get('down', {}).get('rms', 0.0):.6f} | {bk.get('layer_norm', {}).get('rms', 0.0):.6f} |")
        lines += ["", "## Core matrix spectra", "", "| checkpoint | stable rank mean | top8 energy mean |", "|---|---:|---:|"]
        for k, s in out["core_matrix_spectra"].items():
            lines.append(f"| {k} | {s['stable_rank_mean']:.3f} | {s['top8_energy_mean']:.4f} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md), "missing": list(out.get("missing", {}).keys())}, indent=2), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
