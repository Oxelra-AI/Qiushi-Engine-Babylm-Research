#!/usr/bin/env python3
"""Benchmark-independent weight-trajectory geometry for scale1.75 77M-100M.

This script uses existing saved checkpoints only.  It measures stock-vs-adapter
parameter norms, consecutive checkpoint update norms, and consecutive-update
cosines, then joins them to the already-computed cheap7 window.  It is intended
to test whether the 82M competence peak has an obvious legal internal signal in
weight space, without running new model inference or training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import time
from typing import Any

import torch
from safetensors.torch import safe_open

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
A01 = _public_path('experiments/archive/representation_and_objectives')
RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder')
MODEL_ROOT = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model')
SWEEP = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_checkpoint_sweep/summary/scale1p75_checkpoint_sweep_summary.json')
CHCK82 = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
FULL100 = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_100M_repaired_merge/summary/scale1p75_100M_full_eval_hardened_summary.json')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_peak_weight_geometry')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_peak_weight_geometry/scale1p75_peak_weight_geometry.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/scale1p75_peak_weight_geometry/scale1p75_peak_weight_geometry.md')

CHECKPOINTS = [f"chck_{m}M" for m in range(77, 86)] + ["chck_90M", "chck_100M"]
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    pp = pathlib.Path(p)
    try:
        return str(pp.resolve().relative_to(ROOT))
    except Exception:
        return str(pp)


def is_adapter_key(k: str) -> bool:
    return ".adapter." in k or k.startswith("adapter")


def key_group(k: str) -> str:
    if is_adapter_key(k):
        if ".adapter.up." in k:
            return "adapter_up"
        if ".adapter.down." in k:
            return "adapter_down"
        if ".adapter.layer_norm." in k:
            return "adapter_ln"
        return "adapter_other"
    if k.startswith("deberta.embeddings"):
        return "embeddings"
    if k.startswith("cls."):
        return "lm_head"
    if ".attention." in k:
        return "attention"
    if ".intermediate." in k or ".output." in k:
        return "ffn_or_output"
    return "stock_other"


def load_metrics_json(path: pathlib.Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_key_sums(ckpt: str) -> dict[str, Any]:
    path = MODEL_ROOT / ckpt / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(path)
    group_sums: dict[str, dict[str, float]] = {}
    total_sq = 0.0
    total_n = 0
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k).float().reshape(-1)
            sq = float(torch.dot(t, t).item())
            n = int(t.numel())
            g = key_group(k)
            rec = group_sums.setdefault(g, {"sq": 0.0, "n": 0})
            rec["sq"] += sq
            rec["n"] += n
            total_sq += sq
            total_n += n
    out_groups = {}
    for g, rec in group_sums.items():
        out_groups[g] = {
            "n": int(rec["n"]),
            "norm": math.sqrt(rec["sq"]),
            "rms": math.sqrt(rec["sq"] / max(rec["n"], 1)),
            "sq": rec["sq"],
        }
    adapter_sq = sum(rec["sq"] for g, rec in group_sums.items() if g.startswith("adapter"))
    adapter_n = sum(rec["n"] for g, rec in group_sums.items() if g.startswith("adapter"))
    stock_sq = total_sq - adapter_sq
    stock_n = total_n - adapter_n
    return {
        "checkpoint": ckpt,
        "path": rel(path),
        "param_count": total_n,
        "total_norm": math.sqrt(total_sq),
        "total_rms": math.sqrt(total_sq / max(total_n, 1)),
        "stock_norm": math.sqrt(stock_sq),
        "stock_rms": math.sqrt(stock_sq / max(stock_n, 1)),
        "adapter_norm": math.sqrt(adapter_sq),
        "adapter_rms": math.sqrt(adapter_sq / max(adapter_n, 1)),
        "adapter_to_stock_norm_ratio": math.sqrt(adapter_sq) / max(math.sqrt(stock_sq), 1e-30),
        "groups": out_groups,
    }


def pair_delta(prev: str, cur: str, prev_delta: dict[str, torch.Tensor] | None = None):
    p0 = MODEL_ROOT / prev / "model.safetensors"
    p1 = MODEL_ROOT / cur / "model.safetensors"
    sums = {"total": [0.0, 0], "stock": [0.0, 0], "adapter": [0.0, 0]}
    groups: dict[str, list[float]] = {}
    dot_prev = {"total": 0.0, "stock": 0.0, "adapter": 0.0}
    norm_prev = {"total": 0.0, "stock": 0.0, "adapter": 0.0}
    delta_dict: dict[str, torch.Tensor] = {}
    with safe_open(str(p0), framework="pt", device="cpu") as f0, safe_open(str(p1), framework="pt", device="cpu") as f1:
        keys = list(f0.keys())
        if keys != list(f1.keys()):
            raise RuntimeError(f"key mismatch {prev}->{cur}")
        for k in keys:
            d = (f1.get_tensor(k).float() - f0.get_tensor(k).float()).reshape(-1).contiguous()
            delta_dict[k] = d
            sq = float(torch.dot(d, d).item())
            n = int(d.numel())
            kind = "adapter" if is_adapter_key(k) else "stock"
            sums["total"][0] += sq; sums["total"][1] += n
            sums[kind][0] += sq; sums[kind][1] += n
            g = key_group(k)
            rec = groups.setdefault(g, [0.0, 0])
            rec[0] += sq; rec[1] += n
            if prev_delta is not None:
                pd = prev_delta[k]
                dp = float(torch.dot(d, pd).item())
                psq = float(torch.dot(pd, pd).item())
                dot_prev["total"] += dp
                norm_prev["total"] += psq
                dot_prev[kind] += dp
                norm_prev[kind] += psq
    out: dict[str, Any] = {
        "from": prev,
        "to": cur,
        "groups": {},
    }
    for kind in ["total", "stock", "adapter"]:
        sq, n = sums[kind]
        out[f"delta_{kind}_norm"] = math.sqrt(sq)
        out[f"delta_{kind}_rms"] = math.sqrt(sq / max(n, 1))
        out[f"delta_{kind}_param_count"] = int(n)
        if prev_delta is not None:
            den = math.sqrt(max(sq, 0.0) * max(norm_prev[kind], 0.0))
            out[f"cosine_with_previous_delta_{kind}"] = (dot_prev[kind] / den) if den > 0 else None
    out["delta_adapter_norm_fraction"] = out["delta_adapter_norm"] / max(out["delta_total_norm"], 1e-30)
    for g, (sq, n) in groups.items():
        out["groups"][g] = {"delta_norm": math.sqrt(sq), "delta_rms": math.sqrt(sq / max(n, 1)), "param_count": int(n)}
    return out, delta_dict


def score_join() -> dict[str, Any]:
    sweep = load_metrics_json(SWEEP)
    rows = {r["endpoint"]: r for r in sweep.get("rows", [])} if sweep else {}
    ref82 = load_metrics_json(CHCK82)
    full100 = load_metrics_json(FULL100)
    scores: dict[str, Any] = {}
    for ck in CHECKPOINTS:
        rec = rows.get(ck)
        if rec:
            scores[ck] = {k: rec.get(k) for k in CHEAP_COLS if k in rec}
            scores[ck]["cheap7"] = rec.get("cheap7")
            scores[ck]["projected_overall_if_80M_superglue_aoa0"] = rec.get("projected_overall_if_superglue_80M_aoa0")
            scores[ck]["superglue_required_for_41p8_with_aoa0"] = rec.get("superglue_required_for_41p8_with_aoa0")
    if ref82:
        scores.setdefault("chck_82M", {})["official_overall_full"] = ref82["score_arithmetic"]["overall_reported"]
        scores["chck_82M"]["official_superglue_full"] = ref82["score_arithmetic"]["scores"]["SuperGLUE"]
        scores["chck_82M"]["official_aoa_full"] = ref82["score_arithmetic"]["scores"]["AoA"]
        scores["chck_82M"]["cheap7_recomputed_hardened"] = ref82["score_arithmetic"]["cheap7_recomputed"]
    # known 80M full score is in sweep summary.
    if sweep and sweep.get("known_80M_reference"):
        k80 = sweep["known_80M_reference"]
        scores.setdefault("chck_80M", {})["official_overall_full"] = k80.get("Overall")
        scores["chck_80M"]["official_superglue_full"] = k80.get("SuperGLUE")
        scores["chck_80M"]["official_aoa_full"] = k80.get("AoA")
    if full100:
        # two possible schema variants: summary has official_overall/scores or top-level scores
        oo = full100.get("official_overall", {})
        fscores = oo.get("scores", full100.get("scores", {}))
        scores.setdefault("chck_100M", {})["official_overall_full"] = full100.get("Overall") or full100.get("overall") or oo.get("Overall") or oo.get("overall")
        if fscores:
            scores["chck_100M"]["official_superglue_full"] = fscores.get("SuperGLUE")
            scores["chck_100M"]["official_aoa_full"] = fscores.get("AoA")
            if all(c in fscores for c in CHEAP_COLS):
                scores["chck_100M"]["cheap7"] = sum(float(fscores[c]) for c in CHEAP_COLS) / len(CHEAP_COLS)
    return scores


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    score_map = score_join()
    ckpts = [ck for ck in CHECKPOINTS if (MODEL_ROOT / ck / "model.safetensors").exists()]
    norms = []
    for ck in ckpts:
        rec = load_key_sums(ck)
        rec["scores"] = score_map.get(ck, {})
        norms.append(rec)
        print(json.dumps({"event": "norm_done", "checkpoint": ck, "adapter_rms": rec["adapter_rms"], "stock_rms": rec["stock_rms"]}), flush=True)
    deltas = []
    prev_delta = None
    prev_name = None
    for a, b in zip(ckpts[:-1], ckpts[1:]):
        d, prev_delta = pair_delta(a, b, prev_delta)
        deltas.append(d)
        print(json.dumps({"event": "delta_done", "from": a, "to": b, "delta_total_norm": d["delta_total_norm"], "cos_prev_total": d.get("cosine_with_previous_delta_total")}), flush=True)
        prev_name = b
    # Simple internal-signal reading in the measured cheap7 window.
    window = [ck for ck in [f"chck_{m}M" for m in range(77, 84)] if ck in score_map]
    cheap = {ck: score_map[ck].get("cheap7") for ck in window}
    vals = [float(cheap[ck]) for ck in window if cheap[ck] is not None]
    peak = max((ck for ck in window if cheap[ck] is not None), key=lambda ck: cheap[ck]) if vals else None
    local = {
        "window": window,
        "cheap7": cheap,
        "peak_by_cheap7": peak,
        "cheap7_mean": sum(vals) / len(vals) if vals else None,
        "cheap7_range": max(vals) - min(vals) if vals else None,
        "peak_minus_neighbor_mean": None,
    }
    if peak == "chck_82M":
        local["peak_minus_neighbor_mean"] = cheap[peak] - (cheap["chck_81M"] + cheap["chck_83M"]) / 2.0
    # Join per-checkpoint weight metrics to cheap7 window for hand inspection.
    norm_by = {r["checkpoint"]: r for r in norms}
    delta_to = {d["to"]: d for d in deltas}
    local["weight_signal_rows"] = []
    for ck in window:
        n = norm_by.get(ck, {})
        d = delta_to.get(ck, {})
        local["weight_signal_rows"].append({
            "checkpoint": ck,
            "cheap7": cheap.get(ck),
            "adapter_rms": n.get("adapter_rms"),
            "adapter_to_stock_norm_ratio": n.get("adapter_to_stock_norm_ratio"),
            "prev_delta_total_norm": d.get("delta_total_norm"),
            "prev_delta_stock_norm": d.get("delta_stock_norm"),
            "prev_delta_adapter_norm": d.get("delta_adapter_norm"),
            "prev_delta_adapter_norm_fraction": d.get("delta_adapter_norm_fraction"),
            "cosine_with_previous_delta_total": d.get("cosine_with_previous_delta_total"),
            "cosine_with_previous_delta_stock": d.get("cosine_with_previous_delta_stock"),
            "cosine_with_previous_delta_adapter": d.get("cosine_with_previous_delta_adapter"),
        })
    result = {
        "status": "SCALE1P75_PEAK_WEIGHT_GEOMETRY",
        "created_utc": now(),
        "purpose": "Test whether the reproduced 82M cheap7/Overall peak has a benchmark-independent weight-trajectory signature using existing checkpoints only.",
        "run_dir": rel(RUN),
        "checkpoints": ckpts,
        "norms": norms,
        "deltas": deltas,
        "score_map": score_map,
        "local_peak_summary": local,
        "scientific_reading": "This script reports internal weight geometry only. A useful early-stop signal would need to single out chck_82M, or at least warn against 83M-100M degradation, without using official evaluation labels. If the recorded weight rows are monotone/smooth while cheap7 oscillates, then current saved internal traces do not explain the peak.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research scale1.75 peak weight geometry", "", f"Status: `{result['status']}`", "", "## Cheap7 window", "", "| checkpoint | cheap7 | adapter_rms | adapter/stock norm | prev Δ total | prev Δ adapter frac | Δ-cos total | Δ-cos adapter |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in local["weight_signal_rows"]:
        lines.append("| {checkpoint} | {cheap7} | {adapter_rms:.8f} | {ratio:.6f} | {dtn} | {daf} | {ct} | {ca} |".format(
            checkpoint=r["checkpoint"],
            cheap7=r.get("cheap7"),
            adapter_rms=(r.get("adapter_rms") or float("nan")),
            ratio=(r.get("adapter_to_stock_norm_ratio") or float("nan")),
            dtn=("" if r.get("prev_delta_total_norm") is None else f"{r['prev_delta_total_norm']:.6f}"),
            daf=("" if r.get("prev_delta_adapter_norm_fraction") is None else f"{r['prev_delta_adapter_norm_fraction']:.6f}"),
            ct=("" if r.get("cosine_with_previous_delta_total") is None else f"{r['cosine_with_previous_delta_total']:.6f}"),
            ca=("" if r.get("cosine_with_previous_delta_adapter") is None else f"{r['cosine_with_previous_delta_adapter']:.6f}"),
        ))
    lines += ["", "## Local peak summary", "", "```json", json.dumps(local, indent=2), "```", "", result["scientific_reading"], "", f"JSON: `{rel(OUT_JSON)}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD), "peak": peak}, indent=2), flush=True)


if __name__ == "__main__":
    main()
