#!/usr/bin/env python3
"""research: CPU tensor-norm trajectory for DeBERTa adapter-scale runs.

Official scores, not tensor norms, decide competence. This script only checks whether
the train-time adapter scale contrast can be interpreted literally: if scale1.25 grows
larger adapter weights so that effective scaled adapter norms match scale1.75, then
config scale alone is not the mechanism. If effective norms differ, score trajectories
can be read against a real residual-energy difference.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
from statistics import mean
from typing import Any

from safetensors.torch import safe_open

DEFAULT_ENDPOINTS = [f"chck_{m}M" for m in range(70, 101, 2)]


def parse_labeled_run(text: str) -> tuple[str, pathlib.Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use LABEL=RUN_DIR")
    label, path = text.split("=", 1)
    return label, pathlib.Path(path)


def read_scale(ckpt: pathlib.Path) -> float | None:
    cfg = ckpt / "config.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return None
    val = data.get("adapter_scale")
    return float(val) if val is not None else None


def tensor_stats(path: pathlib.Path) -> dict[str, Any]:
    adapter_sq = 0.0
    stock_sq = 0.0
    adapter_count = 0
    stock_count = 0
    up_sq = 0.0
    down_sq = 0.0
    ln_adapter_sq = 0.0
    up_count = 0
    down_count = 0
    layer_up_sq: dict[str, float] = {}
    layer_down_sq: dict[str, float] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            sq = float(t.float().pow(2).sum().item())
            n = int(t.numel())
            if ".adapter." in k:
                adapter_sq += sq
                adapter_count += n
                m = re.search(r"layer\.(\d+)\.adapter\.", k)
                layer = m.group(1) if m else "unknown"
                if ".adapter.up." in k:
                    up_sq += sq
                    up_count += n
                    layer_up_sq[layer] = layer_up_sq.get(layer, 0.0) + sq
                elif ".adapter.down." in k:
                    down_sq += sq
                    down_count += n
                    layer_down_sq[layer] = layer_down_sq.get(layer, 0.0) + sq
                else:
                    ln_adapter_sq += sq
            else:
                stock_sq += sq
                stock_count += n
    return {
        "adapter_l2": math.sqrt(adapter_sq),
        "stock_l2": math.sqrt(stock_sq),
        "adapter_rms": math.sqrt(adapter_sq / adapter_count) if adapter_count else None,
        "stock_rms": math.sqrt(stock_sq / stock_count) if stock_count else None,
        "adapter_to_stock_l2_ratio": math.sqrt(adapter_sq) / math.sqrt(stock_sq) if stock_sq > 0 else None,
        "up_l2": math.sqrt(up_sq),
        "down_l2": math.sqrt(down_sq),
        "adapter_layernorm_l2": math.sqrt(ln_adapter_sq),
        "up_rms": math.sqrt(up_sq / up_count) if up_count else None,
        "down_rms": math.sqrt(down_sq / down_count) if down_count else None,
        "mean_layer_up_l2": mean(math.sqrt(v) for v in layer_up_sq.values()) if layer_up_sq else None,
        "mean_layer_down_l2": mean(math.sqrt(v) for v in layer_down_sq.values()) if layer_down_sq else None,
        "n_adapter_params": adapter_count,
        "n_stock_params": stock_count,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", type=parse_labeled_run, required=True, help="LABEL=RUN_DIR")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--endpoints", nargs="+", default=DEFAULT_ENDPOINTS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "status": "ADAPTER_NORM_TRAJECTORY",
        "endpoints": args.endpoints,
        "runs": {},
        "interpretation_note": "Tensor norms are only mechanism-supporting context. Competence conclusions require official-compatible selected trajectory scores.",
    }
    for label, run_dir in args.run:
        rows = []
        for ep in args.endpoints:
            ckpt = run_dir / "hf_model" / ep
            model = ckpt / "model.safetensors"
            row: dict[str, Any] = {"endpoint": ep, "words": int(ep[len("chck_"):-1]) * 1_000_000, "exists": model.exists()}
            if model.exists():
                scale = read_scale(ckpt)
                row["adapter_scale"] = scale
                stats = tensor_stats(model)
                row.update(stats)
                if scale is not None:
                    row["effective_up_l2_scale_times_up"] = float(scale * stats["up_l2"])
                    row["effective_adapter_l2_scale_times_adapter"] = float(scale * stats["adapter_l2"])
                    row["effective_up_to_stock_ratio"] = float(scale * stats["up_l2"] / stats["stock_l2"])
            rows.append(row)
        valid = [r for r in rows if r.get("exists")]
        result["runs"][label] = {
            "run_dir": str(run_dir),
            "n_valid": len(valid),
            "rows": rows,
            "summary": {
                "first_effective_up_to_stock_ratio": valid[0].get("effective_up_to_stock_ratio") if valid else None,
                "last_effective_up_to_stock_ratio": valid[-1].get("effective_up_to_stock_ratio") if valid else None,
                "max_effective_up_to_stock_ratio": max((r.get("effective_up_to_stock_ratio") for r in valid if r.get("effective_up_to_stock_ratio") is not None), default=None),
                "mean_effective_up_to_stock_ratio": mean([r["effective_up_to_stock_ratio"] for r in valid if r.get("effective_up_to_stock_ratio") is not None]) if valid else None,
            },
        }
    # Compare same endpoints between scale1.25 and reference if labels present.
    labels = list(result["runs"])
    comparisons = {}
    if "scale1p75_seed43022_reference" in result["runs"] and "scale1p25_seed43022_dense" in result["runs"]:
        ref = {r["endpoint"]: r for r in result["runs"]["scale1p75_seed43022_reference"]["rows"] if r.get("exists")}
        alt = {r["endpoint"]: r for r in result["runs"]["scale1p25_seed43022_dense"]["rows"] if r.get("exists")}
        deltas = []
        for ep in args.endpoints:
            if ep not in ref or ep not in alt:
                continue
            r = ref[ep]; a = alt[ep]
            deltas.append({
                "endpoint": ep,
                "words": a["words"],
                "scale1p25_minus_ref_up_l2": a["up_l2"] - r["up_l2"],
                "scale1p25_minus_ref_effective_up_l2": a["effective_up_l2_scale_times_up"] - r["effective_up_l2_scale_times_up"],
                "scale1p25_div_ref_effective_up_l2": a["effective_up_l2_scale_times_up"] / r["effective_up_l2_scale_times_up"] if r["effective_up_l2_scale_times_up"] else None,
                "scale1p25_minus_ref_effective_up_to_stock_ratio": a["effective_up_to_stock_ratio"] - r["effective_up_to_stock_ratio"],
            })
        comparisons["scale1p25_vs_scale1p75_seed43022_reference"] = deltas
    result["comparisons"] = comparisons

    out_json = args.out_dir / "adapter_norm_trajectory.json"
    out_md = args.out_dir / "adapter_norm_trajectory.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = ["# research adapter norm trajectory\n\n"]
    for label, rec in result["runs"].items():
        md.append(f"## {label}\n\n")
        md.append(f"Valid checkpoints: {rec['n_valid']} / {len(args.endpoints)}. Mean effective up/stock ratio: {rec['summary'].get('mean_effective_up_to_stock_ratio')}.\n\n")
        md.append("| endpoint | scale | up_l2 | scale*up_l2 | effective up/stock | adapter/stock |\n")
        md.append("|---|---:|---:|---:|---:|---:|\n")
        for row in rec["rows"]:
            if not row.get("exists"):
                md.append(f"| {row['endpoint']} | — | — | — | — | — |\n")
            else:
                md.append(f"| {row['endpoint']} | {row.get('adapter_scale'):.2f} | {row.get('up_l2'):.4f} | {row.get('effective_up_l2_scale_times_up'):.4f} | {row.get('effective_up_to_stock_ratio'):.6f} | {row.get('adapter_to_stock_l2_ratio'):.6f} |\n")
        md.append("\n")
    if comparisons.get("scale1p25_vs_scale1p75_seed43022_reference"):
        md.append("## scale1.25 vs seed/mask-matched scale1.75 reference\n\n")
        md.append("| endpoint | Δ(scale*up_l2) | ratio scale1.25/ref | Δeffective up/stock |\n")
        md.append("|---|---:|---:|---:|\n")
        for row in comparisons["scale1p25_vs_scale1p75_seed43022_reference"]:
            md.append(f"| {row['endpoint']} | {row['scale1p25_minus_ref_effective_up_l2']:+.4f} | {row['scale1p25_div_ref_effective_up_l2']:.4f} | {row['scale1p25_minus_ref_effective_up_to_stock_ratio']:+.6f} |\n")
    md.append("\nOfficial-compatible selected scores must decide the route; this file only records residual-energy context.\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
