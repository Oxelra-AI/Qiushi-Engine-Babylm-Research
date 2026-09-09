#!/usr/bin/env python3
"""CPU-only private-adapter geometry for research continuation checkpoints."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
PARENT = _public_path('models/frontier')
STANDARD_RUN = _public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_standard_seed43023/hf_model')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/adapter_geometry')


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def layer_key(k: str) -> str:
    parts = k.split(".")
    if "layer" in parts:
        i = parts.index("layer")
        if i + 1 < len(parts):
            return f"L{parts[i+1]}"
    return "other"


def flatten(sd: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    if not keys:
        return torch.zeros(1)
    return torch.cat([sd[k].float().reshape(-1).cpu() for k in keys])


def cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na == 0.0 or nb == 0.0:
        return None
    return float(torch.dot(a, b).item() / (na * nb))


def parse_exposure(path: Path) -> int | None:
    name = path.name
    if name == "final":
        # Read run metrics if nearby.
        m = path.parents[1] / "scientific_metrics.json"
        if m.exists():
            try:
                return int(json.loads(m.read_text()).get("total_consumed_words"))
            except Exception:
                return None
        return None
    if name.startswith("chck_total_") and name.endswith("w"):
        return int(name[len("chck_total_"):-1])
    if name.startswith("chck_") and name.endswith("M"):
        return int(name[len("chck_"):-1]) * 1_000_000
    return None


def stats(parent_sd: dict[str, torch.Tensor], child_path: Path) -> dict[str, Any]:
    child_sd = load_file(str(child_path / "model.safetensors"), device="cpu")
    pkeys = sorted(k for k in parent_sd if ".private_adapter." in k)
    ckeys = sorted(k for k in child_sd if ".private_adapter." in k)
    if pkeys != ckeys:
        raise RuntimeError(f"private keys mismatch for {child_path}: parent {len(pkeys)} child {len(ckeys)}")
    out: dict[str, Any] = {"path": rel(child_path), "exposure_words": parse_exposure(child_path), "layers": {}}
    def one(keys: list[str]) -> dict[str, Any]:
        p = flatten(parent_sd, keys)
        c = flatten(child_sd, keys)
        d = c - p
        pn = float(p.norm().item())
        cn = float(c.norm().item())
        dn = float(d.norm().item())
        return {
            "parent_norm": pn,
            "child_norm": cn,
            "delta_norm": dn,
            "relative_delta_to_parent": dn / max(1e-12, pn),
            "cos_child_parent": cos(c, p),
            "cos_delta_parent": cos(d, p),
            "cos_delta_child": cos(d, c),
        }
    out["all"] = one(pkeys)
    by_layer: dict[str, list[str]] = {}
    by_part: dict[str, list[str]] = {"down": [], "up": [], "layer_norm": []}
    for k in pkeys:
        by_layer.setdefault(layer_key(k), []).append(k)
        if ".down." in k:
            by_part["down"].append(k)
        elif ".up." in k:
            by_part["up"].append(k)
        elif ".layer_norm." in k:
            by_part["layer_norm"].append(k)
    out["layers"] = {lk: one(keys) for lk, keys in sorted(by_layer.items())}
    out["parts"] = {pk: one(keys) for pk, keys in by_part.items() if keys}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default=str(PARENT))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--model", action="append", default=None, help="tag=path; repeatable")
    args = ap.parse_args()
    parent_path = Path(args.parent)
    if args.model:
        models = []
        for spec in args.model:
            tag, path = spec.split("=", 1)
            models.append((tag, Path(path)))
    else:
        names = ["chck_total_87005295w", "chck_total_90005295w", "chck_total_94005295w", "chck_total_98005295w", "final"]
        models = [(f"std_{n}", STANDARD_RUN / n) for n in names]
    parent_sd = load_file(str(parent_path / "model.safetensors"), device="cpu")
    rec = {"status": "ADAPTER_GEOMETRY_COMPLETE", "parent": rel(parent_path), "models": {}}
    for tag, path in models:
        if not path.exists():
            rec["models"][tag] = {"path": rel(path), "status": "missing"}
        else:
            rec["models"][tag] = {"status": "ok", **stats(parent_sd, path)}
            print(json.dumps({"tag": tag, "exposure": rec["models"][tag].get("exposure_words"), **rec["models"][tag]["all"]}), flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "adapter_geometry.json"
    out_json.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# research private-adapter geometry", "", f"Parent: `{rel(parent_path)}`", "", "| tag | exposure | rel Δ/all | Δ norm | cos(Δ,parent) | cos(child,parent) |", "|---|---:|---:|---:|---:|---:|"]
    for tag, m in rec["models"].items():
        if m.get("status") != "ok":
            lines.append(f"| {tag} |  |  |  |  |  |")
            continue
        a = m["all"]
        lines.append(f"| {tag} | {m.get('exposure_words')} | {a['relative_delta_to_parent']:.4f} | {a['delta_norm']:.4g} | {a['cos_delta_parent'] if a['cos_delta_parent'] is not None else ''} | {a['cos_child_parent'] if a['cos_child_parent'] is not None else ''} |")
    out_md = out_dir / "adapter_geometry.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": rec["status"], "out_json": rel(out_json), "out_md": rel(out_md)}), flush=True)


if __name__ == "__main__":
    main()
