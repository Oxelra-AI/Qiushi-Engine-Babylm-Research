#!/usr/bin/env python3
"""research: low-cost same-initialization weight-space sweep for FW compact/breadth endpoints.

Scientific purpose
------------------
The FW compact, row-block breadth, and interleaved breadth endpoints share the same
architecture, tokenizer, seeds, and full-batch training coordinate.  The failed
cross-seed soup from research does not answer whether same-initialization branches
trained on different but aligned data views occupy a compatible basin.  This script
materializes a tiny set of single-model linear combinations with exact provenance,
then later official-compatible readouts can decide whether compact broad capability
and breadth relation movement can coexist in one checkpoint.

No training is performed.  The script only reads completed checkpoints, averages
matching tensors, copies the already-compliant tokenizer/config files, and writes
local HF checkpoint directories under training/runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(".").resolve()
WS = ROOT / "experiments/archive" / 'representation_and_objectives'
A02_WS = ROOT / "experiments/archive" / 'frontier_consolidation'
OUT_RUN_ROOT = WS / "training" / "runs" / "fw_weight_space_sweep"
OUT_DATA = WS / "data" / "fw_weight_space_sweep"
NOTE = (ROOT / 'research/notes/representation_and_objectives/fw_weight_space_sweep.md')

BASES: dict[str, dict[str, Any]] = {
    "compact": {
        "label": "A02 compact same-proposition recurrence anchor",
        "model_path": A02_WS / "training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_100M",
        "source_run": A02_WS / "training/runs/fw_compact_view_shared16k_seed43022",
        "stream_sha256": "c8d7f24b5edd2dad21f589c8cde72671d0b76038d9632fcd3d6c79178a24be68",
        "cheap7": 43.18142857142857,
        "scores": {"Supplement": 58.86, "EWoK": 50.25, "Entity": 28.36, "GlobalPIQA": 38.635, "GlobalPIQA_parallel": 24.27, "GlobalPIQA_nonparallel": 53.0},
    },
    "interleaved": {
        "label": "A01 interleaved whole-sentence independent-breadth arm",
        "model_path": WS / "training/runs/fw_source_breadth_interleaved_wholesentence_fullbatch_shared16k_seed43022/hf_model/chck_100M",
        "source_run": WS / "training/runs/fw_source_breadth_interleaved_wholesentence_fullbatch_shared16k_seed43022",
        "stream_sha256": "1758508e3bfae8bda8d4d66409ebd695499552a495e93d8327be8aa0d9182c44",
        "cheap7": 43.07928571428572,
        "scores": {"Supplement": 56.42, "EWoK": 51.85, "Entity": 25.60, "GlobalPIQA": 41.105, "GlobalPIQA_parallel": 26.21, "GlobalPIQA_nonparallel": 56.0},
    },
    "rowblock": {
        "label": "A02 row-block whole-sentence independent-breadth arm",
        "model_path": A02_WS / "training/runs/fw_source_breadth_shared16k_seed43022/hf_model/chck_100M",
        "source_run": A02_WS / "training/runs/fw_source_breadth_shared16k_seed43022",
        "stream_sha256": "1b98269fb210cc9494885ec47ee26d9fd1b6ead60a1c308f5d9c47d9b385731d",
        "cheap7": 42.63857142857143,
        "scores": {"Supplement": 59.69, "EWoK": 50.50, "Entity": 23.88, "GlobalPIQA": 37.065, "GlobalPIQA_parallel": 29.13, "GlobalPIQA_nonparallel": 45.0},
    },
}

# Minimal predetermined sweep.  The first three test compact->interleaved.  The
# last two inject only a small row-block component because row-block carried the
# strongest hard-GlobalPIQA_parallel movement but also the largest broad damage.
RECIPES: dict[str, dict[str, float]] = {
    "ci_a0p25": {"compact": 0.75, "interleaved": 0.25},
    "ci_a0p50": {"compact": 0.50, "interleaved": 0.50},
    "ci_a0p75": {"compact": 0.25, "interleaved": 0.75},
    "cr_a0p25": {"compact": 0.75, "rowblock": 0.25},
    "cir_i0p25_r0p25": {"compact": 0.50, "interleaved": 0.25, "rowblock": 0.25},
}

SIDE_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_file(path: Path) -> Path:
    p = path / "model.safetensors"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def safetensor_keys(path: Path) -> list[str]:
    with safe_open(model_file(path), framework="pt", device="cpu") as f:
        return list(f.keys())


def file_records(model_path: Path) -> dict[str, Any]:
    out = {"model_path": str(model_path), "exists": model_path.exists(), "files": {}}
    for name in SIDE_FILES + ["model.safetensors"]:
        p = model_path / name
        out["files"][name] = {"exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None, "sha256": sha256_file(p) if p.exists() else None}
    return out


def verify_alignment() -> dict[str, Any]:
    rec: dict[str, Any] = {"status": "OK", "bases": {}, "errors": []}
    key_ref = None
    side_sha_ref: dict[str, str] | None = None
    for name, spec in BASES.items():
        mp = Path(spec["model_path"])
        frec = file_records(mp)
        rec["bases"][name] = {**frec, "label": spec["label"], "stream_sha256": spec["stream_sha256"], "cheap7": spec["cheap7"], "scores": spec["scores"]}
        if not mp.exists():
            rec["errors"].append(f"missing model path for {name}: {mp}")
            continue
        keys = safetensor_keys(mp)
        rec["bases"][name]["n_tensor_keys"] = len(keys)
        rec["bases"][name]["first_keys"] = keys[:5]
        rec["bases"][name]["last_keys"] = keys[-5:]
        side = {sf: frec["files"][sf]["sha256"] for sf in SIDE_FILES}
        if key_ref is None:
            key_ref = keys
            side_sha_ref = side
        else:
            if keys != key_ref:
                rec["errors"].append(f"tensor key mismatch for {name}")
            if side != side_sha_ref:
                rec["errors"].append(f"config/tokenizer side-file SHA mismatch for {name}: {side} vs {side_sha_ref}")
    if rec["errors"]:
        rec["status"] = "FAILED"
    return rec


def safe_float(x: torch.Tensor) -> float:
    v = float(x.item())
    if not math.isfinite(v):
        raise ValueError(v)
    return v


def compute_geometry(keys: list[str]) -> dict[str, Any]:
    """Compute endpoint and delta geometry with all three 138 MB states resident once."""
    states = {name: load_all_state(Path(spec["model_path"])) for name, spec in BASES.items()}
    totals = {
        "norm_compact_sq": 0.0,
        "norm_interleaved_sq": 0.0,
        "norm_rowblock_sq": 0.0,
        "delta_ci_sq": 0.0,
        "delta_cr_sq": 0.0,
        "delta_ir_sq": 0.0,
        "dot_ci_cr": 0.0,
        "dot_c_i": 0.0,
        "dot_c_r": 0.0,
        "dot_i_r": 0.0,
        "num_parameters": 0,
    }
    groups: dict[str, dict[str, float]] = {}

    def group_for_key(k: str) -> str:
        if "embeddings" in k:
            return "embeddings"
        if "encoder.layer" in k:
            parts = k.split(".")
            try:
                idx = parts.index("layer")
                return f"layer_{parts[idx+1]}"
            except Exception:
                return "encoder_other"
        if "lm_predictions" in k or "cls" in k:
            return "mlm_head"
        return "other"

    for key in keys:
        c = states["compact"][key].float()
        i = states["interleaved"][key].float()
        r = states["rowblock"][key].float()
        dci = i - c
        dcr = r - c
        dir_ = r - i
        n = c.numel()
        vals = {
            "norm_compact_sq": torch.sum(c*c),
            "norm_interleaved_sq": torch.sum(i*i),
            "norm_rowblock_sq": torch.sum(r*r),
            "delta_ci_sq": torch.sum(dci*dci),
            "delta_cr_sq": torch.sum(dcr*dcr),
            "delta_ir_sq": torch.sum(dir_*dir_),
            "dot_ci_cr": torch.sum(dci*dcr),
            "dot_c_i": torch.sum(c*i),
            "dot_c_r": torch.sum(c*r),
            "dot_i_r": torch.sum(i*r),
        }
        for kk, vv in vals.items():
            totals[kk] += safe_float(vv.double())
        totals["num_parameters"] += n
        g = group_for_key(key)
        gd = groups.setdefault(g, {"num_parameters": 0, "delta_ci_sq": 0.0, "delta_cr_sq": 0.0, "dot_ci_cr": 0.0})
        gd["num_parameters"] += n
        gd["delta_ci_sq"] += safe_float(vals["delta_ci_sq"].double())
        gd["delta_cr_sq"] += safe_float(vals["delta_cr_sq"].double())
        gd["dot_ci_cr"] += safe_float(vals["dot_ci_cr"].double())
        del c, i, r, dci, dcr, dir_
    eps = 1e-30
    totals["norm_compact"] = math.sqrt(totals["norm_compact_sq"])
    totals["norm_interleaved"] = math.sqrt(totals["norm_interleaved_sq"])
    totals["norm_rowblock"] = math.sqrt(totals["norm_rowblock_sq"])
    totals["delta_ci_norm"] = math.sqrt(totals["delta_ci_sq"])
    totals["delta_cr_norm"] = math.sqrt(totals["delta_cr_sq"])
    totals["delta_ir_norm"] = math.sqrt(totals["delta_ir_sq"])
    totals["cos_delta_ci_cr"] = totals["dot_ci_cr"] / max(math.sqrt(totals["delta_ci_sq"] * totals["delta_cr_sq"]), eps)
    totals["relative_delta_ci_to_compact"] = totals["delta_ci_norm"] / max(totals["norm_compact"], eps)
    totals["relative_delta_cr_to_compact"] = totals["delta_cr_norm"] / max(totals["norm_compact"], eps)
    for gd in groups.values():
        gd["delta_ci_norm"] = math.sqrt(gd["delta_ci_sq"])
        gd["delta_cr_norm"] = math.sqrt(gd["delta_cr_sq"])
        gd["cos_delta_ci_cr"] = gd["dot_ci_cr"] / max(math.sqrt(gd["delta_ci_sq"] * gd["delta_cr_sq"]), eps)
    return {"global": totals, "groups": dict(sorted(groups.items()))}


def load_all_state(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(model_file(path)), device="cpu")


def materialize_recipe(name: str, coeffs: dict[str, float], force: bool) -> dict[str, Any]:
    s = sum(coeffs.values())
    if abs(s - 1.0) > 1e-9:
        raise ValueError({"recipe": name, "coeffs_sum": s})
    out_dir = OUT_RUN_ROOT / name / "hf_model" / "chck_100M"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / "model.safetensors"
    if out_model.exists() and not force:
        return {"name": name, "status": "exists", "coefficients": coeffs, "model_path": str(out_dir), "files": file_records(out_dir)}

    states = {arm: load_all_state(Path(BASES[arm]["model_path"])) for arm in coeffs}
    keys = list(next(iter(states.values())).keys())
    out_state: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key in keys:
            acc = None
            for arm, w in coeffs.items():
                t = states[arm][key]
                if not torch.is_floating_point(t):
                    # These checkpoints should contain only floating tensors, but keep a
                    # deterministic path if an integer buffer appears.
                    if w == 1.0:
                        acc = t.clone()
                    else:
                        raise TypeError({"nonfloating_tensor_in_mixture": key, "arm": arm, "dtype": str(t.dtype)})
                else:
                    part = t.float().mul(float(w))
                    acc = part if acc is None else acc.add(part)
            ref_dtype = states[next(iter(coeffs))][key].dtype
            out_state[key] = acc.to(dtype=ref_dtype).contiguous()  # type: ignore[union-attr]
    save_file(out_state, str(out_model), metadata={"format": "pt", "step": "116", "recipe": name})
    # Copy tokenizer/config files from compact; all side files were SHA-verified equal.
    compact_path = Path(BASES["compact"]["model_path"])
    for sf in SIDE_FILES:
        shutil.copy2(compact_path / sf, out_dir / sf)
    # Parent hf_model files: mirror checkpoint files so evaluator can load either root or chck_100M.
    parent = OUT_RUN_ROOT / name / "hf_model"
    for sf in SIDE_FILES + ["model.safetensors"]:
        shutil.copy2(out_dir / sf, parent / sf)

    meta = {
        "status": "created",
        "name": name,
        "coefficients": coeffs,
        "created_utc": now_utc(),
        "model_path": str(out_dir),
        "source_models": {arm: str(BASES[arm]["model_path"]) for arm in coeffs},
        "source_stream_sha256": {arm: BASES[arm]["stream_sha256"] for arm in coeffs},
        "source_scores": {arm: BASES[arm]["scores"] for arm in coeffs},
        "files": file_records(out_dir),
        "boundary": "No training or data selection: linear parameter combination of same-tokenizer/same-architecture/same-seed completed checkpoints.",
    }
    (OUT_RUN_ROOT / name / "mixture_manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def write_note(manifest: dict[str, Any]) -> None:
    lines = []
    lines.append("# research — FW same-initialization weight-space sweep")
    lines.append("")
    lines.append("This is a low-cost branch-consolidation test, not a new 100M training branch. The scientific question is whether compact same-proposition recurrence and independent-breadth relation movement occupy compatible directions in one aligned checkpoint.")
    lines.append("")
    geom = manifest.get("geometry", {}).get("global", {})
    if geom:
        lines.append("## Endpoint geometry")
        lines.append("")
        lines.append(f"- Parameters compared: {geom.get('num_parameters')}")
        lines.append(f"- ||interleaved - compact|| / ||compact|| = {geom.get('relative_delta_ci_to_compact'):.6f}")
        lines.append(f"- ||rowblock - compact|| / ||compact|| = {geom.get('relative_delta_cr_to_compact'):.6f}")
        lines.append(f"- cosine(interleaved-compact, rowblock-compact) = {geom.get('cos_delta_ci_cr'):.6f}")
        lines.append("")
    lines.append("## Materialized recipes")
    lines.append("")
    for name, rec in manifest.get("recipes", {}).items():
        coeff = ", ".join(f"{k}={v}" for k, v in rec.get("coefficients", {}).items())
        lines.append(f"- `{name}`: {coeff}; status={rec.get('status')}; checkpoint=`{rec.get('model_path')}`")
    lines.append("")
    lines.append("## Readout plan")
    lines.append("")
    lines.append("First read Supplement, Entity, current 7,618-row EWoK, and GlobalPIQA parallel/nonparallel. A useful branch-consolidation signal would preserve compact-like Supplement/Entity while moving EWoK and GlobalPIQA toward the breadth arms. If the mixture only traces the compact↔breadth tradeoff, close this line and move to a changed relational objective or representation rather than more FW allocation variants.")
    lines.append("")
    lines.append("Files:")
    lines.append(f"- manifest: `{OUT_DATA / 'weight_space_sweep_manifest.json'}`")
    lines.append(f"- run root: `{OUT_RUN_ROOT}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", nargs="*", default=list(RECIPES), choices=sorted(RECIPES))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--geometry-only", action="store_true")
    args = ap.parse_args()

    OUT_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    align = verify_alignment()
    if align["status"] != "OK":
        (OUT_DATA / "weight_space_sweep_manifest.json").write_text(json.dumps({"status": "FAILED_ALIGNMENT", "alignment": align}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise RuntimeError(align)
    keys = safetensor_keys(Path(BASES["compact"]["model_path"]))
    geometry = compute_geometry(keys)
    recipes = {}
    if not args.geometry_only:
        for r in args.recipes:
            recipes[r] = materialize_recipe(r, RECIPES[r], force=args.force)
    manifest = {
        "status": "FW_WEIGHT_SPACE_SWEEP_READY",
        "created_utc": now_utc(),
        "alignment": align,
        "geometry": geometry,
        "recipes": recipes,
        "readout_boundary": "Evaluate only a predetermined small sweep. Do not choose a fine-grained lambda on official rows; require broad preservation plus relation movement before any full official evaluation.",
    }
    out = OUT_DATA / "weight_space_sweep_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(manifest)
    print(json.dumps({
        "status": manifest["status"],
        "recipes": list(recipes),
        "cos_delta_ci_cr": geometry["global"].get("cos_delta_ci_cr"),
        "relative_delta_ci_to_compact": geometry["global"].get("relative_delta_ci_to_compact"),
        "relative_delta_cr_to_compact": geometry["global"].get("relative_delta_cr_to_compact"),
        "manifest": str(out),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
