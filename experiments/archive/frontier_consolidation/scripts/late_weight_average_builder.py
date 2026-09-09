#!/usr/bin/env python3
"""research: prepare same-trajectory late-weight-average candidates.

This script is deliberately CPU/file-only.  It does not evaluate official tasks,
submit to the leaderboard, or average across independent random initializations.
It creates reproducible candidate checkpoint directories for later selected
cheap-task evaluation if the seed43122 common-grid evidence makes competence
stabilization the right next route.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep HF dynamic modules/cache writable if optional validation loads a model.
def find_user_root() -> Path:
    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        if (p / "experiments/archive" / 'frontier_consolidation').exists():
            return p
    raise RuntimeError(f"Could not locate user root from cwd={cur}")


ROOT = find_user_root()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
OUT_ROOT = WORKSPACE / "data" / "late_weight_average_scaffold"
HF_CACHE = OUT_ROOT / "runtime_cache"
for key, sub in {
    "HF_HOME": "home",
    "HF_HUB_CACHE": "hub",
    "HUGGINGFACE_HUB_CACHE": "hub",
    "HF_DATASETS_CACHE": "datasets",
    "TRANSFORMERS_CACHE": "transformers",
    "XDG_CACHE_HOME": "xdg",
    "HF_MODULES_CACHE": "modules",
}.items():
    p = HF_CACHE / sub
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p.resolve())
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

STATIC_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "adapter_scaled_modeling.py",
]
EXPECTED_STATIC = {
    "tokenizer.json": "a9cbb830495cb92bbb2996adc256207746282ee67f4f8d40a4f646a634ec139a",
    "tokenizer_config.json": "488d9617b98746cfdb862c62e02c2935bb086a2d0620307e75b9bac967d26d01",
    "special_tokens_map.json": "2be97b602cd0e4a6c2874cbf03c0d1025b3af5474666da931bc04f6c23a9a39d",
    "adapter_scaled_modeling.py": "9fd4ef104f5f8e7d458c7ed10b996d238a6af53a866e8dea907a3cc0e4e156d2",
}
EXPECTED_PARAMS = 35_463_008

TRAJECTORIES = {
    "reference_scale1p75_seed43022": {
        "run_dir": WORKSPACE / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model",
        "status": "already scored on selected 70M-100M grid; chck_84M is the known local cheap7 peak",
        "same_seed_mask_as_reference": True,
        "adapter_scale": 1.75,
    },
    "scale1p25_seed43022": {
        "run_dir": WORKSPACE / "training/runs/adapter128_scale1p25_seed43022_dense100M/hf_model",
        "status": "same seed/mask as reference; lower residual-energy trajectory; selected grid partly pending/resumed",
        "same_seed_mask_as_reference": True,
        "adapter_scale": 1.25,
    },
    "scale1p75_seed43122": {
        "run_dir": WORKSPACE / "training/runs/adapter128_scale1p75_seed43122_dense100M/hf_model",
        "status": "different init plus train/mask RNG; selected grid running as seed43122 robustness test",
        "same_seed_mask_as_reference": False,
        "adapter_scale": 1.75,
    },
}

# Same-trajectory only.  Never average across labels unless a future script proves
# functional alignment, because independent random initializations are not weight
# aligned.
RECIPE_TEMPLATES = {
    "prepeak_78_80_82_84_uniform": {
        "endpoints": ["chck_78M", "chck_80M", "chck_82M", "chck_84M"],
        "weights": [1.0, 1.0, 1.0, 1.0],
        "reading": "low-pass rising late iterates up to the 84M reference peak; excludes immediate 86M decline",
    },
    "center_80_82_84_uniform": {
        "endpoints": ["chck_80M", "chck_82M", "chck_84M"],
        "weights": [1.0, 1.0, 1.0],
        "reading": "short latest-window average ending at the 84M peak; closest LAWA-style candidate for reference trajectory",
    },
    "symmetric_82_84_86_uniform": {
        "endpoints": ["chck_82M", "chck_84M", "chck_86M"],
        "weights": [1.0, 1.0, 1.0],
        "reading": "tests whether the 84M peak sits in a local basin or whether 86M damage should be excluded",
    },
    "wider_76_78_80_82_84_uniform": {
        "endpoints": ["chck_76M", "chck_78M", "chck_80M", "chck_82M", "chck_84M"],
        "weights": [1.0, 1.0, 1.0, 1.0, 1.0],
        "reading": "broader rising-side low-pass; may preserve earlier BLiMP/COMPS while smoothing relation-state churn",
    },
    "ema_decay0p5_76_to_84": {
        "endpoints": ["chck_76M", "chck_78M", "chck_80M", "chck_82M", "chck_84M"],
        "weights": [0.0625, 0.125, 0.25, 0.5, 1.0],
        "reading": "post-hoc EMA-like rising-side average, most weight on 84M while retaining preceding iterate memory",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


def normalize(weights: list[float]) -> list[float]:
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("weights must have positive sum")
    return [float(w) / s for w in weights]


def recipe_key(label: str, template: str) -> str:
    return f"{label}__{template}"


def endpoint_dir(label: str, endpoint: str) -> Path:
    return TRAJECTORIES[label]["run_dir"] / endpoint


def inspect_recipe(label: str, template: str, hash_models: bool = False) -> dict[str, Any]:
    spec = RECIPE_TEMPLATES[template]
    eps = spec["endpoints"]
    static: dict[str, Any] = {}
    for sf in STATIC_FILES:
        vals = []
        for ep in eps:
            p = endpoint_dir(label, ep) / sf
            vals.append({
                "endpoint": ep,
                "exists": p.exists(),
                "sha256": sha256_file(p) if p.exists() else None,
                "size_bytes": p.stat().st_size if p.exists() else None,
            })
        static[sf] = {
            "values": vals,
            "n_unique_sha256": len({v["sha256"] for v in vals if v["sha256"]}),
            "expected_sha256": EXPECTED_STATIC.get(sf),
            "matches_expected_all": all((v["sha256"] == EXPECTED_STATIC.get(sf)) for v in vals) if sf in EXPECTED_STATIC else None,
        }
    weights = normalize(list(spec["weights"]))
    model_files = []
    for ep in eps:
        p = endpoint_dir(label, ep) / "model.safetensors"
        model_files.append({
            "endpoint": ep,
            "path": rel(p),
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else None,
            "sha256": sha256_file(p) if (hash_models and p.exists()) else None,
        })
    return {
        "candidate": recipe_key(label, template),
        "trajectory_label": label,
        "template": template,
        "trajectory_status": TRAJECTORIES[label]["status"],
        "adapter_scale": TRAJECTORIES[label]["adapter_scale"],
        "same_seed_mask_as_reference": TRAJECTORIES[label]["same_seed_mask_as_reference"],
        "endpoints": eps,
        "raw_weights": spec["weights"],
        "normalized_weights": weights,
        "reading": spec["reading"],
        "model_files": model_files,
        "static_files": static,
        "ok_to_build": all(m["exists"] for m in model_files) and all(v["n_unique_sha256"] == 1 for v in static.values()),
        "scientific_boundary": "same-trajectory late-iterate average only; no official evaluation or leaderboard submission is performed by this script",
    }


def copy_static(src_dir: Path, dst_dir: Path) -> dict[str, Any]:
    copied = {}
    for sf in STATIC_FILES:
        src = src_dir / sf
        dst = dst_dir / sf
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied[sf] = {"src": rel(src), "dst": rel(dst), "sha256": sha256_file(dst), "size_bytes": dst.stat().st_size}
    return copied


def build_candidate(label: str, template: str, overwrite: bool = False, validate_load: bool = True) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file, save_file
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Missing torch/safetensors dependency: {type(e).__name__}: {e}") from e

    plan = inspect_recipe(label, template, hash_models=True)
    if not plan["ok_to_build"]:
        raise RuntimeError(f"Recipe is not buildable: {plan}")
    out_dir = OUT_ROOT / "candidates" / recipe_key(label, template)
    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"Candidate exists: {out_dir}; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = plan["normalized_weights"]
    endpoints = plan["endpoints"]
    accum: dict[str, torch.Tensor] = {}
    dtype_record: dict[str, str] = {}
    shape_record: dict[str, list[int]] = {}
    key_order: list[str] | None = None
    for idx, (ep, w) in enumerate(zip(endpoints, weights)):
        mf = endpoint_dir(label, ep) / "model.safetensors"
        state = load_file(str(mf), device="cpu")
        keys = sorted(state.keys())
        if key_order is None:
            key_order = keys
        elif keys != key_order:
            raise RuntimeError(f"Key mismatch for {ep}")
        for k in keys:
            t = state[k]
            if idx == 0:
                dtype_record[k] = str(t.dtype)
                shape_record[k] = list(t.shape)
                accum[k] = t.detach().to(dtype=torch.float32).mul(float(w))
            else:
                if list(t.shape) != shape_record[k]:
                    raise RuntimeError(f"Shape mismatch for {k} in {ep}")
                accum[k].add_(t.detach().to(dtype=torch.float32), alpha=float(w))
        del state
    # Save in original dtypes where safe. Current DeBERTa checkpoints are fp32.
    out_state = {}
    for k, t in accum.items():
        if dtype_record[k] == "torch.float32":
            out_state[k] = t.contiguous()
        else:
            out_state[k] = t.to(dtype=getattr(torch, dtype_record[k].split(".")[-1])).contiguous()
    save_file(out_state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    copied = copy_static(endpoint_dir(label, endpoints[-1]), out_dir)
    model_sha = sha256_file(out_dir / "model.safetensors")

    validation: dict[str, Any] = {"attempted": False}
    if validate_load:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        validation["attempted"] = True
        tok = AutoTokenizer.from_pretrained(str(out_dir), local_files_only=True)
        model = AutoModelForMaskedLM.from_pretrained(str(out_dir), trust_remote_code=True, local_files_only=True)
        model.eval()
        enc = tok(
            [
                "The small child looked at the red ball.",
                "A scientist can compare averaged masked language models.",
                "BabyLM strict small models must learn from limited data.",
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        with torch.no_grad():
            logits = model(**enc).logits
        validation.update({
            "trusted_model_class": model.__class__.__name__,
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "adapter_scale": float(getattr(model.config, "adapter_scale", -1)),
            "adapter_bottleneck": int(getattr(model.config, "adapter_bottleneck", -1)),
            "vocab_size": len(tok),
            "logits_shape": list(logits.shape),
            "logits_all_finite": bool(torch.isfinite(logits).all().item()),
        })
        try:
            native, loading = AutoModelForMaskedLM.from_pretrained(
                str(out_dir), trust_remote_code=False, local_files_only=True, output_loading_info=True
            )
            validation["native_fallback"] = {
                "loads": True,
                "class": native.__class__.__name__,
                "param_count": int(sum(p.numel() for p in native.parameters())),
                "unexpected_keys_count": len(loading.get("unexpected_keys", [])),
                "is_scientifically_wrong_function": int(sum(p.numel() for p in native.parameters())) != EXPECTED_PARAMS or len(loading.get("unexpected_keys", [])) > 0,
            }
        except Exception as e:
            validation["native_fallback"] = {"loads": False, "error_type": type(e).__name__, "error": str(e)}

    summary = {
        "status": "LATE_WEIGHT_AVERAGE_CANDIDATE_BUILT",
        "created_utc": utc_now(),
        "candidate_dir": rel(out_dir),
        "candidate": recipe_key(label, template),
        "model_sha256": model_sha,
        "plan": plan,
        "copied_static_files": copied,
        "validation": validation,
        "official_evaluation_performed": False,
        "leaderboard_submission_performed": False,
        "scientific_reading": "Mechanical candidate only. Its value depends on later official-compatible selected scoring after seed43122 robustness evidence is read.",
    }
    write_json(out_dir / "averaging_manifest.json", summary)
    return summary


def make_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# research late-weight-average scaffold",
        "",
        f"Created UTC: `{manifest['created_utc']}`",
        "",
        "## Scientific purpose",
        "",
        "Same-trajectory late-iterate averaging is prepared as a possible competence-stabilization test, distinct from the closed prediction-vote/per-column-selection route and from the old two-checkpoint average in a different tokenizer coordinate. It must not be read as evidence until an averaged checkpoint is evaluated by the official-compatible selected tasks.",
        "",
        "## Available candidates",
        "",
    ]
    for cand in manifest["candidates"]:
        lines.extend([
            f"### `{cand['candidate']}`",
            f"- trajectory: `{cand['trajectory_label']}`; adapter scale `{cand['adapter_scale']}`; same seed/mask as reference: `{cand['same_seed_mask_as_reference']}`",
            f"- endpoints: `{', '.join(cand['endpoints'])}`",
            f"- normalized weights: `{[round(x, 6) for x in cand['normalized_weights']]}`",
            f"- buildable: `{cand['ok_to_build']}`",
            f"- reading: {cand['reading']}",
            "",
        ])
    lines.extend([
        "## Boundaries",
        "",
        "- Do not average across different random initializations or independent seed trajectories with this script.",
        "- Do not use these candidates for final endpoint claims without full selected cheap-task scoring and, if competitive, SuperGLUE/AoA completion.",
        "- No model evaluation or leaderboard submission is performed by the dry-run manifest.",
        "",
        f"JSON: `{rel(OUT_ROOT / 'late_weight_average_scaffold_manifest.json')}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", choices=sorted(TRAJECTORIES), help="Trajectory label for --build-candidate")
    ap.add_argument("--template", choices=sorted(RECIPE_TEMPLATES), help="Recipe template for --build-candidate")
    ap.add_argument("--build-candidate", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--hash-models", action="store_true", help="Hash source model.safetensors in dry-run; slower")
    ap.add_argument("--no-validate-load", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.build_candidate:
        if not args.trajectory or not args.template:
            raise SystemExit("--build-candidate requires --trajectory and --template")
        summary = build_candidate(args.trajectory, args.template, overwrite=args.overwrite, validate_load=not args.no_validate_load)
        print(json.dumps({
            "status": summary["status"],
            "candidate": summary["candidate"],
            "candidate_dir": summary["candidate_dir"],
            "model_sha256": summary["model_sha256"],
            "validation": summary["validation"],
            "official_evaluation_performed": False,
            "leaderboard_submission_performed": False,
        }, indent=2, ensure_ascii=False), flush=True)
        return

    candidates = []
    for label in TRAJECTORIES:
        for template in RECIPE_TEMPLATES:
            candidates.append(inspect_recipe(label, template, hash_models=args.hash_models))
    manifest = {
        "status": "LATE_WEIGHT_AVERAGE_SCAFFOLD_DRY_RUN",
        "created_utc": utc_now(),
        "out_root": rel(OUT_ROOT),
        "candidates": candidates,
        "n_candidates": len(candidates),
        "official_evaluation_performed": False,
        "leaderboard_submission_performed": False,
        "scientific_reading": "Prepared exact same-trajectory parameter-averaging recipes; use only after pending seed43122/A01 evidence clarifies whether stabilization is the right next route.",
    }
    write_json(OUT_ROOT / "late_weight_average_scaffold_manifest.json", manifest)
    (OUT_ROOT / "late_weight_average_scaffold_manifest.md").write_text(make_markdown(manifest), encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "out_json": rel(OUT_ROOT / "late_weight_average_scaffold_manifest.json"),
        "n_candidates": manifest["n_candidates"],
        "official_evaluation_performed": False,
        "leaderboard_submission_performed": False,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
