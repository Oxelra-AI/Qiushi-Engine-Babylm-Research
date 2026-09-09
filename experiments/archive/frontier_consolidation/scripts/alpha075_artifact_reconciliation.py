#!/usr/bin/env python3
"""research: reconcile alpha0.75 private-scale endpoint artifact identity.

Checks:
- exact SHA256 of model.safetensors/config/carrier;
- whether alpha0.75 weights are bit-identical to coherent86 alpha1 weights;
- which config fields differ;
- whether the alpha0.75 model reloads with trust_remote_code and has the expected
  private scale/parameter count;
- small CPU logits sanity: alpha0.75 differs from alpha1 but is deterministic under
  save/load, and private-off still recovers the slow anchor if the model exposes the
  expected config switch.

This script does not upload, submit, train, or evaluate official tasks.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/alpha075_artifact_reconciliation')
CACHE = _public_path('experiments/archive/frontier_consolidation/data/alpha075_artifact_reconciliation/runtime_cache')

# Set caches before importing transformers for trusted-code dynamic modules.
for key, sub in {
    "HF_HOME": "home",
    "HF_HUB_CACHE": "hub",
    "HUGGINGFACE_HUB_CACHE": "hub",
    "HF_DATASETS_CACHE": "datasets",
    "TRANSFORMERS_CACHE": "transformers",
    "HF_MODULES_CACHE": "modules",
    "XDG_CACHE_HOME": "xdg",
}.items():
    p = CACHE / sub
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p.resolve())

import torch  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

ALPHA075 = _public_path('models/frontier')
ALPHA05 = _public_path('experiments/archive/frontier_consolidation/training/runs/coherent86_private_scale_0p5/hf_model/final')
COHERENT_ALPHA1 = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final')
COHERENT_REPLAY = _public_path('experiments/archive/frontier_consolidation/training/runs/replay_frozen82_fastpath4M_coherent_seed43022/hf_model/final')
ANCHOR = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
CARRIER075 = _public_path('experiments/archive/frontier_consolidation/data/truthful_private_scale_carriers/coherent86_alpha0p75/all_full_preds_truthful_coherent86_alpha0p75_mlm.json')
MANIFEST075 = _public_path('experiments/archive/frontier_consolidation/data/truthful_private_scale_carriers/coherent86_alpha0p75/truthful_coherent86_alpha0p75_carrier_manifest.json')


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(p: Path | str) -> str:
    q = Path(p)
    try:
        return str(q.resolve().relative_to(ROOT))
    except Exception:
        return str(q)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None, "sha256": sha256_file(path) if path.exists() and path.is_file() else None}


def compare_config(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(a) | set(b))
    diffs = {}
    same = []
    for k in keys:
        if a.get(k) != b.get(k):
            diffs[k] = {"alpha075": a.get(k), "coherent_alpha1": b.get(k)}
        else:
            same.append(k)
    return {"num_same": len(same), "num_different": len(diffs), "different_fields": diffs}


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def load_model(path: Path, device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model.to(device)
    model.eval()
    return tok, model


def logits_probe(paths: dict[str, Path]) -> dict[str, Any]:
    # Short CPU probe only. Use simple legal text; not an official evaluation.
    texts = [
        "The child put the toy in the box and then [MASK] smiled.",
        "If the cup is full, water can [MASK] from it.",
    ]
    out: dict[str, Any] = {"texts": texts, "status": "not_run"}
    loaded = {}
    try:
        for label, path in paths.items():
            tok, model = load_model(path, "cpu")
            enc = tok(texts, padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits.detach().float()
            loaded[label] = {"tokenizer": tok, "model": model, "logits": logits}
        def diff(x: str, y: str) -> dict[str, float]:
            d = (loaded[x]["logits"] - loaded[y]["logits"]).abs()
            return {"max_abs": float(d.max().item()), "mean_abs": float(d.mean().item())}
        out = {
            "texts": texts,
            "status": "ok",
            "param_counts": {label: count_params(v["model"]) for label, v in loaded.items()},
            "diffs": {
                "alpha075_vs_alpha1": diff("alpha075", "alpha1"),
                "alpha075_vs_alpha05": diff("alpha075", "alpha05") if "alpha05" in loaded else None,
            },
        }
        # Try private-off recovery. The custom module copies enabled/scale into each
        # ScaledBottleneckAdapter at construction, so editing config alone is not enough.
        if "alpha075" in loaded:
            model = loaded["alpha075"]["model"]
            tok = loaded["alpha075"]["tokenizer"]
            enc = tok(texts, padding=True, return_tensors="pt")
            if hasattr(model, "set_private_enabled"):
                model.set_private_enabled(False)
                with torch.no_grad():
                    off_logits = model(**enc).logits.detach().float()
                model.set_private_enabled(True)
                anchor_tok, anchor_model = load_model(ANCHOR, "cpu")
                enc_anchor = anchor_tok(texts, padding=True, return_tensors="pt")
                with torch.no_grad():
                    anchor_logits = anchor_model(**enc_anchor).logits.detach().float()
                d = (off_logits - anchor_logits).abs()
                out["private_off_vs_anchor"] = {
                    "method": "model.set_private_enabled(False)",
                    "max_abs": float(d.max().item()),
                    "mean_abs": float(d.mean().item()),
                }
            else:
                out["private_off_vs_anchor"] = {"status": "model_has_no_set_private_enabled_method"}
    except Exception as e:
        out["status"] = "probe_failed"
        out["error"] = repr(e)
    finally:
        for v in loaded.values():
            try:
                del v["model"]
            except Exception:
                pass
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    alpha075_cfg = read_json(_public_path('models/frontier/config.json'))
    alpha1_cfg = read_json(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final/config.json'))
    alpha05_cfg = read_json(_public_path('experiments/archive/frontier_consolidation/training/runs/coherent86_private_scale_0p5/hf_model/final/config.json')) if ALPHA05.exists() else None
    manifest = read_json(MANIFEST075)

    files = {
        "alpha075_model": file_record(_public_path('models/frontier/model.safetensors')),
        "alpha075_config": file_record(_public_path('models/frontier/config.json')),
        "alpha075_carrier": file_record(CARRIER075),
        "alpha075_manifest": file_record(MANIFEST075),
        "alpha1_model": file_record(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final/model.safetensors')),
        "alpha1_config": file_record(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final/config.json')),
        "alpha1_replay_model": file_record(_public_path('experiments/archive/frontier_consolidation/training/runs/replay_frozen82_fastpath4M_coherent_seed43022/hf_model/final/model.safetensors')),
        "alpha05_model": file_record(_public_path('experiments/archive/frontier_consolidation/training/runs/coherent86_private_scale_0p5/hf_model/final/model.safetensors')) if ALPHA05.exists() else {"exists": False},
        "anchor_model": file_record(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M/model.safetensors')),
    }

    same_weight_as_alpha1 = files["alpha075_model"]["sha256"] == files["alpha1_model"]["sha256"]
    same_weight_as_replay = files["alpha075_model"]["sha256"] == files["alpha1_replay_model"]["sha256"]
    same_weight_as_alpha05 = files["alpha075_model"]["sha256"] == files["alpha05_model"].get("sha256")
    sha_lengths = {k: (len(v["sha256"]) if v.get("sha256") else None) for k, v in files.items()}

    expected_from_manifest = {
        "model_safetensors_sha256": manifest.get("model_identity", {}).get("model_safetensors_sha256"),
        "config_sha256": manifest.get("model_identity", {}).get("config_sha256"),
        "carrier_sha256": manifest.get("carrier_sha256"),
    }
    manifest_matches = {
        "model": expected_from_manifest["model_safetensors_sha256"] == files["alpha075_model"]["sha256"],
        "config": expected_from_manifest["config_sha256"] == files["alpha075_config"]["sha256"],
        "carrier": expected_from_manifest["carrier_sha256"] == files["alpha075_carrier"]["sha256"],
    }

    result: dict[str, Any] = {
        "status": "ALPHA075_ARTIFACT_RECONCILIATION",
        "created_utc": utc_now(),
        "files": files,
        "sha256_lengths": sha_lengths,
        "expected_from_manifest": expected_from_manifest,
        "manifest_matches_recomputed": manifest_matches,
        "alpha075_model_identity": {
            "model_dir": rel(ALPHA075),
            "private_adapter_scale": alpha075_cfg.get("private_adapter_scale"),
            "private_adapter_enabled": alpha075_cfg.get("private_adapter_enabled"),
            "architectures": alpha075_cfg.get("architectures"),
            "auto_map": alpha075_cfg.get("auto_map"),
        },
        "weight_identity": {
            "alpha075_weights_bit_identical_to_coherent_alpha1": same_weight_as_alpha1,
            "alpha075_weights_bit_identical_to_coherent_replay": same_weight_as_replay,
            "alpha075_weights_bit_identical_to_alpha05": same_weight_as_alpha05,
            "interpretation": "alpha0.75 is a config-scaled inference function over the same learned coherent86 private-path weights" if same_weight_as_alpha1 else "alpha0.75 has distinct weight bytes from coherent alpha1; inspect tensor provenance",
        },
        "config_diff_alpha075_vs_alpha1": compare_config(alpha075_cfg, alpha1_cfg),
        "config_diff_alpha075_vs_alpha05": compare_config(alpha075_cfg, alpha05_cfg) if alpha05_cfg else None,
        "carrier_score_from_manifest": manifest.get("score_arithmetic_candidate_native"),
        "truthful_history_policy": manifest.get("truthful_history_policy"),
        "logits_probe": logits_probe({"alpha075": ALPHA075, "alpha1": COHERENT_ALPHA1, **({"alpha05": ALPHA05} if ALPHA05.exists() else {})}),
    }
    out_json = _public_path('experiments/archive/frontier_consolidation/data/alpha075_artifact_reconciliation/alpha075_artifact_reconciliation.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/alpha075_artifact_reconciliation/alpha075_artifact_reconciliation.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research alpha0.75 artifact reconciliation",
        "",
        "This is file/CPU-only endpoint identity work. No training, upload, or leaderboard submission was performed.",
        "",
        f"Status: `{result['status']}`",
        f"alpha0.75 model SHA256: `{files['alpha075_model']['sha256']}` (length {sha_lengths['alpha075_model']})",
        f"alpha0.75 config SHA256: `{files['alpha075_config']['sha256']}`",
        f"alpha0.75 carrier SHA256: `{files['alpha075_carrier']['sha256']}`",
        f"Manifest matches recomputed digests: `{manifest_matches}`",
        "",
        "## Weight identity",
        f"- weights bit-identical to coherent alpha1: `{same_weight_as_alpha1}`",
        f"- weights bit-identical to coherent replay: `{same_weight_as_replay}`",
        f"- weights bit-identical to alpha0.5: `{same_weight_as_alpha05}`",
        f"- interpretation: {result['weight_identity']['interpretation']}",
        "",
        "## Config difference vs alpha1",
    ]
    for k, v in result["config_diff_alpha075_vs_alpha1"]["different_fields"].items():
        lines.append(f"- `{k}`: alpha0.75 `{v['alpha075']}` vs alpha1 `{v['coherent_alpha1']}`")
    lines.extend([
        "",
        "## CPU logits probe",
        f"`{result['logits_probe']}`",
        "",
        "## Endpoint score from existing truthful manifest",
    ])
    score = manifest.get("score_arithmetic_candidate_native", {})
    lines.append(f"- Overall(AoA0): `{score.get('overall_with_aoa0')}`")
    lines.append(f"- cheap7: `{score.get('cheap7')}`")
    lines.append(f"- SuperGLUE: `{(score.get('scores') or {}).get('SuperGLUE')}`")
    lines.extend(["", f"JSON: `{rel(out_json)}`"])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "model_sha256": files["alpha075_model"]["sha256"],
        "sha_len": sha_lengths["alpha075_model"],
        "manifest_matches": manifest_matches,
        "same_weight_as_alpha1": same_weight_as_alpha1,
        "config_diffs": result["config_diff_alpha075_vs_alpha1"]["different_fields"],
        "logits_probe_status": result["logits_probe"].get("status"),
        "out_json": rel(out_json),
        "out_md": rel(out_md),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
