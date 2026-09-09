#!/usr/bin/env python3
"""research: verify the preserved chck_82M snapshot is a standalone loadable HF-style model.

This is a CPU-only release-carrier check. It does not evaluate the endpoint score;
it tests whether the copied checkpoint snapshot can be loaded from its own directory
with the custom modeling file, tokenizer, config, and weights, and whether a minimal
masked-LM forward pass is finite. The result helps separate score evidence from
packaging/release integrity while long training/fast-eval tasks continue.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
SNAPSHOT = ROOT / "experiments/archive/representation_and_objectives/data/chck82_endpoint_snapshot/checkpoint/chck_82M"
MANIFEST = ROOT / "experiments/archive/representation_and_objectives/data/chck82_endpoint_snapshot/manifest.json"
OUT_DIR = ROOT / "experiments/archive/representation_and_objectives/data/snapshot_release_load_check"
OUT_JSON = OUT_DIR / "snapshot_release_load_check.json"
OUT_MD = OUT_DIR / "snapshot_release_load_check.md"
EXPECTED_MODEL_SHA = "93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3"
EXPECTED_PARAMETER_COUNT = 35_463_008
EXPECTED_VOCAB_SIZE = 16_384
REQUIRED_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "adapter_scaled_modeling.py",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []

    files: dict[str, Any] = {}
    for name in REQUIRED_FILES:
        p = SNAPSHOT / name
        files[name] = {
            "path": rel(p),
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else None,
            "sha256": sha256_file(p) if p.exists() else None,
        }
        if not p.exists():
            errors.append(f"missing_required_file:{name}")

    manifest: dict[str, Any] | None = None
    if MANIFEST.exists():
        manifest = load_json(MANIFEST)
        if manifest.get("candidate_model_sha256") != EXPECTED_MODEL_SHA:
            errors.append(f"manifest_candidate_sha_mismatch:{manifest.get('candidate_model_sha256')}")
    else:
        errors.append("snapshot_manifest_missing")

    config: dict[str, Any] | None = None
    tokenizer_config: dict[str, Any] | None = None
    special_tokens: dict[str, Any] | None = None
    if (SNAPSHOT / "config.json").exists():
        config = load_json(SNAPSHOT / "config.json")
        if config.get("vocab_size") != EXPECTED_VOCAB_SIZE:
            errors.append(f"config_vocab_size_{config.get('vocab_size')}_not_{EXPECTED_VOCAB_SIZE}")
        if config.get("adapter_scale") != 1.75:
            errors.append(f"config_adapter_scale_{config.get('adapter_scale')}_not_1.75")
        if config.get("adapter_bottleneck") != 128:
            errors.append(f"config_adapter_bottleneck_{config.get('adapter_bottleneck')}_not_128")
        auto_map = config.get("auto_map")
        if not isinstance(auto_map, dict) or "AutoModelForMaskedLM" not in auto_map:
            errors.append("config_auto_map_missing_AutoModelForMaskedLM")
    if (SNAPSHOT / "tokenizer_config.json").exists():
        tokenizer_config = load_json(SNAPSHOT / "tokenizer_config.json")
    if (SNAPSHOT / "special_tokens_map.json").exists():
        special_tokens = load_json(SNAPSHOT / "special_tokens_map.json")

    model_sha = files.get("model.safetensors", {}).get("sha256")
    if model_sha != EXPECTED_MODEL_SHA:
        errors.append(f"model_sha_{model_sha}_not_expected_{EXPECTED_MODEL_SHA}")

    load_record: dict[str, Any] = {"attempted": False}
    forward_record: dict[str, Any] = {"attempted": False}
    if not errors:
        try:
            # Keep dynamic-code/cache writes inside the output directory.
            # Do not use setdefault: the runtime may provide a read-only default
            # HF/HOME cache, and trust_remote_code copies custom modules before load.
            cache = OUT_DIR / "runtime_cache"
            env_paths = {
                "HOME": cache / "home",
                "XDG_CACHE_HOME": cache / "xdg_cache",
                "HF_HOME": cache / "hf_home",
                "HF_HUB_CACHE": cache / "hf_home" / "hub",
                "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
                "TRANSFORMERS_CACHE": cache / "transformers",
                "HF_MODULES_CACHE": cache / "hf_modules",
                "TORCH_HOME": cache / "torch",
                "TMPDIR": cache / "tmp",
            }
            for key, path in env_paths.items():
                path.mkdir(parents=True, exist_ok=True)
                os.environ[key] = str(path)
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            import torch
            from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

            torch.set_num_threads(2)
            load_record["attempted"] = True
            hf_config = AutoConfig.from_pretrained(str(SNAPSHOT), trust_remote_code=True, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), use_fast=True, trust_remote_code=True, local_files_only=True)
            model = AutoModelForMaskedLM.from_pretrained(str(SNAPSHOT), trust_remote_code=True, local_files_only=True)
            model.eval()
            param_count = sum(p.numel() for p in model.parameters())
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            load_record.update({
                "ok": True,
                "config_class": type(hf_config).__name__,
                "model_class": type(model).__name__,
                "tokenizer_class": type(tokenizer).__name__,
                "parameter_count": int(param_count),
                "trainable_parameter_count_if_unfrozen": int(trainable_count),
                "vocab_size_from_tokenizer": int(len(tokenizer)),
                "mask_token": tokenizer.mask_token,
                "model_device": str(next(model.parameters()).device),
            })
            if param_count != EXPECTED_PARAMETER_COUNT:
                errors.append(f"parameter_count_{param_count}_not_{EXPECTED_PARAMETER_COUNT}")
            if len(tokenizer) != EXPECTED_VOCAB_SIZE:
                errors.append(f"loaded_tokenizer_vocab_{len(tokenizer)}_not_{EXPECTED_VOCAB_SIZE}")
            if tokenizer.mask_token is None:
                errors.append("tokenizer_missing_mask_token")

            if tokenizer.mask_token is not None:
                forward_record["attempted"] = True
                text = f"The child put the {tokenizer.mask_token} in the box."
                enc = tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    out = model(**enc)
                logits = out.logits
                mask_positions = (enc["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
                finite = bool(torch.isfinite(logits).all().item())
                forward_record.update({
                    "ok": finite and tuple(logits.shape)[-1] == EXPECTED_VOCAB_SIZE,
                    "input_text": text,
                    "input_ids_shape": list(enc["input_ids"].shape),
                    "logits_shape": list(logits.shape),
                    "all_logits_finite": finite,
                    "mask_position_count": int(mask_positions.shape[0]),
                })
                if mask_positions.shape[0] == 1:
                    b, pos = mask_positions[0].tolist()
                    top = torch.topk(logits[b, pos], k=10)
                    ids = top.indices.tolist()
                    vals = top.values.tolist()
                    forward_record["mask_top10"] = [
                        {"token_id": int(i), "token": tokenizer.convert_ids_to_tokens(int(i)), "logit": float(v)}
                        for i, v in zip(ids, vals)
                    ]
                if not finite:
                    errors.append("forward_logits_not_finite")
                if tuple(logits.shape)[-1] != EXPECTED_VOCAB_SIZE:
                    errors.append(f"forward_vocab_dim_{tuple(logits.shape)[-1]}_not_{EXPECTED_VOCAB_SIZE}")
        except Exception as exc:
            load_record["ok"] = False
            load_record["error"] = repr(exc)
            errors.append(f"load_or_forward_failed:{exc!r}")

    out = {
        "status": "PASS" if not errors else "NEEDS_REPAIR",
        "created_utc": now(),
        "purpose": "Check that the preserved chck_82M snapshot is a standalone loadable HF-style masked-LM carrier without touching the score-bearing evaluation.",
        "snapshot": rel(SNAPSHOT),
        "manifest": rel(MANIFEST),
        "expected_model_sha256": EXPECTED_MODEL_SHA,
        "files": files,
        "manifest_core": {
            "status": manifest.get("status") if isinstance(manifest, dict) else None,
            "candidate_model_sha256": manifest.get("candidate_model_sha256") if isinstance(manifest, dict) else None,
            "source_checkpoint": manifest.get("source_checkpoint") if isinstance(manifest, dict) else None,
            "snapshot_checkpoint": manifest.get("snapshot_checkpoint") if isinstance(manifest, dict) else None,
            "checkpoint_exposure": manifest.get("checkpoint_exposure") if isinstance(manifest, dict) else None,
        },
        "config_core": {
            "architectures": config.get("architectures") if isinstance(config, dict) else None,
            "auto_map": config.get("auto_map") if isinstance(config, dict) else None,
            "model_type": config.get("model_type") if isinstance(config, dict) else None,
            "vocab_size": config.get("vocab_size") if isinstance(config, dict) else None,
            "hidden_size": config.get("hidden_size") if isinstance(config, dict) else None,
            "num_hidden_layers": config.get("num_hidden_layers") if isinstance(config, dict) else None,
            "num_attention_heads": config.get("num_attention_heads") if isinstance(config, dict) else None,
            "intermediate_size": config.get("intermediate_size") if isinstance(config, dict) else None,
            "adapter_scale": config.get("adapter_scale") if isinstance(config, dict) else None,
            "adapter_bottleneck": config.get("adapter_bottleneck") if isinstance(config, dict) else None,
        },
        "tokenizer_config_core": {
            "tokenizer_class": tokenizer_config.get("tokenizer_class") if isinstance(tokenizer_config, dict) else None,
            "model_max_length": tokenizer_config.get("model_max_length") if isinstance(tokenizer_config, dict) else None,
            "mask_token": tokenizer_config.get("mask_token") if isinstance(tokenizer_config, dict) else None,
        },
        "special_tokens": special_tokens,
        "load_record": load_record,
        "forward_record": forward_record,
        "warnings": warnings,
        "errors": errors,
        "ready_as_standalone_hf_snapshot": not errors,
        "not_resolved_by_this_check": [
            "This does not establish training reproducibility; it only checks the preserved snapshot carrier.",
            "This does not materialize official --fast checkpoint predictions; research full task and verifier must still pass.",
            "This does not solve the separate context-conditioned alternative-binding mechanism problem.",
        ],
    }
    write_json(OUT_JSON, out)
    lines = [
        "# research snapshot release load check",
        "",
        f"Status: {out['status']}",
        f"Snapshot: `{rel(SNAPSHOT)}`",
        f"Model SHA: `{model_sha}`",
        f"Load ok: {load_record.get('ok')}",
        f"Forward ok: {forward_record.get('ok')}",
        f"Errors: {len(errors)}",
        "",
        f"JSON: `{rel(OUT_JSON)}`",
    ]
    if errors:
        lines += ["", "## Errors"] + [f"- {e}" for e in errors]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "json": rel(OUT_JSON), "load_ok": load_record.get("ok"), "forward_ok": forward_record.get("ok"), "errors": errors}, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
