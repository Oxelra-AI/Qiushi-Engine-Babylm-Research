#!/usr/bin/env python3
"""Mechanical check for research frozen-82M slow path + fresh private adapter.

Checks that loading the verified scale1.75 chck_82M checkpoint into the new wrapper
preserves the slow function exactly and exposes only `.private_adapter.*` for future
tail training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, DebertaV2Config

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(SCRIPTS))
from adapter_scaled_modeling import AdapterDebertaV2ForMaskedLM as SlowAdapterModel  # noqa: E402
from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM  # noqa: E402

ENDPOINT = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/frozen82_private_mech_check')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/frozen82_private_mech_check/frozen82_private_mech_check.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/frozen82_private_mech_check/frozen82_private_mech_check.md')


def rel(p: Path | str):
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def set_private_enabled(model, enabled: bool):
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(enabled)
    else:
        for layer in model.deberta.encoder.layer:
            layer.private_adapter.enabled = bool(enabled)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hf_cache = _public_path('experiments/archive/frontier_consolidation/data/frozen82_private_mech_check/hf_cache')
    (hf_cache / "modules").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_MODULES_CACHE"] = str(hf_cache / "modules")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(ENDPOINT), local_files_only=True)

    # Original research checkpoint for exact functional comparison, loaded without
    # HuggingFace dynamic-module machinery so the check never writes to a read-only
    # global cache.  The endpoint config is a plain DebertaV2Config with adapter attrs.
    cfg = DebertaV2Config.from_pretrained(str(ENDPOINT), local_files_only=True)
    sd = load_file(str(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M/model.safetensors')), device="cpu")
    tied_missing = {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}
    orig = SlowAdapterModel(cfg)
    missing_orig, unexpected_orig = orig.load_state_dict(sd, strict=False)
    if set(missing_orig) - tied_missing or unexpected_orig:
        raise RuntimeError(f"unexpected original load mismatch: {missing_orig=} {unexpected_orig=}")
    orig.tie_weights()
    orig.eval().to(device)

    cfg_new = DebertaV2Config.from_pretrained(str(ENDPOINT), local_files_only=True)
    cfg_new.private_adapter_bottleneck = 128
    cfg_new.private_adapter_scale = 1.0
    cfg_new.private_adapter_enabled = True
    new = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg_new)
    missing, unexpected = new.load_state_dict(sd, strict=False)
    new.tie_weights()
    new.eval().to(device)

    texts = [
        "The small dog runs in the garden.",
        "A child moved the apple from the box to the table.",
        "Because the toy was wet, the boy dried it with a towel.",
    ]
    batch = tok(texts, padding=True, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        orig_logits = orig(**batch).logits.detach()
        set_private_enabled(new, True)
        new_on_logits = new(**batch).logits.detach()
        set_private_enabled(new, False)
        new_off_logits = new(**batch).logits.detach()
        set_private_enabled(new, True)

    slow_vs_orig_max = float((new_off_logits - orig_logits).abs().max().cpu())
    private_on_vs_off_max = float((new_on_logits - new_off_logits).abs().max().cpu())
    private_zero_rms = [float(x) for x in new.private_adapter_rms()]
    slow_rms = [float(x) for x in new.slow_adapter_rms()]

    private_names = {n for n, _ in new.named_parameters() if ".private_adapter." in n}
    slow_names = {n for n, _ in new.named_parameters() if ".private_adapter." not in n}

    # Gradient isolation: freeze slow path and make sure only private parameters get gradients.
    new.train()
    for n, p in new.named_parameters():
        p.requires_grad_(n in private_names)
        p.grad = None
    set_private_enabled(new, True)
    labels = batch["input_ids"].clone()
    labels[:, :] = -100
    # Force a tiny deterministic target at one non-padding position per example.
    for i in range(labels.shape[0]):
        valid = (batch["attention_mask"][i] == 1).nonzero(as_tuple=False).flatten()
        if valid.numel() > 2:
            pos = int(valid[min(2, valid.numel() - 1)].item())
            labels[i, pos] = batch["input_ids"][i, pos]
            batch["input_ids"][i, pos] = tok.mask_token_id
    out = new(**batch)
    vocab = out.logits.shape[-1]
    loss = F.cross_entropy(out.logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100)
    loss.backward()
    private_grad_norm = 0.0
    slow_grad_norm = 0.0
    private_grad_nonzero_names = []
    for n, p in new.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().norm().cpu())
        if n in private_names:
            private_grad_norm += g
            if g > 0:
                private_grad_nonzero_names.append(n)
        else:
            slow_grad_norm += g

    # Verify disabled private path blocks any private gradient path for the same loss.
    # With the slow path frozen and the zero-output private branch disabled, the loss
    # correctly has no grad_fn. That is the desired property for a frozen-slow base
    # branch: there is no accidental gradient route into the private tensors.
    for n, p in new.named_parameters():
        p.grad = None
    set_private_enabled(new, False)
    out2 = new(**batch)
    loss2 = F.cross_entropy(out2.logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100)
    disabled_loss_has_grad_fn = loss2.grad_fn is not None
    disabled_private_grad_norm = 0.0
    if disabled_loss_has_grad_fn:
        loss2.backward()
        for n, p in new.named_parameters():
            if n in private_names and p.grad is not None:
                disabled_private_grad_norm += float(p.grad.detach().norm().cpu())

    checks = {
        "loaded_with_only_private_or_tied_missing": bool(missing) and all(("private_adapter" in x) or (x in tied_missing) for x in missing) and len(unexpected) == 0,
        "slow_path_matches_original": slow_vs_orig_max < 1e-5,
        "private_on_off_equivalent_at_attachment": private_on_vs_off_max < 1e-7,
        "private_rms_zero_at_attachment": max(private_zero_rms) == 0.0,
        "slow_adapter_active": max(slow_rms) > 0.0,
        "only_private_requires_grad_after_freeze": all((p.requires_grad == (n in private_names)) for n, p in new.named_parameters()),
        "private_grad_nonzero_when_enabled": private_grad_norm > 0.0,
        "slow_grad_zero_when_frozen": slow_grad_norm == 0.0,
        "disabled_private_loss_has_no_grad_path_or_zero_private_grad": (not disabled_loss_has_grad_fn) or disabled_private_grad_norm == 0.0,
    }
    checks["all_checks_passed"] = all(checks.values())

    result = {
        "status": "PASS" if checks["all_checks_passed"] else "CHECKS_FAILED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Mechanical validation of a fresh zero-output private pathway attached to the verified frozen scale1.75 chck_82M slow function.",
        "endpoint": rel(ENDPOINT),
        "device": str(device),
        "load_state": {
            "missing_keys_count": len(missing),
            "missing_keys_sample": list(missing)[:12],
            "unexpected_keys_count": len(unexpected),
            "unexpected_keys_sample": list(unexpected)[:12],
        },
        "model_identity": {
            "original_class": orig.__class__.__name__,
            "new_class": new.__class__.__name__,
            "param_count_original": int(sum(p.numel() for p in orig.parameters())),
            "param_count_new": int(sum(p.numel() for p in new.parameters())),
            "private_param_count": int(sum(p.numel() for n, p in new.named_parameters() if n in private_names)),
            "slow_param_count": int(sum(p.numel() for n, p in new.named_parameters() if n in slow_names)),
            "adapter_scale_slow": float(getattr(new.config, "adapter_scale", -1.0)),
            "private_adapter_scale": float(getattr(new.config, "private_adapter_scale", -1.0)),
        },
        "function_checks": {
            "slow_vs_original_max_abs_logit_diff": slow_vs_orig_max,
            "private_on_vs_off_max_abs_logit_diff": private_on_vs_off_max,
            "private_zero_rms": private_zero_rms,
            "slow_adapter_rms": slow_rms,
        },
        "gradient_checks": {
            "private_grad_norm_enabled": private_grad_norm,
            "slow_grad_norm_when_frozen": slow_grad_norm,
            "disabled_loss_has_grad_fn": disabled_loss_has_grad_fn,
            "disabled_private_grad_norm": disabled_private_grad_norm,
            "private_grad_nonzero_names_sample": private_grad_nonzero_names[:12],
        },
        "checks": checks,
        "scientific_reading": "The verified chck_82M score-bearing function can be represented as a frozen slow path with a fresh zero-output private residual attached exactly. This supports the proposed 82M-anchored tail experiment if the research panel justifies it; no new training was launched by this check.",
        "elapsed_sec": round(time.time() - t0, 3),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# research frozen-82M private adapter mechanical check\n\n")
    md.append(f"Status: **{result['status']}**\n\n")
    md.append("## Function preservation\n")
    md.append(f"- Slow path vs original max logit diff: `{slow_vs_orig_max}`.\n")
    md.append(f"- Private ON vs OFF at attachment max logit diff: `{private_on_vs_off_max}`.\n")
    md.append(f"- Slow adapter active RMS max: `{max(slow_rms)}`; private RMS max: `{max(private_zero_rms)}`.\n")
    md.append("\n## Gradient isolation\n")
    md.append(f"- Private grad norm enabled: `{private_grad_norm}`.\n")
    md.append(f"- Slow grad norm while frozen: `{slow_grad_norm}`.\n")
    md.append(f"- Disabled private loss has grad_fn: `{disabled_loss_has_grad_fn}`; private grad norm while disabled: `{disabled_private_grad_norm}`.\n")
    md.append("\n## Checks\n")
    for k, v in checks.items():
        md.append(f"- `{k}`: `{v}`\n")
    md.append("\n" + result["scientific_reading"] + "\n")
    md.append(f"\nJSON: `{rel(OUT_JSON)}`\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD), "all_checks_passed": checks["all_checks_passed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
