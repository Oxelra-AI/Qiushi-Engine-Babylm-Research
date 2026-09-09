#!/usr/bin/env python3
"""Verify research matched-arm smoke artifacts and official loading contracts."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, DebertaV2Config


STUDY = _public_path('experiments/archive/initial_model_studies')
WORKSPACE = _public_path('experiments/archive/initial_model_studies')
SCRIPT_DIR = _public_path('experiments/archive/initial_model_studies/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
from modeling_identity_address import IdentityAddressDebertaForMaskedLM  # noqa: E402


MODES = ("off", "shuffled", "true")
ROOTS = {
    mode: WORKSPACE / f"data/identity_address_smoke_{mode}_constant"
    for mode in MODES
}
OUT = _public_path('experiments/archive/initial_model_studies/data/identity_address_smoke_verification.json')
NOTE = _public_path('research/notes/initial_model_studies/identity_address_smoke_verification.md')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    runs = {}
    for mode, root in ROOTS.items():
        metrics = json.loads((root / "scientific_metrics.json").read_text(encoding="utf-8"))
        log = [json.loads(line) for line in (root / "training_log.jsonl").read_text().splitlines()]
        checkpoint = root / "hf_model/chck_1M"
        runs[mode] = {
            "parameter_count": metrics["parameter_count"],
            "word_exposure": metrics["word_exposure"],
            "masked_tokens_total": metrics["masked_tokens_total"],
            "loss_first": metrics["loss_first"],
            "loss_last": metrics["loss_last"],
            "training_elapsed_sec": log[-1]["elapsed_sec"],
            "identity_address": metrics["identity_address"],
            "checkpoint": str(checkpoint.resolve()),
            "weight_sha256": sha256_file(checkpoint / "model.safetensors"),
            "config_sha256": sha256_file(checkpoint / "config.json"),
            "modeling_sha256": sha256_file(checkpoint / "modeling_identity_address.py"),
        }

    config = DebertaV2Config.from_pretrained(ROOTS["true"] / "hf_model/chck_1M")
    initial_states = []
    for mode in ("off", "true"):
        config.identity_address_mode = mode
        torch.manual_seed(91_313)
        model = IdentityAddressDebertaForMaskedLM(config)
        initial_states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
    initialization_equal = initial_states[0].keys() == initial_states[1].keys() and all(
        torch.equal(initial_states[0][key], initial_states[1][key])
        for key in initial_states[0]
    )

    checkpoint = ROOTS["true"] / "hf_model/chck_1M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=True)
    mlm = AutoModelForMaskedLM.from_pretrained(checkpoint, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    text = f"Alice placed the marble in the garden. Later the marble remained in the {tokenizer.mask_token}."
    batch = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        mlm_shape = list(mlm(**batch).logits.shape)
        encoder_shape = list(encoder(**batch).last_hidden_state.shape)

    same_parameter_count = len({runs[mode]["parameter_count"] for mode in MODES}) == 1
    same_exposure = len({runs[mode]["word_exposure"] for mode in MODES}) == 1
    true_activation = runs["true"]["identity_address"]["identity_address_activation_rate"]
    shuffled_activation = runs["shuffled"]["identity_address"]["identity_address_activation_rate"]
    activation_matched = abs(true_activation - shuffled_activation) < 1e-12
    elapsed_ratio = {
        mode: runs[mode]["training_elapsed_sec"] / runs["off"]["training_elapsed_sec"]
        for mode in ("shuffled", "true")
    }
    remote_load_ok = (
        mlm.__class__.__name__ == "IdentityAddressDebertaForMaskedLM"
        and encoder.__class__.__name__ == "IdentityAddressDebertaModel"
    )
    all_checks = (
        same_parameter_count
        and same_exposure
        and initialization_equal
        and activation_matched
        and true_activation >= 0.20
        and remote_load_ok
        and max(elapsed_ratio.values()) <= 1.10
    )
    decision = (
        "NATURAL_TEXT_IMPLEMENTATION_READY_FOR_GPU_GATE"
        if all_checks
        else "BLOCK_GPU_GATE_AND_REPAIR_IMPLEMENTATION"
    )
    payload = {
        "status": "IDENTITY_ADDRESS_SMOKE_VERIFIED",
        "decision": decision,
        "checks": {
            "same_parameter_count": same_parameter_count,
            "same_word_exposure": same_exposure,
            "initial_state_dicts_exactly_equal": initialization_equal,
            "true_shuffled_activation_exactly_matched": activation_matched,
            "true_activation_rate": true_activation,
            "remote_auto_model_loading": remote_load_ok,
            "parallel_cpu_elapsed_ratio_vs_off": elapsed_ratio,
        },
        "official_load_smoke": {
            "mlm_class": mlm.__class__.__name__,
            "encoder_class": encoder.__class__.__name__,
            "mlm_output_shape": mlm_shape,
            "encoder_output_shape": encoder_shape,
        },
        "runs": runs,
        "scope": "implementation readiness only; no BabyLM score claim",
    }
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT)
    lines = [
        "# research identity-address natural-text smoke",
        "",
        f"Decision: **{decision}**",
        "",
        f"Exact initialization equality: {initialization_equal}",
        f"Parameter count (all arms): {runs['true']['parameter_count']}",
        f"True/shuffled activation: {true_activation:.4%} / {shuffled_activation:.4%}",
        f"CPU elapsed ratio shuffled/true vs off: {elapsed_ratio['shuffled']:.3f} / {elapsed_ratio['true']:.3f}",
        f"Official AutoModel loading: {remote_load_ok}",
        "",
        "This verifies implementation and natural-text coverage only. It is not evidence of an Overall gain.",
        "",
        f"Evidence JSON: `{OUT}`",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "decision": decision, "checks": payload["checks"]}, indent=2))


if __name__ == "__main__":
    main()
