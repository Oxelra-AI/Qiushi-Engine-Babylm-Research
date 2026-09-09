#!/usr/bin/env python3
"""Official-corpus trainer for identity-addressed DeBERTa-v2 MLM.

This wrapper preserves the verified research data, exposure, masking, optimizer,
checkpoint, and logging code.  It changes only model construction and saves the
self-contained remote-code module required by the official evaluators.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import torch

import amlm_masked_train_fullcycle as base
from modeling_identity_address import IdentityAddressDebertaForMaskedLM


HERE = _public_path('experiments/archive/initial_model_studies/scripts')
MODELING_FILE = _public_path('experiments/archive/initial_model_studies/scripts/modeling_identity_address.py')
ORIGINAL_BUILD_MODEL = base.build_model
ORIGINAL_DATA_LOADER = base.DataLoader

STOPWORDS = {
    "the", "and", "for", "was", "were", "are", "with", "from", "into", "that",
    "this", "then", "than", "have", "has", "had", "not", "but", "you", "your",
    "his", "her", "hers", "its", "our", "ours", "their", "they", "them", "there",
    "here", "where", "when", "what", "which", "who", "whom", "why", "how", "can",
    "could", "would", "should", "will", "shall", "may", "might", "must", "been",
    "being", "about", "after", "again", "before", "between", "during", "each",
    "more", "most", "other", "over", "same", "some", "such", "through", "under",
    "until", "very", "while", "does", "doing", "did", "just", "also", "only",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_token_ids(tokenizer) -> list[int]:
    accepted = []
    for token_id in range(len(tokenizer)):
        if token_id in tokenizer.all_special_ids:
            continue
        surface = tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
        if surface.isalpha() and len(surface) >= 3 and surface not in STOPWORDS:
            accepted.append(token_id)
    return accepted


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def install_model_builder(options: argparse.Namespace) -> None:
    def build_model(args, tokenizer):
        if args.model_type != "deberta_v2":
            raise ValueError("identity-address training requires --model_type deberta_v2")
        state = capture_rng_state()
        reference = ORIGINAL_BUILD_MODEL(args, tokenizer)
        config = reference.config
        del reference
        restore_rng_state(state)
        config.identity_address_mode = options.identity_address_mode
        config.identity_address_mask_token_id = tokenizer.mask_token_id
        config.identity_address_local_window = options.identity_address_local_window
        config.identity_address_continuation_window = options.identity_address_continuation_window
        config.identity_address_state_dimensions = options.identity_address_state_dimensions
        config.identity_address_insert_after_layer = options.identity_address_insert_after_layer
        config.identity_address_eligible_token_ids = eligible_token_ids(tokenizer)
        config.identity_address_mechanism_version = "research-v1"
        config.auto_map = {
            "AutoModel": "modeling_identity_address.IdentityAddressDebertaModel",
            "AutoModelForMaskedLM": "modeling_identity_address.IdentityAddressDebertaForMaskedLM",
        }
        config.architectures = ["IdentityAddressDebertaForMaskedLM"]
        return IdentityAddressDebertaForMaskedLM(config)

    base.build_model = build_model


def install_checkpoint_saver() -> None:
    def save_hf_checkpoint(model, tokenizer, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        controller = getattr(model, "identity_address", None)
        if controller is not None:
            telemetry = controller.telemetry()
            model.config.identity_address_active_targets = telemetry["active_targets"]
            model.config.identity_address_total_targets = telemetry["total_targets"]
            model.config.identity_address_activation_rate = telemetry["activation_rate"]
        model.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        base.force_portable_tokenizer_config(destination)
        shutil.copy2(MODELING_FILE, destination / MODELING_FILE.name)

    base.save_hf_checkpoint = save_hf_checkpoint


def install_data_loader(num_workers: int) -> None:
    def data_loader(*args, **kwargs):
        kwargs["num_workers"] = num_workers
        if num_workers == 0:
            kwargs["pin_memory"] = False
        return ORIGINAL_DATA_LOADER(*args, **kwargs)

    base.DataLoader = data_loader


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--identity_address_mode", choices=["off", "shuffled", "true"], required=True
    )
    parser.add_argument("--identity_address_local_window", type=int, default=24)
    parser.add_argument("--identity_address_continuation_window", type=int, default=10)
    parser.add_argument("--identity_address_state_dimensions", type=int, default=160)
    parser.add_argument("--identity_address_insert_after_layer", type=int, default=3)
    parser.add_argument("--identity_address_num_workers", type=int, default=2)
    return parser.parse_known_args()


def augment_metrics(output_dir: Path) -> None:
    metrics_path = output_dir / "scientific_metrics.json"
    config_path = output_dir / "hf_model/config.json"
    if not metrics_path.is_file() or not config_path.is_file():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fields = [
        "identity_address_mode",
        "identity_address_local_window",
        "identity_address_continuation_window",
        "identity_address_state_dimensions",
        "identity_address_insert_after_layer",
        "identity_address_mechanism_version",
        "identity_address_active_targets",
        "identity_address_total_targets",
        "identity_address_activation_rate",
    ]
    metrics["identity_address"] = {field: config.get(field) for field in fields}
    metrics["identity_address_modeling_sha256"] = sha256_file(MODELING_FILE)
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        metrics["cuda_memory"] = {
            "device_index": device,
            "device_name": properties.name,
            "total_bytes": properties.total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metrics_path)


def main() -> None:
    options, remaining = parse_wrapper_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output_dir", required=True)
    output_options, _ = output_parser.parse_known_args(remaining)
    sys.argv = [sys.argv[0], *remaining]
    install_model_builder(options)
    install_checkpoint_saver()
    install_data_loader(options.identity_address_num_workers)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    base.main()
    augment_metrics(Path(output_options.output_dir))


if __name__ == "__main__":
    main()
