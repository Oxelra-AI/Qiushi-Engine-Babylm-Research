#!/usr/bin/env python3
"""Test whether causal use of repeated history separates the public leader.

This is a training-data-only positive-control test. It reuses the corrected
research cases, but compares the protected WWM baseline against the public
41.80 checkpoint on exactly the same counterfactual texts. The primary outcome
is not hidden-state movement. It is whether changing an earlier mention hurts
prediction of later content more than an equal-distance unrelated replacement.

No BabyLM evaluation text or labels are read.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer


STUDY = _public_path('experiments/archive/initial_model_studies')
PATH = _public_path('experiments/archive/initial_model_studies/scripts/corrected_history_content_probe.py')
RAW_DEFAULT = (
    _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/raw_dataset')
)
BASE_TOKENIZER_DEFAULT = (
    _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_80M')
)
PROTECTED_DEFAULT = (
    _public_path('experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M')
)
LEADER_DEFAULT = _public_path('experiments/archive/initial_model_studies/data/leader_package_revision_124/model_side/local')
OUT_DEFAULT = _public_path('experiments/archive/initial_model_studies/data/leader_causal_use_gap.json')
NOTE_DEFAULT = _public_path('research/notes/initial_model_studies/leader_causal_use_gap.md')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def setup_environment() -> None:
    hf_home = _public_path('experiments/archive/initial_model_studies/training/hf_home')
    os.environ.setdefault("HF_HOME", str(hf_home.resolve()))
    os.environ.setdefault("HF_HUB_CACHE", str((hf_home / "hub").resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", str((hf_home / "transformers").resolve()))
    os.environ.setdefault(
        "HF_MODULES_CACHE", str(_public_path('experiments/archive/initial_model_studies/training/hf_modules_cache'))
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")


def load_step273_module():
    spec = importlib.util.spec_from_file_location("probe", PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def encoded_condition(tokenizer, condition, max_length: int) -> dict | None:
    encoded = tokenizer(
        condition.text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")
    positions: list[int] = []
    for start, end in condition.content_spans:
        positions.extend(
            index for index, (left, right) in enumerate(offsets) if right > start and left < end
        )
    positions = sorted(set(positions))
    if not positions:
        return None
    input_ids = list(encoded["input_ids"])
    labels = [-100] * len(input_ids)
    target_ids = []
    for position in positions:
        labels[position] = input_ids[position]
        target_ids.append(input_ids[position])
        input_ids[position] = tokenizer.mask_token_id
    return {
        "input_ids": input_ids,
        "attention_mask": list(encoded["attention_mask"]),
        "labels": labels,
        "target_ids": target_ids,
        "length": len(input_ids),
    }


def pad_batch(items: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    length = max(item["length"] for item in items)
    input_ids = []
    attention_mask = []
    labels = []
    for item in items:
        padding = length - item["length"]
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        attention_mask.append(item["attention_mask"] + [0] * padding)
        labels.append(item["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def score_model(
    name: str,
    model_path: Path,
    cases: list,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.mask_token_id is None:
        raise RuntimeError(f"{name}: tokenizer has no mask token")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError(f"{name}: tokenizer has no pad/eos token")

    flat: list[tuple[int, str, dict]] = []
    condition_names = ("orig", "entity_cf", "unrelated_cf")
    for case_index, case in enumerate(cases):
        encoded = {
            condition_name: encoded_condition(
                tokenizer, getattr(case, condition_name), max_length
            )
            for condition_name in condition_names
        }
        if any(item is None for item in encoded.values()):
            continue
        target_ids = [encoded[condition_name]["target_ids"] for condition_name in condition_names]
        if target_ids[1:] != target_ids[:-1]:
            continue
        for condition_name in condition_names:
            flat.append((case_index, condition_name, encoded[condition_name]))

    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    losses: dict[int, dict[str, float]] = {}
    use_bf16 = device.type == "cuda"
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            chunk = flat[start : start + batch_size]
            batch = pad_batch([item[2] for item in chunk], pad_token_id)
            batch = {key: value.to(device) for key, value in batch.items()}
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else torch.no_grad()
            )
            with context:
                logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits
                token_losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch["labels"].reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape(batch["labels"].shape)
            counts = (batch["labels"] != -100).sum(dim=1).clamp_min(1)
            values = (token_losses.float().sum(dim=1) / counts).cpu().tolist()
            for (case_index, condition_name, _), value in zip(chunk, values):
                losses.setdefault(case_index, {})[condition_name] = float(value)

    rows = []
    for case_index, values in sorted(losses.items()):
        if set(values) != set(condition_names):
            continue
        entity_delta = values["entity_cf"] - values["orig"]
        unrelated_delta = values["unrelated_cf"] - values["orig"]
        raw_extra = entity_delta - unrelated_delta
        scale = max(1e-6, 0.5 * (abs(values["entity_cf"]) + abs(values["unrelated_cf"])))
        rows.append(
            {
                "case_index": case_index,
                "target_word": cases[case_index].target_word,
                "source": cases[case_index].source,
                "orig_loss": values["orig"],
                "entity_delta": entity_delta,
                "unrelated_delta": unrelated_delta,
                "raw_extra_entity_effect": raw_extra,
                "normalized_extra_entity_effect": raw_extra / scale,
            }
        )
    weight = model_path / "model.safetensors"
    identity = {
        "direct_path": str(model_path.resolve()),
        "weight_sha256": sha256_file(weight),
        "weight_bytes": weight.stat().st_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "tokenizer_size": len(tokenizer),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"identity": identity, "rows": rows}


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> dict:
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 1000):
        count = min(1000, repetitions - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "fraction_positive": float((values > 0).mean()),
        "bootstrap_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(RAW_DEFAULT))
    parser.add_argument("--base-tokenizer", default=str(BASE_TOKENIZER_DEFAULT))
    parser.add_argument("--protected-model", default=str(PROTECTED_DEFAULT))
    parser.add_argument("--leader-model", default=str(LEADER_DEFAULT))
    parser.add_argument("--max-docs", type=int, default=3000)
    parser.add_argument("--max-cases", type=int, default=300)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--case-seed", type=int, default=273)
    parser.add_argument("--bootstrap-seed", type=int, default=302042)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--out-json", default=str(OUT_DEFAULT))
    parser.add_argument("--out-note", default=str(NOTE_DEFAULT))
    args = parser.parse_args()

    started = time.time()
    setup_environment()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise RuntimeError("CUDA is unavailable; use --allow-cpu for the bounded fallback")
        device = torch.device("cpu")

    research = load_step273_module()
    cases = research.build_cases(
        Path(args.raw_dir).resolve(),
        args.max_docs,
        args.max_cases,
        args.case_seed,
        Path(args.base_tokenizer).resolve(),
    )
    if len(cases) < 100:
        raise RuntimeError(f"only {len(cases)} corrected cases were constructed")

    model_paths = {
        "protected_wwm42_100M": Path(args.protected_model).resolve(),
        "public_leader": Path(args.leader_model).resolve(),
    }
    scored = {
        name: score_model(name, path, cases, args.max_length, args.batch_size, device)
        for name, path in model_paths.items()
    }
    row_maps = {
        name: {row["case_index"]: row for row in result["rows"]}
        for name, result in scored.items()
    }
    common = sorted(set.intersection(*(set(rows) for rows in row_maps.values())))
    if len(common) < 100:
        raise RuntimeError(f"only {len(common)} cases are common across tokenizers")

    protected = np.asarray(
        [row_maps["protected_wwm42_100M"][index]["normalized_extra_entity_effect"] for index in common]
    )
    leader = np.asarray(
        [row_maps["public_leader"][index]["normalized_extra_entity_effect"] for index in common]
    )
    difference = leader - protected
    rng = np.random.default_rng(args.bootstrap_seed)
    stats = {
        "protected": bootstrap_mean(protected, rng, args.bootstrap_repetitions),
        "leader": bootstrap_mean(leader, rng, args.bootstrap_repetitions),
        "leader_minus_protected": bootstrap_mean(difference, rng, args.bootstrap_repetitions),
        "cross_model_spearman_rho": float(spearmanr(protected, leader).statistic),
    }
    leader_ci = stats["leader"]["bootstrap_ci95"]
    gap_ci = stats["leader_minus_protected"]["bootstrap_ci95"]
    if leader_ci[0] > 0 and gap_ci[0] > 0 and stats["leader"]["fraction_positive"] >= 0.55:
        decision = "H2_LINKED_TO_LEADER_ADVANTAGE"
        next_action = "Design one unavoidable-history prediction objective with matched nulls; do not change data or architecture in that test."
    elif leader_ci[1] <= 0 or gap_ci[1] <= 0:
        decision = "H2_NOT_LINKED_BY_POSITIVE_CONTROL_MOVE_H1"
        next_action = "Do not build another history objective. Move the main route to coverage-preserving semantic compression on a leader-grade FineWeb substrate."
    else:
        decision = "H2_POSITIVE_CONTROL_INCONCLUSIVE"
        next_action = "Permit one stronger event-typed positive-control probe only; otherwise move to H1."

    output = {
        "status": "LEADER_CAUSAL_USE_GAP_COMPLETE",
        "design": {
            "data": "official Strict-Small training text only",
            "cases": len(cases),
            "common_cases_across_tokenizers": len(common),
            "primary_measure": "(entity-history counterfactual loss effect - matched unrelated-history effect) divided by within-model content loss scale",
            "positive_control": "public 41.80 leader versus protected 40.53 WWM baseline",
            "success_rule": "Leader effect and leader-minus-protected gap must both have bootstrap CI95 lower bound > 0; leader positive fraction >= 0.55.",
        },
        "provenance": {
            "script": str(_public_path('experiments/archive/initial_model_studies/scripts/leader_causal_use_gap.py')),
            "script_sha256": sha256_file(_public_path('experiments/archive/initial_model_studies/scripts/leader_causal_use_gap.py')),
            "script": str(PATH),
            "script_sha256": sha256_file(PATH),
            "raw_dir": str(Path(args.raw_dir).resolve()),
            "base_tokenizer": str(Path(args.base_tokenizer).resolve()),
            "device": str(device),
        },
        "models": {name: {"identity": result["identity"]} for name, result in scored.items()},
        "statistics": stats,
        "common_rows": [
            {
                "case_index": index,
                "target_word": cases[index].target_word,
                "source": cases[index].source,
                "protected_normalized_extra": float(protected[position]),
                "leader_normalized_extra": float(leader[position]),
                "leader_minus_protected": float(difference[position]),
            }
            for position, index in enumerate(common)
        ],
        "decision": decision,
        "next_action": next_action,
        "elapsed_sec": time.time() - started,
    }
    out_json = Path(args.out_json).resolve()
    out_note = Path(args.out_note).resolve()
    write_json(out_json, output)
    out_note.parent.mkdir(parents=True, exist_ok=True)
    out_note.write_text(
        "\n".join(
            [
                "# research leader causal-use positive control",
                "",
                f"Decision: **{decision}**",
                "",
                f"Cases common across tokenizers: {len(common)}",
                "",
                "| measure | mean | 95% bootstrap CI | fraction > 0 |",
                "|---|---:|---:|---:|",
                f"| protected | {stats['protected']['mean']:+.6f} | [{stats['protected']['bootstrap_ci95'][0]:+.6f}, {stats['protected']['bootstrap_ci95'][1]:+.6f}] | {stats['protected']['fraction_positive']:.3f} |",
                f"| public leader | {stats['leader']['mean']:+.6f} | [{stats['leader']['bootstrap_ci95'][0]:+.6f}, {stats['leader']['bootstrap_ci95'][1]:+.6f}] | {stats['leader']['fraction_positive']:.3f} |",
                f"| leader - protected | {stats['leader_minus_protected']['mean']:+.6f} | [{stats['leader_minus_protected']['bootstrap_ci95'][0]:+.6f}, {stats['leader_minus_protected']['bootstrap_ci95'][1]:+.6f}] | {stats['leader_minus_protected']['fraction_positive']:.3f} |",
                "",
                f"Cross-model per-case Spearman rho: {stats['cross_model_spearman_rho']:+.4f}",
                "",
                f"Next action: {next_action}",
                "",
                f"Evidence JSON: `{out_json}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": output["status"], "decision": decision, "statistics": stats, "out_json": str(out_json), "elapsed_sec": output["elapsed_sec"]}, indent=2))


if __name__ == "__main__":
    main()
