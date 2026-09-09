#!/usr/bin/env python3
"""research gate for cross-view transfer gain per source word.

For an unseen source/rewrite pair, take one bounded gradient step on masked
source tokens and measure the rewrite loss change.  Compare the true rewrite
with (a) an exact-word-multiset order-destroyed control and (b) a cross-pair
rewrite permutation.  The experiment uses no BabyLM evaluation data.

The proposed data-selection quantity is not raw loss.  It is the marginal loss
reduction on a distinct semantic view per source word.  This script only tests
whether that quantity is measurable and relation-specific in two WWM seeds; it
does not yet train a BabyLM candidate.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer


STUDY = _public_path('experiments/archive/initial_model_studies')
H3_SCRIPT = _public_path('experiments/archive/initial_model_studies/scripts/gradient_forgetting_diagnosis.py')
PAIR_DATA = (
    _public_path('experiments/archive/initial_model_studies/data/aligned_rewrite_revision_150/aligned_w100000_n2091_sel15001_shuf15002_side128-128.jsonl')
)
RUNS = {
    "wwm_seed42_60M": _public_path('experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_60M'),
    "wwm_seed43_60M": _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_60M'),
}
OUT = _public_path('experiments/archive/initial_model_studies/data/transfer_gain_per_word_gate.json')
NOTE = _public_path('research/notes/initial_model_studies/transfer_gain_per_word_gate.md')


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized_words(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)
    }


def jaccard(left: str, right: str) -> float:
    a = normalized_words(left)
    b = normalized_words(right)
    return len(a & b) / max(1, len(a | b))


def destroy_order_same_bag(text: str) -> str:
    words = text.split()
    if len(words) < 9:
        return " ".join(reversed(words))
    first = len(words) // 3
    second = 2 * len(words) // 3
    chunks = [words[:first], words[first:second], words[second:]]
    return " ".join(chunks[2] + chunks[0] + chunks[1])


def load_pairs(count: int, seed: int) -> list[dict]:
    candidates = []
    with PAIR_DATA.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            words = row["text"].split()
            source_words = int(row["source_words"])
            target_words = int(row["target_words"])
            if len(words) != source_words + target_words:
                continue
            source = " ".join(words[:source_words])
            target = " ".join(words[source_words:])
            overlap = jaccard(source, target)
            if not (14 <= source_words <= 70 and 10 <= target_words <= 55):
                continue
            if not (0.18 <= overlap <= 0.72):
                continue
            candidates.append(
                {
                    "source_pair_id": row["source_pair_id"],
                    "source": source,
                    "target": target,
                    "same_bag_control": destroy_order_same_bag(target),
                    "source_words": source_words,
                    "target_words": target_words,
                    "content_jaccard": overlap,
                }
            )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:count]
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)}/{count} eligible pairs")
    # Exact target multiset is preserved across the cross-pair arm.
    shift = max(1, count // 2)
    targets = [row["target"] for row in selected]
    for index, row in enumerate(selected):
        row["cross_pair_control"] = targets[(index + shift) % count]
        row["cross_pair_jaccard"] = jaccard(row["source"], row["cross_pair_control"])
    return selected


def mean_loss(model, batch: dict, device: torch.device, grad: bool = False):
    moved = {key: value.to(device) for key, value in batch.items()}
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        output = model(
            input_ids=moved["input_ids"],
            attention_mask=moved["attention_mask"],
            labels=moved["labels"],
        )
    return output.loss


def gradient_step(
    model,
    source_batch: dict,
    selected_parameters: list[tuple[str, torch.nn.Parameter]],
    device: torch.device,
    learning_rate: float,
    max_norm: float,
):
    model.zero_grad(set_to_none=True)
    source_loss = mean_loss(model, source_batch, device, grad=True)
    source_loss.backward()
    squared = torch.zeros((), device=device)
    for _, parameter in selected_parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().pow(2).sum()
    gradient_norm = float(torch.sqrt(squared).detach().cpu())
    coefficient = min(1.0, max_norm / max(gradient_norm, 1e-12))
    updates = []
    with torch.no_grad():
        for _, parameter in selected_parameters:
            if parameter.grad is None:
                delta = torch.zeros_like(parameter)
            else:
                delta = -learning_rate * coefficient * parameter.grad
            parameter.add_(delta)
            updates.append(delta)
    model.zero_grad(set_to_none=True)
    return float(source_loss.detach().cpu()), gradient_norm, updates


def restore(selected_parameters, updates) -> None:
    with torch.no_grad():
        for (_, parameter), delta in zip(selected_parameters, updates):
            parameter.sub_(delta)


def bootstrap(values: np.ndarray, seed: int, repetitions: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        chosen = rng.integers(0, len(values), size=len(values))
        samples[index] = values[chosen].mean()
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "fraction_positive": float((values > 0).mean()),
        "bootstrap_ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def run_model(
    run_name: str,
    checkpoint: Path,
    pairs: list[dict],
    h3,
    device: torch.device,
    learning_rate: float,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)
    model.eval()
    selected_parameters = h3.select_gradient_parameters(model, last_layers=1)
    selected_ids = {id(parameter) for _, parameter in selected_parameters}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
    builder = h3.FixedMaskBuilder(tokenizer, max_length=128)

    source_batches = []
    target_batches = {"aligned": [], "same_bag": [], "cross_pair": []}
    for index, pair in enumerate(pairs):
        source_batches.append(
            builder.masked_batch(
                [{"text": pair["source"]}] * 4,
                seed=308100 + index,
                mask_probability=0.30,
            )
        )
        for offset, (name, key) in enumerate(
            [
                ("aligned", "target"),
                ("same_bag", "same_bag_control"),
                ("cross_pair", "cross_pair_control"),
            ]
        ):
            target_batches[name].append(
                builder.masked_batch(
                    [{"text": pair[key]}] * 4,
                    seed=308500 + 1009 * index + offset,
                    mask_probability=0.20,
                )
            )

    baseline = {
        name: [
            float(mean_loss(model, batch, device).detach().cpu()) for batch in batches
        ]
        for name, batches in target_batches.items()
    }
    rows = []
    for index, pair in enumerate(pairs):
        source_loss, gradient_norm, updates = gradient_step(
            model,
            source_batches[index],
            selected_parameters,
            device,
            learning_rate,
            max_norm=1.0,
        )
        try:
            source_after = float(mean_loss(model, source_batches[index], device).detach().cpu())
            after = {
                name: float(mean_loss(model, batches[index], device).detach().cpu())
                for name, batches in target_batches.items()
            }
        finally:
            restore(selected_parameters, updates)
        gains = {name: baseline[name][index] - after[name] for name in baseline}
        rows.append(
            {
                "pair_index": index,
                "source_pair_id": pair["source_pair_id"],
                "source_words": pair["source_words"],
                "target_words": pair["target_words"],
                "content_jaccard": pair["content_jaccard"],
                "cross_pair_jaccard": pair["cross_pair_jaccard"],
                "source_loss": source_loss,
                "source_self_gain": source_loss - source_after,
                "gradient_norm": gradient_norm,
                "baseline_target_loss": {name: baseline[name][index] for name in baseline},
                "transfer_gain": gains,
                "aligned_gain_per_100_source_words": 100.0
                * gains["aligned"]
                / pair["source_words"],
            }
        )
        print(
            json.dumps(
                {
                    "event": "pair_complete",
                    "run": run_name,
                    "pair": index + 1,
                    "of": len(pairs),
                    "aligned_gain": gains["aligned"],
                }
            ),
            flush=True,
        )

    arrays = {
        name: np.asarray([row["transfer_gain"][name] for row in rows], dtype=np.float64)
        for name in ["aligned", "same_bag", "cross_pair"]
    }
    contrasts = {
        "aligned_minus_same_bag": bootstrap(
            arrays["aligned"] - arrays["same_bag"], 308042
        ),
        "aligned_minus_cross_pair": bootstrap(
            arrays["aligned"] - arrays["cross_pair"], 308043
        ),
    }
    statistics = {name: bootstrap(values, 308100 + index) for index, (name, values) in enumerate(arrays.items())}
    source_loss = [row["source_loss"] for row in rows]
    self_gain = [row["source_self_gain"] for row in rows]
    aligned_gain = arrays["aligned"].tolist()
    predictors = {
        "raw_source_loss_spearman_with_aligned_transfer": float(
            spearmanr(source_loss, aligned_gain).statistic
        ),
        "source_self_gain_spearman_with_aligned_transfer": float(
            spearmanr(self_gain, aligned_gain).statistic
        ),
        "content_jaccard_spearman_with_aligned_transfer": float(
            spearmanr([row["content_jaccard"] for row in rows], aligned_gain).statistic
        ),
    }
    identity = {
        "direct_path": str(checkpoint.resolve()),
        "weight_sha256": sha256_file(checkpoint / "model.safetensors"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "updated_parameter_count": sum(
            parameter.numel() for _, parameter in selected_parameters
        ),
        "tokenizer_sha256": sha256_file(checkpoint / "tokenizer.json"),
    }
    del model
    return {
        "identity": identity,
        "statistics": statistics,
        "contrasts": contrasts,
        "predictors": predictors,
        "rows": rows,
    }


def main() -> None:
    started = time.time()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    torch.set_num_threads(4)
    h3 = load_module(H3_SCRIPT, "for_step308")
    pairs = load_pairs(24, 30801)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.01
    results = {}
    for run_name, checkpoint in RUNS.items():
        print(json.dumps({"event": "run_start", "run": run_name}), flush=True)
        results[run_name] = run_model(
            run_name, checkpoint, pairs, h3, device, learning_rate
        )
        print(json.dumps({"event": "run_complete", "run": run_name}), flush=True)

    passed = []
    for run_name, result in results.items():
        relation_specific = (
            result["contrasts"]["aligned_minus_same_bag"]["bootstrap_ci95"][0] > 0
            and result["contrasts"]["aligned_minus_cross_pair"]["bootstrap_ci95"][0] > 0
            and result["statistics"]["aligned"]["fraction_positive"] >= 0.60
        )
        passed.append(relation_specific)
    if all(passed):
        decision = "PROMOTE_TRANSFER_GAIN_PER_WORD_TO_NATURAL_DATA_SELECTOR"
        next_action = (
            "Estimate transfer gain on diverse semantic clusters, residualize lexical overlap, "
            "and select a fixed 10M-word corpus against raw-loss and shuffled-utility controls."
        )
    else:
        decision = "REJECT_CURRENT_TRANSFER_GAIN_PER_WORD_ESTIMATOR"
        next_action = (
            "Do not build a corpus selector from this estimator; identify which control failed "
            "before proposing another data-allocation mechanism."
        )

    payload = {
        "status": "TRANSFER_GAIN_PER_WORD_GATE_COMPLETE",
        "design": {
            "principle": "marginal cross-view loss reduction per source word",
            "pairs": len(pairs),
            "pair_source": str(_public_path('experiments/archive/initial_model_studies/data/aligned_rewrite_revision_150/aligned_w100000_n2091_sel15001_shuf15002_side128-128.jsonl')),
            "controls": {
                "same_bag": "same target word multiset with three contiguous blocks reordered",
                "cross_pair": "deranged target assignment preserving the aggregate target multiset",
            },
            "checkpoints": {name: str(path.resolve()) for name, path in RUNS.items()},
            "updated_parameters": "last encoder layer plus MLM head",
            "source_mask_probability": 0.30,
            "target_mask_probability": 0.20,
            "mask_replicates": 4,
            "gradient_clip_norm": 1.0,
            "sgd_learning_rate": learning_rate,
            "device": str(device),
            "no_babylm_eval_data": True,
        },
        "provenance": {
            "script": str(_public_path('experiments/archive/initial_model_studies/scripts/transfer_gain_per_word_gate.py')),
            "script_sha256": sha256_file(_public_path('experiments/archive/initial_model_studies/scripts/transfer_gain_per_word_gate.py')),
            "gradient_helpers": str(_public_path('experiments/archive/initial_model_studies/scripts/gradient_forgetting_diagnosis.py')),
            "gradient_helpers_sha256": sha256_file(H3_SCRIPT),
            "pair_data_sha256": sha256_file(PAIR_DATA),
        },
        "pair_manifest": [
            {
                key: row[key]
                for key in [
                    "source_pair_id",
                    "source_words",
                    "target_words",
                    "content_jaccard",
                    "cross_pair_jaccard",
                ]
            }
            for row in pairs
        ],
        "results": results,
        "decision": decision,
        "next_action": next_action,
        "elapsed_sec": time.time() - started,
    }
    atomic_json(OUT, payload)

    lines = [
        "# research transfer gain per word gate",
        "",
        f"Decision: **{decision}**",
        "",
        "| checkpoint | aligned gain | aligned-same-bag CI95 | aligned-cross-pair CI95 | raw-loss rho |",
        "|---|---:|---:|---:|---:|",
    ]
    for run_name, result in results.items():
        aligned = result["statistics"]["aligned"]
        same = result["contrasts"]["aligned_minus_same_bag"]["bootstrap_ci95"]
        cross = result["contrasts"]["aligned_minus_cross_pair"]["bootstrap_ci95"]
        raw_rho = result["predictors"]["raw_source_loss_spearman_with_aligned_transfer"]
        lines.append(
            f"| {run_name} | {aligned['mean']:+.8f} | "
            f"[{same[0]:+.8f}, {same[1]:+.8f}] | "
            f"[{cross[0]:+.8f}, {cross[1]:+.8f}] | {raw_rho:+.3f} |"
        )
    lines.extend(
        [
            "",
            f"Next action: {next_action}",
            "",
            f"Evidence JSON: `{OUT}`",
        ]
    )
    _public_path('research/notes/initial_model_studies').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": decision,
                "summary": {
                    run: {
                        "statistics": result["statistics"],
                        "contrasts": result["contrasts"],
                        "predictors": result["predictors"],
                    }
                    for run, result in results.items()
                },
                "out": str(OUT),
                "elapsed_sec": payload["elapsed_sec"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
