#!/usr/bin/env python3
"""research: gate for within-experience transfer credit.

The candidate principle is that a useful learning event should improve a
lexically disjoint prediction target in the same natural passage.  For every
official-training passage, this script creates two disjoint whole-word mask
sets A and B.  It takes one bounded update on A and measures the loss reduction
on B.  Exact-word-bag block reordering and a deranged passage are controls.

This is a read-only checkpoint diagnostic.  It uses no BabyLM evaluation data
and does not claim a trained-model improvement.
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
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer


STUDY = _public_path('experiments/archive/initial_model_studies')
WORKSPACE = _public_path('experiments/archive/initial_model_studies')
H3_SCRIPT = _public_path('experiments/archive/initial_model_studies/scripts/gradient_forgetting_diagnosis.py')
RAW_DIR = _public_path('experiments/archive/initial_model_studies/data/reconstruct_tmp/raw_dataset')
RUNS = {
    "wwm_seed42_60M": _public_path('experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_60M'),
    "wwm_seed43_60M": _public_path('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_60M'),
}
OUT_DIR = _public_path('experiments/archive/initial_model_studies/data/within_experience_transfer_credit')
OUT = _public_path('experiments/archive/initial_model_studies/data/within_experience_transfer_credit.json')
NOTE = _public_path('research/notes/initial_model_studies/within_experience_transfer_credit.md')

STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "because", "before",
    "being", "between", "could", "does", "doing", "down", "during", "each",
    "from", "further", "have", "having", "here", "hers", "himself", "into",
    "itself", "more", "most", "other", "ours", "over", "same", "should",
    "some", "such", "than", "that", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "under",
    "until", "very", "what", "when", "where", "which", "while", "who",
    "whom", "with", "would", "your", "yours",
}


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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def is_word_start(token: str) -> bool:
    return token.startswith("▁") or token.startswith("Ġ")


def token_groups(tokenizer, text: str, max_length: int) -> list[dict]:
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    groups: list[list[int]] = []
    for token_id in ids:
        token = str(tokenizer.convert_ids_to_tokens(int(token_id)))
        if not groups or is_word_start(token):
            groups.append([int(token_id)])
        else:
            groups[-1].append(int(token_id))
    records = []
    for index, group_ids in enumerate(groups):
        surface = tokenizer.decode(group_ids, skip_special_tokens=True).strip()
        normalized = surface.lower()
        records.append(
            {
                "original_index": index,
                "ids": group_ids,
                "surface": surface,
                "normalized": normalized,
            }
        )
    return records


def choose_masks(groups: list[dict], seed: int, groups_per_view: int) -> tuple[list[int], list[int]] | None:
    counts = Counter(group["normalized"] for group in groups)
    candidates = [
        group["original_index"]
        for group in groups
        if counts[group["normalized"]] == 1
        and re.fullmatch(r"[A-Za-z][A-Za-z'-]{3,}", group["normalized"])
        and group["normalized"] not in STOPWORDS
        and 1 <= len(group["ids"]) <= 3
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    left: list[int] = []
    right: list[int] = []
    left_ids: set[int] = set()
    right_ids: set[int] = set()
    for group_index in candidates:
        token_ids = set(groups[group_index]["ids"])
        if len(left) < groups_per_view and not (token_ids & right_ids):
            left.append(group_index)
            left_ids.update(token_ids)
            continue
        if len(right) < groups_per_view and not (token_ids & left_ids):
            right.append(group_index)
            right_ids.update(token_ids)
        if len(left) == groups_per_view and len(right) == groups_per_view:
            break
    if len(left) != groups_per_view or len(right) != groups_per_view:
        return None
    if left_ids & right_ids:
        raise AssertionError("A/B target token ids are not disjoint")
    return sorted(left), sorted(right)


def block_reordered(groups: list[dict]) -> list[dict]:
    first = len(groups) // 3
    second = 2 * len(groups) // 3
    return groups[second:] + groups[:first] + groups[first:second]


def masked_batch(
    tokenizer,
    groups: list[dict],
    selected_original_indices: list[int],
    max_length: int,
) -> dict[str, torch.Tensor]:
    selected = set(selected_original_indices)
    input_ids: list[int] = []
    labels: list[int] = []
    for group in groups:
        group_selected = group["original_index"] in selected
        for token_id in group["ids"]:
            input_ids.append(tokenizer.mask_token_id if group_selected else token_id)
            labels.append(token_id if group_selected else -100)
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    attention = [1] * len(input_ids)
    padding = max_length - len(input_ids)
    input_ids.extend([tokenizer.pad_token_id] * padding)
    labels.extend([-100] * padding)
    attention.extend([0] * padding)
    if not any(label != -100 for label in labels):
        raise RuntimeError("masked batch has no targets")
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def reservoir_passages(h3, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    reservoir: list[dict] = []
    for index, example in enumerate(
        h3.iter_pool_examples(RAW_DIR, max_words=10_000_000, words_per_example=160)
    ):
        record = dict(example)
        record["pool_index"] = index
        if len(reservoir) < count:
            reservoir.append(record)
        else:
            chosen = rng.randint(0, index)
            if chosen < count:
                reservoir[chosen] = record
    rng.shuffle(reservoir)
    return reservoir


def build_cases(tokenizer, h3, count: int, max_length: int, groups_per_view: int) -> list[dict]:
    pool = reservoir_passages(h3, max(2000, count * 30), 31001)
    cases = []
    for example in pool:
        groups = token_groups(tokenizer, example["text"], max_length)
        masks = choose_masks(groups, 310_000 + example["pool_index"], groups_per_view)
        if masks is None or len(groups) < 45:
            continue
        left, right = masks
        cases.append(
            {
                "pool_index": example["pool_index"],
                "source": example["source"],
                "text_sha256": sha256_text(example["text"]),
                "groups": groups,
                "mask_a": left,
                "mask_b": right,
                "mask_a_words": [groups[index]["normalized"] for index in left],
                "mask_b_words": [groups[index]["normalized"] for index in right],
            }
        )
        if len(cases) == count:
            break
    if len(cases) != count:
        raise RuntimeError(f"only {len(cases)}/{count} eligible official passages")
    return cases


def mean_loss(model, batch: dict, device: torch.device, grad: bool = False):
    moved = {key: value.to(device) for key, value in batch.items()}
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        return model(**moved).loss


def gradient_step(model, batch, parameters, device, learning_rate: float, max_norm: float):
    model.zero_grad(set_to_none=True)
    loss = mean_loss(model, batch, device, grad=True)
    loss.backward()
    squared = torch.zeros((), device=device)
    for _, parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().pow(2).sum()
    gradient_norm = float(torch.sqrt(squared).cpu())
    coefficient = min(1.0, max_norm / max(gradient_norm, 1e-12))
    updates = []
    with torch.no_grad():
        for _, parameter in parameters:
            delta = (
                torch.zeros_like(parameter)
                if parameter.grad is None
                else -learning_rate * coefficient * parameter.grad
            )
            parameter.add_(delta)
            updates.append(delta)
    model.zero_grad(set_to_none=True)
    return float(loss.detach().cpu()), gradient_norm, updates


def restore(parameters, updates) -> None:
    with torch.no_grad():
        for (_, parameter), delta in zip(parameters, updates):
            parameter.sub_(delta)


def correlation(left: list[float], right: list[float]) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def run_checkpoint(run_name: str, count: int, max_length: int, groups_per_view: int) -> dict:
    started = time.time()
    checkpoint = RUNS[run_name]
    h3 = load_module(H3_SCRIPT, f"for_step310_{run_name}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    cases = build_cases(tokenizer, h3, count, max_length, groups_per_view)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)
    model.eval()
    parameters = h3.select_gradient_parameters(model, last_layers=1)
    selected_ids = {id(parameter) for _, parameter in parameters}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)

    batches = []
    for case in cases:
        batches.append(
            {
                "a": masked_batch(tokenizer, case["groups"], case["mask_a"], max_length),
                "aligned": masked_batch(tokenizer, case["groups"], case["mask_b"], max_length),
                "same_bag": masked_batch(
                    tokenizer, block_reordered(case["groups"]), case["mask_b"], max_length
                ),
            }
        )
    shift = max(1, count // 2)
    for index in range(count):
        batches[index]["cross_passage"] = batches[(index + shift) % count]["aligned"]

    baseline = {
        name: [float(mean_loss(model, row[name], device).cpu()) for row in batches]
        for name in ("aligned", "same_bag", "cross_passage")
    }
    rows = []
    for index, case in enumerate(cases):
        source_loss, gradient_norm, updates = gradient_step(
            model, batches[index]["a"], parameters, device, learning_rate=0.01, max_norm=1.0
        )
        try:
            after = {
                name: float(mean_loss(model, batches[index][name], device).cpu())
                for name in ("aligned", "same_bag", "cross_passage")
            }
        finally:
            restore(parameters, updates)
        gains = {name: baseline[name][index] - after[name] for name in after}
        rows.append(
            {
                "pool_index": case["pool_index"],
                "source": case["source"],
                "text_sha256": case["text_sha256"],
                "mask_a_words": case["mask_a_words"],
                "mask_b_words": case["mask_b_words"],
                "source_loss": source_loss,
                "gradient_norm": gradient_norm,
                "baseline_loss": {name: baseline[name][index] for name in baseline},
                "transfer_gain": gains,
            }
        )
        if (index + 1) % 5 == 0 or index + 1 == count:
            print(
                json.dumps(
                    {
                        "event": "case_complete",
                        "run": run_name,
                        "case": index + 1,
                        "of": count,
                        "aligned_gain": gains["aligned"],
                    }
                ),
                flush=True,
            )

    arrays = {
        name: np.asarray([row["transfer_gain"][name] for row in rows], dtype=np.float64)
        for name in ("aligned", "same_bag", "cross_passage")
    }
    statistics = {
        name: bootstrap(values, 310_100 + offset)
        for offset, (name, values) in enumerate(arrays.items())
    }
    contrasts = {
        "aligned_minus_same_bag": bootstrap(
            arrays["aligned"] - arrays["same_bag"], 310_201
        ),
        "aligned_minus_cross_passage": bootstrap(
            arrays["aligned"] - arrays["cross_passage"], 310_202
        ),
    }
    predictors = {
        "raw_source_loss_spearman_with_aligned_transfer": correlation(
            [row["source_loss"] for row in rows], arrays["aligned"].tolist()
        ),
        "gradient_norm_spearman_with_aligned_transfer": correlation(
            [row["gradient_norm"] for row in rows], arrays["aligned"].tolist()
        ),
    }
    payload = {
        "status": "CHECKPOINT_COMPLETE",
        "run_name": run_name,
        "identity": {
            "direct_path": str(checkpoint.resolve()),
            "weight_sha256": sha256_file(checkpoint / "model.safetensors"),
            "tokenizer_sha256": sha256_file(checkpoint / "tokenizer.json"),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "updated_parameter_count": sum(parameter.numel() for _, parameter in parameters),
        },
        "design": {
            "official_training_only": True,
            "babylm_evaluation_data_used": False,
            "passage_count": count,
            "max_length": max_length,
            "whole_word_groups_per_view": groups_per_view,
            "a_b_target_word_types_disjoint": True,
            "a_b_target_token_ids_disjoint": True,
            "same_bag_control": "exact token/word groups, three-block cyclic reorder",
            "cross_passage_control": "fixed half-sample derangement",
            "updated_parameters": "last encoder layer plus MLM head",
            "learning_rate": 0.01,
            "gradient_clip_norm": 1.0,
            "device": str(device),
        },
        "statistics": statistics,
        "contrasts": contrasts,
        "predictors": predictors,
        "rows": rows,
        "elapsed_sec": time.time() - started,
    }
    atomic_json(OUT_DIR / f"{run_name}.json", payload)
    return payload


def aggregate() -> dict:
    shards = {
        run_name: json.loads((OUT_DIR / f"{run_name}.json").read_text(encoding="utf-8"))
        for run_name in RUNS
    }
    manifests = [
        [(row["pool_index"], row["text_sha256"]) for row in shard["rows"]]
        for shard in shards.values()
    ]
    if any(manifest != manifests[0] for manifest in manifests[1:]):
        raise RuntimeError("checkpoint shards used different passage manifests")

    per_run_pass = {}
    for run_name, shard in shards.items():
        primary = shard["contrasts"]["aligned_minus_same_bag"]
        cross = shard["contrasts"]["aligned_minus_cross_passage"]
        aligned = shard["statistics"]["aligned"]
        per_run_pass[run_name] = (
            primary["bootstrap_ci95"][0] > 0
            and primary["fraction_positive"] >= 0.58
            and cross["bootstrap_ci95"][0] > 0
            and aligned["fraction_positive"] >= 0.80
        )
    if all(per_run_pass.values()):
        decision = "PROMOTE_WITHIN_EXPERIENCE_TRANSFER_CREDIT_TO_TRAINING_GATE"
        next_action = (
            "Implement a matched late-phase two-view meta-objective with standard MLM logits; "
            "compare true within-passage credit with shuffled-credit and ordinary WWM controls."
        )
    else:
        decision = "REJECT_CURRENT_WITHIN_EXPERIENCE_TRANSFER_CREDIT_SIGNAL"
        next_action = (
            "Do not build the meta-objective from this measurement. Reopen the bottleneck at "
            "representation formation rather than manufacturing another contrastive loss."
        )
    payload = {
        "status": "AGGREGATE_COMPLETE",
        "hypothesis": (
            "A bounded update on one set of masked content words improves a lexically disjoint "
            "set in the same natural experience beyond exact-word-bag reorder and cross-passage controls."
        ),
        "pre_registered_success_rule_each_seed": {
            "aligned_minus_same_bag_ci95_lower": "> 0",
            "aligned_minus_same_bag_fraction_positive": ">= 0.58",
            "aligned_minus_cross_passage_ci95_lower": "> 0",
            "aligned_gain_fraction_positive": ">= 0.80",
        },
        "per_run_pass": per_run_pass,
        "decision": decision,
        "next_action": next_action,
        "runs": shards,
        "provenance": {
            "script": str(_public_path('experiments/archive/initial_model_studies/scripts/within_experience_transfer_credit_gate.py')),
            "script_sha256": sha256_file(_public_path('experiments/archive/initial_model_studies/scripts/within_experience_transfer_credit_gate.py')),
            "raw_files": {
                name: sha256_file(RAW_DIR / name)
                for name in (
                    "bnc_spoken.train.txt",
                    "childes.train.txt",
                    "gutenberg.train.txt",
                    "open_subtitles.train.txt",
                    "simple_wiki.train.txt",
                    "switchboard.train.txt",
                )
            },
        },
    }
    atomic_json(OUT, payload)

    lines = [
        "# research within-experience transfer-credit gate",
        "",
        f"Decision: **{decision}**",
        "",
        "One bounded update used mask set A; evaluation used lexically disjoint mask set B.",
        "No BabyLM evaluation text or labels were used.",
        "",
        "| checkpoint | aligned gain | aligned-same-bag mean (CI95) | positive | aligned-cross mean (CI95) | pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run_name, shard in shards.items():
        aligned = shard["statistics"]["aligned"]
        primary = shard["contrasts"]["aligned_minus_same_bag"]
        cross = shard["contrasts"]["aligned_minus_cross_passage"]
        pci = primary["bootstrap_ci95"]
        cci = cross["bootstrap_ci95"]
        lines.append(
            f"| {run_name} | {aligned['mean']:+.8f} | {primary['mean']:+.8f} "
            f"([{pci[0]:+.8f}, {pci[1]:+.8f}]) | {primary['fraction_positive']:.3f} | "
            f"{cross['mean']:+.8f} ([{cci[0]:+.8f}, {cci[1]:+.8f}]) | "
            f"{per_run_pass[run_name]} |"
        )
    lines.extend(
        [
            "",
            "This gate tests whether a new credit-assignment target exists; it is not a model score.",
            "",
            f"Next action: {next_action}",
            "",
            f"Evidence JSON: `{OUT}`",
        ]
    )
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", choices=sorted(RUNS))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--groups-per-view", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)
    if args.aggregate:
        result = aggregate()
        print(json.dumps({"status": result["status"], "decision": result["decision"]}))
        return
    if not args.run_name:
        parser.error("--run-name is required unless --aggregate is used")
    result = run_checkpoint(args.run_name, args.count, args.max_length, args.groups_per_view)
    print(json.dumps({"status": result["status"], "run_name": args.run_name}))


if __name__ == "__main__":
    main()
