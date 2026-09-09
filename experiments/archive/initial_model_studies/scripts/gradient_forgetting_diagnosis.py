#!/usr/bin/env python3
"""Test whether future-stream gradient conflict predicts anchor forgetting.

This is a measurement-only experiment over existing WWM checkpoints. It never
updates or rewrites a model. For each pre-registered transition (40->41M,
60->61M, 80->81M), it:

1. evaluates fixed, training-only anchor masks at both checkpoint endpoints;
2. excludes anchors re-exposed in that exact 1M-word interval;
3. measures gradients for balanced anchor groups at the earlier checkpoint;
4. measures gradients for batches sampled across the subsequent training stream;
5. tests whether negative gradient alignment predicts the observed anchor-loss rise.

The script is intentionally independent of BabyLM evaluation data. It uses only
the exact official training corpus, saved example order, and direct chck_* paths.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer


STUDY = _public_path('experiments/archive/initial_model_studies')
TRAIN_FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]
CHECKPOINT_RE = re.compile(r"^chck_(\d+)M$")


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


def setup_environment(run_root: Path) -> None:
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
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)


def parse_checkpoint_list(value: str) -> list[str]:
    labels: list[str] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        label = raw if raw.startswith("chck_") else f"chck_{raw}M"
        if CHECKPOINT_RE.fullmatch(label) is None:
            raise ValueError(f"invalid checkpoint label: {raw}")
        labels.append(label)
    if not labels:
        raise ValueError("at least one checkpoint is required")
    return sorted(set(labels), key=lambda x: int(CHECKPOINT_RE.fullmatch(x).group(1)))


def parse_transitions(value: str) -> list[tuple[str, str]]:
    transitions: list[tuple[str, str]] = []
    for raw in value.split(","):
        left, right = raw.strip().split(":", 1)
        pair = parse_checkpoint_list(f"{left},{right}")
        if len(pair) != 2:
            raise ValueError(f"transition must have distinct endpoints: {raw}")
        transitions.append((pair[0], pair[1]))
    return transitions


def iter_pool_examples(raw_dir: Path, max_words: int = 10_000_000, words_per_example: int = 160):
    used = 0
    buffer: list[str] = []
    source = ""
    for name in TRAIN_FILES:
        path = raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for word in line.split():
                    if used >= max_words:
                        break
                    if not buffer:
                        source = name
                    buffer.append(word)
                    used += 1
                    if len(buffer) == words_per_example:
                        yield {"text": " ".join(buffer), "words": len(buffer), "source": source}
                        buffer = []
                        source = ""
                if used >= max_words:
                    break
        if used >= max_words:
            break
    if buffer:
        yield {"text": " ".join(buffer), "words": len(buffer), "source": source}
    if used != max_words:
        raise RuntimeError(f"training pool contains {used} selected words, expected {max_words}")


def is_word_start(token: str) -> bool:
    return token.startswith("Ġ") or token.startswith("▁")


class FixedMaskBuilder:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.special_ids = set(tokenizer.all_special_ids)
        self.word_start_cache: dict[int, bool] = {}

    def item(self, example: dict) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            example["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        groups = torch.full_like(input_ids, -1)
        group_id = -1
        for position in range(input_ids.shape[0]):
            if attention_mask[position] == 0:
                continue
            token_id = int(input_ids[position])
            if token_id in self.special_ids:
                continue
            starts = self.word_start_cache.get(token_id)
            if starts is None:
                token = str(self.tokenizer.convert_ids_to_tokens(token_id))
                starts = is_word_start(token)
                self.word_start_cache[token_id] = starts
            if group_id < 0 or starts or position == 0:
                group_id += 1
            groups[position] = group_id
        return {"input_ids": input_ids, "attention_mask": attention_mask, "word_group": groups}

    def masked_batch(self, examples: list[dict], seed: int, mask_probability: float = 0.15) -> dict:
        items = [self.item(example) for example in examples]
        input_ids = torch.stack([item["input_ids"] for item in items])
        attention_mask = torch.stack([item["attention_mask"] for item in items])
        word_group = torch.stack([item["word_group"] for item in items])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        candidate = attention_mask.bool()
        if self.special_ids:
            special_ids = torch.tensor(sorted(self.special_ids), dtype=input_ids.dtype)
            candidate &= ~torch.isin(input_ids, special_ids)
        selected = torch.zeros_like(candidate)
        for row in range(input_ids.shape[0]):
            groups = word_group[row]
            valid_groups = torch.unique(groups[groups >= 0])
            if valid_groups.numel() == 0:
                continue
            draw = torch.rand(valid_groups.numel(), generator=generator)
            chosen = valid_groups[draw < mask_probability]
            if chosen.numel():
                selected[row] = torch.isin(groups, chosen) & candidate[row]
            elif candidate[row].any():
                selected[row, candidate[row].nonzero(as_tuple=False)[0, 0]] = True
        labels = input_ids.clone()
        labels[~selected] = -100
        masked = input_ids.clone()
        replacement_draw = torch.rand(masked.shape, generator=generator)
        use_mask = selected & (replacement_draw < 0.8)
        use_random = selected & (replacement_draw >= 0.8) & (replacement_draw < 0.9)
        masked[use_mask] = self.tokenizer.mask_token_id
        if use_random.any():
            masked[use_random] = torch.randint(
                0, len(self.tokenizer), (int(use_random.sum()),), generator=generator
            )
        return {"input_ids": masked, "attention_mask": attention_mask, "labels": labels}


def concatenate_batches(batches: list[dict], indices: list[int]) -> dict:
    return {key: torch.cat([batches[index][key] for index in indices], dim=0) for key in batches[0]}


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def per_example_losses(model, batches: list[dict], device: torch.device, use_bf16: bool) -> list[float]:
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            moved = batch_to_device(batch, device)
            with autocast_context(device, use_bf16):
                logits = model(
                    input_ids=moved["input_ids"], attention_mask=moved["attention_mask"]
                ).logits
                flat = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    moved["labels"].reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape(moved["labels"].shape)
            counts = (moved["labels"] != -100).sum(dim=1).clamp_min(1)
            losses = flat.float().sum(dim=1) / counts
            values.extend(float(value) for value in losses.cpu())
    return values


def select_gradient_parameters(model, last_layers: int) -> list[tuple[str, torch.nn.Parameter]]:
    layer_count = int(model.config.num_hidden_layers)
    first_layer = max(0, layer_count - last_layers)
    selected = []
    for name, parameter in model.named_parameters():
        in_last_layer = any(
            f".encoder.layer.{index}." in name for index in range(first_layer, layer_count)
        )
        if name.startswith("cls.") or in_last_layer:
            selected.append((name, parameter))
    if not selected:
        raise RuntimeError("gradient parameter selection is empty")
    return selected


def gradient_vector(
    model,
    batch: dict,
    selected_parameters: list[tuple[str, torch.nn.Parameter]],
    device: torch.device,
    use_bf16: bool,
) -> tuple[torch.Tensor, float]:
    model.zero_grad(set_to_none=True)
    moved = batch_to_device(batch, device)
    with autocast_context(device, use_bf16):
        output = model(
            input_ids=moved["input_ids"],
            attention_mask=moved["attention_mask"],
            labels=moved["labels"],
        )
        loss = output.loss
    loss.backward()
    pieces = []
    for _, parameter in selected_parameters:
        if parameter.grad is None:
            pieces.append(torch.zeros(parameter.numel(), device=device, dtype=torch.float32))
        else:
            pieces.append(parameter.grad.detach().reshape(-1).float())
    vector = torch.cat(pieces)
    return vector, float(loss.detach().cpu())


def balanced_groups(indices: list[int], sources: list[str], count: int, seed: int) -> list[list[int]]:
    by_source: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_source[sources[index]].append(index)
    rng = random.Random(seed)
    for values in by_source.values():
        rng.shuffle(values)
    groups = [[] for _ in range(min(count, len(indices)))]
    cursor = 0
    while any(by_source.values()):
        for source in sorted(by_source):
            if by_source[source]:
                groups[cursor % len(groups)].append(by_source[source].pop())
                cursor += 1
    return [group for group in groups if group]


def sample_stream_batches(
    event_ids: list[int], pool: list[dict], builder: FixedMaskBuilder,
    count: int, batch_size: int, seed: int,
) -> list[dict]:
    if len(event_ids) < batch_size:
        raise RuntimeError(f"stream interval contains only {len(event_ids)} events")
    maximum = len(event_ids) - batch_size
    starts = np.linspace(0, maximum, num=count, dtype=int).tolist()
    batches = []
    for index, start in enumerate(starts):
        examples = [pool[event_id] for event_id in event_ids[start:start + batch_size]]
        batches.append(builder.masked_batch(examples, seed + 104729 * index))
    return batches


def checkpoint_exposure_map(metrics: dict) -> dict[str, int]:
    return {
        row["name"]: int(row["actual_cumulative_word_exposure"])
        for row in metrics["saved_checkpoints"]
    }


def checkpoint_event_end(cumulative_event_words: np.ndarray, exposure: int) -> int:
    index = int(np.searchsorted(cumulative_event_words, exposure, side="right"))
    if index <= 0:
        raise RuntimeError(f"cannot map exposure {exposure} into training stream")
    observed = int(cumulative_event_words[index - 1])
    if observed != exposure:
        raise RuntimeError(f"checkpoint exposure {exposure} is not an event boundary; nearest={observed}")
    return index


def transition_analysis(rows: list[dict], seed: int, permutations: int) -> dict:
    if len(rows) < 8:
        return {"n": len(rows), "status": "insufficient_rows"}
    interference = np.asarray([row["interference_score"] for row in rows], dtype=np.float64)
    future_delta = np.asarray([row["future_loss_delta"] for row in rows], dtype=np.float64)
    current_loss = np.asarray([row["current_loss"] for row in rows], dtype=np.float64)
    rho, asymptotic_p = spearmanr(interference, future_delta)
    raw_rho, raw_p = spearmanr(current_loss, future_delta)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        null[index] = spearmanr(interference, rng.permutation(future_delta)).statistic
    permutation_p = float((1 + np.sum(null >= rho)) / (permutations + 1))
    return {
        "n": len(rows),
        "interference_spearman_rho": float(rho),
        "interference_asymptotic_p": float(asymptotic_p),
        "interference_one_sided_permutation_p": permutation_p,
        "raw_loss_spearman_rho": float(raw_rho),
        "raw_loss_asymptotic_p": float(raw_p),
        "future_loss_delta_mean": float(future_delta.mean()),
        "future_loss_delta_positive_fraction": float((future_delta > 0).mean()),
        "null_rho_mean": float(np.nanmean(null)),
        "null_rho_q95": float(np.nanquantile(null, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--anchor-count", type=int, default=256)
    parser.add_argument("--anchor-batch-size", type=int, default=16)
    parser.add_argument("--anchor-groups", type=int, default=24)
    parser.add_argument("--stream-batches", type=int, default=24)
    parser.add_argument("--stream-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--anchor-seed", type=int, default=301042)
    parser.add_argument("--mask-seed", type=int, default=301043)
    parser.add_argument("--last-layers", type=int, default=2)
    parser.add_argument("--loss-checkpoints", default="5,10,15,20,25,30,35,40,41,45,50,55,60,61,65,70,75,80,81,85,90,95,100")
    parser.add_argument("--transitions", default="40:41,60:61,80:81")
    parser.add_argument("--permutations", type=int, default=4000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--skip-gradients", action="store_true")
    args = parser.parse_args()

    started = time.time()
    run_root = Path(args.run_root).resolve()
    model_root = run_root / "hf_model"
    manifest_path = run_root / "example_order_manifest.json"
    metrics_path = run_root / "scientific_metrics.json"
    output_path = Path(args.out_json).resolve()
    setup_environment(run_root)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise RuntimeError("CUDA is required for this diagnosis; use --allow-cpu only for a tiny smoke test")
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    event_ids = [int(value) for value in manifest["consumed_example_ids_in_order"]]
    pool = list(iter_pool_examples(run_root / "raw_dataset"))
    if max(event_ids) >= len(pool):
        raise RuntimeError("training manifest references an example outside the reconstructed pool")
    event_words = np.fromiter((pool[index]["words"] for index in event_ids), dtype=np.int64)
    cumulative_event_words = np.cumsum(event_words)
    if int(cumulative_event_words[-1]) != int(metrics["word_exposure"]):
        raise RuntimeError(
            f"reconstructed exposure {int(cumulative_event_words[-1])} != metrics {metrics['word_exposure']}"
        )

    loss_checkpoints = parse_checkpoint_list(args.loss_checkpoints)
    transitions = parse_transitions(args.transitions)
    required_checkpoints = set(loss_checkpoints)
    for left, right in transitions:
        required_checkpoints.update((left, right))
    exposure_map = checkpoint_exposure_map(metrics)
    for label in required_checkpoints:
        checkpoint = model_root / label
        if not checkpoint.is_dir() or not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"direct checkpoint missing: {checkpoint}")
        if label not in exposure_map:
            raise RuntimeError(f"checkpoint exposure missing from scientific_metrics.json: {label}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_root / loss_checkpoints[-1], use_fast=True, trust_remote_code=True
    )
    builder = FixedMaskBuilder(tokenizer, args.max_length)
    candidate_ids = [index for index, example in enumerate(pool) if example["words"] >= 80]
    rng = random.Random(args.anchor_seed)
    rng.shuffle(candidate_ids)
    anchor_ids = sorted(candidate_ids[: args.anchor_count])
    anchor_examples = [pool[index] for index in anchor_ids]
    anchor_sources = [example["source"] for example in anchor_examples]
    anchor_batches = []
    for start in range(0, len(anchor_examples), args.anchor_batch_size):
        anchor_batches.append(
            builder.masked_batch(
                anchor_examples[start:start + args.anchor_batch_size],
                args.mask_seed + start,
                args.mask_probability,
            )
        )
    anchor_single_batches = [
        {key: value[offset:offset + 1] for key, value in batch.items()}
        for batch in anchor_batches
        for offset in range(batch["input_ids"].shape[0])
    ]

    payload = {
        "status": "GRADIENT_FORGETTING_DIAGNOSIS_RUNNING",
        "run_name": args.run_name,
        "run_root": str(run_root),
        "model_root": str(model_root),
        "design": {
            "claim_under_test": "Negative alignment between an anchor gradient and the subsequent training stream predicts loss increase on unrepeated anchors.",
            "unit": "balanced group of fixed-mask, training-only anchors not re-exposed in the measured interval",
            "transitions": [f"{left}->{right}" for left, right in transitions],
            "gradient_scope": f"last_{args.last_layers}_encoder_layers_plus_mlm_head",
            "nulls": ["current anchor loss", "permuted future-loss deltas", "cross-transition stability", "cross-seed replication"],
            "promotion_rule": "Both WWM seeds must show positive pooled rho; combined permutation p<=0.01 and predictive value beyond raw loss are required before an intervention.",
            "kill_rule": "Reject gradient conflict as the primary bottleneck when both seeds are near zero/opposite or the combined permutation null is not rejected.",
        },
        "provenance": {
            "script": str(_public_path('experiments/archive/initial_model_studies/scripts/gradient_forgetting_diagnosis.py')),
            "script_sha256": sha256_file(_public_path('experiments/archive/initial_model_studies/scripts/gradient_forgetting_diagnosis.py')),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch_version": torch.__version__,
            "bf16_autocast": bool(device.type == "cuda" and not args.no_bf16),
        },
        "training_stream": {
            "pool_examples": len(pool),
            "events": len(event_ids),
            "word_exposure": int(cumulative_event_words[-1]),
            "seed": metrics.get("seed"),
            "mask_mode": metrics.get("mask_mode"),
            "mask_probability": metrics.get("mask_prob"),
        },
        "anchors": {
            "count": len(anchor_ids),
            "ids": anchor_ids,
            "source_counts": dict(Counter(anchor_sources)),
            "fixed_mask_seed": args.mask_seed,
            "masked_token_counts": [int((batch["labels"] != -100).sum()) for batch in anchor_single_batches],
        },
        "checkpoint_losses": {},
        "checkpoint_identity": {},
        "transitions": {},
    }
    write_json(output_path, payload)

    use_bf16 = device.type == "cuda" and not args.no_bf16
    identity_labels = {label for pair in transitions for label in pair} | {"chck_80M", "chck_100M"}
    for label in loss_checkpoints:
        checkpoint = model_root / label
        model = AutoModelForMaskedLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
        losses = per_example_losses(model, anchor_batches, device, use_bf16)
        if len(losses) != len(anchor_ids):
            raise RuntimeError(f"{label}: got {len(losses)} anchor losses, expected {len(anchor_ids)}")
        payload["checkpoint_losses"][label] = losses
        if label in identity_labels:
            weight = checkpoint / "model.safetensors"
            payload["checkpoint_identity"][label] = {
                "direct_path": str(checkpoint),
                "weight_bytes": weight.stat().st_size,
                "weight_sha256": sha256_file(weight),
                "actual_word_exposure": exposure_map[label],
            }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        payload["elapsed_sec"] = time.time() - started
        write_json(output_path, payload)

    if not args.skip_gradients:
        for transition_index, (left, right) in enumerate(transitions):
            left_end = checkpoint_event_end(cumulative_event_words, exposure_map[left])
            right_end = checkpoint_event_end(cumulative_event_words, exposure_map[right])
            interval_ids = event_ids[left_end:right_end]
            seen = set(interval_ids)
            eligible_anchor_indices = [
                index for index, anchor_id in enumerate(anchor_ids) if anchor_id not in seen
            ]
            groups = balanced_groups(
                eligible_anchor_indices,
                anchor_sources,
                args.anchor_groups,
                args.anchor_seed + 1009 * transition_index,
            )
            stream_batches = sample_stream_batches(
                interval_ids,
                pool,
                builder,
                args.stream_batches,
                args.stream_batch_size,
                args.mask_seed + 1_000_003 * transition_index,
            )
            checkpoint = model_root / left
            model = AutoModelForMaskedLM.from_pretrained(checkpoint, trust_remote_code=True).to(device).eval()
            selected_parameters = select_gradient_parameters(model, args.last_layers)
            selected_parameter_count = sum(parameter.numel() for _, parameter in selected_parameters)
            anchor_vectors = []
            anchor_gradient_losses = []
            for group in groups:
                batch = concatenate_batches(anchor_single_batches, group)
                vector, gradient_loss = gradient_vector(
                    model, batch, selected_parameters, device, use_bf16
                )
                anchor_vectors.append(vector)
                anchor_gradient_losses.append(gradient_loss)
            anchor_matrix = torch.stack(anchor_vectors)
            anchor_norms = anchor_matrix.norm(dim=1).clamp_min(1e-12)
            cosine_columns = []
            stream_gradient_losses = []
            for stream_batch in stream_batches:
                vector, gradient_loss = gradient_vector(
                    model, stream_batch, selected_parameters, device, use_bf16
                )
                denominator = anchor_norms * vector.norm().clamp_min(1e-12)
                cosine_columns.append((anchor_matrix @ vector / denominator).detach().cpu())
                stream_gradient_losses.append(gradient_loss)
                del vector
            cosine = torch.stack(cosine_columns, dim=1).numpy()
            left_losses = np.asarray(payload["checkpoint_losses"][left], dtype=np.float64)
            right_losses = np.asarray(payload["checkpoint_losses"][right], dtype=np.float64)
            rows = []
            for group_index, group in enumerate(groups):
                values = cosine[group_index]
                current_loss = float(left_losses[group].mean())
                future_loss = float(right_losses[group].mean())
                rows.append({
                    "group": group_index,
                    "anchor_indices": group,
                    "anchor_ids": [anchor_ids[index] for index in group],
                    "source_counts": dict(Counter(anchor_sources[index] for index in group)),
                    "current_loss": current_loss,
                    "future_loss": future_loss,
                    "future_loss_delta": future_loss - current_loss,
                    "mean_gradient_cosine": float(values.mean()),
                    "interference_score": float(-values.mean()),
                    "negative_cosine_fraction": float((values < 0).mean()),
                    "negative_cosine_magnitude": float(np.maximum(-values, 0).mean()),
                    "anchor_gradient_loss": anchor_gradient_losses[group_index],
                })
            analysis = transition_analysis(
                rows, args.anchor_seed + 7919 * transition_index, args.permutations
            )
            payload["transitions"][f"{left}->{right}"] = {
                "left_checkpoint": left,
                "right_checkpoint": right,
                "left_actual_exposure": exposure_map[left],
                "right_actual_exposure": exposure_map[right],
                "stream_event_start": left_end,
                "stream_event_end": right_end,
                "stream_events": len(interval_ids),
                "anchors_reexposed": len(anchor_ids) - len(eligible_anchor_indices),
                "eligible_unrepeated_anchors": len(eligible_anchor_indices),
                "anchor_groups": len(groups),
                "stream_gradient_batches": len(stream_batches),
                "selected_parameter_count": selected_parameter_count,
                "selected_parameter_names": [name for name, _ in selected_parameters],
                "stream_gradient_loss_mean": float(np.mean(stream_gradient_losses)),
                "rows": rows,
                "analysis": analysis,
            }
            del model, anchor_matrix, anchor_vectors
            if device.type == "cuda":
                torch.cuda.empty_cache()
            payload["elapsed_sec"] = time.time() - started
            write_json(output_path, payload)

    all_rows = [
        row
        for transition in payload["transitions"].values()
        for row in transition.get("rows", [])
    ]
    payload["pooled_analysis"] = transition_analysis(
        all_rows, args.anchor_seed + 99991, args.permutations
    )
    payload["status"] = "GRADIENT_FORGETTING_DIAGNOSIS_COMPLETE"
    payload["elapsed_sec"] = time.time() - started
    write_json(output_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "run_name": args.run_name,
        "out_json": str(output_path),
        "pooled_analysis": payload["pooled_analysis"],
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()
