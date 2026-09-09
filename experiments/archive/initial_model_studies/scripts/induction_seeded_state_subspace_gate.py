#!/usr/bin/env python3
"""research: induction-seeded state-subspace mechanism gate.

Small-data transformers may spend most of training waiting for a useful
induction circuit to emerge.  This candidate gives masked positions a
metadata-free retrieval path: a repeated content cue in the local query finds
its previous occurrence, and a short continuation after that occurrence is
written into a fixed hidden-state subspace.  The edge graph is equivariant to
consistent lexical renaming and the output remains the ordinary MLM logits.

The candidate is compared with path-off, wrong-identity, and wrong-position
controls under matched initialization, parameters, batches, and supervision.
No BabyLM evaluation data is used.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import contextlib
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM


STUDY = _public_path('experiments/archive/initial_model_studies')
SOURCE = _public_path('experiments/archive/initial_model_studies/scripts/identity_edge_state_transport_gate.py')
OUT = _public_path('experiments/archive/initial_model_studies/data/induction_seeded_state_subspace_gate.json')
NOTE = _public_path('research/notes/initial_model_studies/induction_seeded_state_subspace_gate.md')
SHARD_DIR = _public_path('experiments/archive/initial_model_studies/data/induction_seeded_state_subspace_gate')

spec = importlib.util.spec_from_file_location("for_step311", SOURCE)
induction_probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = induction_probe
assert spec.loader is not None
spec.loader.exec_module(induction_probe)
binding_task = induction_probe.binding_task


STOPWORDS = {
    "the", "and", "for", "was", "were", "are", "with", "from", "into", "that",
    "this", "then", "later", "after", "everything", "when", "could", "found",
    "remained", "eventually", "before", "over", "once", "final", "place",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_token_ids(tokenizer) -> set[int]:
    accepted = set()
    for token_id in range(len(tokenizer)):
        if token_id in tokenizer.all_special_ids:
            continue
        surface = tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
        if surface.isalpha() and len(surface) >= 3 and surface not in STOPWORDS:
            accepted.add(token_id)
    return accepted


class InductionSeededMLM(nn.Module):
    """DeBERTa MLM with a hard, parameter-neutral retrieval subspace."""

    MODES = {"off", "wrong_position", "wrong_identity", "identity_continuation"}

    def __init__(
        self,
        config: DebertaV2Config,
        mode: str,
        eligible_ids: set[int],
        mask_token_id: int,
        state_dimensions: int = 32,
        local_window: int = 20,
        continuation_window: int = 10,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(mode)
        if not 0 < state_dimensions < config.hidden_size:
            raise ValueError(state_dimensions)
        self.mlm = DebertaV2ForMaskedLM(config)
        self.mode = mode
        self.runtime_mode: str | None = None
        self.eligible_ids = set(eligible_ids)
        self.mask_token_id = int(mask_token_id)
        self.state_dimensions = state_dimensions
        self.local_window = local_window
        self.continuation_window = continuation_window
        self._input_ids: torch.Tensor | None = None
        self._attention_mask: torch.Tensor | None = None
        self._last_active_fraction = 0.0
        self._hook = self.mlm.deberta.encoder.layer[0].register_forward_hook(
            self._retrieval_hook
        )

    def _retrieval_starts(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = input_ids.shape
        starts = torch.zeros((batch, length), dtype=torch.long, device=input_ids.device)
        active = torch.zeros((batch, length), dtype=torch.bool, device=input_ids.device)
        if mode == "off":
            return starts, active
        ids_rows = input_ids.detach().cpu().tolist()
        mask_rows = attention_mask.detach().cpu().tolist()
        for row, (ids, valid) in enumerate(zip(ids_rows, mask_rows)):
            for target, token_id in enumerate(ids):
                if not valid[target] or token_id != self.mask_token_id:
                    continue
                cue = None
                true_source = None
                for position in range(target - 1, max(-1, target - self.local_window - 1), -1):
                    candidate = ids[position]
                    if candidate not in self.eligible_ids:
                        continue
                    previous = [
                        prior
                        for prior in range(position - 3)
                        if valid[prior] and ids[prior] == candidate
                    ]
                    if previous:
                        cue = position
                        true_source = previous[-1]
                        break
                if cue is None or true_source is None:
                    continue
                source = true_source
                if mode == "wrong_position":
                    source = max(0, true_source - self.continuation_window)
                elif mode == "wrong_identity":
                    alternatives = []
                    cue_id = ids[cue]
                    for prior in range(cue - 3):
                        candidate = ids[prior]
                        if (
                            valid[prior]
                            and candidate in self.eligible_ids
                            and candidate != cue_id
                        ):
                            alternatives.append(prior)
                    source = alternatives[-1] if alternatives else max(0, true_source - 1)
                starts[row, target] = source
                active[row, target] = True
        return starts, active

    def _retrieval_hook(self, module, inputs, output):
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        if self._input_ids is None or self._attention_mask is None:
            return output
        mode = self.runtime_mode or self.mode
        starts, active = self._retrieval_starts(
            self._input_ids, self._attention_mask, mode
        )
        self._last_active_fraction = float(active.float().mean().detach().cpu())
        if not active.any():
            return output
        batch, length, width = hidden.shape
        offsets = torch.arange(self.continuation_window, device=hidden.device)
        positions = starts.unsqueeze(-1) + offsets
        in_range = positions < length
        positions = positions.clamp(max=length - 1)
        flat = hidden.gather(
            1, positions.reshape(batch, -1).unsqueeze(-1).expand(-1, -1, width)
        ).reshape(batch, length, self.continuation_window, width)
        source_valid = self._attention_mask.gather(1, positions.reshape(batch, -1)).reshape(
            batch, length, self.continuation_window
        ).bool()
        valid = active.unsqueeze(-1) & in_range & source_valid
        weights = valid.unsqueeze(-1).to(hidden.dtype)
        memory = (flat * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        state = F.layer_norm(
            memory[..., -self.state_dimensions :], (self.state_dimensions,)
        )
        original_state = hidden[..., -self.state_dimensions :]
        replaced_state = torch.where(active.unsqueeze(-1), state, original_state)
        transported = torch.cat(
            (hidden[..., : -self.state_dimensions], replaced_state), dim=-1
        )
        if isinstance(output, tuple):
            return (transported,) + output[1:]
        return transported

    @contextlib.contextmanager
    def use_edge_mode(self, mode: str | None):
        if mode is not None and mode not in self.MODES:
            raise ValueError(mode)
        old = self.runtime_mode
        self.runtime_mode = mode
        try:
            yield
        finally:
            self.runtime_mode = old

    def forward(self, input_ids, attention_mask, **kwargs):
        del kwargs
        self._input_ids = input_ids
        self._attention_mask = attention_mask
        try:
            output = self.mlm(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            self._input_ids = None
            self._attention_mask = None
        return SimpleNamespace(logits=output.logits)


def train_arm(
    arm: str,
    seed: int,
    config: DebertaV2Config,
    train_pairs,
    pad_id: int,
    eligible_ids: set[int],
    mask_token_id: int,
    device: torch.device,
    steps: int = 360,
    batch_size: int = 24,
):
    torch.manual_seed(seed)
    random.seed(seed)
    model = InductionSeededMLM(
        config, arm, eligible_ids, mask_token_id, state_dimensions=32
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    rng = random.Random(seed * 1009 + 311)
    losses = []
    for _ in range(steps):
        batch = [train_pairs[rng.randrange(len(train_pairs))] for _ in range(batch_size)]
        logits_a, logits_b = binding_task.batch_logits(model, batch, pad_id, device)
        labels_a = torch.tensor([pair.label_a for pair in batch], device=device)
        labels_b = torch.tensor([pair.label_b for pair in batch], device=device)
        loss = 0.5 * (
            F.cross_entropy(logits_a, labels_a) + F.cross_entropy(logits_b, labels_b)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model.eval(), {
        "loss_first20": float(np.mean(losses[:20])),
        "loss_last20": float(np.mean(losses[-20:])),
        "last_active_fraction": model._last_active_fraction,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def main() -> None:
    binding_task.setup_environment()
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(binding_task.TOKENIZER_PATH, use_fast=True)
    if tokenizer.mask_token_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer requires mask and pad tokens")
    locations, train_pairs, heldout = induction_probe.build_data(tokenizer)
    allowed = eligible_token_ids(tokenizer)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=160,
        position_buckets=64,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    arms = ["off", "wrong_position", "wrong_identity", "identity_continuation"]
    seeds = [42, 43]
    results = {}
    vectors = {}
    candidate_models = {}
    for arm in arms:
        results[arm] = {}
        vectors[arm] = {}
        for seed in seeds:
            print(json.dumps({"event": "train_start", "arm": arm, "seed": seed}), flush=True)
            model, training = train_arm(
                arm,
                seed,
                config,
                train_pairs,
                tokenizer.pad_token_id,
                allowed,
                tokenizer.mask_token_id,
                device,
            )
            evaluations = {}
            vectors[arm][str(seed)] = {}
            for split_name, pairs in heldout.items():
                summary, pair_correct, margins = induction_probe.evaluate_detailed(
                    model, pairs, tokenizer.pad_token_id, device
                )
                evaluations[split_name] = summary
                if split_name == "new_all":
                    vectors[arm][str(seed)] = {
                        "pair_correct": pair_correct,
                        "margins": margins,
                    }
            results[arm][str(seed)] = {"training": training, "evaluation": evaluations}
            print(
                json.dumps(
                    {
                        "event": "train_complete",
                        "arm": arm,
                        "seed": seed,
                        "new_all": evaluations["new_all"],
                    }
                ),
                flush=True,
            )
            if arm == "identity_continuation":
                candidate_models[seed] = model
            else:
                del model

    interventions = {}
    for seed, model in candidate_models.items():
        interventions[str(seed)] = {}
        for mode in arms:
            summary, pair_correct, margins = induction_probe.evaluate_detailed(
                model,
                heldout["new_all"],
                tokenizer.pad_token_id,
                device,
                edge_mode=mode,
            )
            interventions[str(seed)][mode] = summary
            if mode == "off":
                interventions[str(seed)]["candidate_minus_off_pair_accuracy_ci95"] = (
                    induction_probe.bootstrap_delta(
                        vectors["identity_continuation"][str(seed)]["pair_correct"],
                        pair_correct,
                        311_000 + seed,
                    )
                )
        del model

    def pair_accuracy(arm: str, seed: int, split: str = "new_all") -> float:
        return results[arm][str(seed)]["evaluation"][split]["pair_accuracy"]

    primary = {arm: [pair_accuracy(arm, seed) for seed in seeds] for arm in arms}
    controls = ["wrong_position", "wrong_identity"]
    candidate_delta = [
        pair_accuracy("identity_continuation", seed)
        - max(pair_accuracy(control, seed) for control in controls)
        for seed in seeds
    ]
    causal_drop = [
        interventions[str(seed)]["identity_continuation"]["pair_accuracy"]
        - interventions[str(seed)]["off"]["pair_accuracy"]
        for seed in seeds
    ]
    update_accuracy = [
        pair_accuracy("identity_continuation", seed, "new_all_updates") for seed in seeds
    ]
    if (
        min(primary["identity_continuation"]) >= 0.65
        and min(candidate_delta) >= 0.15
        and min(causal_drop) >= 0.15
        and min(update_accuracy) >= 0.55
    ):
        decision = "PROMOTE_INDUCTION_SEEDED_STATE_SUBSPACE_TO_NATURAL_TEXT_GATE"
        next_action = (
            "Implement the same parameter-neutral retrieval path in the protected 8x480 trainer, "
            "then compare ordinary WWM, wrong-edge, and true-edge arms on natural training text."
        )
    else:
        decision = "REJECT_CURRENT_INDUCTION_SEEDED_STATE_SUBSPACE"
        next_action = (
            "Do not scale this path. A repeated-cue continuation is insufficient; the next "
            "architecture must form reusable semantic variables rather than retrieve lexical episodes."
        )

    payload = {
        "status": "INDUCTION_SEEDED_STATE_SUBSPACE_GATE_COMPLETE",
        "design": {
            "principle": (
                "seed a renaming-equivariant induction circuit so scarce training need not wait "
                "for repeated-cue state retrieval to emerge spontaneously"
            ),
            "mechanism": (
                "at masked positions, hard-write the previous matching cue's continuation into "
                "32 of 96 hidden dimensions after encoder layer 1 of 2"
            ),
            "arms": arms,
            "matched_parameters_initialization_batches": True,
            "seeds": seeds,
            "steps": 360,
            "batch_pairs": 24,
            "train_pairs": len(train_pairs),
            "heldout_pairs": {name: len(pairs) for name, pairs in heldout.items()},
            "state_vocabulary": locations,
            "eligible_token_count": len(allowed),
            "device": str(device),
            "no_babylm_eval_data": True,
            "promotion_gate": {
                "candidate_pair_accuracy_each_seed": 0.65,
                "candidate_minus_best_active_control_each_seed": 0.15,
                "same_model_candidate_minus_off_each_seed": 0.15,
                "latest_update_pair_accuracy_each_seed": 0.55,
            },
        },
        "provenance": {
            "script": str(_public_path('experiments/archive/initial_model_studies/scripts/induction_seeded_state_subspace_gate.py')),
            "script_sha256": sha256_file(_public_path('experiments/archive/initial_model_studies/scripts/induction_seeded_state_subspace_gate.py')),
            "source_step306_sha256": sha256_file(SOURCE),
            "tokenizer": str(binding_task.TOKENIZER_PATH.resolve()),
            "tokenizer_json_sha256": sha256_file(binding_task.TOKENIZER_PATH / "tokenizer.json"),
        },
        "results": results,
        "interventions": interventions,
        "primary": {
            "new_all_pair_accuracy": primary,
            "candidate_minus_best_active_control": candidate_delta,
            "same_model_candidate_minus_off": causal_drop,
            "candidate_new_all_updates_pair_accuracy": update_accuracy,
        },
        "decision": decision,
        "next_action": next_action,
        "elapsed_sec": time.time() - started,
    }
    _public_path('experiments/archive/initial_model_studies/data').mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT)

    lines = [
        "# research induction-seeded state-subspace gate",
        "",
        f"Decision: **{decision}**",
        "",
        "The primary split changes names, objects, templates, and bindings.",
        "The prediction interface is the ordinary MLM vocabulary logits.",
        "",
        "| arm | seed 42 pair acc | seed 43 pair acc |",
        "|---|---:|---:|",
    ]
    for arm in arms:
        lines.append(f"| {arm} | {primary[arm][0]:.3f} | {primary[arm][1]:.3f} |")
    lines.extend(
        [
            "",
            f"Candidate minus best active control: {candidate_delta}",
            "",
            f"Same trained model, candidate minus path-off: {causal_drop}",
            "",
            f"Candidate latest-update pair accuracy: {update_accuracy}",
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
                "primary": payload["primary"],
                "out": str(OUT),
                "elapsed_sec": payload["elapsed_sec"],
            },
            indent=2,
        ),
        flush=True,
    )


def run_shard(arm: str, seed: int) -> dict:
    binding_task.setup_environment()
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(binding_task.TOKENIZER_PATH, use_fast=True)
    if tokenizer.mask_token_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer requires mask and pad tokens")
    locations, train_pairs, heldout = induction_probe.build_data(tokenizer)
    allowed = eligible_token_ids(tokenizer)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=160,
        position_buckets=64,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model, training = train_arm(
        arm,
        seed,
        config,
        train_pairs,
        tokenizer.pad_token_id,
        allowed,
        tokenizer.mask_token_id,
        device,
    )
    evaluations = {}
    candidate_vector = None
    for split_name, pairs in heldout.items():
        summary, pair_correct, margins = induction_probe.evaluate_detailed(
            model, pairs, tokenizer.pad_token_id, device
        )
        evaluations[split_name] = summary
        if split_name == "new_all":
            candidate_vector = pair_correct
    interventions = None
    if arm == "identity_continuation":
        interventions = {}
        for mode in ["off", "wrong_position", "wrong_identity", "identity_continuation"]:
            summary, pair_correct, margins = induction_probe.evaluate_detailed(
                model,
                heldout["new_all"],
                tokenizer.pad_token_id,
                device,
                edge_mode=mode,
            )
            interventions[mode] = summary
            if mode == "off":
                interventions["candidate_minus_off_pair_accuracy_ci95"] = (
                    induction_probe.bootstrap_delta(
                        candidate_vector, pair_correct, 311_000 + seed
                    )
                )
    payload = {
        "status": "SHARD_COMPLETE",
        "arm": arm,
        "seed": seed,
        "training": training,
        "evaluation": evaluations,
        "interventions": interventions,
        "device": str(device),
        "elapsed_sec": time.time() - started,
    }
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    path = SHARD_DIR / f"{arm}_seed{seed}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "arm": arm,
                "seed": seed,
                "new_all_pair_accuracy": evaluations["new_all"]["pair_accuracy"],
                "elapsed_sec": payload["elapsed_sec"],
            }
        ),
        flush=True,
    )
    return payload


def aggregate_shards() -> dict:
    arms = ["off", "wrong_position", "wrong_identity", "identity_continuation"]
    seeds = [42, 43]
    results = {arm: {} for arm in arms}
    interventions = {}
    for arm in arms:
        for seed in seeds:
            path = SHARD_DIR / f"{arm}_seed{seed}.json"
            shard = json.loads(path.read_text(encoding="utf-8"))
            results[arm][str(seed)] = {
                "training": shard["training"],
                "evaluation": shard["evaluation"],
            }
            if shard["interventions"] is not None:
                interventions[str(seed)] = shard["interventions"]

    def pair_accuracy(arm: str, seed: int, split: str = "new_all") -> float:
        return results[arm][str(seed)]["evaluation"][split]["pair_accuracy"]

    primary = {arm: [pair_accuracy(arm, seed) for seed in seeds] for arm in arms}
    controls = ["wrong_position", "wrong_identity"]
    candidate_delta = [
        pair_accuracy("identity_continuation", seed)
        - max(pair_accuracy(control, seed) for control in controls)
        for seed in seeds
    ]
    causal_drop = [
        interventions[str(seed)]["identity_continuation"]["pair_accuracy"]
        - interventions[str(seed)]["off"]["pair_accuracy"]
        for seed in seeds
    ]
    update_accuracy = [
        pair_accuracy("identity_continuation", seed, "new_all_updates") for seed in seeds
    ]
    if (
        min(primary["identity_continuation"]) >= 0.65
        and min(candidate_delta) >= 0.15
        and min(causal_drop) >= 0.15
        and min(update_accuracy) >= 0.55
    ):
        decision = "PROMOTE_INDUCTION_SEEDED_STATE_SUBSPACE_TO_NATURAL_TEXT_GATE"
        next_action = (
            "Implement the same parameter-neutral retrieval path in the protected 8x480 trainer, "
            "then compare ordinary WWM, wrong-edge, and true-edge arms on natural training text."
        )
    else:
        decision = "REJECT_CURRENT_INDUCTION_SEEDED_STATE_SUBSPACE"
        next_action = (
            "Do not scale this path. A repeated-cue continuation is insufficient; the next "
            "architecture must form reusable semantic variables rather than retrieve lexical episodes."
        )
    payload = {
        "status": "INDUCTION_SEEDED_STATE_SUBSPACE_GATE_COMPLETE",
        "design": {
            "principle": (
                "seed a renaming-equivariant induction circuit so scarce training need not wait "
                "for repeated-cue state retrieval to emerge spontaneously"
            ),
            "mechanism": (
                "at masked positions, hard-write the previous matching cue's continuation into "
                "32 of 96 hidden dimensions after encoder layer 1 of 2"
            ),
            "arms": arms,
            "matched_parameters_initialization_batches": True,
            "seeds": seeds,
            "steps": 360,
            "batch_pairs": 24,
            "train_pairs": 4000,
            "heldout_pairs": {
                "new_entities": 500,
                "new_templates": 500,
                "new_all": 600,
                "new_all_updates": 400,
            },
            "no_babylm_eval_data": True,
            "promotion_gate": {
                "candidate_pair_accuracy_each_seed": 0.65,
                "candidate_minus_best_active_control_each_seed": 0.15,
                "same_model_candidate_minus_off_each_seed": 0.15,
                "latest_update_pair_accuracy_each_seed": 0.55,
            },
        },
        "provenance": {
            "script": str(_public_path('experiments/archive/initial_model_studies/scripts/induction_seeded_state_subspace_gate.py')),
            "script_sha256": sha256_file(_public_path('experiments/archive/initial_model_studies/scripts/induction_seeded_state_subspace_gate.py')),
            "source_step306_sha256": sha256_file(SOURCE),
            "tokenizer": str(binding_task.TOKENIZER_PATH.resolve()),
            "tokenizer_json_sha256": sha256_file(binding_task.TOKENIZER_PATH / "tokenizer.json"),
            "shards": {
                f"{arm}_seed{seed}": sha256_file(SHARD_DIR / f"{arm}_seed{seed}.json")
                for arm in arms
                for seed in seeds
            },
        },
        "results": results,
        "interventions": interventions,
        "primary": {
            "new_all_pair_accuracy": primary,
            "candidate_minus_best_active_control": candidate_delta,
            "same_model_candidate_minus_off": causal_drop,
            "candidate_new_all_updates_pair_accuracy": update_accuracy,
        },
        "decision": decision,
        "next_action": next_action,
    }
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT)
    lines = [
        "# research induction-seeded state-subspace gate",
        "",
        f"Decision: **{decision}**",
        "",
        "The primary split changes names, objects, templates, and bindings.",
        "The prediction interface is the ordinary MLM vocabulary logits.",
        "",
        "| arm | seed 42 pair acc | seed 43 pair acc |",
        "|---|---:|---:|",
    ]
    for arm in arms:
        lines.append(f"| {arm} | {primary[arm][0]:.3f} | {primary[arm][1]:.3f} |")
    lines.extend(
        [
            "",
            f"Candidate minus best active control: {candidate_delta}",
            "",
            f"Same trained model, candidate minus path-off: {causal_drop}",
            "",
            f"Candidate latest-update pair accuracy: {update_accuracy}",
            "",
            f"Next action: {next_action}",
            "",
            f"Evidence JSON: `{OUT}`",
        ]
    )
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "decision": decision, "primary": payload["primary"]}, indent=2))
    return payload


def parallel_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=["off", "wrong_position", "wrong_identity", "identity_continuation"],
    )
    parser.add_argument("--seed", type=int, choices=[42, 43])
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate_shards()
        return
    if args.arm is None or args.seed is None:
        parser.error("--arm and --seed are required unless --aggregate is used")
    run_shard(args.arm, args.seed)


if __name__ == "__main__":
    parallel_main()
