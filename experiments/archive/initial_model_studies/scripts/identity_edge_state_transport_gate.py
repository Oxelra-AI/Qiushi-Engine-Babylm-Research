#!/usr/bin/env python3
"""research mechanism gate for identity-edge state transport.

The candidate adds one inductive bias to an otherwise standard bidirectional
DeBERTa MLM: after an early encoder layer, every repeated token can receive a
gated residual from its nearest previous occurrence.  The edge is constructed
from token identity, not a learned entity/query router.  Consequently, a
consistent renaming of entities preserves the edge graph.

This is a bounded, evaluation-free mechanism test.  It compares matched model
initializations and batches, uses unseen names/objects/templates at test time,
and includes wrong-edge and path-ablation controls.  It does not claim an
official BabyLM score improvement.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import contextlib
import hashlib
import importlib.util
import json
import math
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
SOURCE = _public_path('experiments/archive/initial_model_studies/scripts/counterfactual_binding_credit_gate.py')
OUT = _public_path('experiments/archive/initial_model_studies/data/identity_edge_state_transport_gate.json')
NOTE = _public_path('research/notes/initial_model_studies/identity_edge_state_transport_gate.md')

spec = importlib.util.spec_from_file_location("binding_gate", SOURCE)
binding_task = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = binding_task
assert spec.loader is not None
spec.loader.exec_module(binding_task)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class IdentityEdgeTransportMLM(nn.Module):
    """DeBERTa MLM with a metadata-free residual transport edge.

    All arms instantiate the same parameters.  Only the deterministic source
    rule differs, so parameter count, initialization, optimizer, and batches
    are exactly matched within each seed.
    """

    EDGE_MODES = {"off", "previous_token", "wrong_identity", "identity"}

    def __init__(
        self,
        config: DebertaV2Config,
        edge_mode: str,
        special_token_ids: set[int],
    ) -> None:
        super().__init__()
        if edge_mode not in self.EDGE_MODES:
            raise ValueError(edge_mode)
        self.mlm = DebertaV2ForMaskedLM(config)
        self.edge_mode = edge_mode
        self.runtime_edge_mode: str | None = None
        self.special_token_ids = set(special_token_ids)
        hidden = config.hidden_size
        self.value = nn.Linear(hidden, hidden, bias=False)
        nn.init.eye_(self.value.weight)
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.38629436112)  # sigmoid = 0.2
        self.transport_logit = nn.Parameter(torch.tensor(-1.38629436112))
        self._input_ids: torch.Tensor | None = None
        self._attention_mask: torch.Tensor | None = None
        self._last_active_fraction = 0.0
        self._hook_handle = self.mlm.deberta.encoder.layer[0].register_forward_hook(
            self._transport_hook
        )

    def _source_map(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = input_ids.shape
        sources = torch.zeros((batch, length), dtype=torch.long, device=input_ids.device)
        active = torch.zeros((batch, length), dtype=torch.bool, device=input_ids.device)
        if mode == "off":
            return sources, active

        ids_cpu = input_ids.detach().cpu().tolist()
        mask_cpu = attention_mask.detach().cpu().tolist()
        for row in range(batch):
            last_by_id: dict[int, int] = {}
            for position, token_id in enumerate(ids_cpu[row]):
                if not mask_cpu[row][position] or token_id in self.special_token_ids:
                    continue
                if token_id in last_by_id:
                    true_source = last_by_id[token_id]
                    source = true_source
                    if mode == "previous_token":
                        source = max(0, true_source - 1)
                    elif mode == "wrong_identity":
                        candidates = [
                            prior
                            for other_id, prior in last_by_id.items()
                            if other_id != token_id and prior < position
                        ]
                        if candidates:
                            source = candidates[(token_id + 17 * position) % len(candidates)]
                        else:
                            source = max(0, true_source - 1)
                    sources[row, position] = source
                    active[row, position] = True
                last_by_id[token_id] = position
        return sources, active

    def _transport_hook(self, module, inputs, output):
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        if self._input_ids is None or self._attention_mask is None:
            return output
        mode = self.runtime_edge_mode or self.edge_mode
        sources, active = self._source_map(self._input_ids, self._attention_mask, mode)
        self._last_active_fraction = float(active.float().mean().detach().cpu())
        if not active.any():
            return output
        gathered = hidden.gather(
            1, sources.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        )
        features = torch.cat((hidden, gathered, hidden - gathered), dim=-1)
        gate = torch.sigmoid(self.gate(features))
        scale = torch.sigmoid(self.transport_logit)
        transported = hidden + active.unsqueeze(-1) * scale * gate * self.value(gathered)
        if isinstance(output, tuple):
            return (transported,) + output[1:]
        return transported

    @contextlib.contextmanager
    def use_edge_mode(self, mode: str | None):
        if mode is not None and mode not in self.EDGE_MODES:
            raise ValueError(mode)
        old = self.runtime_edge_mode
        self.runtime_edge_mode = mode
        try:
            yield
        finally:
            self.runtime_edge_mode = old

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


def evaluate_detailed(
    model: IdentityEdgeTransportMLM,
    pairs,
    pad_id: int,
    device: torch.device,
    edge_mode: str | None = None,
    batch_size: int = 64,
):
    model.eval()
    margins: list[float] = []
    pair_correct: list[bool] = []
    condition_correct: list[bool] = []
    nll: list[float] = []
    kinds: list[str] = []
    with model.use_edge_mode(edge_mode), torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            logits_a, logits_b = binding_task.batch_logits(model, chunk, pad_id, device)
            labels_a = torch.tensor([pair.label_a for pair in chunk], device=device)
            labels_b = torch.tensor([pair.label_b for pair in chunk], device=device)
            rows = torch.arange(len(chunk), device=device)
            a_good = logits_a[rows, labels_a] - logits_a[rows, labels_b]
            b_good = logits_b[rows, labels_b] - logits_b[rows, labels_a]
            a_ok = a_good > 0
            b_ok = b_good > 0
            margins.extend((0.5 * (a_good + b_good)).cpu().tolist())
            pair_correct.extend((a_ok & b_ok).cpu().tolist())
            condition_correct.extend(
                torch.stack((a_ok, b_ok), dim=1).reshape(-1).cpu().tolist()
            )
            nll.extend(
                (
                    0.5
                    * (
                        F.cross_entropy(logits_a, labels_a, reduction="none")
                        + F.cross_entropy(logits_b, labels_b, reduction="none")
                    )
                )
                .cpu()
                .tolist()
            )
            kinds.extend(pair.kind for pair in chunk)
    summary = {
        "n": len(pairs),
        "condition_accuracy": float(np.mean(condition_correct)),
        "pair_accuracy": float(np.mean(pair_correct)),
        "cross_margin_mean": float(np.mean(margins)),
        "cross_margin_median": float(np.median(margins)),
        "correct_nll": float(np.mean(nll)),
    }
    for kind in sorted(set(kinds)):
        indices = [index for index, value in enumerate(kinds) if value == kind]
        summary[f"{kind}_pair_accuracy"] = float(
            np.mean([pair_correct[index] for index in indices])
        )
        summary[f"{kind}_margin_mean"] = float(
            np.mean([margins[index] for index in indices])
        )
    return summary, pair_correct, margins


def bootstrap_delta(
    candidate: list[bool], control: list[bool], seed: int, samples: int = 4000
) -> list[float]:
    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(control, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("paired vectors differ in length")
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(a), size=len(a))
        deltas[index] = (a[chosen] - b[chosen]).mean()
    return [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]


def train_arm(
    arm: str,
    seed: int,
    config: DebertaV2Config,
    train_pairs,
    pad_id: int,
    special_ids: set[int],
    device: torch.device,
    steps: int = 240,
    batch_size: int = 24,
):
    torch.manual_seed(seed)
    random.seed(seed)
    model = IdentityEdgeTransportMLM(config, arm, special_ids).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    rng = random.Random(seed * 1009 + 306)
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
        "transport_scale": float(torch.sigmoid(model.transport_logit).detach().cpu()),
        "last_active_fraction": model._last_active_fraction,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def build_data(tokenizer):
    locations = binding_task.one_token_locations(tokenizer)
    train_pairs = binding_task.build_pairs(
        tokenizer,
        4000,
        30601,
        binding_task.TRAIN_NAMES,
        binding_task.TRAIN_OBJECTS,
        locations,
        binding_task.TRAIN_EVENT_TEMPLATES,
        binding_task.TRAIN_UPDATE_TEMPLATES,
        binding_task.TRAIN_QUERY_TEMPLATES,
    )
    heldout = {
        "new_entities": binding_task.build_pairs(
            tokenizer,
            500,
            30602,
            binding_task.HELD_NAMES,
            binding_task.HELD_OBJECTS,
            locations,
            binding_task.TRAIN_EVENT_TEMPLATES,
            binding_task.TRAIN_UPDATE_TEMPLATES,
            binding_task.TRAIN_QUERY_TEMPLATES,
        ),
        "new_templates": binding_task.build_pairs(
            tokenizer,
            500,
            30603,
            binding_task.TRAIN_NAMES,
            binding_task.TRAIN_OBJECTS,
            locations,
            binding_task.HELD_EVENT_TEMPLATES,
            binding_task.HELD_UPDATE_TEMPLATES,
            binding_task.HELD_QUERY_TEMPLATES,
        ),
        "new_all": binding_task.build_pairs(
            tokenizer,
            600,
            30604,
            binding_task.HELD_NAMES,
            binding_task.HELD_OBJECTS,
            locations,
            binding_task.HELD_EVENT_TEMPLATES,
            binding_task.HELD_UPDATE_TEMPLATES,
            binding_task.HELD_QUERY_TEMPLATES,
        ),
        "new_all_updates": binding_task.build_pairs(
            tokenizer,
            400,
            30605,
            binding_task.HELD_NAMES,
            binding_task.HELD_OBJECTS,
            locations,
            binding_task.HELD_EVENT_TEMPLATES,
            binding_task.HELD_UPDATE_TEMPLATES,
            binding_task.HELD_QUERY_TEMPLATES,
            force_update=True,
        ),
    }
    return locations, train_pairs, heldout


def main() -> None:
    binding_task.setup_environment()
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(binding_task.TOKENIZER_PATH, use_fast=True)
    if tokenizer.mask_token_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer requires mask and pad tokens")
    locations, train_pairs, heldout = build_data(tokenizer)
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
    arms = ["off", "previous_token", "wrong_identity", "identity"]
    seeds = [42, 43]
    special_ids = set(tokenizer.all_special_ids)
    results = {}
    vectors = {}
    identity_models = {}
    for arm in arms:
        results[arm] = {}
        vectors[arm] = {}
        for seed in seeds:
            print(json.dumps({"event": "train_start", "arm": arm, "seed": seed}), flush=True)
            model, training = train_arm(
                arm, seed, config, train_pairs, tokenizer.pad_token_id, special_ids, device
            )
            evaluations = {}
            vectors[arm][str(seed)] = {}
            for split_name, pairs in heldout.items():
                summary, pair_correct, margins = evaluate_detailed(
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
            if arm == "identity":
                identity_models[seed] = model
            else:
                del model

    interventions = {}
    for seed, model in identity_models.items():
        interventions[str(seed)] = {}
        for mode in ["identity", "off", "previous_token", "wrong_identity"]:
            summary, pair_correct, margins = evaluate_detailed(
                model,
                heldout["new_all"],
                tokenizer.pad_token_id,
                device,
                edge_mode=mode,
            )
            interventions[str(seed)][mode] = summary
            if mode == "off":
                ci = bootstrap_delta(
                    vectors["identity"][str(seed)]["pair_correct"],
                    pair_correct,
                    306000 + seed,
                )
                interventions[str(seed)]["identity_minus_off_pair_accuracy_ci95"] = ci
        del model

    def pair_accuracy(arm: str, seed: int, split: str = "new_all") -> float:
        return results[arm][str(seed)]["evaluation"][split]["pair_accuracy"]

    primary = {arm: [pair_accuracy(arm, seed) for seed in seeds] for arm in arms}
    active_controls = ["previous_token", "wrong_identity"]
    identity_delta = [
        pair_accuracy("identity", seed)
        - max(pair_accuracy(control, seed) for control in active_controls)
        for seed in seeds
    ]
    causal_drop = [
        interventions[str(seed)]["identity"]["pair_accuracy"]
        - interventions[str(seed)]["off"]["pair_accuracy"]
        for seed in seeds
    ]
    update_accuracy = [pair_accuracy("identity", seed, "new_all_updates") for seed in seeds]
    if (
        min(primary["identity"]) >= 0.65
        and min(identity_delta) >= 0.15
        and min(causal_drop) >= 0.15
        and min(update_accuracy) >= 0.55
    ):
        decision = "PROMOTE_IDENTITY_EDGE_TRANSPORT_TO_NATURAL_TEXT_PILOT"
        next_action = (
            "Test tokenization-robust word-span edges and ordinary MLM retention on natural "
            "pretraining text before any full-scale run."
        )
    else:
        decision = "REJECT_CURRENT_IDENTITY_EDGE_TRANSPORT_MECHANISM"
        next_action = (
            "Do not scale this implementation; use the failed dimensions to revise the "
            "inductive bias rather than repairing thresholds."
        )

    payload = {
        "status": "IDENTITY_EDGE_STATE_TRANSPORT_GATE_COMPLETE",
        "design": {
            "hypothesis": (
                "a deterministic same-token edge can transport contextual state between "
                "mentions without learned role parsing and remain equivariant to entity renaming"
            ),
            "insertion": "gated residual after DeBERTa encoder layer 1 of 2",
            "arms": arms,
            "matched_parameters_initialization_batches": True,
            "seeds": seeds,
            "steps": 240,
            "batch_pairs": 24,
            "train_pairs": len(train_pairs),
            "heldout_pairs": {name: len(pairs) for name, pairs in heldout.items()},
            "state_vocabulary": locations,
            "device": str(device),
            "no_babylm_eval_data": True,
            "promotion_gate": {
                "identity_pair_accuracy_each_seed": 0.65,
                "identity_minus_best_active_control_each_seed": 0.15,
                "same_model_identity_minus_off_each_seed": 0.15,
                "latest_update_pair_accuracy_each_seed": 0.55,
            },
        },
        "provenance": {
            "script": str(_public_path('experiments/archive/initial_model_studies/scripts/identity_edge_state_transport_gate.py')),
            "script_sha256": sha256_file(_public_path('experiments/archive/initial_model_studies/scripts/identity_edge_state_transport_gate.py')),
            "source_step305_sha256": sha256_file(SOURCE),
            "tokenizer": str(binding_task.TOKENIZER_PATH.resolve()),
            "tokenizer_json_sha256": sha256_file(binding_task.TOKENIZER_PATH / "tokenizer.json"),
        },
        "results": results,
        "interventions": interventions,
        "primary": {
            "new_all_pair_accuracy": primary,
            "identity_minus_best_active_control": identity_delta,
            "same_model_identity_minus_off": causal_drop,
            "identity_new_all_updates_pair_accuracy": update_accuracy,
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
        "# research identity-edge state transport gate",
        "",
        f"Decision: **{decision}**",
        "",
        (
            "The primary split changes names, objects, event/update/query templates, and all "
            "bindings. No BabyLM evaluation data is used."
        ),
        "",
        "| arm | seed 42 pair acc | seed 43 pair acc |",
        "|---|---:|---:|",
    ]
    for arm in arms:
        lines.append(f"| {arm} | {primary[arm][0]:.3f} | {primary[arm][1]:.3f} |")
    lines.extend(
        [
            "",
            f"Identity minus best active control: {identity_delta}",
            "",
            f"Same trained model, identity minus path-off: {causal_drop}",
            "",
            f"Identity latest-update pair accuracy: {update_accuracy}",
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


if __name__ == "__main__":
    main()
