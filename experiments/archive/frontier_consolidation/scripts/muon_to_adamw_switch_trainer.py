#!/usr/bin/env python3
"""research: early Muon hidden updates followed by AdamW consolidation.

This is a controlled repair of the research mature reversal. Continuous matched-
decay Muon kept hidden matrices broad through 80M and lowered MLM train loss, but
it redistributed BabyLM competence away from Entity/Reading by mature exposure.
This wrapper keeps the exact research legal trainer/data/masking/scheduler and
changes only hidden attention/FFN update geometry over time:

  * before or at --switch_after_steps: hidden matrices use Muon with matched
    decoupled shrinkage (default active Muon lr 0.008, wd 0.00125);
  * after --switch_after_steps: the same hidden matrices use AdamW at the base
    research learning rate/decay, while interface tensors use AdamW throughout.

The scheduler is the inherited research cosine schedule. The hidden group is given
base AdamW lr to the scheduler; Muon mode multiplies that scheduled lr by
muon_lr / adam_lr internally, so after the switch hidden AdamW lr is exactly the
research scheduled lr.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

BASE_TRAINER = _public_path('experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py')


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int) -> torch.Tensor:
    if g.ndim != 2:
        raise ValueError(f"Muon requires a matrix, got shape={tuple(g.shape)}")
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        aa = x @ x.mT
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
    if transposed:
        x = x.mT
    return x


def muon_update(grad: torch.Tensor, momentum_buffer: torch.Tensor, beta: float, ns_steps: int) -> torch.Tensor:
    momentum_buffer.lerp_(grad, 1.0 - beta)
    update = grad.lerp(momentum_buffer, beta)  # Nesterov form from reference Muon.
    update = zeropower_via_newtonschulz5(update, ns_steps)
    update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    return update


class HiddenMuonThenAdamW(torch.optim.Optimizer):
    """Hidden matrices switch from Muon to AdamW; all other tensors AdamW."""

    def __init__(self, param_groups: list[dict], *, ns_steps: int, switch_after_steps: int):
        self.ns_steps = int(ns_steps)
        self.switch_after_steps = int(switch_after_steps)
        self.global_step = 0
        self._last_hidden_mode: str | None = None
        for group in param_groups:
            role = group.get("role")
            if role == "hidden_switch":
                group.setdefault("betas", (0.9, 0.98))
                group.setdefault("eps", 1e-8)
                group.setdefault("adam_weight_decay", 0.01)
                group.setdefault("muon_weight_decay", 0.00125)
                group.setdefault("muon_momentum", 0.95)
                if group["muon_base_lr"] <= 0 or group["adam_base_lr"] <= 0:
                    raise ValueError("hidden_switch needs positive base lrs")
            elif role == "adam_interface":
                group.setdefault("betas", (0.9, 0.98))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0.01)
            else:
                raise ValueError(f"unknown optimizer group role={role!r}")
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.global_step += 1
        hidden_mode = "muon" if self.global_step <= self.switch_after_steps else "adamw"
        if hidden_mode != self._last_hidden_mode:
            print(json.dumps({
                "event": "hidden_update_mode",
                "optimizer_step": self.global_step,
                "hidden_mode": hidden_mode,
                "switch_after_steps": self.switch_after_steps,
            }), flush=True)
            self._last_hidden_mode = hidden_mode

        for group in self.param_groups:
            role = group["role"]
            scheduled_adam_lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])

            if role == "hidden_switch" and hidden_mode == "muon":
                muon_scale = float(group["muon_base_lr"]) / float(group["adam_base_lr"])
                lr = scheduled_adam_lr * muon_scale
                wd = float(group["muon_weight_decay"])
                beta = float(group["muon_momentum"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.ndim != 2:
                        raise RuntimeError(f"non-matrix in Muon hidden group: {tuple(p.shape)}")
                    state = self.state[p]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta, self.ns_steps)
                    p.mul_(1.0 - lr * wd)
                    p.add_(update.reshape_as(p), alpha=-lr)
                continue

            # AdamW path for interface tensors and hidden tensors after the switch.
            if role == "hidden_switch":
                wd = float(group["adam_weight_decay"])
            else:
                wd = float(group["weight_decay"])
            lr = scheduled_adam_lr
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["adam_step"] = 0
                state["adam_step"] += 1
                state["exp_avg"].lerp_(p.grad, 1.0 - beta1)
                state["exp_avg_sq"].lerp_(p.grad.square(), 1.0 - beta2)
                t = state["adam_step"]
                m = state["exp_avg"] / (1.0 - beta1 ** t)
                v = state["exp_avg_sq"] / (1.0 - beta2 ** t)
                update = m / (v.sqrt() + eps)
                p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


def is_hidden_matrix_name(name: str) -> bool:
    if not name.startswith("deberta.encoder.layer.") or not name.endswith(".weight"):
        return False
    return name.endswith((
        ".attention.self.query_proj.weight",
        ".attention.self.key_proj.weight",
        ".attention.self.value_proj.weight",
        ".attention.output.dense.weight",
        ".intermediate.dense.weight",
        ".output.dense.weight",
    ))


def consume_wrapper_args(argv: list[str]):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--muon_lr", type=float, default=0.008)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--muon_ns_steps", type=int, default=5)
    p.add_argument("--muon_weight_decay", type=float, default=0.00125)
    p.add_argument("--switch_after_steps", type=int, required=True)
    p.add_argument("--switch_label", default="")
    p.add_argument("--grouping_report", default="")
    known, remaining = p.parse_known_args(argv[1:])
    return known, [argv[0], *remaining]


def load_base_module():
    spec = importlib.util.spec_from_file_location("exact_step35_base", BASE_TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base trainer: {BASE_TRAINER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    wrapper, remaining = consume_wrapper_args(sys.argv)
    if wrapper.switch_after_steps < 0:
        raise ValueError("switch_after_steps must be nonnegative")
    if not (0.0 <= wrapper.muon_momentum < 1.0):
        raise ValueError("muon_momentum must be in [0,1)")
    if wrapper.muon_lr <= 0 or wrapper.muon_weight_decay < 0:
        raise ValueError("invalid Muon hyperparameters")

    sys.argv = remaining
    base = load_base_module()

    registry: dict = {}
    original_build_model = base.build_model
    original_adamw = torch.optim.AdamW

    def wrapped_build_model(args, tokenizer):
        model = original_build_model(args, tokenizer)
        named = list(model.named_parameters())
        hidden_named = [(n, p) for n, p in named if is_hidden_matrix_name(n)]
        interface_named = [(n, p) for n, p in named if not is_hidden_matrix_name(n)]
        if len(hidden_named) != 48:
            raise RuntimeError(f"expected 48 hidden matrices, got {len(hidden_named)}")
        if any(p.ndim != 2 for _, p in hidden_named):
            raise RuntimeError("hidden switch group contains non-matrix")
        all_ids = {id(p) for _, p in named}
        hidden_ids = {id(p) for _, p in hidden_named}
        interface_ids = {id(p) for _, p in interface_named}
        if hidden_ids & interface_ids or (hidden_ids | interface_ids) != all_ids:
            raise RuntimeError("parameter grouping is not an exact partition")
        report = {
            "base_trainer": str(BASE_TRAINER),
            "switch_label": wrapper.switch_label,
            "switch_after_steps": wrapper.switch_after_steps,
            "muon_lr": wrapper.muon_lr,
            "muon_momentum": wrapper.muon_momentum,
            "muon_ns_steps": wrapper.muon_ns_steps,
            "muon_weight_decay": wrapper.muon_weight_decay,
            "adam_lr": float(args.learning_rate),
            "adam_weight_decay": float(args.weight_decay),
            "muon_lr_scale_internal": wrapper.muon_lr / float(args.learning_rate),
            "nominal_muon_shrink_per_step_at_base_lr": wrapper.muon_lr * wrapper.muon_weight_decay,
            "adamw_shrink_per_step_at_base_lr": float(args.learning_rate) * float(args.weight_decay),
            "muon_shrink_ratio_vs_step35": (wrapper.muon_lr * wrapper.muon_weight_decay) / (float(args.learning_rate) * float(args.weight_decay)) if float(args.learning_rate) * float(args.weight_decay) else None,
            "hidden_parameter_tensors": len(hidden_named),
            "hidden_parameter_count": sum(p.numel() for _, p in hidden_named),
            "interface_parameter_tensors": len(interface_named),
            "interface_parameter_count": sum(p.numel() for _, p in interface_named),
            "total_unique_parameter_count": sum(p.numel() for _, p in named),
            "hidden_names": [n for n, _ in hidden_named],
            "interface_names": [n for n, _ in interface_named],
        }
        registry.update(hidden_params=[p for _, p in hidden_named], interface_params=[p for _, p in interface_named], report=report)
        print(json.dumps({"event": "parameter_groups", **{k: v for k, v in report.items() if not k.endswith("_names")}}), flush=True)
        if wrapper.grouping_report:
            rp = Path(wrapper.grouping_report)
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return model

    def optimizer_factory(params, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), **kwargs):
        supplied = list(params)
        if not registry:
            raise RuntimeError("optimizer constructed before model grouping")
        if {id(p) for p in supplied} != ({id(p) for p in registry["hidden_params"]} | {id(p) for p in registry["interface_params"]}):
            raise RuntimeError("optimizer input does not match grouped model parameters")
        groups = [
            {
                "params": registry["hidden_params"],
                "lr": lr,  # scheduler sees research AdamW base LR
                "adam_base_lr": float(lr),
                "muon_base_lr": wrapper.muon_lr,
                "muon_momentum": wrapper.muon_momentum,
                "muon_weight_decay": wrapper.muon_weight_decay,
                "adam_weight_decay": weight_decay,
                "betas": betas,
                "eps": kwargs.get("eps", 1e-8),
                "role": "hidden_switch",
            },
            {
                "params": registry["interface_params"],
                "lr": lr,
                "betas": betas,
                "eps": kwargs.get("eps", 1e-8),
                "weight_decay": weight_decay,
                "role": "adam_interface",
            },
        ]
        return HiddenMuonThenAdamW(groups, ns_steps=wrapper.muon_ns_steps, switch_after_steps=wrapper.switch_after_steps)

    base.build_model = wrapped_build_model
    torch.optim.AdamW = optimizer_factory
    try:
        base.main()
    finally:
        torch.optim.AdamW = original_adamw


if __name__ == "__main__":
    main()
