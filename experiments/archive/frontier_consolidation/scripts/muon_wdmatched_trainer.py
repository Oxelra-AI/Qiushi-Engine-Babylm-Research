#!/usr/bin/env python3
"""research: Muon-hidden / AdamW-interface trainer with matched Muon decay.

This is a causal repair of research. research changed two quantities at once on
hidden attention/FFN matrices: Muon orthogonalized updates and the same numeric
weight_decay=0.01 used by research AdamW. Because Muon LR=0.008 is 8x the AdamW
LR=0.001, the decoupled shrinkage p <- p * (1 - lr * wd) was also about 8x
larger on the Muon matrices. This wrapper keeps the research optimizer geometry
but lets the Muon group use a separate decay, normally 0.00125, so its per-step
shrinkage under LR=0.008 matches research's AdamW shrinkage 0.001*0.01.

Everything else is inherited from the exact research/COMPACT_EXPERIENCE trainer: data loading,
model initialization, WWM, RNG, clipping, cosine schedule, checkpointing, and
Hugging Face serialization. Only optimizer parameter grouping and the Muon group
update rule are changed.
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
    """Official Muon quintic Newton-Schulz orthogonalization."""
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
    update = grad.lerp(momentum_buffer, beta)  # Nesterov form used by the reference Muon code.
    update = zeropower_via_newtonschulz5(update, ns_steps)
    update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    return update


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Single-device Muon plus AdamW, with schedulable per-group learning rates."""
    def __init__(self, param_groups: list[dict], ns_steps: int = 5):
        self.ns_steps = int(ns_steps)
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("each group must declare use_muon")
            if group["use_muon"]:
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
            else:
                group.setdefault("betas", (0.9, 0.98))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                lr = float(group["lr"])
                wd = float(group["weight_decay"])
                beta = float(group["momentum"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.ndim != 2:
                        raise RuntimeError(f"non-matrix in Muon group: {tuple(p.shape)}")
                    state = self.state[p]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta, self.ns_steps)
                    # Decoupled weight decay. The scheduler changes group['lr']; with
                    # muon_weight_decay=0.00125 and muon_lr=0.008 this equals the
                    # research AdamW shrinkage 0.001*0.01 at every shared multiplier.
                    p.mul_(1.0 - lr * wd)
                    p.add_(update.reshape_as(p), alpha=-lr)
            else:
                lr = float(group["lr"])
                wd = float(group["weight_decay"])
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    state["exp_avg"].lerp_(p.grad, 1.0 - beta1)
                    state["exp_avg_sq"].lerp_(p.grad.square(), 1.0 - beta2)
                    m = state["exp_avg"] / (1.0 - beta1 ** state["step"])
                    v = state["exp_avg_sq"] / (1.0 - beta2 ** state["step"])
                    update = m / (v.sqrt() + eps)
                    p.mul_(1.0 - lr * wd)
                    p.add_(update, alpha=-lr)
        return loss


def is_muon_hidden_name(name: str) -> bool:
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
    if wrapper.muon_lr <= 0 or wrapper.muon_momentum < 0 or wrapper.muon_momentum >= 1:
        raise ValueError("invalid Muon hyperparameters")
    if wrapper.muon_weight_decay < 0:
        raise ValueError("muon_weight_decay must be nonnegative")
    sys.argv = remaining
    base = load_base_module()

    registry: dict = {}
    original_build_model = base.build_model
    original_adamw = torch.optim.AdamW

    def wrapped_build_model(args, tokenizer):
        model = original_build_model(args, tokenizer)
        named = list(model.named_parameters())
        muon_named = [(n, p) for n, p in named if is_muon_hidden_name(n)]
        adam_named = [(n, p) for n, p in named if not is_muon_hidden_name(n)]
        if len(muon_named) != 48:
            raise RuntimeError(f"expected 48 Muon matrices, got {len(muon_named)}")
        if any(p.ndim != 2 for _, p in muon_named):
            raise RuntimeError("Muon group contains non-matrix")
        all_ids = {id(p) for _, p in named}
        muon_ids = {id(p) for _, p in muon_named}
        adam_ids = {id(p) for _, p in adam_named}
        if muon_ids & adam_ids or (muon_ids | adam_ids) != all_ids:
            raise RuntimeError("parameter grouping is not an exact partition")
        report = {
            "base_trainer": str(BASE_TRAINER),
            "muon_lr": wrapper.muon_lr,
            "muon_momentum": wrapper.muon_momentum,
            "muon_ns_steps": wrapper.muon_ns_steps,
            "muon_weight_decay": wrapper.muon_weight_decay,
            "nominal_muon_shrink_per_step_at_base_lr": wrapper.muon_lr * wrapper.muon_weight_decay,
            "adamw_shrink_per_step_at_base_lr": float(args.learning_rate) * float(args.weight_decay),
            "muon_shrink_ratio_vs_step35": (wrapper.muon_lr * wrapper.muon_weight_decay) / (float(args.learning_rate) * float(args.weight_decay)) if float(args.learning_rate) * float(args.weight_decay) else None,
            "adam_lr": float(args.learning_rate),
            "adam_weight_decay": float(args.weight_decay),
            "muon_parameter_tensors": len(muon_named),
            "muon_parameter_count": sum(p.numel() for _, p in muon_named),
            "adam_parameter_tensors": len(adam_named),
            "adam_parameter_count": sum(p.numel() for _, p in adam_named),
            "total_unique_parameter_count": sum(p.numel() for _, p in named),
            "muon_names": [n for n, _ in muon_named],
            "adam_names": [n for n, _ in adam_named],
        }
        registry.update(muon_params=[p for _, p in muon_named], adam_params=[p for _, p in adam_named], report=report)
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
        if {id(p) for p in supplied} != ({id(p) for p in registry["muon_params"]} | {id(p) for p in registry["adam_params"]}):
            raise RuntimeError("optimizer input does not match grouped model parameters")
        groups = [
            {"params": registry["muon_params"], "lr": wrapper.muon_lr,
             "momentum": wrapper.muon_momentum, "weight_decay": wrapper.muon_weight_decay,
             "use_muon": True},
            {"params": registry["adam_params"], "lr": lr,
             "betas": betas, "eps": kwargs.get("eps", 1e-8),
             "weight_decay": weight_decay, "use_muon": False},
        ]
        return SingleDeviceMuonWithAuxAdam(groups, ns_steps=wrapper.muon_ns_steps)

    base.build_model = wrapped_build_model
    torch.optim.AdamW = optimizer_factory
    try:
        base.main()
    finally:
        torch.optim.AdamW = original_adamw


if __name__ == "__main__":
    main()
