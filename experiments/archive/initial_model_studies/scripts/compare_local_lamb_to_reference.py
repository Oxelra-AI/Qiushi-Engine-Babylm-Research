#!/usr/bin/env python3
"""Numerically compare the isolated-fork local LAMB against a recognized reference.

Reference provenance:
- torch-optimizer Lamb source staged at
  data/external/torch_optimizer_lamb.py
  (downloaded from https://raw.githubusercontent.com/jettify/pytorch-optimizer/master/torch_optimizer/lamb.py)
- That implementation cites cybertronai/pytorch-lamb and the original LAMB paper.

This script avoids importing torch_optimizer as a package (not installed) by copying
its update equations exactly for small tensors. It compares identical parameters
and gradients for two parameter groups and two steps.
"""
from __future__ import annotations
import importlib.util
import json
import pathlib
import math
import sys
import torch
from torch.optim.optimizer import Optimizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAINER = ROOT / "training/scripts/babylm_masked_train_leadershape.py"
REF_SRC = pathlib.Path("data/external/torch_optimizer_lamb.py")
OUT = ROOT / "data/lamb_numeric_comparison.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/lamb_numeric_comparison.md')


def import_local_lamb():
    spec = importlib.util.spec_from_file_location("babylm_masked_train_leadershape", TRAINER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.Lamb


class TorchOptimizerReferenceLamb(Optimizer):
    """Faithful minimal copy of staged torch_optimizer Lamb update.

    Lines 129-137 of the staged reference apply debiasing only if self.debias is
    true; default is false. Lines 139-156 clamp weight_norm, add weight decay to
    adam_step, compute trust_ratio, optionally force adam, and update p.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, clamp_value=10.0, adam=False, debias=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if self.debias:
                    bias_correction = math.sqrt(1 - beta2 ** state["step"])
                    bias_correction /= 1 - beta1 ** state["step"]
                else:
                    bias_correction = 1
                step_size = group["lr"] * bias_correction
                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)
                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(p.data, alpha=group["weight_decay"])
                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state["weight_norm"] = weight_norm.clone() if torch.is_tensor(weight_norm) else weight_norm
                state["adam_norm"] = adam_norm.clone() if torch.is_tensor(adam_norm) else adam_norm
                state["trust_ratio"] = trust_ratio.clone() if torch.is_tensor(trust_ratio) else trust_ratio
                if self.adam:
                    trust_ratio = 1
                p.data.add_(adam_step, alpha=-step_size * trust_ratio)
        return loss


def make_params(seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    p1 = torch.randn(3, 4, generator=g, dtype=torch.float64) * 0.2 + 0.1
    p2 = torch.randn(5, generator=g, dtype=torch.float64) * 0.05 - 0.03
    return [torch.nn.Parameter(p1.clone()), torch.nn.Parameter(p2.clone())]


def assign_grads(params, seed: int, scale: float):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    for p in params:
        p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype) * scale


def state_summary(opt):
    vals=[]
    for group in opt.param_groups:
        for p in group["params"]:
            st=opt.state[p]
            d={"step": st.get("step")}
            for k in ["exp_avg", "exp_avg_sq"]:
                if k in st:
                    d[k+"_norm"] = float(torch.norm(st[k]).item())
            for k in ["weight_norm", "adam_norm", "trust_ratio"]:
                if k in st:
                    v=st[k]
                    d[k] = float(v.item() if torch.is_tensor(v) else v)
            vals.append(d)
    return vals


def max_param_delta(a, b):
    return max(float(torch.max(torch.abs(x.detach() - y.detach())).item()) for x, y in zip(a, b))


def compare_case(name: str, ref_debias: bool, local_kwargs: dict, ref_kwargs: dict):
    LocalLamb = import_local_lamb()
    p_local = make_params(123)
    p_ref = make_params(123)
    opt_local = LocalLamb(p_local, **local_kwargs)
    opt_ref = TorchOptimizerReferenceLamb(p_ref, **ref_kwargs, debias=ref_debias)
    per_step=[]
    for step, (seed, scale) in enumerate([(1001, 0.03), (1002, 0.07)], 1):
        assign_grads(p_local, seed, scale)
        assign_grads(p_ref, seed, scale)
        opt_local.step()
        opt_ref.step()
        per_step.append({
            "step": step,
            "max_abs_param_delta": max_param_delta(p_local, p_ref),
            "local_state": state_summary(opt_local),
            "reference_state": state_summary(opt_ref),
            "local_trust_ratio_mean": getattr(opt_local, "last_trust_ratio_mean", None),
            "local_trust_ratio_min": getattr(opt_local, "last_trust_ratio_min", None),
            "local_trust_ratio_max": getattr(opt_local, "last_trust_ratio_max", None),
        })
    return {"case": name, "reference_debias": ref_debias, "local_kwargs": local_kwargs, "reference_kwargs": ref_kwargs, "per_step": per_step,
            "final_max_abs_param_delta": per_step[-1]["max_abs_param_delta"]}


def main():
    common = dict(lr=0.007, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
    cases=[]
    # Post-patch local class should exactly match the staged torch_optimizer/cybertronai
    # reference default when debias=False, and its optional debias mode when debias=True.
    cases.append(compare_case(
        "local_reference_style_debias_false_vs_torch_optimizer_default",
        ref_debias=False,
        local_kwargs={**common, "trust_clip": 10.0, "debias": False},
        ref_kwargs={**common, "clamp_value": 10.0, "adam": False},
    ))
    cases.append(compare_case(
        "local_reference_style_debias_true_vs_torch_optimizer_debias_true",
        ref_debias=True,
        local_kwargs={**common, "trust_clip": 10.0, "debias": True},
        ref_kwargs={**common, "clamp_value": 10.0, "adam": False},
    ))
    payload={
        "status":"LAMB_NUMERIC_COMPARISON",
        "trainer":str(TRAINER),
        "reference_source":str(REF_SRC),
        "reference_provenance":json.loads((REF_SRC.parent/'provenance.json').read_text()),
        "paper_anchor":"Knowledge/objects/papers/Large-Batch-Optimization-for-Deep-Learning-Training-BERT-in-76-minutes--69bd694887d4--f79503803440/object.md lines 117-130: Algorithm 2",
        "leader_metadata":"Public leader card exposes LAMB, cosine, max LR 0.007, but no training code or exact optimizer dependency.",
        "cases":cases,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    lines=["# research — local LAMB numerical comparison", "", f"JSON: `{OUT}`", "", "## Result", ""]
    for c in cases:
        lines.append(f"- {c['case']}: final max abs parameter delta = `{c['final_max_abs_param_delta']:.12g}`")
    lines += ["", "## Interpretation", "", "The public leader metadata does not expose training code or exact LAMB dependency. The staged `torch_optimizer` implementation (citing cybertronai/pytorch-lamb) is used as the recognized reference. Its default has `debias=False`; original Algorithm 2 in the LAMB paper uses bias-corrected moments. The isolated trainer now exposes this as an explicit `--lamb_debias` switch rather than silently measuring a custom variant. Near-zero deltas here are required before LAMB smokes or 10M screens."]
    NOTE.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"out":str(OUT), "cases":[{"case":c['case'],"final_delta":c['final_max_abs_param_delta']} for c in cases]}, indent=2))


if __name__ == "__main__":
    main()
