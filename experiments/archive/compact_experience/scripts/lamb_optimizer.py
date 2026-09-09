#!/usr/bin/env python3
"""LAMB optimizer (Layer-wise Adaptive Moments for Batch training).

Implements the LAMB optimizer from:
  Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Deepak Narayanan,
  Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh.
  "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes." ICLR 2020.

Key difference from Adam: layer-wise trust ratio that scales the update by
  ||weight|| / ||adam_update||
This allows using much higher learning rates with large batches.
"""
import math
from typing import Iterable, Tuple, Optional
import torch
from torch.optim import Optimizer


class LAMB(Optimizer):
    """LAMB optimizer.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-6)
        weight_decay: Weight decay (L2 penalty) (default: 0.01)
        exclude_from_layer_adaptation: Set of parameter names to exclude from
            layer-wise adaptation (typically bias and LayerNorm)
    """
    
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        exclude_from_layer_adaptation: Optional[set] = None,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._exclude_from_layer_adaptation = exclude_from_layer_adaptation or set()
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                step = state["step"]
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Compute the Adam update (bias-corrected)
                adam_update = (exp_avg / bias_correction1) / (
                    (exp_avg_sq / bias_correction2).sqrt() + eps
                )
                
                # Apply weight decay
                if weight_decay > 0:
                    adam_update = adam_update.add(p, alpha=weight_decay)
                
                # Compute layer-wise trust ratio
                # Exclude bias and layernorm from layer adaptation
                param_name = getattr(p, "_param_name", "")
                if param_name in self._exclude_from_layer_adaptation:
                    # No layer adaptation for excluded params (use standard Adam update)
                    trust_ratio = 1.0
                else:
                    weight_norm = p.norm(2).item()
                    update_norm = adam_update.norm(2).item()
                    
                    if weight_norm > 0 and update_norm > 0:
                        trust_ratio = weight_norm / update_norm
                    else:
                        trust_ratio = 1.0
                
                # Apply update
                p.add_(adam_update, alpha=-lr * trust_ratio)
        
        return loss


def get_lamb_optimizer(
    model: torch.nn.Module,
    lr: float = 0.007,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-6,
) -> LAMB:
    """Create LAMB optimizer with standard exclusions for bias and LayerNorm.
    
    Follows common practice: bias parameters and LayerNorm weights/biases
    are excluded from both weight decay and layer-wise adaptation.
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []
    excluded_names = set()
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Store name for the optimizer's reference
        param._param_name = name
        
        if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
            no_decay_params.append(param)
            excluded_names.add(name)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer = LAMB(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        exclude_from_layer_adaptation=excluded_names,
    )
    
    return optimizer
