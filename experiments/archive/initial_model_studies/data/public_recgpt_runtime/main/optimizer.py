"""
Aurora optimizer code copied and modified from
https://github.com/tilde-research/aurora-release

Modified to implement Cautious Weight Decay (https://arxiv.org/pdf/2510.12402)
CWD decays only coordinates where update and parameter align.

Aurora fixes the dead neuron issue with tall matrixes, and the more rectangular the matrix is the more effective it is.
Therefore it should be a good fit for RecursiveGPT as we use a 16x MLP multiplier.
"""

import torch


def polar(G: torch.Tensor) -> torch.Tensor:
    """Polar factor via 12-step simple-quintic Newton-Schulz."""
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm <= 1 so the iteration converges to polar.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def aurora_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    mu: float = 0.95,
    nesterov: bool = True,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    if grad.ndim != 2:
        raise ValueError(f"aurora expects 2D gradient tensors, got shape {tuple(grad.shape)}")
    if momentum.shape != grad.shape:
        raise ValueError(f"momentum shape {tuple(momentum.shape)} must match grad shape {tuple(grad.shape)}")
    if not (0.0 < mu < 1.0):
        raise ValueError(f"mu must be in (0, 1), got {mu}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if pp_iterations < 1:
        raise ValueError(f"pp_iterations must be >= 1, got {pp_iterations}")
    if pp_beta <= 0.0:
        raise ValueError(f"pp_beta must be positive, got {pp_beta}")

    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum.clone()

    m, n = update.size(-2), update.size(-1)
    if m == n:
        update = polar(update)
    else:
        transposed = m < n
        if transposed:
            update = update.mT
            m, n = n, m

        G32 = update.to(torch.float32)
        target_row_sq = n / m
        row_norm = G32.norm(dim=-1, keepdim=True).clamp_(min=eps)
        D = 1.0 / row_norm
        for k in range(pp_iterations):
            U = polar(D * G32)
            if k < pp_iterations - 1:
                row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp_(min=eps * eps)
                D = D * (target_row_sq / row_sq).pow(pp_beta)

        update = U.mT if transposed else U

    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    if not update.isfinite().all():
        raise RuntimeError("aurora produced non-finite update")
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)

class SingleDeviceAuroraWithAuxAdam(torch.optim.Optimizer):
    """
    Single-device Aurora for 2D weights, with Adam for auxiliary parameters.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["nesterov"] = group.get("nesterov", True)
                group["pp_iterations"] = group.get("pp_iterations", 2)
                group["pp_beta"] = group.get("pp_beta", 0.5)
                group["eps"] = group.get("eps", 1e-7)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {
                    "params", "lr", "momentum", "nesterov", "pp_iterations",
                    "pp_beta", "eps", "weight_decay", "use_muon",
                }
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = aurora_update(
                        p.grad,
                        state["momentum_buffer"],
                        mu=group["momentum"],
                        nesterov=group["nesterov"],
                        pp_iterations=group["pp_iterations"],
                        pp_beta=group["pp_beta"],
                        eps=group["eps"],
                    )
                    if group["weight_decay"] and had_grad:
                        # Cautious Weight Decay
                        lr = group["lr"]
                        wd = group["weight_decay"]
                        mask = (update * p).ge(0)
                        p.add_(p * mask, alpha=-lr * wd)
                    p.add_(update, alpha=-group["lr"])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        # Cautious Weight Decay
                        lr = group["lr"]
                        wd = group["weight_decay"]
                        mask = (update * p).ge(0)
                        p.add_(p * mask, alpha=-lr * wd)
                    p.add_(update, alpha=-group["lr"])

        return loss
