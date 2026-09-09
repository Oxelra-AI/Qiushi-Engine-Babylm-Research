#!/usr/bin/env python3
"""research: LAMB optimizer on research legal compact-view-reinvest substrate.

Replaces AdamW with LAMB (Layer-wise Adaptive Moments optimizer for Batch training,
You et al. 2020) while keeping the exact research corpus, tokenizer, architecture,
WWM masking, seeds, batch size, sequence length, and cosine LR schedule.

LAMB trust ratio = ||w||_2 / ||adam_update||_2 per parameter layer.
Weight decay is decoupled (same as AdamW).

Usage:
  python lamb_trainer.py --lamb_lr 0.007 --gpu 0 --max_words 20000000 \
      --run_label lamb_lr007_seed43022_20M
  python lamb_trainer.py --lamb_lr 0.005 --gpu 1 --max_words 20000000 \
      --run_label lamb_lr005_seed43022_20M
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import sys, os, json, pathlib, math, importlib.util
import torch

# parents: [0]=scripts, [1]=workspace, [2]=frontier_consolidation, [3]=Sessions, [4]=user root
USER_ROOT = _public_path('.')


# ═══════════════════════════════════════════════════════════════════════
# LAMB Optimizer
# ═══════════════════════════════════════════════════════════════════════
class LAMB(torch.optim.Optimizer):
    """Layer-wise Adaptive Moments optimizer for Batch training (You et al., 2020).

    For each parameter: compute bias-corrected Adam momentum and second-moment
    estimates, form the Adam update direction (including decoupled weight decay),
    then scale by trust_ratio = min(clamp, ||w||_2 / ||update||_2).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.01, clamp_value=10.0):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, clamp_value=clamp_value)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            clamp = group['clamp_value']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                t = state['step']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # Exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias-corrected estimates
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                m_hat = exp_avg / bc1
                v_hat = exp_avg_sq / bc2

                # Adam update direction
                update = m_hat / (v_hat.sqrt() + eps)

                # Decoupled weight decay
                if wd > 0:
                    update.add_(p, alpha=wd)

                # Trust ratio
                w_norm = p.norm(2).item()
                u_norm = update.norm(2).item()

                if w_norm > 0 and u_norm > 0:
                    ratio = w_norm / u_norm
                    if clamp > 0:
                        ratio = min(ratio, clamp)
                else:
                    ratio = 1.0

                # Parameter update
                p.add_(update, alpha=-group['lr'] * ratio)

        return loss


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    import argparse
    ap = argparse.ArgumentParser(description="LAMB optimizer trainer wrapper")
    ap.add_argument("--lamb_lr", type=float, required=True,
                    help="Peak learning rate for LAMB")
    ap.add_argument("--lamb_wd", type=float, default=0.01,
                    help="Weight decay for LAMB (default: 0.01)")
    ap.add_argument("--lamb_betas", type=str, default="0.9,0.999",
                    help="LAMB betas as comma-separated (default: 0.9,0.999)")
    ap.add_argument("--lamb_eps", type=float, default=1e-6,
                    help="LAMB epsilon (default: 1e-6)")
    ap.add_argument("--lamb_clamp", type=float, default=10.0,
                    help="Max trust ratio (default: 10.0)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_words", type=int, default=20_000_000)
    ap.add_argument("--run_label", type=str, required=True)
    ap.add_argument("--smoke_steps", type=int, default=0,
                    help="If >0, stop after this many steps (smoke test)")
    args = ap.parse_args()

    betas = tuple(float(x) for x in args.lamb_betas.split(","))
    assert len(betas) == 2, f"Expected 2 betas, got {len(betas)}"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ── Paths ──
    base_trainer = _public_path('experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py')
    run_dir = USER_ROOT / f"experiments/archive/frontier_consolidation/training/runs/{args.run_label}"
    corpus = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
    corpus_meta = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json')
    tok_dir = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')

    assert base_trainer.exists(), f"Base trainer not found: {base_trainer}"
    assert corpus.exists(), f"Corpus not found: {corpus}"
    assert tok_dir.exists(), f"Tokenizer not found: {tok_dir}"

    # ── Monkey-patch torch.optim.AdamW → LAMB ──
    original_adamw = torch.optim.AdamW
    _lamb_lr = args.lamb_lr
    _lamb_wd = args.lamb_wd
    _lamb_betas = betas
    _lamb_eps = args.lamb_eps
    _lamb_clamp = args.lamb_clamp

    def lamb_factory(params, lr=None, weight_decay=None, betas=None, **kw):
        """Intercept the base trainer's AdamW call and return LAMB."""
        opt = LAMB(
            params,
            lr=lr if lr is not None else _lamb_lr,
            betas=_lamb_betas,
            eps=_lamb_eps,
            weight_decay=weight_decay if weight_decay is not None else _lamb_wd,
            clamp_value=_lamb_clamp,
        )
        # Count parameters
        total = sum(p.numel() for g in opt.param_groups for p in g['params'])
        n_tensors = sum(len(g['params']) for g in opt.param_groups)
        print(json.dumps({
            "event": "lamb_optimizer",
            "base_trainer": str(base_trainer),
            "lamb_lr": lr if lr is not None else _lamb_lr,
            "lamb_wd": weight_decay if weight_decay is not None else _lamb_wd,
            "lamb_betas": list(_lamb_betas),
            "lamb_eps": _lamb_eps,
            "lamb_clamp": _lamb_clamp,
            "base_trainer_requested_lr": lr,
            "base_trainer_requested_wd": weight_decay,
            "total_parameter_tensors": n_tensors,
            "total_parameter_count": total,
        }), flush=True)
        return opt

    torch.optim.AdamW = lamb_factory

    try:
        # ── Build sys.argv for the base trainer ──
        cmd = [
            str(base_trainer),
            "--example_jsonl", str(corpus),
            "--example_jsonl_label", args.run_label,
            "--example_jsonl_meta", str(corpus_meta),
            "--output_dir", str(run_dir),
            "--tokenizer_path", str(tok_dir),
            "--tokenizer_label", "compliant16k_reinvest10M",
            "--hidden_size", "480",
            "--n_layer", "8",
            "--n_head", "8",
            "--ffn_mult", "4",
            "--seed", "43",
            "--extra_init_seed", "43022",
            "--train_rng_seed", "43023",
            "--batch_size", "256",
            "--seq_length", "256",
            "--max_seq_length", "256",
            "--learning_rate", str(args.lamb_lr),
            "--warmup_fraction", "0.06",
            "--weight_decay", str(args.lamb_wd),
            "--masking_curriculum", "wwm_fixed",
            "--mask_prob_start", "0.15",
            "--mask_prob_end", "0.15",
            "--checkpoint_words", "1000000",
            "--max_word_exposure", str(args.max_words),
            "--num_workers", "0",
            "--log_every", "50",
            "--dynamics_trace_every", "200",
        ]
        if args.smoke_steps > 0:
            cmd.extend(["--smoke_steps", str(args.smoke_steps)])

        sys.argv = cmd

        # ── Execute base trainer ──
        mod_name = "masking_curriculum_trainer"
        spec = importlib.util.spec_from_file_location(mod_name, str(base_trainer))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod          # dataclass decorator needs this
        spec.loader.exec_module(mod)
        # The base trainer gates on __name__=="__main__"; call main() explicitly
        mod.main()

    finally:
        torch.optim.AdamW = original_adamw


if __name__ == "__main__":
    main()
