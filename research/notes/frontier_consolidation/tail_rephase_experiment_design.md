# tail rephase experiment design: Tail Rephase Restart Experiment

## Scientific Question

Does restarting the optimizer from the legal chck_80M checkpoint (discarding
accumulated AdamW momentum/variance) improve the 80-100M tail phase enough to
narrow the legal-tokenizer gap?

## Motivation from COMPACT_EXPERIENCE

COMPACT_EXPERIENCE endpoint decision ledger and route quality ran a clean-tail restart from chck_80M on the OLD tokenizer:
- Continuation LR ≈ 0.000109 (matched to the original schedule's 80% point)
- No warmup, cosine decay to 0 over ~506 steps
- Fresh AdamW optimizer state (momentum/variance reset)

Results showed a clear 90M peak:
- chck_85M: mean7 42.704 (below clean-Qwen reference 43.113)
- chck_90M: mean7 43.436 (+0.323 vs clean-Qwen) ← PEAK
- chck_95M: mean7 43.116 (+0.003 vs clean-Qwen)
- chck_100M: mean7 43.213 (+0.100 vs clean-Qwen)

Key insight: the restart gives the fresh optimizer a productive learning window in
80-90M that the original near-zero LR schedule cannot exploit. The effect peaks at
90M and partially decays by 100M.

## Current Legal Deficit Context

Original schedule LR at 80M: 0.000107 → decays to 0 by 100M
80M→100M cheap7 gain: only +0.057 (42.949 → 43.006)
This means the original schedule makes almost no progress in the tail.

Legal100M vs old100M gap: Overall 41.258 vs 42.033 = −0.775
Same corpus, model, recipe, seeds — only tokenizer differs.

## Two Bounded Experiments

### Variant A: Matched-LR (COMPACT_EXPERIENCE-style)
- Load chck_80M, discard optimizer state
- Peak LR = 0.000108 (matched to original schedule at 80%)
- No warmup, cosine to 0
- 20M continuation on 10M pool with fresh shuffle (seed 43044)
- Expected: moderate 90M peak (+0.1-0.3 mean7 over existing 100M)

### Variant B: Base-LR (aggressive)
- Same as Variant A except peak LR = 0.001 (10× higher) with warmup 0.06
- This is genuinely new — COMPACT_EXPERIENCE never tested high-LR restart
- Tests whether aggressive late optimization can recover more of the legal deficit
- Risk: model destabilization; mitigation: 85M checkpoint allows early detection

## Decision Criteria

- If any 90M variant exceeds existing legal100M cheap7 (43.006) by ≥+0.3:
  continue that line to full evaluation (SuperGLUE + AoA)
- If both variants show <+0.3 at 90M: rephase is not the mechanism
- If base-LR diverges (loss spike, cheap7 < 40 at 85M): confirms high-LR is destructive

## Files

- Trainer: `scripts/rephase_restart_trainer.py`
- Evaluator: `scripts/eval_rephase_comparison.py`
- Matched-LR run: `training/runs/rephase_matched_lr_seed43044/`
- Base-LR run: `training/runs/rephase_base_lr_seed43044/`
- Evaluation output: `data/rephase_eval/`

## Broader Context

This is NOT a generic LR sweep. The COMPACT_EXPERIENCE evidence provides a specific mechanism:
fresh AdamW state peaks at 90M. The two variants test whether the mechanism
transfers to the legal tokenizer (Variant A) and whether higher LR amplifies it
(Variant B). Both are bounded to 20M words (~30 min GPU each).

If both fail, the legal-tokenizer gap is not explained by stale optimizer state in
the tail, and the research must return to representation or corpus-level mechanisms.
