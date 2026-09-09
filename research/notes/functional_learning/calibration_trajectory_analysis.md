# accumulated mechanism synthesis analysis: Compatible learning trajectories exist

## Central finding

The conflict between context prediction and held-symbol binding is **not inherent** but arises from how the full objective is introduced. Two compatible trajectory approaches preserve substantially more held-symbol transfer while achieving comparable context prediction:

1. **Interleaved answer/full** preserves the most binding (mean held_top4 0.746 vs direct's 0.418, hB +7.77 vs +3.11)
2. **Gradual context ramp** preserves substantial binding (mean held_top4 0.587, hB +5.70)

Both achieve final context CE ~1.23–1.32 nats, matching direct_full's 1.23.

## Endpoint comparison (mean across 3 seeds)

| arm | start h4 | final h4 | final hB | final hSel | train4 | ctx_ce |
|---|---:|---:|---:|---:|---:|---:|
| direct_full | 0.753 | 0.418 | +3.11 | +0.256 | 0.921 | 1.233 |
| calibrate_out50 | 0.753 | 0.423 | +3.31 | +0.296 | 0.915 | 1.233 |
| gradual_ctx100 | 0.753 | **0.587** | **+5.70** | **+0.442** | 1.000 | 1.229 |
| slow_lr25 | 0.753 | 0.486 | +4.13 | +0.314 | 1.000 | 1.231 |
| interleaved | 0.753 | **0.746** | **+7.77** | **+0.665** | 1.000 | 1.316 |

## Per-seed final held metrics

| arm | sd42 h4/hB | sd43 h4/hB | sd100 h4/hB |
|---|---|---|---|
| direct_full | 0.449/+4.33 | 0.406/+2.29 | 0.398/+2.71 |
| calibrate_out50 | 0.473/+4.73 | 0.387/+2.37 | 0.410/+2.82 |
| gradual_ctx100 | 0.441/+3.81 | **0.742/+6.92** | **0.578/+6.37** |
| slow_lr25 | 0.480/+4.78 | 0.543/+4.95 | 0.434/+2.65 |
| interleaved | **0.570/+5.74** | **0.855/+9.25** | **0.812/+8.31** |

Seed 42 has weak preparation binding (h4=0.484), so effects are small. Seeds 43 and 100
show the decisive contrast: direct_full collapses to h4~0.40 while interleaved preserves
to h4>0.81 and gradual preserves to h4>0.57.

## Trajectory dynamics

### Interleaved: binding preserved and even strengthened

Seed 43 interleaved: preparation h4=0.820, **peak h4=0.961** at be=200, final h4=0.855.
The binding actually *increases* during interleaved training because answer-only epochs
strengthen the marker while full epochs provide context learning.

Seed 100 interleaved: preparation h4=0.953, dips to h4=0.535 at be=250, then recovers
to h4=0.812 at be=500.  Even the dip is far above direct_full's collapse to 0.22 at be=5.

### Gradual ramp: binding held during calibrated introduction

Seed 43 gradual: h4 stays above 0.82 through be=50 (ctx_weight=0.5), drops to 0.75 at
be=100 (full weight), then stabilizes at 0.64–0.77 during full continuation.  Context CE
drops from 29 to 1.23 over the same period.

Seed 100 gradual: h4 drops from 0.953 to 0.50 during the ramp (more sensitive than sd43),
then recovers to 0.57–0.61 during full continuation.  Still far above direct's 0.40.

### Calibration: readout alone cannot bridge the gap

Out-only calibration barely reduces context CE (30→25 after 50 epochs for seed 100).
When the body is unfrozen at be=51, the same shock occurs: seed 100 calibrate drops to
h4=0.316 at be=75 (25 body epochs), similar to direct_full's be=25.  The readout cannot
learn context prediction without body changes because the body's hidden states at context
positions don't contain next-token-predictive features.

### Slow lr: delays but doesn't prevent collapse

Seed 100 slow_lr: h4=0.934 at be=1 (lr=3e-5), h4=0.848 at be=5, h4=0.629 at be=10,
then collapses to h4=0.227 at be=25 when lr switches to 3e-4.  The reduced step size
delays but doesn't prevent the destructive trajectory once full-rate context gradients
arrive.

## Mechanism interpretation

### Why interleaved works

Every other epoch is answer-only, which reinforces the binding marker through body gradients
that *strengthen* query-conditioned context marking (body objective ablation showed body-only answer-only
increases hB from +7.77 to +11.45 in 25 epochs).  On full epochs, context gradients perturb
the marker, but the next answer-only epoch repairs it.  The alternation creates a dynamic
equilibrium: context features accumulate incrementally while the binding marker is actively
maintained.

The cost: slightly higher final context CE (1.32 vs 1.23), reflecting that only half the
epochs contribute context gradients.  This is a tiny price for dramatically better transfer.

### Why gradual ramp partially works

Small context weights at early epochs give the body time to find representations that serve
both purposes.  By the time full weight arrives, the body has already adapted to a compatible
state.  But without ongoing answer reinforcement, some binding degrades under sustained full
pressure.  The ramp is a one-time intervention; interleaved is continuous.

### Why calibration fails

The readout (out.weight) alone can't learn context prediction because the body's hidden states
at context positions were shaped entirely by answer-only training.  Context positions contain
answer-irrelevant features.  Only body adaptation can install context-predictive features, and
that's exactly what destroys binding when done abruptly.

## Scientific principle

**Compatible representations exist but require managed learning trajectories.**

The body of this small Transformer can simultaneously support query-conditioned binding
(reusable, counterfactual) and next-token context prediction (local, statistical).  The
architecture has sufficient capacity for both.  The standard practice of abruptly switching
objectives creates a destructive optimization trajectory that overwrites the binding marker
with local prediction features.  Two alternative trajectories avoid this:

1. **Active reinforcement** (interleaved): periodic answer-only epochs maintain the binding
   computation while context learning proceeds on alternating epochs.
2. **Gradual introduction** (ramp): slowly increasing context weight gives the body time to
   find compatible representations.

This principle generalizes beyond the synthetic task: under finite data and compute, reusable
computation is fragile not because it is representationally incompatible with local prediction,
but because standard training dynamics prioritize the steepest descent direction (local
prediction improvement) over preservation of useful-but-not-immediately-rewarded features.
Managed objective transitions can prevent this.

## Files

- Script: `scripts/calibration_trajectory.py`
- Data: `data/calibration_trajectory/results.json`
- Figure: `figures/calibration_trajectory.png`
- Note: `notes/calibration_trajectory.md` (auto-generated)
- Framework: `notes/framework.md`
