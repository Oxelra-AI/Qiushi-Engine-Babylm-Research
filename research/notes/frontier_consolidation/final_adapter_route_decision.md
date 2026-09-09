# adapter matched horizon plan final — corrected adapter inference and next discriminating work

## What changed from adapter 20M closure

The schedule comparison confirms that adapter 20M closure's 37-point adapter runs were not a fair test of function-preserving residual capacity. The zero-output branch must recruit, but adapter 20M closure used a standalone 20M cosine schedule that placed the LR peak too early and decayed LR to zero by 20M. adapter matched horizon plan reran the minimum matched comparison on the original spatial repair route status 100M LR horizon.

## Matched-horizon evidence

Two arms used the exact protected spatial repair route status legal compact-view-reinvest substrate and stopped at the exact spatial repair route status 20M checkpoint exposure of 20,008,711 words / 506 batches:

- live adapter128: `adapter_enabled=1`, zero-output residual bottleneck after each encoder layer.
- disabled adapter128: same wrapper and unused parameters, but branch returns zero.

The disabled arm exactly reproduced spatial repair route status at every logged watch step: same loss, LR, words, masks, final loss 3.755582, adapter RMS 0, stock displacement 0. This validates the wrapper, gradient checkpointing, custom model serialization, and horizon setting as a faithful execution control.

The live matched-horizon arm also recovered normal training: final loss 3.756545, only +0.000962 over spatial repair route status/disabled. Thus adapter 20M closure's ~6.88 final losses were a schedule artifact. The artifact is specifically the LR timing mismatch: adapter 20M closure reached peak LR by legal tokenizer clean control trajectory design and had already decayed while spatial repair route status's loss-descent phase was beginning; adapter matched horizon plan shares spatial repair route status's LR at all watch steps.

## Behavior at 20M

| arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading | cheap7 | Δcheap7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial repair route status/disabled 20M | 59.69 | 55.45 | 50.73 | 18.65 | 50.26 | 34.195 | 8.67 | 39.6636 | 0.0000 |
| adapter 20M closure adapter128 compressed | 53.81 | 50.13 | 51.08 | 17.07 | 50.34 | 33.75 | 7.17 | 37.6214 | -2.0421 |
| adapter matched horizon plan adapter128 live matched horizon | 60.38 | 56.23 | 49.49 | 18.65 | 50.44 | 32.225 | 8.31 | 39.3893 | -0.2743 |

The matched horizon removes the catastrophic syntax loss and produces real local gains: BLiMP +0.69, Supplement +0.78, COMPS +0.18. But it also loses EWoK -1.24, GlobalPIQA -1.97, and Reading -0.36, with Entity unchanged. At 20M this is competence redistribution, not a broad sample-efficiency gain.

## Branch use and backbone movement

Live branch is clearly recruited: adapter output RMS mean 0.05657, max 0.06987, adapter-up norm sum 28.23. The disabled control has adapter RMS/up norm 0.

independent_review correctly warned that the original unweighted per-tensor displacement statistic over-read small tensors. The norm-weighted readout is more precise:

| comparison | whole-vector cosine | relative L2 to reference | interpretation |
|---|---:|---:|---|
| live128 20M vs disabled128 20M | 0.89024 | 0.46809 | substantial different stock-backbone trajectory at same exposure |
| disabled128 20M vs spatial repair route status 20M | 1.00000 | 0.00000 | exact execution control |
| spatial repair route status 20M vs init | 0.76161 | 0.85123 | normal 20M training moves far from init |
| spatial repair route status 21M vs spatial repair route status 20M | 0.99563 | 0.09435 | one additional million words moves weights modestly |
| spatial repair route status 50M vs spatial repair route status 20M | 0.88866 | 0.52704 | live-vs-disabled movement is comparable to a normal 20M→50M trajectory length |

The displacement is not a small-tensor artifact: whole-vector rel-L2 is large and top squared-difference mass includes word embeddings and relative-position projection weights. But the correct interpretation is not simply "harmful co-adaptation". Parameter movement of this scale occurs during normal training; the functional evidence (EWoK/GP/Reading down while BLiMP/Supp up) is what makes the ordinary joint adapter unpromoted at 20M.

## Scientific interpretation

adapter matched horizon plan revises the residual-capacity line rather than closing it. The zero-output post-layer adapter is trainable and exactly function-preserving at initialization under official-compatible loading. However, in ordinary joint training it does not preserve the spatial repair route status behavior surface at 20M: it recruits and keeps scalar loss unchanged while shifting competence from world/relation/reading columns toward BLiMP/Supplement/COMPS.

This means the current problem is not "adapters cannot train" but **coupling control**. Zero-output initialization protects the initial forward function only. Once the branch recruits, two coupling channels can alter the backbone trajectory:

1. adapter input Jacobian back-injected into earlier layers; and
2. forward residual stream shift that changes later-layer inputs and thus their ordinary gradients.

The next architecture work should isolate these channels rather than merely increasing adapter width or launching another compressed-schedule run.

## What should happen next

Do not jump to curriculum, seed sweeps, or a new model family from adapter 20M closure. clean curriculum evidence remains pending and should be used when available, but it is not a substitute for resolving the residual-capacity mechanism.

The next Execute step should produce one of these direct discriminating evidence objects:

1. **Ordinary adapter maturation check**: run live adapter128 to a later midpoint on the original 100M horizon, preferably 50M, because the 20M matched result is only modestly negative and BabyLM route rankings can change late. Since optimizer state at 20M was not saved, a faithful 50M run must start from the same random initialization and use `max_word_exposure=50021468`, `lr_total_steps=2529`, same seeds/settings. Evaluate live50M and spatial repair route status chck50M cheap7. Continue ordinary adapter only if EWoK/GlobalPIQA/Reading recover and cheap7 meets/exceeds spatial repair route status; otherwise do not promote it to 80M.
2. **No-training live-branch ablation**: evaluate the live20M checkpoint with adapter disabled at inference. This separates direct adapter output from stock-backbone drift and helps decide whether a stop-gradient or frozen-reader design is the correct repair.
3. **Gradient-isolated adapter construction**: implement `h_out = h + Adapter(detach(h))` with zero-output up projection and the same 100M-horizon 20M screen against the disabled control. This tests whether removing the adapter-input Jacobian preserves EWoK/GP/Reading while retaining BLiMP/Supp gains.
4. **Frozen-backbone side reader/residual head**: if trajectory protection is the main need, freeze the verified backbone (preferably from a spatial repair route status prefix checkpoint or from matched random init) and train only a zero-output residual path. This tests whether any additive capacity can raise columns with stock displacement exactly zero.

The lowest-risk immediate order is: run the 20M live-branch ablation and a 50M ordinary-adapter maturation check; if ordinary adapter still redistributes, the next proposed comparison is the stop-gradient side path and/or frozen reader as the next real architecture rather than continuing the ordinary joint adapter.

## Evidence files

- Training dynamics: `data/adapter_training_dynamics/adapter_training_dynamics.{json,md}`.
- Matched-horizon evaluation and mechanism: `data/adapter_matched_horizon_eval/adapter_matched_horizon_summary.{json,md}`.
- Norm-weighted displacement: `data/backbone_displacement_deepread/backbone_displacement_deepread.{json,md}`.
- Main synthesis before independent_review corrections: `notes/adapter_matched_horizon_synthesis.md`.
- independent_review generator/verifier: , .

The best fully legal complete endpoint remains spatial repair route status Overall 41.257770896404615. adapter matched horizon plan produced no complete endpoint and no SOTA/submission-ready result.
