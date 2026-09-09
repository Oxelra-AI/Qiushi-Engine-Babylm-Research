# adapter matched horizon plan — matched-horizon residual-adapter test

## Why this experiment is needed

adapter 20M closure tested zero-output bottleneck residual adapters with a standalone 20M-word cosine schedule. That was not a fair test of the function-preserving capacity idea, because the adapter branch begins at zero output and must first recruit. The standalone 20M schedule decayed the learning rate to zero by the checkpoint, while the spatial repair route status 20M comparator is an early snapshot of the original 100M schedule with LR still high. Therefore the 37-point adapter 20M closure runs establish only that the compressed 20M schedule is bad for adapters; they do not securely establish that separately routed residual capacity is bad.

## Minimum discriminating run

Run exactly two 20M-prefix arms on the original 100M LR horizon:

1. **live adapter128**: `adapter_enabled=1`, bottleneck 128, zero-output residual branch after each full DeBERTa-v2 layer.
2. **disabled adapter128**: same custom architecture, bottleneck 128, but `adapter_enabled=0`, so the path returns exact zero throughout. This controls gradient checkpointing, custom-model serialization, extra unused parameters, and the wrapper implementation.

Both arms use the protected spatial repair route status legal compact-view-reinvest stream, compliant 16k tokenizer, 8×480 DeBERTa-v2, AdamW LR=0.001, warmup_fraction=0.06, WWM 0.15, batch 256, seed 43, init seed 43022, train RNG seed 43023, gradient checkpointing enabled, and `lr_total_steps=2529` to reproduce the original 100M schedule. Stop after the exact spatial repair route status 20M checkpoint boundary, 20,008,711 words (first 506 batches / 129,536 rows), not a full 80M/100M endpoint.

## What it decides

- If the disabled path matches spatial repair route status 20M behavior while the live path still loses broadly, the simple post-layer residual adapter is genuinely interfering or failing to recruit usefully even under the correct schedule; do not continue it to 80M.
- If the live path recovers to roughly disabled/spatial repair route status 20M and shows nonzero adapter output with moderate backbone displacement, adapter 20M closure mainly measured the compressed schedule; the architecture route remains open and may deserve a later-emergence continuation.
- If the disabled path itself deviates strongly from spatial repair route status, the execution setting rather than the adapter mechanism is suspect, so interpret live-vs-disabled rather than live-vs-spatial repair route status and repair the implementation before any longer run.

## Measurements

Cheap official-compatible columns at `chck_20M`: BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA_parallel/nonparallel, Reading. Mechanism readout: adapter output RMS on fixed sentences, adapter up/down weight norms, and stock-backbone displacement live vs disabled and vs spatial repair route status 20M.

This is the lowest-cost reliable experiment that can correct or support the adapter 20M closure architecture inference without launching an 80M/100M endpoint.
