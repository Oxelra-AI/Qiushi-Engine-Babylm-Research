# adapter matched horizon plan — matched-horizon adapter result and revised architecture interpretation

## Question corrected from adapter 20M closure

adapter 20M closure trained zero-output residual adapters for 20M words with a standalone 20M cosine schedule. Because the adapter branch starts at exact zero output and must first recruit, that compressed schedule was not a fair test against spatial repair route status 20M, which is an early checkpoint on the original 100M LR horizon. adapter matched horizon plan therefore ran the minimum matched experiment on the protected spatial repair route status legal compact-view-reinvest substrate.

## Matched experiment

Both arms use the exact spatial repair route status legal16k compact-view-reinvest data stream and training recipe, with gradient checkpointing and custom adapter model wrapper:

- 8×480 DeBERTa-v2 MLM, compliant 16k tokenizer, WWM 0.15, AdamW lr=0.001, warmup_fraction=0.06, batch 256, seed 43, init seed 43022, train RNG seed 43023.
- `lr_total_steps=2529`, matching the original 100M schedule.
- Stop at the exact spatial repair route status 20M checkpoint exposure: 20,008,711 words, 506 batches.
- Live arm: bottleneck 128 adapter after each complete encoder layer, zero-output initialized, enabled.
- Disabled arm: same architecture and unused parameters, but adapter returns exact zero throughout.

## Training dynamics

The disabled arm exactly reproduces spatial repair route status watch-step loss, LR, words, and mask counts. Its final loss is 3.755582, identical to spatial repair route status chck_20M. The live arm also returns to spatial repair route status-scale learning: final loss 3.756545, only +0.000962 versus spatial repair route status. This directly shows that adapter 20M closure's final losses near 6.88 were caused by the standalone 20M LR schedule, not an inherent inability of the adapter model to train.

Evidence: `data/adapter_training_dynamics/adapter_training_dynamics.md`.

## Cheap official-compatible behavior at chck_20M

| arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δcheap7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial repair route status/disabled | 59.69 | 55.45 | 50.73 | 18.65 | 50.26 | 34.20 | 8.67 | 39.6636 | 0.0000 |
| adapter 20M closure adapter128 compressed | 53.81 | 50.13 | 51.08 | 17.07 | 50.34 | 33.75 | 7.17 | 37.6214 | -2.0421 |
| adapter matched horizon plan live adapter128 matched horizon | 60.38 | 56.23 | 49.49 | 18.65 | 50.44 | 32.23 | 8.31 | 39.3893 | -0.2743 |

The corrected schedule removes the catastrophic BLiMP/Supplement failure and makes the branch useful for local syntax-like surfaces: BLiMP +0.69, Supplement +0.78, COMPS +0.18. But it still loses EWoK -1.24, GlobalPIQA -1.97, and Reading -0.36, with Entity unchanged. The result is not a broad collapse, but it is a familiar competence redistribution, not a broad sample-efficiency gain.

Evidence: `data/adapter_matched_horizon_eval/adapter_matched_horizon_summary.{json,md}`.

## Mechanism readout

- Disabled control: adapter RMS mean 0.0, adapter up norm 0.0, stock displacement vs spatial repair route status exactly zero. This validates the execution setting and gradient-checkpoint/custom-model wrapper.
- Live branch: adapter RMS mean 0.05657, max 0.06987; adapter up norm sum 28.23, down norm sum 66.34. The branch is clearly recruited.
- Live stock-backbone displacement vs disabled/spatial repair route status 20M is already large by this crude parameter-space readout: mean tensor cosine 0.8659, mean relative L2 0.4506, max relative L2 1.7559. Thus the live branch is not acting as a purely additive capacity reserve; joint training causes strong co-adaptation of the original backbone.

## Current interpretation

adapter 20M closure's inference was too strong: separately routed residual capacity did not fail catastrophically when trained on the correct horizon. However, the simple joint-trained post-layer adapter does not yet satisfy the reason it was introduced. It recruits and preserves scalar training loss, but it pulls the stock backbone away from the verified spatial repair route status trajectory and exchanges EWoK/GlobalPIQA/Reading for BLiMP/Supplement/COMPS at 20M.

This does not warrant immediate full 80M/100M continuation of the same ordinary live adapter: the minimum matched experiment already shows no broad early gain and shows the same relation/world/reading damage direction seen in other failed modifications. At the same time, it also should not be treated as closing all function-preserving residual-capacity designs. The sharper bottleneck is **gradient and representation isolation**: the added path must be recruited without adding a large extra Jacobian term into the backbone update and without rewriting the compact-view trajectory.

A concrete next architecture refinement, if pursued, should therefore change the coupling rather than merely the bottleneck width: for example, a stop-gradient adapter input `update = Adapter(stopgrad(h))` added back to `h`, or a gated side path whose branch receives features but does not backpropagate its own input Jacobian into earlier layers. The next minimum test should compare such a gradient-isolated live path against the disabled control on the same 100M horizon for 20M, using the same branch-use and stock-displacement readout. The ordinary joint-trained adapter should not be continued to 80M unless independent analysis provides a reason that the EWoK/GlobalPIQA/Reading deficits are expected to reverse rather than mature into another redistribution.

The best legal complete endpoint remains spatial repair route status Overall 41.2578. adapter matched horizon plan produced no new complete endpoint and does not authorize final expression or submission work.
