# mature checkpoint averaging hard surface null — Route C mature checkpoint averaging: hard-surface null

## What was done (no training, no new data, no tokenizer)

From the route portfolio context alternative binding portfolio, Route C (cheap mature-window consolidation) was the only
unexecuted cheap route. I built four predeclared loss-agnostic weight averages of the
existing **legal40k fixed-256 compact-view** trajectory (legal40k accum training completion), preserving the exact
legal tokenizer/config:

- `avg_uniform_80_100`: uniform mean of chck_80M..chck_100M (21 checkpoints)
- `avg_uniform_90_100`: uniform mean of chck_90M..chck_100M (11)
- `avg_exp_70_100_hl10`: late-weighted exponential mean chck_70M..chck_100M, half-life 10M (31)
- `avg_linear_90_100`: linear late-weighted mean chck_90M..chck_100M (11)

Builder: `scripts/mature_checkpoint_averages.py`;
artifacts: `training/runs/mature_checkpoint_averages/<spec>/hf_model/chck_avg`;
manifest: `data/mature_checkpoint_averages/mature_checkpoint_averages_manifest.json`.
All four load cleanly as `AutoModelForMaskedLM` (trust_remote_code) and produce finite
40000-dim logits; mask id 4, pad id 3 as expected.

## Hard-surface readout (CPU-only, validated fw globalpiqa relevant substrate all-option reader)

Reader: `scripts/mature_globalpiqa_reader.py`;
summary: `data/mature_globalpiqa_reader/mature_globalpiqa_summary.json`.

GlobalPIQA_parallel (103 rows) and nonparallel (100 rows), all-option length-normalized
pseudo-log-likelihood:

| target | parallel acc | hard-52 acc | hard-52 mean top-minus-correct (nats) | nonparallel acc |
|---|---|---|---|---|
| anchor_fixed256_100M | 22.33 | 0.0 | 1.9534 | 47.0 |
| avg_uniform_80_100 | 23.30 | 0.0 | 1.9449 | 47.0 |
| avg_uniform_90_100 | 22.33 | 0.0 | 1.9517 | 46.0 |
| avg_exp_70_100_hl10 | 23.30 | 0.0 | 1.9496 | 47.0 |
| avg_linear_90_100 | 22.33 | 0.0 | 1.9524 | 47.0 |

Parallel accuracy moves by at most one row (22.33 -> 23.30). The 52-row cross-endpoint
always-wrong core stays **0/52 correct** in every average, with correct option still at
rank 3/4 and mean top-minus-correct ~1.95 nats — statistically identical to the anchor.
Category structure (spatial 9.1%, direction_spatial 8.3%, tool_affordance 0%,
time/counting 0%) is unchanged. nonparallel margins are unchanged.

## Scientific conclusion

Weight-space averaging of a single compliant legal40k trajectory does **not** touch the
context-conditioned alternative-binding deficit. This is consistent with the whole
experimental record: the hard GlobalPIQA_parallel deep-rank failure and the EWoK stable
conditional reversals are not a variance/consolidation artifact of the mature trajectory;
they are a structural property the trajectory never learns. Averaging blurs likelihood
without creating the missing competition, exactly the failure mode predicted for linear
weight consolidation (fw weight space sweep branch soups also collapsed).

Route C therefore cannot be a hard-surface fix. The only residual value of these averages
is a possible small broad-cheap7 gain (BLiMP/Supplement/COMPS), which does not close the
frontier by itself (the reheat endpoint already reaches broad cheap7 43.4957 and also
leaves the hard surfaces flat). I did **not** spend a GPU lane on the broad-only cheap
eval: GPU1 is fully occupied by the running earlier analysis packet screens and GPU0 is running
scale1.75 mature run; a broad-only measurement is low value relative to that cost.
If a free lane appears, the averages are ready for a bounded cheap eval via
`scripts/screen_readout.py`-style calls or the official collator coordinate seed43022 evaluator,
but this is optional and not decision-changing for the SOTA route.

## Route status after mature checkpoint averaging hard surface null

- Route C (mature checkpoint averaging): **closed as a hard-surface fix.** No effect on
  GlobalPIQA_parallel hard-52 ranks/margins. Broad-cheap7 effect unmeasured and low value.
- Route B (role-switch vs role-fixed 80M from-scratch screen, shared tokenizer): **running**
  (`s147_t30_tool1`, `s147_t31_tool1`), at ~20M words each on GPU1. Readout apparatus ready.
- Route A (protected context-difference logit side path): still the strongest untested
  architecture idea; owned jointly with coupling-controlled adapter machinery.

The central unresolved weakness remains context-conditioned alternative binding
(EWoK stable reversals + GlobalPIQA_parallel deep ranks). No compliant Overall SOTA
endpoint exists; best compliant full endpoint is legal40k 8x480 at Overall 41.1406,
best compliant broad reference is reheat 43.4957 cheap7.
