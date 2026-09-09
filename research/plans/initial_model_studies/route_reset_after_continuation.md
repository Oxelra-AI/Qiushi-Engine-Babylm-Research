# earlier analysis — Route reset after strict continuation failure

## What changed

The strict prefix-continuation experiment did exactly what it was designed to do computationally, but not scientifically.

- The 3D prefix-causal mask was structurally sound: future suffix tokens had exactly zero effect on earlier continuation logits, while prefix and earlier suffix tokens affected them.
- The 1M run trained a dense directional signal: continuation loss fell from 9.80 to 5.98 and supplied about 4× as many supervised targets as WWM.
- Yet the fast profile versus matched WWM was mostly worse: BLiMP −0.56, Supplement −0.80, EWoK +0.28, Entity −0.48, COMPS −0.66, Reading −0.155.
- The prefix probe was strongly negative for the desired same-source effect: continuation-minus-WWM same-vs-cross specificity mean −0.735.

This closes the current strict continuation formulation for scaling. It also sharpens a more important unresolved issue: the probes used in RTD, CPC, and continuation mostly sampled ordinary later tokens that were nearly prefix-insensitive under WWM. In cont profile and prefix probe the WWM sensitivity score averaged only about 0.00038 log-prob, and even the high quartile averaged about 0.00119. A route can fail such a probe because it lacks the mechanism, or because the probe target population almost never depends on earlier context. We have not separated those possibilities.

## Immediate route decision

Do **not** launch another broad auxiliary loss, another CPC margin/lambda/source variant, another continuation weighting run, or a 3M/100M extension.

The next work should be no-training measurement plus route selection:

1. A prefix-dependent target census on official text, using existing WWM checkpoints.
2. A measurement recheck for the current-best endpoint and the local score definition.
3. Only after those, choose between representation/cadence/tokenizer work or a high-precision prefix-target route.

## Work item 1 — Prefix-dependent target census

### Purpose

Find out whether official BabyLM text contains a usable population of later tokens whose MLM gold-token log-prob is actually helped by the true earlier context under existing strong WWM models.

This is not training data construction. It is a measurement of whether the cross-sentence credit-assignment route has a real target population.

### Models

Use exact checkpoint directories, never parent `hf_model` plus local `revision`.

Primary two independent WWM seeds:

- seed42 full-cycle WWM: `training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_40M`, `chck_80M`, `chck_100M`
- seed43 full-cycle WWM: `training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_40M`, `chck_80M`, `chck_100M`

Optional low-exposure comparison:

- matched prior data event binding panel WWM 1M: `training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched/hf_model/chck_1M`

Do not call seeds 456/789 independent WWM seeds; those are initialization/training RNG components inside runs.

### Text samples

Run the census in two layers:

1. The exact 1M official slice used by prior data event binding panel (mostly BNC + CHILDES), so it directly interprets the recent failed mechanisms.
2. A broader sample from all six Strict-Small official files, so the route does not overfit to the early slice selected by the 1M experiments.

Construct examples by the same tokenizer and word accounting conventions as `babylm_masked_train.py`. For each example, split by token position into prefix and suffix. Choose later target tokens using only pre-model information:

- alphabetic content-ish tokens;
- token frequency bucket;
- subword length and whether the surface word is split;
- source file;
- distance from prefix boundary;
- repeated capitalized/string entity presence in prefix and suffix when detectable by simple rules;
- local-context leakage proxies such as repeated exact target in nearby suffix window.

Avoid selecting targets because one model already shows a large response, except for a separate held-out analysis. The main census must report model-agnostic strata.

### Variants to score

For every target, score the gold target log-prob under:

- `full`: true prefix + suffix with target masked;
- `deleted`: suffix only;
- `block_shuffled`: prefix block-shuffled;
- `same_source_near`: prefix from another same-source example with similar prefix length;
- `cross_source_near`: prefix from a different source with similar prefix length.

For each WWM checkpoint, compute:

- `full - deleted`;
- `full - block_shuffled`;
- `full - same_source_near`;
- `full - cross_source_near`;
- `(full - same_source_near) - (full - cross_source_near)`.

### Main outputs

Report counts and fractions at thresholds |delta| ≥ 0.02, 0.05, 0.10, and 0.20 log-prob, with same sign across seed42 and seed43 at the same exposure. Report separately for 40M, 80M, and 100M.

The most important quantity is the fraction of targets where the true prefix helps over `same_source_near`, not merely over `cross_source_near`.

Save:

- per-target rows to `data/prefix_target_census_rows.csv`;
- aggregate JSON to `data/prefix_target_census_summary.json`;
- concise interpretation note to `notes/prefix_target_census.md`.

### Route consequences

If sign-stable true-prefix-over-same-source targets are extremely rare, stop broad cross-sentence objective work and move to representation/cadence/tokenizer/architecture routes.

If a usable target population exists, do not immediately train on it. First inspect examples and define a high-precision evaluation surface. Then any future CPC/continuation variant must improve that surface under candidate-minus-WWM comparisons.

## Work item 2 — Measurement recheck

The prior summary identifies `wwm_seed43 chck_80M` is the current best internal coordinate, but fineweb relation vs random 3m direct checkpoint trajectory found a local checkpoint loading artifact and explicitly warned that multi-checkpoint endpoint evidence needed rechecking. A separate prior summary claims that the best checkpoint was validated, but that claim requires an attributable validation result or a direct-checkpoint comparison; it is not established by the summary alone.

Concrete work:

- Verify exact path `training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_80M` scored as the new best verification and assessment current best.
- Verify `chck_100M` from the same run with direct checkpoint paths.
- Do this for the columns involved in endpoint choice at minimum: BLiMP, Supplement, EWoK, Entity, COMPS, Reading, and if affordable SuperGLUE/GlobalPIQA.
- Save a short note at `notes/current_best_direct_checkpoint_recheck.md`.

This protects the SOTA gap estimate and prevents endpoint artifact from driving route choices.

## Work item 3 — Next training route after the census

The strongest training routes left should be ranked after the census, not before.

Candidate ordering from the accumulated experimental evidence:

1. **Tokenizer / word-morph anchor or morphology-aware representation**: field evidence points to Entity/EWoK benefits from morphology/tokenization, and this attacks a different bottleneck: low-frequency word/entity parameter sharing. First use corpus/task tokenization statistics before training.
2. **Factorized WWM cadence**: sequence length schedule, mask-rate decay, and smaller effective batch are supported by BabyLM findings and do not require fragile prefix-negative construction. Test factors separately.
3. **Cross-block sparse memory**: scientifically aligned with Entity, but heavier engineering and must include persistent-vs-reset/random-carry comparisons.
4. **High-precision prefix-target CPC/continuation**: only if the census finds enough sign-stable true-prefix targets and example inspection supports that they are not source/register artifacts.

Do not combine these in a single first run. The next training experiment should isolate one active factor with a matched WWM control and two-seed 1M/3M evidence before larger compute.
