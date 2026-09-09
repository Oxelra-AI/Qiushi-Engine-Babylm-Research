# scale1p75 pre80m state — scale1.75 adapter pre-80M state

## Scientific question

Does function-preserving residual side capacity with fixed train-time branch amplitude 1.75 create a mature broad sample-efficiency gain on the legal spatial repair route status compact-view-reinvestment substrate, or is the 20M/50M signal another competence redistribution that will reverse by 70–80M?

This matters because the best complete legal endpoint remains spatial repair route status Overall 41.257770896404615, below 41.8 by 0.542229. The compact-view reinvestment corpus mechanism is protected, but many optimizer/target/tokenizer/architecture variants have failed or reversed at maturity. Scale1.75 is the first fair adapter setting to produce a score-bearing positive signal, but it must survive mature exposure before any full endpoint work.

## Exact-prefix replication

Evidence: `data/scale1p75_prefix_repro/scale1p75_prefix_repro.{json,md}`.

The 50M scale1.75 run embeds the exact standalone 20M scale1.75 trajectory:

- normalized train rows through 20M: 506 vs 506, first mismatches 0 after dropping elapsed seconds;
- last 20M row identical: earlier analysis, 20,008,711 words, loss 3.765514850616455, lr 0.0009460119426666896;
- `chck_20M/model.safetensors` hash identical;
- all 218 tensors exactly equal, global L2 difference 0.

Thus the adapter scale sweep plan 20M +0.642 cheap7 result is not a launch-to-launch execution discrepancy.

## Matched 20/30/40/50M trajectory

Evidence: `data/scale1p75_trajectory/scale1p75_trajectory.{json,md}`.

| exposure | spatial repair route status cheap7 | scale1.75 cheap7 | delta | main positive/negative structure |
|---|---:|---:|---:|---|
| 20M | 39.6636 | 40.3057 | +0.6421 | EWoK +1.49, GlobalPIQA +2.955, Supplement +0.82; Entity -0.43, COMPS -0.29 |
| 30M | 41.1943 | 41.1571 | -0.0371 | EWoK +0.98, Supplement +0.66; GlobalPIQA -0.91, COMPS -0.48 |
| 40M | 42.2214 | 41.7464 | -0.4750 | BLiMP +0.78, Entity +1.62, EWoK +0.18; GlobalPIQA -3.955, Reading -0.89 |
| 50M | 41.9729 | 42.3843 | +0.4114 | BLiMP +1.84, Entity +2.79, COMPS +0.73; EWoK -0.73, GlobalPIQA -1.475, Reading -0.235 |

The trajectory is nonmonotone. The positive 50M point is not an isolated execution artifact because the 20M prefix is exact, but the route is not broadly clean: EWoK and GlobalPIQA trade against Entity/BLiMP/COMPS at 50M.

## 50M mechanism and subtask readout

Evidence:

- `data/scale1p75_50m_mechanism/scale1p75_50m_mechanism.{json,md}`
- `data/subtask_report_trajectory/subtask_report_trajectory.{json,md}`

Mechanism facts:

- scale1.75 stock 20M vs spatial repair route status 20M: cosine 0.8791, relL2 0.4913;
- scale1.75 stock 50M vs spatial repair route status 50M: cosine 0.7559, relL2 0.6982;
- scale1.75 20M→50M stock update norm / spatial repair route status 20M→50M stock update norm = 0.9991;
- cosine between scale1.75 and spatial repair route status 20M→50M stock updates = 0.3050;
- adapter all-parameter RMS is nearly stable from 20M to 50M (0.066006→0.066315), while adapter up RMS grows 0.012710→0.018637;
- core stock-matrix spectra remain close to spatial repair route status (stable-rank mean spatial repair route status 50M 32.754, scale1.75 50M 32.963).

Interpretation: the side path does not simply add inert capacity or Muon-like spectral broadening. It co-trains the stock backbone along a similarly long but different direction.

Subtask report facts at 50M:

- strongest positives: BLiMP superlative_quantifiers_1 +24.10, wh_questions_object_gap +21.07, superlative_quantifiers_2 +19.78, EWoK physical-dynamics +15.83, Entity move_contents_5_ops +12.07, Entity regular_4_ops +8.76, regular_0_ops +8.51;
- strongest negatives: EWoK active-passive -16.66, game -15.00, number -10.00, material-properties -9.41, quantitative-properties -6.05; GlobalPIQA parallel -1.95, nonparallel -1.0;
- selected delta trajectories show physical-dynamics stays strongly positive (+12.50, +6.67, +13.34, +15.83 from 20→50M), while material/number/active-passive deteriorate after 20M.

Thus the route specifically improves high-operation entity-state tracking and physical dynamics, not EWoK as a whole. Mature 70/80M scores must decide whether those gains persist while EWoK/GlobalPIQA recover or whether the route becomes another redistribution.

## 80M bounded mature test

A first scale1p75 pre80m state 80M launch attempt failed before training because of wrong flags. Corrected command copied from the successful 50M run and changed only output dir, label, and exposure boundary. The active corrected training task is:

- Training configuration: `Train adapter128 scale1p75 exact spatial repair route status-80M GPU0`
- run dir: `training/runs/adapter128_scale1p75_h100M80M_seed43022`
- exact target exposure: 80,034,368 words, matching spatial repair route status `chck_80M`
- same legal tokenizer, compact-view reinvestment stream, DeBERTa 8×480, AdamW/WWM recipe, seeds 43/43022/43023, and lr_total_steps=2529.

A managed wait/eval task is also running:

- Evaluation configuration: `Wait-eval scale1p75 70M80M GPU1`
- output: `data/scale1p75_70_80_eval/`
- it waits for `chck_70M` and `chck_80M` and evaluates only cheap official-compatible columns for the adapter arm. spatial repair route status 70/80 references are already in `data/legal_mature_treatment_effect_eval/per_target/`.

A mature merger is ready:

- `scripts/merge_scale1p75_mature.py`
- output: `data/scale1p75_mature_decision/`

Do not launch 100M/full SuperGLUE/AoA before reading the 70/80M adapter scores and matched deltas. If 80M retains a meaningful cheap7 advantage with no enlarged EWoK/GlobalPIQA/Reading collapse, this exact trajectory deserves 100M and then full official-compatible evaluation. If 80M is negative or the tradeoff deepens, stop fixed scale1.75 as another mature redistribution and hand to construction for a more protected side-path or adaptive-amplitude mechanism.
