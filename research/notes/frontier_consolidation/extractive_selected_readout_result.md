# extractive selected readout result — source-only extractive selected readout result

## What was tested

The earlier analysis balanced and wide source-only extractive DeBERTa arms finished cleanly and were read with the predeclared lowest-cost selected panel: `chck_80M` and `chck_100M` for `extractive_balanced`, `extractive_wide`, and reused `legal_compact` references. No SuperGLUE, AoA, upload, or leaderboard action occurred.

Integrity carrier: `data/extractive_training_integrity_final/extractive_training_integrity.json`.
Selected panel: `data/extractive_selected_eval_stage1/extractive_selected_panel_summary.json`.
Paired intervals: `data/extractive_selected_interval_analysis/`.

## Mechanical status

Both extractive runs match the intended compliant tokenizer retrain status stock DeBERTa coordinate:

- 100,000,000 counted words, 2,529 updates, 10 saved checkpoints through `chck_100M`;
- legal tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`;
- 34,467,424 parameters, 8 layers, width 480, 8 heads;
- fixed WWM p=0.15, AdamW lr=0.001, batch 256, seq 256;
- balanced stream SHA `8a4e77af3ae7049b889e5d8f174616606a930bcd8371188ea35db3197175776c`;
- wide stream SHA `b9e4666844f966a2fedbe6a37eb61aa3b1bcd5fe08cdc738cd0a15fda4258753`.

Terminal training losses were balanced 2.5676 and wide 2.5328. These losses are not used as the learning-capability readout.

## Stage-1 selected scores

All deltas below are extractive minus legal compact at the same checkpoint.

### `chck_80M`

| arm | cheap7 | cheap6 no GlobalPIQA | cheap5 no GlobalPIQA/Reading | EWoK+Entity | Supplement | Entity | COMPS | GlobalPIQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | +0.260 | -0.198 | -0.234 | -0.940 | +0.480 | -0.550 | -0.740 | +3.010 |
| wide | -0.119 | -0.807 | -0.888 | -3.270 | -2.100 | -2.300 | -0.010 | +4.010 |

### `chck_100M`

| arm | cheap7 | cheap6 no GlobalPIQA | cheap5 no GlobalPIQA/Reading | EWoK+Entity | Supplement | Entity | COMPS | GlobalPIQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | +0.231 | -0.316 | -0.340 | -2.220 | +0.760 | -0.970 | -0.640 | +3.515 |
| wide | -0.077 | -0.516 | -0.604 | -2.370 | -2.030 | -2.410 | -0.060 | +2.555 |

### Mean over 80M/100M

| arm | cheap7 | cheap6 no GlobalPIQA | cheap5 no GlobalPIQA/Reading | EWoK+Entity | Supplement | Entity | COMPS | BLiMP | GlobalPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced - compact | +0.246 | -0.257 | -0.287 | -1.580 | +0.620 | -0.760 | -0.690 | +0.215 | +3.263 | -0.107 |
| wide - compact | -0.098 | -0.662 | -0.746 | -2.820 | -2.065 | -2.355 | -0.035 | +1.190 | +3.283 | -0.240 |

## Paired item intervals

The interval analyzer confirms complete common selected item coverage: 170,722 common items, zero loader warnings for every extractive-vs-compact pair. Wrapper/official selected scores remain primary because BabyLM aggregation is not identical to raw item averaging.

Interval details that matter for interpretation:

- balanced 100M: EWoK+Entity raw item delta -1.118 pp with item interval [-1.852, -0.395] and cluster interval [-1.870, -0.305]; BLiMP raw item delta +0.429 pp; GlobalPIQA raw item point +3.448 pp with wide intervals because it has only 203 item rows.
- wide 100M: EWoK+Entity raw item delta -1.056 pp with item interval [-1.747, -0.221]; Entity raw item delta -1.726 pp; BLiMP raw item delta +1.425 pp with positive item and cluster intervals.
- Supplement raw item movement is negative for both extractive arms at both checkpoints even when balanced's official Supplement score is positive. This mismatch reinforces using official/wrapper scores for aggregate movement and raw intervals for item-motion structure.

## Scientific reading

The source-only arms do not reproduce natural compact's stable selected competence. Balanced is the closer source-only approximation and even beats compact on cheap7, but the cheap7 gain is carried by GlobalPIQA and partly BLiMP/Supplement; the stable composite excluding GlobalPIQA is below compact at both 80M and 100M, with the relation/state composite (EWoK+Entity) lower at both checkpoints. Wide has broader source coverage than balanced and compact, but it is worse on the stable composite and much worse on Entity and Supplement.

This resolves the immediate source-selection-sufficiency question: in the same legal stock-DeBERTa coordinate, simply selecting original source words at compact-like or wider coverage is not sufficient to recover the natural compact-view benefit on stable official-compatible families. The result does **not** isolate a single missing factor. Natural compact differs from extractive in continuity/fluency, source-absent generated content, token geometry, density, and their coupling. The result only rules out the explanation that missing source/tail access alone accounts for compact's advantage.

Balanced being closer than wide suggests that compact-like density/token geometry is more useful than maximum source span under this coordinate. Wide's high BLiMP and GlobalPIQA together with poor Entity/Supplement suggests a redistribution toward broad lexical/syntax surfaces while damaging relation/state and fine-grained selected competence.

## Next research implication

Do not spend on 60M/70M/90M for this same source-only pair unless a later reviewer finds a concrete contradiction in the 80M/100M readout. The two read checkpoints agree on the stable-family direction, and the predeclared readout already distinguishes the route-relevant cases.

The next same-coordinate mechanism work, if pursued, should be designed from the observed deficit pattern rather than the earlier over-clean factor table. Useful separations would target continuity/fluency, generated source-absent content, and BPE/token-geometry coupling inside DeBERTa with the lowest reliable cost before any full 100M training. RoBERTa factorial should be used later as cross-coordinate survival evidence, not as a numerical decomposition of the DeBERTa effect.
