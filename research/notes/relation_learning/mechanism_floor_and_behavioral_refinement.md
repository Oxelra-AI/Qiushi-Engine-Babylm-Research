# mechanism floor and behavioral refinement mechanism, duplicate-run floor, and behavioral refinement

## Source-token mass separates identity misfire from pure displacement

The seed43222 entity clean integration source-token readout scored 2,585 held-out compact-rewrite token-nonoverlap target masks at 80M/90M/100M for each DeBERTa V/C/R seed. The target token is absent from the source-token set, so probability mass on source tokens is not correct content use.

Across seeds, REPEAT-CLEAN source-token mass deltas are +0.1035, +0.1418, +0.0927, mean +0.1127. REPEAT also lowers target probability relative to CLEAN by mean -0.0092 and raises the top-1-is-source rate by mean +0.0889.
VIEW is different: V-C source-mass delta mean is only -0.0012, while V-C target-probability delta mean is +0.0576. Thus VIEW improves the correct nonoverlap target without pulling mass to source tokens, whereas REPEAT pulls mass to source tokens and suppresses the target.
With only three seeds, R-C source-mass elevation and the existing R-C excess true-source cost have Pearson r=0.853; treat this as consistency, not a precise scaling law.

Scientific reading: the active recurrence cost is accompanied by output-level identity competition/misfire, not merely by an absence of content reading. This agrees in form with functional_learning #403's synthetic result that identity can create copy-output competition, while this BabyLM readout supplies the natural-language masked-LM source-token mass evidence directly.

## Entity R versus C should be stated as retrieval benefit plus update cost

The third CLEAN seed changes the Entity framing. The stable behavioral fact is not a three-seed VIEW-over-CLEAN benefit. It is that exact recurrence raises unchanged-state retrieval relative to CLEAN, while after the queried state is updated REPEAT is at or below CLEAN on the recorded aggregate cells.

| seed | group | n | R-C accuracy delta | R acc | C acc |
|---:|---|---:|---:|---:|---:|
| 43022 | rel_updates_0 | 1541 | +8.78 | 48.24 | 39.45 |
| 43022 | rel_updates_1 | 1323 | -1.39 | 14.08 | 15.47 |
| 43022 | rel_updates_2 | 1266 | -0.18 | 18.80 | 18.98 |
| 43022 | rel_updates_3 | 1230 | -5.83 | 17.80 | 23.63 |
| 43022 | rel_updates_4 | 1112 | -1.59 | 23.71 | 25.30 |
| 43022 | rel_updates_5 | 308 | -2.06 | 21.65 | 23.70 |
| 43022 | rel_ge3_weighted_from_rel_updates_3_5 | 2650 | -3.61 | 20.73 | 24.34 |
| 43122 | rel_updates_0 | 1541 | +9.09 | 48.39 | 39.30 |
| 43122 | rel_updates_1 | 1323 | -3.22 | 14.11 | 17.33 |
| 43122 | rel_updates_2 | 1266 | -1.79 | 18.14 | 19.93 |
| 43122 | rel_updates_3 | 1230 | -3.09 | 18.43 | 21.52 |
| 43122 | rel_updates_4 | 1112 | -0.24 | 23.86 | 24.10 |
| 43122 | rel_updates_5 | 308 | +2.06 | 27.71 | 25.65 |
| 43122 | rel_ge3_weighted_from_rel_updates_3_5 | 2650 | -1.30 | 21.79 | 23.08 |
| 43222 | rel_updates_0 | 1541 | +3.11 | 39.56 | 36.45 |
| 43222 | rel_updates_1 | 1323 | -1.01 | 15.75 | 16.75 |
| 43222 | rel_updates_2 | 1266 | -0.97 | 19.93 | 20.91 |
| 43222 | rel_updates_3 | 1230 | -3.33 | 21.76 | 25.09 |
| 43222 | rel_updates_4 | 1112 | -5.64 | 25.57 | 31.21 |
| 43222 | rel_updates_5 | 308 | -5.09 | 24.89 | 29.98 |
| 43222 | rel_ge3 | 2650 | -4.50 | 23.72 | 28.23 |

This preserves Entity as a behavioral face of the cost center: identity practice buys retrieval of an unchanged stated fact and can hurt discrimination once a state has been superseded. The held-out copy/rewrite probes remain the primary installed-competence evidence because their three-seed magnitudes are more stable.

## Duplicate CLEAN run did not introduce an additional run-nondeterminism floor

The default earlier analysis CLEAN seed43222 and the roberta probe result parallel CLEAN seed43222 checkpoint files are byte-identical for model.safetensors, config.json, and tokenizer.json at 80M/90M/100M: True. Entity scores are identical at 26.41/27.15/26.59, with max absolute delta 0.000000. Probe summary files are also byte-identical: True.
This means the duplicate CLEAN does not set a new stochastic floor; under this recipe and hardware path the same seed/data run is deterministic at the measured checkpoints. The hierarchy of evidence should therefore be based on seed-to-seed variation and probe specificity, not duplicate-run drift.

## 1.82x DeBERTa pair preserves the V/R relation signature

The existing 1.82x DeBERTa VIEW/REPEAT pair gives R-V held-out copy gain +0.1789, V-R rewrite gain +1.3267 overall and +1.5974 on token-nonoverlap targets. The Entity cue-ablation rel_ge3 V-R effect when all relevant updates are removed is +1.0080; R-V no-initial effect is +0.3942.
Thus the relation-specific V/R split is not unique to the 2.64x MAX dose, although this two-arm lower-dose check lacks a matched CLEAN baseline for active-cost statements.

## Open causal work

Both split arms, REPEAT_SPLIT and VIEW_SPLIT, were training at this stage. Their results require the held-out copy/rewrite framework and the predictions in `notes/split_control_numeric_prestatement.md`. Locality cannot be inferred before those results are available and scored.

Data outputs: `experiments/archive/relation_learning/data/mechanism_floor_integration`.
