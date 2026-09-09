# Route evidence synthesis

This historical note records completed measurements and remaining uncertainties at the point when the format-attribution controls had been integrated. Pending measurements below describe that point in the research chronology, not their later outcomes.

## Faithful v4 is now numerically defined at seed42

Completed faithful AutoModel SuperGLUE evaluation established the following reference values:

- coherent86 alpha0.75 repaired AutoModel SuperGLUE: `68.94571192183594`
- coherent86 cheap7: `44.18142857142857`
- coherent86 AoA: `0.0`
- faithful coherent86 Overall: `(7*44.18142857142857 + 68.94571192183594 + 0)/9 = 42.023967991315104`
- chck82 repaired AutoModel SuperGLUE: `69.47199304980661`
- chck82 Overall with cheap7/AoA: `41.90979357610763`

This is consistent with the earlier repaired-coordinate evaluation. The historical `42.1210` used stripped AutoModel SuperGLUE and is not the faithful v4 bar. The bridge/fidelity movement is not a training improvement.

Key files:

- `experiments/archive/relation_learning/data/faithful_superglue/coherent86_alpha075/faithful_superglue_summary.json`
- `experiments/archive/relation_learning/data/faithful_superglue/chck82_scale1p75/faithful_superglue_summary.json`
- consolidated table: `research/documents/relation_learning/data/faithful_format_boundary_synthesis/summary.md`

## Dense remains a replicated trade under the current rule

The earlier dense-continuation comparison measured two seeds against faithful coherent86:

- coherent86 Overall `42.023967991315104`
- dense62064 `42.14909111936738`, delta `+0.125123128052273`
- dense62065 `42.168422702303516`, delta `+0.144454710988413`
- mean delta `+0.134788919520343`
- AoA remains `0.0` for all.

Stable gains: Entity `+1.05/+1.10`, GlobalPIQA `+1.485/+1.485`, small COMPS/Reading gains. Stable costs: BLiMP `-0.45/-0.49`, Supplement `-0.60/-0.55`, EWoK `-0.10/-0.25`; SuperGLUE negative and seed-sensitive (`-0.414/-0.140`). This is not an admitted v5 component under the earlier analysis rule because the macro gain is inside the coherent private-seed band while important column losses exceed their bands.

The dense-mask/sparse-label contrast is important mechanistically. Dense-mask/sparse-label `(M,S)` reproduces the dense source/rank signature while dense labels add almost nothing on those diagnostics:

- `(M,S)-(S,S)` Qwen Tfit `+0.1447`, rank `+72.38`; common source-follow swing Tfit `+0.5894`, rank `+115.56`.
- `(M,S)-(M,M)` Qwen Tfit about `-0.005`, rank `-2.92`; common source-follow swing about `+0.0058`, rank `+2.59`.
- CDI endpoint NLL/rank costs persist for dense and `(M,S)`; dense input corruption/context availability, not broad dense target labels, appears to carry the source-responsive shift.

The preservation-control audit corrected the interpretation of the earlier preservation arms: full-row standard-WWM preservation still contains source evidence, the earlier implementation confounded lambda with global seed and auxiliary train-mode RNG, and train/eval identical-weight KL has a large dropout component (`~0.1806` nats/target). Clean eval-mode/train-mode preservation runs had started, but their results were not yet incorporated in this note.

Key files:

- `experiments/archive/functional_learning/data/trusted_comparison_two_seed/two_seed_measured_comparison.json`
- `research/documents/functional_learning/data/densemask_causal_reader/full_after_delivery/densemask_causal_reader.md`
- `research/documents/functional_learning/data/source_margin_decomposition_five_models/source_margin_decomposition_five_models.md`
- `research/documents/functional_learning/data/cdi_endpoint_temperature_rank_profile_five_models/cdi_endpoint_temperature_rank_profile_five_models.md`
- `research/notes/functional_learning/preservation_controls_and_corrected_interpretation.md`

## Format/boundary controls close simple private-phase geometry repair as a direct lever

By the time of this note, the no-special coherent and embedding-row-only controls were complete and had been consolidated with the other format arms.

Family means versus faithful coherent86:

| family | mean cheap7 | delta cheap7 | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coherent_unsplit_no_special_control | 44.0168 | -0.1646 | -0.1200 | -1.0650 | -0.4200 | +0.0650 | -0.0400 | +0.4925 | -0.0650 |
| coherent_unsplit_special | 43.6689 | -0.5125 | +0.3300 | -2.7750 | -0.5650 | +0.0300 | +0.2100 | -0.7450 | -0.0725 |
| half_coherent_half_isolated | 43.7521 | -0.4293 | +1.5200 | -3.1950 | +0.4950 | +0.2250 | +0.3750 | -2.7100 | +0.2850 |
| isolated_all | 43.5136 | -0.6679 | +1.5300 | -2.7650 | +0.7350 | -1.5750 | +0.3050 | -3.2100 | +0.3050 |
| special_embedding_only_control | 43.9300 | -0.2514 | +0.3950 | -1.9400 | -0.6050 | +0.8500 | +0.3000 | -0.9850 | +0.2250 |

No-special coherent replay is a trainer-mechanism control: technically clean, lower KL than special-token arm, but not a component. Special embedding-only changes just `<s>`/`</s>` tied embedding rows and produces stable BLiMP/Entity/Reading improvements, but also large Supplement/EWoK and GlobalPIQA costs. Thus the special-token edge channel alone cannot be used as a safe repair. Half-format and isolated-all remain broader format/context redistributions, not admitted gains.

The full no-boundary cheap7 diagnostic was pending at the time of this note. A preliminary no-Reading run had been stopped during BLiMP before completion; it did not supply a complete diagnostic result.

Key files:

- `research/documents/relation_learning/data/faithful_format_boundary_synthesis/summary.md`
- pending full no-boundary diagnostic output root: `experiments/archive/relation_learning/data/no_boundary_cheap7_full`

## AoA: pass-1 timing instruments corrected, but simple legal timing proxies do not yet justify trunk training

The earlier half-exposure crossing instrument cannot see within-pass order under ten identical passes. The corrected instruments therefore use pass-1-only, ladder-resolution measurements. They change only first-pass row order and leave checkpoints after 10M unchanged in exposure counts.

1. Fixed-effect exposure model:

- file: `research/documents/relation_learning/data/aoa_pass1_order_predictor/summary.md`
- model: word effect + checkpoint effect + shared `beta*log1p(cumulative exposure)` fitted to coherent86 mean surprisals
- validation against measured model AoA: r `0.5866` (orientation differs from frequency baseline, but magnitude is moderate), compared with whole-stream frequency vs measured model AoA r `-0.6214`
- current predicted official-like score clips to `0.0`, r `-0.0193`, n `245`
- best legal pass-1 candidate under this model: `sort__spoken_freq`, clipped `0.0`, unclipped r `0.0445`, p `0.478`, n `256`; all legal candidates clip to zero.

2. Sensitivity on actual coherent86 mean curves:

- file: `research/documents/relation_learning/data/aoa_pass1_sensitivity_parallel/summary.md`
- method: actual mean-surprisal curves plus `beta * (log exposure_candidate - log exposure_current)` at 1M..10M; after 10M deltas exactly zero.
- reproduces earlier analysis coherent86 exactly: r `-0.034854`, p `0.603`, n `225`, clipped `0.0`.
- across beta grid `[-0.1,-0.2,-0.36,-0.5,-0.75,-1.0,-1.5]`, all legal candidates clip to zero; best legal rows remain near zero or negative and non-significant.
- even oracle row sorting by child AoA is not a positive ceiling under this row-score/exposure law; at strong beta it becomes significant negative (`-0.131` at beta `-1.0`, p `0.053`; `-0.118` at beta `-1.5`, p `0.091`). This should not be overinterpreted as impossibility of AoA improvement, but it rejects the simple row-order/frequency-proxy route as currently instrumented.

Interpretation: AoA remains the largest structural opportunity, but these analyses did not produce a reliable positive predictor. They did not justify 10M or 100M trunk AoA training from simple CHILDES/spoken/whole-frequency pass-1 sorting alone. A future AoA route needs a stronger acquisition-order model (for example, modeling boundary-token/lower-surprisal range effects, lexical concreteness/length/class if legally available or model-internal, or a designed early objective) that predicts positive official score movement before large-scale trunk training.

## Outstanding measurements at the time

- Faithful coherent86 SuperGLUE ftseed43 was pending as a comparison to seed42 for estimating downstream seed spread.
- The full no-boundary cheap7 diagnostic was pending.

The coherent86 SuperGLUE seed42 reference, no-special trainer control and embedding-only control had been completed and are summarized above.
