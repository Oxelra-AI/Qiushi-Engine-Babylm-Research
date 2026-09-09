# Whole-word copied-control integration

## Why this matters now

The compact ordered-vs-scrambled training/evaluation comparison remained in progress. The whole-word copied-content control artifacts were complete and provide evidence about the compact source-absent signal for interpreting the pending comparison.

## earlier analysis strengthens the local source-absent compact-side signal

earlier analysis repaired the copied-label comparator from earlier analysis. Instead of deleting arbitrary copied token pieces, it deletes whole copied-content target-word groups matched to the source-absent content target groups under the exact roberta selected transfer resolution own-visible 20M interface.

Key matching facts from `experiments/archive/representation_and_objectives/data/wholeword_copied_content_selection/wholeword_copied_content_selection_audit.json`:

| quantity | value |
|---|---:|
| absent-content target groups | 5,779 |
| absent-content BPE pieces | 7,649 |
| selected copied-content whole groups | 5,779 |
| selected copied-content BPE pieces | 7,649 |
| exact BPE-length match fraction | 1.0 |
| match class | 5,779 exact-length, same epoch/exposure |
| support log mean delta selected-minus-absent | +0.002011 |
| relative word position delta | -0.001583 |
| row-order fraction delta | +0.000153 |

Training facts from `experiments/archive/representation_and_objectives/training/runs/target_selective_drop_copied_content_wholeword_20M/scientific_metrics.json`:

- Same data/interface/init/masks as the roberta selected transfer resolution/225 20M compact-side geometry.
- Exposure: exactly 20,000,000 words over 578 updates.
- Dropped copied-content whole-word pieces: 7,649.
- Dropped source-absent pieces: 0.
- Parameters: 34,467,424.
- Final loss: 3.53703.

Central fixed-event denoising result from `experiments/archive/representation_and_objectives/data/wholeword_control_readout/wholeword_control_readout.json`:

| checkpoint | category | drop_abs minus drop_copied_word piece-weighted delta | 95% interval | fraction > 0 |
|---|---|---:|---|---:|
| 10M | source_absent_content | +0.083625 | [+0.069129,+0.098691] | 1.0 |
| 10M | retained_content | +0.051188 | [+0.037438,+0.064687] | 1.0 |
| 10M | function_other | -0.053565 | [-0.070833,-0.037231] | 0.0 |
| 20M | source_absent_content | +0.080267 | [+0.058346,+0.102292] | 1.0 |
| 20M | retained_content | -0.067648 | [-0.091635,-0.046236] | 0.0 |
| 20M | function_other | -0.053149 | [-0.075724,-0.028516] | 0.0 |

At 20M, the source-absent category interaction is strong: source_absent_content exceeds retained_content by +0.147915 nats and exceeds function_other by +0.133416 nats. This means the earlier analysis local signal survives a stronger whole-word copied-content comparator and becomes more category-selective at 20M.

Exposure split from `experiments/archive/representation_and_objectives/data/wholeword_exposure_split/wholeword_event_exposure_split.json`:

| checkpoint | source-absent split | events | delta | 95% interval | fraction > 0 |
|---|---|---:|---:|---|---:|
| 10M | selected0 | 2,990 | +0.072765 | [+0.054212,+0.089646] | 1.0 |
| 10M | selected_any | 1,106 | +0.113397 | [+0.086887,+0.142386] | 1.0 |
| 20M | selected0 | 2,990 | +0.048983 | [+0.026405,+0.073794] | 1.0 |
| 20M | selected_any | 1,106 | +0.166025 | [+0.120385,+0.206621] | 1.0 |
| 20M | all | 4,096 | +0.080267 | [+0.057275,+0.100260] | 1.0 |

The effect is larger when the probed word was selected by WWM, as expected, but remains positive for probe words not selected in the two-epoch schedule. This supports a category-level compact-side learning effect rather than only exact masked-instance memorization.

## External 20M BabyLM-style surface is suggestive but early

The 20M whole-word control was also scored on the cheap external surface:

`drop_abs - drop_copied_word` at 20M:

| column | delta |
|---|---:|
| BLiMP | +0.60 |
| Supplement | +1.30 |
| EWoK | +0.82 |
| Entity | -0.83 |
| COMPS | -0.05 |
| GlobalPIQA | +3.475 |
| Reading | +0.265 |
| cheap7 equal mean | +0.797143 |

This is compatible with the local source-absent channel but cannot decide mature endpoint value because it is a 20M geometry-specific surface and GlobalPIQA contributes a large share. The historical compact-view benefit matured late, so this early score should guide mechanism work rather than replace the matched result.

## Essential distinction from the failed earlier analysis intervention

earlier analysis already showed a dangerous failure mode: directly increasing pressure on strict source-absent content innovations made the intended targets easier but damaged broad mature competence. At 80M, earlier analysis improved strict-target true loss by -0.6476 nats versus token-mean 80M, yet scored -1.0529 cheap7 versus spatial repair route status reinvest, with Supplement -2.72, EWoK -1.85, GlobalPIQA -2.445, COMPS -0.75, and BLiMP -0.46.

The live source-absent result is therefore not a justification for another source-absent target-prioritization run. The scientifically different object is natural faithful compression: the data distribution itself introduces source-absent compact-content words in normal compact contexts while ordinary WWM samples targets without hand-marking that class.

## Consequence for the pending ordered-vs-scrambled result

When ordered/scrambled matched result arrives, interpret it as a test of whether natural compact-view word order helps the compact source-absent signal at fixed lexical multiset. A positive source-absent channel should be combined with the earlier whole-word control as evidence that compact data creates a useful local learning signal, but it should not be converted into explicit source-absent mask/loss selection.

If the signal survives, the next high-value experiment should preserve ordinary WWM and manipulate the natural data distribution or model/objective coordinate:

1. Test compact versus matched noncompact/repetition data in another bidirectional MLM architecture, still with legal tokenizer/data accounting and ordinary WWM.
2. Build matched faithful-compression pools with different naturally arising source-absent content density, while controlling word budget, source coverage, retained-content mass, and ordinary WWM.

Both proposed comparisons depend on completion and joint interpretation of the four compact-order experiments. No new GPU work is justified from this note alone.
