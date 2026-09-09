# complete 2x2 evidence synthesis complete two-seed 2x2 evidence synthesis

Evidence files:
- Two-seed zero-shot/reading aggregate: `data/2x2_two_seed_zero_shot_aggregate.json`
- Aggregate with SuperGLUE field: `data/2x2_two_seed_with_superglue_aggregate.json`
- SuperGLUE output validation: `data/superglue_output_validation.json`
- Compact zero-shot summary: `notes/two_seed_zero_shot_summary.md`

## Mean factorial effects across seeds 42 and 43

| contrast | BLiMP | Supplement | Entity | EWoK | COMPS | GlobalPIQA | Reading | zero-shot NLP mean | SuperGLUE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Structured data under WWM (B-A) | +1.690 | -0.825 | -0.095 | +1.745 | +0.620 | +0.965 | +0.110 | +0.601 | +0.479 |
| Structured data under AMLM (D-C) | +2.220 | -1.190 | -0.830 | +0.535 | +0.050 | -1.000 | -0.030 | -0.035 | 0.000 |
| AMLM on official data (C-A) | -0.470 | -0.210 | +0.580 | +0.785 | +0.135 | +1.953 | +0.120 | +0.413 | 0.000 |
| AMLM on structured data (D-B) | +0.060 | -0.575 | -0.155 | -0.425 | -0.435 | -0.012 | -0.020 | -0.223 | -0.479 |
| Structured × AMLM interaction | +0.530 | -0.365 | -0.735 | -1.210 | -0.570 | -1.965 | -0.140 | -0.636 | -0.479 |

## SuperGLUE validation

The patched SuperGLUE jobs produced prediction files for every arm, seed, and task, but validation shows the scores are dominated by constant majority-class behavior rather than informative transfer differences. For both seeds and nearly all arms, boolq predicts all label 1, multirc/rte/wsc/qqp/mnli predict all label 0, and mrpc predicts all label 1. Seed43 structured-WWM differs mainly on QQP. Therefore the SuperGLUE mean in this 10M screen is a valid record of the current finetuning behavior, but it should not be treated as strong positive or negative evidence that one pretraining arm has better transferable semantic representations.

## Scientific interpretation

The current SynCSE CHILDES-replacement structured corpus under ordinary WWM reproducibly improves BLiMP and EWoK, with smaller COMPS and Reading movement, but it consistently costs Supplement and does not improve Entity on average. The structured corpus combined with AMLM is antagonistic for the central target cluster: Entity, GlobalPIQA, COMPS, Reading, and the zero-shot NLP mean all move negatively relative to the additive expectation.

Progress-normalized AMLM on the official corpus is the cleanest current 10M signal: it improves Entity, EWoK, GlobalPIQA, Reading, COMPS, and the zero-shot NLP mean across the two seeds, with only small BLiMP/Supplement costs. This does not prove it will close the 100M SOTA gap, but it is the only arm in the strict 2x2 whose reproduced movement touches the persistent Entity/EWoK/GlobalPIQA deficit cluster without adding a custom data route.

## Execution consequence

The next full-budget candidate to test is official-corpus AMLM on the protected DeBERTa-v2 8×480 backbone, using the full official 10M-word corpus for 100M word exposure under the same legal repeated-exposure accounting as the protected WWM reference. The present structured+AMLM combination should not be scaled. Structured WWM may be worth redesigning later as a separate data-principle route, but its current form is not the strongest immediate full-budget candidate because Entity is not reproducibly improved and Supplement declines.
