# frontier consolidation seed43122 incorpus readout readout of inherited frontier_consolidation register and in-corpus cells

This note is research-facing memory for relation_learning. It reads already-produced frontier_consolidation artifacts and does not run model inference, training, official scoring, uploads, or leaderboard actions.

## Predictors tied to these cells before any new training

The old frontier_consolidation files already contain the numerical predictors used here: earlier analysis profile-register prediction, earlier analysis removed-block late-rate prediction, earlier analysis in-corpus static profile prediction, and earlier analysis post-training in-corpus late-rate readout. relation_learning uses those numbers as comparators and asks whether they supply a computable conversion term rather than simply describe outcomes.

- Register childspeech-minus-adultprose: profile distance and removed-block late rate both predict a positive broad contrast if adult-prose removal is costlier; word-unigram predicts a small negative contrast. A seed-stable principle would preserve the sign across seed43022 and seed43122.
- In-corpus adult prose: static profile predicts a small positive broad4/cheap5 movement; the measured in-corpus 60M→100M admitted-block loss drop predicts a positive late broad gain close to the view/breadth regime; the word-unigram control predicts a small negative movement; a pure out-of-corpus-novelty reading expects the in-corpus cell to stay near clean while FineWeb full1x remains higher.
- The open term in this study is conversion: whether residual work is connected to reusable learner coordinates rather than privately fitted. The present file-only readout cannot measure that term; it identifies where a checkpoint-computable shared/private fitting index must enter next.

## Register seed replication

Positive means the arm that removes child/spoken material scores above the arm that removes adult prose, with identical admitted MAX FineWeb text.

| seed | checkpoint | BLiMP | Supplement | EWoK | COMPS | exEntity4 | Entity/cheap if present |
|---|---:|---:|---:|---:|---:|---:|---:|
| 43022 | chck_80M | -1.130 | -0.020 | -0.050 | -0.770 | -0.492 | -0.512 |
| 43022 | chck_100M | -1.450 | -1.560 | +0.940 | -0.380 | -0.613 | -0.714 |
| 43122 | chck_80M | -0.760 | +0.180 | +1.730 | +1.060 | +0.552 | NA |
| 43122 | chck_100M | -1.290 | +0.690 | +2.030 | +1.070 | +0.625 | NA |

Seed43022 mean exEntity4 over 80/100M is -0.552; seed43122 mean exEntity4 is +0.589. The broad sign therefore flips between basins. Seed43122 gives positive broad4 mostly through EWoK/COMPS/Supplement while BLiMP remains negative; seed43022 gave a negative broad4 contrast. This makes the sacrificed-register sign real but not stable enough to be the general principle.

## In-corpus adult-prose cell

| contrast | checkpoint | BLiMP | Supplement | EWoK | COMPS | Entity | exEntity4 | cheap5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| incorpus_minus_clean | chck_70M | -0.040 | +0.610 | -0.580 | +0.730 | +1.080 | +0.180 | +0.360 |
| incorpus_minus_clean | chck_80M | +0.450 | +0.250 | +1.410 | +0.350 | +1.180 | +0.615 | +0.728 |
| incorpus_minus_clean | chck_100M | +0.500 | -0.150 | +0.480 | +0.470 | +0.750 | +0.325 | +0.410 |
| incorpus_minus_subdose_full | chck_70M | -0.950 | -0.440 | -0.980 | +1.120 | +2.010 | -0.312 | +0.152 |
| subdose_full_minus_clean | chck_70M | +0.910 | +1.050 | +0.400 | -0.390 | -0.930 | +0.492 | +0.208 |

In-corpus minus clean is positive on broad4 at 70M/80M/100M (+0.180, +0.615, +0.325), with late 80/100M mean +0.470. The post-training admitted-block loss drop from 60M to 100M is 0.274, near the view/breadth active range rather than the repeat-like low range. At 70M, in-corpus is below full1x on broad4 (-0.312) but above it on Entity (+2.010), so Entity should remain separate from the broad learning principle.

## What this changes for the principle

The in-corpus result gives residual work numerical content in DeBERTa: official adult-prose rows still carried late reducible loss and their training moved broad evaluation scores. The register replicate prevents turning removal-side register opportunity into a seed-invariant scalar: the same sign does not hold across seed43022 and seed43122. The stronger relation_learning proposition should therefore be coordinate conversion, not register mixture by itself: finite experience helps when late residual work is aligned with a learner coordinate that carries improvement beyond the trained rows. The next execution must measure that alignment directly as shared versus private fitting on existing checkpoints, especially DeBERTa versus RoBERTa on identical MAX admitted view text.

## Files

- summary_json: `experiments/archive/relation_learning/data/frontier_consolidation_readout/frontier_consolidation_readout_summary.json`
- register_arm_csv: `experiments/archive/relation_learning/data/frontier_consolidation_readout/register_arm_scores.csv`
- register_contrast_csv: `experiments/archive/relation_learning/data/frontier_consolidation_readout/register_seed_contrasts.csv`
- incorpus_arm_csv: `experiments/archive/relation_learning/data/frontier_consolidation_readout/incorpus_arm_scores.csv`
- incorpus_contrast_csv: `experiments/archive/relation_learning/data/frontier_consolidation_readout/incorpus_contrasts.csv`
- predictor_csv: `experiments/archive/relation_learning/data/frontier_consolidation_readout/predictor_outcomes.csv`
- note_md: `research/notes/relation_learning/frontier_consolidation_seed43122_incorpus_readout.md`
