# step044 copy like diagnosis diagnosis of earlier analysis accepted packet geometry

The hardened earlier analysis lexical state validator is not sufficient for the intended relation-retargeted arm, because a final use sentence can satisfy the state-content rule by repeating the source sentence or a long source fragment. This note quantifies that failure before any training is launched.

## Aggregate copy-like measurements

| subset | n | exact use=source | use substring of source | LCS>=6 | LCS>=8 | content-overlap-min>=0.94 | content-Jaccard>=0.80 | combined DUP-like | mean content Jaccard | mean LCS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 3059 | 0.006 | 0.208 | 0.574 | 0.401 | 0.530 | 0.073 | 0.575 | 0.438 | 6.82 |
| UNCHANGED_DISTRACTOR_USE | 2742 | 0.006 | 0.232 | 0.640 | 0.447 | 0.591 | 0.082 | 0.641 | 0.475 | 7.42 |
| UPDATED_USE | 317 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.003 | 0.115 | 1.68 |

## Interpretation

The intended zero-relevant-update packet should train retained-state readout under changed form after an irrelevant update. Packets with exact/substring use sentences or long source spans instead train source copying with an intervening sentence, close to the DUP/REPEAT behavior already known to create changed-form liability. These packets must not be assembled into the practical SOTA arm. The step044 copy like diagnosis prompt and validator therefore require changed-form use sentences, no long shared source/update spans, and balanced subsampling of UPDATED_USE and UNCHANGED_DISTRACTOR_USE after validation.

Machine-readable summary: `experiments/archive/relation_learning/data/copy_like_diagnosis/copy_like_summary.json`. Per-packet metrics: `experiments/archive/relation_learning/data/copy_like_diagnosis/copy_like_packet_metrics.jsonl`. Worst examples: `experiments/archive/relation_learning/data/copy_like_diagnosis/copy_like_worst_examples.jsonl`.
