# seed43222 entity clean integration split-control numeric prestatement

## Scientific purpose

The split controls remove same-window source/companion co-occurrence while preserving the selected text exposure and 100M fixed-budget stream. They test whether the installed copy and nonidentical-content effects come from practicing a relation between spans inside one training window, or from duplicated/paraphrased token exposure without local co-occurrence.

The comparisons use the existing CLEAN arm in the same DeBERTa seed43022 coordinate. The measured quantities are late means over 80M/90M/100M from the held-out probe framework.

## Original three-seed DeBERTa ranges

| quantity | seed43022 | seed43122 | seed43222 | three-seed mean | original range |
|---|---:|---:|---:|---:|---:|
| REPEAT − CLEAN, token-nonoverlap rewrite gain | -0.7517 | -1.0433 | -0.8503 | -0.8818 | [-1.0433, -0.7517] |
| REPEAT − CLEAN, held-out natural copy gain | +0.4996 | +0.6678 | +0.5594 | +0.5756 | [+0.4996, +0.6678] |
| VIEW − CLEAN, token-nonoverlap rewrite gain | +0.6837 | +0.8938 | +0.8392 | +0.8056 | [+0.6837, +0.8938] |

## Reading the split arms before results are known

A split-vs-CLEAN contrast within ±0.25 nats/points of zero is read as: the corresponding same-window relation is necessary for most of the installed effect in this setup.

A split-vs-CLEAN contrast inside the original three-seed range is read as: token exposure and fixed-budget substitution are sufficient to reproduce the original installed quantity without same-window co-occurrence.

Intermediate values are read as mixed dependence: cross-row exposure contributes, but same-window relation practice supplies a substantial part of the installed quantity. This is the most likely outcome for at least one side and should not be reinterpreted after seeing the number.

For REPEAT_SPLIT, the two quantities must be read together:

- If token-nonoverlap rewrite gain moves near zero and copy gain also moves near zero, exact same-window identity practice is the central cause of both the copy benefit and the nonidentical-source cost.
- If rewrite cost remains original-like while copy weakens, repeated-token budget can damage nonidentical source use independently of a strong copy-use computation.
- If copy remains original-like while rewrite cost weakens, exact recurrence can preserve copy use without installing the harmful nonidentical-source response.
- If both remain original-like, duplicated-token budget/cross-row recurrence is sufficient; the in-window formulation would need to be narrowed.

For VIEW_SPLIT:

- If token-nonoverlap VIEW − CLEAN rewrite gain moves near zero, VIEW's content-conditioning benefit requires practicing source/rewrite co-occurrence in one context.
- If it remains original-like, cross-row paraphrase augmentation is sufficient for this positive content-conditioning effect, and the novelty shifts toward the recurrence-cost side plus the locality contrast.
- Intermediate values indicate that both paraphrase exposure and in-window source/rewrite relation practice contribute.

## Files and planned scoring

Training streams: `experiments/archive/relation_learning/data/split_inwindow_rowholdout_pools/compact_repeat_split_dose2p64x_100M.jsonl` and `.../compact_view_split_dose2p64x_100M.jsonl`.

After training, score `chck_80M`, `chck_90M`, and `chck_100M` with the held-out copy/rewrite probe framework against CLEAN seed43022 before reading any benchmark behavior.
