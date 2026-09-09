# tokenizer learning contingency and gpu priority — tokenizer-learning contingency and GPU priority

## Research role

The decisive missing evidence is the compliant-tokenizer `compact_view_reinvest` endpoint. Active training directories were not inspected. The recorded training status was:

- Compliant-tokenizer `compact_view_reinvest` retraining: in progress at approximately 2905 s; submission-relevant endpoint.
- Clean-Qwen fixed-tokenizer control: starting at approximately 2898 s; scientific control only.

The old-tokenizer 42.0331347900748 endpoint remains scientific evidence for redundancy-reduced semantic second views plus reinvested source diversity, but it is not submit-ready because tokenizer training used outside-budget text. The current decisive endpoint is the reinvest retrain with tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9` trained only on the reinvest 10M pool.

## Peer update incorporated

CPU-only evidence shows that old-tokenizer seed behavior and MLM loss are poor guides to the corrected-tokenizer coordinate:

- In old-tokenizer reinvest, seed43022 beats seed43122 by +0.7849 Overall while having slightly higher all-step mean MLM loss (+0.003586) and last-50-step loss (+0.000543), so MLM loss should not choose seeds or endpoints.
- Old-tokenizer focused EWoK relation advantage forms late and correlates by 80M with endpoint seed delta, but corrected-tokenizer 50M reverses on the same focused rows: seed43022 40.14, seed43122 56.60, with correlation to old endpoint -0.168.
- Consequence: do not infer the compliant 100M endpoint from old-tokenizer dynamics; evaluate the delivered compliant checkpoint in the current coordinate.

This reinforces waiting for the real 100M official-compatible result and makes the interval useful only for evidence that can change the response to a below-leader score.

## CPU-only tokenizer-learning contingency

Script:

- `experiments/archive/frontier_consolidation/scripts/tokenizer_learning_contingency.py`

Outputs:

- `experiments/archive/frontier_consolidation/data/tokenizer_learning_contingency/tokenizer_learning_contingency.json`
- `research/documents/frontier_consolidation/data/tokenizer_learning_contingency/tokenizer_learning_contingency.md`

Run completed in 318.363 s. The tokenizer library printed a max-length warning during count-only tokenization of long text; the script used `truncation=False`, performed no model inference, and did not evaluate any checkpoint.

Official evaluation text was used only to interpret possible score movement after the model result arrives. It must not be used to choose pretraining text or tokenizer vocabulary.

## Main findings

Raw vocabulary overlap between old 100M-trained tokenizer and compliant 10M-trained tokenizer is moderate:

- 16,384 tokens each.
- 13,809 shared token strings = 0.842834 of the compliant vocabulary.
- 2,575 token strings exist only in each side.

Occurrence mass is much less alarming than raw type overlap:

| surface | new/old token ratio | old-only occurrence % | new-only occurrence % | mean token delta |
|---|---:|---:|---:|---:|
| reinvest 10M pool, all rows | 0.9915 | 0.79 | 1.49 | -1.930 |
| reinvest changed block | 0.9811 | 1.18 | 2.89 | -3.907 |
| inherited Qwen pairs | 0.9845 | 0.91 | 2.22 | -2.858 |
| official evaluation text, all families | 0.9993 | 1.02 | 1.08 | -0.039 |

The compliant tokenizer is not simply a fragmented replacement on the allowed pretraining pool. It produces fewer tokens on the reinvest pool, especially on the changed block and inherited Qwen pair rows. It also allocates new token mass to material present in the allowed reinvest pool: changed-block top new-only tokens include `Ġresearchers`, `Ġprotein`, `ĠResearchers`, `Ġdiabetes`, `Ġantib`, `Ġvitamin`, `Ġhorm`, `Ġimmune`, `Ġinteract`, `Ġinfections`, `Ġexposure`, and `Ġpollution`.

On official evaluation text, family-level token length changes are small and mostly positive except SuperGLUE, which becomes shorter:

| family | new/old token ratio | old-only occurrence % | new-only occurrence % | mean token delta |
|---|---:|---:|---:|---:|
| BLiMP | 1.0042 | 2.29 | 1.94 | +0.043 |
| Supplement | 1.0016 | 1.56 | 1.34 | +0.026 |
| EWoK | 1.0128 | 1.98 | 0.94 | +0.116 |
| Entity | 1.0015 | 0.15 | 0.00 | +0.247 |
| COMPS | 1.0099 | 1.65 | 0.83 | +0.145 |
| GlobalPIQA parallel | 1.0061 | 1.03 | 0.36 | +0.183 |
| GlobalPIQA nonparallel | 1.0091 | 2.11 | 1.03 | +0.223 |
| Reading sentence | 1.0059 | 0.75 | 0.20 | +0.068 |
| Reading word | 1.0035 | 0.55 | 0.25 | +0.004 |
| SuperGLUE | 0.9954 | 1.28 | 1.69 | -0.786 |

Together with compliant tokenizer shift and eval readiness's result that zero-shot columns gain no new seq256 truncation and SuperGLUE has slightly fewer seq512 truncations, this weakens a simple explanation in which compliant-tokenizer replacement mainly hurts by cropping evaluation text.

EWoK domain tokenization pressure is uneven but not a one-to-one replay of the old relation instability:

| EWoK domain | new/old token ratio | old-only occurrence % | new-only occurrence % | mean token delta |
|---|---:|---:|---:|---:|
| material-dynamics | 0.9856 | 0.39 | 2.72 | -0.110 |
| physical-dynamics | 1.0257 | 2.57 | 0.00 | +0.250 |
| physical-interactions | 1.0248 | 1.83 | 1.65 | +0.315 |
| physical-relations | 1.0000 | 0.00 | 0.00 | 0.000 |
| spatial-relations | 1.0184 | 1.11 | 0.71 | +0.269 |
| social-relations | 1.0350 | 4.68 | 0.19 | +0.342 |

The old seed interaction emphasized spatial/material/physical dynamics. In tokenizer surface, social-relations has the largest length increase, physical-relations is unchanged, and material-dynamics becomes shorter. Therefore a future compliant-score loss should not be automatically attributed to the old spatial-relation mechanism.

There are nevertheless concrete physical/object words where the compliant tokenizer breaks old whole-word pieces into more subwords:

- EWoK overall: `screwdriver` appears 404 times in eval text but only 43 times in the reinvest pool; old segmentation was a whole-word token, compliant segmentation is `Ġscrew`/`dri`/`ver`.
- EWoK spatial-relations: the only changed word type at this granularity is `screwdriver`, causing 624 weighted extra pieces.
- EWoK physical-dynamics: `sinking`, `sliding`, `slippery`, and `underwater` are each longer under the compliant tokenizer, with train counts 41–70.
- GlobalPIQA practical/object words such as `cupboard`, `backpack`, `heavier`, `baking`, `freeze`, `sweeping`, `cans`, `floors`, `loaf`, and `parked` become longer, but counts are small.

## How to use this after the compliant result arrives

If the compliant reinvest endpoint is above the visible leader, complete the official-compatible evaluation and pristine collation; do not stop at a partial projection.

If it lands below the visible leader, compare the score movement against this tokenizer surface:

1. If losses concentrate in EWoK physical/spatial/object words, COMPS animal/object families, or GlobalPIQA practical-object readouts where this analysis shows increased fragmentation or old-only occurrence mass, then a legal tokenizer-learning route may be worth studying. Any such route must train only on allowed 10M text and use generic tokenizer objectives or corpus-derived priors; official evaluation words must not drive vocabulary choices.
2. If losses are broad, or occur in columns/domains with small tokenization pressure, then the likely cause is changed MLM target geometry, optimizer/initialization dynamics, or the data mechanism under the new vocabulary rather than crude token length. In that case choose the next scientific factor from this evidence before any new full retrain.
3. Do not use MLM loss alone as the continuation signal, especially after evidence that old-tokenizer seed/MLM dynamics fail to transfer to corrected-tokenizer behavior.

## GPU priority

The submission-relevant reinvest retrain has priority over the fixed-tokenizer clean-Qwen control. If a resource collision appears before terminal delivery, keep GPU priority on the reinvest retrain; the clean-Qwen arm is valuable for fixed-tokenizer causality but is not an end-to-end submission candidate. If the reinvest retrain fails from resource collision or timeout before a complete 100M checkpoint, relaunch the identical scientific command in a fresh run directory using the repaired wrappers. Do not change tokenizer, corpus, architecture, seed, batch geometry, masking, optimizer, or exposure.
