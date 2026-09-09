# public leader available coordinate — public leader local re-score and route consequence

## Public checkpoint staging

Public model repo: `go76dof/wwm_curriculum_simplification_40k`

The first attempt to download weights failed because Hugging Face tried to cache into a read-only shared model root. Retried with writable local cache:

- `HF_HOME=training/hf_home_public_leader`
- staged weights: `data/leader_package_revision_124/model_side/local/model.safetensors`
- load record: `data/public_leader_weight_stage_load.json`

The staged public checkpoint loads locally:

- tokenizer length: 40,000
- model class: `DebertaV2ForMaskedLM`
- parameter count: 34,677,952
- local staged directory: `data/leader_package_revision_124/model_side/local/`

No gated training data was downloaded or used.

## Local re-score

Evaluation scripts:

- `scripts/eval_public_leader_available_coordinate.py`
- `scripts/eval_public_leader_full_ewok.py`

Evidence:

- available-coordinate JSON: `data/public_leader_available_coordinate.json`
- full local EWoK JSON: `data/public_leader_full_ewok_word_tokenize_score.json`
- notes: `notes/public_leader_available_coordinate.md`, `notes/public_leader_full_ewok_word_tokenize_score.md`

Scores from the local direct-checkpoint evaluator:

| column/task | local public leader score | card score | comment |
|---|---:|---:|---|
| BLiMP | 67.20 | 67.20 | exact |
| Supplement | 56.04 | 56.01 | rounding/surface difference only |
| EWoK | 56.07 | 56.07 | exact with local word-tokenize EWoK |
| Entity | 28.45 | 28.45 | exact |
| COMPS | 53.57 | 53.57 | exact |
| GlobalPIQA parallel | 22.33 | not separately reported | local subcolumn |
| GlobalPIQA nonparallel | 57.00 | not separately reported | local subcolumn |
| GlobalPIQA mean | 39.665 | 39.67 | rounding |
| Reading eye | 6.83 | not separately reported | local subcolumn |
| Reading self-paced | 4.02 | not separately reported | local subcolumn |
| Reading mean | 5.425 | 5.42 | rounding |

This resolves a major uncertainty from earlier analysis: the visible leader's reported Entity/EWoK/GlobalPIQA package is real under our local evaluator path. The gap is not a card-only or scorer-incompatibility artifact.

## Comparison to our best runs

| column | public leader local | protected 8×480 | S1 12×384 | true S2 | leader - protected | leader - S1 | leader - S2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 67.20 | 66.76 | 66.84 | 64.24 | +0.44 | +0.36 | +2.96 |
| Supplement | 56.04 | 59.88 | 60.31 | 59.09 | -3.84 | -4.27 | -3.05 |
| EWoK | 56.07 | 52.19 | 52.02 | 51.64 | +3.88 | +4.05 | +4.43 |
| Entity | 28.45 | 22.62 | 20.24 | 18.47 | +5.83 | +8.21 | +9.98 |
| COMPS | 53.57 | 52.19 | 52.26 | 50.61 | +1.38 | +1.31 | +2.96 |
| GlobalPIQA | 39.665 | 35.635 | 37.605 | 38.635 | +4.03 | +2.06 | +1.03 |
| Reading | 5.425 | 7.62 | 7.25 | 7.52 | -2.20 | -1.83 | -2.10 |

Interpretation:

1. The target is now empirically anchored in our own local evaluator. The leader really has the large Entity/EWoK gains that our S1/S2 routes failed to produce.
2. S1/S2 evidence remains valid: shape and true word-clock curriculum are insufficient on official corpus + baseline16k + AdamW. In fact, S2 moves toward the leader on GlobalPIQA but away from it on Entity/EWoK.
3. It remains premature to attribute the entire remaining gap to the gated paired data because 40k under the 12×384 shape and LAMB/cosine optimization are still unrun under official data. The public leader re-score makes these factor tests more urgent, not less: the actual model-side package is now verified locally.

## Immediate execution consequence

The next clean executable cell should complete the legal model-side decomposition before expensive data reconstruction:

- **S3 40k-under-12×384 arm**: S1 architecture, official corpus, legal local 40k tokenizer, flat WWM or leader curriculum as a separate condition, exact word accounting. Need tokenizer-aware Reading interpretation because the previous 8×480 official40k arm collapsed Reading.
- **LAMB arm**: S1 architecture and official corpus with baseline16k, LAMB/cosine optimization and a safe LR smoke/sweep around the leader card regime. LAMB has not been tested at all.

Given engineering readiness, the most direct next step is likely S3 40k under the already-working isolated leader-shape fork if it supports `--tokenizer_label official40k`, followed by LAMB support if not already available. Neither should use the gated training data. If both fail to move Entity/EWoK, the evidence would strongly justify legal simplification-pair reconstruction with aligned-vs-shuffled controls.
