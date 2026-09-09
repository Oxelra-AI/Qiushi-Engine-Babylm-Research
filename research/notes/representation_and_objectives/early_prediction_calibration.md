# early prediction calibration: Early-prediction calibration
## Purpose
Determine whether short (10–20M) checkpoint screens can reliably distinguish tokenizer/recipe routes before committing a full 100M run.

## Semantic view trajectory calibration (treatment vs packet-local control)
| Checkpoint | delta eq7 | Pearson(delta,endpoint) | rank_corr | sign_agree |
|-----------|-----------|------------------------|-----------|------------|
| chck_10M | +0.404 | -0.055 | 0.107 | 0.714 |
| chck_20M | -0.352 | 0.081 | 0.214 | 0.571 |
| chck_30M | +0.486 | -0.556 | -0.750 | 0.286 |
| chck_40M | +1.720 | -0.574 | -0.464 | 0.571 |
| chck_50M | +0.416 | 0.663 | 0.607 | 1.000 |
| chck_60M | +0.136 | -0.046 | 0.107 | 0.714 |
| chck_70M | -0.120 | 0.738 | 0.929 | 0.857 |
| chck_80M | +0.263 | 0.943 | 0.893 | 1.000 |
| chck_90M | +0.211 | 0.990 | 0.893 | 1.000 |

Endpoint eq7 delta: +0.172

## Per-task delta sign stability
- **BLiMP**: endpoint delta 0.190, sign match fraction 0.778
- **Supplement**: endpoint delta -1.820, sign match fraction 0.556
- **EWoK**: endpoint delta 2.190, sign match fraction 0.556
- **Entity**: endpoint delta 1.860, sign match fraction 0.778
- **COMPS**: endpoint delta 0.540, sign match fraction 0.889
- **GlobalPIQA**: endpoint delta -1.915, sign match fraction 0.667
- **Reading**: endpoint delta 0.160, sign match fraction 1.000

## EWoK phase dynamics
- old_tok_50M_to_100M_delta_corr: 0.42729235333485205
- old_tok_80M_to_100M_delta_corr: 0.9384874255689193
- corrected_tok_50M_accuracy_430_vs_431: 40.1 vs 56.6
- corrected_tok_100M_accuracy_430_vs_431: 48.5 vs 45.9
- corrected_50M_seed_delta_corr_with_old_endpoint: -0.1677177398335928
- corrected_100M_seed_delta_corr_with_old_endpoint: -0.06709872172099708

## Conclusions
- Semantic view treatment-control equal7 delta oscillated from +0.40 (10M) to -0.35 (20M) to +1.72 (40M) to +0.17 (100M). The sign flipped at 20M despite positive final effect.
- Supplement delta sign match fraction across 9 early checkpoints: 0.556. Endpoint delta: -1.82.
- EWoK delta sign match fraction: 0.556. EWoK is nearly at noise floor for this treatment size.
- EWoK seed-delta from old tokenizer: 50M→100M correlation 0.427, 80M→100M correlation 0.938. Under corrected tokenizer, dynamics were different and less polarized.
- First checkpoint with cross-task delta Pearson > 0.7: chck_70M.
- A 10–20M early screen cannot reliably predict 100M endpoint quality for this model family. The minimum informative exposure for EWoK/relational tasks is around 80M. For aggregate equal7, even 40M delta was 10× the endpoint delta.
- Route selection should rely on representation geometry (tokenizer interface), established mechanism evidence, and staged continuation to high-exposure evaluation, not on 10–20M model scores.

## Artifacts
- JSON: `experiments/archive/representation_and_objectives/data/early_prediction_calibration/early_prediction_calibration.json`
- note: `research/notes/representation_and_objectives/early_prediction_calibration.md`
