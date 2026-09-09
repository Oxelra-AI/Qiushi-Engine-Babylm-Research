# earlier analysis AoA estimator rescore: representation_frontier_studies_step217_alpha075_19ckpt_official

Surprisal: `experiments/archive/representation_and_objectives/data/alpha075_aoa_minctx0/collate_fast/results/hf_model/main/zero_shot/mlm/AoA_word/surprisal.json`
Tokenizer: `experiments/archive/functional_learning/data/automodel_repair/repaired_coherent86_alpha075` (vocab 16384)
Rows/steps/words: `152095` / `19` / `485`; row counts per step `[8005]`

## Estimator provenance

- Current public BabyLM evaluation commit expected: `6f825c291e2c4c78ad33b1935fd64d45f52642dc`; local head: `6f825c291e2c4c78ad33b1935fd64d45f52642dc`.
- `utils.py` SHA256: `b88b5bf4cf2ec1a55ee598d837faeb8b4df40fea6498b334a5fd1c22fb236b21`; Knowledge archive byte-equal to local: `True`.
- Official import check for current estimator: `True`; diff `0.0`.
- The saved Space-side submission code reads the numeric `aoa` object and carries `aoa_surprisals`; display code zeroes AoA when `aoa_surprisals` is absent. Thus estimator identity belongs to the participant-side evaluation commit that produced `aoa_score.json`.

## Scores from the same surprisal measurements

| Estimator | Platform coordinate? | curve_fitness | leaderboard units | n fitted words | p value | raw corr |
|---|---:|---:|---:|---:|---:|---:|
| `current_official_main_6f825c2` | `True` | 0 | 0 | 225 | 0.6030225874932932 | -0.03485418140387646 |

Non-platform rows are estimator-sensitivity analyses. They are useful for developmental interpretation but must not be mixed into the BabyLM frontier coordinate.

Full JSON: `experiments/archive/functional_learning/data/aoa_estimator_provenance/frontier_alpha075_19ckpt_official_aoa_estimator_rescore.json`
