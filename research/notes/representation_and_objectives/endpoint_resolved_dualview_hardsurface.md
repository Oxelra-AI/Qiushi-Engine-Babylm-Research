# endpoint resolved dualview hardsurface endpoint resolution and dual-view hard-surface readout

Status: **PASS**

## chck_82M endpoint
- From-corpus reproduction: `experiments/archive/representation_and_objectives/data/from_corpus_reproduction_compare/from_corpus_reproduction_compare.json` status PASS; chck_82M hash equal = True.
- Full-plus-fast prediction carrier: `experiments/archive/representation_and_objectives/data/chck82_fast_submission_materialization/fast_submission_verification.json` status PASS; final collated file `experiments/archive/representation_and_objectives/data/chck82_fast_submission_materialization/collate_fast/results/hf_model/all_full_preds_and_fast_scores_mlm.json` SHA `dcad3d8cf285a69d31910a32c62c80689022a3ff459a0956401ab7f4b3542237`; 133 fast prediction files checked.
- Full endpoint blocks in the full-plus-fast carrier are byte-equal to the trusted earlier analysis full collation; fast input provenance is PASS.

## Dual-view broad scores
| target | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlm_only_20M | 39.786429 | 59.54 | 58.43 | 50.1 | 18.39 | 50.7 | 32.665 | 8.68 |
| sep_sparse20_aligned_20M | 40.292857 | 61.26 | 58.6 | 50.77 | 18.8 | 50.48 | 33.665 | 8.475 |
| sep_sparse20_shuffled_20M | 40.063571 | 60.56 | 57.45 | 50.88 | 18.35 | 50.74 | 34.195 | 8.27 |
| coupled_sparse20_aligned_20M | 38.793571 | 57.02 | 55.0 | 50.15 | 17.93 | 49.72 | 34.165 | 7.57 |

## Fixed hard surfaces
| target | GPIQA parallel | GPIQA hard52 acc | hard52 mean top-correct | EWoK hard acc | EWoK stable failures | EWoK interaction wrong mean |
|---|---:|---:|---:|---:|---:|---:|
| mlm_only_20M | 22.33009708737864 | 3.8461538461538463 | 1.622895339167021 | 0.26580557443915703 | 922 | -1.632786623834572 |
| sep_sparse20_aligned_20M | 22.33009708737864 | 7.6923076923076925 | 1.53736199530253 | 0.2229775662814412 | 988 | -1.2584356781682333 |
| sep_sparse20_shuffled_20M | 20.388349514563107 | 1.9230769230769231 | 1.6378646220456103 | 0.21210061182868797 | 1032 | -1.6099947531106122 |
| coupled_sparse20_aligned_20M | 22.33009708737864 | 9.615384615384615 | 1.443233044866988 | 0.512576478585996 | 430 | -0.06628491602204169 |

## Main deltas
- `sep_aligned_minus_mlm_only`: cheap7 0.5064285714285717, GPIQA hard52 acc 3.8461538461538463, GPIQA hard mean margin -0.08553334386449096, EWoK hard accuracy -0.04282800815771584, EWoK stable failures 66, EWoK interaction wrong mean 0.37435094566633875.
- `sep_aligned_minus_sep_shuffled`: cheap7 0.22928571428571587, GPIQA hard52 acc 5.769230769230769, GPIQA hard mean margin -0.10050262674308041, EWoK hard accuracy 0.010876954452753218, EWoK stable failures -44, EWoK interaction wrong mean 0.3515590749423789.
- `coupled_aligned_minus_mlm_only`: cheap7 -0.9928571428571402, GPIQA hard52 acc 5.769230769230768, GPIQA hard mean margin -0.179662294300033, EWoK hard accuracy 0.24677090414683894, EWoK stable failures -492, EWoK interaction wrong mean 1.5665017078125303.
- `coupled_aligned_minus_sep_aligned`: cheap7 -1.499285714285712, GPIQA hard52 acc 1.9230769230769225, GPIQA hard mean margin -0.09412895043554204, EWoK hard accuracy 0.2895989123045548, EWoK stable failures -558, EWoK interaction wrong mean 1.1921507621461915.

## Scientific reading
- The chck_82M endpoint is now a reproducible score-bearing candidate: from-corpus training regenerated the key checkpoint hashes bit-for-bit, including chck_82M, and the full-plus-fast prediction carrier passes while preserving the trusted full endpoint blocks.
- The endpoint result remains an exposure-selected scale1.75 substrate; it does not by itself explain or repair context-conditioned alternative binding.
- Separated sparse20 true alignment preserves broad early-training score better than coupled sparse20, but its fixed hard-surface movement is mixed: it improves GlobalPIQA hard52 over MLM-only and shuffled, yet worsens the full ewok interaction synthesis EWoK stable-failure subset versus MLM-only.
- Coupled sparse20 true alignment strongly improves the full ewok interaction synthesis EWoK hard subset and GlobalPIQA hard52, showing that dual-view coupling can move the missing relation surface, but it damages broad columns enough that it is not the usable training recipe.
- The live scientific opportunity is to isolate the coupled hard-surface signal inside a broad-preserving pathway; launching an 82M private tail is premature unless a small controlled construction shows both broad preservation and EWoK/GlobalPIQA hard-row movement.

JSON: `experiments/archive/representation_and_objectives/data/endpoint_and_dualview_consolidation/endpoint_and_dualview_consolidation.json`
