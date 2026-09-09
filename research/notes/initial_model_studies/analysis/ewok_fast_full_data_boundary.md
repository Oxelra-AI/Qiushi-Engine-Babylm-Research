# Historical EWoK Fast/Full Data Boundary

This record preserves the data-identity findings from the early BERT-WWM 8x256, seed-42, 100M-word baseline evaluation. It describes the evidence available during that source-resolution investigation, not the later availability of full EWoK or the scores of later DeBERTa models. No new download, experiment or evaluation was performed to prepare this note.

## Distinct Evaluation Populations

The inspected BabyLM evaluation bundle, revision `8d52da9424a9ff30b9e8266c4f751aba9c504233`, contained 175 files. Its only EWoK data object was `evaluation_data/fast_eval/ewok_fast.zip`, 34,719 bytes. The fast population contained 11 domains with 100 rows each, or 1,100 rows. Neither the presence of this archive nor its nested extraction directory established availability of the full population.

The full-data source was identified as `ewok-core/ewok-core-1.0`, repository revision `34d912a608066c92e2990a0328ffc3bd9a716042`, object `data/test/ewok-core-1.0.parquet`, 110,797 bytes, object commit `a374d875fa62d54081ff82e455aba9c0f80ef7ec`. At the time of the investigation, metadata was accessible but anonymous content requests failed; the exact full-data object was not obtained. Public metadata did not establish possession of the evaluation items.

The historical fast-set accuracy on the exact BERT-WWM `chck_100M` checkpoint was **46.181818%**, reported as **46.18%**. The saved [fast evaluation summary](../../../../experiments/archive/initial_model_studies/analysis/fast_eval_summary.json) records that measurement. It remains a fast-set result, not a substitute for full EWoK. The broader baseline and missing-full-score context is retained in [the original evaluation note](../ewok_aoa_and_34m_smoke.md).

## Recorded Full-Data Shape

The inspected evaluator expected 3,809 vocabulary-filtered originals, each written in two context/target orientations, giving 7,618 rows. The recorded per-domain shape was:

| Domain | Filtered Originals | Full Rows | Fast Rows |
|---|---:|---:|---:|
| agent-properties | 1,105 | 2,210 | 100 |
| material-dynamics | 385 | 770 | 100 |
| material-properties | 85 | 170 | 100 |
| physical-dynamics | 60 | 120 | 100 |
| physical-interactions | 278 | 556 | 100 |
| physical-relations | 409 | 818 | 100 |
| quantitative-properties | 157 | 314 | 100 |
| social-interactions | 147 | 294 | 100 |
| social-properties | 164 | 328 | 100 |
| social-relations | 774 | 1,548 | 100 |
| spatial-relations | 245 | 490 | 100 |
| **Total** | **3,809** | **7,618** | **1,100** |

These counts are necessary shape checks for that evaluator revision, not sufficient proof of item identity. The vocabulary filter used all four sentence fields, the evaluator vocabulary and NLTK word tokenization; reproducing the total without reproducing the same source and filtering semantics does not establish the same test population.

## Rejected Substitute and Availability Limitation

A historical comparison with the protected paper-era EWoK release reproduced the total of 3,809 filtered originals but not their allocation across domains: physical relations differed by **+10**, social interactions by **-8**, and social properties by **-2**. Because the score is macro-averaged over domains, an unchanged grand total does not make the populations interchangeable. That release was therefore not accepted as a validated substitute for the pinned test Parquet and did not establish a full score.

At this historical boundary, the scientifically valid state was a missing full-EWoK measurement plus a separately labeled fast interim measurement. Access to the authorized source, matching item identity, successful filtering and a completed full-population evaluation were still required before filling the full coordinate. This note redistributes no evaluation items, protected archives or corpus excerpts and makes no claim about current access status.
