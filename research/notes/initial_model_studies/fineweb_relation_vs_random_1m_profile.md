# fineweb relation vs random 1m profile FineWeb relation-explicit vs random-quality 1M profile

Evidence JSON: `experiments/archive/initial_model_studies/data/fineweb_relation_vs_random_1m_profile.json`

Both arms: FineWeb-Edu sample-10BT, 1M words, DeBERTa-v2 8x480, baseline16k, WWM p=0.15, seed/init fixed, batch 128 after batch-256 OOM. Difference is same-source relation-explicit filtering versus random-quality FineWeb text.

| column | random_quality | relation_explicit | relation-random |
|---|---:|---:|---:|
| blimp_fast | 53.7500 | 54.4600 | +0.7100 |
| supplement_fast | 47.2000 | 47.2000 | +0.0000 |
| ewok_fast | 49.1800 | 52.4500 | +3.2700 |
| entity_tracking_fast | 20.0900 | 19.9700 | -0.1200 |
| comps | 49.7500 | 49.4600 | -0.2900 |
| reading_eye_tracking | 10.4100 | 9.7800 | -0.6300 |
| reading_self_paced | 3.1400 | 2.9500 | -0.1900 |
| Reading_mean | 6.7750 | 6.3650 | -0.4100 |

## Training summary

| arm | loss first | loss last | steps | trunc frac | kept tokens/word | masked tokens/word |
|---|---:|---:|---:|---:|---:|---:|
| random_quality | 9.7859 | 7.4608 | 49 | 0.2368 | 1.4688 | 0.2199 |
| relation_explicit | 9.7716 | 7.4835 | 49 | 0.2416 | 1.4695 | 0.2205 |
