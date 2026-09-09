# pair discriminator 1m profile same-entity pair discriminator 1M profile

Evidence JSON: `experiments/archive/initial_model_studies/data/pair_discriminator_1m_profile.json`

| column | true_pair | hard_neg | orig | shuffled | true-hardneg | true-orig | true-shuf |
|---|---:|---:|---:|---:|---:|---:|---:|
| blimp_fast | 53.1700 | 52.2500 | 53.8700 | 53.2200 | +0.9200 | -0.7000 | -0.0500 |
| supplement_fast | 48.4000 | 52.4000 | 44.4000 | 42.8000 | -4.0000 | +4.0000 | +5.6000 |
| ewok_fast | 49.1800 | 48.3600 | 49.1800 | 49.0900 | +0.8200 | +0.0000 | +0.0900 |
| entity_tracking_fast | 17.8800 | 16.6000 | 17.7300 | 17.6000 | +1.2800 | +0.1500 | +0.2800 |
| comps | 49.7900 | 50.0100 | 50.2000 | 49.8300 | -0.2200 | -0.4100 | -0.0400 |
| reading_eye_tracking | 5.6200 | 5.8500 | 9.6500 | 5.7100 | -0.2300 | -4.0300 | -0.0900 |
| reading_self_paced | 1.4700 | 1.5700 | 2.9400 | 2.4100 | -0.1000 | -1.4700 | -0.9400 |
| Reading_mean | 3.5450 | 3.7100 | 6.2950 | 4.0600 | -0.1650 | -2.7500 | -0.5150 |

## Training summary

| arm | loss first | loss last | steps | trunc frac | kept tokens/word | masked tokens/word |
|---|---:|---:|---:|---:|---:|---:|
| true_pair_adjacent | 9.826553344726562 | 7.5237650871276855 | 49 | 0.25696 | 1.494502 | 0.223589 |
| hard_negative_same_entity | 9.822012901306152 | 7.590243816375732 | 49 | 0.23664 | 1.502273 | 0.224873 |
| orig_only | 9.819293022155762 | 7.483432769775391 | 49 | 0.21872 | 1.503456 | 0.225315 |
| shuffled_pair_adjacent | 9.817875862121582 | 7.611945152282715 | 49 | 0.13072 | 1.469868 | 0.220052 |
