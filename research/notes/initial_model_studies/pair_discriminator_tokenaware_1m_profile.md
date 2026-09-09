# pair discriminator tokenaware 1m profile token-aware/equal-row pair discriminator 1M profile

Evidence JSON: `experiments/archive/initial_model_studies/data/pair_discriminator_tokenaware_1m_profile.json`

| column | true_pair | hard_neg | orig | shuffled | true-hardneg | true-orig | true-shuf |
|---|---:|---:|---:|---:|---:|---:|---:|
| blimp_fast | 52.6100 | 53.6800 | 53.4200 | 52.5300 | -1.0700 | -0.8100 | +0.0800 |
| supplement_fast | 49.6000 | 48.0000 | 45.2000 | 46.0000 | +1.6000 | +4.4000 | +3.6000 |
| ewok_fast | 49.2700 | 52.8200 | 50.7300 | 51.8200 | -3.5500 | -1.4600 | -2.5500 |
| entity_tracking_fast | 15.9800 | 16.2600 | 17.6700 | 17.7700 | -0.2800 | -1.6900 | -1.7900 |
| comps | 49.7800 | 49.8300 | 49.5900 | 50.0200 | -0.0500 | +0.1900 | -0.2400 |
| reading_eye_tracking | 5.6300 | 6.0800 | 9.8400 | 6.2300 | -0.4500 | -4.2100 | -0.6000 |
| reading_self_paced | 1.5600 | 1.7000 | 3.0700 | 1.9900 | -0.1400 | -1.5100 | -0.4300 |
| Reading_mean | 3.5950 | 3.8900 | 6.4550 | 4.1100 | -0.2950 | -2.8600 | -0.5150 |

## Training/tokenization summary

| arm | loss first | loss last | steps | trunc frac | max untrunc tokens | kept tokens/word | masked tokens/word |
|---|---:|---:|---:|---:|---:|---:|---:|
| true_pair_adjacent | 9.81294059753418 | 7.568047523498535 | 47 | 0.0 | 256 | 1.528698 | 0.229258 |
| hard_negative_same_entity | 9.82654094696045 | 7.530747890472412 | 47 | 0.0 | 256 | 1.528698 | 0.227988 |
| orig_only | 9.826565742492676 | 7.444480895996094 | 47 | 0.0 | 256 | 1.523976 | 0.228027 |
| shuffled_pair_adjacent | 9.814155578613281 | 7.623090744018555 | 47 | 0.0 | 256 | 1.483888 | 0.221906 |
