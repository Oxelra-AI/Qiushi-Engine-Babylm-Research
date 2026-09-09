# changed state bias threat and route decision MAX binding changed-state-bias probe

CPU-only cross data binding comparison result-style counterbalanced binding readout on MAX view/repeat in both DeBERTa basins. The key threat model is a signed shift toward predicting changed state: affected rows improve while unaffected rows decline, leaving EEBF nearly unchanged.

## View-minus-repeat held recombination deltas

| basin | checkpoint | affected Δ pp | unaffected Δ pp | EEBF Δ pp | affected margin Δ | unaffected margin Δ | signed margin shift | bias-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed43022_first_basin | chck_80M | +23.438 | -20.312 | +1.563 | +0.8353 | -0.8966 | +1.7320 | True |
| seed43122_second_basin | chck_80M | +0.521 | +7.812 | +4.167 | -0.3949 | +0.8501 | -1.2450 | False |

## Compact interpretation

- **late_pair_summary**: Across late 80/90/100M pairs, mean affected Δ=0.11979166666666667, mean unaffected Δ=-0.0625, mean EEBF Δ=0.02864583333333337.
- **bias_like_pair_count**: 1/2 all checkpoint pairs and 1/2 late pairs have affected gain, unaffected loss, and smaller balanced movement.
- **use_before_permuted_training**: If second-basin and numops splits show the same signed bias, do not treat Entity V-R as record-addressability evidence; if affected gains persist without an unaffected cost, the aligned-versus-permuted arm becomes more informative.

Files: summary JSON and CSVs in this directory.
