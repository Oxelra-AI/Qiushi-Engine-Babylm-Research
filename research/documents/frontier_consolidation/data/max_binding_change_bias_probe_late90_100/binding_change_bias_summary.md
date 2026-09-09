# changed state bias threat and route decision MAX binding changed-state-bias probe

CPU-only cross data binding comparison result-style counterbalanced binding readout on MAX view/repeat in both DeBERTa basins. The key threat model is a signed shift toward predicting changed state: affected rows improve while unaffected rows decline, leaving EEBF nearly unchanged.

## View-minus-repeat held recombination deltas

| basin | checkpoint | affected Δ pp | unaffected Δ pp | EEBF Δ pp | affected margin Δ | unaffected margin Δ | signed margin shift | bias-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed43022_first_basin | chck_90M | +19.792 | -19.792 | +0.000 | +0.7711 | -0.8356 | +1.6067 | True |
| seed43022_first_basin | chck_100M | +20.833 | -19.271 | +0.781 | +0.7808 | -0.8473 | +1.6281 | True |
| seed43122_second_basin | chck_90M | -2.083 | +8.333 | +3.125 | -0.2075 | +0.6623 | -0.8698 | False |
| seed43122_second_basin | chck_100M | -1.042 | +6.771 | +2.865 | -0.1550 | +0.6095 | -0.7645 | False |

## Compact interpretation

- **late_pair_summary**: Across late 80/90/100M pairs, mean affected Δ=0.09375, mean unaffected Δ=-0.059895833333333315, mean EEBF Δ=0.01692708333333337.
- **bias_like_pair_count**: 2/4 all checkpoint pairs and 2/4 late pairs have affected gain, unaffected loss, and smaller balanced movement.
- **use_before_permuted_training**: If second-basin and numops splits show the same signed bias, do not treat Entity V-R as record-addressability evidence; if affected gains persist without an unaffected cost, the aligned-versus-permuted arm becomes more informative.

Files: summary JSON and CSVs in this directory.
