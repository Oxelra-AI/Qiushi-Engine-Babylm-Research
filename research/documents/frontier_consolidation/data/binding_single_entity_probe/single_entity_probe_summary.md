# earlier analysis single-entity binding probe
Inference-only probe over earlier analysis held recombination binding rows. Affected rows are rewritten as one-entity update sentences; unaffected rows are rewritten as one-entity no-change sentences.
## Main variant accuracies
| model | original affected two-entity | single affected update | original unaffected two-entity | single unaffected no-change | affected single-two delta |
|---|---:|---:|---:|---:|---:|
| scale1p75_chck100 | 0.297 | 0.432 | 0.781 | 0.599 | 0.135 |
| scale1p75_chck82 | 0.318 | 0.417 | 0.755 | 0.589 | 0.099 |
| scale1p75_chck84 | 0.286 | 0.443 | 0.771 | 0.562 | 0.156 |

## Interpretation notes
- If `single_affected_update` is high while original affected two-entity is low, the existing model knows many verb->result mappings but fails entity binding under competition.
- If `single_affected_update` remains low, the route needs transition-result signal or representation learning, not only slot separation.
- `single_unaffected_nochange` is an easy preservation control; high values there do not establish binding, only retention without competing event.
- This is not BabyLM official evaluation and not model training. It only sharpens the next mechanism route.
