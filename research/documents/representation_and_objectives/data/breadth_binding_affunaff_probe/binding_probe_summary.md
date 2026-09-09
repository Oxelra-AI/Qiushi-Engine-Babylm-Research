# earlier analysis breadth binding affected/unaffected probe

CPU-only inference on compositional update test design counterbalanced binding rows, applied to already-trained first-basin DeBERTa MAX view/repeat/breadth checkpoints. It asks whether V-B is a balanced correspondence-like gain or an affected/unaffected offset.

## Per-arm held recombination readout

| arm | checkpoint | affected acc | unaffected acc | EEBF | affected margin | unaffected margin | signed margin | train binding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| view | chck_80M | +45.312 | +56.771 | +51.042 | -0.442 | +0.538 | -0.981 | +51.953 |
| view | chck_90M | +43.750 | +58.333 | +51.042 | -0.524 | +0.653 | -1.177 | +52.344 |
| view | chck_100M | +44.792 | +58.333 | +51.562 | -0.512 | +0.634 | -1.146 | +52.539 |
| repeat | chck_80M | +21.875 | +77.083 | +49.479 | -1.278 | +1.435 | -2.713 | +48.438 |
| repeat | chck_90M | +23.958 | +78.125 | +51.042 | -1.295 | +1.489 | -2.784 | +48.633 |
| repeat | chck_100M | +23.958 | +77.604 | +50.781 | -1.293 | +1.481 | -2.774 | +48.438 |
| breadth | chck_80M | +34.896 | +61.458 | +48.177 | -0.663 | +0.616 | -1.280 | +50.391 |
| breadth | chck_90M | +39.583 | +61.979 | +50.781 | -0.588 | +0.536 | -1.124 | +49.414 |
| breadth | chck_100M | +38.542 | +62.500 | +50.521 | -0.548 | +0.488 | -1.036 | +49.414 |

## Contrasts: affected/unaffected deltas

| contrast | checkpoint | affected Δ pp | unaffected Δ pp | EEBF Δ pp | affected margin Δ | unaffected margin Δ | signed margin Δ | offset-like |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| VminusR | chck_80M | +23.438 | -20.312 | +1.563 | +0.835 | -0.897 | +1.732 | True |
| VminusB | chck_80M | +10.417 | -4.688 | +2.865 | +0.221 | -0.078 | +0.299 | True |
| BminusR | chck_80M | +13.021 | -15.625 | -1.302 | +0.614 | -0.818 | +1.433 | True |
| VminusR | chck_90M | +19.792 | -19.792 | +0.000 | +0.771 | -0.836 | +1.607 | True |
| VminusB | chck_90M | +4.167 | -3.646 | +0.260 | +0.064 | +0.118 | -0.054 | True |
| BminusR | chck_90M | +15.625 | -16.146 | -0.260 | +0.707 | -0.953 | +1.660 | True |
| VminusR | chck_100M | +20.833 | -19.271 | +0.781 | +0.781 | -0.847 | +1.628 | True |
| VminusB | chck_100M | +6.250 | -4.167 | +1.042 | +0.036 | +0.146 | -0.110 | True |
| BminusR | chck_100M | +14.583 | -15.104 | -0.260 | +0.745 | -0.993 | +1.738 | True |

## Late-window means

| contrast | quantity | n | mean | min | max |
|---|---|---:|---:|---:|---:|
| VminusR | affected_delta | 3 | +21.354 | +19.792 | +23.438 |
| VminusR | unaffected_delta | 3 | -19.792 | -20.312 | -19.271 |
| VminusR | EEBF_delta | 3 | +0.781 | +0.000 | +1.563 |
| VminusR | affected_margin_delta | 3 | +0.796 | +0.771 | +0.835 |
| VminusR | unaffected_margin_delta | 3 | -0.860 | -0.897 | -0.836 |
| VminusR | signed_bias_margin_delta | 3 | +1.656 | +1.607 | +1.732 |
| VminusR | offset_like_count | 3 | +3.000 | +3.000 | +3.000 |
| VminusB | affected_delta | 3 | +6.944 | +4.167 | +10.417 |
| VminusB | unaffected_delta | 3 | -4.167 | -4.688 | -3.646 |
| VminusB | EEBF_delta | 3 | +1.389 | +0.260 | +2.865 |
| VminusB | affected_margin_delta | 3 | +0.107 | +0.036 | +0.221 |
| VminusB | unaffected_margin_delta | 3 | +0.062 | -0.078 | +0.146 |
| VminusB | signed_bias_margin_delta | 3 | +0.045 | -0.110 | +0.299 |
| VminusB | offset_like_count | 3 | +3.000 | +3.000 | +3.000 |
| BminusR | affected_delta | 3 | +14.410 | +13.021 | +15.625 |
| BminusR | unaffected_delta | 3 | -15.625 | -16.146 | -15.104 |
| BminusR | EEBF_delta | 3 | -0.608 | -1.302 | -0.260 |
| BminusR | affected_margin_delta | 3 | +0.689 | +0.614 | +0.745 |
| BminusR | unaffected_margin_delta | 3 | -0.922 | -0.993 | -0.818 |
| BminusR | signed_bias_margin_delta | 3 | +1.611 | +1.433 | +1.738 |
| BminusR | offset_like_count | 3 | +3.000 | +3.000 | +3.000 |

## Interpretation

- **VminusB_binding**: If V-B were the same changed-state offset as V-R/B-R, it should show affected gains paired with unaffected losses. The saved table should therefore be read through affected_delta, unaffected_delta, and EEBF_delta rather than aggregate Entity alone.
- **late_mean_VminusB**: late 80/90/100 V-B mean affected_delta=0.06944444444444446, unaffected_delta=-0.04166666666666663, EEBF_delta=0.013888888888888914
- **permuted_companion_decision**: A balanced positive V-B on binding rows would strengthen the case for a permuted companion arm; an offset-like V-B would weaken correspondence/addressability and push interpretation toward decision bias or dataset/evaluation mixture.
- **scope**: This remains a cheap binding panel over a synthetic-natural bridge substrate, not official BabyLM evaluation and not a completed general principle.

## Files

- plan_json: `experiments/archive/representation_and_objectives/data/breadth_binding_affunaff_probe/binding_probe_plan.json`
- summary_json: `experiments/archive/representation_and_objectives/data/breadth_binding_affunaff_probe/binding_probe_summary.json`
- summary_md: `research/documents/representation_and_objectives/data/breadth_binding_affunaff_probe/binding_probe_summary.md`
- per_model_csv: `experiments/archive/representation_and_objectives/data/breadth_binding_affunaff_probe/binding_per_model_summary.csv`
- contrast_csv: `experiments/archive/representation_and_objectives/data/breadth_binding_affunaff_probe/binding_contrast_deltas.csv`
- window_summary_csv: `experiments/archive/representation_and_objectives/data/breadth_binding_affunaff_probe/binding_window_summary.csv`
- per_item_csv: `experiments/archive/representation_and_objectives/data/breadth_binding_affunaff_probe/binding_per_item_records.csv`
