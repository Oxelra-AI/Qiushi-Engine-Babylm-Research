# earlier analysis interpretation of frozen structural contrast-margin v1

Panel: `experiments/archive/frontier_consolidation/data/structural_contrast_scores/scale1p75_late_panel.json`
Quality report: `experiments/archive/frontier_consolidation/data/structural_probe_quality/structural_probe_quality_report.json`

## Main result

- Frozen composite Pearson vs known scale1.75 cheap7: **-0.293627**; Spearman: **-0.119048**.
- Frozen composite maximum: **chck_77M** with value -2.371481; rank of `chck_82M`: **4**.
- Structural-only weighted score without the surface subtraction maximum: **chck_77M**; Pearson -0.389537; rank of `chck_82M`: **6**.
- Composite movement from 80M to 82M: 0.011918; from 82M to 100M: -0.062241.

## Readouts with positive association but no 82M peak

| readout | best ckpt | rank of 82M | Pearson | Spearman | 80→82 | 82→100 |
|---|---:|---:|---:|---:|---:|---:|
| entity_depth_hard_accuracy/depth3 | chck_79M | 4 | 0.636747 | 0.538932 | 0.0 | -0.05 |
| entity_depth_mean/depth3 | chck_79M | 4 | 0.411777 | 0.52381 | -0.024781 | -0.357887 |
| family_hard_accuracy/directed_relation | chck_79M | 4 | 0.327435 | 0.239525 | -0.004808 | -0.014423 |
| family_hard_accuracy/temporal_order | chck_83M | 2 | 0.561211 | 0.60151 | 0.004 | -0.008 |
| family_mean/polarity_relation | chck_81M | 3 | 0.410486 | 0.238095 | 0.010872 | -0.007663 |
| family_trimmed/directed_relation | chck_100M | 3 | 0.492887 | 0.261905 | 0.003472 | 0.014493 |
| family_trimmed/polarity_relation | chck_81M | 3 | 0.374183 | 0.261905 | 0.008378 | -0.007812 |

## Static construction context

- double_period: 256
- perturbed_lowercase_start: 249
- entity_put_to_template: 91
- childes_markup: 11
- space_before_punct: 4
- perturbed_keeps_negator: 4
- focus_token_count_imbalance_ge3: 1

## Scientific reading

- **frozen_v1_result**: The frozen composite does not track the known scale1.75 late official cheap7 peak: Pearson is negative and its maximum is chck_77M, not chck_82M.
- **family_result**: Some single-family readouts have modest positive association with cheap7, especially directed_relation, polarity_relation, and entity_state depth3, but none produces a reliable 82M peak and their best checkpoints differ.
- **quality_context**: The scorer sees every pair, but the v1 pair set contains construction artifacts: temporal perturbations often start lowercase and carry double punctuation, entity_state uses synthetic templates with some awkward put-to wording, and surface substitutions produce very large margins that dominate the frozen surface subtraction.
- **next_research_direction**: Do not spend official evaluation on a cross-trajectory v1 test or retune v1 after seeing these scores. Use the failure to refine the mechanism search or build a cleaner structural source only from a newly frozen construction before model scoring.

JSON: `experiments/archive/frontier_consolidation/data/structural_contrast_panel_interpretation/structural_contrast_v1_panel_interpretation.json`
