# earlier analysis register interpretation scale

File-only synthesis from existing tables; no model loading, training, evaluation, GPU, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard.

## Scalar anchors

- Old DeBERTa MAX companion-form specificity, V-B exEntity5 common10_80: -0.1694; late80_100: +0.0503.
- Old DeBERTa broad stream/composition leg, B-C_old exEntity5 common10_80: +0.7675.
- Old DeBERTa total V-C_old exEntity5 common10_80: +0.5981; matched-clean MAX late V-Cmax exEntity5: +0.3853.
- RoBERTa MAX view-clean late exEntity5: -0.6873; late cheap6: -0.5283.
- Two-seed same-data treatment-delta mean absolute spread: cheap6 full10_100 +0.4199, mature70_100 +0.2833; exEntity5 full10_100 +0.2897, mature70_100 +0.1765.

## Register design

| arm | rho | FineWeb words | displaced child/sub frac | displaced adult frac | row mean |
|---|---:|---:|---:|---:|---:|
| childsub_posmatched | 0.010596 | 105962 | 0.9412 | 0.0352 | 3977.3 |
| adult_posmatched | 0.010596 | 105962 | 0.1253 | 0.7966 | 3992.5 |

## Reading rules for incoming scores

The primary contrast is adult-prose-removal versus child/subtitle-removal at identical admitted FineWeb. A mixed quarter_1x null should be treated as potentially cancelled signal, not as an automatic stop.

If both arms agree and both beat the two-clean anchor on cheap6/exEntity5 over the common window, the DeBERTa result supports a small admixture/register-presence effect. If they agree near zero or below clean, the old admission leg is not stable at rho≈0.011 under this control. If they diverge, the principle must include the value of the removed clean register; direct child-minus-adult sign tells which register is costly to sacrifice.

Entity must remain separate because previous positive Entity movement was operation-skewed. RoBERTa negative late V-C keeps any positive DeBERTa register law coordinate-conditional until transfer is tested or explained.

CSV: `experiments/archive/frontier_consolidation/data/register_interpretation_scale/interpretation_scale_rows.csv`
JSON: `experiments/archive/frontier_consolidation/data/register_interpretation_scale/register_interpretation_scale_summary.json`
