# coherence margin signal isolation coherence-margin NLL-gap comparison

Status: **COMPLETE**

| probe | coherent NLL | disrupted NLL | bad-coh gap | SE | z | pos frac |
|---|---:|---:|---:|---:|---:|---:|
| pilot_train64 | 6.648963 | 6.649938 | +0.000975 | 0.000342 | +2.85 | 0.5070 |
| ref2m_train64 | 6.715376 | 6.714922 | -0.000453 | 0.000218 | -2.08 | 0.4945 |
| ref4m_train64 | 6.256158 | 6.281168 | +0.025010 | 0.004073 | +6.14 | 0.5547 |
| pilot_holdout64_after2M | 6.736957 | 6.737282 | +0.000326 | 0.000365 | +0.89 | 0.5016 |
| ref2m_holdout64_after2M | 6.792955 | 6.793036 | +0.000080 | 0.000157 | +0.51 | 0.5128 |
| ref4m_holdout64_after2M | 6.293997 | 6.311919 | +0.017922 | 0.003463 | +5.18 | 0.5272 |

## Key contrasts

| contrast | delta gap | SE | z |
|---|---:|---:|---:|
| train64_pilot_minus_ref2m | +0.001428 | 0.000405 | +3.52 |
| holdout64_pilot_minus_ref2m | +0.000246 | 0.000398 | +0.62 |
| train64_pilot_minus_ref4m | -0.024035 | 0.004087 | -5.88 |
| holdout64_pilot_minus_ref4m | -0.017596 | 0.003482 | -5.05 |
| train64_ref4m_minus_ref2m | +0.025463 | 0.004078 | +6.24 |
| holdout64_ref4m_minus_ref2m | +0.017842 | 0.003466 | +5.15 |

On these fixed legal-row probes, the reopen structured context margin route lambda>0 pilot has only near-zero coherent-over-disrupted preference and is close to the coherent-word matched 2M ordinary reference, while the 4M ordinary reference already shows a much larger positive preference. Therefore the short margin pilot has not yet established the intended coherent-context likelihood separation; any encouraging official score must be isolated against a same-charge same-row lambda-zero arm before being attributed to the margin signal.

Do not extend coherence-margin to 20M from pilot score alone. If cheap7 is promising, run the minimal 4M charged lambda-zero same-row arm first and compare both official cheap7 and this NLL-gap probe.

JSON: `experiments/archive/frontier_consolidation/data/cohmargin_nll_gap_comparison/cohmargin_nll_gap_comparison.json`
