# dose arm correction and pilot relation-arm state-margin: T/U/N and format-matched T-U

Created UTC: 2026-09-06T23:14:12Z

## Why this rerun was necessary

calibrated route options used T-N to subtract neutral phrase preference, but the split arms were themselves trained in a source-alone-plus-query-like format. Their higher N therefore could be a construction-specific neutral-format shift. This note treats the format-matched quantity as primary: on UPDATED_USE packets, T and U both contain source, update sentence, and query frame; the contrast T(new-source)-U(new-source) asks whether the true target update adds more new-state preference than a foreign update does. T, U, and N remain shown side by side so the neutral-format shift is visible rather than hidden.

## UPDATED_USE extended_nontrain: arm-minus-CLEAN central quantities

| seed | arm minus CLEAN | n | raw T | raw U | T-U | T-N | content T-U |
|---:|---|---:|---:|---:|---:|---:|---:|
| 43022 | repeat_minus_clean | 1125 | +0.1887 | -0.0197 | +0.2083 | +0.1608 | +0.2300 |
| 43022 | repeat_split_minus_clean | 1125 | +0.2564 | +0.4353 | -0.1788 | -0.1372 | -0.1912 |
| 43022 | view_minus_clean | 1125 | +0.3644 | -0.0012 | +0.3656 | +0.4519 | +0.3848 |
| 43022 | view_split_minus_clean | 1125 | +0.2499 | +0.2803 | -0.0304 | +0.0464 | -0.0392 |
| 43122 | repeat_minus_clean | 1125 | +0.2974 | -0.0344 | +0.3318 | +0.3736 | +0.3732 |
| 43122 | repeat_split_minus_clean | 1125 | +0.1025 | +0.3480 | -0.2455 | -0.2126 | -0.2505 |
| 43122 | view_minus_clean | 1125 | +0.4789 | +0.0753 | +0.4036 | +0.4942 | +0.4725 |
| 43122 | view_split_minus_clean | 1125 | +0.2128 | +0.3439 | -0.1311 | +0.0145 | -0.1189 |

## Per-arm T/U/N means for UPDATED_USE extended_nontrain

| seed | arm | n | T | U | N | T-U | T-N |
|---:|---|---:|---:|---:|---:|---:|---:|
| 43022 | clean | 1125 | +0.8413 | -4.4812 | -4.7642 | +5.3224 | +5.6055 |
| 43022 | repeat | 1125 | +1.0299 | -4.5008 | -4.7363 | +5.5308 | +5.7662 |
| 43022 | view | 1125 | +1.2057 | -4.4824 | -4.8517 | +5.6881 | +6.0573 |
| 43022 | repeat_split | 1125 | +1.0977 | -4.0459 | -4.3706 | +5.1436 | +5.4683 |
| 43022 | view_split | 1125 | +1.0911 | -4.2009 | -4.5607 | +5.2920 | +5.6519 |
| 43122 | clean | 1125 | +0.8196 | -4.3961 | -4.6947 | +5.2157 | +5.5143 |
| 43122 | repeat | 1125 | +1.1170 | -4.4305 | -4.7709 | +5.5475 | +5.8879 |
| 43122 | view | 1125 | +1.2986 | -4.3208 | -4.7099 | +5.6194 | +6.0085 |
| 43122 | repeat_split | 1125 | +0.9221 | -4.0482 | -4.3796 | +4.9703 | +5.3017 |
| 43122 | view_split | 1125 | +1.0324 | -4.0522 | -4.4963 | +5.0846 | +5.5287 |

## Interpretation boundary

This table decides whether the state-margin readout can be used as causal-locality support. If VIEW remains above CLEAN while split arms stay near CLEAN on T-U in both seeds, the result is relation-locality evidence. If T-U collapses or reverses while T-N shows the old pattern, the result is instead a phrase-prior/source-format shift in split training. REPEAT sharing VIEW's sign on T-U would point to a shared second-sentence-use component rather than the exact-copy liability seen on compact changed-form probes.
