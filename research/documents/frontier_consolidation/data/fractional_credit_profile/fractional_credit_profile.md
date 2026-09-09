# successor route comparison while minfreq runs fractional whole-word credit profile

Training-text-only CPU analysis of actual selected WWM groups; no model forward, no official evaluation, no GPU.

Sampled `79` batches, `453739` selected groups, `662183` selected tokens; mean selected pieces/group `1.4594`.

## Key category share shifts

Shares are average-batch loss-credit shares. Alpha 0 is token-mean; alpha 1 is global word-mean.

| category | tokenmean a=0 | a=0.25 Δ | a=0.5 Δ | a=0.75 Δ | wordmean a=1 Δ |
|---|---:|---:|---:|---:|---:|
| bpe_len::1 | 47.595% | +5.952% | +11.644% | +16.961% | +21.828% |
| bpe_len::3 | 16.385% | -2.375% | -4.606% | -6.629% | -8.412% |
| has_upper::True | 26.692% | -2.291% | -4.379% | -6.241% | -7.865% |
| bpe_len::2 | 27.319% | -1.471% | -3.271% | -5.279% | -7.387% |
| cue::none | 78.606% | -1.283% | -2.431% | -3.438% | -4.306% |
| source::childes | 28.401% | -0.827% | -1.669% | -2.488% | -3.254% |
| bpe_len::4 | 4.247% | -0.868% | -1.603% | -2.209% | -2.697% |
| has_digit::True | 3.198% | -0.786% | -1.392% | -1.852% | -2.199% |
| cue::mental_social_dialogue | 8.010% | +0.566% | +1.082% | +1.542% | +1.946% |
| source::inherited_qwen_pair_packed | 15.607% | +0.343% | +0.688% | +1.024% | +1.339% |
| cue::spatial_state | 4.773% | +0.328% | +0.625% | +0.890% | +1.121% |
| bpe_len::8plus | 0.532% | -0.199% | -0.326% | -0.406% | -0.456% |
| cue::physical_dynamics | 1.077% | +0.083% | +0.160% | +0.229% | +0.291% |
| cue::material_property | 1.403% | +0.051% | +0.096% | +0.135% | +0.167% |
| source::fineweb_source_compact_view_pair | 4.314% | +0.008% | +0.024% | +0.045% | +0.071% |

## Family-level half-L1 share shift versus tokenmean

| alpha | family | mean half-L1 shift |
|---:|---|---:|
| 0.25 | bpe_len | 5.952% |
| 0.25 | cue | 1.319% |
| 0.25 | has_digit | 0.786% |
| 0.25 | has_upper | 2.291% |
| 0.25 | source | 1.254% |
| 0.50 | bpe_len | 11.644% |
| 0.50 | cue | 2.501% |
| 0.50 | has_digit | 1.392% |
| 0.50 | has_upper | 4.379% |
| 0.50 | source | 2.393% |
| 0.75 | bpe_len | 16.961% |
| 0.75 | cue | 3.542% |
| 0.75 | has_digit | 1.852% |
| 0.75 | has_upper | 6.241% |
| 0.75 | source | 3.409% |
| 1.00 | bpe_len | 21.828% |
| 1.00 | cue | 4.441% |
| 1.00 | has_digit | 2.199% |
| 1.00 | has_upper | 7.865% |
| 1.00 | source | 4.299% |

## Interpretation
- Alpha=0 is token-mean and alpha=1 is the closed global word-mean objective; intermediate alpha values are quantitatively different only if they materially reduce the content/name/digit downweighting while preserving a small amount of word-unit normalization.
- This profile cannot predict BabyLM scores. It is useful for deciding whether a later fractional/late credit fork is a targeted optimization-mechanism test rather than an unprincipled rerun of the failed global word-mean route.

Full JSON: `experiments/archive/frontier_consolidation/data/fractional_credit_profile/fractional_credit_profile.json`
