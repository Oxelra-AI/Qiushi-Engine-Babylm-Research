# wordmean screen and substrate constraints word-mean MLM credit-scale analysis

CPU-only actual training-stream batches; no model forward pass, official evaluation text, GPU, corpus/tokenizer change, or new training.

## Why this matters

On the same masked batch, token-mean MLM uses weight `1/T` for every selected BPE token. The earlier analysis word-mean objective uses `1/(G*k)` for a selected token inside a selected whole-word group of BPE length `k`. The per-token ratio is therefore `mean(k_selected)/k`. This changes both relative credit across word lengths and the squared weight norm of the logit-level gradient, so unchanged learning rate is not a pure isolation of credit allocation.

## Sample

- Loaded words: `80000000` from `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl`
- Total batches: `2024`; sampled batches: `126` (front `32`, stride request `96`)
- Mask probability: `0.15`; tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`

## Selected masked word groups

- Selected groups: `726919`; selected tokens: `1060869`; mean tokens/group: `1.4594`
- Uncorrelated-position logit-gradient RMS weight scale, wordmean/tokenmean: `1.0980`; LR multiplier to match that scale: `0.9107`
- Perfect within-word group-coherent approximation scale, wordmean/tokenmean: `0.8629`; LR multiplier under that approximation: `1.1588`

### Credit by selected word length

| k BPE tokens in selected word | groups | group frac = wordmean credit | token frac = tokenmean credit | Δ wordmean-tokenmean | per-token weight ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 504476 | 69.399% | 47.553% | 21.846% | 1.459 |
| 2 | 144510 | 19.880% | 27.244% | -7.364% | 0.730 |
| 3 | 58630 | 8.066% | 16.580% | -8.514% | 0.486 |
| 4 | 11275 | 1.551% | 4.251% | -2.700% | 0.365 |
| 5 | 4454 | 0.613% | 2.099% | -1.486% | 0.292 |
| 6 | 2556 | 0.352% | 1.446% | -1.094% | 0.243 |
| 7 | 486 | 0.067% | 0.321% | -0.254% | 0.208 |
| 8 | 228 | 0.031% | 0.172% | -0.141% | 0.182 |
| 9 | 71 | 0.010% | 0.060% | -0.050% | 0.162 |
| 10 | 33 | 0.005% | 0.031% | -0.027% | 0.146 |
| 11 | 66 | 0.009% | 0.068% | -0.059% | 0.133 |
| 12 | 44 | 0.006% | 0.050% | -0.044% | 0.122 |
| 13 | 52 | 0.007% | 0.064% | -0.057% | 0.112 |
| 14 | 26 | 0.004% | 0.034% | -0.031% | 0.104 |
| 15 | 4 | 0.001% | 0.006% | -0.005% | 0.097 |
| 17 | 2 | 0.000% | 0.003% | -0.003% | 0.086 |
| 19 | 1 | 0.000% | 0.002% | -0.002% | 0.077 |
| 22 | 1 | 0.000% | 0.002% | -0.002% | 0.066 |
| 23 | 1 | 0.000% | 0.002% | -0.002% | 0.063 |
| 28 | 1 | 0.000% | 0.003% | -0.003% | 0.052 |
| 32 | 1 | 0.000% | 0.003% | -0.003% | 0.046 |
| 70 | 1 | 0.000% | 0.007% | -0.006% | 0.021 |

## Visible candidate word groups before masking

- Candidate visible groups: `4850899`; candidate tokens: `7080181`; mean tokens/group: `1.4596`

| k BPE tokens in visible word | groups | group frac | token frac | wordmean-tokenmean credit shift if selected uniformly | per-token ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 3368042 | 69.431% | 47.570% | 21.861% | 1.460 |
| 2 | 962163 | 19.835% | 27.179% | -7.344% | 0.730 |
| 3 | 391041 | 8.061% | 16.569% | -8.508% | 0.487 |
| 4 | 75094 | 1.548% | 4.242% | -2.694% | 0.365 |
| 5 | 30560 | 0.630% | 2.158% | -1.528% | 0.292 |
| 6 | 17297 | 0.357% | 1.466% | -1.109% | 0.243 |
| 7 | 3292 | 0.068% | 0.325% | -0.258% | 0.209 |
| 8 | 1351 | 0.028% | 0.153% | -0.125% | 0.182 |
| 9 | 495 | 0.010% | 0.063% | -0.053% | 0.162 |
| 10 | 260 | 0.005% | 0.037% | -0.031% | 0.146 |
| 11 | 427 | 0.009% | 0.066% | -0.058% | 0.133 |
| 12 | 314 | 0.006% | 0.053% | -0.047% | 0.122 |
| 13 | 314 | 0.006% | 0.058% | -0.051% | 0.112 |
| 14 | 166 | 0.003% | 0.033% | -0.029% | 0.104 |
| 15 | 17 | 0.000% | 0.004% | -0.003% | 0.097 |
| 16 | 9 | 0.000% | 0.002% | -0.002% | 0.091 |
| 17 | 8 | 0.000% | 0.002% | -0.002% | 0.086 |
| 18 | 3 | 0.000% | 0.001% | -0.001% | 0.081 |
| 19 | 7 | 0.000% | 0.002% | -0.002% | 0.077 |
| 20 | 5 | 0.000% | 0.001% | -0.001% | 0.073 |
| 21 | 1 | 0.000% | 0.000% | -0.000% | 0.070 |
| 22 | 2 | 0.000% | 0.001% | -0.001% | 0.066 |
| 23 | 3 | 0.000% | 0.001% | -0.001% | 0.063 |
| 24 | 1 | 0.000% | 0.000% | -0.000% | 0.061 |
| 25 | 2 | 0.000% | 0.001% | -0.001% | 0.058 |
| 28 | 4 | 0.000% | 0.002% | -0.001% | 0.052 |
| 29 | 1 | 0.000% | 0.000% | -0.000% | 0.050 |
| 30 | 4 | 0.000% | 0.002% | -0.002% | 0.049 |
| 31 | 2 | 0.000% | 0.001% | -0.001% | 0.047 |
| 32 | 1 | 0.000% | 0.000% | -0.000% | 0.046 |
| 35 | 1 | 0.000% | 0.000% | -0.000% | 0.042 |
| 36 | 3 | 0.000% | 0.002% | -0.001% | 0.041 |
| 37 | 1 | 0.000% | 0.001% | -0.001% | 0.039 |
| 38 | 1 | 0.000% | 0.001% | -0.001% | 0.038 |
| 43 | 2 | 0.000% | 0.001% | -0.001% | 0.034 |
| 45 | 1 | 0.000% | 0.001% | -0.001% | 0.032 |
| 48 | 1 | 0.000% | 0.001% | -0.001% | 0.030 |
| 51 | 2 | 0.000% | 0.001% | -0.001% | 0.029 |
| 70 | 1 | 0.000% | 0.001% | -0.001% | 0.021 |

## Per-batch scale summary

- Mean selected tokens/batch: `8419.60`; mean selected groups/batch: `5769.20`; mean selected tokens/group: `1.4600`
- Mean uncorrelated logit-gradient RMS scale: `1.0980` (p10 `1.0949`, p90 `1.1019`)
- Mean group-coherent approximation scale: `0.8633`

## Interpretation for the running earlier analysis screen

The earlier analysis run remains a useful 80M screen of a general legal objective, but its result must not be read as pure evidence for relative word credit alone. It upweights one-piece words and downweights multi-piece words on matched selected WWM batches, while also changing the RMS norm of the logit-level loss weights. A positive result should be followed by scale-aware comparison before firm mechanism attribution; a negative result should stop this objective family rather than erode the already validated compact-view reinvestment data mechanism.

Full JSON: `experiments/archive/frontier_consolidation/data/wordmean_credit_scale_analysis/wordmean_credit_scale_analysis.json`
