# tail rephase probe and experience utilization route tail-rephase score probe and next-route reading

## Score probe

| model | BLiMP | Suppl | EWoK | Entity | COMPS | GPIQA | Reading | cheap7 | Δ vs legal100M | Δ vs continuous90M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legal100M_ref | 65.87 | 61.17 | 50.39 | 27.40 | 52.01 | 36.065 | 8.140 | 43.0062 | +0.0000 | -0.0738 |
| continuous_90M | 66.13 | 61.40 | 50.22 | 27.76 | 51.90 | 36.065 | 8.085 | 43.0800 | +0.0738 | +0.0000 |
| matched_lr_chunkshuffle_90M | 66.48 | 60.70 | 51.02 | 27.40 | 51.84 | 37.050 | 8.635 | 43.3036 | +0.2973 | +0.2236 |
| base_lr_chunkshuffle_90M | 65.89 | 59.29 | 51.68 | 27.44 | 52.03 | 37.550 | 7.665 | 43.0779 | +0.0716 | -0.0021 |
| matched_lr_original_tail_90M | 65.84 | 61.24 | 50.53 | 28.13 | 51.66 | 36.065 | 8.170 | 43.0907 | +0.0845 | +0.0107 |

## Reading

- tail rephase experiment design matched-LR token-chunk shuffle 90M: cheap7 `43.3036`, Δ vs legal100M `+0.2973`, Δ vs continuous90M `+0.2236`.
- tail rephase probe and experience utilization route matched-LR original-tail replay 90M: cheap7 `43.0907`, Δ vs legal100M `+0.0845`, Δ vs continuous90M `+0.0107`.
- Base-LR token-chunk shuffle 90M: cheap7 `43.0779`, Δ vs legal100M `+0.0716`.
- The original-tail replay consumes row 518144 through 647399 of the frozen 100M stream, 19,965,632 words, ending at actual 100,000,000 words. It removes the post-80M data-order/presentation change from the matched-LR comparison.
- The larger tail rephase experiment design matched-LR movement does not reproduce under the original row stream. Fresh optimizer state / low-LR tail rephase is not the missing broad mechanism.
- No full official evaluation, second seed, 95M/100M endpoint sweep, base-LR replay, or optimizer/LR continuation is supported by these numbers.

## CPU-only route asset preserved

The distinct next candidate is faithful word-boundary experience utilization on the legal16 compact-view reinvest stream. It preserves the corpus and tokenizer but changes whether every charged word's tokenizer tokens become visible and predictable within the 100M-word allowance.

| arm | charged words | steps | active tokens per epoch | verification |
|---|---:|---:|---:|---|
| U256 | 100000000 | 2530 | 14664519 | True |
| U64_128_256 | 100000000 | 2530 | 14664519 | True |

JSON: `experiments/archive/frontier_consolidation/data/tail_rephase_summary/tail_rephase_summary.json`
