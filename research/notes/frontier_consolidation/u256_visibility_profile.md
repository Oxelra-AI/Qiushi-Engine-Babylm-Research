# active endpoint consumption and u256 interpretation — legal16k U256 visibility/source-tail profile

CPU-only accounting on the exact legal compact-view reinvest corpus and spatial repair route status legal16k tokenizer. It does not launch training or evaluation.

## Row256 versus U256 exposure

- 10M pool words: 10,000,000; legal16k raw tokens: 14,664,519.
- spatial repair route status row256 active tokens: 14,294,893; U256 active tokens: 14,664,519.
- U256 recovers 369,626 tokens per 10M epoch, an active-token ratio of 1.025857 and recovered fraction 2.5205% of raw tokens.
- Fully hidden words recovered by U256: 197,806 / 10,000,000 (1.9781%); boundary-hidden token pieces recovered: 10,461; boundary-split rows: 6,381.

This confirms U256 is a small but real visibility repair. Its 20M score movement (+0.9179 cheap7) is much larger than the +2.59% token-count change, so the endpoint question is whether the newly visible row tails change useful credit flow or only rotate competence.

## Update mass and masking

- First 20M row256 reconstructed updates: 506 steps, 28,602,220 active tokens, 4,292,423 masked targets, loss 9.837543→3.755582.
- First 20M U256 actual log: 506 steps, 29,329,038 active tokens from the matching dry-run accounting, 4,405,957 realized masked targets, loss 9.826857→3.718591.
- U256/row256 first-20M active-token ratio: 1.025411; masked-target ratio: 1.026450; masked-per-active ratio: 1.001013.
- Full U256 dry-run: 2530 steps, 146,645,190 active tokens, 21,998,645 masked targets, pad fraction 0.2829.
- U64_128_256 full dry-run has the same charged/raw-token exposure but 186,120,384 nominal slots and pad fraction 0.2121; it adds short-context/order effects, so U256 remains the cleaner first endpoint.

## Source-tail distribution

- childes: recovered tokens 223,846 / raw 4,261,468 (5.253%); hidden full words 124,156; suffix cues action=3554, causal=1895, physical=1456, spatial=4210, material=448, quantitative=2880, social=3419.
- simple_wiki: recovered tokens 74,970 / raw 1,693,623 (4.427%); hidden full words 41,532; suffix cues action=266, causal=613, physical=135, spatial=2387, material=55, quantitative=3315, social=245.
- open_subtitles: recovered tokens 38,835 / raw 2,644,137 (1.469%); hidden full words 19,319; suffix cues action=594, causal=434, physical=114, spatial=794, material=46, quantitative=577, social=615.
- gutenberg: recovered tokens 29,993 / raw 2,483,064 (1.208%); hidden full words 11,741; suffix cues action=101, causal=111, physical=86, spatial=290, material=29, quantitative=2507, social=161.
- qwen_pair_packed: recovered tokens 1,279 / raw 2,226,570 (0.057%); hidden full words 671; suffix cues action=5, causal=13, physical=7, spatial=32, material=1, quantitative=64, social=7.
- cleanqwen_fineweb_compact_view_reinvest: recovered tokens 636 / raw 607,898 (0.105%); hidden full words 341; suffix cues action=2, causal=10, physical=5, spatial=8, material=0, quantitative=13, social=5.
- bnc_spoken: recovered tokens 67 / raw 717,852 (0.009%); hidden full words 46; suffix cues action=1, causal=1, physical=0, spatial=2, material=0, quantitative=6, social=2.
- neutral_cleanqwen_topup_compact_reinvest::open_subtitles: recovered tokens 0 / raw 11 (0.000%); hidden full words 0; suffix cues action=0, causal=0, physical=0, spatial=0, material=0, quantitative=0, social=0.
- switchboard: recovered tokens 0 / raw 29,896 (0.000%); hidden full words 0; suffix cues action=0, causal=0, physical=0, spatial=0, material=0, quantitative=0, social=0.

The recovered suffix mass is source-concentrated rather than uniform. Later U256 endpoint interpretation should compare any EWoK material/social/quantitative or Reading movement against these recovered-tail cues, but these lexicon counts are descriptive only and must not be used as benchmark-conditioned data selection.

## Files

- JSON: `experiments/archive/frontier_consolidation/data/u256_visibility_profile/u256_visibility_profile.json`
- Source CSV: `experiments/archive/frontier_consolidation/data/u256_visibility_profile/recovered_suffix_by_source_legal16k.csv`
- Top recovered rows CSV: `experiments/archive/frontier_consolidation/data/u256_visibility_profile/top_recovered_suffix_rows_legal16k.csv`
- Update summary CSV: `experiments/archive/frontier_consolidation/data/u256_visibility_profile/update_mass_summary_legal16k.csv`
- Token-length bucket CSV: `experiments/archive/frontier_consolidation/data/u256_visibility_profile/row_token_length_buckets_legal16k.csv`
