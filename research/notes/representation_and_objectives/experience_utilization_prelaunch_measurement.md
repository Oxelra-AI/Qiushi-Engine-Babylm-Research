# experience utilization prelaunch measurement — Experience-utilization prelaunch measurement

CPU-only analysis with no new training or model evaluation. Depth training and its dependent official-compatible evaluation remain pending.

## What U256 actually recovers from row256

- Legal40k 10M pool raw tokens: 13,942,644; row256 active tokens: 13,706,162; U256 recovered tokens: 236,482 (1.017254x active tokens).
- Fully hidden charged words recovered by U256: 128,664 / 10,000,000 (1.2866%); rows with any hidden word/token: 11,407.
- Boundary-hidden tokens inside the last visible word: 7,147; full hidden-word tokens: 229,335; rows with only boundary-token recovery: 277.

The visible change is therefore small in total mass but semantically concentrated in long-row tails; it should be interpreted as fixed-length suffix visibility plus the ordinary WWM target pressure on those now-visible suffixes.

## Per-update mass compared with row256

- Row256 baseline reconstruction matched the training log batch words with 0 mismatches over 2529 updates.
- Row256: 137,061,620 active tokens, 20,568,519 realized masked targets, 2529 updates, pad fraction 0.1730.
- U256@0.15: 139,426,440 active tokens, 20,916,090 realized masked targets, 2530 updates, pad fraction 0.2849.
- U64_128_256@0.15: 139,426,440 active tokens, 20,915,364 realized masked targets, 2530 updates, pad fraction 0.2201.
- U256 versus row256: active-token ratio 1.017254; masked-target ratio 1.016898; update-count difference +1.

U256 keeps maximum length 256 and stream order, but it changes the example object from rows to row-internal chunks, raises pad slots substantially, and uses one more stage-reset update than the continuous row baseline. These are small enough for a SOTA-facing endpoint test, but the target-matched U256 arm remains the attribution follow-up if U256@0.15 improves.

## Source and tail profile

- childes: hidden words 101,114 / source words 2,574,880 (3.927%); hidden tokens 180,968; suffix cues action=2600, causal=1021, physical=990, spatial=3202, names/numbers=23905.
- simple_wiki: hidden words 13,205 / source words 1,059,040 (1.247%); hidden tokens 23,234; suffix cues action=47, causal=133, physical=33, spatial=659, names/numbers=5047.
- gutenberg: hidden words 7,359 / source words 1,859,360 (0.396%); hidden tokens 19,155; suffix cues action=30, causal=31, physical=40, spatial=132, names/numbers=4961.
- open_subtitles: hidden words 6,811 / source words 1,829,440 (0.372%); hidden tokens 12,837; suffix cues action=180, causal=121, physical=39, spatial=231, names/numbers=3671.
- qwen_pair_packed: hidden words 152 / source words 1,656,800 (0.009%); hidden tokens 256; suffix cues action=1, causal=2, physical=0, spatial=7, names/numbers=33.
- cleanqwen_fineweb_compact_view_reinvest: hidden words 17 / source words 423,511 (0.004%); hidden tokens 24; suffix cues action=0, causal=0, physical=1, spatial=0, names/numbers=1.
- bnc_spoken: hidden words 6 / source words 575,840 (0.001%); hidden tokens 8; suffix cues action=0, causal=0, physical=0, spatial=0, names/numbers=0.
- neutral_cleanqwen_topup_compact_reinvest::open_subtitles: hidden words 0 / source words 9 (0.000%); hidden tokens 0; suffix cues action=0, causal=0, physical=0, spatial=0, names/numbers=0.

The recovered tail mass is not uniformly distributed across sources. The source table and top hidden rows should be read before choosing a data-mechanism interpretation for any U256 score movement.

## Consequence for the next launch

This measurement supports the independent_review reading: if depth is coherent and competitive, the first combined chunk-stream endpoint should be U256@0.15 on the 12x384 substrate; if depth is flat or worse, U256@0.15 on the 8x480 legal40k substrate is the cleaner fixed-length visibility comparison. U64_128_256 should follow a meaningful U256 signal, not precede it, because it adds context-length/order/packing changes on top of visibility repair.

JSON: `experiments/archive/representation_and_objectives/data/experience_utilization_prelaunch_measurement/experience_utilization_prelaunch_measurement.json`
CSV: `experiments/archive/representation_and_objectives/data/experience_utilization_prelaunch_measurement/update_mass_summary.csv`, `experiments/archive/representation_and_objectives/data/experience_utilization_prelaunch_measurement/recovered_suffix_by_source.csv`, `experiments/archive/representation_and_objectives/data/experience_utilization_prelaunch_measurement/row256_baseline_update_stats.csv`
