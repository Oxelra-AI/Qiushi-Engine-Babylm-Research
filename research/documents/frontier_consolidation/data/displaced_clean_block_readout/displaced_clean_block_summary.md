# earlier analysis displaced clean-block readout

CPU/file-only pool readout; no model loading, training, official evaluation, GPU work, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.

## Central result

The clean rows removed by the sub-dose ladder are not the protected `qwen_pair_packed` rows. They are ordinary heldout rows from CHILDES, Gutenberg, OpenSubtitles, Simple Wikipedia, BNC Spoken, and Switchboard. The protected Qwen paired block sits in the shared common filler and is identical across all arms.

## Cumulative displaced clean rows

| Dose | Rows | Words | rho(clean removed) | qwen_pair_packed words | Main source-word counts | Content types |
|---|---:|---:|---:|---:|---|---:|
| clean_rho0 | 0 | 0 | 0.000000 | 0 |  | 0 |
| quarter_1x | 758 | 105962 | 0.010596 | 0 | cleanqwen_lengthmatched_dose2p64x::childes:37133, cleanqwen_lengthmatched_dose2p64x::gutenberg:24029, cleanqwen_lengthmatched_dose2p64x::open_subtitles:22480, cleanqwen_lengthmatched_dose2p64x::simple_wiki:13672, cleanqwen_lengthmatched_dose2p64x::bnc_spoken:8648 | 11163 |
| half_1x | 1513 | 211853 | 0.021185 | 0 | cleanqwen_lengthmatched_dose2p64x::childes:70882, cleanqwen_lengthmatched_dose2p64x::open_subtitles:47645, cleanqwen_lengthmatched_dose2p64x::gutenberg:46943, cleanqwen_lengthmatched_dose2p64x::simple_wiki:28195, cleanqwen_lengthmatched_dose2p64x::bnc_spoken:17612, cleanqwen_lengthmatched_dose2p64x::switchboard:576 | 16808 |
| full_1x | 3005 | 423559 | 0.042356 | 0 | cleanqwen_lengthmatched_dose2p64x::childes:140323, cleanqwen_lengthmatched_dose2p64x::gutenberg:95801, cleanqwen_lengthmatched_dose2p64x::open_subtitles:95009, cleanqwen_lengthmatched_dose2p64x::simple_wiki:59596, cleanqwen_lengthmatched_dose2p64x::bnc_spoken:32147, cleanqwen_lengthmatched_dose2p64x::switchboard:683 | 24739 |
| max_dose2p64 | 7923 | 1118587 | 0.111859 | 0 | cleanqwen_lengthmatched_dose2p64x::childes:364916, cleanqwen_lengthmatched_dose2p64x::gutenberg:263557, cleanqwen_lengthmatched_dose2p64x::open_subtitles:256165, cleanqwen_lengthmatched_dose2p64x::simple_wiki:150554, cleanqwen_lengthmatched_dose2p64x::bnc_spoken:80511, cleanqwen_lengthmatched_dose2p64x::switchboard:2884 | 40996 |

## Incremental displaced segments

| Segment | Rows | Words | qwen_pair_packed words | Source-word counts |
|---|---:|---:|---:|---|
| quarter_only_rows_0_757 | 758 | 105962 | 0 | cleanqwen_lengthmatched_dose2p64x::childes:37133, cleanqwen_lengthmatched_dose2p64x::gutenberg:24029, cleanqwen_lengthmatched_dose2p64x::open_subtitles:22480, cleanqwen_lengthmatched_dose2p64x::simple_wiki:13672, cleanqwen_lengthmatched_dose2p64x::bnc_spoken:8648 |
| half_increment_rows_758_1512 | 755 | 105891 | 0 | cleanqwen_lengthmatched_dose2p64x::childes:33749, cleanqwen_lengthmatched_dose2p64x::open_subtitles:25165, cleanqwen_lengthmatched_dose2p64x::gutenberg:22914, cleanqwen_lengthmatched_dose2p64x::simple_wiki:14523, cleanqwen_lengthmatched_dose2p64x::bnc_spoken:8964, cleanqwen_lengthmatched_dose2p64x::switchboard:576 |
| full_increment_rows_1513_3004 | 1492 | 211706 | 0 | cleanqwen_lengthmatched_dose2p64x::childes:69441, cleanqwen_lengthmatched_dose2p64x::gutenberg:48858, cleanqwen_lengthmatched_dose2p64x::open_subtitles:47364, cleanqwen_lengthmatched_dose2p64x::simple_wiki:31401, cleanqwen_lengthmatched_dose2p64x::bnc_spoken:14535, cleanqwen_lengthmatched_dose2p64x::switchboard:107 |
| max_increment_rows_3005_7922 | 4918 | 695028 | 0 | cleanqwen_lengthmatched_dose2p64x::childes:224593, cleanqwen_lengthmatched_dose2p64x::gutenberg:167756, cleanqwen_lengthmatched_dose2p64x::open_subtitles:161156, cleanqwen_lengthmatched_dose2p64x::simple_wiki:90958, cleanqwen_lengthmatched_dose2p64x::bnc_spoken:48364, cleanqwen_lengthmatched_dose2p64x::switchboard:2201 |
| neutral_topup_row_7923 | 1 | 133 | 0 | cleanqwen_lengthmatched_dose2p64x::childes:133 |

## Interpretation for the sub-dose score curve

A plateau by rho≈0.011 would not mean that the removed block is Qwen-generated paired text; file evidence says the removed material is ordinary BabyLM heldout text. The contrast would instead read as replacing a small amount of ordinary source-balanced BabyLM text with authentic FineWeb source+view packets. Because the common qwen_pair_packed portion is identical in every arm, it cannot explain V≈R≈B≫C through direct displacement in this instrument.

JSON summary: `experiments/archive/frontier_consolidation/data/displaced_clean_block_readout/displaced_clean_block_summary.json`
Cumulative CSV: `experiments/archive/frontier_consolidation/data/displaced_clean_block_readout/cumulative_displaced_clean_rows.csv`
Incremental CSV: `experiments/archive/frontier_consolidation/data/displaced_clean_block_readout/incremental_displaced_clean_rows.csv`
