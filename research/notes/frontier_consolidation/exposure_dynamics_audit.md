# exposure dynamics audit — exposure dynamics audit before pilot training

JSON: `experiments/archive/frontier_consolidation/data/exposure_audit/exposure_dynamics_audit.json`

This audit uses only the real clean-Qwen 10M training corpus and baseline16k tokenizer; it does not read any downstream evaluation items.

## Key measurements

| schedule | steps | padded/word | visible groups/word | visible token fraction | qwen visible groups/word | qwen visible token fraction | first long batch |
|---|---:|---:|---:|---:|---:|---:|---|
| old_A0_batch64_fixed256 | 1006 | 1.648154 | 0.977694 | 0.971848 | 0.999488 | 0.999296 |  |
| repaired_A0_batch256_fixed256 | 252 | 1.648154 | 0.977694 | 0.971848 | 0.999488 | 0.999296 |  |
| old_A1_A2_A3_batch256_seq64_to_256 | 252 | 0.778163 | 0.491334 | 0.483427 | 0.551711 | 0.545631 | reciprocal multiview mechanism and scaffold after 7040004 words |
| stagewise_pair_atomic | 687 | 2.245158 | 0.984189 | 0.980666 | 0.966402 | 0.962611 |  |
| stagewise_row_chunked | 711 | 2.323891 | 0.989684 | 0.986254 | 0.99957 | 0.999191 |  |

## Scientific reading

- The component gap analysis and leader reverse engineering baseline launch used batch64, which would create 1006 updates for a 10M pass; the repaired baseline batch256 uses 252 updates and matches the 2,515-step 100M clean-Qwen geometry (one tenth of the full run).
- The unchanged seq64→256 schedule with batch256 would count 10M words but expose only 0.491 visible word groups per counted word overall; for Qwen pair rows the ratio is 0.552. This is not the leader's inverse-batch low-truncation dynamics.
- The repaired pair-atomic stagewise candidate uses dynamic batches 512/256/128 and reaches 0.984 visible word groups per counted word overall, but it still preserves Qwen pairs as atomic rows; truncated qwen pair units by stage are {'stage1_seq64': 3945, 'stage2_seq128': 117, 'stage3_seq256': 0}. A full data-mechanism interpretation therefore still needs pair-length-aware construction, not just this training control.
- Loss values from these schedules are not comparable across masking modes; the next decision must use matched fast official-task outcomes across all trained arms.
