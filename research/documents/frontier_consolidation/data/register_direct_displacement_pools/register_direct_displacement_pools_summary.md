# earlier analysis direct register-displacement pools

CPU/file-only materialization. No generation, streaming, model loading, training, official evaluation, upload, or leaderboard action.

## Scientific contrast
Both arms admit the same quarter_1x FineWeb source+compact-view text multiset (758 rows, 105,962 words, rho=0.0105962). Each FineWeb row is placed onto a selected clean row of exactly the same word count. Unselected rows keep their original clean text at their original positions. Thus the intended changing variable is the register of clean material directly replaced.

## Arm summaries
### childsub_direct
10M: `experiments/archive/frontier_consolidation/data/register_direct_displacement_pools/regdisp_direct_childsub_direct_samefw_quarter_10M.jsonl`
100M: `experiments/archive/frontier_consolidation/data/register_direct_displacement_pools/regdisp_direct_childsub_direct_samefw_quarter_100M.jsonl`
SHA10: `492e8db89c81dfd44de28c4309edbd1245b81cd29db572574404364f46e5e3c3`
SHA100: `58370b995eafdc2ad0ccc2fdac5ff4069960d8120692c072ca191705df9e83e1`
Displaced rows/words: 758 / 105962
Displaced composition: child+subtitle 105867 (0.999103), Gutenberg+SimpleWiki 95 (0.000897), other 0
Displaced row-index span/mean: 13..5199 / 1269.1

### adult_direct
10M: `experiments/archive/frontier_consolidation/data/register_direct_displacement_pools/regdisp_direct_adult_direct_samefw_quarter_10M.jsonl`
100M: `experiments/archive/frontier_consolidation/data/register_direct_displacement_pools/regdisp_direct_adult_direct_samefw_quarter_100M.jsonl`
SHA10: `84d7e190bff9f8f23b2e1aef50a16d72a36253fafeb58f56683af7533a0352ca`
SHA100: `07631d8eab7a304a0cc1bf14747e0c1b12ab288cd105662ec962c5794b57c5ff`
Displaced rows/words: 758 / 105962
Displaced composition: child+subtitle 213 (0.002010), Gutenberg+SimpleWiki 105672 (0.997263), other 77
Displaced row-index span/mean: 0..7869 / 2580.2

## Relationship to fixed-position auxiliary pair
The earlier `register_displacement_pools/` pair holds FineWeb positions fixed and relocates clean rows to alter the omitted set. This direct pair should be the primary H100 discriminator because it directly replaces the selected clean register and does not relocate unselected clean rows.

## Interpretation use
Train only when the sub-dose/clean-anchor curve makes rho≈0.011 worth resolving. If childsub_direct and adult_direct land together against the same clean anchor, the effect is tied to admitting a second register rather than to the register removed. If they diverge, the sacrificed clean register is part of the data-efficient learning law at this scale.

Metadata: `experiments/archive/frontier_consolidation/data/register_direct_displacement_pools/register_direct_displacement_pools_metadata.json`
