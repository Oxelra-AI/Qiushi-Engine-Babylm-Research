# earlier analysis register-contrast displacement pools

CPU/file-only materialization. No generation, streaming, model loading, training, official evaluation, upload, or leaderboard action.

## Scientific contrast
Both arms admit exactly the same quarter_1x FineWeb source+compact-view rows at the same row positions (758 rows, 105,962 words, rho=0.0105962). They preserve the MAX-row 65,313-row / 10M-word geometry and differ in which clean rows are absent.

## Arm summaries
### childsub_removed
10M: `experiments/archive/frontier_consolidation/data/register_displacement_pools/regdisp_childsub_removed_samefw_quarter_10M.jsonl`
100M: `experiments/archive/frontier_consolidation/data/register_displacement_pools/regdisp_childsub_removed_samefw_quarter_100M.jsonl`
SHA10: `db7b7043f4db2d1a888df80f79c521335b4c00fdff5ba851940ab55bd953048e`
SHA100: `a459ca12107a465de464adb1a442aacfe50baefaf5f4b4d4d623c3f59caf957c`
Displaced rows/words: 758 / 105962
Displaced composition: child+subtitle 105867 (0.999103), Gutenberg+SimpleWiki 95 (0.000897), other 0
Overlap with original quarter active clean rows: 262; necessary relocated clean rows: 496 (69444 words)

### adult_removed
10M: `experiments/archive/frontier_consolidation/data/register_displacement_pools/regdisp_adult_removed_samefw_quarter_10M.jsonl`
100M: `experiments/archive/frontier_consolidation/data/register_displacement_pools/regdisp_adult_removed_samefw_quarter_100M.jsonl`
SHA10: `f3dcdc0999169414231321a94daab948e8bd63b9f6770d4031be3686608eb14b`
SHA100: `17e1c6a9c5f4fd530c581176448c84aed578c661a74487c050ed5dcd9cf0288a`
Displaced rows/words: 758 / 105962
Displaced composition: child+subtitle 213 (0.002010), Gutenberg+SimpleWiki 105672 (0.997263), other 77
Overlap with original quarter active clean rows: 126; necessary relocated clean rows: 632 (88578 words)

## Interpretation use
Train these two arms only after the current sub-dose/clean-anchor curve indicates that rho≈0.011 is a meaningful region. If their stable-family V-C curves land together, the evidence supports an admission-of-second-register mixture effect rather than value-of-removed-register; if they diverge, the clean material removed is itself part of the principle.

Metadata: `experiments/archive/frontier_consolidation/data/register_displacement_pools/register_displacement_pools_metadata.json`
