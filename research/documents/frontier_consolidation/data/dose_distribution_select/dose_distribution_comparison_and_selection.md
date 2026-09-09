# matched max dose execution note dose distribution comparison and matched MAX selection

The inherited 1x block is preserved as the inner set. The increment is selected from unused medium accepted pairs plus the web-artifact-removed expansion so that source length, compression ratio, content density, and novel rewrite content stay close to the inherited block.

Old 1x: 12155 pairs, {'n': 12155, 'mean': 34.842533936651584, 'weighted_mean': 39.2564325365811, 'median': 33.0, 'q10': 20.0, 'q25': 25.0, 'q75': 42.0, 'q90': 52.0, 'min': 14.0, 'max': 86.0, 'sum': 423511.0} pair words.
All available MAX: 34917 pairs, {'n': 34917, 'mean': 33.11220895265917, 'weighted_mean': 37.759268244796004, 'median': 31.0, 'q10': 19.0, 'q25': 24.0, 'q75': 40.0, 'q90': 51.0, 'min': 14.0, 'max': 92.0, 'sum': 1156179.0} pair words, dose 2.730x.
Matched MAX: 33291 pairs, {'n': 33291, 'mean': 33.600282358595415, 'weighted_mean': 38.11081569873421, 'median': 31.0, 'q10': 19.0, 'q25': 24.0, 'q75': 40.0, 'q90': 51.0, 'min': 14.0, 'max': 92.0, 'sum': 1118587.0} pair words, dose 2.641x.
Matched increment origins: {"war_expansion_accepted": 490535, "medium_unused_accepted": 204541}.

## Mean shifts vs inherited 1x selected block (in old-block standard deviations)
### war_expansion_accepted
- source_words: -0.287 (old 21.5387, group 19.2965)
- rewrite_words: -0.231 (old 13.3038, group 12.1754)
- pair_words: -0.272 (old 34.8425, group 31.4719)
- length_ratio: +0.162 (old 0.6247, group 0.6406)
- pair_content_density: +0.252 (old 0.6024, group 0.6261)
- rewrite_content_density: +0.066 (old 0.6922, group 0.7007)
- novel_content_fraction: +0.167 (old 0.1609, group 0.1813)
### all_available_increment
- source_words: -0.214 (old 21.5387, group 19.8639)
- rewrite_words: -0.200 (old 13.3038, group 12.3244)
- pair_words: -0.214 (old 34.8425, group 32.1882)
- length_ratio: +0.077 (old 0.6247, group 0.6322)
- pair_content_density: +0.152 (old 0.6024, group 0.6167)
- rewrite_content_density: +0.008 (old 0.6922, group 0.6932)
- novel_content_fraction: +0.292 (old 0.1609, group 0.1966)
### matched_increment
- source_words: -0.161 (old 21.5387, group 20.2811)
- rewrite_words: -0.143 (old 13.3038, group 12.6048)
- pair_words: -0.158 (old 34.8425, group 32.8859)
- length_ratio: +0.083 (old 0.6247, group 0.6328)
- pair_content_density: +0.133 (old 0.6024, group 0.6149)
- rewrite_content_density: -0.017 (old 0.6922, group 0.6900)
- novel_content_fraction: +0.161 (old 0.1609, group 0.1806)
### matched_max_old_plus_increment
- source_words: -0.102 (old 21.5387, group 20.7403)
- rewrite_words: -0.091 (old 13.3038, group 12.8600)
- pair_words: -0.100 (old 34.8425, group 33.6003)
- length_ratio: +0.052 (old 0.6247, group 0.6298)
- pair_content_density: +0.085 (old 0.6024, group 0.6103)
- rewrite_content_density: -0.011 (old 0.6922, group 0.6908)
- novel_content_fraction: +0.103 (old 0.1609, group 0.1734)

JSON: `experiments/archive/frontier_consolidation/data/dose_distribution_select/dose_distribution_comparison_and_selection.json`
Selected MAX pairs: `experiments/archive/frontier_consolidation/data/dose_distribution_select/selected_matched_max_pairs.jsonl`
