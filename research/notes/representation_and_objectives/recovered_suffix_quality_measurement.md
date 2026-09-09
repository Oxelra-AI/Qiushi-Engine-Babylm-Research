# experience utilization prelaunch measurement — Recovered suffix quality/content measurement

CPU-only. No data were selected or changed, and no model was trained or evaluated.

U256 would expose 128,664 fully hidden suffix words and 236,482 suffix/boundary tokens per 10M corpus pass. These are small in corpus mass, but not neutral in content.

## Source concentration
- childes: 101,114 suffix words (78.59%), 180,968 hidden tokens (76.53%), non-ASCII fraction 0.000.
- simple_wiki: 13,205 suffix words (10.26%), 23,234 hidden tokens (9.82%), non-ASCII fraction 0.004.
- gutenberg: 7,359 suffix words (5.72%), 19,155 hidden tokens (8.10%), non-ASCII fraction 0.004.
- open_subtitles: 6,811 suffix words (5.29%), 12,837 hidden tokens (5.43%), non-ASCII fraction 0.010.
- qwen_pair_packed: 152 suffix words (0.12%), 256 hidden tokens (0.11%), non-ASCII fraction 0.012.
- cleanqwen_fineweb_compact_view_reinvest: 17 suffix words (0.01%), 24 hidden tokens (0.01%), non-ASCII fraction 0.000.
- bnc_spoken: 6 suffix words (0.00%), 8 hidden tokens (0.00%), non-ASCII fraction 0.000.

## Content/noise categories
- dialogue_or_transcript_tail: 95,336 words (74.10%), 173,113 tokens (73.20%), rows 7,602.
- name_heavy_tail: 53,262 words (41.40%), 105,306 tokens (44.53%), rows 4,300.
- relation_action_social_tail: 30,208 words (23.48%), 53,182 tokens (22.49%), rows 1,470.
- ordinary_tail: 11,397 words (8.86%), 17,190 tokens (7.27%), rows 2,454.
- index_or_catalog_tail: 3,839 words (2.98%), 11,956 tokens (5.06%), rows 92.
- encoding_noise: 2,533 words (1.97%), 6,710 tokens (2.84%), rows 67.
- number_heavy_tail: 2,101 words (1.63%), 5,449 tokens (2.30%), rows 113.

## Scientific reading
U256 is still the clean first chunk-stream experiment because it changes visibility at fixed maximum length 256. But it is not simply adding uniformly good experience: most hidden full words are Childes tails, and a measurable subset is encoding-heavy or index/catalog-like text from OpenSubtitles/Gutenberg. If U256 helps, the result is stronger because it survived this heterogeneous tail; if it hurts, this profile gives a concrete explanation and points toward visibility-aware data repair rather than repeating the same chunking route.

JSON: `experiments/archive/representation_and_objectives/data/recovered_suffix_quality_measurement/recovered_suffix_quality_measurement.json`
CSV: `experiments/archive/representation_and_objectives/data/recovered_suffix_quality_measurement/suffix_quality_by_source.csv`, `experiments/archive/representation_and_objectives/data/recovered_suffix_quality_measurement/suffix_quality_categories.csv`, `experiments/archive/representation_and_objectives/data/recovered_suffix_quality_measurement/suffix_quality_samples.csv`
