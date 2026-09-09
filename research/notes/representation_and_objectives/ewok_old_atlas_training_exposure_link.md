# EWoK old-atlas link to compliant reinvest training exposure

Purpose: use the delivered full old EWoK atlas to test whether the seed-specific old compact-view EWoK effect is mostly ordinary surface exposure of EWoK concept words in the allowed 10M reinvest corpus or changed block. This is interpretation only, not training data construction.

Corpus: 64740 rows / 10000000 words; changed block identified by 3006 example ids / 423520 words.
Unique EWoK concept keys counted: 167 across 7618 official EWoK rows.

## Exposure summaries by row class
- All rows: both concepts seen in full 10M for 76.8%; both seen in changed block for 55.9%; min-occ median 53.0.
- Negative old treatment-by-seed rows: n=1766, both seen full 84.8%, both seen changed 60.9%, min-occ mean 419.85.
- Stable all-correct rows: n=1703, both seen full 67.3%, min-occ mean 435.89.

## Correlations with old treatment-by-seed interaction
- log1p(min concept occurrences in full 10M) vs accuracy interaction: r=-0.0171.
- log1p(sum concept occurrences in full 10M) vs accuracy interaction: r=-0.0168.
- log1p(min changed-block occurrences) vs accuracy interaction: r=-0.0107.
- log1p(sum changed-block occurrences) vs accuracy interaction: r=-0.0075.
- log1p(min full occurrences) vs margin interaction: r=-0.0062.

## Domain exposure and interaction
- Most negative domains: material-dynamics int=-0.175 bothSeen=100.0%; physical-dynamics int=-0.108 bothSeen=100.0%; spatial-relations int=-0.084 bothSeen=100.0%; physical-interactions int=-0.059 bothSeen=85.6%; social-relations int=-0.021 bothSeen=100.0%; quantitative-properties int=-0.010 bothSeen=100.0%
- Most positive domains: material-properties int=+0.082 bothSeen=72.9%; social-interactions int=+0.065 bothSeen=100.0%; social-properties int=+0.027 bothSeen=100.0%; agent-properties int=+0.017 bothSeen=27.6%; physical-relations int=-0.005 bothSeen=95.1%; quantitative-properties int=-0.010 bothSeen=100.0%

## Reusable outputs
- JSON: `experiments/archive/representation_and_objectives/data/ewok_old_atlas_training_exposure_link/ewok_old_atlas_training_exposure_link.json`
- High-exposure negative rows: `experiments/archive/representation_and_objectives/data/ewok_old_atlas_training_exposure_link/negative_interaction_high_exposure_examples.csv`
- Zero-pair-exposure negative rows: `experiments/archive/representation_and_objectives/data/ewok_old_atlas_training_exposure_link/negative_interaction_zero_pair_exposure_examples.csv`

Scientific reading: if the correlations above are small, the next repair should not be naive evaluation-word counting. The compact-view effect likely depends on relational framing, target geometry, or optimization dynamics; any data repair should be general and learning-scale, using independent source selection and neutral controls rather than benchmark-lexeme injection.
