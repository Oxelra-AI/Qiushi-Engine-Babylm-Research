# anchor matched controls — core transition filter and length-matched controls

## Purpose

fw ewok interaction reader found a large corpus-derived transition reservoir, but sampling showed metaphor/social/narrative noise. This CPU-only step filters it toward explicit relation markers plus concrete/action/spatial/temporal anchors and builds neutral controls matched by source and sentence-length bins. No official evaluation item text is read, and no training is launched.

## Main counts

- Input fw ewok interaction reader candidates: 22662 sentences / 747986 words.
- Core strict candidates: 5353 sentences / 190928 words.
- Core very-strict candidates: 1007 sentences / 41847 words.
- Neutral candidate pool for matching: 210960 sentences / 3427382 words.

## Balanced treatment slices and length-matched controls

See `core_slice_summary.csv` and `length_source_match_summary.csv` for exact matching measurements. The intended use is a future cheap treatment-vs-neutral probe only if FW compact/breadth endpoints fail to move the shared EWoK/GlobalPIQA relational weakness.

## Scientific reading

The filter creates a cleaner substrate than fw ewok interaction reader but still remains a research asset, not a training decision. It should not supersede the running FW compact-vs-breadth experiment. Before any GPU probe, sample the actual slice, check whether the neutral control is semantically neutral rather than merely marker-free, and choose the minimum reliable probe scale.

## Files

- summary JSON: `experiments/archive/representation_and_objectives/data/core_transition_filter/core_transition_filter_and_controls.json`
- core strict candidates: `experiments/archive/representation_and_objectives/data/core_transition_filter/core_transition_candidates.jsonl`
- core very-strict candidates: `experiments/archive/representation_and_objectives/data/core_transition_filter/core_transition_very_strict.jsonl`
- treatment slices: `experiments/archive/representation_and_objectives/data/core_transition_filter/core_transition_treatment_50k.jsonl`, `experiments/archive/representation_and_objectives/data/core_transition_filter/core_transition_treatment_100k.jsonl`, `experiments/archive/representation_and_objectives/data/core_transition_filter/core_transition_treatment_200k.jsonl`
- length-matched controls: `experiments/archive/representation_and_objectives/data/core_transition_filter/core_transition_neutral_lengthmatched_50k.jsonl`, `experiments/archive/representation_and_objectives/data/core_transition_filter/core_transition_neutral_lengthmatched_100k.jsonl`, `experiments/archive/representation_and_objectives/data/core_transition_filter/core_transition_neutral_lengthmatched_200k.jsonl`
- neutral pool: `experiments/archive/representation_and_objectives/data/core_transition_filter/neutral_candidate_pool_for_length_matching.jsonl`
- slice summary CSV: `experiments/archive/representation_and_objectives/data/core_transition_filter/core_slice_summary.csv`
- match summary CSV: `experiments/archive/representation_and_objectives/data/core_transition_filter/length_source_match_summary.csv`
