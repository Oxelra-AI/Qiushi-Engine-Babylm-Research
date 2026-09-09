# anchor matched controls — ultra-clean transition probe assets v2

## Purpose

The first ultra-clean attempt showed that the strict pool is only about 38k words and the 50k exact selector failed. This repaired version creates a 30k-word treatment slice and a no-explicit-relation anchored control. It remains a future cheap probe asset, not a full route.

## Counts

- Ultra-clean candidate pool: 928 sentences / 37917 words.
- Treatment: 7 sentences / 200 words; bucket words {'physical_material_transition': 48, 'spatial_transition': 46, 'temporal_quantity_transition': 106}.
- Anchor control: 9 sentences / 232 words; word difference 32; source L1 0.0566; length-bin L1 0.1848; required-capability L1 0.2759.

## Scientific reading

This is the cleanest transition-substrate probe asset produced so far, but it is only 30k words. It can test whether explicit corpus-derived transition material has a detectable direction relative to anchored non-transition content after a pretrained shared checkpoint, but it cannot by itself establish a SOTA route. Because older INITIAL_MODEL_STUDIES BSM screens raised EWoK while hurting GlobalPIQA, any future use must include the GlobalPIQA margin reader and EWoK interaction reader.

## Files

- summary JSON: `experiments/archive/representation_and_objectives/data/ultraclean_transition_probe_v2/ultraclean_transition_probe_assets_v2.json`
- treatment: `experiments/archive/representation_and_objectives/data/ultraclean_transition_probe_v2/ultraclean_transition_30k.jsonl`
- anchor control: `experiments/archive/representation_and_objectives/data/ultraclean_transition_probe_v2/ultraclean_anchor_control_30k.jsonl`
- candidate pool: `experiments/archive/representation_and_objectives/data/ultraclean_transition_probe_v2/ultraclean_transition_candidate_pool_v2.jsonl`
- summary CSV: `experiments/archive/representation_and_objectives/data/ultraclean_transition_probe_v2/ultraclean_probe_v2_summary.csv`
