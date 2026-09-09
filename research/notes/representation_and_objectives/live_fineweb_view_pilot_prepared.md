# fineweb slot pairing feasibility live FineWeb faithful-view pilot preparation

This CPU-only file set prepares a possible source+view test without launching generation or training. It uses the fineweb source benchmark motif overlap live-FineWeb seed and the exact same-focus ratio-0.88 slot contract from the fineweb slot pairing feasibility pairing study.

Slot manifest: 5,985 source/distinct pairs; simulated source+view packet words 288,454; source words 153,533; simulated view words 134,921.

Prompt slice: 256 unique sources and 512 prompts across variants faithful_simplify, relation_preserving_restate. Focus counts: {'causal_temporal_process': 96, 'physical_spatial_object': 64, 'social_entity_state': 40, 'other_expository_relation': 40, 'definition_taxonomic_fact': 16}.

If generation is later run, actual accepted view lengths must be measured and the distinct-source slot should be rebuilt to those actual lengths before any matched training family.

Summary JSON: `experiments/archive/representation_and_objectives/training/data/live_fineweb_view_pilot/live_fineweb_view_pilot_preparation_summary.json`

Prompts: `experiments/archive/representation_and_objectives/training/data/live_fineweb_view_pilot/live_fineweb_view_pilot_prompts_512.jsonl`

Manifest: `experiments/archive/representation_and_objectives/training/data/live_fineweb_view_pilot/live_fineweb_slot_manifest_ratio0p88_samefocus_exact.jsonl`
