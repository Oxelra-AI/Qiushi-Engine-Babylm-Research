# fineweb slot pairing feasibility live FineWeb view pilot source repair

This is a narrow prompt-readiness repair of the prepared source+view pilot, not a new broad selector and not a generation/training run. It removes rows that are likely to make a generator hallucinate or copy because of corrupt names, epistemic/future wording, title/report metadata, discourse-open starts, unresolved references, or high numeric load.

Input slot manifest: 5,985 pairs. Usable for view prompts: 4,625 pairs (77.28%).

Repaired prompt slice: 256 sources / 512 prompts; source words 7,090; simulated view-slot words 6,233; packet words 13,323.

Focus counts in repaired slice: {"causal_temporal_process": 96, "physical_spatial_object": 64, "social_entity_state": 40, "other_expository_relation": 40, "definition_taxonomic_fact": 16}

Top rejected reasons: [["length_outside_10_38_for_generation", 568], ["epistemic_future_or_time_relative", 465], ["titlelike_colon_fragment", 167], ["discourse_dependent_start", 155], ["compressed_catalogue_or_clause_stack", 51], ["media_procedure_or_web_residue", 41], ["report_title_citation_or_metadata", 35], ["unresolved_reference_risk", 33], ["mojibake_or_corrupt_name", 5]]

Use this repaired slice, not the unrepaired prompt file, if a later step runs the compact live FineWeb view generation. Actual accepted view lengths and substantive-change rates still decide whether any training corpus is materialized.

Summary JSON: `experiments/archive/representation_and_objectives/training/data/live_fineweb_view_pilot_repaired/live_fineweb_view_pilot_source_repair_summary.json`

Prompts: `experiments/archive/representation_and_objectives/training/data/live_fineweb_view_pilot_repaired/live_fineweb_view_pilot_repaired_prompts_512.jsonl`
