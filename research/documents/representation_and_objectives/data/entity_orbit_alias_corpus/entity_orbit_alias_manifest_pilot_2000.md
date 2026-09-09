# earlier analysis entity-orbit alias corpus

Input 10M: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
Output 10M: `experiments/archive/representation_and_objectives/data/entity_orbit_alias_corpus/entity_orbit_alias_pilot_2000.jsonl`
Rows/words: 2000 / 281178
Changed rows: 1788 (0.894)
Entity spans replaced: 10889; entity words replaced per 10M words: 619003
Word-mismatch reverted rows: 0
Label counts: `{'ORG': 3688, 'LOC': 634, 'GPE': 2246, 'NORP': 1146, 'PERSON': 2434, 'PRODUCT': 155, 'FAC': 174, 'EVENT': 94, 'WORK_OF_ART': 175, 'LANGUAGE': 87, 'LAW': 56}`

This construction preserves exact whitespace word counts and the existing tokenizer lineage; it deliberately changes entity identity statistics while keeping within-row coreference for exact mentions.
Manifest JSON: `experiments/archive/representation_and_objectives/data/entity_orbit_alias_corpus/entity_orbit_alias_manifest_pilot_2000.json`
