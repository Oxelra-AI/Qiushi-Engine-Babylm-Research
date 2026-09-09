# earlier analysis v2 entity-orbit alias corpus

Input 10M: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
Output 10M: `experiments/archive/representation_and_objectives/data/entity_orbit_alias_corpus_v2/entity_orbit_alias_compact_reinvest_10M_pilot_2000.jsonl`
Rows/words: 2000 / 281178
Changed rows: 1247 (0.624)
Entity spans replaced: 5139; entity words per 10M words: 236967
Rows with repeated entity keys: 1247
Word-mismatch reverted rows: 0
Label counts: `{'LOC': 444, 'GPE': 1869, 'NORP': 927, 'PERSON': 1825, 'FAC': 74}`
Top rejections: `{'excluded_label:ORG': 3692, 'excluded_label:DATE': 2757, 'excluded_label:CARDINAL': 2417, 'singleton_identity_key': 1255, 'excluded_label:ORDINAL': 659, 'excluded_label:QUANTITY': 392, 'excluded_label:PERCENT': 277, 'excluded_label:TIME': 235, 'excluded_label:WORK_OF_ART': 177, 'not_title_like': 176, 'excluded_label:PRODUCT': 155, 'excluded_label:MONEY': 103}`

This construction preserves exact whitespace word counts and existing tokenizer lineage. It is intentionally high precision: replacement volume is lower than v1, but semantic corruption from false labels is reduced.
Manifest JSON: `experiments/archive/representation_and_objectives/data/entity_orbit_alias_corpus_v2/entity_orbit_alias_manifest_pilot_2000.json`
