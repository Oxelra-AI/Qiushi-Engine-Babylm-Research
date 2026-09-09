# earlier analysis v2 entity-orbit alias corpus

Input 10M: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
Output 10M: `experiments/archive/representation_and_objectives/data/entity_orbit_alias_corpus_v2/entity_orbit_alias_compact_reinvest_10M.jsonl`
Rows/words: 64740 / 10000000
Changed rows: 34968 (0.540)
Entity spans replaced: 155835; entity words per 10M words: 195719
Rows with repeated entity keys: 34976
Word-mismatch reverted rows: 0
Label counts: `{'LOC': 3673, 'GPE': 37882, 'NORP': 14477, 'PERSON': 98226, 'FAC': 1577}`
Top rejections: `{'singleton_identity_key': 136346, 'excluded_label:ORG': 99451, 'excluded_label:DATE': 81319, 'excluded_label:CARDINAL': 72287, 'excluded_label:ORDINAL': 17038, 'excluded_label:TIME': 15867, 'excluded_label:WORK_OF_ART': 10820, 'not_title_like': 9220, 'excluded_label:QUANTITY': 4402, 'excluded_label:PRODUCT': 4253, 'excluded_label:EVENT': 3792, 'excluded_label:MONEY': 3269}`

This construction preserves exact whitespace word counts and existing tokenizer lineage. It is intentionally high precision: replacement volume is lower than v1, but semantic corruption from false labels is reduced.
Manifest JSON: `experiments/archive/representation_and_objectives/data/entity_orbit_alias_corpus_v2/entity_orbit_alias_manifest.json`
