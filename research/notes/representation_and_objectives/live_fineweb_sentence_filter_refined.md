# refined live FineWeb sentence filter

Manual inspection of the first live high-anchor sample found false anchors from generic sentence-initial capitals (`In`, `If`, `For`, `By`) and web-instruction sentences. This offline filter removes those without another network pass.

Input basic candidates: 910 rows / 21,374 words.

Strict-anchor retained: 302 rows / 7,612 words (35.61% of candidate words).

Relation-dense retained: 147 rows / 3,570 words (16.70% of candidate words).

Flag counts after repair: `{'weak_relation_structure': 252, 'unbalanced_or_fragmentary_quote': 40, 'web_instruction_or_advice': 31, 'pronoun_instruction_heavy': 29, 'event_listing_or_contact_fragment': 6, 'byline_or_metadata': 2}`.

Scientific implication: live FineWeb can provide large volume, but naive capitalized-phrase heuristics are unsafe. A scaled source+view arm should use relation-dense filters, row provenance, and a small Qwen faithfulness pilot before generation at scale.

JSON: `experiments/archive/representation_and_objectives/data/live_fineweb_sentence_filter_refined/live_fineweb_sentence_filter_refined_summary.json`
