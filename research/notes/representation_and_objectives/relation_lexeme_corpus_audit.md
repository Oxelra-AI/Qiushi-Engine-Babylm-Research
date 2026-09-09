# ewok tokenizer eval interface audit — relation lexeme corpus audit

JSON: `experiments/archive/representation_and_objectives/data/relation_lexeme_corpus_audit/relation_lexeme_corpus_audit_fast.json`

This is a surface text-composition audit for relation words highlighted by the official-compatible EWoK margin analysis. It is not a competence result.

## Corpus accounting
- clean_qwen_10M: rows=64381, words=10000000, top sources=[('childes', 16954), ('gutenberg', 12243), ('qwen_pair_packed', 12236), ('open_subtitles', 12045), ('simple_wiki', 6973)]
- compact_view_reinvest_10M: rows=64740, words=10000000, top sources=[('childes', 16093), ('qwen_pair_packed', 12236), ('gutenberg', 11621), ('open_subtitles', 11434), ('simple_wiki', 6619)]
- compact_repeat_reinvest_10M: rows=64740, words=10000000, top sources=[('childes', 16093), ('qwen_pair_packed', 12236), ('gutenberg', 11621), ('open_subtitles', 11434), ('simple_wiki', 6619)]
- lengthmatched_compact_reinvest_10M: rows=64740, words=10000000, top sources=[('childes', 16093), ('qwen_pair_packed', 12236), ('gutenberg', 11621), ('open_subtitles', 11434), ('simple_wiki', 6619)]

## Selected pair-pattern row counts
- above_below: clean=66, compact_view=67, compact_repeat=66, lengthmatched=64, delta_view-clean=1
- boss_subordinate: clean=0, compact_view=0, compact_repeat=0, lengthmatched=0, delta_view-clean=0
- parent_child: clean=158, compact_view=180, compact_repeat=180, lengthmatched=159, delta_view-clean=22
- teacher_student: clean=65, compact_view=84, compact_repeat=83, lengthmatched=66, delta_view-clean=19
- kick_drop_touch: clean=0, compact_view=0, compact_repeat=0, lengthmatched=0, delta_view-clean=0
- sink_float: clean=8, compact_view=8, compact_repeat=8, lengthmatched=8, delta_view-clean=0
- rise_fall: clean=202, compact_view=194, compact_repeat=193, lengthmatched=198, delta_view-clean=-8
- grow_shrink: clean=9, compact_view=9, compact_repeat=9, lengthmatched=9, delta_view-clean=0

## Selected lexeme counts
- above: clean=1253, compact_view=1331, compact_repeat=1343, lengthmatched=1253, delta_view-clean=78
- below: clean=617, compact_view=692, compact_repeat=691, lengthmatched=617, delta_view-clean=75
- boss: clean=382, compact_view=359, compact_repeat=358, lengthmatched=382, delta_view-clean=-23
- subordinate: clean=7, compact_view=6, compact_repeat=6, lengthmatched=7, delta_view-clean=-1
- parent: clean=116, compact_view=136, compact_repeat=134, lengthmatched=116, delta_view-clean=20
- child: clean=2185, compact_view=2328, compact_repeat=2308, lengthmatched=2185, delta_view-clean=143
- teacher: clean=882, compact_view=914, compact_repeat=905, lengthmatched=882, delta_view-clean=32
- student: clean=303, compact_view=416, compact_repeat=397, lengthmatched=303, delta_view-clean=113
- kick: clean=239, compact_view=226, compact_repeat=227, lengthmatched=239, delta_view-clean=-13
- drop: clean=771, compact_view=764, compact_repeat=759, lengthmatched=771, delta_view-clean=-7
- touch: clean=1086, compact_view=1054, compact_repeat=1052, lengthmatched=1086, delta_view-clean=-32
- break: clean=1435, compact_view=1404, compact_repeat=1402, lengthmatched=1435, delta_view-clean=-31
- stir: clean=149, compact_view=149, compact_repeat=148, lengthmatched=149, delta_view-clean=0
- sink: clean=230, compact_view=230, compact_repeat=231, lengthmatched=230, delta_view-clean=0
- float: clean=66, compact_view=54, compact_repeat=54, lengthmatched=66, delta_view-clean=-12

## Reading
The compact-view corpus is not grossly devoid of the main relation words, but surface counts are not a relation-learning mechanism. Compare these counts with official-margin behavior only as a clue: relation failures with many ordinary occurrences point toward relation-direction/composition and seed-dependent representation dynamics, not simple word absence.
