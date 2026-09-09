# Semantic-View Training and Source-Screen Criteria

Status: semantic-view training completed; matched downstream interpretation and the proposed FineWeb rewrite study were unresolved in this record.

The semantic-view treatment used DeBERTa-v2 with 8 layers, hidden size 480, a 16k tokenizer and 34,467,424 parameters. Training consumed exactly 100,000,000 words in 2,665 updates and saved 100 checkpoints. First and last losses were 9.802993774414062 and 2.5477638244628906. The semantic-view source contributed 8,201,870 words across ten passes. These are training observations, not BabyLM scores.

The next data question was whether factual source sentences paired with faithful simplifications outperform broader factual exposure alone. An initial complete-sentence preparation still admitted dialogue, abbreviation fragments and web lists, and was superseded. A stricter prepared subset had 13,198 sources, 309,191 source words and 4,484 documents; its expected paired mass was about 587,463 words at a rewrite/source ratio of 0.90.

The broader screening measurement distinguished these tiers:

| Tier | Rows | Words | Documents |
|---|---:|---:|---:|
| Broad clean rows | 59,769 | 1,246,944 | 5,419 |
| Complete nonfragment | 40,770 | 861,862 | 5,357 |
| Web artifacts removed | 39,550 | 819,521 | 5,343 |
| Balanced source-by-rewrite | 32,836 | 703,975 | 5,283 |
| Strict factual/expository | 14,385 | 332,669 | 4,566 |

The prepared strict subset and the measured strict tier are different selections and must not be conflated. The balanced tier offered more coverage but retained some low-relation material. The proposed next test was a small faithfulness/acceptance study followed, only if warranted, by source-plus-rewrite versus same-source repetition. Neither a raw-source fallback nor a full generation run was justified by training loss alone.
