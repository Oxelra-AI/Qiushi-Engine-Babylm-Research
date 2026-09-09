# semantic view contrast materialization — simpara source coverage and semantic-view interpretation

## Measurement
- Prompt file: `experiments/archive/representation_and_objectives/training/data/factual_prompts_shard1_simpara.jsonl`; rows=30000; sha256 `155a5f4f6044e51ded46de69d1a8e231ff62cf922c8e33bb157bc6f67e880525`.
- Type counts: {"paraphrase": 15000, "simplification": 15000}.
- Unique source texts: 14985 with 502,284 source words across 8990 SimpleWiki article labels.
- Source texts with both simplification and paraphrase prompts: 14985.
- Using generation slice quality slice length ratios, the maximum semantic-view pool if every generated row were accepted is about 1,423,975 words (14.24% of a 10M corpus).
- Domain row hits over unique source texts: {"geography_places": 2973, "people_history": 3811, "quant_numeric": 5147, "institutions_society": 2130, "causal_relational": 1427, "media_culture": 2229, "science_physical": 535}

## Scientific meaning
The pending simplification/paraphrase shard is not a broad new factual source: it is two generated linguistic views of the same 15k SimpleWiki source rows. That can still be scientifically useful, but as a semantic-view learning mechanism rather than as evidence that factual/entity/causal breadth alone closes the leader gap.

The next corpus comparison therefore needs two matched arms: a treatment that uses original+accepted simplification/paraphrase views, and an original-only contrast that allocates the same word exposure to the same SimpleWiki source articles with the same row-length sequence. This makes any score movement interpretable as transformation/view learning rather than mere extra passes over the selected articles.

## Top source articles by selected source rows
- Jeremy Corbyn: 23
- Alcohol withdrawal: 19
- Mike Pence: 19
- Hamis Kiggundu: 18
- Charles Reid: 17
- Laramie: 17
- History of the Catholic Church: 17
- Glosa (auxiliary language): 16
- Free Syrian Army: 15
- High Court of Justiciary: 15
- Rudolf Höss: 14
- Brihadeshwara Temple: 14
- Exposure (toxicology): 14
- Israel–Hamas war: 13
- Lipan Apache people: 13
- Novial: 12
- Elmdale, Kansas: 12
- Cranial nerve: 11
- Cabinet of prime minister Akhund: 11
- 1st Provisional Marine Brigade: 11

Full JSON: `experiments/archive/representation_and_objectives/training/data/semantic_view/simpara_prompt_source_coverage.json`
