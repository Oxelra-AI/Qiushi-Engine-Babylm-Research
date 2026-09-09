# causal gpt relation boundary model tiedness probe

This records whether representative BabyLM coordinates tie input embeddings to the output readout matrix, for comparison with functional_learning's tied-softmax mechanism note.

| label | model class | tie_word_embeddings | same tensor | input shape | output shape |
|---|---|---:|---:|---|---|
| deberta_representation_frontier_studies_clean | DebertaV2ForMaskedLM | True | True | [16384, 480] | [16384, 480] |
| deberta_compact_experience_qwen_aligned | DebertaV2ForMaskedLM | True | True | [16384, 480] | [16384, 480] |
| roberta_functional_relation_studies_repeat_split | RobertaForMaskedLM | True | True | [16384, 480] | [16384, 480] |
| causal_gpt_repeat | GPT2LMHeadModel | True | True | [16384, 480] | [16384, 480] |

JSON: `experiments/archive/relation_learning/data/model_tiedness_probe/model_tiedness.json`.
