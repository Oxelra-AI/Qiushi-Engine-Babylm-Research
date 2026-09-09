# pvdm compliance and control design — PVDM compliance and deterministic control design

## Rule/provenance resolution

The corpus directional pair census spaCy dependency pilot is not used to create training labels. The official BabyLM FAQ says that ancillary language-learned models count toward the 100M word budget and gives the concrete example that an off-the-shelf POS tagger cannot be used in the pipeline. The installed `en_core_web_sm` metadata reports OntoNotes 5, ClearNLP conversion guidelines, and WordNet sources; it is therefore an external language-learned parser/tagger for our purpose.

## Replacement extractor

This construction uses only deterministic pattern rules plus word-frequency bins computed from the allowed compact 10M pool. There are no learned parser, tagger, NER, or external model outputs in the labels. The rule list is saved in `scripts/pvdm_deterministic_label_builder.py` and is inspectable submission-side code.

## Matched-control construction

For each accepted PVDM event, the target word index is identical in treatment and control. PVDM protects the true pivot from masking; the control protects a surrogate visible anchor chosen from the same row by deterministic class/frequency/distance scoring. Thus the dependent-target identity sequence is exactly shared by treatment and control, and the trainer will normalize masking probabilities so total masked mass is matched per batch.

## Tail label facts

- tail: 70,000,000→100,000,000 words from `experiments/archive/representation_and_objectives/data/fw_source_breadth_arm/fw_preserved_compact_view_100M.jsonl`
- rows: 192,549; words: 30,000,000
- eligible rows: 192,318 (0.999)
- accepted events: 1,527,594; unique target positions: 1,527,594
- target positions per 1000 words: 50.92
- target identity sequence hash shared by both arms: `54db584a61bc7923074051b5fe848dd7482c36761594db2acea869c43d2b0d96`
- label file SHA: `91acbdb597de9bc9f560cae0c54960b9a4ca5e472a0b65efbe8bd071b79869bc`

Full summary: `experiments/archive/representation_and_objectives/data/pvdm_deterministic_labels/pvdm_label_summary.json`
Samples: `experiments/archive/representation_and_objectives/data/pvdm_deterministic_labels/pvdm_tail_event_samples.jsonl`
