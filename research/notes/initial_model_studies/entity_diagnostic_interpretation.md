# debertav2 entity diagnostics — Entity diagnostic interpretation

Evidence:

- Diagnostic JSON: `data/debertav2_entity_diagnostics.json`
- Diagnostic note: `notes/debertav2_entity_diagnostics.md`
- Official report: `training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/eval_results_available/hf_model/chck_100M/zero_shot/mlm/entity_tracking/entity_tracking/best_temperature_report.txt`

## Official score vs pooled diagnostic score

The diagnostic script exactly consumed all saved Entity predictions for the DeBERTa-v2 b256 model after applying the official skip rule that drops examples with a `nothing` option. It computed a pooled exact-option accuracy:

- pooled exact-option accuracy: **21.95%** = 1488 / 6780

The official evaluator reports **22.62%** because Entity Tracking is not pooled over all examples. `sentence_zero_shot/run.py` macro-averages operation-count subsets within each split and then averages the three split scores:

- regular macro over 0–5 ops: **22.40**
- ambiref macro over 0–5 ops: **23.49**
- move_contents macro over 0–5 ops: **21.97**
- official Entity = mean(regular, ambiref, move_contents) = **22.62**

Therefore the diagnostic is valid as a pooled/error-profile view, but the official coordinate should continue to use the evaluator’s 22.62 macro score.

## Mechanistic implications

The Entity weakness is broad rather than confined to one split:

- pooled split accuracies are close: regular 21.81, ambiref 22.46, move_contents 21.55.
- official split macros are also close: regular 22.40, ambiref 23.49, move_contents 21.97.
- operation-count behavior is not monotonic in difficulty under the pooled view: 0 ops 23.30, 1 op 16.33, 2 ops 19.23, 3 ops 22.74, 4 ops 26.47, 5 ops 28.23. This likely reflects subset/sample composition rather than clean reasoning depth.

The tokenizer comparison on actual Entity options found **no option-token-length difference** between baseline16k and official40k for these synthetic option strings:

- all option mean tokens: 6.474 under both tokenizers
- correct option mean tokens: 6.474 under both tokenizers
- 33,900 / 33,900 option strings have equal token length

Thus a 40k tokenizer is unlikely to improve Entity by shortening the synthetic answer options themselves. If 40k helps Entity, the mechanism would more likely be improved pretraining lexical/entity representation, different WWM grouping, shorter corpus token sequences, or training dynamics from the larger embedding matrix—not direct compression of the Entity evaluation option strings.

The broad Entity weakness supports later relation/entity-state mechanisms after the official40k run is evaluated: relation-preserving entity-chain masking or data that increases supervised exposure to identity binding and object-state updates, rather than a narrow answer-token segmentation fix.
