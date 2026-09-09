# fullctx aux budget audit — full-context auxiliary budget audit

The unchanged full context pivot substitution probe trainer reports only the base WWM stream words. Its auxiliary loss, however, forwards full row contexts for selected events. This audit reuses the same selection and donor-attachment code without loading a model.

- Base segment words: 9971308

- Auxiliary events: 11346; categories: {'temporal': 2000, 'comparative': 1346, 'physical_change': 2000, 'negation': 2000, 'causal_connector': 2000, 'spatial': 2000}

- Repaired two-view semantic objective (true+shuffle) would add 3527690 row-word passes, ratio 0.354 of base, total debit for the full base segment 13498998. Starting from 80,011,326, that would land at 93510324.

- Unchanged four-view implementation adds 7055380 row-word passes, ratio 0.708 of base, total debit for the full base segment 17026688. Starting from 80,011,326, that would land at 97038014.

- To spend only a 10M-word debit after 80M, observed ratios imply at most 7386702 base words for a two-view objective, or 5856281 base words for the unchanged four-view trainer.

Consequence: no future relation-auxiliary continuation should call its endpoint 90M by base words alone. The WWM reference and auxiliary branches must share the same debited exposure, and the trainer should compute only the views required by the chosen loss.

Files: `experiments/archive/representation_and_objectives/data/fullctx_aux_budget_audit/fullctx_aux_budget_audit.json`, `experiments/archive/representation_and_objectives/data/fullctx_aux_budget_audit/batch_budget_records.jsonl`

