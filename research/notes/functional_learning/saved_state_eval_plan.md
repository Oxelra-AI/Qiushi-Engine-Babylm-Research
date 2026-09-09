# saved state eval plan saved-state replication and competence plan

## Why this step matters

The relation-first route has moved from a negative baseline to a positive acquired computation. The remaining scientific question is whether the operation is robust enough to matter for the BabyLM goal: can an identifiable coherent86-derived state acquire entity-conditioned source-value selection while retaining ordinary language competence?

deeper acquisition probe design was strong but transient: it retrained an in-memory model, then scored it. earlier analysis produced a better script to save exact trained checkpoints and add two missing behavioral distinctions:

1. **Neutral source retrieval versus correction-resistance.** If the parent already retrieves the correct source value in neutral contexts, answer-only training might mainly protect retrieval after an irrelevant update. If neutral retrieval is also absent in the parent and emerges after training, the intervention creates or exposes a more basic context-reading operation.
2. **Three-entity query-indexed selection.** In two-entity RETAIN rows, the queried entity is always the one not updated. A model could use a two-entity heuristic. The three-entity probe updates a third entity and switches the query among two unaffected entities, testing whether the learned rule selects by query identity among multiple unchanged source bindings.

## Actions started or prepared in saved state eval plan

- The first attempt used `saved_state_replicate_neutral_threeentity.py`; it failed before training because `main()` passed `parent_cache` where the tokenizer was expected, so `RowDataset` received `tokenizer=None`.
- Repaired the argument order at the call site. Syntax check passed.
- The repaired replication was restarted, writing to `data/saved_state_replicate_neutral_threeentity/` with construction seed 40040, training seeds 40040/40041/40042, 80 epochs, trusted private-adapter scale 0.75.
- Built `scripts/eval_saved_state_cheap7_entity.py`, a separate compatibility evaluator for a saved earlier analysis checkpoint. It records a trust-remote-code identity preamble and can run Entity alone or the full causal interface trajectory fast Cheap7 screen.
- Initial evaluator preflight failed trusted loading for the coherent86 parent because Transformers had already been imported before `HF_MODULES_CACHE` was redirected to a writable evaluation-scoped directory; this reproduced the earlier loader failure.
- Repaired the evaluator by importing `AutoConfig`/`AutoModelForMaskedLM` inside `checkpoint_identity()` after setting evaluation-scoped HF caches. Syntax check passed.
- Second preflight on the coherent86 parent succeeded: class `FrozenSlowPrivateDebertaV2ForMaskedLM`, private_adapter params `995584`; stock/plain loader has zero private adapters and counts ignored private keys, as expected.

## Reference values for interpreting the saved-state competence screen

The causal interface trajectory common-screen coherent86 alpha0.75 reference has:

- BLiMP 69.17
- Supplement 66.40
- EWoK 49.82
- Entity 27.78
- COMPS 52.05
- GlobalPIQA parallel/nonparallel 29.13 / 48.00, mean 38.565
- Reading 8.165
- equal_valid_mean 44.56428571428571

These are fast-screen research values, not official Overall. The first saved-state competence test should at minimum evaluate Entity; if the saved state is not badly damaged there, evaluate the full fast Cheap7 screen for the same checkpoint.

## Interpretation to preserve

A positive saved-state result must not be promoted directly to a universal principle. It would support a bounded principle: finite relation-first experience can install contextual source-value selection in coherent86 private adapters when answer-position credit is concentrated and relation evidence remains visible. The BabyLM-relevant question is whether this operation can coexist with the inherited ALN/ordinary language substrate under legal exposure and limited compute. Negative or mixed results should separate behavioral acquisition from general competence preservation instead of collapsing them into a single failure.
