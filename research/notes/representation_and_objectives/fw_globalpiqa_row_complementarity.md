# execution synthesis — FW 100M GlobalPIQA row complementarity

This file reads existing 100M all-option row CSVs. It asks whether the compact and row-block breadth arms are complementary row-wise or merely trade broad accuracy for hard-relation margins.

## Provenance

For both arms, parent `hf_model` files equal `hf_model/chck_100M` for `model.safetensors`, `tokenizer.json`, and `config.json`; the fw ewok interaction reader wrapper's parent+revision convention is therefore safe for 100M on these runs, though direct checkpoint paths remain safer for nonfinal checkpoints.

## Parallel (103 four-choice rows)

- compact accuracy: 24.27; breadth accuracy: 29.13; oracle union of exact correct sets: 34.95; correct-set Jaccard: 0.528
- compact-only correct: 6; breadth-only correct: 11; both correct: 19; both wrong: 67
- breadth better/same/worse rank rows: 34/50/19; breadth lower/higher margin rows: 49/35
- fw globalpiqa relevant substrate hard52: compact correct 2, breadth correct 3, compact-only 1, breadth-only 2, both wrong 48, breadth better rank 20, lower margin 30, mean margin delta -0.284 nats.

## Nonparallel (100 two-choice rows)

- compact accuracy: 53.00; breadth accuracy: 45.00; oracle union of exact correct sets: 60.00; correct-set Jaccard: 0.633
- compact-only correct: 15; breadth-only correct: 7; both correct: 38; both wrong: 40

## Scientific reading

The row sets are not identical: breadth repairs compact on some parallel rows and damages it on others. But the complementarity is not yet a usable mechanism because it is split across arms: breadth's parallel/hard-row gain comes with a larger nonparallel loss, while compact's nonparallel strength coexists with the hard52 deep-rank weakness.

This supports the execution synthesis route interpretation: do not extend FW allocation variants unless the pending interleaved arm uniquely combines compact-like broad capability with breadth-like hard-row margin movement. If not, the transition probe should be used only to study a stronger mechanism rather than scaled directly.

JSON: `experiments/archive/representation_and_objectives/data/fw_globalpiqa_row_complementarity/fw_globalpiqa_row_complementarity.json`
