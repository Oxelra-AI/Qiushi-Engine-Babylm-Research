# Compact anchor match

Status: `A02_COMPACT_CAN_BE_USED_AS_FULLBATCH_ANCHOR`.

compact run uses the same compact 100M stream (`c8d7f24b...`), the same shared legal 16k tokenizer (`e70d167f...`), the same DeBERTa-v2 8x480 recipe, fixed WWM 0.15, AdamW settings, seeds 43/43022/43023, 100M exposure, 641,830 rows, and the same row order recorded by the exact stream path.

The only substantive execution difference from the cancelled companion analysis queued pair is the trainer coordinate: companion analysis uses the original COMPACT_EXPERIENCE full-batch trainer, while companion analysis had queued the accumulated memory-repair trainer. The two are intentionally close but not bit-identical, because dropout is executed over different forward-call shapes. Therefore companion analysis compact should not be used as the anchor for an companion analysis microbatch-coordinate comparison; instead the companion analysis interleaved breadth arm should be trained with the same COMPACT_EXPERIENCE full-batch trainer so the shared compact anchor is valid.

Current compact progress record: `{'step': 438, 'loss': 3.7484724521636963, 'lr': 0.0009636418201259489, 'batch_words': 40008, 'cumulative_word_exposure': 17470124, 'seq_len': 256, 'masked_tokens': 8561, 'effective_mask_rate': 0.1464, 'mask_mode': 'wwm', 'mask_prob_nominal': 0.15, 'elapsed_sec': 761.6}`.

Decision: keep the duplicate compact run cancelled; train only the interleaved whole-sentence breadth arm in the full-batch coordinate, then compare it to the compact model when the compact training result is present.

JSON: `experiments/archive/representation_and_objectives/data/compact_anchor_match/compact_anchor_match.json`
