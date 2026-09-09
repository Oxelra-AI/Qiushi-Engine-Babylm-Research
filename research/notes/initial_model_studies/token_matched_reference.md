# token matched reference — Token-level supervision matching for the BSM reference

## Confound addressed

matched reference repair matched row count, word-length sequence, and update count, but the matched reference still had a different number of loss-bearing target tokens per targeted row/batch (reference 45–64 vs BSM 36–50 in the earlier smoke). Different supervision strength on targeted rows means BSM-minus-reference could still reflect target-token count, not binding content.

## Repair

`scripts/materialize_token_matched_official_reference.py` builds an official-only reference from a BSM schedule corpus and, for each targeted row, chooses an official target span whose tokenizer/offset-overlap target-token count under the trainer's rule (`train_legal_bsm_screen.py`, seq_length 128) exactly equals the corresponding BSM binding row's target-token count. It records `schedule_target_token_count` and `reference_target_token_count` per row.

Static materialization result on the 100k coherent schedule:
- same_row_count: true
- same_word_sequence: true
- same_target_token_count_sequence: true
- target_match_failures: 0
- unshuffled per-batch targeted tokens: schedule [35,52], reference [35,52]

## Batch-level verification through the real trainer path

`scripts/verify_token_matched_batches.py` loads both corpora through the trainer's `BSMDataset`/`collate_fn`, resets RNG with seed 42, and iterates the shuffled DataLoader. Result (`data/token_matched_reference/batch_match_100k.json`):

Coherent BSM vs token-matched reference, all true:
- same_rows, same_words, same_word_lengths
- same_training_modes
- same_per_row_target_tokens
- same_batch_target_tokens
- same_batch_targeted_rows
- same_batch_words

So under the actual training order, the two arms present identical row structure, identical shuffled batch composition, and identical loss-bearing target-token counts per batch. The only remaining difference is the content of targeted rows: entity-value binding text vs official text.

## Consequence for the 4M trace

Interpretation rules for the developmental trace are now cleaner:

- coherent − swapped: relation-consistency effect under fully matched structure and supervision (both are BSM corpora built from the same master events; they already share word/target structure by construction).
- BSM arms − token-matched reference: content effect of targeted binding vs targeted official text, with row/update/target-token supervision matched.
- coherent − official_standard remains structure+content confounded and must not be used as the relation or content effect.

For the 4M run, regenerate BOTH the swapped corpus and the token-matched reference from the actual 4M coherent schedule, then re-run `verify_token_matched_batches.py` on all arms before training, requiring all comparison flags true (for the reference) and matched structure for coherent/swapped.

## Note on swapped-vs-coherent target tokens

The paired materializer (earlier analysis) builds coherent and swapped from the same master events with the same templates and answer words, so their per-row target-token counts already match by construction. The 4M pre-flight verification should still confirm this with the same batch verifier to guarantee identical shuffled batch target-token sequences across coherent, swapped, and the token-matched reference.
