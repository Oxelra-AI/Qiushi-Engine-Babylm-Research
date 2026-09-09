# role switch from scratch screen plan — legal role-switch from-scratch natural-transfer screen

## Scientific purpose

role switch packet screen synthesis already answered the synthetic-learning question. Balanced role exchange is learnable and transfers inside the hand-written grammar, including the held-out container family, but the exchange-specific movement on natural BabyLM hard surfaces was essentially null: GlobalPIQA hard-52 ranks stayed flat and EWoK treatment-specific stable-failure movement was only 15 rows over the role-fixed control.

The next screen therefore tests only the remaining legal/natural-transfer question: if the same packet/control text is embedded inside the <=10M corpus from the beginning, with legal tokenizers trained on the revised corpora, does the role-switch arm move the natural hard surfaces beyond the structurally matched role-fixed arm while preserving broad cheap7?

## Data object selected for the screen

Use the strict in-place packed corpora, not the first role switch packet screen synthesis short-row corpora and not the role switch from scratch screen plan appended packed corpora:

- Treatment 10M: `experiments/archive/representation_and_objectives/data/inplace_packed_role_switch_replacement_corpus/role_switch_inplace_packed_replacement_10M.jsonl`
- Control 10M: `experiments/archive/representation_and_objectives/data/inplace_packed_role_switch_replacement_corpus/role_fixed_inplace_packed_replacement_10M.jsonl`
- Manifest: `experiments/archive/representation_and_objectives/data/inplace_packed_role_switch_replacement_corpus/manifest.json`
- Tokenizer summary: `experiments/archive/representation_and_objectives/data/inplace_replacement_tokenizers/inplace_replacement_tokenizer_summary.json`

These corpora remove exactly the same 138 intact 160-word OpenSubtitles rows (22,080 words) from the source compact-view 10M pool and replace those same row positions (base row indices 3012..3710) with 138 packed 160-word packet rows. They preserve:

- exact 10,000,000 words per corpus;
- original 64,740-row count and therefore optimizer-batch geometry;
- matched treatment/control row positions and packet word positions;
- zero official-evaluation 7/8/10-gram overlap inherited from the packet suite;
- legal per-arm tokenizers trained only on each revised 10M corpus.

Tokenizer audit: treatment/control tokenizers have the same 40,000 token set and nearly identical surfaces (`tokens_per_word=1.3937251`, 11,404 rows over 256; packet rows have mean about 183.48 tokens and never exceed 256). They differ in 553 common-token IDs/merge order because BPE tie-breaking is stream-sensitive, so read the screen as a legal per-arm-tokenizer comparison; the text multiset and surface geometry are matched.

## Why 80M is the minimum reliable H100 screen

A 20M from-scratch screen would see only 44,160 packet words (two passes) and could fail to reproduce the role switch packet screen synthesis synthetic packet learning for lack of packet exposure. role switch packet screen synthesis's disposable learner used 8 passes / 176,640 packet-word exposure. Training to 80M reaches that same packet exposure while still avoiding two full 100M retrains. The 80M runs also retain 10M checkpoints so early severe broad damage can be read without needing a 100M endpoint.

The LR schedule remains the original 100M horizon (`lr_total_steps=2529`, warmup fraction 0.06), so the 80M checkpoints correspond to the same learning-rate regime as the legal40k accum training completion anchor rather than an artificially compressed 80M schedule.

## Predeclared interpretation

Route B continues only if the role-switch arm, relative to the role-fixed arm, shows natural-transfer movement of the kind role switch packet screen synthesis did not show:

1. GlobalPIQA_parallel: improvement on the fw globalpiqa relevant substrate hard-52 subset must be visible as rank/margin movement, not only one or two noisy flips. Useful signs are more correct hard-52 rows, fewer rank-4 correct options, and lower mean top-minus-correct margin.
2. EWoK: stable conditional-reversal failures must fall meaningfully beyond the role-fixed control, not merely by the shared effect of seeing packet-shaped text. The role switch packet screen synthesis exchange-specific reduction of 15 rows is too small.
3. Broad cheap7 must be preserved. A role-switch hard-surface movement that damages Supplement/Entity/Reading or cheap7 substantially repeats old INITIAL_MODEL_STUDIES synthetic-binding behavior and should not be scaled.
4. Synthetic packet transfer is not a success condition anymore. If from-scratch training shows large synthetic/held-out gains but leaves the natural hard surfaces flat above role-fixed, close Route B.

## Training commands

Both arms use DeBERTa-v2 8x480 legal40k, WWM fixed 0.15, AdamW LR 0.001, batch 256 via 64-row microbatches, seed 43 / init 43022 / train RNG 43023, seq256, 80M word exposure, 10M checkpoints, and 100M LR horizon.

Treatment output:
`experiments/archive/representation_and_objectives/training/runs/role_switch_inplace_packed_80M_legal40k_seed43022`

Control output:
`experiments/archive/representation_and_objectives/training/runs/role_fixed_inplace_packed_80M_legal40k_seed43022`

## Planned readout after completion

At minimum evaluate `chck_80M` for both arms with the official-compatible cheap columns (BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA_parallel, GlobalPIQA_nonparallel, Reading), then run the GlobalPIQA hard-margin reader and EWoK four-cell stable-reversal reader. If 20M/50M checkpoints are available before 80M completion in a later step, they can be used only to stop a clearly damaging route; they should not be used to declare natural transfer absent before equivalent packet exposure is reached.


## earlier analysis update — tokenizer confound removed; shared-tokenizer relaunch

The role switch from scratch screen plan per-arm tokenizers differed in 553 common-token IDs/merge ranks (BPE tie-breaking is stream-order sensitive). Given the large tokenizer sensitivity measured in these studies and role switch packet screen synthesis's tiny natural-surface signal, that difference could dominate the effect. The confounded screens were therefore stopped at only ~1.94M words, with no usable checkpoint, and the tokenizer mismatch was removed.

### Shared tokenizer on the exact intersection
- Script: `scripts/shared_intersection_tokenizer.py`
- The two in-place packed 10M corpora differ only in 138 packet rows / 22,080 words at base row indices 3012..3710. The common intersection is exactly 9,977,920 words / 64,602 rows.
- One legal 40k byte-BPE tokenizer trained ONLY on that intersection: `data/shared_intersection_tokenizer/tokenizers/shared_intersection_legal_byte_bpe_40k`. Special IDs 0/1/2/3/4; `tokens_per_word` ~1.39375; packet rows never exceed 256 tokens.
- Audit (`shared_intersection_tokenizer_summary.json`): under the shared tokenizer the common rows tokenize identically across arms (identical `common_row_token_hash`), the first 256-row prefix is identical, and only the 138 packet rows differ (35 packet rows differ in token count, as intended by the role text).

### Bitwise apparatus alignment on the pre-packet first batch
- Script: `scripts/shared_tokenizer_one_step_alignment.py`; summary `data/shared_tokenizer_one_step_alignment/shared_tokenizer_one_step_alignment_summary.json`.
- First effective batch = 256 rows / 35,716 words, all pre-packet (first packet row is 3012). Across arms: input IDs, word groups, attention masks, masked inputs, labels, initial model weights, gradients, grad norm, loss (10.666473), scheduler LR after step, and post-update weights are all identical. `all_alignment_keys_identical=true`.

### Relaunched paired 80M screen (single shared tokenizer)
- `s147_t30_tool1` role_switch, GPU1, output `training/runs/role_switch_sharedtok_80M_legal40k_seed43022`.
- `s147_t31_tool1` role_fixed, GPU1, output `training/runs/role_fixed_sharedtok_80M_legal40k_seed43022`.
- Same legal40k accum training completion legal40k 8x480 AdamW/WWM recipe, seed 43 / init 43022 / train RNG 43023, batch 256 via 64-row microbatches, seq256, LR 0.001, warmup 0.06, `lr_total_steps=2529`, 80M exposure, 10M checkpoints.
- Both arms share the identical tokenizer, so any arm difference at 80M is attributable to the 138 role-switch vs role-fixed packet rows, not representation/initialization.

The predeclared natural-transfer interpretation and stop rule are unchanged from the role switch from scratch screen plan plan above.
