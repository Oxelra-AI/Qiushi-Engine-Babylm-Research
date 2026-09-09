# probe results interpretation probe predictions before reading seed43222

This note fixes the measurement targets for the pending seed43222 VIEW / REPEAT / CLEAN arms before any seed43222 result is read.

## Quantities

1. **Held-out natural copy gain**: build source-plus-rotated-copy packets from `heldout_cleanqwen_rows.jsonl`, which was never trained by any arm. Score masked tokens in the copied companion under repeated versus source-corrupted contexts. Gain = NLL(source-corrupted) - NLL(source-present). If exact-repetition training learned a natural text copy computation rather than only fitting its own MAX packet rows, REPEAT should have larger gain than CLEAN and VIEW on these held-out natural packets. The random-token probe from copy and relevant update result remains the nonsemantic comparison: REPEAT need not win there.

2. **Held-out rewrite content-conditioning gain**: use unselected accepted compact source/rewrite pairs from `combined_all_accepted_pairs.jsonl` after excluding `selected_matched_max_pairs.jsonl`. Score masked rewrite content tokens under source-present versus unrelated-source controls. Gain = NLL(unrelated source) - NLL(true source). If VIEW learned to use an earlier span by content, VIEW should have larger gain than CLEAN and REPEAT, especially on rewrite tokens not copied verbatim from the source.

3. **Entity item cue ablations**: on official-filtered Entity items where the initial state is a non-gold available foil after one or more relevant updates, score gold-vs-stale margins under full context, queried-box initial-clause removal, last relevant-update removal, and all relevant-updates removal. If the current mechanism is right, removing the initial clause should help REPEAT's gold-vs-stale margin more than VIEW's, while removing relevant update sentences should hurt VIEW's margin more than REPEAT's. This is the item-level counterpart of the depth/relevant-update crossover.

## Seed43222 prediction

When the pending seed43222 arms become available, score these same quantities before interpreting the third-seed Entity results. The expected ordering is:

- natural held-out copy gain: REPEAT > CLEAN and REPEAT > VIEW;
- rewrite content-conditioning gain: VIEW > CLEAN and VIEW > REPEAT on held-out compact pairs, strongest on non-overlap rewrite tokens;
- Entity cue use: REPEAT remains more initial-state dependent on stale-foil update items, while VIEW remains more relevant-update dependent.

If seed43222 breaks any of these orderings, the principle should be rewritten around the computation actually supported by the three-seed evidence rather than preserving the current copy/content-reading account.
