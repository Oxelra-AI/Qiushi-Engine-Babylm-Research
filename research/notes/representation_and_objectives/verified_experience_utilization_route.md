# chunk stream preflight — Verified experience-utilization route

This note integrates the chunk stream preflight construction preflight with independent_review verification. It does not report any new training or evaluation result.

## Corrected mechanism wording

The central mechanism is charged-experience utilization: the training pipeline should not debit words that cannot enter the model or the prediction candidate set. Current row256 to U256 reads a fixed-length chunk-stream intervention that restores row-truncated suffix material; U256 to U64_128_256 reads a short-to-long context/order/packing bundle after full visibility is restored.

## legal40k

- Current row256: 137,061,620 active tokens, 1,286,640 hidden-but-debited words (1.2866%).
- U256: 139,426,440 active tokens (1.0173x current), 761,640 chunks, mask probability 0.147456 to match current expected targets.
- U64_128_256: 139,426,440 active tokens, 1,546,514 chunks, same 100M charged words and 2,530 stage-reset updates as U256; L64 has 1 zero-charge continuation chunk per epoch.
- First-update accumulation microbatches: L64 16, L128 9, L256 5.

## minfreq25

- Current row256: 138,713,400 active tokens, 1,448,730 hidden-but-debited words (1.4487%).
- U256: 141,385,120 active tokens (1.0193x current), 770,920 chunks, mask probability 0.147165 to match current expected targets.
- U64_128_256: 141,385,120 active tokens, 1,564,093 chunks, same 100M charged words and 2,530 stage-reset updates as U256; L64 has 1 zero-charge continuation chunk per epoch.
- First-update accumulation microbatches: L64 16, L128 9, L256 5.

## Before any H100 run

- Build stage-specific chunk streams from the exact 10M corpus and tokenizer SHA, with every whitespace word charged once per epoch and every tokenizer token visible once per epoch.
- Use the preflight-corrected overlong split counts: one L64 overlong continuation chunk per epoch for both legal40k and minfreq25, charged_words=0 on continuation.
- Form exactly 253 optimizer updates per 10M-word epoch for both U256 and U64_128_256 by distributing chunks across updates; do not add a tail update at stage boundaries.
- Apply WWM once on the full effective batch before microbatch forwards; keep one mask generator stream and record realized masked tokens, selected groups, active tokens, charged words, and accumulation depth per update.
- Weight microbatch losses by masked-token count so the update objective is the masked-token mean over the effective batch, not an average of microbatch means.
- Keep optimizer, LR scheduler, warmup, clipping, global step, initialization seed, train RNG seed, and data order continuous across stage boundaries unless intentionally changed in a paired arm.
- Verify a dry run over a full 10M stage epoch before H100 launch: exact 10M charged words, exact preflight active-token totals, no duplicate or missing chunks, no special-token insertion drift, and all checkpoint word thresholds reachable.

## Route use

Do not launch this while depth training is pending. If depth and support-floor results leave the gap, use the U256 vs U64_128_256 pair as the first experience-utilization wave. If U64_128_256 beats U256, add a length-mixture matched interleaved or permuted stream before attributing the effect to chronological order.

JSON: `experiments/archive/representation_and_objectives/data/verified_experience_utilization_route/verified_experience_utilization_route.json`
independent_review: `data/external/independent_review01_verifier1_integration.md`
