# Porting guide: conditional-innovation WWM

## Recommended treatment

Port `innovation_biased_wwm.py` as a narrow branch of the existing fixed-256,
fixed-15% WWM path.  Preserve the spatial repair route status tokenizer, frozen 100M JSONL, 8x480
model, token-mean MLM reduction, optimizer, LR schedule, seed, batch size, and
word-exposure accounting.  The only treatment variable is the selected WWM
group set.

For each batch, reproduce the baseline Bernoulli WWM selection first.  Each
changed row proposes at most one strict source-absent rewrite group, chosen by a
stable hash of `(seed, step, example_id, group_id)`.  If baseline already chose
it, record a collision and do nothing.  Otherwise replace one already-selected
ordinary group of the same token length.  Donors must be outside every
source/rewrite pair group; prefer ordinary rows.  If no donor exists, skip the
proposal.  This makes selected group and token mass exactly equal per batch.

Do not prioritize relation cues in the first screen.  They remain a diagnostic
slice only.  Do not force copyable targets, and do not take a donor from any
paired source or rewrite group.

## Minimal trainer integration

1. Add `example_id` to `MaskedChunkDataset.__getitem__` and stack it in
   `collate`.  The frozen JSONL already supplies it and repeats each changed id
   exactly ten times with identical text.
2. Load `innovation_metadata_train.jsonl` once into a dictionary keyed by
   `example_id`.  Before training, assert the frozen stream SHA, tokenizer JSON
   SHA, metadata SHA, `seq_length=max_seq_length=256`, `wwm_fixed`, and
   `mask_prob=0.15`.
3. In the training loop, build a metadata list parallel to the batch.  Call
   `innovation_biased_selection(..., priority_salt=f"{seed}:{step}")` instead
   of the baseline selection sub-block.  Then apply the inherited per-token
   80/10/10 corruption and unchanged token-mean MLM loss.
4. Log baseline/biased selected groups and tokens, proposals, collisions,
   successful swaps, no-donor skips, forced relation-cue count, forced
   copyable count, donor provenance, and source/copyable selection deltas.
5. Assert every step: batch group delta zero; batch token delta zero; forced
   copyables zero; protected-pair donors zero.  A no-donor skip is safe because
   it leaves the baseline selection unchanged.

Only positions, group ids, and audit categories control selection.  The lean
trainer payload contains no normalized strings.  Never pass category flags,
pair ids, or source spans as model inputs.
Metadata is computed from the same legal training text and tokenizer, not from
evaluation data, model losses, or held-out labels.

## Whole-word and corruption caveat

Strict metadata guarantees every selected target includes all tokenizer pieces
of its WWM group.  The inherited trainer nevertheless applies 80/10/10
corruption independently per token, so a selected multi-piece group can have an
intentional gold keep-branch piece.  Keep this behavior for the first screen to
avoid adding a corruption-policy variable.  Log it.  Do not call those events
span leakage: all pieces remain selected and labeled.  If the initial screen is
ambiguous, a later separately controlled experiment could use group-coherent
corruption, but it is not part of this construction.

## Pre-screen gates

Run the unit tests and CPU smoke after porting, then a one-step no-update or tiny
optimizer smoke with the real trainer.  A 20M/40M H100 screen is justified only
if the port reproduces exact batch mass and zero pair-donor/copyable violations.
At 20M/40M, read the route reopen conditional innovation innovation true-source/source-masked/decoy probe and
broad Supplement/EWoK/BLiMP-compatible columns.  Continue only if innovation
loss improves without replaying the GlobalPIQA-nonparallel-only trade.

## Why not an auxiliary predictor first

A source-to-rewrite predictor adds a head, loss weight, representation pooling,
and usually an extra attention/forward-path decision, so its loss mass is not
naturally matched to baseline MLM.  It is also easier to leak the target because
a source representation from the ordinary bidirectional pass can attend to the
rewrite.  A leakage-safe predictor would require a source-only attention view or
separate source encoding, mask every target piece before encoding, and prevent
all rewrite target tokens from entering the predictor context.  That is a
different multi-variable experiment.  The exact-swap WWM treatment is the safer
first test.
