# nonce state tracking failed run — nonce-state experiment first run: implementation failures

The first execution of `scripts/nonce_state_tracking.py` completed, but its scores are not scientifically interpretable.

## Failures exposed by the run

1. **No query mask was inserted.** `Episode.text` ends with `Where is E?` and contains no answer. `prepare_batch()` attempted to replace `? answer.` with `? [MASK] .`, which never matched. Therefore `[MASK]` was absent and `query_mask_pos` stayed at its zero default. Endpoint predictions used token position 0 rather than the query.

2. **Unseen-state labels collapsed to class 0.** `state2idx` was built only from training states. `prepare_batch()` used `state2idx.get(answer, 0)`, so every held-out state became label 0. This explains the impossible WESS values of exactly 1.000 on `eval_new_state` and `eval_new_both`; they are label-collapse artifacts.

3. **WESS produced NaNs on padded events.** A batch uses `max_events`, but episodes with fewer events have `(start,end)=(0,0)` for padded event indices. WESS then runs multi-head attention with every event token masked. Softmax over all-masked keys produces NaNs; later multiplication by a zero participation mask does not remove NaNs. WESS loss was NaN from the first epoch.

4. **A fixed train-state classifier cannot measure unseen-state compositional extrapolation.** Even without label collapse, output weights for unseen nonce states receive no training. The appropriate measurement is episode-local retrieval: score the query/slot representation against the four candidate state-token representations in that episode.

5. **Step-state supervision reads invalid token positions for unmentioned entities.** `entity_positions` defaults to zero while labels are filled for the full state table at every event. Thus the step head predicts persistent states from token 0 for entities not mentioned at that event, rather than from a controlled entity-indexed representation.

6. **Held-out template event boundaries are parsed using training templates.** Event lengths are reconstructed from `EVENT_TEMPLATES[template_idx]`, not from the actual rendered event text, so new-template boundaries can be wrong.

## Consequence

The reported endpoint/step-state/WESS accuracies in `data/nonce_state_tracking/results.json` do not compare the intended mechanisms. They must not be used as evidence about role binding, step-state supervision, WESS, or generalization.

## Required v2 representation

- Append `[MASK]` explicitly after the query and assert exactly one mask exists.
- Store rendered event text/token boundaries directly in each episode; never reconstruct them from template indices.
- Use episode-local four-way state retrieval. Encode each candidate state token; score query/slot representations against candidate representations. Gold label is the candidate index, so unseen state strings are meaningful.
- For step-state supervision, predict each entity's current state from entity-indexed representations, not token position 0. A fair credit-assignment arm can pool the entity mention history up to each event or use an explicit per-entity table interface without recurrence.
- Skip padded events entirely or carry an event-valid mask; never call attention with all keys masked.
- Assert finite losses/logits after every forward pass in the pilot.
- Run one-seed/short-epoch pilot first; only after correctness checks should the five-seed comparison run.
