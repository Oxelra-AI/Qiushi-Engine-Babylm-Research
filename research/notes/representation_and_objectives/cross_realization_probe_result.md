# cross realization probe result — Cross-realization probe: input-distribution enrichment dominates; source-absent target labels not necessary

Evaluation-only saved-checkpoint experiment. No training launched.
Script `experiments/archive/representation_and_objectives/scripts/cross_realization_probe.py`;
result `experiments/archive/representation_and_objectives/data/cross_realization_probe/cross_realization_probe.json`;
task `s238_t9_tool1`.

## What was tested

For 9,109 source-shared **retained content** words that occur exactly once in both a
source sentence and its genuine compact rewrite (identical target token ids in both,
same BPE mass), score the SAME masked target under three contexts on five saved 100M
checkpoints:

- `source` — source sentence, target masked
- `rewrite` — genuine compact rewrite, target masked
- `counterfactual` — a DIFFERENT compact rewrite, matched to the genuine rewrite on
  BPE-piece count (exact), capitalization (exact), number (exact), rel/event class
  (99.6%), hyphenation (98.3%), rewrite word/BPE length, relative position, compression
  ratio, and source-side overlap; the counterfactual context is verified to NOT contain
  the target word (0.0% leakage). Donor slot has the same piece count so identical target
  labels sit at identical mask mass.

Arms: `full_compact_100M`, `drop_abs_100M`, `drop_copied_word_100M`,
`compact_repeat_100M`, `adjbreak_100M`. Metrics per event: piece-normalized target loss
in each context; `rewrite_advantage = counterfactual_loss - rewrite_loss` (positive =
genuine compact context predicts the shared target better than a matched compact context
that does not contain it); `agreement_advantage = cos(h_source, h_rewrite) - cos(h_source,
h_counterfactual)` on final-layer masked-target hidden states. Bootstrap by source
document / source–rewrite pair, 1000 iters.

Matching quality (all events): piece/cap/number match 1.0, rel_event 0.9963, counterfactual
contains target 0.0, |Δrel_pos| 0.0141, Δrewrite_word_len −0.11, Δrewrite_bpe_len −0.35,
Δcompression 0.0028, Δsource_side_overlap 0.0060. These small residuals are CONSTANT across
arms (identical examples), so the load-bearing between-arm contrasts are unbiased.

## Per-arm cross-realization advantage (all events)

| arm | rewrite_advantage | agreement_advantage | rewrite_loss |
|---|---:|---:|---:|
| full_compact | 3.1967 | 0.4427 | 6.7485 |
| adjbreak | 3.2952 | 0.4378 | 6.2907 |
| drop_abs | 3.1353 | 0.4401 | 6.9432 |
| drop_copied_word | 3.1278 | 0.4417 | 6.8342 |
| compact_repeat | 3.0654 | 0.4261 | 6.7542 |

Every arm has a large genuine-rewrite advantage (~3.1–3.3 nats; f>0=1.0) and strong
source↔rewrite representation agreement over counterfactual (~0.43; f>0=1.0). All trained
models build a cross-realization representation that predicts a shared word better from its
own faithful rewrite context than from a matched compact context lacking it.

## Decisive between-arm contrasts (rewrite_advantage; positive = a > b)

| contrast | all | rel_event | plain | frac_gt0(all) |
|---|---:|---:|---:|---:|
| **drop_abs − drop_copied_word** | **+0.0075** | +0.1027 | +0.0012 | 0.657 (interval crosses 0) |
| full − drop_abs | +0.0614 | +0.1444 | +0.0647 | 1.0 |
| full − drop_copied_word | +0.0689 | +0.2471 | +0.0659 | 1.0 |
| drop_abs − compact_repeat | +0.0699 | −0.0222 | +0.0727 | 0.999 |
| **adjbreak − compact_repeat** | **+0.2297** | +0.2923 | +0.2538 | 1.0 |
| **full − compact_repeat** | **+0.1313** | +0.1222 | +0.1374 | 1.0 |
| drop_abs − adjbreak | −0.1599 | −0.3144 | −0.1811 | 0.0 |

agreement_advantage contrasts agree: adjbreak−repeat +0.0116 (f>0=1.0), full−repeat +0.0166
(f>0=1.0), drop_abs−repeat +0.0140 (f>0=1.0); drop_abs−drop_copied −0.0016 (crosses 0).

## What this establishes

1. **Source-absent target labels are NOT necessary for the cross-realization competence.**
   `drop_abs ≈ drop_copied_word` on both rewrite_advantage (+0.0075, interval crossing zero)
   and agreement_advantage (−0.0016). Removing all source-absent compact-content loss labels
   does not remove the genuine-rewrite-over-counterfactual advantage. This is the
   input-exposure/contextual-enrichment signature, and it matches earlier analysis's official-surface
   dissociation (drop_abs ≈ drop_copied on tasks; the strong local source-absent NLL channel
   does not carry the cross-realization skill).

2. **The dominant driver is the compact rewrite INPUT distribution.** The largest, cleanest
   contrasts are `adjbreak − compact_repeat` (+0.23) and `full − compact_repeat` (+0.13),
   both f>0=1.0. adjbreak and full both contain compact rewrites in the input (adjbreak breaks
   source adjacency but keeps the rewrite marginal); compact_repeat contains literal source
   repetition instead of rewrites. The cross-realization representation tracks the presence of
   faithful rewrites in the input, not the loss-label class. adjbreak even slightly EXCEEDS
   full (drop_abs − adjbreak −0.16), consistent with the earlier analysis triangle where adjbreak beat
   repeat and recovered much of the compact benefit.

3. **Joint supervision adds a small, real, rel/event-concentrated increment.** full exceeds
   both deletion arms by +0.06 to +0.07 (f>0=1.0), and full−drop_copied is enriched on
   rel/event targets (+0.247, f>0=0.992). This is the same +0.459 equal7 "small nonzero joint
   contribution" seen on the matched official surface (matched triangle and route judgment) reappearing as a small
   cross-realization increment — but it is ~2–4× smaller than the input-distribution effect
   and drop_abs alone loses almost none of it (full−drop_abs +0.06 vs adjbreak−repeat +0.23).

## Route judgment

The joint-target-complementarity signature (full uniquely and largely exceeding BOTH deletion
arms) is NOT observed: full's advantage over the deletion arms is small and roughly symmetric,
while the compact-input contrasts (adjbreak/full vs repeat) are large. Under the independent_review-endorsed
decision rule, a training-time same-input target-cooccurrence-decoupling 100M arm is therefore
**not warranted as the next step**: the saved-checkpoint interaction does not show the
full-unique signature that would justify it, and the input-distribution effect is where the
signal concentrates.

**Strategist caveat retained (not resolved by this probe):** drop_abs ≈ drop_copied here shows
absent-target labels are unnecessary for the cross-realization effect; it does NOT prove
"compact input alone" caused the compact-vs-repeat gain, because all three deletion/full arms
still receive dense COPIED/shared-content target gradients on the same compact input. The
clean causal separation that remains is: does compact input WITHOUT any content-target
supervision still produce the effect, or is copied-content supervision on compact input
required? The cheapest remaining separation is a saved/analysis route on the existing arms
(there is no arm that saw compact input with zero content supervision), so the next causal
question needs a different construction, not a target-class-decoupling arm.

## Files
- script `experiments/archive/representation_and_objectives/scripts/cross_realization_probe.py`
- result `experiments/archive/representation_and_objectives/data/cross_realization_probe/cross_realization_probe.json` and `.md`
- per-arm per-event metrics `.../cross_realization_probe/<arm>_per_event_metrics.jsonl`
- prepared events + match audit `.../cross_realization_probe/prepared_events.jsonl`, `prep_summary.json`
- task `experiments/archive/representation_and_objectives/tasks/s238_t9_tool1`
