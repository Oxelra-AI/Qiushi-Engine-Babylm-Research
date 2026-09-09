# target channel semantic structure: source-absent target channel is semantically structured

CPU-only, using the completed 20M target-selective loss files (`wholeword copied control plan` train-fixed
event losses, `source disjoint target channel result` source-disjoint held-out event losses). Contrast is
`drop_abs_content − drop_copied_word` on compact-side `source_absent_content` events;
positive = removing source-absent labels hurt that subset more than removing matched
copied whole-word labels. Semantic groups use the packed targetselect static load and readout patch heuristic flags (no learned
tagger). Groups overlap except the explicit relational/event vs not partition.

## Key result (chck_20M, source_absent_content)

Train-fixed (`wholeword copied control plan`):
- all: n=4096, delta **+0.0803** nats, boot[+0.0587,+0.1002], frac>0 1.0
- relational_or_event_state: n=533, delta **+0.3028**, boot[+0.2172,+0.3920], frac>0 1.0
- not_relational_or_event_state: n=3563, delta +0.0592, boot[+0.0349,+0.0838], frac>0 1.0
- ordinary_nonrel_nonentity (novel content, not names/numbers/rel/event): n=3238, delta +0.0947, boot[+0.0706,+0.1196], frac>0 1.0
- capitalized_or_number (names/dates): n=282, delta **−0.1241**, boot[−0.1858,−0.0687], frac>0 **0.0**

Exact-source-disjoint held-out (`source disjoint target channel result` source_disjoint_quality):
- all: n=974, delta +0.0796, boot[+0.0366,+0.1257], frac>0 1.0
- relational_or_event_state: n=159, delta **+0.4370**, boot[+0.2778,+0.6011], frac>0 1.0
- not_relational_or_event_state: n=815, delta +0.0350, boot[−0.0117,+0.0835], frac>0 0.93
- ordinary_nonrel_nonentity: n=732, delta +0.0657, boot[+0.0123,+0.1189], frac>0 0.993
- capitalized_or_number: n=79, delta **−0.1022**, boot[−0.2020,+0.0127], frac>0 0.04

## Reading

1. The validated source-absent channel is not a uniform "novelty" effect. It is
   ~3.8x (train) to ~5.5x (held-out) stronger on relational/event-state rewrite words
   than the overall source-absent mean, and it **reverses sign** on capitalized/number
   tokens (proper names, dates). Removing source-absent name/date labels actually helped
   the model on those same categories relative to removing matched copied labels — i.e.
   compact rewrites' novel proper-noun/number targets are, if anything, harmful noise,
   while their novel relational/event-state targets carry the positive channel.

2. This is robust to exact source-disjointness (source disjoint target channel result): the relational/event
   concentration and the name/number sign reversal reproduce on held-out rewrites whose
   exact source sentences were never trained. So it is not repeated-row memorization and
   not tied to a particular document; it is a property of *which kind of novel word* the
   compact rewrite introduces.

3. Interpretive fork for the pending 100M packed endpoint (strategist note): if
   `drop_abs_content` loses Supplement + relational EWoK at 100M, the naturally coupled
   population that carries the endpoint is the novel *relational/event-state* abstractive
   content, not source-absence per se and not novel names/numbers. The removed target
   population is itself enriched in exactly the semantics (relation/state/abstraction)
   that Supplement and the correctness transition analysis relational EWoK domains reward.

4. `ordinary_nonrel_nonentity` (novel content words that are neither names/numbers nor
   flagged relational/event) still carries a clear positive channel (+0.095 train,
   +0.066 held-out). So novelty-of-content contributes broadly, but relational/event
   semantics is where the channel is by far strongest, and names/numbers are where it is
   negative. Source novelty and relational-content semantics are still confounded in the
   natural compact marginal.

## Next separation (before any new GPU spend)

To distinguish source novelty from semantic content selection, the future target-set
design should cross two axes on matched copied-vs-source-absent labels:
- axis A: source-novel vs source-copied (already the current contrast)
- axis B: relational/event-state vs ordinary-content vs name/number

A clean test holds relational/event semantics fixed while varying source novelty (do
copied relational/event target labels carry a similar channel?), and holds source
novelty fixed while varying relational/event semantics. If the endpoint pattern tracks
relational/event content rather than source novelty, the operative property is compact
abstractive-relational content, and "source absence" is only a marker of where that
content tends to sit in teacher rewrites.

## Files
- `experiments/archive/representation_and_objectives/scripts/target_channel_semantic_stratification.py`
- `experiments/archive/representation_and_objectives/data/target_channel_semantic_stratification/target_channel_semantic_stratification.json`
- inputs: wholeword copied control plan event_losses SHA + source disjoint target channel result source_disjoint_event_losses SHA recorded in JSON `inputs`.

CPU-only analysis; neither pending packed-target-selective 100M endpoint was inspected or inferred.

## Step234b: novelty × relational interaction (decisive cross-cut)

Using the same wholeword copied control plan 20M event losses, split by category (source_absent vs retained
copied content) × semantics (rel/event vs other), `drop_abs − drop_copied_word` at chck_20M:

| cell | pieces | piece-weighted delta | boot95 |
|---|---|---|---|
| source_absent × rel/event | 582 | **+0.3028** | [+0.2186, +0.3864] |
| source_absent × other | 6156 | +0.0592 | [+0.0370, +0.0827] |
| retained(copied) × rel/event | 237 | **+0.0082** | [−0.1329, +0.1630] |
| retained(copied) × other | 7498 | −0.0700 | [−0.0920, −0.0473] |

Interaction I = (SA_rel − SA_other) − (RC_rel − RC_other) = **+0.1653**.

Reading: the channel is not "relational/event words are hard" (copied relational/event
content shows a near-zero gap, +0.008) and it is not merely "novel content" (novel
non-rel/event content is +0.059). It is the **conjunction**: novel relational/event
words carry the largest gap by far. Source novelty and relational/event semantics
interact rather than either being sufficient alone. This is the sharpest statement of
what the source-absent target channel actually teaches in this DeBERTa/FineWeb family:
the model gains predictive capability specifically for teacher-introduced novel
relational/event/abstractive content, exactly the semantics rewarded by Supplement and
the correctness transition analysis relational EWoK domains.

This makes the pending 100M interpretation more precise: if `drop_abs_content` loses
Supplement + relational EWoK, the operative removed population is the novel
relational/event abstractive content, not source-absence in the abstract and not novel
names/dates (which showed a negative gap). If it does not, the local channel is real but
does not carry official competence — the constraint ordered/scrambled 40M result
already demonstrated is possible.

File: `experiments/archive/representation_and_objectives/data/target_channel_semantic_stratification/novelty_x_relational_interaction.json`
Script: `experiments/archive/representation_and_objectives/scripts/revision_234b_novelty_x_relational_interaction.py`
