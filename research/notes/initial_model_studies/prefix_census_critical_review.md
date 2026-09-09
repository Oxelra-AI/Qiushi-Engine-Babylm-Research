# Prefix-target census: interpretation and controls

## What is solid

The prefix target census v2 census repaired the largest artifact in v1. In v2, deleted, same-source, cross-source, and block-shuffled variants all keep the suffix and masked target at the same absolute positions. Deleted prefix positions are attention-masked pads, and same/cross prefixes are padded or truncated to the true prefix length. Targets are word-start alphabetic tokens, reducing fragment artifacts.

The fixed-position result is therefore scientifically meaningful as a first answer to the earlier analysis question. Mature WWM models do have many later tokens whose score changes when the true earlier prefix is replaced by another same-source prefix:

- 100M: 172/600 targets have sign-stable `full - same_source_near >= 0.05` across seed42 and seed43.
- 100M: 146/600 have sign-stable `>= 0.10` and 99/600 have sign-stable `>= 0.20`.
- The signal appears in both exact 1M-slice and broader official examples.

This means the near-zero WWM sensitivity in the earlier RTD/CPC/continuation probes should not be interpreted as proof that official text lacks prefix-dependent prediction targets. Those probes likely sampled many locally predictable or prefix-insensitive tokens.

## What is not yet solid

The v2 signal is not clean evidence for the Entity/EWoK bottleneck or for state-like credit assignment.

The prefix census critical review review script `scripts/review_prefix_census.py` compared the same-source effect against deleted, cross-source, and block-shuffled controls and decoded representative cases. It saved:

- `data/prefix_census_review.json`
- `notes/prefix_census_review.md`

At 100M:

- 172/600 targets are positive against same-source-near at threshold 0.05.
- 82 of those are not positive against deleted prefix.
- 83 are not positive against cross-source prefix.
- 146 are not positive against block-shuffled prefix.
- Only 13/600 are positive against same-source, deleted, cross-source, and block-shuffled prefixes simultaneously.
- Same-source deltas correlate strongly with deleted-prefix deltas (0.754) and cross-source deltas (0.636), and weakly with block-shuffled deltas (0.202).

This pattern says the population is heterogeneous. Many effects are not an ordered-context or entity-state effect. Some are broad local document/topic coherence; some are source/register effects; some are target/suffix phrase effects; some may be true discourse dependence.

## Representative-row reading

The decoded examples are useful. Several high positive cases are plausible discourse continuations but not necessarily Entity Tracking mechanisms:

- `philosopher`: true prefix discusses individuation/history/body; suffix contains “poetical or philosopher”. The true prefix helps topic and technical register. This is meaningful context, but not a state update.
- `eyes`: CHILDES/Frosty context; suffix asks what was used for Frosty’s eyes. This is a plausible cross-turn discourse dependency and worth preserving as a candidate type.
- `simple`: music/time-signature dialogue; target is locally repeated in the suffix window and marked by local structure, so the `local_leak_10tok` feature may miss some leakage.
- `attempt`, `live`, `mountains`, `House`, `relationship`: true prefixes usually provide topic/register continuity. Some are useful discourse coherence, but they are not controlled same-topic wrong-state contrasts.

The robust-all-controls examples are mostly strong lexical/topic continuity:

- `ninety`: prefix and suffix both contain the phrase “ninety five per cent”; strong lexical continuation.
- `alphabet`: true prefix and suffix both discuss alphabet/letters/graphs; strong topic continuity.
- `shape`, `local`, `Russian`: clear lexical or topical carryover.

The negative examples also matter. Some true prefixes lower the target score relative to replacements, showing that true context can introduce strong priors that conflict with the local suffix, or that the replacement prefix sometimes supplies easier topical cues. This argues against treating all prefix dependence as beneficial supervision.

## Judgment for the route

The v2 census supports this narrower statement:

> Mature WWM checkpoints contain a nontrivial population of later tokens whose MLM score depends on earlier prefix content under fixed-position same-source comparisons.

It does not support the stronger statement:

> We have a clean entity/state target population that can be used directly for training to improve Entity/EWoK/GlobalPIQA.

So the current v2 census should not be turned directly into a CPC/continuation training set. It is a route-finding measurement. It shows where to look, but it also shows why broad all-suffix objectives failed: useful effects are a minority and mixed with lexical/topic/register patterns.

## Next scientific work

The next step should build a stricter prefix-target surface, not train yet.

Required improvements:

1. Add a true-prefix-vs-block-shuffled requirement. If full prefix does not beat block-shuffled prefix, the target is not using ordered prefix structure.
2. Add a same-document or same-conversation nonadjacent negative when possible, not only random same-source. Same-source BNC examples can still differ wildly in topic.
3. Decode and classify a small high-response sample into categories: lexical repetition, topic/register, discourse anaphora, numeric/quantity continuation, entity/state, and artifact.
4. Use candidate selection based on seed42 and evaluate held-out seed43, or select at 80M and test at 100M, to reduce response-selection bias.
5. Keep fixed positions and word-start targets from v2.
6. For any future training objective, train only if there is a substantial subset where full prefix beats same-source, same-document/same-topic negative, deleted prefix, and block-shuffled prefix with sign stability.

If the stricter surface yields mostly lexical/topic cases and very few entity/state or ordered-discourse cases, the cross-sentence target route should stop. The next training route should then pivot to a different bottleneck: word/morph anchoring, factorized WWM cadence, or another architecture-level change grounded in BabyLM findings.

The current-best direct-checkpoint recheck from earlier analysis is still unfinished and should be done before interpreting endpoint-level Overall gaps.
