# relation bias execution replay relation-route readiness while legal40k evaluations run

This step did **not** launch training, official evaluation, packaging, or endpoint interpretation. The two decisive legal40k official evaluations remain managed asynchronously: `s65_t22_tool1` (seed43022) and `s66_t9_tool1` (seed43122). The work here makes one possible post-40k branch safer if those evaluations fail.

## Why this CPU-only work matters

If legal40k fails, two genuinely distinct routes remain: relation-cue WWM masking or true 12x384 depth. Before an expensive H100 relation-masking run, this analysis tests whether the prepared relation route is well defined on the real legal corpus and trainer path.

## Large 10M-pool relation-dose audit

Script: `experiments/archive/representation_and_objectives/scripts/relation_bias_large_stream_audit.py`

Output: `experiments/archive/representation_and_objectives/data/relation_bias_large_stream_audit/relation_bias_large_stream_audit.json`

- Exact audited pool/tokenizer: pool SHA and legal40k tokenizer SHA both matched expected.
- Rows/words: `64,740` / `10,000,000`.
- Visible groups/tokens under legal40k seq256: `9,871,336` groups and `13,706,162` tokens.
- Relation-bearing visible groups/tokens: `1,079,828` (`0.109390`) and `1,316,350` (`0.096041`).
- Row relation-group fraction p50/p90/p99: `0.1074` / `0.1625` / `0.2125`.

Candidate WWM boosts:

| setting | relation selected-group fraction | relation target-token fraction | selected-group multiplier | target-token multiplier | clipped rows |
|---|---:|---:|---:|---:|---:|
| boost 2.0 | 0.218781 | 0.194706 | 1.000000 | 0.986522 | 0 |
| boost 3.0 | 0.328166 | 0.296100 | 1.000000 | 0.973046 | 5 |
| boost 4.0 | 0.437356 | 0.400149 | 1.000000 | 0.959609 | 120 |

Boost 2.0 remains the cleanest first relation-pressure setting: it makes a large dose change, has no row-level clipping on the full pool, preserves expected selected word-group count, and induces only a modest expected target-token reduction from selecting shorter relation cue words more often. Across source classes under boost 2.0, relation selected-group fraction ranges from `0.1221` in `simple_wiki` to `0.2637` in `bnc_spoken`; the route is broad rather than confined to the compact FineWeb block, but its effective dose is source-dependent.

## 100M launch-prefix identity

Script: `experiments/archive/representation_and_objectives/scripts/relation_bias_prefix_identity.py`

Output: `experiments/archive/representation_and_objectives/data/relation_bias_prefix_identity/relation_bias_prefix_identity.json`

independent_review correctly noted that the trainer consumes the 100M JSONL, not the standalone 10M JSONL. The check shows the first 10M charged words of the 100M launch stream are exactly a row-object multiset permutation of the audited 10M pool:

- Pool rows/words: `64,740` / `10,000,000`.
- 100M first-pass rows/words: `64,740` / `10,000,000`.
- Row-object multiset equal: `true`.
- Source words equal: `true`.
- Stream-only / pool-only row counts: `0` / `0`.

The sequence order is different, which is expected because the 100M stream is permuted by pass. The large-pool dose statistics describe the actual first-pass row multiset, while order-specific learning effects remain part of the fixed training stream.

## Execution-path replay on real launch-order rows

Script: `experiments/archive/representation_and_objectives/scripts/relation_bias_execution_replay.py`

Output: `experiments/archive/representation_and_objectives/data/relation_bias_execution_replay/relation_bias_execution_replay.json`

A full three-seed replay over all 64k rows timed out at 900s, so I separated the exact full-prefix identity above from a 10,000-row real launch-order replay. The replay used actual train RNG seeds `43023` and `43123`, disabled/base WWM and enabled boost-2 WWM, without model forward or optimization.

For the 10k launch-order subset:

- Stream subset rows/words: `10,000` / `1,544,605`; standalone 10M first 10k rows/words: `10,000` / `1,507,476`. The subset row multiset is not expected to match because the stream is permuted.
- Relation dataset matched the base dataset for first 256 rows on `input_ids`, `attention_mask`, and `word_group`.
- Relation selected-group fraction base → boost2: `0.109144` → `0.220130`.
- Relation selected-token fraction base → boost2: `0.095623` → `0.195182`.
- Realized boost/base selected-group ratio: `1.000142`.
- Realized boost/base selected-token ratio: `0.986858`.
- Expected boost selected-group multiplier vs uniform: `1.000000`; expected token multiplier: `0.986387`.
- Boost clipped rows: `0`; zero-selected row fraction base/boost: `0` / `0`; partial selected groups base/boost: `0` / `0`.
- Observed special-token random replacements: base `8`, boost `5` over this replay. This is inherited from the base trainer's random-token corruption policy and is not unique to relation bias.

## independent_review verification

Integrated independent_review memo: `data/external/independent_review01_verifier1_integration.md`

Main reading: the route is a genuine objective/target-selection intervention if tokenizer, architecture, data stream, optimizer, seeds, WWM curriculum, and sequence schedule are fixed. Boost 2.0 is the best-supported first elevated dose. Remaining confounds: cue frequency/predictability, source-dependent dose, lexicon precision/provenance, unpaired mask randomness, visible-word/truncation accounting, and inherited special-token random replacements.

## Practical route implication

Do not launch this route until the pending legal40k official vectors arrive. If legal40k clears or nearly clears the frontier, protect and reproduce that compliant endpoint instead. If legal40k fails in the relation/world-knowledge columns, relation-cue WWM boost 2.0 is now a better specified, GPU-ready learning-signal route than it was at relation bias trainer preflight and is distinct from another vocabulary-size sweep. If legal40k fails broadly or mostly in non-relation columns, true 12x384 depth may be the better next mechanism.

Consolidated JSON: `experiments/archive/representation_and_objectives/data/relation_route_readiness/relation_route_readiness.json`
