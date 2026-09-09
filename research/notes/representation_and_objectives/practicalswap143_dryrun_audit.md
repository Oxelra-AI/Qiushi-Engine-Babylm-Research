# globalpiqa practical repair blueprint practicalswap143 dry-run corpus and audit

This CPU-only materialization replaces 143 selected compact-view packets with valid unused practical packets of identical pair-word length at the same pair positions. It writes a 10M dry-run pool and row metadata only; no 100M training stream, GPU job, or evaluation was launched.

## Corpus invariants

- Replacements: 143 pair positions; added pair words 5928, removed pair words 5928, net 0.
- Repaired selected pair list: 12155 pairs / 423511 words; original selected words 423511.
- Pair-row count 3005 and changed-row count 3006 match the original; top-up rows copied from original changed block: 1.
- 10M pool words: 10000000; row word-length sequence identical to original: True.

## Tokenizer and visibility

- Total pair-token delta after-minus-original over all 12,155 pair positions: -737 (source -140, rewrite -597).
- Pair-token delta per position: mean -0.061, p95 0.000, min -55.0, max 48.0.
- Seq256 full source+rewrite visibility after swap: 12063/12155 = 0.992431; source-visible 0.998684; hidden pairs 3.
- Rows over seq256 with special tokens: 89; pair-position loss counts: {3: 45, 4: 28, 2: 17, 1: 1, 5: 1}.

## GlobalPIQA overlap screen

- Added packets with any exact 7-gram shared with GlobalPIQA prompt/options: 0 / 143.
- Added packets with any exact 5-gram shared with GlobalPIQA prompt/options: 1 / 143.
- Added max content-Jaccard to a GlobalPIQA item: mean 0.076, p95 0.133, max 0.200.

## Scientific reading

The dry-run corpus demonstrates that the proposed practical repair can be represented as a very small, exact-whitespace-budget perturbation of the existing compact_view_reinvest stream while preserving row lengths and high joint visibility. The tokenizer-token delta is not zero, so any future trained comparison would still have a small lexical/token-exposure shift in addition to the semantic practical-action shift.
The GlobalPIQA overlap screen should be read as a contamination safeguard, not as a selector. If high overlap examples appear in the JSON tail they must be inspected before training; if the pending endpoint evidence does not survive, this dry-run should remain unused.

Pool: `experiments/archive/representation_and_objectives/data/practicalswap_dryrun_audit/cleanqwen_fineweb_compact_view_reinvest_practicalswap143_10M.jsonl`
Audit JSON: `experiments/archive/representation_and_objectives/data/practicalswap_dryrun_audit/practicalswap143_dryrun_audit.json`

## Decisive judgment (earlier analysis): this 143-swap repair is deprioritized, not to be trained

The dry-run is technically clean (exact 10M, identical row-length sequence, full visibility 0.9924, near-zero token delta −737, zero exact 5/7-gram GlobalPIQA overlap), but the scientific case for training it is weak on three independent grounds:

1. **Wrong scale.** The swap touches 5,928 of 423,520 changed-block words (1.40%) and 143 of 12,155 pairs (1.18%). Even if the injected content were exactly the missing ability, this magnitude cannot plausibly move a benchmark that all three compact arms fail on 66/103 parallel + 38/100 nonparallel items. Spending an H100 run on a 1.4% composition nudge is not a decision-changing experiment.

2. **Wrong content.** The added packets are not genuine affordance / action-outcome / physical-causal experience. Inspection of the highest-overlap additions (Social Security income mix, chondroitin sulfate warning, if-then reasoning definition, Pew mobility report, English-tense course description) shows generic expository text selected by a practical-lexicon heuristic, exactly the "test-shaped lexicon selection" failure the Strategist warned against. Shared tokens with GlobalPIQA items are isolated function/content words (height, tip, out, team, small), never phrase-level, confirming both no contamination and no real practical-reasoning transfer.

3. **Negative composition tradeoff.** Added packets have mean content recall 0.657 vs removed 0.846 and length ratio 0.597 vs 0.690. The swap systematically replaces higher-fidelity compact views with lower-fidelity, more-compressed ones. This is a plausible small *harm* to the consolidation mechanism, not a repair.

Conclusion: the practical-swap route is shelved as a completed-but-unjustified prepared option. It should not be materialized to a 100M training stream or given GPU time. If the pending endpoint evidence later shows compact_view_reinvest is broadly strong but selectively weak on practical causal reasoning, the correct response is a *general* affordance/action-outcome experience intervention selected independently of evaluation wording, at a scale (e.g. a substantial fraction of the changed block or a dedicated new source stream) capable of changing learning, compared against a word-, geometry-, and content-recall-matched neutral swap — not this 143-pair lexicon patch.

The decisive evidence remains the seed43022 full evaluation and the seed43122 training/fast evaluation. Corpus-repair training is not justified before those completed results are interpreted.
