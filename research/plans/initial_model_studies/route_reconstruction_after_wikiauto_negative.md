# wikiauto pair signal analysis — route reconstruction after the WikiAuto aligned-pair negative result

## What changed

The aligned-vs-shuffled experiment cleanly tested one narrow mechanism:

> Does correct WikiAuto/Turk simplification correspondence in the same MLM window produce better Entity/EWoK representations than the same source and target texts with wrong correspondence?

The answer at 10M exposure under the S1 base is no.

Evidence:

- available-coordinate comparison: `data/aligned_vs_shuffled_10m_available_comparison.json`
- full EWoK comparison: `data/aligned_vs_shuffled_10m_full_ewok_comparison.json`
- synthesis: `notes/aligned_rewrite_mechanism_result.md`
- source analysis: `data/wikiauto_pair_signal_analysis.json`, `notes/wikiauto_pair_signal_analysis.md`

Primary deltas, aligned minus shuffled:

| column | aligned - shuffled |
|---|---:|
| Entity | -0.42 |
| full EWoK | -1.75 |
| GlobalPIQA mean | -1.955 |
| COMPS | -0.15 |
| BLiMP | +1.82 |
| Supplement | +0.18 |
| Reading mean | +0.10 |

The source itself was not empty of surface entities: 81.7% of selected pairs had a source-side noninitial capitalized token, and source capitalized tokens were partly preserved. But the selected WikiAuto/Turk pairs were a weak match to the desired representation pressure:

- content-token Jaccard mean 0.475, median 0.444;
- target/source word ratio median 0.875, with wide spread;
- target content copy fraction mean 0.711, source covered by target mean 0.589;
- relation-word overlap mean 0.497 conditional on any relation cue;
- compared with the official corpus, the selected WikiAuto text has lower pronoun density, lower state/transfer verb density, lower social-word density, lower causal/temporal density, and slightly lower aggregate relation-cue density, despite far more names and numbers.

Scientific interpretation: the experiment measured passive sentence-level simplification adjacency, not dense entity persistence, state transition, physical process, or evidence-to-answer structure. Correct sentence-pair adjacency by itself did not make WWM use cross-surface correspondence in the way needed for Entity/EWoK.

## Important boundary on the S1 comparison

S1 official-corpus 10M is useful as an overall reference, but it is not a perfectly isolated data-only control because its packing and schedule differ:

- S1 official 10M: 245 optimizer steps, packed examples near length 256, about 29.2% truncated, `lr_total_steps=2442`.
- WikiAuto aligned/shuffled: 829 optimizer steps, no truncation, shorter examples, `lr_total_steps=829`.

Therefore the strongest evidence is aligned-vs-shuffled, where text multiset and training geometry are matched. The S1 comparison still matters for route value: the WikiAuto arms did not approach the public leader and damaged several important columns relative to the best available 10M S1 reference.

## What not to do next

Do not scale this exact aligned WikiAuto construction to 100M as the next main experiment.

Do not automatically train all remaining materialized arms (`unpaired_mix`, `source_only`, `rewrite_only`) as a batch. After shuffled already beat aligned on EWoK and Entity did not improve, those arms would mostly subdivide the same weak WikiAuto distribution. They are lower value than rebuilding the mechanism.

`unpaired_mix` remains the only old arm with a clearly distinct residual question: whether the shuffled effect is from wrong same-window co-occurrence or simply from the mixed source+target marginal distribution. It can be run later if the route needs that causal separation, but it should not displace mechanism reconstruction.

## New central hypothesis

The better general principle is:

> In strict small-data pretraining, performance is driven by the density of recoverable relation structure per training word, where the objective must make the model use that structure. Entity persistence, state dynamics, physical/affordance processes, and social/causal relations are partly separate budget axes.

This principle subsumes the failed WikiAuto result: simplification only helps if it preserves and exposes recoverable entity/relation structure. Passive high-overlap sentence pairs are not enough.

## Next main experiment: official-corpus structure-density selection

Before generating or rewriting text, use the legal official corpus itself to test whether ability-bearing structure density predicts gains.

Construct a script to score official-corpus sentence windows or line windows for features such as:

- repeated entity-like mentions across a window;
- pronoun and definite-reference density;
- state/transfer/location/action verbs;
- physical/material/affordance words;
- social interaction and communication words;
- causal/temporal connectives;
- number/quantity and spatial-relation cues;
- source file and length.

Build matched pools using only official-corpus text:

1. `high_entity_state`: high repeated-entity + pronoun + state/transfer score.
2. `high_affordance_physical`: high physical/material/action/spatial score.
3. `high_social_causal`: high social/communication/causal/temporal score.
4. `matched_low_structure`: length/source-matched windows with low scores.
5. `uniform_reference`: deterministic sample under the same pool size/exposure rule.

A first efficient design should use a small unique pool, e.g. 2M words per arm repeated for 10M exposure under the ≤10-epoch rule, because high-density selection necessarily trades off coverage. The matched low-structure and uniform arms separate structure density from repetition and coverage.

Fixed training base:

- S1 DeBERTa-v2 12×384/intermediate1280;
- baseline16k tokenizer;
- AdamW, flat WWM;
- sequence length 256;
- same seeds as recent S1 comparisons;
- no LAMB, no 40k tokenizer, no curriculum, no generated/gated text in the first structure-density test.

Evaluation:

- Entity, full EWoK, GlobalPIQA parallel/nonparallel, COMPS;
- BLiMP, Supplement, Reading as damage indicators;
- training metrics and tokenization/truncation summaries;
- report per-arm structure density per 1k words, not just arm labels.

Predictions:

- `high_entity_state` should mainly move Entity and perhaps EWoK if entity persistence is load-bearing.
- `high_affordance_physical` should mainly move EWoK physical domains and GlobalPIQA parallel/nonparallel if physical process density is load-bearing.
- `high_social_causal` should affect social/causal EWoK and perhaps Reading/Entity if discourse structure matters.
- If no high-density official selection helps, then selection by shallow lexical structure is insufficient; the next mechanism must make the objective explicitly use relations, not only increase their frequency.

## Second mechanism if density alone is insufficient

Build correspondence-aware masking or evidence-to-answer closure only after the official-density test tells which structure axis is alive.

For a selected high-entity/state or high-physical subset, construct pairs or windows where masking deliberately hides one side's entity/relation span while leaving the corresponding evidence visible elsewhere in the window. Compare:

- ordinary WWM;
- span-targeted masking of entity/relation tokens;
- shuffled evidence with the same marginal text;
- wrong-answer or swapped-role counterfactuals.

This tests whether the bottleneck is data structure or the objective's failure to require use of structure.

## Longer route implication

If a high-density official selection works at 10M, it gives a legal, interpretable principle that can be scaled to 100M and combined later with the better verified model-side components. If it does not, the next route should move toward explicit evidence/query/state objectives or richer legal data sources, not more generic simplification variants.
