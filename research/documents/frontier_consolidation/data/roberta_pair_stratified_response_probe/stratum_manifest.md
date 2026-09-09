# roberta stratified bridge and early local response RoBERTa pair-stratified response probe event set

Created: `2026-09-02T03:02:26Z`

## Purpose

This frozen event set bridges the aggregate RoBERTa compact-vs-repeat trajectory to the measured natural data marginal. It is designed to ask whether local compact-trained response is concentrated in pairs with high tail/source-wide coverage, high compact content density, or high source-absent compact-content fraction.

The strata are not causal interventions. They only become informative when read together with official-compatible stable-family gains or losses and existing DeBERTa/GPT2/ordered-scrambled evidence.

## Inputs

- Pair file SHA: `6d0ac85dec1718e5f5663d09123e2f62a23f7cceb0b491b83a34f7bfca59d32d`
- Atlas CSV SHA: `7fb667e17d5198a2b44d1ea5381bd0fea7fc32161e81103f9ca8be6c3b8f223f`
- Legal tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Frozen selection

- Candidate events: `323065`
- Selected events: `5470`
- Unique selected pairs: `4338`
- Frozen events SHA: `c172b378873553f209a6bfd9bf53a63e0bde251c1f77690f8abea0ef2d18709a`

Selected events by view/category:

- `compact|function_other`: 1129
- `compact|retained_content`: 1122
- `compact|source_absent_content`: 965
- `repeat|function_other`: 1125
- `repeat|retained_content`: 1129

## Feature thresholds

- `compact_tail_content_coverage`: {"kind": "tertile", "q33": 0.6, "q67": 0.8333333333333335, "n": 12102}`
- `compact_content_fraction`: {"kind": "tertile", "q33": 0.6, "q67": 0.7, "n": 12155}`
- `compact_source_absent_content_fraction_of_content`: {"kind": "zero_low_high_positive", "q33": 0.1111111111111111, "q67": 0.2222222222222222, "n": 12155, "positive_median": 0.2, "positive_n": 9221, "zero_n": 2934}`

## Later evaluation reading

Run with `--evaluate` only after both RoBERTa training arms and their selected official-compatible trajectory have completed. The local response quantity is compact-trained minus repeat-trained NLL on the same masked event; negative is a compact-trained local advantage. The derived `repeat_minus_compact_advantage` flips the sign so positive means compact-trained better.

Concordant evidence would be: stable late official-family gains plus local compact advantages concentrated in the same high-tail/high-density/high-source-absent strata that the DeBERTa mechanism predicts. Discordance should choose one clean future intervention, not launch a broad sweep.
