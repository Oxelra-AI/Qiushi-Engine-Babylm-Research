# Qualitative Audit of Natural Variation-Set Pairs

This is a historical qualitative audit of the existing CHILDES variation-set probe. No new sampling, model scoring or experiment was performed to prepare this record. The [construction summary](variation_set_summary.md) describes the probe population and its surface-exposure restrictions.

## Sample and Observations

The inspection covered ten deterministic pairs from each lexical-overlap bin, **30 pairs in total**, together with decoded masked inputs from all three bins. In the inspected examples, the utterances were readable participant speech and locally adjacent, and the lexical-overlap labels agreed with the displayed texts. Decoded inputs placed `<mask>` at the intended content word. No malformed special-token sequence or annotation-only target was observed in this inspected sample.

The illustrative low-, partial- and high-overlap pairs had Jaccard values **0.143**, **0.333** and **0.600**, respectively. These are individual example values, not bin thresholds; the transcript quotations are omitted.

These observations establish sample-level readability and target-placement checks, not semantic equivalence of every retained pair.

## Interpretation Limits

1. **Adjacency and overlap are not a paraphrase annotation.** In particular, some low-overlap pairs were topic continuations or successive narration rather than reformulations. The instrument should not be described as a uniformly verified paraphrase set.
2. **Transcription artifacts remain.** Dysfluencies and occasional missing-space forms were retained where cleaning would require linguistic reconstruction. Readable examples do not establish fully normalized text.
3. **Non-child speaker filtering is not caregiver identification.** The literal `speaker != CHI` criterion admits sibling codes such as `BRO` and `SIS`. A stricter caregiver sensitivity analysis would require a trustworthy corpus-specific speaker-role map; it was a proposed sensitivity analysis, not an established result.

The [existing machine validation](../../../../../experiments/archive/relation_learning/analysis/childes_variation_set_probe/validation_results.json) checked selected pairs against physical source lines and found no provenance or adjacency-rule mismatch. Such structural validation does not resolve the semantic and speaker-category limitations above. Raw transcript quotations are not reproduced here.
