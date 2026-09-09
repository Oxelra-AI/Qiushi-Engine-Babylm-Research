# Dose-arm correction, pilot, and pre-training interpretation

Created UTC: 2026-09-06

## Why calibrated route options dose extraction had to be repaired

The calibrated route options design correctly identified the restatement-dose question, but its extraction script compared inherited compact-experience clean pair ids (`rw2s0_*`, `rw2s1_*`, `rw_*`) against prompt ids in the unsplit extra prompt file (`rw2_*`). That id mismatch counted nearly all extra prompts as rejected and produced `91,057` salvageable originals. It also did not apply the exact inherited source-completeness function that defined the `source_incomplete_or_fragment` rejection bucket.

The corrected pool was rebuilt from the original compact-experience validation sources:

- `experiments/archive/compact_experience/data/qwen_aligned/source_sentences.jsonl` (`rw_*`), 45,000 rows;
- `experiments/archive/compact_experience/data/qwen_clean_aligned/extra_source_sentences_shard0.jsonl` (`rw2s0_*`), 32,804 rows;
- `experiments/archive/compact_experience/data/qwen_clean_aligned/extra_source_sentences_shard1.jsonl` (`rw2s1_*`), 32,804 rows.

Using exact ids, the inherited pool has 110,608 source records, 37,704 clean pairs, 37,594 selected pairs, and 72,904 rejected source ids. Recomputing the original `is_complete_pair_text(..., is_source=True)` removes 18,218 source fragments, matching the inherited metadata. Minor additional exclusions removed 12 low-diversity rows and 27 rows with too many `xxx` placeholders. The corrected salvageable pool is therefore `54,647`, not `91,057`.

Corrected files:

- extractor: `experiments/archive/relation_learning/scripts/extract_dose_originals_corrected.py`
- corrected pool: `experiments/archive/relation_learning/data/dose_arm_originals/salvageable_rejected_originals_corrected.jsonl`
- corrected all prompts: `experiments/archive/relation_learning/data/dose_arm_originals/dose_arm_rewrite_prompts_corrected.jsonl`
- corrected extraction metadata: `experiments/archive/relation_learning/data/dose_arm_originals/extraction_metadata_corrected.json`

## Corrected dose sizes

Existing compact-experience selected aligned-restatement pair words: `1,656,800` per 10M pass, i.e. `16.568%` of the word budget.

Using the existing clean-pair mean pair length, the pre-pilot dose needs are:

| target total ALN pair fraction | target pair words / 10M | additional pair words / 10M | estimated accepted pairs |
|---:|---:|---:|---:|
| 21.0% | 2,100,000 | 443,200 | 10,061 |
| 25.0% | 2,500,000 | 843,200 | 19,142 |

The ~21% arm is now primary for scientific shape, not optional: with 16.6%, 21%, and 25% points, the experiment can distinguish approximately linear return from saturation/crossover much better than a single 25% endpoint.

## Qwen3.5-9B pilot result

Recorded generation settings:

| Setting | Recorded value |
|---|---|
| Model | Qwen3.5-9B (model identifier `qwen3.5-9b`) |
| Prompt input | `experiments/archive/relation_learning/data/dose_arm_originals/dose_arm_rewrite_prompts_corrected_pilot512.jsonl` |
| Generated output | `experiments/archive/relation_learning/data/dose_arm_originals/raw_outputs_qwen35_9b_pilot512.jsonl` |
| Batch size | 32 |
| Maximum new tokens | 80 |
| Temperature | 0.45 |
| Device | `cuda` |

The run produced 512 outputs in 169.18 s with 32,352 generated tokens. Validation used `experiments/archive/relation_learning/scripts/validate_dose_arm_outputs_a06bea42.py`, which fixes two calibrated route options validator issues: it recovers source metadata from prompt indices in the batch-generation output, and uses longest common contiguous span rather than longest common subsequence for the copy filter.

Pilot validation result:

- accepted `241/512` outputs (`47.07%`);
- accepted pair words: `10,908`;
- accepted pair words per prompt: `21.30`;
- mean accepted pair length: `45.26` words;
- accepted content Jaccard mean/median/p95/max: `0.2987 / 0.2667 / 0.5625 / 0.7778`;
- accepted content-overlap-min mean/median/p95/max: `0.4759 / 0.4545 / 0.8 / 0.9091`;
- accepted contiguous-copy span mean/median/p95/max: `3.45 / 3 / 5 / 5`; no accepted row has a six-token contiguous copy;
- main rejection buckets: contiguous LCS > 5 (`182`), entity recall (`76`), high Jaccard (`47`), copy overlap (`41`), rewrite incomplete/fragment (`22`), low content overlap (`16`), number mismatch (`6`).

Accepted register distribution in the pilot was Gutenberg-heavy by pair words: Gutenberg 5,178 words, CHILDES 2,032, SimpleWiki 1,834, OpenSubtitles 1,016, BNC Spoken 816, Switchboard 32. This partly reflects the corrected source pool and partly generation yield; it should remain explicit when materializing filler replacement.

Validation artifacts:

- `experiments/archive/relation_learning/data/dose_arm_validated/validation_metadata_pilot512.json`
- `experiments/archive/relation_learning/data/dose_arm_validated/accepted_dose_arm_pairs_pilot512.jsonl`
- `research/documents/relation_learning/data/dose_arm_validated/accepted_manual_read_sample_pilot512.md`

Manual reading of the sample found mostly usable meaning-preserving restatements with strong copy avoidance, but also several noisy CHILDES/transcript rows and occasional transformations that preserve gist while regularizing syntax or pragmatics. This is not a flaw unique to the pilot; it is the same natural-source regime as the compact-experience study, and the validation holds the important geometry: no accepted contiguous copy span above five and no high-Jaccard accepted row. The dose arm should be interpreted as extra aligned restatement under this noisy official-source distribution, not as perfect semantic paraphrase annotation.

## Full generation sizing

Pilot yield implies:

| target total ALN fraction | additional pair words / 10M | estimated accepted pairs | estimated prompts at pilot yield |
|---:|---:|---:|---:|
| 21.0% | 443,200 | 9,793 | 20,803 |
| 25.0% | 843,200 | 18,630 | 39,579 |

The corrected construction therefore produced two 22,500-prompt shards (45,000 prompts total), enough buffer for the 25% target without generating all 54,647 corrected candidates. Shard files:

- `experiments/archive/relation_learning/data/dose_arm_originals/generation_shards/dose_arm_prompts_corrected_45k_shard0.jsonl`
- `experiments/archive/relation_learning/data/dose_arm_originals/generation_shards/dose_arm_prompts_corrected_45k_shard1.jsonl`

At the time of this note, shard1 generation was in progress and shard0 generation remained planned. These were not yet completed full-generation results.

## Readout rules before training

The dose arm is not an Entity-primary intervention. Entity aggregation is too noisy relative to the expected effect and was misleading for the state-update arm. The first two deciding readouts are:

1. ordinary held-out MLM loss at matched checkpoints and two seeds;
2. in-family source-conditioned T/U/N margins for compact/Wikipedia/source-recurring restatement use.

The ordinary-heldout screen needs two separate uses. For **curve shape**, the 21% and 25% dose points are compared with the line through OFF and the inherited 16.6% ALN point. Because 21% adds only about 4.4% of the pool, linear return from ALN's approximately `-0.0125` ordinary-heldout movement predicts only about `-0.0033` nats there; the former `0.004` rule would reject the primary curve point even if returns were perfectly linear. Use paired within-seed values rather than a pooled mean: previous within-seed ALN/OFF and SHUF/OFF heldout contrasts replicated at roughly the `0.001` level across seeds.

For **practical continuation**, gate expensive cheap/common, official, and coherent-replay work mainly on the 25% point: it adds about 8.4% of the pool, with a linear prediction around `-0.0063` nats. Both new points below the OFF→ALN line mean diminishing return; 25% worse than 21% means crossover; points on or above the line mean added restatement is still paying. Proceed beyond ordinary-heldout and in-family margins only if the 25% point shows absolute improvement together with dose-ordered source-conditioned margins and no broad degradation pattern. Read cheap7 as a column vector, not as an aggregate decision, because BLiMP, EWoK, Supplement, and GlobalPIQA have shown low-SNR or tradeoff behavior in comparable contrasts.

## How the credit-allocation result changes the ceiling

The focused-credit binding pilot shows that entity binding needs credit allocated to the answer computation: standard WWM on a small paired pilot partly closed a retention gap but destroyed update detection, whereas focused answer masking closed the retention-neutral gap while strengthening update detection. This explains why the state-update packets under ordinary WWM induced a coarse retention rule rather than entity-gated binding.

For the restatement-dose arm, this means the experiment should not be expected to install negative-selection entity binding. It tests dose-return for an already working corresponding-restatement relation under the standard objective. If the field goal is entity binding specifically, the better scientific route is contrastive same-source packets with answer-credit concentration, not merely more ALN under 15% WWM.
