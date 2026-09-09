# ordered prefix route decision — Ordered-prefix route decision

Source: `experiments/archive/initial_model_studies/data/ordered_prefix_analysis.json`

## Decisive measurement

Targets needing BOTH the true prefix specifically (full > same_source_near >= 0.05, sign-stable)
AND ordered prefix content (full > block_shuffled >= 0.05, sign-stable):

| exposure | ordered+specific | fraction | at threshold 0.02 | fraction |
|---|---:|---:|---:|---:|
| 40M | 29 | 0.0483 | 44 | 0.0733 |
| 80M | 28 | 0.0467 | 52 | 0.0867 |
| 100M | 26 | 0.0433 | 47 | 0.0783 |

Cross-exposure stability: 26 at 100M, 28 at 80M, overlap 21.

## Route interpretation

The ordered-prefix filter is the key discriminator between topic/lexical coherence
and genuine ordered discourse dependence. With the block-shuffled requirement:

- At 100M only 26/600 targets (4.3%) survive both filters at threshold 0.05.
- At threshold 0.02, 47/600 (7.8%) survive.
- The remaining 146 same-source-positive targets that fail block-shuffled
  are primarily topic/lexical continuity rather than ordered-state dependence.

## Route decision

The ordered-discourse target population on official text is too sparse (<5% of sampled targets)
to support a reliable training objective. Any CPC, continuation, or contrastive mechanism
that trains on broad suffix targets will be dominated by the much larger locally-predictable
and topic-continuity populations, explaining why RTD, CPC, and prefix-continuation all
learned source/register contrasts instead of ordered-state effects.

**Cross-sentence ordered-state objective route is closed for official BabyLM text.**

The next training routes should attack the Entity/EWoK/GlobalPIQA deficit through
channels that do NOT require ordered prefix-to-suffix credit assignment:

1. **Factorized WWM cadence** (sequence-length schedule, mask-rate decay, smaller effective batch)
   — directly supported by BabyLM 2025 findings; targets context integration and optimization.
2. **Word/morphology-level representation anchoring** — targets low-frequency entity/property
   parameter sharing without requiring a new training interface.
3. **Architecture/tokenizer changes** — more radical but with strong prior evidence.
