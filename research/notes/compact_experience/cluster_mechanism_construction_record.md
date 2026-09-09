# cluster mechanism construction record Construction Record: Natural Entity-Relation Cluster Mechanism Test

## Experimental Design

**Hypothesis**: Packing naturally co-occurring complementary predicate/attribute evidence
about the same entity into adjacent training positions can improve EWoK/COMPS/GlobalPIQA
(world knowledge and commonsense) while preserving clean-Qwen strengths in Supplement,
Entity, Reading, BLiMP, and SuperGLUE.

**Active ingredient being tested**: Complementary predicate evidence — NOT anchor 
repetition, NOT topic continuity, NOT exposure increase.

## Construction Summary

- **Clusters selected**: 3,107 quality-filtered natural entity-relation clusters
  - 97.6% named-entity (`cap:`) anchors
  - Source balanced: gutenberg 38%, simple_wiki 42%, bnc_spoken 11%
  - 1,276 clusters with held-out third sentence for internal diagnostic
  - Total cluster words: 200,002 (2% of 10M pool)
  - Packed into 1,810 row-groups at 160 words each

- **Arms** (all identical 10M pools with 1,810 rows replaced):
  - E1 (true_cluster): Cluster sentences packed adjacently + official padding
  - E2 (anchor_shuffle): Same anchors, cross-block sentences + official padding
  - E3 (anchor_repeat): First sentence repeated + official padding
  - E4 (untouched_tail): Original clean-Qwen pool unchanged

- **Training**: 80M→100M continuation from `chck_80M`
  - Fresh optimizer, cosine-tail LR (~1.06e-4 starting)
  - 20M word exposure (~503 steps per arm)
  - 4 checkpoints: chck_85M, chck_90M, chck_95M, chck_100M
  - Same batch256, seq256, WWM 0.15 masking

## Decision Criteria

**Positive outcome** (E1 promotes to full nine-column evaluation):
- E1 outperforms E2 (anchor exposure alone doesn't explain the gain)
- E1 outperforms E3 (diverse evidence better than repetition)
- E1 outperforms E4 (cluster packing adds value beyond continuation)
- E1 shows favorable EWoK/COMPS/GlobalPIQA profile while preserving Reading/Supplement/Entity

**Negative outcome** (close cluster route, move to evidence-visible masking):
- E1 ≤ E2: effect is just anchor frequency/exposure
- E1 ≤ E3: effect is just repetition
- E1 ≤ E4: packing doesn't add value
- E1 shows same column-redistribution pattern as previous routes

## Potential confounds to watch

1. **Anchor concentration**: Top clusters may over-represent Gutenberg ship names
   and Simple-Wiki geographic entities. If E1 gains only on Entity while losing 
   elsewhere, it's the same redistribution seen in SynCSE experiments.

2. **Padding content**: Cluster rows are padded with official filler to reach 160w.
   The padding may dilute the cluster signal or create discontinuous context.

3. **Continuation drift**: All arms start from same chck_80M, but 20M exposure may
   not be enough for 2% dose to show measurable broad effects. The diagnostic
   (held-out loss) may be more sensitive than no-AoA equal7.

## Fallback route: Evidence-Visible Masking

If clusters fail, a proposed alternative is **evidence-visible masking**: 
selectively masking tokens that require relational/compositional reasoning 
(entity attributes, event outcomes, quantity comparisons) while leaving context 
tokens visible. This changes the loss gradient distribution without changing data
composition. Design would need:
- A corpus-internal scheme to identify "evidence-requiring" token positions
- A control where mask positions are random (standard MLM)
- A control where the same positions are masked but with scrambled context
