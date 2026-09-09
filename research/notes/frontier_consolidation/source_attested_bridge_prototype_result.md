# mechanism evidence synthesis: Improved source-attested fluent bridge — decisive prototype result

## Scientific question
Can source-attested fluent re-expression (using only words from the source, but in fluent compressed sentences) reproduce the characteristic of natural compact views that source-only extractive selection could not?

## Evidence produced
- 250 stratified pairs sampled from 12,155 (hard_absent_relation 80, hard_absent_norelation 40, relation_zero_absent 70, ordinary 60)
- 750 first-pass prompts (3 regimes: lexclosed_compress, anchor_gapfill, clause_bridge) with improved completeness requirements
- 144 repair prompts for repairable failures
- Strict validation with finite verb, protected complement, polarity, geometry, grammar pattern, and source-copy degree checks

## Main result

### Constructibility: moderate
- First pass: 189/750 rows accepted (25.2%), 103/250 pairs accepted (41.2%)
- Repair pass: 0/144 succeeded (136/144 too long; Qwen3.5-9B does not follow word-count constraints under repair)
- By bucket: hard_absent_relation 37/80 (46.3%), hard_absent_norelation 10/40 (25.0%), relation_zero_absent 31/70 (44.3%), ordinary 25/60 (41.7%)
- Best regime: anchor_gapfill 81/250 (32.4%); worst: lexclosed_compress 51/250 (20.4%)

### Source-copy degree: decisive
- **87.4% of accepted outputs (90/103) are full source-word copies** (source_copy_degree ≥ 0.999)
- Only 13/103 (12.6%) show any content substitution beyond deletion/reordering
- Mean source-copy degree: 0.981; median: 1.000
- The source-attested constraint inherently prevents the kind of semantic re-expression that distinguishes natural compact from source repetition

### Geometry and overlap
- Generated/natural word ratio: mean 1.137, median 1.167 (bridge outputs ~14% longer than natural compact)
- Natural content overlap: mean 0.874 (bridge shares ~87% of content words with natural compact)
- Source content recall: mean 0.918 (bridge covers ~92% of source content words)
- Generated/source word ratio: mean 0.750 (bridge is ~75% of source length — real compression)

## Comparison with earlier experiments

| Arm | Source-absent content | Fluency | Source span | Stable-family result |
|-----|----------------------|---------|-------------|---------------------|
| Natural compact | ~17,891 instances | Fluent | 0.807 | **+1.35 mean7 @ 80M** |
| Extractive balanced | Zero | Telegraphic | 0.972 | -0.26 cheap6 |
| Extractive wide | Zero | Telegraphic | 0.928 | -0.66 cheap6 |
| Source-attested bridge (this) | Zero | Fluent (but mostly source-copy) | N/A | **Not trained** |

The bridge was meant to isolate fluent sentence form from novel vocabulary. But the actual outputs are not genuinely re-expressed — they are slightly edited/shortened source sentences. This makes the bridge structurally closer to extractive (same source words, different packaging) than to natural compact (novel vocabulary, genuine re-expression).

## Scientific conclusion

**The source-attested constraint cannot produce the kind of contextual re-expression that natural compact views provide.** When forced to use only source-attested content lemmas, the generator overwhelmingly copies source words directly (87.4%), producing outputs that differ from the source only by deletion and minor reordering. This is extractive with grammar polish, not the semantic recoding that characterizes natural compact views.

Natural compact's distinctive features — source-absent vocabulary ("link" for "connection", "means" for "associated with", "cooperated" for "was very cooperative"), re-expressed proposition structure, and novel contextual framing — are inherently excluded by the source-attested constraint. The bridge therefore cannot test whether fluent re-expression separates from novel vocabulary; the constraint collapses fluent re-expression into fluent extraction.

**This means a full 100M training with source-attested bridge views would not be informative**: the contrast with extractive is too small (same words, slight grammar improvement), and the extractive experiment already showed source-only selection is insufficient for stable selected families.

## Implication for mechanism understanding

The accumulated evidence now points toward natural compact's advantage being carried by the coupled system of:
1. **Novel vocabulary (source-absent content)**: provides genuinely new prediction targets for MLM
2. **Faithful semantic compression**: preserves the source proposition while changing its expression
3. **Fluent syntactic form**: enables DeBERTa's masked denoising to learn from grammatical contexts
4. **Source-wide coverage with reinvested diversity**: broadens the experience base

These factors cannot be separated by source-only construction because the first factor (novel vocabulary) is what makes compact views qualitatively different from any source-word rearrangement.

## What this means for the score-improvement path

The existing endpoints remain:
- chck_82M: Overall 41.94 (submitted)
- chck_84M: projected Overall 42.02 (not submitted)
- coherent86 α=0.75: projected Overall 42.12 (not submitted)

**The most direct score improvement candidate is chck_84M coherent replay**: freeze 84M as anchor, train private adapter on the 84M→88M tail, apply the α=0.75 interpolation that improved 82M→42.12. Since 84M starts from a stronger base (+0.08 Overall), the resulting endpoint could exceed 42.12.

## Files
- Prompts: `data/improved_fluent_bridge/prompts.jsonl` (750)
- Generation outputs: `data/improved_fluent_bridge/gen_outputs.jsonl`
- Analysis: `data/improved_fluent_bridge/analyze_summary.json`
- Final accepted: `data/improved_fluent_bridge/final_accepted.jsonl` (103 pairs)
- Final summary: `data/improved_fluent_bridge/final_summary.json`
- Review sample: `data/improved_fluent_bridge/final_review_sample.jsonl`
- Repair (failed): `data/improved_fluent_bridge/repair_outputs.jsonl`
