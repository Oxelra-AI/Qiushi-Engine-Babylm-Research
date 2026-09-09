# cluster mechanism construction record Gap Analysis and Strategy Implications

## Score Gap Breakdown (clean-Qwen vs visible leader)

| Column | Clean-Qwen | Leader | Gap | Direction |
|--------|-----------|--------|-----|-----------|
| BLiMP | 66.84 | 67.2 | -0.36 | small deficit |
| Supplement | 62.84 | 56.01 | **+6.83** | ahead |
| EWoK | 50.19 | 56.07 | **-5.88** | largest deficit |
| Entity | 25.76 | 28.45 | -2.69 | deficit |
| COMPS | 51.78 | 53.57 | -1.79 | deficit |
| GlobalPIQA | 36.62 | 39.67 | -3.05 | deficit |
| SuperGLUE | 70.31 | 69.79 | +0.52 | ahead |
| Reading | 7.76 | 5.42 | +2.34 | ahead |
| AoA | 0.0 | 0.0 | 0 | tied |
| **Overall** | **41.34** | **41.8** | **-0.46** | target gap |

## What This Means

To close 0.46 gap (arithmetic mean of 9 columns):
- Need total of 4.1 points improvement across all columns
- Deficit columns: EWoK(-5.88) + GlobalPIQA(-3.05) + Entity(-2.69) + COMPS(-1.79) = 13.41
- Only need 4.1 total → partial EWoK recovery (~4 pts) alone would suffice
- Can afford to lose some Supplement/Reading if EWoK/COMPS gains are large enough

## Leader's Advantage Likely Source

Leader uses FineWeb sentence-level simplification pairs → brings in **new factual content**
from a different corpus distribution (FineWeb). This likely explains their EWoK/COMPS/Entity
advantage: more diverse entity-relation facts during training.

Our Qwen approach generates second FORMS of existing content → gives linguistic variety
(Supplement +6.83, Reading +2.34) but doesn't add new facts.

## Implication for Current Route

The cluster mechanism test (cluster mechanism construction record) directly tests whether natural corpus-internal
complementary evidence can recover EWoK/COMPS/GlobalPIQA. If it works at 2% dose,
a stronger version could close the gap.

If clusters fail, the **highest-leverage pivot** may be:
1. Evidence-visible masking (changes what gets predicted, targets relational tokens)
2. **Qwen fact-generation** (instead of paraphrasing official text, generate complementary
   factual statements about entities appearing in the corpus — this mimics what FineWeb
   simplification pairs do for the leader, but using legal Qwen generation)

Both are legal under BabyLM 2026 rules.

## Critical path to SOTA

1. Run cluster continuation → evaluate → decide within this step
2. If positive: scale to higher dose or full training
3. If negative: implement evidence-visible masking OR Qwen fact-generation
4. Any mechanism that adds +4 to EWoK while preserving other columns → SOTA achieved
