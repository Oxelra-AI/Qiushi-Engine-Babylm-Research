# Compact pilot analysis and relation-learning synthesis

## Pilot Quality Summary (512 pairs, Qwen3.5-9B, temp=0.0)

### Word Savings
- Effective compression: 0.484 (compact is 48.4% of current rewrite length)
- Practical pass rate: 92.6% (474/512, gates: savings + min 8 words + number preservation)
- Mean savings per passed pair: 19.7 words
- Total saved from 474 passed: 9,317 words
- Extrapolation to 26,567 manifest: ~24,595 passed, ~483K saved words, ~3,021 top-up rows

### Entity/Number Preservation
- Raw entity recall 0.75 is inflated failure: entity_source includes common words
  (e.g., "yeah", "we've", "soda", "supper"), not just proper names
- Number preservation: 0.93 mean, 89.5% perfect recall (95/512 have numbers)
- The noisy entity metric should not gate the production run; number preservation gates suffice

### Structural Markers (concerning but interpretable)
- Negation: 38.3% preserved (median 0.0) 
- Modality: 35.1% preserved
- Causal/temporal: 36.3% preserved
- These are low BUT the compact is a full regeneration, not a trim. Qwen3.5-9B
  restructures the sentence rather than removing words. Many structural markers
  are implicit in the compact form (e.g., "cannot afford to renovate" → compact
  preserves the inability meaning without explicit negation words)

### Expression Diversity (key scientific observation)
- Compact-original Jaccard: 0.476 (moderate overlap, different expression)
- Current-original Jaccard: 0.594 (current rewrite shares more with original)
- Compact-current Jaccard: 0.365 (compact is quite different from current rewrite)

**The compact views create MORE diverse expression of the same content** than the
current rewrites. If what matters for learning is cross-span correspondence between
different expressions of the same meaning, compact views may be a STRONGER signal
than the current near-parity rewrites.

## Connection to the correspondence finding

The earlier SHUF/ALN dose comparison reports:
- Same-text wrong correspondence produces OPPOSITE column movements from correct pairing
- The learning signal depends on CORRESPONDENCE, not text length
- Dose curve shows a KNEE: more compact restatement saturates

Our compact views are NOT redundant restatement of current rewrites (Jaccard 0.365).
They are independently generated second views with fresh lexical choices. This means:
1. They should not saturate the same way as dose increments of similar rewrites
2. They provide genuinely new correspondence evidence for the MLM learner
3. The diversity (0.476 compact-orig vs 0.594 current-orig) increases the information
   content per word of supervised correspondence

## Practical Route Decision

Based on the pilot:
1. Full generation quality is sufficient for a training pilot (92.6% practical pass rate)
2. Savings are larger than expected (0.484 vs target 0.60 ratio)
3. More words saved than can be reinvested through top-up rows alone
4. The compact views provide genuinely different correspondence, not just shorter text

The proposed next comparison required full 26,567-pair generation, quality filtering, and then a 4-arm training comparison. These were not completed results in this note.

## Binding comparison across substrates

Answer-only specialist on relation-first rows: 28/30 held four-condition (~93%)
Balanced answer-credit on natural packets: 44/200 gating (~22%)

Interpretation: Cleaner rows with explicit entity-value mappings produce much higher
selector success, but the answer-only specialist is incompatible with broad competence
(Cheap7 43.35 vs coherent86 44.56). the chck_82M binding candidate tests whether
gating can coexist with broad competence using the coherent86 architecture.

## Measurements pending at the time
- Full generation: 26,567 pairs, estimated ~4-5 hours; incomplete.
- Reference tail Cheap7 evaluation: nearing completion.
- Bridge Cheap7 suite: answer_allocation checkpoints remained to be evaluated.
