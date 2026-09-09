# edit route closed: Predeclared edit-operator inventory thresholds

## Purpose
Decide whether the legal compact-view source→rewrite pairs contain dense,
source-absent, full-word, operator-labeled transformations that supply
equivariant training signal for context-conditioned alternative binding.

## Data
All ~22,820 compact-view pairs from `selected_compact_reinvest_pairs.jsonl`.
Each pair has source (original FineWeb) and rewrite (Qwen3.5-9B compact).
Word-level diff identifies 1:1 substitutions (same syntactic slot).

## Key metric: source-absent full-word content-word 1:1 substitutions (SAFW)
- "1:1": exactly one source word replaced by one rewrite word in alignment
- "full-word": rewrite word ≥3 alphabetic chars, not punctuation/number
- "content-word": not a closed-class function word
- "source-absent": rewrite word does NOT appear anywhere in the source sentence
- Also measure stem-level leakage: does a ≥4-char prefix of the rewrite word
  appear as a prefix of any source word? (approximate morphological copy)

## Route decision thresholds

### VIABLE → proceed to paired-context equivariant mechanism design
All of:
- ≥2000 SAFW substitutions
- ≥500 unique (source_word, rewrite_word) pairs
- ≥10% of all pairs contribute at least one SAFW substitution
- <30% of SAFW have stem-level source leakage
- Bidirectional density: ≥50 word pairs appearing in both A→B and B→A direction

### MARGINAL → needs cheap model-based plausibility readout before commit
- 500-2000 SAFW substitutions, ≥100 unique pairs, <50% stem leakage
- Next step: frozen-model readout of whether both alternatives are plausible
  in both contexts (both have reasonable probability)

### CLOSED → edit route not viable; move immediately to different mechanism
Any of:
- <500 SAFW substitutions
- <50 unique word pairs
- >60% stem-level leakage
- >70% of SAFW are actually function/closed-class

## If CLOSED, the edit-derived equivariant route is closed and the research
moves to a representation-forming mechanism that does not depend on finding
or manufacturing paired-context examples from the corpus. Candidates:
1. Context-conditioned candidate metric loss initialized on mature LM
2. Masked prediction with context perturbation
3. Entity-role binding from legal corpus structure
4. Direct SuperGLUE/EWoK-targeted finetuning strategies

## Binding to scale1.75 endpoint
Any viable mechanism must be expressible as either:
(a) from-beginning training on the legal 10M corpus, or
(b) continuation from scale1.75 80M checkpoint preserving broad gains.
The target is crossing 41.80 Overall, currently at 41.7716.
