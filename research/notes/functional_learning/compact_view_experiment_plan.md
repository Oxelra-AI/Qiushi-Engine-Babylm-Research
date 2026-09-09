# compact pilot analysis Compact-View Reinvestment Experiment Plan

## Scientific question
Can we improve coherent86's Overall 42.1210 by shortening redundant Qwen second views
and reinvesting saved words in new source coverage?

## Key scientific distinction
Redundancy in expressed meaning ≠ redundancy for learning. A shortened view may preserve
every entity/number/proposition yet remove the alternative-expression or cross-span
evidence through which the model learns correspondences. Conversely, it may retain those
correspondences while eliminating cheaply predicted repetition.

## Pilot phase (compact pilot analysis, this step)
1. Generate compact rewrites for 512 stratified pairs via Qwen3.5-9B
2. Measure preservation: entity, number, negation, modality, causal/temporal, role, 
   and expression diversity (compact-vs-original Jaccard)
3. Audit: quality-filter outputs, count realized savings, estimate full-set yield
4. Decision gate: proceed to training only if:
   - Entity recall ≥ 0.95 for pairs with entities
   - Number recall ≥ 0.95 for pairs with numbers
   - Negation preservation ≥ 0.80
   - Compact-original overlap distinctly different from 1.0 (still alternative expression)
   - Realized savings ≥ 80% of expected

## Training comparison (if pilot passes)
Four arms, all starting from coherent86 parent, same trainer, same 354 macro-updates:

| Arm | Data | Purpose |
|---|---|---|
| reference_tail | Current 10M pool (no changes) | Same-trainer baseline |
| compact_only | Compact pairs, no reinvestment | Isolate compaction effect |
| compact_reinvest | Compact + top-up official rows | Full compact+reinvest |
| compact_neutral | Compact + neutral source-matched | Control for reinvestment |

Evaluation: full Cheap7 screen + Entity + relation held probes for all arms.

The comparison separates:
- compact_only vs reference: does compaction alone help or hurt?
- compact_reinvest vs compact_only: does reinvestment add value?
- compact_reinvest vs compact_neutral: new source vs. recurrence?
- All vs coherent86 parent: practical improvement?

## Training entry point considerations
- coherent86 continuation tests suffix adaptation (practical route)
- legal replay from earlier checkpoint tests representation formation (scientific route)
- Start with continuation; if promising, test formation route separately

## Inherited assets
- Compaction manifest: 26,567 pairs, expected 233,903 saved words (full scenario)
- Top-up rows: 1,461 rows / 233,680 words (high-affordance official rows)
- Neutral matched: 1,461 rows / 233,680 words (source-distribution-matched control)
- Clean-Qwen pool: 64,381 rows / 10M words with pair boundaries preserved
- Trusted trainer: corrected_bridge_trainer.py (macro-update accounting)

## Experimental status at the time
- Compact generation: 512 prompts, Qwen3.5-9B; in progress.
- Reference tail Cheap7 evaluation: in progress.
- Bridge Cheap7 suite: in progress; ordinary_wwm completed with 44.272.

## Bridge Cheap7 interim results
ordinary_wwm_update0354 equal_valid_mean: 44.272 (coherent86: 44.564, historical ref: 44.308)
This is near the reference level, meaning the bridge data overlay did not substantially
harm broad competence even though it failed to acquire relation selection.
