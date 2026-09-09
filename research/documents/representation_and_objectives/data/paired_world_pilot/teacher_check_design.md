# Teacher-Check Design (paired world pilot result)

## Purpose
Verify that approved teacher models (Qwen3.5-9B, Llama3.1-8B-Instruct)
can reliably read role assignment from diverse score-ablated contexts.

## Procedure
1. Sample ~200 families (100 train + 100 held).
2. For each family, present 4 score-ablated NLI queries:
   - "Context: {c1_ablated}. Hypothesis: {A} defeated {B}. Answer ENTAILED or NOT_ENTAILED."
   - (repeat for B defeated A, and for C2)
3. Each query → Qwen3.5-9B and Llama3.1-8B-Instruct independently.
4. Score:
   - Expected-label accuracy per teacher (target ≥0.95)
   - Cross-teacher agreement (target ≥0.95)
   - Train-template vs held-template comparison

## Pass Criteria
- Both teachers ≥0.95 expected-label accuracy on score-ablated queries.
- Cross-teacher agreement ≥0.95.
- No significant accuracy drop on held templates vs train templates.
- If any criterion fails, investigate failure modes before teacher-
  supported student training.

## Notes
- This is the CHEAPEST reliable test before any model training.
- Score-visible queries serve as a verification ceiling (near 1.0).
- Do NOT use score-visible-only data as learning signal.
- Results feed into the broader route decision: if teachers can read
  role assignment from diverse templates, the substrate is viable for
  student distillation probes.
