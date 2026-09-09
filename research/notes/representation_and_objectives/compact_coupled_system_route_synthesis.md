# compact coupled system route synthesis — Route synthesis after the 100M packed target-selective readout

No H100 training was launched for this synthesis. Purpose: decide how
the compact-view mechanism line must be reformulated now that the source-absent-label
mediation hypothesis has a clean 100M negative, and specify at most one premise-changing
experiment to run after mature RoBERTa selected result arrives.

## What earlier analysis–235 actually established (100M packed intervention, DeBERTa 8x480 legal16k)

Three 100M models, identical init SHA `f13f1f85…509`, identical stream/order/tokenizer/
optimizer/LR/WWM/CUDA-RNG, differing only in which already-corrupted target labels
contribute loss:

- `full_compact` (historical intact compact_view_reinvest trajectory; equal7 ~44.29 in triangle)
- `drop_abs` = source-absent compact-content labels removed (48,105 BPE deleted)
- `drop_copied_word` = matched whole-word copied-content labels removed (48,387 BPE deleted)

### Official-surface arm-to-arm (earlier analysis)
`drop_abs − drop_copied_word` = **+0.645 equal7**:
BLiMP −1.68, Supplement **+3.89**, EWoK −0.42, Entity +1.05, COMPS +0.02,
GlobalPIQA **+1.575**, Reading +0.08.

So removing source-absent labels did NOT hurt the official surface. Removing matched
copied-content labels hurt MORE (drop_copied is worse overall). The predeclared correctness transition analysis
relational-EWoK signature is ABSENT in the arm comparison: relational-domain delta
−0.11 pp, net −2 items, bootstrap interval crosses zero; historical-vector cosine ~0.

**Decisive conclusion:** within this DeBERTa/FineWeb packed coordinate, the source-absent
compact-content target labels are NOT the mediator of the historical compact-view
Supplement/relational-EWoK advantage. The target selective source absent result–227 source-absent target channel is
real locally but does not carry official competence at 100M. This is a clean negative for
the "source-absent labels are the mechanism" hypothesis.

### Local denoising (earlier analysis/235, train_fixed_probe, piece-weighted loss on source_absent_content)
- full_compact 5.001, drop_copied_word 4.994, drop_abs higher (worst)
- `drop_abs − drop_copied_word` on source_absent_content = **+0.650** train-fixed,
  +0.386 source-disjoint quality, +0.359 doc-disjoint accepted (all strongly positive)
- retained_content and function_other do NOT share this positive movement (near-zero/neg)
- earlier analysis stratification: local channel concentrated in NOVEL relational/event words
  (+0.96 train relational/event vs +0.62 other), reverses on names/numbers.

So the source-absent labels do teach a strong, semantically-structured LOCAL compact-side
pseudolikelihood channel — but that local skill is dissociated from official competence.
This mirrors ordered/scrambled 40M result (large local SA channel, worse selected
surface) — a recurring local↔global dissociation.

### The subtle fact that reframes the mechanism
On source-absent local denoising, `drop_copied_word` (4.994) is even slightly BETTER than
intact `full_compact` (5.001), yet `full_compact` is the strongest official model of the
three. And on the official surface, copied-target deletion is the more damaging removal.
Neither single target population reproduces the full model:
- removing copied labels → worse official surface (esp. BLiMP, Supplement)
- removing source-absent labels → worse local SA denoising, but FINE / slightly better official

=> Compact views behave as a COUPLED target system, not a sum of separable channels.

## Revised Hypothesis

Do not read this as "hard source-absent labels should be suppressed." Read it as:
- **copied-content targets** anchor reusable, source-shared semantics and are the more
  load-bearing population for broad official competence (BLiMP/Supplement) in this coordinate;
- **novel relational/event source-absent targets** create a specialized abstractive
  local channel that is semantically aligned with Supplement/relational-EWoK but does not,
  by itself, convert into official competence when isolated;
- the intact compact trajectory exceeds BOTH deletions, so the operative property is the
  INTERACTION — how these two target populations together convert semantic density into
  transfer — not either alone.

The unresolved scientific question is therefore the *coupling*: does the compact advantage
come from (a) target complementarity (copied anchors + novel abstractive targets trained
jointly on the same sequences), or (b) an input-side compact-distribution effect (the model
simply seeing high-density faithful compact text as input, largely independent of which
targets carry loss)?

## Distinguishing (a) target complementarity vs (b) input-side distribution — design space

Key observation: the target-selective arms only changed WHICH labels contribute loss; they
kept the SAME input text (intact compact sequences) in all three arms. So all three arms
already share the input-side compact distribution. That means:

- The fact that drop_abs ≈ full on the official surface (drop_abs 43.44 vs full triangle
  equal7 ~44.29 — full is still ahead but the isolated-label removal is not catastrophic)
  is consistent with a LARGE input-side component: even with source-absent labels removed,
  the model still trains on intact compact input and stays close to full official surface.
- The catastrophic direction is removing copied labels, which removes the densest
  source-shared supervision.

This suggests the next premise-changing experiment should test the INPUT-side hypothesis
directly, because the target-side has been substantially explored and the biggest official
movement came from copied-target removal, not source-absent-target removal.

### Candidate premise-changing experiment (choose one, after RoBERTa selected result)

**E1 — Input-distribution vs target-coupling separation (single 100M contrast).**
Two matched 100M arms sharing the historical compact input stream, differing only in a
minimal, theory-motivated change to the target coupling:
- Arm A (coupling-preserved reference): full compact (already have it, equal7 ~44.29).
- Arm B (decoupled targets, same inputs): train with the SAME intact compact input
  sequences, but corrupt/select targets so copied and source-absent targets never co-occur
  in the same sequence's loss (e.g. alternate which population is supervised per row), while
  keeping total supervised token mass and per-category mass matched to full.
  If Arm B ≈ full official surface → coupling within a sequence is not required; input
  distribution + total supervision dominate. If Arm B < full → within-sequence target
  coupling matters, supporting the complementarity hypothesis.

**E2 — Input-side compact vs matched-noncompact input, targets held (already partly closed).**
The triangle (compact vs repeat vs adjbreak) is the input-side test and already shows
compact >> repeat. adjbreak (rewrite marginal, source adjacency broken) recovers much of
the gain. This says the rewrite marginal / high-density input is a large driver. Rather
than re-run, USE this: the triangle already argues input-side density is a major component.
Combined with earlier analysis (target-selective removal is mild for source-absent, severe for
copied), the leading integrated hypothesis is: **the compact advantage is primarily an
input-side high-density faithful-rewrite distribution effect, amplified by dense
source-shared (copied) supervision, with the novel relational/event targets a
semantically-aligned but non-load-bearing byproduct at the official surface.**

If that integrated hypothesis holds, the transferable principle is NOT "add source-absent
targets" but "train on high-density faithful compact rewrites of the same content, which
concentrate reusable source-shared supervision per token." That is an input/data principle,
architecture-agnostic in spirit, and directly testable in RoBERTa coordinate.

## Route rule (what decides the next expensive step)

Integrate two results before any new 100M run:
1. This endpoint dissociation (copied-target removal is the damaging one; source-absent
   removal is mild; intact > both).
2. mature RoBERTa compact-vs-repeat SELECTED official result (`s211_t37_tool1`),
   which tests whether the INPUT-side compact distribution transfers across architecture on
   stable selected families (not local NLL, not GlobalPIQA/Reading alone).

- If RoBERTa compact > repeat on stable selected families → input-side high-density
  faithful-rewrite distribution is a cross-architecture principle. Then the strongest
  single experiment is a decisive input-side data ablation isolating "faithful high-density
  rewrite" from confounds (density, coverage, lexical recurrence), NOT a target-weighting run.
- If RoBERTa is neutral/negative on stable families while local SA channel is positive
  → another local↔global split; the compact effect is coordinate-specific and we bound it,
  then attack the coupling question with E1 as the single admissible 100M run.

## Do NOT
- Do not re-open explicit source-absent target up-weighting as a SOTA route: 100M endpoint
  says it is not the mediator.
- Do not launch any new 100M run before selected RoBERTa result is integrated.
- Do not treat local pseudolikelihood or training loss as downstream evidence.

## Practical endpoints unchanged
coherent86 alpha0.75 Overall 42.1210247 (complete), chck_82M 41.9424812 (reproducible),
compact_view_reinvest 42.0867857 (compliant). These remain frozen; not the scientific endpoint.

## compact coupled system route synthesis context-perturbation probe on saved 100M checkpoints (completed)

independent_review proposed that source-absent compact words might matter not as targets but as visible
context conditioning copied/shared anchor prediction ("anchor-conditioned abstraction").
The compact coupled system route synthesis saved-model probe tested this directly and **weakens that clean hypothesis**.

Probe: `experiments/archive/representation_and_objectives/scripts/anchor_context_perturbation_probe.py`
Artifacts: `experiments/archive/representation_and_objectives/data/anchor_context_perturbation_probe`
(`anchor_context_perturbation_probe.json`, `.md`, event/loss JSONLs). 3,596 retained-anchor
events across train-fixed and held-out compact pairs; three saved 100M checkpoints
(full_compact, drop_abs, drop_copied_word).

Result:
- Masking visible source-absent compact-content context DOES raise copied/retained-anchor
  loss: full +0.404639 nats, drop_abs +0.360792, drop_copied_word +0.391266
  (all bootstrap >0). So the models use source-absent compact words as context.
- But matched function/other context perturbation raises anchor loss by nearly the same
  amount: full +0.338483, drop_abs +0.298764, drop_copied +0.312859. On the exact
  BPE-matched subset, `abs_minus_function` is ~ −0.033 with intervals crossing zero or
  negative (fraction_gt0 ~0.16-0.18). Thus source-absent context is **not a special
  anchor-conditioning signal** relative to matched generic context.
- Matched copied-context perturbation is slightly WEAKER than source-absent context:
  `abs_minus_copied` ~ +0.052 (full), +0.029 (drop_abs), +0.051 (drop_copied); only the
  source-disjoint full subset is cleanly positive. So anchor prediction is not dominated
  by copied lexical anchors either.
- Subgroup nuance: within full, contexts containing relational/event source-absent words
  have larger abs sensitivity than non-rel contexts (0.453 vs 0.392), but the matched
  function control also behaves similarly, so the specific semantic advantage is not
  established.
- Between arms, `full_minus_drop_abs` source-absent context sensitivity is only
  +0.043848 nats (small), and `full_minus_drop_copied` is +0.013374 (weak). This is not a
  large coupling signal.

Scientific consequence: the novel source-absent words act like ordinary useful context,
not like a uniquely load-bearing semantic conditioning channel for copied anchors. That,
combined with the earlier analysis endpoint negative for source-absent target labels and the
triangle oom repair and gc preflight triangle (compact >> repeat, adjbreak recovers most), strengthens the leading
account toward an **input-side high-density faithful-rewrite distribution effect** plus
dense source-shared (copied) supervision, and away from a target-specific or
context-specific novel-abstraction coupling mechanism.

## Remaining matched-score caveat

The claim "intact compact exceeds both deletion arms" currently mixes the historical
compact score surface (compact reinvest full eval summary full / triangle oom repair and gc preflight no-AoA equal7) with the newly evaluated
deletion arms. The arm-to-arm source-absent mediation negative is secure, but the exact
magnitude of full-minus-deletion on the CURRENT evaluator invocation is not yet measured.
A matched evaluation of the full compact 100M endpoint with the same deletion-arm evaluator
is running as managed background task `s236_t28_tool1`; do not treat the historical
full-vs-deletion gaps as precise until that result is integrated.
