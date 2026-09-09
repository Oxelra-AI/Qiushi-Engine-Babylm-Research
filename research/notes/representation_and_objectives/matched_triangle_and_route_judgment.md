# matched triangle and route judgment — Matched full-compact endpoint, integrated route judgment

No training was launched. This synthesis closed the one unresolved
prerequisite (matched full-compact endpoint on the deletion-arm evaluator), integrated the
newly-available RoBERTa selected endpoint, obtained independent independent_review verification, and
states the single admissible next experiment. It does NOT accept "input-side distribution"
by elimination.

## 1. Matched-evaluator triangle now complete (one evaluator surface)

All three 100M packed-geometry models are now scored on the SAME evaluator
(`pristine_official_coordinate`, no AoA, equal7 = mean of 7 columns).

| model | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | equal7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_compact (compact coupled system route synthesis task s236_t28_tool1) | 66.87 | 63.28 | 53.54 | 27.75 | 51.97 | 35.62 | 8.24 | **43.896** |
| drop_abs (earlier analysis) | 65.64 | 63.73 | 51.51 | 26.72 | 51.60 | 36.665 | 8.195 | 43.437 |
| drop_copied_word (earlier analysis) | 67.32 | 59.84 | 51.93 | 25.67 | 51.58 | 35.09 | 8.115 | 42.792 |

- full_compact − drop_abs = **+0.459 equal7**
- full_compact − drop_copied_word = **+1.104 equal7**
- drop_abs − drop_copied_word = +0.645 equal7 (earlier analysis, secure)

full_compact per-column vs drop_abs: BLiMP +1.23, Supplement −0.45, EWoK +2.03,
Entity +1.03, COMPS +0.37, GlobalPIQA −1.045, Reading +0.045.

**Load-bearing correction to the compact coupled system route synthesis synthesis:** the "intact compact exceeds both
deletion arms" claim is now VERIFIED on a single matched surface (previously it mixed the
historical compact reinvest full eval summary/triangle oom repair and gc preflight surface with the deletion surface). The intact advantage over
drop_abs is real but MODEST (+0.459), concentrated in EWoK/BLiMP/Entity, and offset on
GlobalPIQA. The intact advantage over drop_copied_word is larger (+1.104).

Interpretation: all three arms share the intact compact INPUT distribution and differ only
in which labels carry loss. So the +0.459 that full adds over drop_abs is an upper bound on
the endpoint contribution of *jointly supervising source-absent labels alongside everything
else* — it is not zero, but it is small relative to the +2.4164 compact−repeat effect that
also included the input change. This is direct quantitative evidence that the compact−repeat
gain is dominated by factors present in ALL three arms (the compact input distribution +
dense shared/copied supervision), with source-absent joint target supervision a small
positive add-on, NOT the mediator, and NOT a source of the predeclared relational-EWoK
signature (earlier analysis arm-to-arm relational delta −0.11 pp, interval crossing zero).

## 2. Available RoBERTa Selected Endpoint

`experiments/archive/frontier_consolidation/data/roberta_minimal_selected_eval/minimal_selected_eval_summary.json`
(chck_100M, RoBERTa 8x480, compliant16k, same corpus contrast compact vs repeat):

- compact cheap7 39.149 vs repeat 39.106 → compact−repeat cheap7 **+0.042** (flat)
- cheap6_no_GlobalPIQA **−0.115**, cheap5_no_GlobalPIQA_Reading **−0.194**
- Supplement **−2.21**, Entity −0.41; EWoK +1.05, EWoK+Entity +0.64, COMPS +0.11
- GlobalPIQA +0.985, Reading +0.28
- stable_selected_positive: False; stable_selected_negative: True

Late-band 60M/70M summaries present but no integrated late-band decision; at 60M compact
cheap7 38.326 < repeat 38.859. So the single available RoBERTa selected panel does NOT show
stable cross-architecture transfer on the families that exclude GlobalPIQA/Reading, even
though local source-absent pseudolikelihood channel is positive and semantically
structured (same direction as companion analysis). This is a SECOND local↔global dissociation
(after ordered/scrambled DeBERTa at 40M).

**Consequence for the goal:** the compact-view effect is strong and reproducible in the
DeBERTa masked-denoising coordinate, but the current evidence does NOT establish it as an
architecture-general downstream principle. "Compact views + source-absent targets" therefore
does not yet meet the criterion of a transferable learning principle.

## 3. Integrated mechanism account and supporting comparisons

Best joint account consistent with all facts: **faithful compact rewrites diversify and
densify the contexts around shared (copied) content, and MLM supervision on those
copied/shared targets is the reusable, downstream-consequential channel; source-absent
relational/event targets acquire a genuine but downstream-dissociated local pseudolikelihood
skill.** This positively explains:
- compact >> repeat (repetition adds exposure without context diversification/compression);
- adjbreak recovers much of it (rewrite marginal survives adjacency breaking);
- drop_copied_word is the most damaging removal (removes densest shared-content supervision);
- drop_abs ≈ full on the official surface (source-absent labels are a small joint add-on);
- source-absent words are useful context but not a privileged anchor-conditioning channel
  (abs_minus_function ≈ −0.033, interval crossing zero);
- RoBERTa local channel positive, selected endpoint flat/negative (local ≠ downstream).

Input exposure and copied-target supervision remain CORRELATED across all three arms and are
not yet causally separated. Do not overclaim "input alone."

## 4. Single admissible next experiment (lowest cost first)

independent_review-endorsed ordering:

1. **Prerequisite DONE this step:** matched full_compact endpoint (above). No longer pending.
2. **Next low-cost, saved-checkpoint, no training:** paired cross-realization probe on
   EXISTING checkpoints (full_compact, drop_abs, drop_copied_word, and if loadable repeat,
   adjbreak). For source-shared content words present in both source and compact rewrite,
   mask the identical target under (a) its source context, (b) its genuine compact context,
   (c) a length/position/overlap-matched counterfactual compact context; apply identical
   deterministic masks to all arms; measure paired target NLL and representation agreement
   A_m = cos(h_source, h_rewrite) − cos(h_source, h_counterfactual).
   - Input-exposure/contextual-enrichment signature: drop_abs KEEPS the genuine-rewrite
     advantage over counterfactual/repeat despite never receiving source-absent target loss;
     repeat lacks it; adjbreak retains much; full adds little beyond drop_abs.
   - Joint-complementarity signature: full uniquely exceeds BOTH deletion arms on the
     cross-realization interaction, not merely both copied-supervised models.
   This uses only inference on models we already have; it is the cheapest way to decide
   whether a training-time cooccurrence-decoupling arm is even warranted.
3. **Only if leader analysis and route pivot shows a full-unique interaction:** one same-input target-cooccurrence
   decoupling 100M arm (alternate which target class is supervised per row, matched total
   AND per-category supervised mass CUMULATIVELY and per step, matched update counts, mask
   density, packing, RNG, exposure). decoupled ≈ coupled ⇒ within-sequence cooccurrence
   unnecessary; decoupled < coupled on stable families ⇒ complementarity (conditional on
   compact input). Requires a supplied seed-variance band; a single pair cannot settle a
   modest official difference.

Confounds that would invalidate the probe (must all be matched): target identity/frequency,
capitalization, BPE-piece count, target position, sequence length, visible-token count,
lexical overlap, mask pattern, domain, compression ratio, rewrite quality; counterfactual
context must not contain the target; bootstrap by source document / source–rewrite pair, not
token; never select probe items from model outputs.

## 5. Route rule and prohibitions

- Do NOT re-open explicit source-absent target up-weighting as a SOTA route (100M endpoint
  is a clean negative for mediation).
- Do NOT launch any new 100M training before the saved-checkpoint cross-realization probe
  (leader analysis and route pivot) is run and read.
- Do NOT treat training loss or local pseudolikelihood as downstream evidence.
- Do NOT declare "input-side distribution" the mechanism by elimination; the +0.459
  full−drop_abs gap shows a small but nonzero joint-target contribution that must be
  explained, not dismissed.
- Practical endpoints unchanged and frozen: coherent86 alpha0.75 Overall 42.1210247
  (complete), chck_82M 41.9424812 (reproducible), compact_view_reinvest 42.0867857.
  These exceed the visible 41.80 leader but are practical assets, not the scientific
  endpoint under the earlier analysis mandate.

## Artifacts
- matched full_compact: `experiments/archive/representation_and_objectives/data/full_compact_matched_endpoint_eval/per_target/fullcompact100.json`; task result `experiments/archive/representation_and_objectives/tasks/s236_t28_tool1/result.json`
- arm-to-arm signature: `experiments/archive/representation_and_objectives/data/packed_targetselect_armonly_signature`
- local 100M denoising probe + stratification: `experiments/archive/representation_and_objectives/data/packed_targetselect_denoising_probe_100M`, `.../target_channel_semantic_stratification_100M/`
- context probe: `experiments/archive/representation_and_objectives/data/anchor_context_perturbation_probe`
- RoBERTa selected: `experiments/archive/frontier_consolidation/data/roberta_minimal_selected_eval`
- independent_review verification: `data/external/independent_review01_verifier1_integration.md`
