# frozen transfer test and route decision — Frozen-evidence transfer test and route decision after the compact-view factorial closure

No training or model evaluation was launched. This analysis converts the factorial interaction item review
negative RoBERTa factorial into a route-portfolio decision, using the strategist's exact
question: does a concrete DeBERTa-specific compact-view signature predict *both* the large
DeBERTa compact gain *and* the RoBERTa/GPT failures from frozen evidence? If yes, one
decisive test is warranted; if not, compact views are demoted from the central transferable
route and the stronger-principle portfolio reopens.

## What was computed (frozen, no GPU)

Script `experiments/archive/representation_and_objectives/scripts/frozen_compact_transfer_predictor.py`;
result `experiments/archive/representation_and_objectives/data/frozen_compact_transfer_predictor`.

Joined **105,778** official item ids that are present in ALL of:
- DeBERTa triangle item correctness (compact_view / compact_repeat / adjbreak, 100M, from the
  correctness transition analysis prediction files) → per-item `d_view_minus_repeat`, `d_view_minus_adjbreak`,
  `d_adjbreak_minus_repeat`;
- RoBERTa compact-vs-repeat per-item movement (`roberta100_selected_prediction_movement`);
- RoBERTa HS/LS/HD/LD factorial per-item interaction (`factorial_item_interaction_terminal`).
Family-level causal-GPT reciprocity (`reciprocal_minus_oneway`) added as directional context.

The test: if the compact-view effect is one architecture-general mechanism, then the DeBERTa
items/subtasks that compact *helps* should be the same items/subtasks that move coherently
positive in RoBERTa compact-repeat and in the RoBERTa contextual-anchor interaction. A
frozen predictor built on DeBERTa gains should carry sign to the other coordinates.

## Result: the DeBERTa compact signature ANTI-predicts cross-architecture movement

Conditional item means, stable non-BLiMP pool (Supplement+EWoK+COMPS, n=92,378), grouped by
the DeBERTa view−repeat item outcome:

| DeBERTa view−repeat | RoBERTa compact−repeat (mean pp) | RoBERTa factorial interaction (mean pp) | n |
|---|---:|---:|---:|
| −1 (compact lost) | −0.061 | **+2.746** | 16,386 |
| 0 (no change) | +0.177 | +1.792 | 59,324 |
| +1 (compact gained) | **−0.510** | **−0.150** | 16,668 |

- DeBERTa-gain items → RoBERTa compact−repeat gain-minus-loss coefficient **−0.449 pp**.
- DeBERTa-gain items → RoBERTa factorial interaction gain-minus-loss coefficient **−2.896 pp**.

The items where DeBERTa compact views help are precisely where the RoBERTa contextual-anchor
interaction is *lowest*, and where RoBERTa compact-repeat is *most negative*. This is the
opposite of a shared mechanism.

Subtask-vector correlations (87 common subtasks) are weak or negative:
- DeBERTa view−repeat vs RoBERTa compact−repeat: pearson **+0.060** (weighted +0.308);
- DeBERTa view−adjbreak vs RoBERTa factorial interaction: pearson **−0.295**;
- DeBERTa adjbreak−repeat vs RoBERTa compact−repeat: pearson **−0.186**;
- DeBERTa view−repeat vs RoBERTa factorial interaction: pearson **−0.193**.

Family table (item-weighted pp):

| column | n | D view−rep | D view−adj | D adj−rep | R compact−rep | R factorial |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP | 13,400 | −0.06 | −1.62 | +1.56 | +0.40 | +2.45 |
| COMPS | 91,028 | +0.27 | +0.18 | +0.09 | −0.01 | +1.56 |
| EWoK | 1,100 | +2.18 | +4.64 | −2.45 | +2.91 | +5.73 |
| Supplement | 250 | +4.40 | +3.60 | +0.80 | **−3.60** | +0.80 |

Supplement is the single DeBERTa family with genuine retained-plus-new competence
(correctness transition analysis: +4.4, 90.3% retention). In RoBERTa compact-repeat it is **−3.60**. The one place
the DeBERTa mechanism is cleanest is exactly where RoBERTa reverses it. EWoK is the only
family with same-sign positivity across coordinates, but it is 1,100 items with ~30% answer
churn (correctness transition analysis) and the RoBERTa factorial EWoK item interaction was −1.14 pp at the correctly
item-weighted level (factorial interaction item review), so the aggregate EWoK agreement is not a stable relational
signal.

Family-level causal-GPT reciprocity is near-zero/negative on exactly the DeBERTa-gain
families: EWoK reciprocal−oneway −0.425, COMPS −0.100, Supplement +0.075, BLiMP +0.125.

## Scientific conclusion

There is **no** concrete frozen DeBERTa compact-view signature that predicts the RoBERTa or
causal-GPT movement. The item-level association is not merely weak — it is negative on the
stable relational families. This is a real scientific result: the compact-view advantage is a
**coordinate-specific** property of DeBERTa masked denoising (relative/disentangled attention +
WWM + this FineWeb compact geometry), not an architecture-general data-efficient learning
principle. Three independent coordinates now agree — DeBERTa extractive stable-family
negative (earlier analysis), RoBERTa compact-repeat selected negative (crossview v2 20M result), and this
frozen anti-correlation — that "faithful compact rewrites" do not transfer as a downstream
principle across encoder families.

What remains true and reproducible:
1. In DeBERTa masked denoising, compact_view − compact_repeat = **+2.4164 equal7** (earlier analysis),
   beyond the seed band; a real coordinate-specific finding.
2. A cross-realization representation tracks compact rewrite INPUT distribution in all arms
   including repeat-trained (cross realization probe result); real, but input-side and not transfer-carrying.
3. A local source-absent denoising channel, dissociated from downstream competence
   (target selective source absent result); real but local.

## Route decision (this changes the portfolio, not just the interpretation)

- **Demote compact views from the central transferable-principle route.** Per the
  expensive-work admission rule, do NOT spend GPU on further compact-view decomposition
  variants (chck_60M/80M of the factorial, seed replication, new content-density/coverage/
  recurrence/naturalness arms, target-class-decoupling arms). No cheaper-or-equal evidence
  could now flip the transfer verdict, and the frozen predictor shows the DeBERTa signature
  does not even point at the same items the other coordinates move.
- **Preserve compact views as a coordinate-specific practical asset.** The compliant DeBERTa
  practical endpoints stay frozen: coherent86 α0.75 Overall 42.1210247099666, chck_82M
  41.942481167385985, compact_view_reinvest 42.0867857191. These already exceed the visible
  41.80 public leader and remain the submission-facing fallback. Submission is outside this analysis.
- **Reopen the stronger-principle portfolio** with the accumulated diagnosis as the anchor,
  not as a fresh guess. The most-supported unresolved deficit across these studies is
  the persistent **context-conditioned binding / main-path compositional update** failure
  (ewok contrast preservation anatomy, 195–199): models learn local compatibility and affected-entity identity but
  never bind event-result state back to the entity and expose it to output computation. This
  is architecture-level, it is the deficit that the coordinate-specific compact data effect
  never repaired, and it is a candidate for a transferable inductive-bias/objective principle
  rather than a data-surface trick.
- The next Explore work must compare serious candidate transferable-principle routes (e.g.
  a main-path compositional-update mechanism / objective; a credit-assignment or
  representation-geometry principle; a learning-signal principle) against this diagnosis and
  against the frozen negative evidence, and choose the one with the best chance of an
  architecture-general sample-efficiency gain, with complementary tests of the mechanism and its transfer.

## Artifacts
- `scripts/frozen_compact_transfer_predictor.py`
- `data/frozen_compact_transfer_predictor/transfer_predictor_summary.json`
- `data/frozen_compact_transfer_predictor/transfer_predictor_summary.md`
- `data/frozen_compact_transfer_predictor/subtask_alignment_table.csv`
- Prior: `notes/factorial_interaction_item_review.md`, `notes/cross_realization_probe_result.md`,
  `notes/matched_triangle_and_route_judgment.md`, `notes/extractive_result_review_and_route.md`.

## Addendum: independent_review-requested component-conditioned refinement

independent_review verifier agreed that frozen transfer test and route decision demotes intact compact views as the central transferable route, but asked for one remaining no-training refinement: test correctness transition analysis's components separately after baseline/subtask conditioning:

- A = `adjbreak − repeat` (rewrite marginal / information density)
- B = `view − adjbreak` (source-own adjacency / correspondence)
- T = `view − repeat` (total compact effect)

I implemented the fast frozen analysis in `experiments/archive/representation_and_objectives/scripts/component_transfer_conditioned_fast.py`; result directory `experiments/archive/representation_and_objectives/data/component_transfer_conditioned_fast`. It residualizes within column/subtask plus relevant DeBERTa and RoBERTa baseline correctness groups, and uses a descriptive subtask-cluster bootstrap over fixed residual sums.

Key stable non-BLiMP results (Supplement+EWoK+COMPS):

| DeBERTa component → RoBERTa outcome | residual slope pp | subtask-bootstrap 95% interval | reading |
|---|---:|---:|---|
| A → RoBERTa compact−repeat | +3.97 | [−2.22, +7.56] | weak/unstable; not reliable transfer |
| A → RoBERTa factorial interaction | **−3.50** | [−10.61, +0.15] | opposite to contextual-anchor route |
| B → RoBERTa compact−repeat | +7.44 | [+5.09, +8.50] | positive but COMPS-dominated and not mediator-aligned |
| B → RoBERTa factorial interaction | −0.58 | [−1.33, +1.10] | no source-own-anchor mediator |
| T → RoBERTa compact−repeat | +7.36 | [+4.74, +8.74] | positive residual slope dominated by COMPS, despite raw D-gain items being worse |
| T → RoBERTa factorial interaction | −0.83 | [−1.58, +1.02] | no contextual-transfer mechanism |

Important family detail: Supplement, the cleanest DeBERTa retained-plus-new family, remains negative after conditioning: A→RoBERTa compact−repeat slope −5.76 with interval [−10.71, −2.33], B→compact−repeat −5.90, and T→compact−repeat −6.98. EWoK component slopes are positive but intervals are wide and the factorial interaction item review item-weighted RoBERTa factorial verdict on EWoK+Entity was flat/negative. COMPS drives the positive residual compact-repeat slopes, but COMPS has already shown large internal reversals (`wugs_dist_before` vs `wugs_dist_in_between`) and does not carry the relation/state mechanism.

Refined conclusion: there may be a narrow DeBERTa/RoBERTa COMPS-like distributional response to compact/rewrite components after strong baseline conditioning, but the component test does not rehabilitate compact views as a central transferable principle. The mediator that would matter for the scientific goal — source-own contextual/relational transfer on Supplement/EWoK/Entity/COMPS without BLiMP/GlobalPIQA artifacts — still fails. Compact views remain a practical DeBERTa recipe and perhaps a limited distributional data asset, not the main route.

independent_review files:
- `data/external/independent_review01_generator1_integration.md`
- `data/external/independent_review01_verifier1_integration.md`
