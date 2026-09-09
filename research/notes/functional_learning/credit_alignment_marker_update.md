# accumulated mechanism synthesis/024 consolidated update: effective relation-aligned credit and a candidate query-match marker

## What changed in this Execute step

The previous body-objective ablation result showed that interleaving answer-only and full objectives preserved held-symbol binding much better than an abrupt switch to full next-token training. The comparison audit found that the proposed constant-weight controls were not the right comparison because the implemented loss normalizes over active positions. The subsequent experiment corrected the comparison, ran the matched static objective, separated allocation from generic answer exposure, separated context-dose reduction from relation-compatible answer credit, and added a first activation-level probe.

## 1. Correct static match for answer/full alternation

Let \(A\) be answer-position loss and \(C\) mean loss over the 15 non-answer context positions. Static context weight \(w\) implements

\[
L_w = \frac{A+15wC}{1+15w}.
\]

Equal alternation between answer-only and full objectives has nominal average

\[
\frac12 A + \frac12\frac{A+15C}{16}=\frac{17A+15C}{32},
\]

so the matched static objective is \(w=1/17\), not \(w=0.5\).

Artifacts:

- modified script: `experiments/archive/functional_learning/scripts/calibration_trajectory.py`
- static matched data: `experiments/archive/functional_learning/data/revision_021b_credit_matched/results.json`
- analysis: `research/notes/functional_learning/021b_credit_matched_analysis.md`
- figure: `experiments/archive/functional_learning/figures/revision_021b_credit_matched_unified.png`

Endpoint means across seeds 42/43/100:

| arm | A/C coefficients | held top-4 | held B | held selectivity | train top-4 | context CE |
|---|---:|---:|---:|---:|---:|---:|
| direct full | 0.062/0.938 | 0.418 | +3.111 | +0.256 | 0.921 | 1.233 |
| interleaved answer/full | 0.531/0.469 | 0.746 | +7.765 | +0.665 | 1.000 | 1.316 |
| static \(w=1/17\) | 0.531/0.469 | 0.694 | +7.548 | +0.574 | 1.000 | 1.238 |
| static \(w=0.1\) | 0.400/0.600 | 0.652 | +6.754 | +0.515 | 1.000 | 1.234 |
| static \(w=0.5\) | 0.118/0.882 | 0.445 | +4.011 | +0.307 | 0.992 | 1.234 |

The matched static arm recovers most of interleaving's advantage over direct full while learning context prediction to the same CE as direct full. Thus temporal alternation is useful but not necessary for most of the preservation effect. The core variable is effective answer/context credit allocation.

## 2. Generic answer/RWT exposure is insufficient

Artifacts:

- script: `experiments/archive/functional_learning/scripts/answer_credit_alignment.py`
- data: `experiments/archive/functional_learning/data/answer_credit_alignment/results.json`
- note: `research/notes/functional_learning/answer_credit_alignment.md`

The control keeps answer position, RWT answer family, query-first contexts, absence of held entities in continuation, and either static \(w=1/17\) or interleaving format. It changes only the answer target to a bag-independent random context attribute, removing query-conditioned relation alignment.

Final means:

| arm | held top-4 | held B | held selectivity | train top-4 | context CE | answer CE |
|---|---:|---:|---:|---:|---:|---:|
| bag-independent static \(w=1/17\) | 0.225 | -0.040 | -0.007 | 0.261 | 1.256 | 1.471 |
| bag-independent interleaved | 0.219 | -0.071 | -0.016 | 0.240 | 1.337 | 1.610 |

This rejects generic answer-position pressure, RWT-family rehearsal, and formal alternation as sufficient explanations.

## 3. Reduced context dose alone is insufficient; learnable misaligned answer credit is insufficient

Artifacts:

- script: `experiments/archive/functional_learning/scripts/alignment_vs_dose.py`
- data: `experiments/archive/functional_learning/data/alignment_vs_dose/results.json`
- analysis: `research/notes/functional_learning/alignment_vs_dose_analysis.md`

Two stronger controls:

- `context_only_lr_half`: no answer loss, half learning rate, context CE learned to 1.237; held binding collapses to top-4 0.272 and held-B -0.144.
- `slot0_static_1over17`: deterministic visible first-attribute target at matched \(w=1/17\), answer CE \(3.6\times10^{-5}\), context CE 1.232; query-conditioned held binding collapses to top-4 0.238 and held-B +0.578.

These controls show that the successful bound static \(w=1/17\) arm is not merely weaker context training or generic learnable answer training. The answer credit must be compatible with the query-conditioned relation.

## 4. Gradient pressure at the preparation point

Artifacts:

- script: `experiments/archive/functional_learning/scripts/gradient_pressure.py`
- data: `experiments/archive/functional_learning/data/gradient_pressure/results.json`
- note: `research/notes/functional_learning/gradient_pressure.md`

At the answer-only preparation checkpoint, bound answer loss is already tiny (0.024) while context loss is huge (33.638). Body gradient norms are about 1.16 for bound answer and 21.36 for context. The matched \(w=1/17\) gradient is still context-direction initially, but with about half the normalized context coefficient of direct full.

One-epoch effects:

| condition | Δheld top-4 | Δheld B | Δtrain top-4 |
|---|---:|---:|---:|
| bound direct full | -0.151 | -1.955 | -0.184 |
| bound static \(w=1/17\) | -0.082 | -1.175 | -0.068 |
| bag-independent static \(w=1/17\) | -0.276 | -3.584 | -0.392 |

The local picture is not that matched static updates are answer-aligned from the start. They still contain strong context pressure, but the relation-compatible answer term and reduced context coefficient keep the model in a recoverable/preserving trajectory.

## 5. Activation-level probe: retained behavior tracks a query-match signal but causal evidence is still needed

Artifacts:

- script: `experiments/archive/functional_learning/scripts/activation_marker_probe.py`
- data: `experiments/archive/functional_learning/data/activation_marker_probe/results.json`
- note: `research/notes/functional_learning/activation_marker_probe.md`

A shared scalar linear probe was trained on train-entity hidden states at the four context attribute positions to select the slot whose entity matches the query, then evaluated on held-entity rows. This was run for the two strong-preparation seeds (43,100) at 250 continuation epochs.

| arm | held top-4 | held B | probe held acc | probe held margin |
|---|---:|---:|---:|---:|
| preparation | 0.887 | +6.639 | 0.854 | +5.832 |
| direct full | 0.389 | +1.787 | 0.484 | +0.059 |
| static \(w=1/17\) | 0.863 | +9.360 | 0.833 | +3.817 |
| context-only half-lr | 0.244 | -0.062 | 0.482 | -0.068 |
| slot0 static \(w=1/17\) | 0.238 | +0.536 | 0.479 | -0.330 |

This association is strong: the successful bound static arm preserves a linearly accessible held query-match signal, while direct/context-only/slot0 reduce it to near chance on average. The seed43 context-only and slot0 margins show that partial separability can exist without bound answer behavior, so this probe must not be treated as causal proof. The next experiment should ablate or patch the candidate query-match subspace inside the Transformer and measure held behavior.

## Current best mechanism

Within the query-first orbit-binding task, cross-symbol transfer survives broad context prediction only when two conditions hold together:

1. competing local-prediction credit is not allowed to dominate the body update, and
2. answer-position credit remains aligned with the same query-conditioned relation whose transfer is needed.

Neither reduced context dose alone nor answer/RWT pressure alone preserves the computation. This makes the principle more precise than the earlier interleaving story: data efficiency is governed by **effective relation-aligned credit**, not by repetition, answer exposure, or schedule form in isolation.

This is still controlled synthetic evidence. It should now be strengthened by a causal activation intervention and then connected to the natural-language fixed-budget phenomena from relation_learning.
