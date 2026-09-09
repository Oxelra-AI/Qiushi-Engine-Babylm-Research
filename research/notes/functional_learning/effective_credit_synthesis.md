# answer credit alignment synthesis: relation-aligned effective credit preserves cross-symbol computation

## Scientific question

020 mechanism synthesis--020 located the held-symbol transfer loss in the contextual body rather than in held input/output rows. accumulated mechanism synthesis showed that the loss is not forced by representational capacity: interleaving answer/full training and gradual context-weight ramps preserve much more held-symbol binding while learning context prediction.

answer credit alignment asked a sharper question about the earlier interpretation: is the accumulated mechanism synthesis preservation a special temporal repair effect, or mostly a consequence of effective objective allocation? And if objective allocation matters, is it generic answer/RWT rehearsal or answer credit aligned with the query-conditioned relation?

## Correct effective-credit comparison

Let \(A\) be the mean answer-position loss and \(C\) the mean loss over the 15 non-answer context positions. The implemented static weighted objective with context weight \(w\) is

\[
L_w = \frac{A + 15wC}{1 + 15w}.
\]

Full training has \(w=1\):

\[
L_{\rm full}=\frac{A+15C}{16}.
\]

Alternating answer-only and full epochs has nominal average

\[
\frac12 A + \frac12 \frac{A+15C}{16}
= \frac{17A+15C}{32}.
\]

The matching static objective is therefore \(w=1/17\), not \(w=0.5\), because

\[
L_{1/17}=\frac{A+(15/17)C}{1+15/17}=\frac{17A+15C}{32}.
\]

This matters because static \(w=0.5\) still gives context loss 0.882 of the normalized update, close to direct full's 0.938. The matched static and interleaved objectives both give nominal answer/context coefficients 0.531/0.469.

## Result 1: static matched allocation reproduces most preservation

Artifacts:

- data: `experiments/archive/functional_learning/data/revision_021b_credit_matched/results.json`
- analysis: `experiments/archive/functional_learning/data/revision_021b_credit_matched/credit_matched_analysis.json`
- note: `research/notes/functional_learning/021b_credit_matched_analysis.md`
- figure: `experiments/archive/functional_learning/figures/revision_021b_credit_matched_unified.png`

Endpoint means across seeds 42/43/100 on the same probe bank:

| arm | nominal A coef | nominal C coef | final held top-4 | final held B | final held selectivity | final train top-4 | final context CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct full | 0.062 | 0.938 | 0.418 | +3.111 | +0.256 | 0.921 | 1.233 |
| interleaved answer/full | 0.531 | 0.469 | 0.746 | +7.765 | +0.665 | 1.000 | 1.316 |
| static \(w=1/17\) | 0.531 | 0.469 | 0.694 | +7.548 | +0.574 | 1.000 | 1.238 |
| static \(w=0.1\) | 0.400 | 0.600 | 0.652 | +6.754 | +0.515 | 1.000 | 1.234 |
| static \(w=0.5\) | 0.118 | 0.882 | 0.445 | +4.011 | +0.307 | 0.992 | 1.234 |

The coefficient-matched static arm recovers 84.1% of interleaving's held-top4 gain over direct full, 95.3% of its held-B gain, and 77.8% of its held-selectivity gain. Context CE is not worse than direct full in the static arm (1.238 versus 1.233) and is better than the interleaved arm (1.316), so preservation is not caused by failure to learn context prediction.

Per-seed endpoint held top-4 / held-B:

| seed | direct full | interleaved | static \(w=1/17\) | static \(w=0.1\) | static \(w=0.5\) |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.449/+4.33 | 0.570/+5.74 | 0.547/+5.49 | 0.500/+5.20 | 0.480/+5.20 |
| 43 | 0.406/+2.29 | 0.855/+9.25 | 0.758/+8.24 | 0.793/+7.48 | 0.547/+4.95 |
| 100 | 0.398/+2.71 | 0.812/+8.31 | 0.777/+8.92 | 0.664/+7.58 | 0.309/+1.89 |

Interpretation: periodic answer-only epochs are not necessary for most of the preservation effect. The stronger working mechanism is compatible effective allocation: the context objective can be learned without erasing held-symbol transfer when relation-relevant answer credit remains strong enough during the same training period.

Interleaving may still have a smaller schedule-specific effect. It gives the best mean held top-4 and selectivity, especially seed43, but the static matched arm gives comparable held-B and lower context CE. The remaining gap is a real tradeoff, not evidence that temporal alternation is the main phenomenon.

## Result 2: preservation requires relation-aligned answer credit

Artifacts:

- script: `experiments/archive/functional_learning/scripts/answer_credit_alignment.py`
- data: `experiments/archive/functional_learning/data/answer_credit_alignment/results.json`
- note: `research/notes/functional_learning/answer_credit_alignment.md`

The control keeps the same query-first context rows, same answer position, same RWT answer family, same absence of held entities in continuation rows, and either the same nominal static coefficient \(w=1/17\) or the same interleaving format. The only change is the answer target: instead of the queried entity's attribute, the target is a bag-independent random context attribute. This supplies answer/RWT pressure but removes the query-conditioned relation.

Final means:

| arm | final held top-4 | final held B | final held selectivity | final train top-4 | final context CE | final answer CE |
|---|---:|---:|---:|---:|---:|---:|
| bag-independent static \(w=1/17\) | 0.225 | -0.040 | -0.007 | 0.261 | 1.256 | 1.471 |
| bag-independent interleaved | 0.219 | -0.071 | -0.016 | 0.240 | 1.337 | 1.610 |

Both arms learn the context positions to normal CE but drive bound query-conditioned behavior to chance. This separates the preservation effect from generic answer-position rehearsal, RWT-family rehearsal, and the formal presence of alternating answer-only epochs. The answer credit must remain aligned with the same relation whose transfer is being preserved.

The final answer CE near \(\log 4\) is expected because the bag-independent answer target is not deterministically recoverable from the visible sequence: it asks for a random one of the four context attributes. Its role is not to train a stable alternative relation, but to provide high answer-position pressure with the wrong information structure. It fails to preserve binding, so high answer pressure alone is not sufficient.

## Result 3: local gradient pressure explains the first update without settling long-run representation

Artifacts:

- script: `experiments/archive/functional_learning/scripts/gradient_pressure.py`
- data: `experiments/archive/functional_learning/data/gradient_pressure/results.json`
- note: `research/notes/functional_learning/gradient_pressure.md`

At the preparation checkpoint, bound answer loss is already tiny (mean 0.024), whereas context loss is huge (mean 33.638). Body gradient norms are approximately 1.16 for bound answer loss and 21.36 for context loss. Therefore even the coefficient-matched \(w=1/17\) gradient is still nearly context-direction at the exact starting point (mean cosine with context +0.997). It nevertheless has about half the normalized context coefficient of direct full training and includes relation-aligned answer credit.

Finite one-epoch effects across seeds:

| condition | mean change held top-4 | mean change held B | mean change trained top-4 | answer CE | context CE |
|---|---:|---:|---:|---:|---:|
| bound direct full | -0.151 | -1.955 | -0.184 | 0.101 | 28.738 |
| bound static \(w=1/17\) | -0.082 | -1.175 | -0.068 | 0.058 | 28.786 |
| bag-independent static \(w=1/17\) | -0.276 | -3.584 | -0.392 | 5.599 | 29.688 |

The matched static bound update still perturbs transfer, but much less than direct full. The bag-independent matched update is more destructive despite the same answer-position weight. The long-run static preservation is therefore not because the initial update is answer-gradient aligned; it is because the competing context coefficient is small enough and the answer term remains relation-aligned as the model moves through training.

## Current mechanism

The controlled task now supports the following mechanism:

1. Query-first answer-only training can form a shared query-conditioned context computation that works on trained entities and, for seeds 43/100, held entity symbols never used in continuation.
2. Direct full next-token training gives almost all normalized credit to non-answer context prediction. Early body updates follow the huge context-gradient direction and quickly suppress held-symbol transfer, even though trained-symbol binding can later return.
3. Held input/output row drift is not the cause; fixing the token tables does not prevent collapse, and final held rows do not break the preparation network.
4. A compatible objective exists: static relation-aligned answer/context allocation with \(w=1/17\) keeps cross-symbol transfer while learning context prediction. Alternation gives a small additional endpoint advantage in some seeds but is not required for most of the effect.
5. Generic answer/RWT pressure is insufficient. When answer credit no longer points to the query-conditioned relation, both static and interleaved versions lose binding despite normal context learning.

A concise principle emerging from this substrate is: **finite experience produces reusable computation when later learning continues to assign enough relation-aligned credit to the internal variables that carry that computation; high-volume local prediction objectives can suppress transfer not because the reusable computation is impossible, but because their effective credit dominates the training trajectory.**

## What remains unresolved

The result is still a controlled synthetic mechanism, not a BabyLM-scale law. The next work should identify the representation-level object being preserved and connect it to natural-language fixed-budget effects.

Concrete predictions now available:

1. In bound static \(w=1/17\), trained-entity answer gradients should correct perturbations in attribute-position states in a direction that benefits held entities, whereas bag-independent answer gradients should not.
2. If a direct-full run is given additional relation-aligned answer credit only after held transfer has collapsed, recovery of held behavior should be slower or incomplete compared with starting from the preparation checkpoint; that would separate prevention from reacquisition.
3. Activation probes should find a query-match signal at context attribute positions that is retained in static \(w=1/17\)/interleaved arms and reduced in direct-full or bag-independent arms. Such probes must be tied to behavior through interventions, not treated as sufficient on their own.
4. A natural-language bridge should look for cases where examples that continue to reward the same relation-variable structure protect transfer across held lexical symbols, while high-frequency local prediction or unaligned answer-like objectives improve familiar contexts but weaken transfer.

## Updated relation to the overall research goal

The synthetic binding research has moved from an observed SOTA-adjacent phenomenon to a candidate general learning principle with a defined mechanism and tests: reusable knowledge is not just a representation that appears after enough examples; it is a computation maintained by continuing credit on the variables that make it reusable. Data-efficient learning under limited compute therefore depends on the interaction between data structure and effective credit allocation. Experience that looks plentiful can be low-value or destructive if it mostly trains local predictive shortcuts; sparse relation-aligned experience can have high value because it keeps a shared computation usable for symbols and cases not replayed.

The principle is not yet complete. It needs activation-level evidence, stronger tests across schedules and model sizes, and a bridge to the peer natural-language evidence where recurrence/view/split effects vary by architecture and target form.
