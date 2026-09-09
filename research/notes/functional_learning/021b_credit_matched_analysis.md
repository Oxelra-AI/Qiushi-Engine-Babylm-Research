# Step021b effective-credit matched comparison

## Why w=1/17 is the matched static objective

Let A be mean answer-position loss and C be mean context-position loss over the 15 non-answer context positions. The implemented weighted loss for static context weight w is

\[L_w = \frac{A + 15 w C}{1 + 15 w}.\]

Full training is \(L_1=(A+15C)/16\), and answer-only is \(L_0=A\). Alternating answer-only and full gives the nominal average

\[\tfrac12 L_0 + \tfrac12 L_1 = (17A+15C)/32.\]

The static objective matching those coefficients is therefore \(w=1/17\), since \(L_{1/17}=(17A+15C)/32\). Static w=0.5 and w=0.1 are useful dose-response arms but do not isolate temporal alternation.

## Endpoint comparison on the same probe set

| arm | A coef | C coef | final h4 | final hB | final hSel | train4 | blocked4 | ctx CE | ans CE | Δh4 vs direct | Δctx vs direct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct full | 0.062 | 0.938 | 0.418 | +3.111 | +0.256 | 0.921 | 0.258 | 1.233 | 0.1298 | +0.000 | +0.000 |
| gradual ramp |  |  | 0.587 | +5.702 | +0.442 | 1.000 | 0.264 | 1.229 | 0.0831 | +0.169 | -0.003 |
| interleaved | 0.531 | 0.469 | 0.746 | +7.765 | +0.665 | 1.000 | 0.238 | 1.316 | 0.0003 | +0.328 | +0.083 |
| static w=0.5 | 0.118 | 0.882 | 0.445 | +4.011 | +0.307 | 0.992 | 0.255 | 1.234 | 0.0584 | +0.027 | +0.001 |
| static w=0.1 | 0.400 | 0.600 | 0.652 | +6.754 | +0.515 | 1.000 | 0.270 | 1.234 | 0.0002 | +0.234 | +0.002 |
| static w=1/17 | 0.531 | 0.469 | 0.694 | +7.548 | +0.574 | 1.000 | 0.264 | 1.238 | 0.0001 | +0.276 | +0.005 |

The coefficient-matched static arm recovers most of the interleaved endpoint advantage over direct full training: 84.1% of the h4 gain, 95.3% of the hB gain, and 77.8% of the hSel gain.

## Per-seed final h4/hB

| seed | direct | interleaved | static w=1/17 | static w=0.1 | static w=0.5 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.449/+4.33 | 0.570/+5.74 | 0.547/+5.49 | 0.500/+5.20 | 0.480/+5.20 |
| 43 | 0.406/+2.29 | 0.855/+9.25 | 0.758/+8.24 | 0.793/+7.48 | 0.547/+4.95 |
| 100 | 0.398/+2.71 | 0.812/+8.31 | 0.777/+8.92 | 0.664/+7.58 | 0.309/+1.89 |

## Interpretation

The matched static objective preserves held-symbol transfer almost as well as interleaving while learning context prediction on the same held-out probe distribution. This changes the accumulated mechanism synthesis interpretation: periodic answer-only epochs are not required for the preservation effect. The main cause is compatible effective allocation between answer credit and context credit. In this task, reducing the normalized context coefficient from 0.938 in direct full training to 0.469 in the matched static/interleaved allocation is enough for the trained-entity answer signal to keep shaping a shared query-conditioned computation that also works on held entities.

Temporal ordering may still matter, but as a smaller effect rather than the main explanation. Interleaving has the best mean final h4 (0.746 vs 0.694 for static w=1/17) and hSel (0.665 vs 0.574), with the clearest extra h4 in seed43. The static arm, however, has comparable hB and lower context CE (1.238 vs 1.316). Because both use the same common probes and training distribution, the remaining difference is a real tradeoff to investigate, not grounds for calling alternation necessary.

The half/tenth arms show the response to static allocation. w=0.5 is close to direct full and still loses much held transfer, while w=0.1 and w=1/17 preserve far more. This nonlinearity indicates that the destructive transition is controlled by the answer/context coefficient ratio rather than by the mere presence of context prediction. Context CE by 500 epochs remains around 1.23--1.24 for all static weights, so the preserved held behavior is not achieved by failing to learn the context positions.

The result is also not ordinary held-symbol rehearsal: held entity tokens never occur in the continuation rows, and blocked query-to-context evaluation stays near chance. What is being maintained must therefore be a computation shared across entity symbols under the query-first format. The present evidence does not yet identify its activation-level form or prove that the direct-full path erases a specific internal marker; it shows that the transfer-bearing computation survives when answer-position credit is kept strong enough during context learning.

## Prediction for the next experiment

If effective allocation is the main mechanism, answer-position gradients computed on trained entities should have a positive projection on the transfer-bearing query-conditioned pathway, while context gradients at the full coefficient should dominate or oppose that pathway early after the switch. A coefficient-matched static update should reduce the destructive projection without requiring a separate answer-only epoch. If temporal scheduling adds a real second-order effect, the interleaved trajectory should show different optimizer-state or activation-marker evolution from the static w=1/17 trajectory despite matched nominal coefficients. This can be tested by saving matched checkpoints and measuring gradient alignment plus linear/causal probes on attribute-position states for direct, static w=1/17, and interleaved arms.

## Files

- accumulated mechanism synthesis data: `experiments/archive/functional_learning/data/calibration_trajectory/results.json`
- Step021b data: `experiments/archive/functional_learning/data/revision_021b_credit_matched/results.json`
- Analysis JSON: `experiments/archive/functional_learning/data/revision_021b_credit_matched/credit_matched_analysis.json`
- Unified figure: `experiments/archive/functional_learning/figures/revision_021b_credit_matched_unified.png`
