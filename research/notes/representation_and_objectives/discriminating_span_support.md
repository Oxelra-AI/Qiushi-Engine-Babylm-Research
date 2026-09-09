# discriminating span support — Low-support tokens in answer-discriminating spans

CPU-only. No training, no model evaluation, no managed-task query. This isolates the decisive object flagged by independent_review: whether legal40k low-support tokens actually sit in the discriminating part of each item (the parts that differ across candidates or contexts), which the model must use to choose the answer, versus the shared context.

For each item the SHARED span is text common to every candidate; the DISCRIMINATING span is the multiset difference across candidates (EWoK: Context1 vs Context2; GlobalPIQA: solutions; COMPS: acceptable vs unacceptable prefix; Entity: options).

| family | items | disc token share | shared frac<50 | disc frac<50 | disc/shared<50 | disc share of all <50 mass | shared frac<100 | disc frac<100 | disc share of all <100 mass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EWoK | 7618 | 0.1050 | 0.0878 | 0.0865 | 0.9845 | 0.1036 | 0.1483 | 0.1685 | 0.1176 |
| GlobalPIQA | 203 | 0.2243 | 0.0382 | 0.0713 | 1.8669 | 0.3506 | 0.0894 | 0.1283 | 0.2932 |
| COMPS | 91028 | 0.2187 | 0.0640 | 0.1871 | 2.9225 | 0.4500 | 0.0900 | 0.2646 | 0.4514 |
| Entity | 9483 | 0.0666 | 0.0029 | 0.0142 | 4.8195 | 0.2558 | 0.1153 | 0.0202 | 0.0123 |

## Scientific reading

If the discriminating span has clearly higher low-support fraction than the shared span (disc/shared > 1) and holds a large share of the total low-support mass, then undertrained legal40k rows plausibly sit exactly where the answer is decided, strengthening the support-aware representation route. If low-support mass is dominated by shared context, then the support corner is weaker: rare tokens are seen by both candidates and cannot directly explain discrimination, so U256, depth, or a different objective is more likely the operative lever. Read together with the pending depth and minfreq50 score vectors before any launch.

JSON: `experiments/archive/representation_and_objectives/data/discriminating_span_support/discriminating_span_support.json`
CSV: `experiments/archive/representation_and_objectives/data/discriminating_span_support/discriminating_span_support_by_family.csv`
