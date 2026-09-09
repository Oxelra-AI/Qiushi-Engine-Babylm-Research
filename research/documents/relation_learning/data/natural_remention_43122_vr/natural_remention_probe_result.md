# research synthesis natural re-mention probe result

Scored 258 held-out natural re-mention records (129 verbatim + 129 nonidentical).
antecedent_gain = NLL(replaced) - NLL(present). Positive = antecedent helps.

## Late-mean contrasts

| arch | seed | group | V gain | R gain | C gain | V−C | R−C | V−R |
|---|---|---|---:|---:|---:|---:|---:|---:|
| D | 43122 | ALL | +1.8571 | +1.7117 | NA | NA | NA | +0.1454 |
| D | 43122 | nonidentical_remention | +1.0521 | +0.9249 | NA | NA | NA | +0.1272 |
| D | 43122 | verbatim_remention | +2.6621 | +2.4985 | NA | NA | NA | +0.1636 |

## Prediction check

Pre-stated: REPEAT > CLEAN on verbatim gain; VIEW > CLEAN on nonidentical gain.
Confirmatory: (R-C)_verbatim > (R-C)_nonidentical AND (V-C)_nonidentical > (V-C)_verbatim.

  D_43122 nonidentical_remention: V−C=None, R−C=None
  D_43122 verbatim_remention: V−C=None, R−C=None

