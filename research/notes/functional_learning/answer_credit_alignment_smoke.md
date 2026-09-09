# answer credit alignment answer-credit alignment test

Continuation starts from the same query-first bound answer-only preparation checkpoints used in accumulated mechanism synthesis.
The two arms keep the context rows and answer-position marginal RWT vocabulary, but train the answer target as bag-independent rather than the queried entity's attribute.
Held entity tokens remain absent from all continuation rows.

## Final means

| arm | final h4 | final hB | final hSel | train4 | blocked4 | ctx CE | ans CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| bag_static_1over17 | 0.396 | +0.886 | +0.309 | 0.521 | 0.240 | 20.872 | 3.559 |
| bag_interleaved_ans_full | 0.458 | +1.418 | +0.390 | 0.542 | 0.229 | 24.217 | 3.992 |

## Per-seed trajectories

### Seed 100
**bag_static_1over17**
  be=  0 h4=0.958 hB=+7.646 hSel=+0.901 tr4=1.000 ctx=0.000 cw=0.0
  be=  1 h4=0.896 hB=+7.009 hSel=+0.859 tr4=1.000 ctx=30.615 cw=0.0588
  be=  5 h4=0.500 hB=+3.222 hSel=+0.592 tr4=0.802 ctx=25.610 cw=0.0588
  be= 10 h4=0.396 hB=+0.886 hSel=+0.309 tr4=0.521 ctx=20.872 cw=0.0588

**bag_interleaved_ans_full**
  be=  0 h4=0.958 hB=+7.646 hSel=+0.901 tr4=1.000 ctx=0.000 cw=0.0
  be=  1 h4=0.896 hB=+6.822 hSel=+0.846 tr4=1.000 ctx=30.615 cw=0.0
  be=  5 h4=0.625 hB=+4.029 hSel=+0.610 tr4=0.906 ctx=27.696 cw=0.0
  be= 10 h4=0.458 hB=+1.418 hSel=+0.390 tr4=0.542 ctx=24.217 cw=1.0

## Interpretation

If these bag-independent answer-credit arms lose binding while Step021b's bound static w=1/17 and bound interleaving preserve it, the preservation cannot be explained by answer-token exposure, RWT-family rehearsal, or temporal alternation alone. It depends on answer gradients that remain aligned with the query-conditioned relation. Conversely, if they preserve, the previous result would reduce to generic answer/RWT rehearsal or loss allocation without relation-specific credit.
