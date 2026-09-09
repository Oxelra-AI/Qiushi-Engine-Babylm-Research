# scale1p75 bottleneck and next route scale1.75 bottleneck and next-route analysis

The complete 80M scale1.75 score is a near miss, not SOTA. It uses the corrected SuperGLUE primary metrics (MRPC/QQP F1; others accuracy) and AoA=0.0.

- 80M Overall: **41.771634940531**, margin vs 41.80 = **-0.028365059469**. Equivalent one-column shortfall: 0.255286.
- 80M SuperGLUE: **69.259714464777**, threshold required at fixed other columns: 69.515000000000.
- 100M fixed columns require SuperGLUE **71.405000000000** to exceed 41.80; if it merely matches aoa mincontext discrepancy audit 100M SuperGLUE (70.279868), projected Overall is 41.674985.
- Moving from 80M to 100M changes non-SuperGLUE column sum by -1.890000: {'BLiMP': 0.519999999999996, 'Supplement': 0.28000000000000114, 'EWoK': -0.1600000000000037, 'Entity': -0.7399999999999984, 'COMPS': 0.18999999999999773, 'GlobalPIQA': -2.0, 'Reading': 0.019999999999999574, 'AoA': 0.0}.

Mechanism interpretation remains negative for binding: matched legal16k EWoK degrades at both 80M and 100M, and earlier analysis matched GlobalPIQA hard-surface evidence showed scale1.75 100M worse than the matched aoa mincontext discrepancy audit legal16k base on parallel accuracy and hard52 margin.

If 100M SuperGLUE does not exceed the high 71.405 requirement, the route should not continue by exposure-only or simple amplitude variants. The next useful work is a representation or learning-signal change that preserves the broad score gains of the adapter trajectory while repairing matched EWoK and GlobalPIQA hard rows.
