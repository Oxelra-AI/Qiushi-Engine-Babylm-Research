# routeB bridge predictive value — Route B and bridge predictive-value synthesis
## Legal earlier analysis role-switch screen
- `anchor_fixed256_80M`: cheap7=51.5580, BLiMP=67.7, Supplement=60.65, EWoK=51.16, Entity=26.47, COMPS=51.81, GlobalPIQA=None, Reading=None; GP_parallel=22.33, hard52=0.00, hard52_margin=1.957; EWoK_acc=0.509451, stable_fail=2545; bridge_M=-1.0014290020448646, bridge_both=0.13541666666666666.
- `role_fixed_80M`: cheap7=50.7320, BLiMP=68.44, Supplement=57.32, EWoK=50.29, Entity=24.67, COMPS=52.94, GlobalPIQA=None, Reading=None; GP_parallel=23.30, hard52=1.92, hard52_margin=1.861; EWoK_acc=0.500919, stable_fail=2673; bridge_M=-2.1382323048439704, bridge_both=0.11805555555555555.
- `role_switch_80M`: cheap7=51.1380, BLiMP=68.23, Supplement=58.38, EWoK=49.62, Entity=26.67, COMPS=52.79, GlobalPIQA=None, Reading=None; GP_parallel=23.30, hard52=1.92, hard52_margin=1.965; EWoK_acc=0.509714, stable_fail=2574; bridge_M=-1.3978733137385764, bridge_both=0.1545138888888889.

Role-switch minus role-fixed cheap deltas: {"BLiMP": -0.20999999999999375, "Supplement": 1.0600000000000023, "EWoK": -0.6700000000000017, "Entity": 2.0, "COMPS": -0.14999999999999858, "GlobalPIQA": null, "Reading": null, "cheap7": 0.4059999999999988}

## earlier analysis small updates sorted by bridge delta
| target | obj | lr | upd | bridge ΔM | bridge Δboth | clean Δloss | GP par | GP hard52 | hard52 margin Δ | EWoK acc | stable fail Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor80_correct_ce_lr5e-5_u80 | correct_ce | 5e-05 | 80 | 3.891 | 0.109 | 0.1072 | 26.21 | 1.92 | 0.111 | 0.515358 | -128 |
| pareto_correct_ce_lr2e-5_u20 | correct_ce | 2e-05 | 20 | 3.361 | 0.170 | 0.0477 | 28.16 | 1.92 | 0.038 | 0.513914 | -46 |
| pareto_correct_ce_lr1e-5_u40 | correct_ce | 1e-05 | 40 | 2.372 | 0.118 | 0.0480 | 28.16 | 1.92 | 0.053 | 0.509320 | 5 |
| anchor80_exchange_lr5e-5_u80 | exchange | 5e-05 | 80 | 2.134 | 0.115 | 0.0879 | 25.24 | 1.92 | 0.068 | 0.502625 | -30 |
| anchor80_fixed_ce_lr5e-5_u80 | fixed_ce | 5e-05 | 80 | 1.593 | 0.045 | 0.0622 | 26.21 | 0.00 | 0.014 | 0.507351 | -40 |
| pareto_exchange_lr2e-5_u20 | exchange | 2e-05 | 20 | 1.267 | 0.056 | -0.0026 | 22.33 | 0.00 | -0.012 | 0.505251 | 16 |
| pareto_exchange_lr1e-5_u40 | exchange | 1e-05 | 40 | 1.233 | 0.066 | 0.0022 | 22.33 | 0.00 | 0.005 | 0.508532 | 19 |

## Interpretation
- The legal role-switch replacement arm learned more constructed packet behavior than role-fixed, but GlobalPIQA_parallel is identical between role-switch and role-fixed and the 52-row hard subset remains one row correct in both; role-switch has worse hard52 mean top-minus-correct than role-fixed.
- Broad cheap7 does not rescue the route: role-switch is below the fixed-256 80M anchor and only modestly above role-fixed through Entity/Supplement tradeoffs; it is not a SOTA seed.
- The low-dose bridge does not select the natural hard surfaces. Ordinary correct CE gives the largest bridge gain, but GlobalPIQA hard behavior stays at or below the anchor and EWoK movement is small or harmful. The fixed-CE arm also moves the bridge without the desired natural repair.
- Therefore the constructed bridge remains useful as an unsolved transfer object and stress test, but not as a direct optimization target. The next work should change representation or learning signal source rather than elaborate packet dose/placement or tune on the bridge.
