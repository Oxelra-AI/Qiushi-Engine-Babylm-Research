# routeB bridge predictive value — corrected Route B and bridge predictive-value synthesis

This note replaces the first synthesis broad-score calculation, which omitted GlobalPIQA and Reading while reading per-target score JSONs. The hard-surface interpretation is unchanged.

## Legal earlier analysis role-switch screen

- `anchor_fixed256_80M`: cheap7=42.8907; BLiMP=67.7, Supplement=60.65, EWoK=51.16, Entity=26.47, COMPS=51.81, GlobalPIQA=34.665, Reading=7.779999999999999; GP_parallel=22.33, hard52=0.00, hard52_margin=1.957; EWoK_acc=0.509451, stable_fail=2545; bridge_M=-1.001, bridge_both=0.135.
- `role_fixed_80M`: cheap7=42.5893; BLiMP=68.44, Supplement=57.32, EWoK=50.29, Entity=24.67, COMPS=52.94, GlobalPIQA=36.65, Reading=7.815; GP_parallel=23.30, hard52=1.92, hard52_margin=1.861; EWoK_acc=0.500919, stable_fail=2673; bridge_M=-2.138, bridge_both=0.118.
- `role_switch_80M`: cheap7=42.9043; BLiMP=68.23, Supplement=58.38, EWoK=49.62, Entity=26.67, COMPS=52.79, GlobalPIQA=36.65, Reading=7.99; GP_parallel=23.30, hard52=1.92, hard52_margin=1.965; EWoK_acc=0.509714, stable_fail=2574; bridge_M=-1.398, bridge_both=0.155.

Corrected role-switch minus role-fixed cheap deltas: {"BLiMP": -0.20999999999999375, "Supplement": 1.0600000000000023, "EWoK": -0.6700000000000017, "Entity": 2.0, "COMPS": -0.14999999999999858, "GlobalPIQA": 0.0, "Reading": 0.17499999999999982, "cheap7": 0.3149999999999977, "GlobalPIQA_parallel": 0.0, "GlobalPIQA_nonparallel": 0.0}
Corrected role-switch minus anchor cheap deltas: {"BLiMP": 0.5300000000000011, "Supplement": -2.269999999999996, "EWoK": -1.5399999999999991, "Entity": 0.20000000000000284, "COMPS": 0.9799999999999969, "GlobalPIQA": 1.9849999999999994, "Reading": 0.21000000000000085, "cheap7": 0.013571428571424349, "GlobalPIQA_parallel": 0.9700000000000024, "GlobalPIQA_nonparallel": 3.0}

## earlier analysis small updates: bridge movement versus official hard surfaces

| target | obj | lr | upd | bridge ΔM | bridge Δboth | clean Δloss | GP par Δ | GP hard52 | hard52 margin Δ | EWoK acc Δ | stable fail Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor80_correct_ce_lr5e-5_u80 | correct_ce | 5e-05 | 80 | 3.891 | 0.109 | 0.1072 | 3.88 | 1.92 | 0.111 | 0.005907 | -128 |
| pareto_correct_ce_lr2e-5_u20 | correct_ce | 2e-05 | 20 | 3.361 | 0.170 | 0.0477 | 5.83 | 1.92 | 0.038 | 0.004463 | -46 |
| pareto_correct_ce_lr1e-5_u40 | correct_ce | 1e-05 | 40 | 2.372 | 0.118 | 0.0480 | 5.83 | 1.92 | 0.053 | -0.000131 | 5 |
| anchor80_exchange_lr5e-5_u80 | exchange | 5e-05 | 80 | 2.134 | 0.115 | 0.0879 | 2.91 | 1.92 | 0.068 | -0.006826 | -30 |
| anchor80_fixed_ce_lr5e-5_u80 | fixed_ce | 5e-05 | 80 | 1.593 | 0.045 | 0.0622 | 3.88 | 0.00 | 0.014 | -0.002100 | -40 |
| pareto_exchange_lr2e-5_u20 | exchange | 2e-05 | 20 | 1.267 | 0.056 | -0.0026 | 0.00 | 0.00 | -0.012 | -0.004201 | 16 |
| pareto_exchange_lr1e-5_u40 | exchange | 1e-05 | 40 | 1.233 | 0.066 | 0.0022 | 0.00 | 0.00 | 0.005 | -0.000919 | 19 |

Correlations across the seven small updates: {"bridge_delta_M_vs_gp_parallel_delta": 0.6891131675733861, "bridge_delta_M_vs_gp_hard52_margin_delta": 0.8320696612161457, "bridge_delta_M_vs_ewok_accuracy_delta": 0.784613612300945, "bridge_delta_M_vs_ewok_stable_failure_delta": -0.8302909405874167, "bridge_delta_both_vs_gp_parallel_delta": 0.6982251075132491, "bridge_delta_both_vs_gp_hard52_margin_delta": 0.5497744017695675, "bridge_delta_both_vs_ewok_accuracy_delta": 0.4933729805477738, "bridge_delta_both_vs_ewok_stable_failure_delta": -0.33713109694626003}

## Scientific reading

- Route B exact recipe is closed: role-switch learned the packet grammar relative to role-fixed, but role-switch and role-fixed are identical on GlobalPIQA_parallel (23.30) and hard52 accuracy (1/52), and role-switch has worse hard52 mean top-minus-correct than role-fixed.
- Corrected broad scores do not rescue it: role-switch cheap7 is 42.9043, only +0.315 over role-fixed and +0.0136 over the 80M anchor, while losing Supplement and EWoK relative to anchor. This is not a SOTA trajectory and cannot justify 100M continuation.
- The constructed bridge is not a sufficient selector for official hard behavior. Correct-CE variants give the largest bridge improvements and some non-hard GlobalPIQA_parallel movement, but hard52 remains 0/52 or 1/52 with mean wrong margins not improved, and EWoK movement is inconsistent. Low-dose exchange preserves clean MLM better but leaves GP_parallel flat and worsens EWoK stable failures.
- The next mechanism should change the representation or source of learning signal, not vary packet dose/placement or optimize the bridge itself.

Full corrected JSON: `experiments/archive/representation_and_objectives/data/routeB_bridge_synthesis/routeB_bridge_synthesis_corrected.json`
