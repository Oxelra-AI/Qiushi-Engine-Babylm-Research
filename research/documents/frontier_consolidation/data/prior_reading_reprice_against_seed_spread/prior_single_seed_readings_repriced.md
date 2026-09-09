# earlier analysis prior readings priced against full-DeBERTa seed spread

This compares magnitudes; it does not retroactively erase mechanically matched negative results. It tells which prior few-tenth readings require seed-spread-aware wording and which remain large relative to the measured full-DeBERTa treatment-spread scale.

## Measured treatment-delta seed spread
- cheap6_no_GlobalPIQA: median_abs=0.33666666666666245, p75_abs=0.5737499999999986, p90_abs=0.7407499999999991, max_abs=0.8599999999999994, n=10
- cheap5_no_GlobalPIQA_Reading: median_abs=0.5549999999999997, p75_abs=0.6289999999999996, p90_abs=0.7025999999999996, max_abs=0.9059999999999988, n=10
- EWoK_plus_Entity_sum: median_abs=2.8600000000000065, p75_abs=3.2050000000000054, p90_abs=3.9540000000000046, max_abs=4.439999999999998, n=10
- Supplement: median_abs=0.9149999999999956, p75_abs=1.4450000000000056, p90_abs=3.064, max_abs=3.6400000000000006, n=10
- Entity: median_abs=0.8199999999999967, p75_abs=1.727500000000001, p90_abs=2.2250000000000014, max_abs=2.719999999999999, n=10
- COMPS: median_abs=0.6099999999999994, p75_abs=0.8524999999999974, p90_abs=1.089, max_abs=1.259999999999998, n=10

## Prior readings
- GPT2 causal compact-minus-repeat mean over six endpoints (cross-architecture transfer): source `data/causal_compact_repeat_selected_contrast/causal_compact_repeat_selected_contrast.json`
  - cheap6_no_GlobalPIQA: value=-0.1446, abs/median_spread=0.4294544554455499, abs/p90_spread=0.19518461019237282, same_scale=True
  - cheap5_no_GlobalPIQA_Reading: value=-0.1513, abs/median_spread=0.2726720720720722, abs/p90_spread=0.21538998007401094, same_scale=True
  - EWoK_plus_Entity_sum: value=-0.3192, abs/median_spread=0.11159685314685289, abs/p90_spread=0.08072003034901355, same_scale=True
- RoBERTa compact-minus-repeat 100M (cross-architecture transfer): source `notes/roberta_selected_transfer_resolution.md`
  - cheap6_no_GlobalPIQA: value=-0.1150, abs/median_spread=0.3415841584158459, abs/p90_spread=0.1552480593992577, same_scale=True
  - cheap5_no_GlobalPIQA_Reading: value=-0.1940, abs/median_spread=0.34954954954954975, abs/p90_spread=0.2761172786791918, same_scale=True
  - EWoK_plus_Entity_sum: value=+0.6400, abs/median_spread=0.22377622377622328, abs/p90_spread=0.16186140617096592, same_scale=True
  - Supplement: value=-2.2100, abs/median_spread=2.415300546448099, abs/p90_spread=0.7212793733681462, same_scale=True
  - Entity: value=-0.4100, abs/median_spread=0.500000000000002, abs/p90_spread=0.1842696629213482, same_scale=True
  - COMPS: value=+0.1100, abs/median_spread=0.1803278688524592, abs/p90_spread=0.10101010101010101, same_scale=True
- RoBERTa compact-minus-repeat late 60M/70M stable mean (cross-architecture transfer): source `notes/roberta_selected_transfer_resolution.md`
  - cheap6_no_GlobalPIQA: value=-0.2659, abs/median_spread=0.7898019801980297, abs/p90_spread=0.35896051299358805, same_scale=True
  - cheap5_no_GlobalPIQA_Reading: value=-0.3040, abs/median_spread=0.547747747747748, abs/p90_spread=0.4326786222601767, same_scale=True
  - EWoK_plus_Entity_sum: value=-0.1450, abs/median_spread=0.05069930069930058, abs/p90_spread=0.036671724835609466, same_scale=True
- Extractive balanced minus natural compact mean 80M/100M (data-factor contrast): source `notes/extractive_selected_readout_result.md`
  - cheap6_no_GlobalPIQA: value=-0.2570, abs/median_spread=0.7633663366336729, abs/p90_spread=0.34694566317921066, same_scale=True
  - cheap5_no_GlobalPIQA_Reading: value=-0.2870, abs/median_spread=0.5171171171171174, abs/p90_spread=0.4084827782522063, same_scale=True
  - EWoK_plus_Entity_sum: value=-1.5800, abs/median_spread=0.5524475524475512, abs/p90_spread=0.3995953464845721, same_scale=True
  - Supplement: value=+0.6200, abs/median_spread=0.6775956284153039, abs/p90_spread=0.2023498694516971, same_scale=True
  - Entity: value=-0.7600, abs/median_spread=0.9268292682926866, abs/p90_spread=0.34157303370786496, same_scale=True
  - COMPS: value=-0.6900, abs/median_spread=1.1311475409836076, abs/p90_spread=0.6336088154269972, same_scale=True
- Extractive wide minus natural compact mean 80M/100M (data-factor contrast): source `notes/extractive_selected_readout_result.md`
  - cheap6_no_GlobalPIQA: value=-0.6620, abs/median_spread=1.966336633663391, abs/p90_spread=0.89368882888964, same_scale=True
  - cheap5_no_GlobalPIQA_Reading: value=-0.7460, abs/median_spread=1.3441441441441448, abs/p90_spread=1.0617705664674075, same_scale=False
  - EWoK_plus_Entity_sum: value=-2.8200, abs/median_spread=0.9860139860139837, abs/p90_spread=0.7132018209408185, same_scale=True
  - Supplement: value=-2.0650, abs/median_spread=2.256830601092907, abs/p90_spread=0.6739556135770235, same_scale=True
  - Entity: value=-2.3550, abs/median_spread=2.8719512195122063, abs/p90_spread=1.0584269662921342, same_scale=False
  - COMPS: value=-0.0350, abs/median_spread=0.057377049180327926, abs/p90_spread=0.03213957759412305, same_scale=True
- Plain no-disentangle minus full interaction mean 80M/100M before common-copy (architecture-coordinate confounded interaction): source `data/architecture_interaction_selected_panel_final/architecture_interaction_selected_panel_rows.csv`
  - cheap6_no_GlobalPIQA: value=-0.6908, abs/median_spread=2.0518811881188377, abs/p90_spread=0.9325683428957149, same_scale=True
  - cheap5_no_GlobalPIQA_Reading: value=-0.6080, abs/median_spread=1.095495495495496, abs/p90_spread=0.8653572445203535, same_scale=True
  - EWoK_plus_Entity_sum: value=-3.1200, abs/median_spread=1.0909090909090884, abs/p90_spread=0.7890743550834589, same_scale=True
- Common-copy no-disentangle minus full interaction mean 80M/100M (architecture-coordinate causal interaction): source `notes/commoncopy_architecture_interaction_result.md`
  - cheap6_no_GlobalPIQA: value=+0.0679, abs/median_spread=0.20168316831683422, abs/p90_spread=0.09166385420182259, same_scale=True
  - cheap5_no_GlobalPIQA_Reading: value=+0.1550, abs/median_spread=0.2792792792792794, abs/p90_spread=0.22060916595502433, same_scale=True
  - EWoK_plus_Entity_sum: value=-0.3850, abs/median_spread=0.1346153846153843, abs/p90_spread=0.09736975214972168, same_scale=True
- Scale1.25 residual-adapter trajectory minus scale1.75 reference mean (residual-capacity route): source `notes/lead_cross_seed_decision_framework.md`
  - cheap6_no_GlobalPIQA: value=-0.8152, abs/median_spread=2.4214099009901293, abs/p90_spread=1.1005170435369571, same_scale=False
  - cheap5_no_GlobalPIQA_Reading: value=-0.9491, abs/median_spread=1.710135135135136, abs/p90_spread=1.3508753202391126, same_scale=False
  - EWoK_plus_Entity_sum: value=-0.4253, abs/median_spread=0.14871083916083883, abs/p90_spread=0.10756525037936254, same_scale=True
- Reference ordinary late chck84-over-chck82 selected cheap7 shift (late-training allocation shift): source `Research memory partial deberta grid and endpoint branch`

JSON: `experiments/archive/frontier_consolidation/data/prior_reading_reprice_against_seed_spread/prior_single_seed_readings_repriced.json`
