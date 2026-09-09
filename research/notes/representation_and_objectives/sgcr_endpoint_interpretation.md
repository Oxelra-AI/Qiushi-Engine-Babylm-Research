# sgcr endpoint interpretation — SGCR endpoint interpretation

Status: `SGCR_ENDPOINT_HAS_INTEGRITY_ERRORS`

SGCR Overall: `40.331709`; margin vs 41.80 `-1.468291`

## SGCR movement vs matched depth 12x384 seed43022
- BLiMP: `+0.215412`
- Supplement: `-2.390269`
- EWoK: `-0.019702`
- Entity: `-1.730575`
- COMPS: `+0.194468`
- SuperGLUE: `-1.143231`
- GlobalPIQA: `-1.485437`
- Reading: `+0.096356`
- AoA: `+0.000000`
- Overall: `-0.695886`

## SGCR movement vs shallower legal40k 8x480 seed43022
- BLiMP: `-0.258853`
- Supplement: `-3.216289`
- EWoK: `-0.946496`
- Entity: `-1.762823`
- COMPS: `+1.222148`
- SuperGLUE: `-1.425960`
- GlobalPIQA: `-0.514563`
- Reading: `-0.376980`
- AoA: `+0.000000`
- Overall: `-0.808868`

## sgcr official span burden burden-map anchors
- COMPS: score_delta_vs_depth `0.1944679557143374`, residual ratio `2.5987900215439748`, lt50 ratio `2.922520795506795`, disc residual share `0.421139710234526`
- GlobalPIQA: score_delta_vs_depth `-1.4854368932038824`, residual ratio `1.5194709613873472`, lt50 ratio `1.8669243881296869`, disc residual share `0.30527773473988024`
- GlobalPIQA_parallel: score_delta_vs_depth `-1.4854368932038824`, residual ratio `1.9445340105888722`, lt50 ratio `2.6791356306785623`, disc residual share `0.4748105325355171`
- GlobalPIQA_nonparallel: score_delta_vs_depth `-1.4854368932038824`, residual ratio `1.7512595428958508`, lt50 ratio `2.420653533458412`, disc residual share `0.16947273997785076`
- Entity: score_delta_vs_depth `-1.730575352866886`, residual ratio `1.0287355123794735`, lt50 ratio `5.233865376872995`, disc residual share `0.07459027307062506`
- EWoK: score_delta_vs_depth `-0.019701599830121097`, residual ratio `1.1362750340158727`, lt50 ratio `0.9844985950885389`, disc residual share `0.11765961274199163`

## Integrity
- ERROR: missing or invalid example_order_manifest.json

## Scientific reading
- The SGCR files have integrity mismatches; repair provenance or official-coordinate artifacts before using the endpoint scientifically.
- SGCR underperforms both matched depth and shallower legal40k baselines; close exact-prefix SGCR as implemented unless a specific column-level result reveals a different useful mechanism.
- Only part of the strongest on-mechanism pair moves over depth; inspect COMPS/GlobalPIQA subtasks and consider whether the gain is support-local or generic.
- At least one broad language column falls by more than 0.5 over depth; if SGCR is otherwise promising, any next comparison must separate support-dependent gains from interference caused by the added auxiliary estimator.

## Files
- json: `experiments/archive/representation_and_objectives/data/sgcr_endpoint_interpretation/sgcr_endpoint_interpretation.json`
- column_csv: `experiments/archive/representation_and_objectives/data/sgcr_endpoint_interpretation/sgcr_endpoint_column_comparison.csv`
- detail_csv: `experiments/archive/representation_and_objectives/data/sgcr_endpoint_interpretation/sgcr_endpoint_detail_comparison.csv`
- note: `research/notes/representation_and_objectives/sgcr_endpoint_interpretation.md`
