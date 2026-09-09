# fixed budget learning principle after review RoBERTa terminal active-error ladder

CPU-only forward MLM measurement on existing RoBERTa MAX-dose view/repeat/clean checkpoints. No training, no official benchmark evaluation, no upload, no leaderboard action.

## Why this measurement matters
RoBERTa already has a negative broad late MAX view-clean downstream contrast (exEntity5 -0.6873, cheap6 -0.5283), while DeBERTa retains positive late value for distinct admitted content. This readout asks whether RoBERTa lacks the terminal active-error source itself, or has the source but does not convert it into broad competence.

## Clean-prior mature 80M→100M rates
- view: {'n': 3, 'mean': 0.130971, 'sd': 0.001738, 'min': 0.12954, 'max': 0.132905, 'values': [0.130469, 0.132905, 0.12954]}
- repeat: {'n': 3, 'mean': 0.132226, 'sd': 8.1e-05, 'min': 0.132143, 'max': 0.132304, 'values': [0.132143, 0.13223, 0.132304]}
- clean: {'n': 3, 'mean': 0.149805, 'sd': 0.001629, 'min': 0.148129, 'max': 0.151382, 'values': [0.151382, 0.148129, 0.149904]}

## Own-arm changed-block terminal rates
### view
- changed: {'rate_60_to_100': {'n': 3, 'mean': 0.920573, 'sd': 0.026726, 'min': 0.891573, 'max': 0.944212, 'values': [0.944212, 0.925933, 0.891573]}, 'rate_80_to_100': {'n': 3, 'mean': 0.20592, 'sd': 0.002897, 'min': 0.202602, 'max': 0.207943, 'values': [0.207943, 0.207216, 0.202602]}, 'loss_100M': {'n': 3, 'mean': 6.191808, 'sd': 0.049958, 'min': 6.141828, 'max': 6.241743, 'values': [6.141828, 6.191852, 6.241743]}}
- unchanged: {'rate_60_to_100': {'n': 3, 'mean': 1.462965, 'sd': 0.03389, 'min': 1.441476, 'max': 1.502032, 'values': [1.441476, 1.445386, 1.502032]}, 'rate_80_to_100': {'n': 3, 'mean': 0.209723, 'sd': 0.010693, 'min': 0.197808, 'max': 0.218485, 'values': [0.197808, 0.212877, 0.218485]}, 'loss_100M': {'n': 3, 'mean': 4.445996, 'sd': 0.019686, 'min': 4.427386, 'max': 4.466606, 'values': [4.466606, 4.443995, 4.427386]}}
### repeat
- changed: {'rate_60_to_100': {'n': 3, 'mean': 0.228958, 'sd': 0.008166, 'min': 0.221626, 'max': 0.237758, 'values': [0.237758, 0.221626, 0.227489]}, 'rate_80_to_100': {'n': 3, 'mean': 0.052248, 'sd': 0.006892, 'min': 0.046956, 'max': 0.060041, 'values': [0.060041, 0.046956, 0.049746]}, 'loss_100M': {'n': 3, 'mean': 6.783145, 'sd': 0.032945, 'min': 6.763121, 'max': 6.821169, 'values': [6.763121, 6.765145, 6.821169]}}
- unchanged: {'rate_60_to_100': {'n': 3, 'mean': 0.33376, 'sd': 0.005823, 'min': 0.328529, 'max': 0.340034, 'values': [0.328529, 0.332718, 0.340034]}, 'rate_80_to_100': {'n': 3, 'mean': 0.09644, 'sd': 0.001751, 'min': 0.095296, 'max': 0.098456, 'values': [0.095569, 0.098456, 0.095296]}, 'loss_100M': {'n': 3, 'mean': 5.653208, 'sd': 0.060593, 'min': 5.597339, 'max': 5.717618, 'values': [5.644666, 5.717618, 5.597339]}}
### clean
- changed: {'rate_60_to_100': {'n': 3, 'mean': 1.257573, 'sd': 0.009483, 'min': 1.251809, 'max': 1.268518, 'values': [1.252391, 1.251809, 1.268518]}, 'rate_80_to_100': {'n': 3, 'mean': 0.149805, 'sd': 0.001629, 'min': 0.148129, 'max': 0.151382, 'values': [0.151382, 0.148129, 0.149904]}, 'loss_100M': {'n': 3, 'mean': 4.213847, 'sd': 0.043887, 'min': 4.171361, 'max': 4.259013, 'values': [4.171361, 4.259013, 4.211168]}}
- unchanged: {'rate_60_to_100': {'n': 3, 'mean': 1.18005, 'sd': 0.038618, 'min': 1.136491, 'max': 1.210093, 'values': [1.210093, 1.136491, 1.193565]}, 'rate_80_to_100': {'n': 3, 'mean': 0.149514, 'sd': 0.005719, 'min': 0.144143, 'max': 0.155526, 'values': [0.148872, 0.144143, 0.155526]}, 'loss_100M': {'n': 3, 'mean': 4.427617, 'sd': 0.08575, 'min': 4.342486, 'max': 4.513973, 'values': [4.342486, 4.513973, 4.426393]}}

## Interpretation
RoBERTa does not preserve the DeBERTa-like view>repeat clean-prior terminal-rate ordering. The active-error component itself is architecture/coordinate-sensitive, so conversion cannot be the only boundary. Downstream anchors remain: DeBERTa late view-clean exEntity5 +0.3853, DeBERTa repeat-clean +0.0343, RoBERTa late view-clean exEntity5 -0.6873, with same-coordinate mature exEntity5 resolution floor 0.1765.

## Files
- summary_json: `experiments/archive/frontier_consolidation/data/roberta_terminal_rate_ladder/roberta_terminal_rate_ladder.json`
- measurements_csv: `experiments/archive/frontier_consolidation/data/roberta_terminal_rate_ladder/loss_measurements.csv`
- rate_rows_csv: `experiments/archive/frontier_consolidation/data/roberta_terminal_rate_ladder/rate_rows.csv`
- summary_md: `research/documents/frontier_consolidation/data/roberta_terminal_rate_ladder/roberta_terminal_rate_ladder.md`
