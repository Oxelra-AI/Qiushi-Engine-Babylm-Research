# fw globalpiqa relevant substrate — GlobalPIQA option-margin reader

This CPU-only diagnostic mirrors the official MLM length-normalized completion scoring for GlobalPIQA and records all candidate scores. It is not a custom submission scorer and must not be tuned on official rows.

## `legal16k8x480_43022`

Model: `experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022/hf_model` revision `chck_100M`; load 0.179 sec on CPU.
- parallel: n=103, accuracy=26.21, chance-adjusted=0.016, rank counts={'3': 28, '4': 26, '1': 27, '2': 22}, choice counts={'0': 25, '2': 27, '1': 23, '3': 28}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=0.00, rank counts={'4': 22, '3': 20, '2': 10}, mean top-minus-correct=1.783, small wrong margins ≤0.25/≤0.50 nats=1/7

## `depth12x384_43022`

Model: `experiments/archive/representation_and_objectives/training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/hf_model` revision `chck_100M`; load 0.073 sec on CPU.
- parallel: n=103, accuracy=24.27, chance-adjusted=-0.010, rank counts={'2': 28, '4': 24, '3': 26, '1': 25}, choice counts={'1': 21, '0': 26, '2': 27, '3': 29}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=0.00, rank counts={'4': 21, '3': 18, '2': 13}, mean top-minus-correct=1.859, small wrong margins ≤0.25/≤0.50 nats=4/10

## `inherited16k_noncompliant_43022`

Model: `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model` revision `chck_100M`; load 0.345 sec on CPU.
- parallel: n=103, accuracy=25.24, chance-adjusted=0.003, rank counts={'2': 18, '4': 23, '3': 36, '1': 26}, choice counts={'1': 27, '0': 20, '2': 21, '3': 35}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=0.00, rank counts={'4': 18, '3': 24, '2': 10}, mean top-minus-correct=1.838, small wrong margins ≤0.25/≤0.50 nats=3/7

Files:
- combined JSON: `experiments/archive/representation_and_objectives/data/globalpiqa_margin_reader/globalpiqa_margin_reader_results.json`
- per-target JSON/CSV under `experiments/archive/representation_and_objectives/data/globalpiqa_margin_reader`
