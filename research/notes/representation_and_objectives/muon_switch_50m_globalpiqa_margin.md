# fw globalpiqa relevant substrate — GlobalPIQA option-margin reader

This CPU-only diagnostic mirrors the official MLM length-normalized completion scoring for GlobalPIQA and records all candidate scores. It is not a custom submission scorer and must not be tuned on official rows.

## `adamw50`

Model: `experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_50M` revision `main`; load 0.09 sec on CPU.
- parallel: n=103, accuracy=28.16, chance-adjusted=0.042, rank counts={'2': 24, '4': 23, '1': 29, '3': 27}, choice counts={'1': 30, '0': 23, '3': 27, '2': 23}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=3.85, rank counts={'4': 17, '2': 12, '3': 21, '1': 2}, mean top-minus-correct=1.762, small wrong margins ≤0.25/≤0.50 nats=2/10
- nonparallel: n=100, accuracy=45.00, chance-adjusted=-0.100, rank counts={'1': 45, '2': 55}, choice counts={'1': 35, '0': 65}

## `continuous_muon50`

Model: `experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_80M/hf_model/chck_50M` revision `main`; load 0.026 sec on CPU.
- parallel: n=103, accuracy=27.18, chance-adjusted=0.029, rank counts={'1': 28, '4': 22, '3': 28, '2': 25}, choice counts={'3': 28, '0': 28, '1': 27, '2': 20}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=5.77, rank counts={'4': 18, '3': 21, '2': 10, '1': 3}, mean top-minus-correct=1.849, small wrong margins ≤0.25/≤0.50 nats=3/6
- nonparallel: n=100, accuracy=48.00, chance-adjusted=-0.040, rank counts={'2': 52, '1': 48}, choice counts={'0': 56, '1': 44}

## `muon20toadamw50`

Model: `experiments/archive/frontier_consolidation/training/runs/muon20toadamw_seed43022_80M/hf_model/chck_50M` revision `main`; load 0.032 sec on CPU.
- parallel: n=103, accuracy=26.21, chance-adjusted=0.016, rank counts={'1': 27, '4': 26, '3': 24, '2': 26}, choice counts={'3': 35, '1': 25, '0': 23, '2': 20}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=5.77, rank counts={'4': 21, '3': 17, '2': 11, '1': 3}, mean top-minus-correct=1.699, small wrong margins ≤0.25/≤0.50 nats=1/5
- nonparallel: n=100, accuracy=51.00, chance-adjusted=0.020, rank counts={'1': 51, '2': 49}, choice counts={'1': 37, '0': 63}

## `muon40toadamw50`

Model: `experiments/archive/frontier_consolidation/training/runs/muon40toadamw_seed43022_80M/hf_model/chck_50M` revision `main`; load 0.026 sec on CPU.
- parallel: n=103, accuracy=23.30, chance-adjusted=-0.023, rank counts={'2': 23, '3': 31, '4': 25, '1': 24}, choice counts={'1': 26, '2': 24, '0': 29, '3': 24}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=7.69, rank counts={'3': 18, '4': 21, '1': 4, '2': 9}, mean top-minus-correct=1.680, small wrong margins ≤0.25/≤0.50 nats=4/8
- nonparallel: n=100, accuracy=45.00, chance-adjusted=-0.100, rank counts={'2': 55, '1': 45}, choice counts={'0': 63, '1': 37}

Files:
- combined JSON: `experiments/archive/representation_and_objectives/data/muon_switch_50m_globalpiqa_margin/globalpiqa_margin_reader_results.json`
- per-target JSON/CSV under `experiments/archive/representation_and_objectives/data/muon_switch_50m_globalpiqa_margin`
