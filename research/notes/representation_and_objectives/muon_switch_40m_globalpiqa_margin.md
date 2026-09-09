# fw globalpiqa relevant substrate — GlobalPIQA option-margin reader

This CPU-only diagnostic mirrors the official MLM length-normalized completion scoring for GlobalPIQA and records all candidate scores. It is not a custom submission scorer and must not be tuned on official rows.

## `adamw40`

Model: `experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_40M` revision `main`; load 0.091 sec on CPU.
- parallel: n=103, accuracy=26.21, chance-adjusted=0.016, rank counts={'1': 27, '3': 27, '4': 24, '2': 25}, choice counts={'3': 25, '1': 29, '0': 22, '2': 27}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=1.92, rank counts={'3': 19, '4': 19, '2': 13, '1': 1}, mean top-minus-correct=1.730, small wrong margins ≤0.25/≤0.50 nats=2/9
- nonparallel: n=100, accuracy=51.00, chance-adjusted=0.020, rank counts={'1': 51, '2': 49}, choice counts={'1': 47, '0': 53}

## `continuous_muon40`

Model: `experiments/archive/frontier_consolidation/training/runs/muon_lr008_wd00125_seed43022_80M/hf_model/chck_40M` revision `main`; load 0.026 sec on CPU.
- parallel: n=103, accuracy=22.33, chance-adjusted=-0.036, rank counts={'2': 24, '4': 28, '3': 28, '1': 23}, choice counts={'1': 31, '0': 26, '3': 22, '2': 24}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=1.92, rank counts={'4': 21, '3': 19, '2': 11, '1': 1}, mean top-minus-correct=1.872, small wrong margins ≤0.25/≤0.50 nats=5/8
- nonparallel: n=100, accuracy=45.00, chance-adjusted=-0.100, rank counts={'1': 45, '2': 55}, choice counts={'1': 49, '0': 51}

## `muon20toadamw40`

Model: `experiments/archive/frontier_consolidation/training/runs/muon20toadamw_seed43022_80M/hf_model/chck_40M` revision `main`; load 0.025 sec on CPU.
- parallel: n=103, accuracy=20.39, chance-adjusted=-0.061, rank counts={'2': 25, '4': 26, '1': 21, '3': 31}, choice counts={'1': 27, '0': 24, '2': 23, '3': 29}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=1.92, rank counts={'4': 20, '2': 9, '3': 22, '1': 1}, mean top-minus-correct=1.841, small wrong margins ≤0.25/≤0.50 nats=2/8
- nonparallel: n=100, accuracy=56.00, chance-adjusted=0.120, rank counts={'1': 56, '2': 44}, choice counts={'1': 44, '0': 56}

## `muon40toadamw40`

Model: `experiments/archive/frontier_consolidation/training/runs/muon40toadamw_seed43022_80M/hf_model/chck_40M` revision `main`; load 0.026 sec on CPU.
- parallel: n=103, accuracy=22.33, chance-adjusted=-0.036, rank counts={'2': 24, '4': 28, '3': 28, '1': 23}, choice counts={'1': 31, '0': 26, '3': 22, '2': 24}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=1.92, rank counts={'4': 21, '3': 19, '2': 11, '1': 1}, mean top-minus-correct=1.872, small wrong margins ≤0.25/≤0.50 nats=5/8
- nonparallel: n=100, accuracy=45.00, chance-adjusted=-0.100, rank counts={'1': 45, '2': 55}, choice counts={'1': 49, '0': 51}

Files:
- combined JSON: `experiments/archive/representation_and_objectives/data/muon_switch_40m_globalpiqa_margin/globalpiqa_margin_reader_results.json`
- per-target JSON/CSV under `experiments/archive/representation_and_objectives/data/muon_switch_40m_globalpiqa_margin`
