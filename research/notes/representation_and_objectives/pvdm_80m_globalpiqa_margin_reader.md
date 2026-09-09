# fw globalpiqa relevant substrate — GlobalPIQA option-margin reader

This CPU-only diagnostic mirrors the official MLM length-normalized completion scoring for GlobalPIQA and records all candidate scores. It is not a custom submission scorer and must not be tuned on official rows.

## `pvdm_treatment_80m`

Model: `experiments/archive/representation_and_objectives/training/runs/pvdm_treatment_70M_to_80M_seed43022/hf_model` revision `chck_80M`; load 0.088 sec on CPU.
- parallel: n=103, accuracy=22.33, chance-adjusted=-0.036, rank counts={'2': 27, '4': 26, '3': 27, '1': 23}, choice counts={'1': 28, '0': 23, '3': 27, '2': 25}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=3.85, rank counts={'4': 23, '3': 20, '2': 7, '1': 2}, mean top-minus-correct=1.820, small wrong margins ≤0.25/≤0.50 nats=2/8
- nonparallel: n=100, accuracy=48.00, chance-adjusted=-0.040, rank counts={'1': 48, '2': 52}, choice counts={'1': 42, '0': 58}

## `pvdm_control_80m`

Model: `experiments/archive/representation_and_objectives/training/runs/pvdm_control_70M_to_80M_seed43022/hf_model` revision `chck_80M`; load 0.026 sec on CPU.
- parallel: n=103, accuracy=24.27, chance-adjusted=-0.010, rank counts={'2': 34, '4': 16, '1': 25, '3': 28}, choice counts={'1': 24, '0': 24, '3': 30, '2': 25}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=5.77, rank counts={'4': 15, '2': 14, '3': 20, '1': 3}, mean top-minus-correct=1.603, small wrong margins ≤0.25/≤0.50 nats=2/10
- nonparallel: n=100, accuracy=51.00, chance-adjusted=0.020, rank counts={'2': 49, '1': 51}, choice counts={'0': 59, '1': 41}

Files:
- combined JSON: `experiments/archive/representation_and_objectives/data/pvdm_80m_readouts/globalpiqa_margin/globalpiqa_margin_reader_results.json`
- per-target JSON/CSV under `experiments/archive/representation_and_objectives/data/pvdm_80m_readouts/globalpiqa_margin`
