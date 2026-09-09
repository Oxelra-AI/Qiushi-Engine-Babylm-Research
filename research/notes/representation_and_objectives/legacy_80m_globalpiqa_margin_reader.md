# fw globalpiqa relevant substrate — GlobalPIQA option-margin reader

This CPU-only diagnostic mirrors the official MLM length-normalized completion scoring for GlobalPIQA and records all candidate scores. It is not a custom submission scorer and must not be tuned on official rows.

## `standard_legacy_80m`

Model: `experiments/archive/representation_and_objectives/training/runs/standard_legacy_70M_to_80M_seed43022/hf_model` revision `chck_80M`; load 0.091 sec on CPU.
- parallel: n=103, accuracy=27.18, chance-adjusted=0.029, rank counts={'3': 34, '4': 22, '2': 19, '1': 28}, choice counts={'1': 29, '0': 22, '3': 27, '2': 25}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=3.85, rank counts={'4': 17, '2': 8, '3': 25, '1': 2}, mean top-minus-correct=1.800, small wrong margins ≤0.25/≤0.50 nats=1/4
- nonparallel: n=100, accuracy=53.00, chance-adjusted=0.060, rank counts={'1': 53, '2': 47}, choice counts={'1': 39, '0': 61}

Files:
- combined JSON: `experiments/archive/representation_and_objectives/data/legacy_80m_readouts/globalpiqa_margin/globalpiqa_margin_reader_results.json`
- per-target JSON/CSV under `experiments/archive/representation_and_objectives/data/legacy_80m_readouts/globalpiqa_margin`
