# fw globalpiqa relevant substrate — GlobalPIQA option-margin reader

This CPU-only diagnostic mirrors the official MLM length-normalized completion scoring for GlobalPIQA and records all candidate scores. It is not a custom submission scorer and must not be tuned on official rows.

## `cr_a0p25`

Model: `experiments/archive/representation_and_objectives/training/runs/fw_weight_space_sweep/cr_a0p25/hf_model` revision `chck_100M`; load 0.094 sec on CPU.
- parallel: n=103, accuracy=23.30, chance-adjusted=-0.023, rank counts={'2': 22, '3': 33, '4': 24, '1': 24}, choice counts={'1': 30, '0': 27, '3': 29, '2': 17}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=0.00, rank counts={'3': 21, '2': 10, '4': 21}, mean top-minus-correct=1.467, small wrong margins ≤0.25/≤0.50 nats=5/14
- nonparallel: n=100, accuracy=48.00, chance-adjusted=-0.040, rank counts={'1': 48, '2': 52}, choice counts={'1': 46, '0': 54}

## `cir_i0p25_r0p25`

Model: `experiments/archive/representation_and_objectives/training/runs/fw_weight_space_sweep/cir_i0p25_r0p25/hf_model` revision `chck_100M`; load 0.029 sec on CPU.
- parallel: n=103, accuracy=19.42, chance-adjusted=-0.074, rank counts={'3': 41, '4': 23, '1': 20, '2': 19}, choice counts={'1': 28, '0': 26, '2': 23, '3': 26}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=3.85, rank counts={'4': 18, '3': 26, '1': 2, '2': 6}, mean top-minus-correct=1.384, small wrong margins ≤0.25/≤0.50 nats=4/12
- nonparallel: n=100, accuracy=51.00, chance-adjusted=0.020, rank counts={'1': 51, '2': 49}, choice counts={'1': 51, '0': 49}

Files:
- combined JSON: `experiments/archive/representation_and_objectives/data/weight_sweep_rowblock_screen/globalpiqa_margin/globalpiqa_margin_reader_results.json`
- per-target JSON/CSV under `experiments/archive/representation_and_objectives/data/weight_sweep_rowblock_screen/globalpiqa_margin`
