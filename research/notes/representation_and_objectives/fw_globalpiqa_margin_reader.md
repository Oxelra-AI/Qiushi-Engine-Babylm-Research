# fw globalpiqa relevant substrate — GlobalPIQA option-margin reader

This CPU-only diagnostic mirrors the official MLM length-normalized completion scoring for GlobalPIQA and records all candidate scores. It is not a custom submission scorer and must not be tuned on official rows.

## `fw_breadth_interleaved_fullbatch_seed43022`

Model: `experiments/archive/representation_and_objectives/training/runs/fw_source_breadth_interleaved_wholesentence_fullbatch_shared16k_seed43022/hf_model` revision `chck_100M`; load 0.095 sec on CPU.
- parallel: n=103, accuracy=26.21, chance-adjusted=0.016, rank counts={'3': 28, '2': 25, '4': 23, '1': 27}, choice counts={'1': 27, '0': 20, '3': 31, '2': 25}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=3.85, rank counts={'3': 19, '2': 13, '4': 18, '1': 2}, mean top-minus-correct=1.660, small wrong margins ≤0.25/≤0.50 nats=2/7
- nonparallel: n=100, accuracy=56.00, chance-adjusted=0.120, rank counts={'1': 56, '2': 44}, choice counts={'1': 44, '0': 56}

Files:
- combined JSON: `experiments/archive/representation_and_objectives/data/fw_globalpiqa_margin_reader/globalpiqa_margin_reader_results.json`
- per-target JSON/CSV under `experiments/archive/representation_and_objectives/data/fw_globalpiqa_margin_reader`
