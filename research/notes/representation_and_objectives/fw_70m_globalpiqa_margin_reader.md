# fw globalpiqa relevant substrate — GlobalPIQA option-margin reader

This CPU-only diagnostic mirrors the official MLM length-normalized completion scoring for GlobalPIQA and records all candidate scores. It is not a custom submission scorer and must not be tuned on official rows.

## `a02_fw_compact_70M_seed43022`

Model: `experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_70M` revision `main`; load 0.097 sec on CPU.
- parallel: n=103, accuracy=26.21, chance-adjusted=0.016, rank counts={'3': 24, '2': 26, '4': 26, '1': 27}, choice counts={'1': 32, '0': 24, '3': 26, '2': 21}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=3.85, rank counts={'3': 17, '2': 12, '4': 21, '1': 2}, mean top-minus-correct=1.703, small wrong margins ≤0.25/≤0.50 nats=1/6

## `a02_fw_breadth_rowblock_70M_seed43022`

Model: `experiments/archive/frontier_consolidation/training/runs/fw_source_breadth_shared16k_seed43022/hf_model/chck_70M` revision `main`; load 0.031 sec on CPU.
- parallel: n=103, accuracy=26.21, chance-adjusted=0.016, rank counts={'4': 16, '3': 33, '2': 27, '1': 27}, choice counts={'1': 30, '0': 21, '3': 25, '2': 27}
  - 52-row cross-endpoint always-wrong subset in this target: accuracy=1.92, rank counts={'3': 25, '2': 15, '1': 1, '4': 11}, mean top-minus-correct=1.476, small wrong margins ≤0.25/≤0.50 nats=2/9

Files:
- combined JSON: `experiments/archive/representation_and_objectives/data/fw_70m_globalpiqa_margin_reader/globalpiqa_margin_reader_results.json`
- per-target JSON/CSV under `experiments/archive/representation_and_objectives/data/fw_70m_globalpiqa_margin_reader`
