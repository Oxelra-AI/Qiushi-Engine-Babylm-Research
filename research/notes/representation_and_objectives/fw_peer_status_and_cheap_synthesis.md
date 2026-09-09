# fw 70m ewok prediction overlap — FW peer training and cheap-eval status

## Training artifacts

- **compact_view** (same-source compact rewrite companion): status `READY`, chck_100M=True, checkpoint dirs=100, tokenizer SHA `e70d167f62066813`; word exposure=100000000, steps=2508, loss 9.790523529052734 → 2.466791868209839. Errors: []
- **source_breadth_rowblock** (whole-sentence independent-source breadth companion, row-block layout): status `READY`, chck_100M=True, checkpoint dirs=100, tokenizer SHA `e70d167f62066813`; word exposure=100000000, steps=2508, loss 9.788363456726074 → 2.4748852252960205. Errors: []

## Available cheap-eval files

- 70M compact_view: 7/7 cheap columns, cheap7=42.6571, path `experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/compact_view_chck_70M.json`
- 70M source_breadth_rowblock: 7/7 cheap columns, cheap7=42.0314, path `experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/source_breadth_chck_70M.json`
- 80M compact_view: 4/7 cheap columns, cheap7=NA, path `experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/compact_view_chck_80M.json`
- 80M source_breadth_rowblock: 0/7 cheap columns, cheap7=NA, path `experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/source_breadth_chck_80M.json`
- 100M compact_view: 0/7 cheap columns, cheap7=NA, path `experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/compact_view_chck_100M.json`
- 100M source_breadth_rowblock: 0/7 cheap columns, cheap7=NA, path `experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/source_breadth_chck_100M.json`

## Paired deltas now safe to read

- 70M: compact-breadth cheap7 = +0.6257; compact-ref = +0.0485; breadth-ref = -0.5772.
- 80M: only 0 common cheap columns (); not a cheap7 mechanism decision yet.
- 100M: only 0 common cheap columns (); not a cheap7 mechanism decision yet.

## Evidence files

- JSON: `experiments/archive/representation_and_objectives/data/fw_peer_status/fw_peer_status_and_cheap_synthesis.json`
- Score CSV: `experiments/archive/representation_and_objectives/data/fw_peer_status/fw_peer_cheap_scores.csv`
- Delta CSV: `experiments/archive/representation_and_objectives/data/fw_peer_status/fw_peer_cheap_deltas.csv`
