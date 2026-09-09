# rowchunk pairfit screen and route decision — row-chunked vs pairfit matched screen

## Why this experiment mattered

The previous exposure repair showed that the repaired pilot eval row-chunked stagewise data removed hidden-word waste, but mechanism preservation summary showed a different mechanism problem: row chunking broke the clean-Qwen paired-view signal by placing many original+rewrite pairs in different attention windows. The pairfit corpus fixed exactly that representation problem while keeping the same training recipe:

- DeBERTa-v2 12×384, 40k tokenizer
- LAMB lr 0.007, weight decay 0.01
- stagewise seq64/batch512 → seq128/batch256 → seq256/batch128
- WWM until 7M word exposure, then token masking
- bf16 forward, same initialization/training seeds
- 10M total word exposure and only `chck_10M` saved

The comparison isolates whether preserving aligned semantic dependency units under the leader-style curriculum is a large enough lever to keep investing in this curriculum arm.

## Training completion

Row-chunked arm:

- run: `experiments/archive/frontier_consolidation/training/runs/qwen10_stagewise_rowchunk_40k_12x384_lamb_bf16_seed44011`
- checkpoint: `hf_model/chck_10M`
- params: 34,677,184
- steps: 612
- elapsed: 1515.5 s
- loss: 10.7238 → 3.9474
- WWM→token switch occurred at 7M exposure

Pairfit arm:

- run: `experiments/archive/frontier_consolidation/training/runs/qwen10_stagewise_pairfit_40k_12x384_lamb_bf16_seed44011`
- checkpoint: `hf_model/chck_10M`
- params: 34,677,184
- steps: 585
- elapsed: 433.6 s on free GPU1
- loss: 10.7248 → 3.8795
- WWM→token switch occurred at 7M exposure

Pairfit preserved all 37,594 Qwen pairs exactly once and restored complete pair joint visibility to 1.0; row-chunked complete pair joint visibility was only 0.572006 overall.

## Fast official-compatible screen

Summary JSON:

`experiments/archive/frontier_consolidation/data/repaired_pilot_eval/repaired_pilot_eval_summary.json`

Per-target raw files:

- row-chunked: `experiments/archive/frontier_consolidation/data/repaired_pilot_eval/per_target/stagewise_qwen10_12x384_40k_seed44011.json`
- pairfit: `experiments/archive/frontier_consolidation/data/repaired_pilot_eval/per_target/pairfit_qwen10_12x384_40k_seed44011.json`

| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA mean | Reading | equal7 fast | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| row-chunked | 53.95 | 55.60 | 48.18 | 17.29 | 16.35 | 49.44 | 35.635 | 6.795 | 38.127 | 37.993 |
| pairfit | 54.86 | 56.80 | 48.36 | 17.30 | 17.43 | 49.39 | 33.725 | 6.750 | 38.169 | 38.188 |

Pairfit minus row-chunked:

- BLiMP +0.91
- Supplement +1.20
- EWoK +0.18
- Entity fast +0.01
- Entity full +1.08
- COMPS −0.05
- GlobalPIQA parallel −5.82
- GlobalPIQA nonparallel +2.00
- GlobalPIQA mean −1.91
- Reading −0.045
- equal7 fast +0.042
- equal7 full-Entity +0.195

## Scientific interpretation

Preserving original+rewrite joint visibility is mechanistically cleaner and modestly improves BLiMP, Supplement, EWoK, and full Entity in this 10M screen, but the improvement is small relative to the SOTA gap and is cancelled or nearly cancelled by GlobalPIQA and Reading tradeoffs. The result rules out the idea that the poor row-chunked screen was mainly caused by broken pair visibility and that simply restoring semantic dependency-unit visibility inside the same leader-style curriculum is a major shortcut to 41.8–42.0.

The 12×384/40k/LAMB/seq-curriculum arm remains a useful control, but it should not be the main frontier_consolidation route. The immediate next research should return to the higher-value data-efficiency problem identified in earlier analysis/earlier analysis: building faithful denser second views that expose many more distinct source instances per generated word while preserving entities, roles, negation, modality, and relation structure. Any new dense-view construction should be tested by strict transformation audits before H100 training and then by a low-cost official-task screen, rather than scaling another secondary curriculum variant.

## Resource note

The row-chunked training completed successfully. GPU0 continued to report a large allocation with no visible compute process, but inference on the row-chunked checkpoint succeeded there. GPU1 became free and was used for the pairfit training and evaluation.
