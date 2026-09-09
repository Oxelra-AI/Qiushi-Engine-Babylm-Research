# fw 70m execution synthesis — FW 70M GlobalPIQA margin synthesis

## Implementation repair

The first wrapper invocation passed the local `hf_model` parent plus `revision=chck_70M`. That parent also contains final weights, so the first run did not faithfully read the 70M checkpoint. The repaired wrapper points directly at `hf_model/chck_70M` with `revision=main`. The final all-option scores reproduce the official 70M GlobalPIQA_parallel value 26.2136 for both arms.

## Arm-level parallel margins

| arm | parallel acc | rank1/2/3/4 | all mean top-correct | hard52 acc | hard52 rank1/2/3/4 | hard52 mean top-correct | hard52 <=0.50 nats |
|---|---:|---:|---:|---:|---:|---:|---:|
| `compact_view_70M` | 26.21 | 27/26/24/26 | 1.036 | 3.85 | 2/12/17/21 | 1.703 | 6 |
| `source_breadth_rowblock_70M` | 26.21 | 27/27/33/16 | 0.919 | 1.92 | 1/15/25/11 | 1.476 | 9 |

## Compact versus row-block breadth movement

| subset | n | compact correct | breadth correct | breadth-compact correct | breadth better/same/worse rank | breadth lower/higher margin | mean margin delta | median margin delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all103` | 103 | 27 | 27 | 0 | 32/49/22 | 48/37 | -0.117 | 0.000 |
| `hard52` | 52 | 2 | 1 | -1 | 18/24/10 | 30/22 | -0.227 | -0.086 |
| `other51` | 51 | 25 | 26 | 1 | 14/25/12 | 18/15 | -0.004 | 0.000 |

## Scientific reading

At 70M, row-block breadth does not improve official GlobalPIQA_parallel accuracy: both arms are 26.21, and on the fw globalpiqa relevant substrate hard52 set compact is 2/52 while breadth is 1/52. This blocks any interpretation that the row-block breadth EWoK gain has already solved the load-bearing GlobalPIQA weakness.

However, the all-option margins reveal a softer movement than exact accuracy: breadth lowers the hard52 mean top-minus-correct margin from 1.703 to 1.476 nats, reduces hard52 rank4 from 21 to 11, and increases hard52 rows within 0.50 nats from 6 to 9. This is mechanistically meaningful as a weak probability-mass shift toward conditional physical/spatial/temporal answers, but it is far too small for SOTA and coexists with worse 70M cheap7 through Entity and nonparallel GlobalPIQA losses.

The next evidence should therefore remain the complete 100M cheap/full vector when available. If 100M cheap7 remains far below the leader, the FW row-block breadth route should not consume full official evaluation just because it softened margins. If a 100M arm is near range, the same direct-checkpoint margin wrapper should be run on 100M before interpreting GlobalPIQA movement.

## Files

- JSON: `experiments/archive/representation_and_objectives/data/fw_70m_globalpiqa_margin_synthesis/fw_70m_globalpiqa_margin_synthesis.json`
- arm summary CSV: `experiments/archive/representation_and_objectives/data/fw_70m_globalpiqa_margin_synthesis/fw_70m_globalpiqa_margin_summary.csv`
- subset delta CSV: `experiments/archive/representation_and_objectives/data/fw_70m_globalpiqa_margin_synthesis/fw_70m_globalpiqa_margin_subset_deltas.csv`
- row delta CSV: `experiments/archive/representation_and_objectives/data/fw_70m_globalpiqa_margin_synthesis/fw_70m_globalpiqa_margin_row_deltas.csv`
- historical hard-row reference CSV: `experiments/archive/representation_and_objectives/data/fw_70m_globalpiqa_margin_synthesis/fw_70m_vs_hard_margin_reference.csv`
