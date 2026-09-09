# fw weight space sweep — FW same-initialization weight-space sweep

This is a low-cost branch-consolidation test, not a new 100M training branch. The scientific question is whether compact same-proposition recurrence and independent-breadth relation movement occupy compatible directions in one aligned checkpoint.

## Endpoint geometry

- Parameters compared: 34467424
- ||interleaved - compact|| / ||compact|| = 0.782172
- ||rowblock - compact|| / ||compact|| = 0.789242
- cosine(interleaved-compact, rowblock-compact) = 0.537540

## Materialized recipes

- `ci_a0p25`: compact=0.75, interleaved=0.25; status=created; checkpoint=`experiments/archive/representation_and_objectives/training/runs/fw_weight_space_sweep/ci_a0p25/hf_model/chck_100M`
- `ci_a0p50`: compact=0.5, interleaved=0.5; status=created; checkpoint=`experiments/archive/representation_and_objectives/training/runs/fw_weight_space_sweep/ci_a0p50/hf_model/chck_100M`
- `ci_a0p75`: compact=0.25, interleaved=0.75; status=created; checkpoint=`experiments/archive/representation_and_objectives/training/runs/fw_weight_space_sweep/ci_a0p75/hf_model/chck_100M`
- `cr_a0p25`: compact=0.75, rowblock=0.25; status=created; checkpoint=`experiments/archive/representation_and_objectives/training/runs/fw_weight_space_sweep/cr_a0p25/hf_model/chck_100M`
- `cir_i0p25_r0p25`: compact=0.5, interleaved=0.25, rowblock=0.25; status=created; checkpoint=`experiments/archive/representation_and_objectives/training/runs/fw_weight_space_sweep/cir_i0p25_r0p25/hf_model/chck_100M`

## Readout plan

First read Supplement, Entity, current 7,618-row EWoK, and GlobalPIQA parallel/nonparallel. A useful branch-consolidation signal would preserve compact-like Supplement/Entity while moving EWoK and GlobalPIQA toward the breadth arms. If the mixture only traces the compact↔breadth tradeoff, close this line and move to a changed relational objective or representation rather than more FW allocation variants.

Files:
- manifest: `experiments/archive/representation_and_objectives/data/fw_weight_space_sweep/weight_space_sweep_manifest.json`
- run root: `experiments/archive/representation_and_objectives/training/runs/fw_weight_space_sweep`
