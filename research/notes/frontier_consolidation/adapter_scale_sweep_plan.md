# adapter scale sweep plan adapter endpoint-sensitivity plan

The adapter matched horizon plan matched-horizon adapter128 result corrected the adapter 20M closure schedule artifact but left a functional tradeoff at 20M: BLiMP/Supplement/COMPS improved while EWoK/GlobalPIQA/Reading declined. The interpretation of adapter-off requires a qualification: disabling the branch after joint training is not a clean separation of direct branch output from backbone drift, because the entire checkpoint is co-adapted. This analysis therefore treats adapter-scale variants as an **endpoint sensitivity map** only.

## Lowest-cost question before more training

A faithful 50M ordinary-adapter maturation run would consume a long training prefix. A no-training scale sweep on the existing live20M checkpoint is cheaper and can decide whether that expenditure is worthwhile:

- If an interior adapter scale improves cheap7 and especially restores EWoK/GlobalPIQA/Reading while retaining BLiMP/Supplement gains, residual capacity may need amplitude control; then a maturation run or controlled-scale training is scientifically justified.
- If all nonzero scales preserve the same tradeoff or fall below spatial repair route status, ordinary post-layer residual capacity is not currently a promising 50M route; the next construction should change coupling/amplitude rather than merely train longer.
- If scale=0 is better than live, this alone does not prove that gradient isolation or frozen-backbone training is the remedy; it only shows endpoint sensitivity under a co-adapted system.

## Execution

Create shadow checkpoints from `training/runs/adapter128_live_h100M20M_seed43022/hf_model/chck_20M` with identical weights and tokenization but `adapter_scale` in {0, .25, .5, .75, 1, 1.25, 1.5}. Evaluate the official-compatible cheap columns: BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA_parallel/nonparallel, and Reading. Compare against spatial repair route status/disabled 20M and adapter matched horizon plan live128.

This is not a final endpoint and not submission work; it is a cheap decision object for the residual-capacity route.
