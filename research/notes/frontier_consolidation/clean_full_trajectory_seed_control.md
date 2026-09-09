# clean full trajectory seed control clean-Qwen full-trajectory seed-control context

CPU-only summary from existing COMPACT_EXPERIENCE clean-Qwen checkpoint trajectory files. This is context for the reinvest seed-stability question, not a substitute for the matched fast-slice probe.

## Seed43122 minus seed43022 on selected full-trajectory exposures
- 10M: BLiMP=-0.890, Supplement=0.320, EWoK=2.480, Entity=1.170, COMPS=-0.460, GlobalPIQA=-0.500, Reading=-0.010, equal7_full_eval=0.301
- 40M: BLiMP=0.660, Supplement=-3.330, EWoK=-1.400, Entity=-4.400, COMPS=0.550, GlobalPIQA=1.430, Reading=-0.345, equal7_full_eval=-0.976
- 100M: BLiMP=-0.820, Supplement=-1.330, EWoK=0.240, Entity=-0.500, COMPS=0.280, GlobalPIQA=-2.000, Reading=-0.555, equal7_full_eval=-0.669

## Official 100M full surface from COMPACT_EXPERIENCE semantic repair frontier and refill plan
BLiMP=-0.820, Supplement=-1.330, EWoK=0.240, Entity=-0.500, COMPS=0.280, GlobalPIQA=-2.000, SuperGLUE=-1.563, Reading=-0.555, AoA=0.000, Overall=-0.694, NLP_average=-0.813

At 100M, clean-Qwen seed43122 is not an equal replicate of seed43022: it loses Overall -0.694, NLP_average -0.813, SuperGLUE -1.563, Supplement -1.330, BLiMP -0.820, Entity -0.500, GlobalPIQA -2.000, and Reading -0.555, while EWoK +0.240 and COMPS +0.280 rise slightly. This makes base-recipe seed spread a real alternative explanation for part of the reinvest seed gap.

The exact matched answer requires the clean full trajectory seed control clean sparse temporal run because it uses the same fast Supplement/EWoK slices, selected BLiMP files, Entity rows, and 1/10/40/100M exposures as the reinvest sparse probe.

Machine-readable output: `experiments/archive/frontier_consolidation/data/clean_full_trajectory_seed_control/clean_full_trajectory_seed_control.json`
