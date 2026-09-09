# Depth Geometry as a Controlled Follow-Up

Status: historical training design; final depth results are recorded separately.

The two legal-40k 8x480 Overall scores were 41.140577774478444 and 40.42013171935382, with mean 40.78035474691613. This motivated a depth-over-width test rather than another nearby vocabulary size.

The proposed contrast changed 8 layers, hidden size 480, 8 heads and intermediate size 1920 to 12 layers, hidden size 384, 12 heads and intermediate size 1280. The latter had 38,421,952 parameters. The compact-view data, legal-40k tokenizer, fixed WWM 0.15, sequence length 256, AdamW learning rate 0.001, warmup 0.06, weight decay 0.01, effective batch 256 through microbatch 64, data-order seed 43 and initialization/training seeds 43022/43023 were retained.

The first-seed result would decide whether a second seed was justified: coherent gains without a new collapse would support reproduction; a flat or worse result would not. GlobalPIQA, Entity and relational/stateful task movement mattered alongside the complete vector. Early MLM loss was not the decision metric.

Evaluation was specified on the same 7,618-row EWoK coordinate and the AoA ladder with 19 checkpoints and 8,005 rows per checkpoint, using minimum context zero. These were planned evidence requirements, not measurements of the new model at this stage.
