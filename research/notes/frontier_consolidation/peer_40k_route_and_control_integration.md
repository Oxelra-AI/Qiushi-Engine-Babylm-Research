# legal tokenizer clean control trajectory design integration of representation route with matched control

## action

A memory-safe clean-Qwen control trajectory through 80M was launched under the legal tokenizer:

- run dir: `experiments/archive/frontier_consolidation/training/runs/complianttok_cleanqwen_seed43022_80M`
- wrapper: `experiments/archive/frontier_consolidation/scripts/wait_and_train_clean_control_80m.py`
- retained checkpoints: every 1M word through 80M, including `chck_20M`, `chck_70M`, `chck_80M`
- purpose: determine whether compact-view reinvestment still beats clean-Qwen under the same legal spatial repair route status tokenizer at early and mature exposures

Matched reinvest reference evaluation was also launched at mature checkpoints:

- script: `experiments/archive/frontier_consolidation/scripts/eval_reinvest_mature_refs.py`
- outputs: `experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json` and `_80M.json`

The planned clean-control evaluation is:

- script: `experiments/archive/frontier_consolidation/scripts/wait_and_eval_clean_control_refs.py`
- behavior: waits for the clean 80M trajectory to have `scientific_metrics.json` and `chck_20M`, `chck_70M`, `chck_80M`, then evaluates the same cheap official-compatible columns under isolated roots

CPU comparison script:

- `experiments/archive/frontier_consolidation/scripts/compare_legal_treatment_trajectory.py`
- output root: `experiments/archive/frontier_consolidation/data/legal_treatment_trajectory`

## Peer evidence received in legal tokenizer clean control trajectory design



The fully legal 16k/byte-like compact-view-reinvest endpoints remained below the 41.8 reference:

- seed43022 corrected-tokenizer compact-view-reinvest Overall 40.703956
- seed43122 corrected-tokenizer compact-view-reinvest Overall 41.023994
- two-seed mean 40.863975
- main mean losses versus the inherited-tokenizer mechanism coordinate: Supplement about -5.09 and EWoK about -3.08

early-prediction calibration directly supports the legal tokenizer clean control trajectory design choice not to run another 20M-only clean control:

- treatment-control equal7 deltas oscillated: +0.40 at 10M, -0.35 at 20M, +1.72 at 40M, -0.12 at 70M, +0.26 at 80M, +0.17 endpoint
- first cross-task delta Pearson above 0.7 appeared at `chck_70M`
- EWoK relation/seed information stabilized around `chck_80M`

representation map suggests legal 40k byte-BPE is a plausible next representation route because it is trained on the exact same 10M pool and reduces subword length while preserving WWM word boundaries:

- legal 16k tokens/word on the pool: 1.4669
- legal 40k tokens/word on the pool: 1.3943
- legal 40k reduces seq256 overflow rows from 15,143 to 11,407
- legal 40k evaluation-token ratios vs legal 16k: EWoK 0.9114, Supplement 0.9447, SuperGLUE 0.9234, BLiMP 0.8984
- all byte-BPE tokenizers keep WWM groups/word at exactly 1.0000

The legal40k compact-view-reinvest route was launched. Direct batch-256 training failed from memory pressure, so the implemented comparison uses effective batch 256, microbatch 64 and accumulation 4. The intended factor is vocabulary/segmentation/embedding change, but dropout realization and the floating-point path also differ from the full-batch 16k trainer. The accounting comparison supports matched example order, masking batches, optimizer schedule and loss weighting. A short accumulated-16k control remains proposed to bound the remaining implementation effect after the 40k training completes.

## How to use these facts

The same-tokenizer clean-versus-reinvest trajectory tests whether the compact-view reinvestment data effect survives the legal-tokenizer coordinate. It is scientifically distinct from the pending legal40k comparison.

Interpretation after results:

- If reinvest beats clean broadly at 70M/80M, then compactness plus source reinvestment remains a useful data principle in the legal coordinate. The route should preserve the data mechanism and look to legal representation/optimization repairs; 40k evidence becomes highly relevant.
- If reinvest does not beat clean at 70M/80M, then the legal-tokenizer failure is not merely an endpoint representation problem. The next route should modify the learning signal while preserving the strict data budget, with paired source-view consistency or information/relation-weighted masking as the first serious candidates.
- The 20M comparison is only early behavior evidence and should not decide the route by itself.

Relevant files read in legal tokenizer clean control trajectory design:

- `research/notes/representation_and_objectives/corrected_tokenizer_official_vectors_and_route_decision.md`
- `research/notes/representation_and_objectives/early_prediction_calibration.md`
- `research/notes/representation_and_objectives/fast_representation_map.md`
- `research/notes/representation_and_objectives/route_recommendation.md`
- `research/notes/representation_and_objectives/legal40k_launch_and_memory_repair.md`
- `research/notes/representation_and_objectives/accum_trainer_equivalence_and_confound.md`
