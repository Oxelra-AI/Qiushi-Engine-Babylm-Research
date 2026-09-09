# masking schedule preflight and resource decision WWM-to-token training measurement

This CPU-only measurement prepares the orthogonal masking-schedule route without using an H100. It compares the completed WWM->token run with the COMPACT_EXPERIENCE clean-Qwen fixed-WWM run.

## Shared setup

- Target run: `experiments/archive/representation_and_objectives/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022`
- Fixed-WWM reference: `experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022`
- Both report 100M word exposure and 2515 training steps with baseline16k / DeBERTa-v2 8x480 / clean-Qwen data.

## Training trace comparison

- First non-elapsed training-log difference appears at earlier analysis.
- Rows are identical through earlier analysis, cumulative exposure 70023296 words, with mask mode `wwm`.
- WWM->token first token-masking row: earlier analysis, cumulative exposure 70062953, loss 2.4968814849853516.
- Target mask-mode counts: {'wwm': 1761, 'token': 754}.
- Fixed-WWM mask-mode counts: {'wwm': 2515}.

## Loss windows

- pre_switch_last_100_steps_target: n=100, first=2.505844, last=2.611357, mean=2.589173.
- pre_switch_last_100_steps_fixed_wwm: n=100, first=2.505844, last=2.611357, mean=2.589173.
- post_switch_first_100_steps_target: n=100, first=2.496881, last=2.341065, mean=2.354555.
- post_switch_same_100_steps_fixed_wwm: n=100, first=2.665667, last=2.542888, mean=2.549343.
- final_100_steps_target: n=100, first=2.287349, last=2.266498, mean=2.212637.
- final_100_steps_fixed_wwm: n=100, first=2.489538, last=2.507568, mean=2.450531.

The post-switch token objective gives a much lower MLM loss, but this changes the prediction target and cannot be read as BabyLM competence by itself.

## Smallest downstream conversion

Evaluate only post-switch checkpoints `chck_80M, chck_90M, chck_100M` first. The optional identity anchor is `chck_70M`. This should decide whether the leader-style WWM7->Token3 factor is worth combining with whichever data mechanism survives.

JSON: `experiments/archive/representation_and_objectives/data/wwm_to_token_training_measurement/wwm_to_token_training_measurement.json`
