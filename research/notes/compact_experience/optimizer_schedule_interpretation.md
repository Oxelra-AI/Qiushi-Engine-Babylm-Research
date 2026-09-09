# highlr 10M training completion — schedule-horizon interpretation for optimizer interaction screen

The optimizer by paired data experiment high-LR 10M optimizer × corpus run is internally matched across AdamW/LAMB and official/Qwen, so its column-wise interaction

\[
I_t = [S(\text{LAMB,Qwen},t)-S(\text{LAMB,Official},t)] - [S(\text{AdamW,Qwen},t)-S(\text{AdamW,Official},t)]
\]

is meaningful within that compact experimental coordinate.

However, the 10M trainer uses `get_cosine_schedule_with_warmup(..., num_training_steps=total_steps)` with `total_steps` determined by the selected-word horizon. Thus the `chck_10M` checkpoint in the 10M screen is a short-horizon endpoint where the LR has decayed to near zero. It is **not** the same dynamical state as `chck_10M` inside a 20M or 100M schedule, where the cosine schedule would still be mid-training. This does not invalidate the compact screen, but it changes the interpretation:

- Read the 10M result as a low-cost matched interaction probe, not as a direct miniature of the 100M learning trajectory.
- A positive 10M interaction should motivate a fresh 20M four-arm run using a 20M schedule, not a continuation from the 10M endpoint.
- A negative 10M interaction is strong evidence against a large, easy optimizer×paired-data effect at this coordinate, but if it is only mildly negative/noisy and the column profile is ambiguous, direction choice should consider the known fact that clean-Qwen effects can emerge later.
- No 100M run should be launched from loss acceleration or a single favorable short-horizon checkpoint.

The repaired 20M launcher `scripts/launch_20M_optimizer_interaction.sh high` creates fresh 20M runs with a 20M schedule; its repaired evaluator `scripts/eval_20M_optimizer_interaction.sh 0.007` reads chck_5M/chck_10M/chck_20M under that schedule.
