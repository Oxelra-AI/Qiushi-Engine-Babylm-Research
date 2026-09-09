# preservation controls and corrected interpretation preservation drift profile

Status: `PRESERVATION_DRIFT_PROFILE_DONE`
Rendering: ordinary 15% WWM on full packed Qwen pair rows; source evidence remains present
Examples/targets/words: `384` / `10348` / `51945`

## Model results
- coherent86: KL(t||s) `0.00000000`, ΔCE(s-t) `0.000000`, Δrank(s-t) `0.000`, CE-improved `0.000`, rank-improved `0.000`
- sparse_focus_seed62064: KL(t||s) `0.00690126`, ΔCE(s-t) `0.000919`, Δrank(s-t) `-0.578`, CE-improved `0.366`, rank-improved `0.098`
- densemask_sparselabel_seed62064: KL(t||s) `0.02448642`, ΔCE(s-t) `0.028969`, Δrank(s-t) `5.882`, CE-improved `0.473`, rank-improved `0.094`
- dense_focus_seed62064: KL(t||s) `0.02827254`, ΔCE(s-t) `0.031519`, Δrank(s-t) `6.282`, CE-improved `0.459`, rank-improved `0.096`
- pres_lambda1_trainmode_confounded: KL(t||s) `0.01381098`, ΔCE(s-t) `0.022016`, Δrank(s-t) `4.651`, CE-improved `0.533`, rank-improved `0.087`

## Comparisons
- densemask_minus_sparse: {"kl_teacher_to_student_mean_delta": 0.017585160352569035, "delta_ce_student_minus_teacher_delta": 0.02804977717803259, "delta_rank_student_minus_teacher_delta": 6.459122535755702}
- lambda1_minus_densemask: {"kl_teacher_to_student_mean_delta": -0.010675434721543791, "delta_ce_student_minus_teacher_delta": -0.006952852736488935, "delta_rank_student_minus_teacher_delta": -1.2309625048318518}

This is a preservation-target relevance probe, not official BabyLM scoring.
