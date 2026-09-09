# preservation controls and corrected interpretation preservation drift profile

Status: `PRESERVATION_DRIFT_PROFILE_DONE`
Rendering: ordinary 15% WWM on full packed Qwen pair rows; source evidence remains present
Examples/targets/words: `384` / `10348` / `51945`

## Model results
- coherent86: KL(t||s) `0.00000000`, ΔCE(s-t) `0.000000`, Δrank(s-t) `0.000`, CE-improved `0.000`, rank-improved `0.000`
- ordinary_inherited_wwm_seed62064: KL(t||s) `0.00055117`, ΔCE(s-t) `-0.002291`, Δrank(s-t) `-0.737`, CE-improved `0.478`, rank-improved `0.078`
- densemask_sparselabel_seed62064: KL(t||s) `0.02448639`, ΔCE(s-t) `0.028969`, Δrank(s-t) `5.882`, CE-improved `0.473`, rank-improved `0.094`
- clean_pres_lambda1_eval_seed62064: KL(t||s) `0.00941133`, ΔCE(s-t) `0.009986`, Δrank(s-t) `2.432`, CE-improved `0.468`, rank-improved `0.090`
- clean_pres_lambda1_eval_seed62065: KL(t||s) `0.00945284`, ΔCE(s-t) `0.009880`, Δrank(s-t) `2.381`, CE-improved `0.464`, rank-improved `0.092`

This is a preservation-target relevance probe, not official BabyLM scoring.
