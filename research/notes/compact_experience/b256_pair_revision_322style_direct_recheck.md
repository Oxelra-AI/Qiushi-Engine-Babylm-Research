# b256 fixedseq pair eval raw — b256 pair earlier analysis-style direct recheck

JSON: `experiments/archive/compact_experience/data/b256_pair_eval/b256_pair_revision_322style_direct_recheck.json`
Log: `research/notes/compact_experience/b256_pair_revision_322style_direct_recheck.log`

This uses the INITIAL_MODEL_STUDIES earlier analysis invocation pattern: exact checkpoint directories as `--model_path_or_name`, no `revision_name`, batch size 64, and the same fast BLiMP/Supplement/EWoK/Entity/COMPS/Reading paths.

| checkpoint | BLiMP | Supplement | EWoK | Entity | COMPS | Reading | mean6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact_experience_b256_fixed_chck_100M | 67.36 | 57.20 | 50.91 | 24.95 | 52.72 | 8.230 | 43.562 |
| compact_experience_b256_wwm_to_token_chck_100M | 67.63 | 61.20 | 50.45 | 23.78 | 52.18 | 7.885 | 43.854 |

## WWM→token minus fixed

| BLiMP | Supplement | EWoK | Entity | COMPS | Reading | mean6 |
|---:|---:|---:|---:|---:|---:|---:|
| +0.27 | +4.00 | -0.46 | -1.17 | -0.54 | -0.345 | +0.292 |

## INITIAL_MODEL_STUDIES earlier analysis reference fixed-WWM chck_100M

INITIAL_MODEL_STUDIES fixed-WWM chck_100M: BLiMP 67.34, Supplement 65.20, EWoK 49.64, Entity 21.24, COMPS 53.11, Reading 7.330.
The fixed-WWM comparison earlier analysis-style direct score differs by Supplement -8.00, Entity +3.71, Reading +0.900; this localizes the Supplement mismatch away from the b256 fixedseq pair eval raw wrapper evaluator and toward training, software-environment or data-order differences unless another hidden environment factor is found.
