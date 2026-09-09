# preservation controls and corrected interpretation preservation KL mode probe

Status: `PRESERVATION_KL_MODE_PROBE_DONE`
Limited examples/targets: `8` / `244`

Implemented orientation: `KL(teacher || student)` via `F.kl_div(log_softmax(student), softmax(teacher))`.

## Mode results
- eval_eval: KL teacher||student mean `0.00000000`, reverse `0.00000000`, student CE `1.749682`, teacher CE `1.749682`, tokens `244`
- train_eval: KL teacher||student mean `0.18062807`, reverse `0.21511504`, student CE `1.875261`, teacher CE `1.749682`, tokens `244`
- train_eval_repeat_same_seed: KL teacher||student mean `0.18062807`, reverse `0.21511504`, student CE `1.875261`, teacher CE `1.749682`, tokens `244`
- train_eval_different_seed: KL teacher||student mean `0.16987394`, reverse `0.19060477`, student CE `1.827566`, teacher CE `1.749682`, tokens `244`
- train_train_same_seed_reset: KL teacher||student mean `0.21295691`, reverse `0.22951475`, student CE `1.882552`, teacher CE `1.847321`, tokens `244`
- eval_train: KL teacher||student mean `0.21511504`, reverse `0.18062807`, student CE `1.749682`, teacher CE `1.875261`, tokens `244`

The train/eval identical-weight value is stochastic regularization pressure, not acquired functional drift.
