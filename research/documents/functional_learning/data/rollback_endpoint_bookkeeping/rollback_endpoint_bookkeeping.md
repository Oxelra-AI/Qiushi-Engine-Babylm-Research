# earlier analysis rollback endpoint bookkeeping

This note separates NLL numerical reproduction from integer rank/tie differences in the earlier analysis frozen-bank rollback endpoint checks.

## rollback_alpha_0_vs_parent
- original combined pass: `True`
- NLL pass at 2e-05: `True`; max |ΔNLL| `0.0`
- exact rank pass: `True`; max |Δrank| `0.0`; rank<=1 `True`

## rollback_alpha_1_vs_densemask
- original combined pass: `False`
- NLL pass at 2e-05: `True`; max |ΔNLL| `1.049041748046875e-05`
- exact rank pass: `False`; max |Δrank| `1.0`; rank<=1 `True`

## Scientific reading

The repaired frozen-bank source comparison remains valid: the dramatic negative Qwen rollback exception from earlier analysis was a mixed-bank artifact. On one frozen bank, rollback alpha near the clean KL keeps positive Qwen source specificity and common source-following. The endpoint reproduction issue in the earlier analysis printed pass flag is bookkeeping: alpha=1 reproduces dense-mask NLLs to tiny tolerance, while one Qwen rank differs by one.
