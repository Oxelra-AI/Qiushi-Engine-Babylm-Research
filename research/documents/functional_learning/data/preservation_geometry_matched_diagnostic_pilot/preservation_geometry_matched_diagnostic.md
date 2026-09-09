# earlier analysis preservation-geometry matched diagnostic

Created: `2026-09-08T07:10:59Z`

This diagnostic is run at the exact acquisition-only `(M,S)` checkpoint. It compares clean's ordinary-WWM preservation rendering with a dense-corrupted rendering before launching any full preservation-geometry training arm.

## Selected acquisition macros

- Macro indices (0-based): `[0]`
- Selected rows: `252`; Qwen rows: `33`; selected words: `39652`

## Target support

| quantity | value |
|---|---:|
| ordinary_full_targets | 972 |
| dense_masked_tokens | 1598 |
| dense_sparse_label_targets | 247 |
| dense_nonlabel_targets | 1351 |
| common_targets | 213 |
| ordinary_common_examples | 33 |
| ordinary_full_examples | 33 |
| dense_nonlabel_full_examples | 33 |

The common-support comparison uses `common_targets = ordinary-WWM target positions ∩ dense-corruption masked non-label positions`; KL is evaluated at the same token positions under the two input renderings.

## Acquisition gradients

| gradient | loss mean | L2 norm | focus targets | ordinary targets |
|---|---:|---:|---:|---:|
| acquisition_full_weighted | 2.65498 | 0.140029 | 247 | 7894 |
| acquisition_focus_mean | 3.61011 | 0.825218 | 247 | 7894 |

## Preservation branches

| branch | tokens | KL(t||s) | teacher entropy | teacher top1 | teacher target CE | student target CE | grad L2 | cos vs full acquisition | cos vs focus acquisition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary_common | 213 | 0.0275071 | 2.27339 | 0.628811 | 2.63034 | 2.60425 | 0.129201 | 0.174899 | -0.28696 |
| dense_common | 213 | 0.532946 | 4.26484 | 0.382131 | 4.45987 | 3.66683 | 1.63282 | 0.246058 | -0.383304 |
| ordinary_full | 972 | 0.0248768 | 1.7365 | 0.698845 | 1.82339 | 1.86285 | 0.0966362 | 0.24336 | -0.340992 |
| dense_nonlabel_full | 1351 | 0.485378 | 4.19375 | 0.382879 | 4.62777 | 3.94329 | 1.36948 | 0.273684 | -0.395645 |

## Direct rendering contrast on common support

{
  "dense_minus_ordinary_teacher_entropy": 1.991452118600478,
  "dense_minus_ordinary_teacher_top1_prob": -0.24668044551437446,
  "dense_minus_ordinary_teacher_target_ce": 1.829529471240693,
  "dense_minus_ordinary_student_target_ce": 1.0625869947979707,
  "dense_minus_ordinary_kl_teacher_to_student": 0.5054390490614752,
  "dense_over_ordinary_grad_l2": 12.637903696441958,
  "cos_ordinary_dense_common_grad": 0.6060449834311841,
  "cos_ordinary_common_vs_acq_full": 0.17489922724145243,
  "cos_dense_common_vs_acq_full": 0.2460578373776931,
  "cos_ordinary_common_vs_acq_focus": -0.2869598723679284,
  "cos_dense_common_vs_acq_focus": -0.38330424897825793
}

## Interpretation

The common-support branch separates input rendering from target support: the same token positions are preserved under ordinary-WWM context and dense-corrupted context. Any difference in teacher entropy, KL, or gradient alignment here is a prediction-state geometry effect rather than a support-count effect for these positions. On the selected macros, dense-common KL has 12.6× the ordinary-common private-gradient L2. Its cosine with the full acquisition gradient is 0.2460578373776931, versus 0.17489922724145243 for ordinary-common. The full dense-nonlabel branch remains a joint rendering-and-support intervention because its target set differs from clean ordinary WWM. It should be interpreted through both the full-support measurements and the common-support diagnostic, not as a pure geometry isolation. Because gradients are evaluated at the acquisition-shifted exact `(M,S)` endpoint, nonzero KL reflects actual drift from the coherent86 parent. The same test at initialization would be uninformative: teacher and student predictions coincide and the deterministic KL gradient is zero.

This file is a local gradient/uncertainty diagnostic. It does not change the repaired BabyLM coordinate and should not be used as a benchmark score.
