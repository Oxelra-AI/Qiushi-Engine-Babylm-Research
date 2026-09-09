# earlier analysis preservation-geometry matched diagnostic

Created: `2026-09-08T07:12:47Z`

This diagnostic is run at the exact acquisition-only `(M,S)` checkpoint. It compares clean's ordinary-WWM preservation rendering with a dense-corrupted rendering before launching any full preservation-geometry training arm.

## Selected acquisition macros

- Macro indices (0-based): `[0, 20, 40, 60]`
- Selected rows: `1021`; Qwen rows: `172`; selected words: `158522`

## Target support

| quantity | value |
|---|---:|
| ordinary_full_targets | 4762 |
| dense_masked_tokens | 7956 |
| dense_sparse_label_targets | 1291 |
| dense_nonlabel_targets | 6665 |
| common_targets | 1003 |
| ordinary_common_examples | 169 |
| ordinary_full_examples | 172 |
| dense_nonlabel_full_examples | 172 |

The common-support comparison uses `common_targets = ordinary-WWM target positions ∩ dense-corruption masked non-label positions`; KL is evaluated at the same token positions under the two input renderings.

## Acquisition gradients

| gradient | loss mean | L2 norm | focus targets | ordinary targets |
|---|---:|---:|---:|---:|
| acquisition_full_weighted | 2.64819 | 0.0821472 | 1291 | 30719 |
| acquisition_focus_mean | 3.65221 | 0.530635 | 1291 | 30719 |

## Preservation branches

| branch | tokens | KL(t||s) | teacher entropy | teacher top1 | teacher target CE | student target CE | grad L2 | cos vs full acquisition | cos vs focus acquisition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary_common | 1003 | 0.0288476 | 2.32474 | 0.631347 | 2.73 | 2.70832 | 0.111892 | 0.22411 | -0.482101 |
| dense_common | 1003 | 0.502657 | 3.95773 | 0.413733 | 4.04548 | 3.28843 | 1.32425 | 0.283407 | -0.505417 |
| ordinary_full | 4762 | 0.0245852 | 1.71541 | 0.708179 | 1.76288 | 1.79381 | 0.0907623 | 0.242256 | -0.500555 |
| dense_nonlabel_full | 6665 | 0.473825 | 3.92194 | 0.418142 | 4.20389 | 3.55537 | 1.22175 | 0.290248 | -0.51629 |

## Direct rendering contrast on common support

{
  "dense_minus_ordinary_teacher_entropy": 1.6329882356485839,
  "dense_minus_ordinary_teacher_top1_prob": -0.2176142528073738,
  "dense_minus_ordinary_teacher_target_ce": 1.3154774263634872,
  "dense_minus_ordinary_student_target_ce": 0.5801086387748375,
  "dense_minus_ordinary_kl_teacher_to_student": 0.47380962477605104,
  "dense_over_ordinary_grad_l2": 11.835130863814776,
  "cos_ordinary_dense_common_grad": 0.7287606757020876,
  "cos_ordinary_common_vs_acq_full": 0.2241102193890417,
  "cos_dense_common_vs_acq_full": 0.2834065326628716,
  "cos_ordinary_common_vs_acq_focus": -0.48210133441885655,
  "cos_dense_common_vs_acq_focus": -0.5054166518596868
}

## Interpretation

The common-support branch separates input rendering from target support: the same token positions are preserved under ordinary-WWM context and dense-corrupted context. Any difference in teacher entropy, KL, or gradient alignment here is a prediction-state geometry effect rather than a support-count effect for these positions. On the selected macros, dense-common KL has 11.8× the ordinary-common private-gradient L2. Its cosine with the full acquisition gradient is 0.2834065326628716, versus 0.2241102193890417 for ordinary-common. The full dense-nonlabel branch remains a joint rendering-and-support intervention because its target set differs from clean ordinary WWM. It should be interpreted through both the full-support measurements and the common-support diagnostic, not as a pure geometry isolation. Because gradients are evaluated at the acquisition-shifted exact `(M,S)` endpoint, nonzero KL reflects actual drift from the coherent86 parent. The same test at initialization would be uninformative: teacher and student predictions coincide and the deterministic KL gradient is zero.

This file is a local gradient/uncertainty diagnostic. It does not change the repaired BabyLM coordinate and should not be used as a benchmark score.
