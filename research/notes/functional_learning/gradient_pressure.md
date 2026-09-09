# answer credit alignment gradient pressure

## Gradient summary

| target/scope | ans_loss | ctx_loss | ans_norm | ctx_norm | cos(A,C) | direct cos mix,C | one17 cos mix,A | one17 cos mix,C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bound/body | 0.024 | 33.638 | 1.16 | 21.4 | -0.026 | +1.000 | +0.035 | +0.997 |
| bound/all | 0.024 | 33.638 | 1.17 | 21.5 | -0.026 | +1.000 | +0.035 | +0.997 |
| bound/tables | 0.024 | 33.638 | 0.0964 | 2.43 | -0.003 | +1.000 | +0.041 | +0.999 |
| bag_indep/body | 6.788 | 33.638 | 10.8 | 21.4 | -0.047 | +0.999 | +0.460 | +0.862 |
| bag_indep/all | 6.788 | 33.638 | 10.9 | 21.5 | -0.047 | +0.999 | +0.460 | +0.863 |
| bag_indep/tables | 6.788 | 33.638 | 1.04 | 2.43 | -0.029 | +1.000 | +0.407 | +0.897 |

## One-epoch finite update

| condition | Δheld h4 | Δheld B | Δtrain4 | ans CE | ctx CE |
|---|---:|---:|---:|---:|---:|
| bound/direct | -0.151 | -1.955 | -0.184 | 0.101 | 28.738 |
| bound/one17 | -0.082 | -1.175 | -0.068 | 0.058 | 28.786 |
| bag_indep/one17 | -0.276 | -3.584 | -0.392 | 5.599 | 29.688 |

## Interpretation

The component gradients are measured at the answer-only preparation checkpoint on the first continuation epoch rows. Direct full weighting is dominated by the context-gradient direction. The coefficient-matched w=1/17 mixture does **not** become answer-gradient aligned at the preparation point, because the bound answer loss is already tiny and the answer-gradient norm is much smaller than the context-gradient norm; instead it mainly reduces the context-gradient magnitude while preserving a relation-aligned answer term that can grow when the selector is perturbed. Finite one-epoch effects distinguish aligned from misaligned answer credit: bound w=1/17 causes less immediate damage than direct full, whereas bag-independent w=1/17 damages the trained/held selector despite the same answer-position and RWT-family emphasis. Thus the useful pressure is not generic answer exposure; it is relation-aligned answer credit with a sufficiently reduced competing context coefficient.
