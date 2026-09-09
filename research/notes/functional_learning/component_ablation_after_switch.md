# body objective ablation component ablation after full-objective switch

This analysis starts from the saved query-first answer-only preparation checkpoints from Step019b and runs short continuations with selective trainable parameter sets. It asks whether early held-symbol binding loss is caused by the tied held input rows, the output head, the contextual body, or the switch of objective pressure itself.

## Branch means

| branch | n | be1 train4 | be1 held4 | be1 heldB | be1 heldSel | final train4 | final held4 | final heldB | final heldSel | final blocked train4 | final held L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tied_all_full | 3 | 0.811 | 0.602 | 3.751 | 0.437 | 0.346 | 0.315 | 0.206 | 0.058 | 0.267 | 0.208 |
| untied_all_full | 3 | 0.812 | 0.602 | 3.749 | 0.436 | 0.346 | 0.316 | 0.193 | 0.055 | 0.266 | 0.005 |
| untied_all_answer_only | 3 | 0.986 | 0.742 | 6.001 | 0.657 | 0.997 | 0.777 | 8.448 | 0.730 | 0.258 | 0.005 |
| untied_all_context_only | 3 | 0.794 | 0.592 | 3.688 | 0.428 | 0.276 | 0.255 | -0.024 | -0.002 | 0.262 | 0.005 |
| untied_body_only_full | 3 | 0.812 | 0.602 | 3.736 | 0.435 | 0.348 | 0.315 | 0.194 | 0.054 | 0.269 | 0.000 |
| untied_out_only_full | 3 | 0.995 | 0.754 | 5.702 | 0.668 | 0.997 | 0.749 | 5.670 | 0.668 | 0.246 | 0.000 |
| untied_input_only_full | 3 | 0.995 | 0.753 | 5.721 | 0.670 | 0.996 | 0.772 | 6.092 | 0.693 | 0.251 | 0.005 |

## Per-seed one-epoch and final deltas

| seed | branch | start h4 | be1 h4 | Δ1 h4 | be1 hB | Δ1 hB | final h4 | Δfinal h4 | final hB | Δfinal hB | final train4 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | tied_all_full | 0.484 | 0.484 | 0.000 | 4.372 | 0.539 | 0.375 | -0.109 | 0.530 | -3.303 | 0.410 |
| 42 | untied_all_full | 0.484 | 0.484 | 0.000 | 4.376 | 0.543 | 0.371 | -0.113 | 0.492 | -3.341 | 0.402 |
| 42 | untied_all_answer_only | 0.484 | 0.477 | -0.008 | 3.826 | -0.007 | 0.488 | 0.004 | 4.929 | 1.096 | 0.992 |
| 42 | untied_all_context_only | 0.484 | 0.484 | 0.000 | 4.380 | 0.547 | 0.285 | -0.199 | -0.006 | -3.838 | 0.328 |
| 42 | untied_body_only_full | 0.484 | 0.484 | 0.000 | 4.350 | 0.517 | 0.367 | -0.117 | 0.500 | -3.333 | 0.412 |
| 42 | untied_out_only_full | 0.484 | 0.484 | 0.000 | 3.834 | 0.001 | 0.484 | 0.000 | 3.839 | 0.006 | 0.992 |
| 42 | untied_input_only_full | 0.484 | 0.484 | 0.000 | 3.859 | 0.026 | 0.492 | 0.008 | 4.380 | 0.547 | 0.990 |
| 43 | tied_all_full | 0.820 | 0.672 | -0.148 | 3.528 | -1.975 | 0.273 | -0.547 | -0.031 | -5.535 | 0.279 |
| 43 | untied_all_full | 0.820 | 0.672 | -0.148 | 3.517 | -1.986 | 0.273 | -0.547 | -0.031 | -5.535 | 0.281 |
| 43 | untied_all_answer_only | 0.820 | 0.809 | -0.012 | 5.690 | 0.187 | 0.875 | 0.055 | 8.888 | 3.384 | 1.000 |
| 43 | untied_all_context_only | 0.820 | 0.652 | -0.168 | 3.424 | -2.079 | 0.246 | -0.574 | -0.020 | -5.524 | 0.256 |
| 43 | untied_body_only_full | 0.820 | 0.672 | -0.148 | 3.509 | -1.994 | 0.277 | -0.543 | -0.032 | -5.536 | 0.279 |
| 43 | untied_out_only_full | 0.820 | 0.824 | 0.004 | 5.503 | -0.001 | 0.812 | -0.008 | 5.480 | -0.024 | 1.000 |
| 43 | untied_input_only_full | 0.820 | 0.820 | 0.000 | 5.521 | 0.018 | 0.867 | 0.047 | 5.940 | 0.436 | 0.998 |
| 100 | tied_all_full | 0.953 | 0.648 | -0.305 | 3.353 | -4.420 | 0.297 | -0.656 | 0.118 | -7.656 | 0.348 |
| 100 | untied_all_full | 0.953 | 0.648 | -0.305 | 3.353 | -4.421 | 0.305 | -0.648 | 0.118 | -7.656 | 0.355 |
| 100 | untied_all_answer_only | 0.953 | 0.941 | -0.012 | 8.487 | 0.713 | 0.969 | 0.016 | 11.529 | 3.755 | 1.000 |
| 100 | untied_all_context_only | 0.953 | 0.641 | -0.312 | 3.260 | -4.514 | 0.234 | -0.719 | -0.045 | -7.819 | 0.244 |
| 100 | untied_body_only_full | 0.953 | 0.648 | -0.305 | 3.349 | -4.425 | 0.301 | -0.652 | 0.113 | -7.661 | 0.352 |
| 100 | untied_out_only_full | 0.953 | 0.953 | 0.000 | 7.770 | -0.004 | 0.949 | -0.004 | 7.691 | -0.083 | 1.000 |
| 100 | untied_input_only_full | 0.953 | 0.953 | 0.000 | 7.783 | 0.009 | 0.957 | 0.004 | 7.957 | 0.183 | 1.000 |

## Interpretation scaffold

If `untied_out_only_full` collapses held behavior while the body and input embeddings are fixed, then output/readout adaptation is sufficient. If `untied_body_only_full` collapses held behavior while input and output tables are fixed, then contextual state specialization is sufficient. If `untied_input_only_full` collapses despite fixed body/output and held rows absent, then training-symbol input geometry is sufficient. `untied_all_answer_only` is the continuing-answer-pressure control; `untied_all_context_only` tests whether answer loss is needed to keep the selector alive. These are early-dynamics tests; long reacquisition under full objective is a separate phenomenon.
