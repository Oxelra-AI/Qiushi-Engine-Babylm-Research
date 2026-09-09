# body objective ablation component ablation after full-objective switch

This analysis starts from the saved query-first answer-only preparation checkpoints from Step019b and runs short continuations with selective trainable parameter sets. It asks whether early held-symbol binding loss is caused by the tied held input rows, the output head, the contextual body, or the switch of objective pressure itself.

## Branch means

| branch | n | be1 train4 | be1 held4 | be1 heldB | be1 heldSel | final train4 | final held4 | final heldB | final heldSel | final blocked train4 | final held L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| untied_body_only_full | 3 | 0.812 | 0.602 | 3.736 | 0.435 | 0.348 | 0.315 | 0.194 | 0.054 | 0.269 | 0.000 |
| untied_body_only_answer_only | 3 | 0.986 | 0.742 | 5.995 | 0.657 | 0.999 | 0.773 | 8.503 | 0.729 | 0.259 | 0.000 |
| untied_body_only_context_only | 3 | 0.795 | 0.592 | 3.678 | 0.427 | 0.263 | 0.253 | -0.024 | -0.002 | 0.264 | 0.000 |

## Per-seed one-epoch and final deltas

| seed | branch | start h4 | be1 h4 | Δ1 h4 | be1 hB | Δ1 hB | final h4 | Δfinal h4 | final hB | Δfinal hB | final train4 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | untied_body_only_full | 0.484 | 0.484 | 0.000 | 4.350 | 0.517 | 0.367 | -0.117 | 0.500 | -3.333 | 0.412 |
| 42 | untied_body_only_answer_only | 0.484 | 0.477 | -0.008 | 3.826 | -0.007 | 0.488 | 0.004 | 5.013 | 1.180 | 0.996 |
| 42 | untied_body_only_context_only | 0.484 | 0.484 | 0.000 | 4.357 | 0.524 | 0.258 | -0.227 | -0.006 | -3.839 | 0.301 |
| 43 | untied_body_only_full | 0.820 | 0.672 | -0.148 | 3.509 | -1.994 | 0.277 | -0.543 | -0.032 | -5.536 | 0.279 |
| 43 | untied_body_only_answer_only | 0.820 | 0.809 | -0.012 | 5.680 | 0.176 | 0.879 | 0.059 | 9.045 | 3.541 | 1.000 |
| 43 | untied_body_only_context_only | 0.820 | 0.652 | -0.168 | 3.419 | -2.084 | 0.250 | -0.570 | -0.021 | -5.524 | 0.248 |
| 100 | untied_body_only_full | 0.953 | 0.648 | -0.305 | 3.349 | -4.425 | 0.301 | -0.652 | 0.113 | -7.661 | 0.352 |
| 100 | untied_body_only_answer_only | 0.953 | 0.941 | -0.012 | 8.478 | 0.704 | 0.953 | 0.000 | 11.453 | 3.679 | 1.000 |
| 100 | untied_body_only_context_only | 0.953 | 0.641 | -0.312 | 3.259 | -4.515 | 0.250 | -0.703 | -0.046 | -7.820 | 0.240 |

## Interpretation scaffold

If `untied_out_only_full` collapses held behavior while the body and input embeddings are fixed, then output/readout adaptation is sufficient. If `untied_body_only_full` collapses held behavior while input and output tables are fixed, then contextual state specialization is sufficient. If `untied_input_only_full` collapses despite fixed body/output and held rows absent, then training-symbol input geometry is sufficient. `untied_all_answer_only` is the continuing-answer-pressure control; `untied_all_context_only` tests whether answer loss is needed to keep the selector alive. These are early-dynamics tests; long reacquisition under full objective is a separate phenomenon.
