# body objective ablation component ablation after full-objective switch

This analysis starts from the saved query-first answer-only preparation checkpoints from Step019b and runs short continuations with selective trainable parameter sets. It asks whether early held-symbol binding loss is caused by the tied held input rows, the output head, the contextual body, or the switch of objective pressure itself.

## Branch means

| branch | n | be1 train4 | be1 held4 | be1 heldB | be1 heldSel | final train4 | final held4 | final heldB | final heldSel | final blocked train4 | final held L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| untied_all_full | 1 | 1.000 | 0.917 | 7.456 | 0.880 | 1.000 | 0.917 | 7.456 | 0.880 | 0.198 | 0.000 |
| untied_body_only_full | 1 | 1.000 | 0.917 | 7.455 | 0.880 | 1.000 | 0.917 | 7.455 | 0.880 | 0.198 | 0.000 |
| untied_out_only_full | 1 | 1.000 | 0.958 | 7.646 | 0.901 | 1.000 | 0.958 | 7.646 | 0.901 | 0.208 | 0.000 |

## Per-seed one-epoch and final deltas

| seed | branch | start h4 | be1 h4 | Δ1 h4 | be1 hB | Δ1 hB | final h4 | Δfinal h4 | final hB | Δfinal hB | final train4 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | untied_all_full | 0.958 | 0.917 | -0.042 | 7.456 | -0.190 | 0.917 | -0.042 | 7.456 | -0.190 | 1.000 |
| 100 | untied_body_only_full | 0.958 | 0.917 | -0.042 | 7.455 | -0.191 | 0.917 | -0.042 | 7.455 | -0.191 | 1.000 |
| 100 | untied_out_only_full | 0.958 | 0.958 | 0.000 | 7.646 | -0.000 | 0.958 | 0.000 | 7.646 | -0.000 | 1.000 |

## Interpretation scaffold

If `untied_out_only_full` collapses held behavior while the body and input embeddings are fixed, then output/readout adaptation is sufficient. If `untied_body_only_full` collapses held behavior while input and output tables are fixed, then contextual state specialization is sufficient. If `untied_input_only_full` collapses despite fixed body/output and held rows absent, then training-symbol input geometry is sufficient. `untied_all_answer_only` is the continuing-answer-pressure control; `untied_all_context_only` tests whether answer loss is needed to keep the selector alive. These are early-dynamics tests; long reacquisition under full objective is a separate phenomenon.
