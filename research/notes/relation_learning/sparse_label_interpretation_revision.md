# Sparse Labels Are Not Answer-Only Supervision

Scientific status: corrected intervention description; the exclusive-source interpretation is withdrawn.

The acquisition loss was not restricted to a task-specific answer position. It covered deterministically selected content groups in the second view. For seed62064 the recorded counts were 21,479 label groups and 28,590 focus targets, compared with 592,858 ordinary targets. The dense second-view corruption set was larger than this sparse label set. Nonlabel masked positions altered available evidence without directly contributing acquisition loss.

The design therefore separates input corruption from loss attribution. Dense masking suppresses nearby second-view completion cues; leaving the original view available makes source-to-rewrite correspondence more consequential for the sparse targets. An early interpretation said the targets could only be explained through the first view. That wording was subsequently narrowed: removing one family of cues does not rule out every residual prior or prediction path.

The state-use and binding controls show why this qualification matters. A training example may be organized around a desired operation while a simpler cue solves it. Removing long copied substrings or obvious temporal words does not prove that state tracking or correspondence has become the only available solution.

The preservation residual also remains a combined-policy effect. A matched no-teacher, extra-presentation control was not run. The two continuation seeds reproduce the policy comparison, not an independently isolated causal contribution from the teacher term.

Evidence: [dense-mask/sparse-label implementation](../../../experiments/archive/functional_learning/scripts/densemask_sparselabel_train.py), [clean preservation implementation](../../../experiments/archive/functional_learning/scripts/clean_preservation_train.py), [later mechanistic limits](../functional_learning/mechanistic_claim_boundaries.md).
