# Rewrite Generation and Paired-View Design

Scientific status: reconstruction of the documented construction recipe and its design rationale. It is not a new generation run or a guarantee of semantic equivalence.

The documented extraction selected 45,000 sentences of 12-50 words using seed27042. The scientific generation template was:

> Rewrite the following sentence to convey the same meaning using different words and sentence structure. Output only the rewritten sentence, nothing else.

Validation required 5-80 words in the rewrite, a rewrite/source length ratio between 1/3 and 3, and absence of meta-answer formatting. Accepted pairs were packed into approximately 160-word training rows, usually containing 3-4 pair segments; the intended fraction of rows containing aligned pairs was about 25%. Ten presentations formed the nominal 100M-word construction budget, with generated rewrites included in exposure accounting. The note did not establish an exact generator checkpoint or complete decoding configuration; these must not be guessed.

The illustrative comparison used a real source/rewrite pair to contrast three relationships: aligned nonidentical wording, a source paired with an unrelated rewrite, and exact duplication of the source. Shared entity strings and number recall do not establish faithful meaning: changes in modality, event attachment, or agency require semantic review. The raw source sentences are not reproduced here.

For dense-mask/sparse-label acquisition, the original view stays visible while detected content groups in the rewrite view are masked. Only a deterministic sparse subset receives loss. Other masked groups change the evidence available for prediction without adding acquisition labels. For preservation, the same complete row is presented with ordinary 15% whole-word masking to the frozen starting model and student.

This construction motivates correspondence-dependent prediction by suppressing local rewrite completion. It does not establish that every row forces a unique cross-view solution, or that a rewrite passing length and formatting checks is semantically safe.

Implementations and bounds: [dense-mask/sparse-label training](../../../experiments/archive/functional_learning/scripts/densemask_sparselabel_train.py), [semantic judging prompt and parser](../../../experiments/archive/functional_learning/scripts/semantic_judge.py), [sparse-label correction](sparse_label_interpretation_revision.md).
