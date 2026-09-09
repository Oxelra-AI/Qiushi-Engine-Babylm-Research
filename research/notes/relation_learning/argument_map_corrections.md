# argument map corrections argument-map corrections

This note records two corrections to `argument_map_relation_typed_composition.md` prompted by delivered data and exact target-class summaries.

## 1. ICLM positioning must separate broad deduplication from in-window relation practice

The current experiments do not measure the cause of ICLM's broad perplexity degradation when deduplication is removed. ICLM removes near-duplicate documents at corpus level before composition; that intervention is confounded with corpus-level duplication and training instability. Our split controls instead separate same-content exposure from same-window adjacency. The result is sharper and narrower: at BabyLM dose, window adjacency produces a targeted source-conditioned computation and active cost on related-nonidentical targets, while the ordinary held-out same-window share is small (+0.021 nats for R−RS and +0.020 nats for V−VS).

Thus the safe positioning is: ICLM identifies that sequence composition helps and that near-duplicates must be filtered; this study supplies the matched-content instrument and T/U/N readout that isolate the in-window component, not a full explanation of ICLM's broad PPL harm.

## 2. Mechanism wording: transferring trigger, non-transferring precise readout

The source-specificity 2×2 shows REPEAT recognizes related true sources and moves mass onto source-content tokens. But the target-class readout shows this should not be called a precise identity-copy readout inside nonidentical rewrite contexts.

Across DeBERTa seeds from `component_targetclass_checks`:

- R−C overlap-target source-conditioned gain Δ mean -0.4203 (signs -/-/-); nonoverlap gain Δ mean -0.8965 (signs -/-/-).
- R−C true-source NLL Δ on overlap targets mean +0.0112 (signs -/-/+); on nonoverlap targets mean +0.5384 (signs +/+/+).
- V−C overlap gain Δ mean +0.5515; nonoverlap gain Δ mean +0.8029. VIEW improves source-conditioned gain on both classes and more strongly on nonoverlap targets.

Reading: exact local recurrence transfers a recognition trigger to related spans, but its precise answer readout is format-bound to exact natural repeats. In nonidentical rewrite contexts, even when the answer token is present in the source, REPEAT has lower source-conditioned gain than CLEAN in all three seeds. The mass shift is therefore best described as diffuse recognized-source content pull or source-content competition, not a generally precise copy of the correct source token.

Data: `experiments/archive/relation_learning/data/integrated_scope_and_mechanism/rewrite_targetclass_across_seed_summary.csv`
