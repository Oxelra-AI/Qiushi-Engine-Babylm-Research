# cross realization probe resolution — Cross-realization probe result and next compact-data experiment

## What was measured

This analysis ran the saved-checkpoint cross-realization probe prepared in earlier analysis. It loaded existing DeBERTa-family checkpoints only and measured copied/shared content targets under three realizations: original source context, genuine compact rewrite context, and a matched counterfactual compact context. It did not train, run the official selected evaluator, upload a model, compute SuperGLUE/AoA, or submit to the leaderboard.

Mechanical evidence:

- Script parsed after a deterministic bootstrap-seed repair: `scripts/cross_realization_probe.py`.
- 16-event all-arm smoke completed with 16/16 events per arm: `data/cross_realization_probe_smoke/`.
- Full split-GPU run completed 4,500/4,500 events for every arm:
  - `data/cross_realization_probe_gpu0/` for `full_compact_100M`, `drop_abs_100M`, `drop_copied_word_100M`.
  - `data/cross_realization_probe_gpu1/` for `repeat_100M`, `adjbreak_100M`.
- Integrated paired result: `data/cross_realization_probe_integrated/cross_realization_probe_integrated.{json,md}` and `all_event_arm_contrasts.csv`.
- Construction robustness file: `data/cross_realization_probe_audit/cross_realization_probe_audit.{json,md}`.

The integrated events are fully paired across arms: 4,500 events, 2,009 source–compact pairs, 7,715 target BPE pieces, identical event sets, no cross-arm metadata mismatch. Event construction also has zero target/counterfactual doc overlap and zero target/counterfactual pair overlap; all counterfactual target-absence and BPE-identity flags are present. Compact versus counterfactual context length is close on average (mean +0.206 tokens, p95 +10), and target relative position is tightly matched (absolute compact–counterfactual rel-pos delta mean 0.000752, p95 0.0).

## Main numbers

All-event arm means for compact-context advantage, defined as counterfactual target NLL minus genuine compact-context target NLL per target BPE piece:

- `full_compact_100M`: 0.189814
- `drop_abs_100M`: 0.177078
- `drop_copied_word_100M`: 0.181702
- `repeat_100M`: 0.150132
- `adjbreak_100M`: 0.145007

The pre-stated full-unique same-sequence target-complementarity pattern is absent:

- `full_minus_drop_abs` compact-context advantage: +0.012736, pair-bootstrap 95% interval [-0.046339, +0.069389]; document-bootstrap interval [-0.049584, +0.068792].
- `full_minus_drop_copied_word` compact-context advantage: +0.008111, pair-bootstrap interval [-0.043802, +0.059521]; document-bootstrap interval [-0.046838, +0.060923].
- Representation interaction `cos(source, compact) - cos(source, counterfactual)` does not rescue full uniqueness: `full_minus_drop_abs` is -0.002139, pair interval [-0.005312, +0.000977], and `full_minus_drop_copied_word` is +0.003087, pair interval [-0.000154, +0.006295].

Relative to repeat, deletion arms retain small local signals, but the NLL effect is not uniform across event strata:

- `drop_abs_minus_repeat` compact-context advantage: +0.026947 all events, pair interval [-0.040906, +0.090991], document interval [-0.034510, +0.088882]. Its representation interaction is consistently positive: +0.006681, pair interval [+0.003282, +0.010263].
- `drop_copied_word_minus_repeat` compact-context advantage: +0.031571, pair interval [-0.027136, +0.089372]; representation interaction +0.001456 with interval crossing zero.
- `full_minus_repeat` compact-context advantage: +0.039682, pair interval [-0.021966, +0.100469]; representation interaction +0.004543 with positive interval [+0.001277, +0.007900].
- `adjbreak_minus_repeat` compact-context advantage: -0.005124, pair interval [-0.060265, +0.052783].

The cleanest NLL-positive stratum is `doc_disjoint_all_accepted`, where `drop_abs_minus_repeat` is +0.106346 with a positive interval, but `source_disjoint_quality` and `train_fixed_probe` are negative/uncertain. This means the local compact-context effect is real in some held-out document-like probes but cannot be treated as a universal downstream surrogate.

## Scientific resolution

The probe removes the new reason for a same-input target-cooccurrence training arm. If full compact training were dominated by a same-sequence complementarity between source-absent and copied/shared targets, full would need to separate clearly from both deletion arms on the cross-realization interaction. It does not. The matched endpoint triangle from bridge route recovered from sourcecopy error is still important: full compact beats `drop_abs` by +0.459 equal7 and `drop_copied_word` by +1.104 equal7 on the same no-AoA evaluator surface. cross realization probe resolution therefore does not say joint supervision is zero. It says joint target-complementarity is not the dominant explanation for compact-minus-repeat.

The surviving local object is more specific: compact input/context exposure combined with learning around shared/copied content. Drop-absent preserves compact inputs and copied/shared target supervision, and it retains much of the cross-realization local behavior; drop-copied also retains a similar aggregate NLL contrast but has weaker representation support and much worse endpoint performance. Input/context exposure and copied/shared-target supervision are still entangled. cross realization probe resolution supports studying this coupled object, not reducing the mechanism to input exposure alone.

Together with the RoBERTa selected-transfer result and the ordered/scrambled result, the stronger lesson is that local pseudolikelihood and representation alignment do not by themselves establish BabyLM selected competence. Future work must use official-compatible selected movement on stable families as the decisive downstream signal.

## independent_review reading

independent_review independently agreed with the core interpretation:

- Verifier integration:  — strong same-sequence target-complementarity is disfavored; contextual learning around shared/copied content is the better local explanation, but input-side causality and architecture-general downstream value remain unresolved.
- Generator integration:  — the highest-information next ordinary-WWM experiment is a source-wide extractive compact-view contrast.

## Next research object

The next experiment should not be target-cooccurrence decoupling. The best next compact-data experiment is a legal ordinary-WWM DeBERTa data contrast that separates source-wide coverage and density from generated rewrite-specific re-expression.

Preferred construction: a source-wide extractive compact view. For each source–compact pair, build an order-preserving view using only original source words/clauses, aiming to match the natural compact view on actual packed-stream properties:

- view word count and row/packed sequence-length distribution;
- source-position coverage, especially tail recovery and decile spread;
- content-word fraction and source-content coverage;
- source-document frequencies in the final 10M pool;
- active-token, candidate-token, and word-group mass under the legal spatial repair route status tokenizer;
- ordinary WWM probability, seed geometry, optimizer, update count, checkpoint schedule, and official-compatible selected readout.

Interpretation after selected scoring:

- Extractive approximately matches compact and both beat repeat: source-wide coverage/reduced prefix redundancy dominates; generation is not necessary for the principle.
- Compact exceeds extractive while extractive exceeds repeat: coverage contributes, but contextual re-expression/abstraction adds value.
- Compact exceeds extractive while extractive is near repeat: tail/source-wide access alone is insufficient; compact re-realization, density, or context recomposition is central.
- Extractive exceeds compact: original wording/discourse continuity may be more useful than generated compression.
- Any local-only signal without stable selected movement repeats the local/downstream split and should not be promoted.

If one extractive construction cannot simultaneously match compact-like breadth and compact-like density, split the construction into two source-only variants before any training: breadth-matched extractive and density-matched/front-loaded extractive. This would separate source-position spread from density per budgeted word without changing the objective.

No new 100M training should begin until an exact source-wide extractive pool construction preflight verifies row identity, word budget, packed-stream token/candidate/WWM-mass balances, and the selected-evaluation plan.
