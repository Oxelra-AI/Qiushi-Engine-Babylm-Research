# compact directional internal result compact directional internal result

## What was checked

Both compact fork branches completed with exact shared prefixes within branch pairs. The internal readout uses the intended directional interaction

`I_loss = 0.5*(FR + RF) - 0.5*(FF + RR)`

where lower loss is better. The pair-objective probe directly evaluated every trained model on the 12,155 compact pairs in both causal orientations, separated into target tokens that were copied from the context segment and target tokens not present in that context.

## Main internal numbers

- Training epoch-summary interaction (loss, lower better): overall `0.000639`, copied `0.008741`, noncopied `0.020481`. This is not favorable.
- Direct pair-probe forward target I: side `-0.005953`, copied `-0.004898`, noncopied `-0.010725`.
- Direct pair-probe reverse target I: side `-0.010365`, copied `0.001657`, noncopied `-0.028863`.
- Noncopied-minus-copied I: forward `-0.005828`, reverse `-0.030520`. Negative means the mixed schedule helps noncopied targets more than copied targets.

## Scientific interpretation

The paired causal objective was learned: all trained second epochs lower the directly evaluated side-target losses in their trained direction by large amounts relative to the corresponding prefix, including noncopied targets. The mixed-direction signal is real enough to track but small. It appears most clearly in direct cross-orientation pair probing, especially reverse noncopied targets; it is not visible as a favorable aggregate in the raw epoch training summaries. Therefore the compact-only 20M run is an internal trajectory clue, not a compact-specific mechanism result. hypothesis comparison source-conditioned ordering result further weakens any compact-specific reading: ordered source conditioning is large for compact, prefix, and onegap views, and compact has the smallest frozen I_f because many rewrite targets are source-absent. Any positive endpoint transfer must therefore be tested against copy-matched extractive controls before being interpreted as compact semantic exchange.

## Decision rule for the pending endpoint readout

The first endpoint-scoring attempt failed before execution because the working directory did not match the command's path base; it produced no scores. Scoring was restarted with an explicit root working directory and remained pending, with output expected under `experiments/archive/representation_and_objectives/data/compact_directional_cheap7_eval_r2`.

- If cheap7/EWoK-domain transfer is absent, close this 20M compact reciprocal-causal instantiation.
- If the correctness transition analysis relational domains or cheap7 show a meaningful positive directional interaction, run copy-matched `semantic_extract` and `random_extract` v4 fork controls before interpreting compact specificity.
- If endpoint transfer is weak but the internal noncopied effect is the only favorable sign, consider only a bounded continuation if it is cheaper and more decisive than full controls for distinguishing delayed transfer from null.

JSON evidence: `experiments/archive/representation_and_objectives/data/compact_directional_internal/internal_synthesis.json`. Pair-probe evidence: `experiments/archive/representation_and_objectives/data/compact_pair_objective/pair_objective_probe.json`.
