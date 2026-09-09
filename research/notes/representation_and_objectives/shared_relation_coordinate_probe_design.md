# shared factorization result synthesis — Shared latent relation-coordinate probe design

## Scientific Motivation exists

earlier analysis showed that balanced same-initial counterexamples can be fit locally while the DeBERTa+independent binary head keeps an anti-copy-like rule off support: k16 train rows fit at 1.000, but held same-initial changed/pair-both stayed near 0.010 and mixed held-seen signed orientation was absent. earlier analysis removed the connectivity substrate construction and audit exposure flaw and tested exact-dyad pair-binding alignment with a one-seed learned pilot. The decomposition in `data/pair_binding_decomposition/` sharpened the result: direct-anchor h0/h2 state rows can be learned, but h1/h3 graph-transfer state rows collapse; held-held edge closure is also weak/unstable, so the comparison graph is not being cleanly learned and then merely failing to couple to state.

The strategist note therefore changes the next intervention: do not hand explicit entity or event roles to a model, because that would reproduce the earlier supplied-coordinate ceiling. Instead, keep role assignment latent in raw text and impose only a **shared low-dimensional relation coordinate** used by both relation-comparison and state-update decisions. Compare this to an equally expressive untied model on identical experience.

## Probe implemented

Script: `training/scripts/shared_relation_coordinate_probe.py`.

Default data: `data/information_budget_substrate/replace_k16_spread/` rather than earlier analysis neutral-block data. Reason: static slot confounded factorial and balanced budget preparation/284 k16 is the already-audited repaired case where anti-copy/copy-initial/static-slot equivalences were removed and DeBERTa fit local examples perfectly. earlier analysis neutral blocks are valuable for exposure-matched pair binding, but their rank-zero neutral labels can be fit by voice-pattern rules and would complicate a strict scalar relation-coordinate objective.

Model inputs are restricted to raw strings:

- event text spans from `event1`, `event2`, or `cause_event`; these are natural strings, not relation IDs;
- candidate/event participant strings parsed from surface capitalization and represented only as `<cand>` versus `<other>` inside each candidate-specific event query;
- premise/hypothesis strings for unchanged-state static readout.

The model does **not** receive `relation`, `voice`, `arg_order`, `correct_slot`, `inverted_bridge_label`, formal assignment bits, or event-role labels as input features. Metadata are used only for training targets and analysis grouping.

Two variants are contrasted on identical training rows:

1. `tied`: one candidate-event scorer is used for both changed-state updates and relation-comparison decisions. Relation comparison computes `P(same final owner) = sum_i softmax(score_event1)_i * softmax(score_event2)_i`, forcing comparison labels and state labels to share the same scalar event-output coordinate.
2. `untied`: state and comparison event scorers are separate, with at least equal capacity. It can fit both tasks locally but has no forced transfer of the comparison coordinate into state updating.

Unchanged-state queries use a separate static scorer because preserving unaffected facts is a different operation; the positive evidence must include unchanged preservation but should not force static ownership through the event coordinate.

## Runs started

The smoke test ran on CPU only:

```bash
python3 -B experiments/archive/representation_and_objectives/training/scripts/shared_relation_coordinate_probe.py \
  --models tied --arms aligned_state_bridge --seeds 28800 --epochs 3 --device cpu \
  --out experiments/archive/representation_and_objectives/data/shared_coordinate_smoke --print-every 1
```

It verified parsing/path/output plumbing. At 3 epochs, train comparison accuracy was only 0.542, so it is not scientific evidence.

Two one-seed GPU pilots were launched as the minimum learned contrast able to change the route:

- aligned arm, tied vs untied, seed 28800, 220 epochs, output `data/shared_coordinate_aligned_seed28800/`.
- inverted arm, tied vs untied, seed 28800, 220 epochs, output `data/shared_coordinate_inverted_seed28800/`.

No BabyLM training, official evaluation, or upload is involved.

## How to read the outcome

A positive factorization result requires the tied model, but not the untied equal-capacity model, to show all of:

- high train fit on both changed-state and held-held comparison objectives;
- high same-initial graph-transfer h1/h3 changed-state exact choice under the arm-installed coordinate;
- high graph-transfer pair-both changed+unchanged conservation;
- preserved unchanged facts;
- corresponding signed mixed held-seen orientation, especially as aligned/inverted arms differ in the expected direction under true versus inverted targets.

Direct h0/h2 success alone is insufficient: it only means bridge state rows were fit. If train comparison fit is poor, the result is an optimization/architecture non-result, not evidence against the factorization principle. If both tied and untied succeed, the effect is likely extra capacity/easier optimization rather than the sharing constraint. If tied succeeds and untied fails on graph-transfer while both fit train, this identifies shared computational factorization as the missing inductive bias in the previous independent-head probes, without supplying explicit roles or addresses.

## Input-access audit after first pilot

CPU audit `data/factorized_probe_input_audit/` shows that under `--vocab-scope train`, train vocabulary has 62 tokens versus 94 if train+eval are included; all eval participant names are replaced by `<cand>`/`<other>` and zero raw eval or train name tokens remain after normalization. Eval-only unknowns are object words (e.g. goblet, key, telescope, etc.), with unknown fractions 0.083 for changed-state/comparison event spans and 0.115 for unchanged-state premise+hypothesis strings. Thus a successful train-only replication cannot be attributed to learned embeddings for eval names or eval object words.

A fresh-seed train-only replication was launched:

- aligned, tied vs untied, seed 28801, output `data/shared_coordinate_trainvocab_aligned_seed28801/`.
- inverted, tied vs untied, seed 28801, output `data/shared_coordinate_trainvocab_inverted_seed28801/`.

This is still a one-seed replication, not final proof. It is the minimum check that removes the eval-vocabulary artifact and tests seed sensitivity before any broader expansion or paper-level principle statement.
