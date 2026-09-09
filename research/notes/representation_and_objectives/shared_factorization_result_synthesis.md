# shared factorization result synthesis — Shared computational factorization as a learner-usable coordinate

## Result-producing artifacts

- Decomposition of earlier analysis pair-binding pilot: `data/pair_binding_decomposition/pair_binding_decomposition_summary.md` and JSON.
- Shared-coordinate raw-text probe script: `training/scripts/shared_relation_coordinate_probe.py`.
- First one-seed tied/untied pilot, train+eval vocabulary: `data/shared_coordinate_aligned_seed28800/`, `data/shared_coordinate_inverted_seed28800/`.
- Input-access audit: `data/factorized_probe_input_audit/`.
- Train-only vocabulary replication: `data/shared_coordinate_trainvocab_aligned_seed28801/`, `data/shared_coordinate_trainvocab_inverted_seed28801/`.
- Merged analysis: `data/shared_coordinate_all_merged_analysis/shared_coordinate_merged_analysis_summary.md` and JSON.

## What changed scientifically

earlier analysis showed that exposure-matched pair-binding alignment was not enough for DeBERTa with independent NLI-style binary heads. The shared factorization result synthesis decomposition sharpened the failure: direct h0/h2 state bridge rows can be learned, but h1/h3 graph-transfer state rows collapse; held-held comparison closure in the earlier analysis pilot is itself unstable. This means the previous failure was not simply a learned comparison graph failing to talk to state. It was a failure to create a single reusable event-output coordinate across comparison and state-update decisions.

The shared factorization result synthesis tied-vs-untied probe directly tests that missing computation on the cleaner static slot confounded factorial and balanced budget preparation/284 `replace_k16_spread` substrate. This substrate was chosen because it had already removed the known anti-copy, copy-initial, and static-slot logical shortcuts, and because the earlier analysis DeBERTa probe fit local k16 rows perfectly yet failed graph transfer.

## Input restriction

The probe keeps role assignment latent in raw text. The model sees event/premise/hypothesis strings; participant names are parsed from the string and replaced with `<cand>` and `<other>` for each candidate. It does not receive relation IDs, voice metadata, subject/object roles, arg_order, correct slots, assignment bits, or event-role labels as input features. Metadata are used only to assemble supervised labels and analyze outputs.

The tied model uses one candidate-event scorer for both changed-state update and relation comparison. Relation comparison computes:

\[
P(\mathrm{same\ final\ owner}) = \sum_i \mathrm{softmax}(s(e_1, i))\,\mathrm{softmax}(s(e_2, i)).
\]

The untied model has separate event scorers for comparison and state, so it has at least as much capacity but no forced shared coordinate.

The input audit showed that with `--vocab-scope train`, train vocab has 62 tokens vs 94 for train+eval; all participant names are replaced and zero raw name tokens remain after normalization. Eval-only object words become `<unk>`. Thus the train-only replication cannot exploit eval name/object embeddings.

## Central evidence

All tied and untied models fit train state and train comparison objectives at 1.000 by epoch 55 or later.

Merged central readout (`data/shared_coordinate_all_merged_analysis/shared_coordinate_merged_analysis_summary.md`):

- Seed 28800, train+eval vocab:
  - tied aligned: graph same-initial changed = 1.000, graph pair-both = 1.000, mixed orientation acc = 1.000, margin 13.692, unchanged = 1.000;
  - untied aligned: graph same-initial changed = 0.500, graph pair-both = 0.500, mixed orientation acc = 0.000, margin −7.716, unchanged = 1.000;
  - tied inverted: graph same-initial changed = 1.000, graph pair-both = 1.000, mixed orientation acc = 1.000 under the inverted arm target, unchanged = 1.000;
  - untied inverted: graph same-initial changed = 0.500, graph pair-both = 0.500, unchanged = 1.000.

- Seed 28801, train-only vocab:
  - tied aligned: graph same-initial changed = 1.000, graph pair-both = 1.000, mixed orientation acc = 1.000, margin 13.809, unchanged = 1.000;
  - untied aligned: graph same-initial changed = 0.250, graph pair-both = 0.250, mixed orientation acc = 0.500, margin 4.698, unchanged = 1.000;
  - tied inverted: graph same-initial changed = 1.000, graph pair-both = 1.000, mixed orientation acc = 1.000 under the inverted arm target, margin 13.600, unchanged = 1.000;
  - untied inverted: graph same-initial changed = 0.500, graph pair-both = 0.500, mixed orientation acc = 0.500, margin −4.698, unchanged = 1.000.

The tied-minus-untied graph-transfer gain is +0.5 to +0.75 in aligned and +0.5 in inverted, with no train-fit advantage and no unchanged-state advantage. This is exactly the selective pattern needed to separate shared factorization from extra capacity.

## Current interpretation

This is the first positive controlled result after the earlier analysis/287 failures that connects the emerging principle to a concrete mechanism: limited experience becomes reusable when evidence is forced through a shared low-dimensional computational coordinate that both comparison and state-update decisions must use. Logical identifiability and data counterexamples alone were insufficient for a generic DeBERTa independent-head learner; adding more capacity without tying did not solve h1/h3 state transport. The result supports a principle of **learner-usable coordinate factorization**: data-efficient transfer requires not just informative examples, but an architecture/objective that makes those examples update the same latent variable used by the target behavior.

This is not yet a final general law. It is a controlled, synthetic raw-text result. The model still uses a designed candidate-event scoring form and a surface candidate parser, so it is not a BabyLM-scale language model result and not natural semantic role induction in full language. It does, however, avoid handing roles/coordinates directly: the event scorer must infer candidate final-owner tendencies from raw event wording.

## Immediate controls

A heldheld-only train-only vocabulary anchor control has been launched (`s288_t40_tool1`, output `data/shared_coordinate_trainvocab_heldheld_anchor_control/`). It tests whether the tied model has a default surface coordinate even without mixed bridge state evidence. The expected informative outcome is: held-held comparison closure may be high, but state graph-transfer under an absolute arm target should not be anchored to both aligned/inverted bridge targets without bridge rows. If heldheld-only also gives perfect state transfer, then the tied model has a surface/default-state shortcut and the bridge-anchor interpretation weakens.

Next useful controls after that, if the heldheld-only result supports anchoring, are:

1. seed expansion for tied/untied train-only vocabulary on aligned/inverted arms (one more seed may be enough if the pattern remains deterministic);
2. ablate held-held comparison rows from the tied model to confirm bridge h0/h2 alone does not propagate to h1/h3;
3. ablate bridge state rows to confirm comparison rows alone do not set an absolute state coordinate;
4. apply the same tying idea to a less hand-built encoder or a small transformer module to test whether the principle transfers beyond the GRU scorer.
