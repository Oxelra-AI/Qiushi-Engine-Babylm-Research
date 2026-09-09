# transfer boundary and accounting transfer boundary and corrected bridge accounting

## Corrected bridge accounting result

The full static accounting analysis completed with `status=CORRECTED_BRIDGE_ACCOUNTING_READY`. It validates the experimental apparatus for the corrected interspersed bridge comparison:

- overlay: `experiments/archive/functional_learning/data/bridge_schedule_plan/overlay_tail_relation_e080_interspersed_wordpaced.jsonl`
- updates: 354 word-paced macro-updates
- rows seen: 108,727
- ordinary rows compared across arms: 79,927
- ordinary corruption pairing mismatch count: 0
- answer arm totals:
  - relation rows 28,800
  - ordinary rows 79,927
  - relation words 1,698,480
  - ordinary words 12,296,225
  - relation target tokens 60,160
  - ordinary target tokens 2,698,730
  - relation zero-label rows 0
  - answer span rows 28,800
- ordinary WWM arm totals:
  - relation rows 28,800
  - ordinary rows 79,927
  - relation target tokens 421,829
  - ordinary target tokens 2,698,730
  - relation zero-label rows 3
  - WWM relation rows 28,800
- mean lambda in both arms: 0.12136591529119578
- mean relation target-token fraction:
  - answer allocation: 0.02181848227285948
  - ordinary relation WWM: 0.13520327782980351

This supports interpreting the pending ordinary-WWM and answer-allocation bridge runs as the intended **policy-level** comparison: the same legal interspersed stream and paired ordinary WWM, with relation rows treated either by relation-row WWM or by answer-span allocation under an explicit macro-update relation-word coefficient. It does not isolate a single mechanism because the policy changes supervised positions, target count, masking pattern, per-target weight, and relation-row input corruption.

## Why the recombination export is useful but not decisive transfer evidence

Inspection of the recombination export identified a real boundary. Its rows keep source and update fixed while changing the query, but their candidate alternatives often differ strongly in entity type, semantic role, or grammatical compatibility. Thus the full-phrase contract can be solved partly by entity/candidate priors rather than contextual state assignment.

Concrete examples:

- `rw2s0_017880`: SoundCloud -> "one of the largest music streaming services" and Alexander Ljung -> "retired founder". Both entities occur in source, but the candidate phrases strongly encode which entity type they belong to.
- `rw2s0_011096`: Mr Rimbolt -> "condoned his offence" and the memory of Bolsover -> "forgotten by everyone". Both entities occur in source, but candidate grammatical and semantic roles differ.

I built `scripts/revision_048b_recombination_shortcut_audit.py` to quantify this instead of merely asserting it. For the two concrete examples scored on the coherent86 parent:

- no-context frame `The relevant state of {query_entity} is ___` already gives joint success for 1/2 pairs, specifically SoundCloud/Alexander Ljung (`no_context_min_margin=0.1449`), without source or update context.
- source-only also gives joint success for 1/2.
- update-only gives the updated-row direction in both examples but fails the unchanged row.
- full context fails joint success in both examples under this scorer, but the no-context result is enough to show the subset is shortcut-susceptible and should not be used as decisive selector-transfer evidence.

A full 200-pair shortcut audit for the parent and the answer-only specialist was in progress. Its purpose was descriptive: estimate how much of the recombination screen is solvable without contextual assignment and whether specialist movement exceeds simple query/candidate prior shifts. It would not by itself prove transfer of the relation-first selector because the recombination rows are not recipient-only reversals.

## Operation-preserving expression-transfer test

The direct operation-transfer test uses `scripts/expression_transfer_probe.py`. It keeps the relation-first state maps fixed:

- entity_a -> value_a
- entity_b -> value_b
- update to either entity assigns the same `shared_new_value`
- candidate set for a query is always `(value_a, value_b, shared_new_value)`
- changing only the update recipient should flip whether the queried entity's source value or shared_new is correct
- all candidate spans are masked simultaneously, matching the relation readout contract

This avoids the dense focus official and mechanism state shortcut more directly because value_a, value_b, and shared_new are candidates of the same typed relation, mostly place names; entity type alone should not determine the answer.

A six-pair CPU pilot (`data/expression_transfer_pilot6/`) compared the coherent86 parent and answer-only specialist seed40040 across original wording and five rephrasing variants. The pilot should not be treated as complete evidence, but it gives a useful mechanistic clue:

- parent: 0/6 original recipient-only flips; 0/30 non-original expression flips. It self-updates in 29/30 non-original records but cannot retain the queried source against the shared replacement; mean expression retain correct-vs-replacement = -5.393.
- specialist seed40040: 5/6 original flips; 10/30 non-original expression flips. It retains much stronger source discrimination under expression changes than the parent: mean expression retain correct-vs-wrong = +5.172 and correct-vs-replacement = +3.986. But full recipient-only flip success is brittle under query/update rephrasing.

The key failure pattern is not uniform forgetting. In many failed expression variants, the specialist still strongly selects the correct source over the wrong source but loses against the replacement, or retains source well while self-update recognition weakens after rewording. Thus the acquired operation has partially reusable source-assignment structure but is still wording-sensitive at the update/query interface.

The full 30-pair operation-preserving expression-transfer screen had started on CPU. It was intended as a more controlled expression-transfer readout for the answer-only specialist and future bridge checkpoints.

## Operation-preserving data export

I built `scripts/revision_048c_operation_preserving_export.py` and materialized `data/operation_preserving_export/`:

- `a01_operation_preserving_held_maps.jsonl`: 30 held relation-first state maps with exact entity/value/shared_new candidate contract.
- `a01_low_entity_overlap_examples.jsonl`: the 5 held maps with zero held-entity overlap with relation-first training.
- `summary.json`: contract and overlap counts.

Export summary:

- 30 held maps: 4 birthplace, 26 death_place
- entity overlap with relation training: 5 pairs with 0 overlap, 13 with 1, 12 with 2
- source-value overlap with relation training: 3 pairs with 0 overlap, 12 with 1, 15 with 2

The export preserves the operation during re-expression rather than defining a new shortcut-prone task: same relation type for both entities, same candidate set, recipient-only reversal, fixed source state map, and simultaneous full-span scoring.

## Current scientific interpretation

The bridge remains the central experiment for the active goal: can relation-sensitive experience be integrated into a legal continuation without destroying broad competence? The transfer work here does not replace it. It clarifies how to read any bridge success:

- Bridge relation success plus preserved BabyLM scores would show coexistence/acquisition inside ordinary limited-data learning on this substrate.
- Operation-preserving expression success would add evidence that the learned operation is not only an exact-template specialist.
- Recombination-screen descriptive success alone cannot establish selector transfer if no-context or entity-type priors already solve many pairs.

No BabyLM improvement, general data-efficient learning principle, or SOTA result is established by transfer boundary and accounting so far. The strengthened research state is methodological and empirical: the corrected bridge is now accountably valid; the shortcut boundary for cross-format testing is explicit; and a safer expression-transfer instrument exists and is partly piloted.
