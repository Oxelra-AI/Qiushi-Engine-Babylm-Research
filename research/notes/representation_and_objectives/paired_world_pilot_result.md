# paired world pilot result — Paired-World Role-Equivariant Construction Pilot

## Result

Built **500 role-equivariant four-cell families** from three independently sourced competitive event archives:
- 350 tennis (Sackmann ATP/WTA, 133,102 reversed pairs available)
- 100 badminton (BWF match data, 6,162 reversed pairs available)
- 50 football (La Liga, 149 reversed pairs available)
- **841 unique participants** across all families

Each family contains:
- Two independently sourced contexts (C1, C2) with the same two participants
- In C1 one participant won; in C2 the other won (source-attested)
- **Score-visible** and **score-ablated** verbalization variants
- 20 diverse templates (15 train, 5 held) balanced across winner-first/loser-first/mixed ordering
- Random A/B label assignment (fraction A-wins-C1 = 0.462 ≈ 0.50)
- Deterministic four-cell labels: `defeated(A,B)` × {C1, C2} × {TRUE, FALSE}
- Symmetric invariant checks (both-play-sport always TRUE)
- NLI-format queries ready for teacher verification
- Train/held split: 400 train, 100 held

## Shortcut Resistance

**Bag-of-words classifier cannot predict who won from score-ablated text.**
- Group-CV accuracy: **0.525** ± 0.039 (chance = 0.50) → **no shortcut detected**
- Within-family Jaccard: mean 0.361, range [0.161, 0.810]
- After score ablation, the only discriminating feature is syntactic role assignment

This means a model must read "defeated/lost to/was beaten by" and bind it to the correct participant — exactly the latent occurrence-role assignment problem from rawtoken bridge screen and route judgment.

## Label Consistency

- **0 label-consistency errors** across all 500 families
- Four cells verified: ab/ba complementary within each context, ab/ba flip across contexts
- Winner labels differ across C1/C2 in every family

## Connection to the Research Line

The paired-world substrate directly instantiates the missing operation from route2 factorial causal review/244/245:
- **Mention → entity**: participant names map to entities
- **Event → affected entity + state**: match outcome assigns winner/loser roles
- **Query → entity to read**: "Did A defeat B?" requires reading context-dependent role
- **Untouched facts**: symmetric invariants persist across both contexts
- **Role equivariance**: swapping A↔B in hypotheses must flip all four cells

The construction overcomes the earlier analysis-249 same-world paraphrase failure by using genuinely independent source records where the asymmetric relation is reversed.

## What This Does NOT Establish

- Teacher models can read diverse templates (needs teacher check)
- The substrate can train a model to learn role assignment (needs model probe)
- The operation transfers to non-competitive-outcome relations (needs broader families)
- The operation transfers to natural EWoK/Entity items (needs bridge panel test)

## Next babylm2026 live surface. **Teacher check** (cheap GPU): Qwen3.5-9B + Llama3.1-8B on ~200 families' score-ablated NLI queries; target ≥0.95 cross-teacher agreement
2. **Extend relation families**: add event-to-state and temporal families from same sources (e.g., "became champion", title sequences) + KG-sourced asymmetric relations
3. **TinyMLM/DeBERTa probe** with predeclared write/read permutation, actor-swap, query-swap, paraphrase, untouched-entity controls
4. **Natural EWoK bridge panel** item-level conditional-reversal test

## Files

- Script: `scripts/paired_world_pilot.py`
- Train families: `data/paired_world_pilot/families_train.jsonl` (400)
- Held families: `data/paired_world_pilot/families_held.jsonl` (100)
- Shortcut test: `data/paired_world_pilot/shortcut_test_result.json`
- Summary: `data/paired_world_pilot/pilot_summary.{json,md}`
- Teacher-check design: `data/paired_world_pilot/teacher_check_design.md`
