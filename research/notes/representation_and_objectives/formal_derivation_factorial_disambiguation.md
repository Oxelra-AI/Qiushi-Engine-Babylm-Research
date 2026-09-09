# analysis framework for factorial probe — Formal observational equivalence and factorial disambiguation

## Why this note exists

matched state and initial owner results showed that equivariant symmetry repair and macro context's state-query "transfer" follows a non-initial-owner heuristic rather than event-role inference. This note derives the formal equivalence, specifies the factorial disambiguation design, and states the transition prediction that connects role coordinate anchor and state probe identifiability to language learning.

## Formal observational equivalence on underdetermined data

### Setup
- Relations R with slot assignment s(R) ∈ {0, 1}
- Event "A R B" with participants (A, B) in surface order
- Event-associated object's final owner: participant(s(R))
- equivariant symmetry repair and macro context construction sets initial_owner = participant(1 − s(R))

### Two rules that produce identical behavior

**Event-role rule** (compositional, relation-dependent):
> After "A R B", the event-associated object belongs to participant(s(R)).

**Non-initial-owner rule** (simple, relation-independent):
> After any event, the event-associated object belongs to complement(initial_owner).

### Equivalence proof on equivariant symmetry repair and macro context data

equivariant symmetry repair and macro context fixes initial_owner = participant(1 − s(R)). Then:

    complement(initial_owner) = complement(participant(1 − s(R))) = participant(s(R))

This equals the event-role prediction for any relation R, any participant pair, and any voice realization. QED.

The formal Z2 assignment enumeration (equivariant symmetry repair and macro context) ranges only over relation-to-slot maps and does not include the non-initial-owner rule as a candidate. The formal algebra remains valid within the restricted assignment hypothesis class, but a learner has access to a larger hypothesis class that includes relation-independent rules.

### Key point
The initial-owner pattern is not a "bug" in equivariant symmetry repair and macro context. It is an instance of a deeper structural property: when finite experience leaves two rules observationally equivalent, the learner has no reason to prefer the compositional one. The simpler rule (anti-copy) is sufficient and relation-independent, so it is naturally preferred under any reasonable complexity bias.

## Factorial disambiguation

### Design

The **disambiguated** condition includes both initial-ownership patterns in training:
- **initial=opposite**: initial_owner = participant(1 − s(R)) → both rules agree
- **initial=same**: initial_owner = participant(s(R)) → rules disagree

For initial=same rows:
- Event-role predicts: participant(s(R)) — CORRECT
- Anti-copy predicts: complement(participant(s(R))) = participant(1 − s(R)) — WRONG

Half the state training rows are initial=opposite, half are initial=same. Comparison rows are identical across conditions. The total state row count is the same.

### Formal check
- **Underdetermined condition**: anti-copy accuracy on training state labels = 1.0
- **Disambiguated condition**: anti-copy accuracy on training state labels = 0.5 (fails on all initial=same)
- **Event-role accuracy**: 1.0 on both conditions (it is the correct rule regardless of initial pattern)

## Predicted transition

### Under underdetermined training (equivariant symmetry repair and macro context original)
1. Anti-copy fits all state labels → model learns anti-copy
2. No need to represent per-relation slot assignments → mixed held-seen orientation stays at chance
3. Eval on initial=opposite: high changed accuracy; eval on initial=same: low changed accuracy
4. Pair-both on initial=same: low (changed fails)

### Under disambiguated training (factorial)
1. Anti-copy fails on half the state labels → model must learn event-role
2. Event-role requires representing per-relation slot assignments → mixed orientation MAY separate (aligned > inverted)
3. Eval on BOTH patterns: high changed accuracy
4. Pair-both on BOTH patterns: high (compositional state understanding)

### The transition is the scientific result
The transition from no-orientation to orientation (if observed) would demonstrate:
- Models learn the simplest sufficient rule from finite data (anti-copy when underdetermined)
- When the simpler rule is insufficient, they discover the compositional one (event-role when disambiguated)
- Compositional structure enables relational transfer (orientation on unbridged comparisons)
- This connects role coordinate anchor and state probe's abstract identifiability law to language learning

### Connection to role coordinate anchor and state probe

role coordinate anchor and state probe showed: an internally consistent new predicate cluster can be globally permuted unless sparse mixed held-seen anchors attach it to an existing coordinate system.

The factorial substrate extends this: the anti-copy rule is an "internally consistent but disconnected cluster" — it gives consistent answers within the training set without anchoring to the global slot assignment. Factorial initial-ownership data acts as sparse anchors that force orientation identification by breaking the lower-complexity alternative.

## Conservation prediction

Under anti-copy (underdetermined):
- changed on initial=opposite eval: high (anti-copy works)
- changed on initial=same eval: low (anti-copy predicts wrong person)
- unchanged: high on both (copy static owner)
- pair-both on initial=same: low

Under event-role (disambiguated):
- changed on BOTH patterns: high
- unchanged: high
- pair-both on BOTH: high

**The strongest single readout**: pair-both on initial=same eval rows. This should transition from low (underdetermined) to high (disambiguated) if compositional event-role inference replaces anti-copy.

## Files
- Substrate generator: `scripts/factorial_initial_ownership_substrate.py`
- Substrate outputs: `data/factorial_initial_ownership/`
- equivariant symmetry repair and macro context base: `data/equivariant_symmetry_substrate/`
- matched state and initial owner results confound evidence: `notes/verified_initial_owner_shortcut_and_open_controls.md`
