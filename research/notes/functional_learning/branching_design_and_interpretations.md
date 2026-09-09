# branching experiment Branching Experiment: Design and Pre-Registered Interpretations

## Scientific purpose

query first binding compact summary established that a 153K-parameter causal Transformer can acquire
near-perfect counterfactual entity-attribute binding when the query appears
before the context (query-first order, answer-only training), while the same
architecture fails to bind in original order under any tested objective
allocation. branching experiment tests what the acquired computation depends on and whether
it constitutes genuinely reusable knowledge.

## Design

**Phase 1: Acquisition.** Train query-first answer-only (BOUND targets) up to
600 epochs per seed. Save model + optimizer state at the first evaluation point
meeting the binding threshold: tie-safe top4 ≥ 0.95, B-swap frac ≥ 0.95,
Q-swap frac ≥ 0.95, query-novel selectivity ≥ 0.8.

**Phase 2: Branching (500 epochs from bound checkpoint).**
Five arms, all starting from the same bound checkpoint:

| Arm | Training order | Objective | Mask | Scientific question |
|---|---|---|---|---|
| continue_qfirst_ans_only | query-first | answer-only | standard | Retention baseline |
| switch_qfirst_full | query-first | full next-token | standard | Retention under diluted answer pressure |
| switch_qfirst_ctx_only | query-first | context-only (zero answer weight) | standard | Retention without any answer pressure |
| qfirst_block_qctx_ans | query-first | answer-only | blocked | Query-access dependency |
| transfer_orig_ans_only | original | answer-only | standard | Transfer to harder format |

The blocked mask prevents context positions (3-14) from attending to the query
entity (pos 1) and SEP (pos 2) in all layers. This closes both direct and
SEP-mediated indirect query access for context states. IS (pos 15) and the
answer input (pos 16) retain full attention.

**Phase 3: Matched-compute control.**
Original-order answer-only from scratch for (acq_epoch + 500) total epochs.
Same seeds, same latent data stream (same `make_epoch_rows` per epoch), just
rendered in original rather than query-first format.

**Evaluation.** All arms evaluated in both query-first and original formats
plus query-first with blocked mask, using the standard binding probes (correct
NLL, bag mass, tie-safe top4, query margin, B-swap, Q-swap, query-novel
selectivity, held-entity variants, corruption selectivity).

## Pre-registered outcome patterns and interpretations

### Retention

**R1: Binding persists under continued answer-only** (expected).
→ The computation is stable under the training that produced it.
Interpretation: retention baseline; any decay in other arms is relative to this.

**R2: Binding persists under full objective.**
→ The selector is not overwritten when context-token losses are added.
→ Full next-token training can maintain (but may not easily acquire) binding.
This would reconcile the query first binding compact summary finding that full training sometimes transitions
to binding but unreliably: maintenance is easier than acquisition.

**R3: Binding decays under full objective.**
→ Context-token losses actively compete with or overwrite the selector.
→ Data-efficient binding requires dedicated selector pressure, not merely
  finite experience that happens to contain the right information.

**R4: Binding decays under context-only.**
→ The selector needs active answer-pressure reinforcement, not just passive
  context processing.
This is the strongest test of whether binding is a stable state or a reinforced
computation.

### Query-access dependency

**Q1: Blocking query-context attention immediately (bep=0 eval) destroys
binding metrics under the blocked mask.**
→ The bound computation uses prospective slot marking: context states form
  query-specific representations that the IS/answer position reads.
If the standard-mask evaluation of the same arm still works, the model has
dual paths: it uses prospective marking when available but may also have a
non-prospective fallback.

**Q2: Blocking does NOT immediately destroy binding at evaluation.**
→ The binding computation does not depend on query-conditioned context states.
  The IS position performs a content-addressable lookup at inference time using
  entity tokens that are representation-rich from training.
This would be surprising given the query first binding compact summary finding that original-order (where
context forms without query) fails.

**Q3: Training with blocked mask gradually erodes binding.**
→ The model cannot maintain the computation when query-conditioned context
  formation is prevented during learning, even though the IS position retains
  full access. The selector needs ongoing query-to-context interaction.

### Transfer test

**T1: Bound init learns original-order binding faster than control.**
→ The most valuable outcome. "Temporarily making a computation easier to
  acquire (through causal format) can increase the usefulness of later
  experience." This is a concrete data-efficient learning principle.
Measure: compare the original-order binding trajectory of the transfer arm
against the control at the same total epoch. If the transfer arm has higher
original-order top4 or B-swap, the bound init provided a useful starting point.

**T2: Bound init does NOT improve original-order binding over control.**
→ The computation is format-dependent. The bound parameters encode a
  query-conditioned context-marking strategy that is irrelevant when the
  query comes after context. The "reusable computation" exists only within
  the favorable causal format.
This is still informative: it says data efficiency of a computation depends
on whether the format that exposes it to learning matches the format in which
it will later be used. A curriculum that builds a computation in a favorable
format only helps if the computation transfers.

**T3: Bound init initially shows original-order binding (at bep=0) that then
decays during original-order training.**
→ The bound parameters encode representations that work across formats at
  evaluation, but original-order training gradients push the model away from
  this state. The interesting question is whether the decay is fast or slow.

## Connection to relation-typed composition

The relation-learning experiments have established that under a fixed experience budget:
- Exact recurrence installs source-recognition/content-pull (Δ(T−N) positive)
- Nonidentical restatement installs content-conditioned support (Δ(T−N) negative)
- Both effects require within-window co-occurrence (split controls collapse them)

The controlled binding experiments isolate a complementary axis: the conditions under
which finite training pressure creates a **reusable selector** versus a
**statistical bag approximation**. The connection is:

- The relation-learning REPEAT arm installs a source-recognition routine that helps with unchanged
  queries but hurts when the target form changes. the controlled binding study's bag-level convergence
  is analogous: the model learns what's in the context (bag membership) without
  learning which entity owns which attribute (binding).

- The relation-learning VIEW arm installs content-conditioned support. the controlled query-first binding task
  installs a query-conditioned selector. Both require that the learning setup
  supply the right relational structure to the computation.

- The key difference: the controlled binding task makes the distinction between bag-level and
  binding explicit and measurable, while the relation-learning study measures the consequence of the
  installed computation on downstream NLL differences.

If the branching experiment shows that a bound init transfers to original order
and makes later examples more effective, the combined principle would be:
"Data-efficient learning requires that the training format and objective route
sufficient credit to the target computation. When this is difficult, a
curriculum that first acquires the computation in a favorable format can
bootstrap later learning. The same content becomes more or less useful depending
on whether the co-occurrence structure practices the right relational
computation."

## What must NOT be concluded from this experiment

1. The orbit-binding task is not natural language. Any principle must note this
   scope and clearly state what would need to hold for the bridge to BabyLM.

2. A positive transfer result does not by itself establish a general
   temporal-credit principle. It establishes transfer for this specific
   architecture, task, and format pair. The generality claim requires
   mechanism (prospective marking → format-independent selector) and replication.

3. A negative transfer result does not invalidate the query-first binding
   finding. It constrains the kind of computation that was acquired (format-
   specific vs format-independent).

4. The blocked-mask arm blocks both direct and SEP-mediated indirect access.
   If it destroys binding, this does not distinguish which pathway was critical
   (entity-to-query direct vs entity-to-SEP-to-implicit-query indirect).
