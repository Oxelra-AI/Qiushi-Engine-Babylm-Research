# full ewok interaction specificity — natural two-context/two-target feasibility (negative)

CPU-only, no training, no official evaluation. This probe tested whether the narrowed
EWoK interaction object (conditional target reversal) can be trained from an
evaluation-independent, corpus-derived signal using *natural* adjacent-sentence
transitions from the actual compact-view-reinvest 10M corpus.

## Construction

- Mine natural (context -> target) sentence transitions from real corpus rows.
- Pair two transitions (C1->T1, C2->T2) whose targets share content terms but come
  from different rows, so each target appears once as its own observed continuation
  and once as a cross negative.
- Score the four-cell matrix S(C_i, T_j) with the official EWoK MLM completion
  convention, plus empty-context target priors.

## Result: the naive natural-pair form is both redundant and confounded

Pilot (40 scored pairs, both legal40 checkpoints, CPU):

- Models already choose the observed continuation in most pairs: depth within-context
  positive fraction 0.60 (C1) / 0.75 (C2); mean-normalized positive 0.675 / 0.85.
- The mean-normalized within-context margins are solidly positive (median ~1.5-1.6
  nats), so length normalization does not manufacture a failure population here.
- The "interaction sum" is dominated by target length and content differences
  (median ~40 nats), not by a comparable relational contrast. Two natural targets are
  not minimally contrasting, so their target priors do NOT cancel. This is the exact
  confound independent_review warned about in ewok contrast preservation anatomy.
- The candidate pairs are heavily biased toward the same `cleanqwen_fineweb_compact_view_reinvest`
  source (near-paraphrase FineWeb rewrites), so the miner mostly recovers rewrite
  paraphrases rather than genuinely distinct natural contexts.

Example (pair 0): C1 "In the white phase they are completely white except for black
wing tips." / T1 "They are white except for black wing tips." has c1 target margin
+69.8; the cross target T2 (a different bird-plumage sentence) is strongly dispreferred.
This is fluency/topic preference, not conditional relational reversal.

## Scientific conclusion for the route

The EWoK interaction object requires **minimally contrasting** two-context/two-target
structure: T1 and T2 near-identical in surface form except a swapped relational element,
and C1/C2 that flip which target is correct. Natural corpus sentence transitions do not
supply that structure — their targets differ in wording and topic, so target priors and
lexical plausibility do not cancel, and the models already win the easy version.

Therefore an evaluation-independent generator for this object cannot rely on natural
sentence transitions. It would have to *synthesize* minimally contrasting target pairs
and matched context pairs, which risks (a) reinventing EWoK's own construction (evaluation
contamination) and (b) the INITIAL_MODEL_STUDIES RTD/contrast shortcut failure, where synthetic negatives
are locally implausible rather than context-dependent.

## Route implication

- The naive natural-pair objective is closed.
- Any surviving interaction-specific route must produce minimally contrasting
  target/context structure that (1) does not reuse EWoK wording or relation inventories,
  (2) makes target priors cancel by construction, and (3) has hard negatives whose
  wrongness depends on the paired context, not on local implausibility. Whether such a
  generator exists at all under Strict-Small compliance and without evaluation overlap is
  the open question for the pending full-EWoK measurement and subsequent generator design.

## Files

- JSON: `experiments/archive/representation_and_objectives/data/natural_two_context_target_feasibility/natural_two_context_target_feasibility.json`
- Candidates: `experiments/archive/representation_and_objectives/data/natural_two_context_target_feasibility/natural_two_context_target_candidates.csv`
- Scored pairs: `experiments/archive/representation_and_objectives/data/natural_two_context_target_feasibility/natural_two_context_target_scored_pairs.csv`
- Script: `experiments/archive/representation_and_objectives/scripts/natural_two_context_target_feasibility.py`
