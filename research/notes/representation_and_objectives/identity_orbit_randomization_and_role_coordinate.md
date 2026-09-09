# identity orbit randomization and role coordinate — Identity-orbit randomization converts entity memorization into reusable relation structure

## Scientific Motivation existed

name memorization finding found that a small BiGRU trained on the paired world pilot result source-attested paired-world
NLI task reaches ~0.94 training accuracy while transferring nothing: new-name/known-predicate
accuracy was 0.513 and shared-name/reversed-role accuracy 0.301. The tempting reading was
"role-coordinate anchoring is unreachable in raw text". That reading was wrong in an
important way: with fixed natural participant names, the task *admits* a dominant
identity-specific solution, so the learner never has to represent the relation at all.
The correct repair is to remove the identity solution while keeping the coreference
structure that binding requires.

## The manipulation

Three identity regimes, otherwise identical text, labels, model, and optimizer:

- `fixed_names`: real participant names (paired world pilot result source records).
- `family_alias`: one alias pair per source family, stable across all its rows,
  drawn from a shared 64-token alias pool.
- `per_item_alias`: fresh alias pair resampled deterministically per encoded item,
  consistent between that item's context sentence and hypothesis.

Crossed with sparse anchor evidence on 5 held-out probe predicates
(`overcame`, `proved too strong for`, `was beaten by`, `was unable to overcome`,
`was victorious over`), at k ∈ {0, 2, 8, 32} labeled source events per probe predicate,
in three arms:

- `true`: correctly aligned labels;
- `shuffled`: systematically anti-aligned labels (label flipped), same exposure;
- `exposure`: exposure-matched but uninformative labels.

Anchor predicates (`defeated`, `beat`, `won against`, `lost to`, `fell to`,
`was defeated by`, `triumphed over`, `emerged victorious over`, `prevailed against`,
`came out on top against`) are always fully labeled. Evaluation is on **new source
families** with no participant overlap in the fixed-name mode; five never-trained
predicates (`edged out`, `succumbed to`, `resulted in a victory for`, `claimed the win
against`, `went down to`) form an additional surface.

## Result 1: identity randomization changes the learning regime

Minimal-predicate sentences, held-family probe accuracy, 5 seeds each, all runs train-fit
(`data/identity_orbit_template_verify/`):

| identity regime | k=0 | true k=2 | shuffled k=2 | exposure k=2 | true k=8 | shuffled k=8 | exposure k=8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_names | 0.550 | 0.546 | 0.497 | 0.534 | 0.570 | 0.476 | 0.500 |
| per_item_alias | 0.788 | 0.835 | 0.571 | 0.687 | **0.905** | **0.302** | 0.647 |

With fixed names every arm sits within a few points of chance while training accuracy is
0.998–1.000. With per-item aliases the same architecture reaches 0.905 on new families and
new predicates. The training objective and data volume are matched; only the identity orbit changed.

## Result 2: the effect is not alias-vocabulary compression

On the richer full-sentence surface (dates, tournaments, rounds retained), the
family-stable alias control uses the *same* 684-token vocabulary as the per-item mode:

| regime (full surface) | vocab | k=0 | true k=8 | shuffled k=8 | exposure k=8 |
|---|---:|---:|---:|---:|---:|
| fixed_names | 1941 | 0.508 | 0.508 | 0.487 | 0.488 |
| family_alias | 684 | 0.559 | 0.552 | 0.423 | 0.492 |
| per_item_alias | 684 | 0.688 | **0.834** | **0.399** | 0.562 |

Family-stable aliases with an identical compact vocabulary stay near chance
(0.552 with correctly aligned anchors). Only per-item resampling produces transfer.
The active variable is therefore the *statistical independence of identity from role*
within each encoded item, not lexical simplification, vocabulary size, or anonymization.
Files: `data/full_surface_expanded_alias_probe/`,
`data/full_surface_family_alias_control/`.

## Result 3: sparse aligned anchors act as a signed role coordinate

Within the per-item regime the three anchor arms separate sharply and in the direction
predicted by the role coordinate anchor and state probe factorized identifiability analysis:

- correctly aligned k=8: 0.834–0.905
- exposure-matched k=8: 0.562–0.647 (worse than k=0, so extra rows alone do not help)
- anti-aligned k=8: 0.399 (minimal surface 0.302), i.e. **far below chance**

Below-chance behavior under anti-aligned anchors is the load-bearing signature. It means
the learner does not ignore the sparse evidence and does not treat each probe predicate
independently: it attaches the new predicate to a shared signed role direction and inherits
whatever orientation the anchors specify. This is the raw-text realization of the role coordinate anchor and state probe
result that a new relation cluster can be internally consistent yet globally inverted, and
that sparse correctly aligned links to established roles fix the orientation. Anti-aligned
anchors invert it; uninformative anchors leave it partly unresolved.

Per-template breakdown at per-item alias k=8 (held families) confirms it is not one template:
T04 0.906, T05 0.901, T09 0.933, T10 0.871, T15 0.933, versus anti-aligned
T04 0.416, T05 0.180, T09 0.334, T10 0.331, T15 0.193.
Never-trained predicates T16–T20 stay near 0.75–0.79 in every per-item arm, i.e. they are
supported by the anchor predicates and are largely insensitive to probe-anchor sign.

## Result 4: identity randomization alone is not sufficient

`data/original_surface_alias_probe/` uses each source event exactly once,
in its originally assigned template (1086 anchor rows instead of 4800). There, per-item
aliases still fit training (0.996–0.999) but transfer collapses to 0.500–0.538, while fixed
names do not even fit (0.558 train). So the mechanism needs both conditions:

1. identity must be uninformative within each item, and
2. the same relation must be encountered in several distinct realizations
   (multiple predicates/orderings per source event) so that a shared role direction
   is the only consistent solution.

This is a genuine two-factor statement about data structure, and the second factor is
still confounded with row count in this run; separating realization coverage from data
volume at matched row count is the immediate next measurement.

## Scientific reading

The candidate principle emerging here is about *what limited experience must look like*
for relational knowledge to become reusable, rather than about model capacity:

> When entity identity is predictive of a relation's outcome within the training
> distribution, a limited-data learner will spend its capacity on identity-specific
> associations and will not form a reusable relation representation, even when it fits
> the training set perfectly. Randomizing the identity orbit per encoded item, while
> preserving within-item coreference, removes that solution and forces a role-structured
> representation. Once that representation exists, a small amount of correctly aligned
> evidence for a new relation is enough to place it on the shared role coordinate; the same
> amount of anti-aligned evidence installs the opposite orientation, and exposure-matched
> uninformative evidence is worse than none.

Concretely: 8 labeled events per new predicate (160 rows against 4800 anchor rows, ~3%)
moved held-family accuracy from 0.688 to 0.834 on the full surface and from 0.788 to 0.905
on the minimal surface. That is the disproportionate-transfer signature the route required.

## What is not established

- This is a small BiGRU on a bounded sports-derived relation family. It is not yet an
  architecture-general or BabyLM-scale statement.
- Realization coverage and row count are confounded (Result 4).
- The relation family is a single asymmetric outcome relation. Held predicates
  T16–T20 are paraphrases of the same relation, not new relations.
- No connection yet to the natural EWoK/Entity conditional-reversal panel
  (`data/final_bridge_synthesis/`, 7,618 rows / 1,756 relational).
- Nothing here yet tests entity-state update or temporal overwrite; that remains the
  equivariance control and state substrate repair/255 ATP `state_at_time` substrate.

## Immediate next measurements

1. Matched-row-count test of realization coverage: hold total anchor rows fixed and vary
   how many distinct predicates/orderings each source event appears in.
2. Extend beyond one relation family: add an independent asymmetric relation
   (e.g. contemporaneous ranking state from the ATP substrate) and test whether the same
   sparse-anchor mechanism places it on the shared coordinate, and whether untouched facts
   are conserved under interference.
3. Then, and only then, a bounded masked-LM test at small scale: does per-item identity
   randomization applied to a fraction of a limited pretraining corpus change relational
   generalization measured on the prepared EWoK/Entity panel?

## Evidence files

- `scripts/identity_orbit_anchor_test.py` → `data/identity_orbit_anchor_test/`
- `scripts/identity_orbit_template_verify.py` → `data/identity_orbit_template_verify/`
- `scripts/full_surface_expanded_alias_probe.py` → `data/full_surface_expanded_alias_probe/`
- `scripts/full_surface_family_alias_control.py` → `data/full_surface_family_alias_control/`
- `scripts/original_surface_alias_probe.py` → `data/original_surface_alias_probe/`
- name memorization finding background task partial record: `tasks/s256_t17_tool1/` (timed out at 1500 s; fixed-name arms only, all probe surfaces at or below chance, consistent with Result 1)
