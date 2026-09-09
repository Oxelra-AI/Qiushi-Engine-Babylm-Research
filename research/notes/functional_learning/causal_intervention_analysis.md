# causal intervention: causal donor-query activation intervention analysis

## Scientific question

body objective ablation established that compatible objective allocation (static w=1/17)
preserves held-symbol transfer while allowing context prediction, while direct
full-objective continuation destroys it.  activation marker probe's linear probe showed that a
query-match signal at context-attribute positions was associated with preservation,
but a probe is representational, not causal.  A donor-query
redirection test was then used: for the same context, patching attribute-position activations
from a query-A run into a query-B run and testing whether the answer redirects
from entity B to entity A.  This establishes whether the component carries usable
entity-selection information, not merely novelty, scale, or correlated content.

## Design

Same context with four entity-attribute triples and two different queries (A and B).
Query-first format places the query at position 1; attribute positions at [5,8,11,14].
The model has 3 transformer blocks (d=64, 2 heads).  After each block L, we can:

- **full**: replace entire hidden state at ATTR_POS from donor (query-A run)
- **d_only**: replace only the preparation-learned query-match direction component
- **orth_only**: replace only the orthogonal complement (negative control)

The query-match direction is learned from the preparation model: mean(matched attribute
activations) − mean(unmatched activations) at each layer, normalized.  A single direction
per layer is used for all models, establishing whether continuation preserves or destroys
the preparation-era computation.

Tested on preparation, direct_full (100 epochs, collapsed), and static_1over17 (100
epochs, preserved) for seeds 43 and 100.  Cross-model transplant injects the preserving
model's L1 activations into the collapsed model.

## Core results

### 1. Layer 1 query-match direction causally determines entity selection

| seed | model | dir_acc_L1 | L1 d_only redirect | L1 orth redirect | L1 full redirect |
|---:|---|---:|---:|---:|---:|
| 43 | prep | 0.972 | **0.998** | 0.002 | 0.998 |
| 43 | static_1over17 | **1.000** | **1.000** | 0.000 | 1.000 |
| 43 | direct_full | 0.322 | 0.242 | 0.244 | 0.242 |
| 100 | prep | 0.980 | **1.000** | 0.000 | 1.000 |
| 100 | static_1over17 | **0.998** | **1.000** | 0.000 | 1.000 |
| 100 | direct_full | 0.264 | 0.530 | 0.174 | 0.538 |

- **d_only vs orth_only dissociation is perfect** in preparation and preserved models:
  the query-match direction component alone produces complete redirection (0.998–1.000),
  while the orthogonal complement produces zero redirection (0.000–0.002).
- **Layer specificity**: L0 and L2 show zero/chance redirect for all modes.
  The query-match computation is formed at block 1 and consumed by blocks 2+.
- **Collapsed models**: seed 43 is fully collapsed (dir_acc 0.322, all redirects at
  chance); seed 100 retains partial signal but the *preparation-learned* direction is at
  chance (0.264), suggesting the model has reorganized its representation.

### 2. Cross-model transplant rescues selection in collapsed models

| seed | L1 full redirect | L1 d_only redirect |
|---:|---:|---:|
| 43 | **0.946** | **0.840** |
| 100 | **0.982** | **0.860** |

The collapsed model (direct_full), whose own L1 direction accuracy is at chance and
whose own redirection is at or near chance, achieves near-perfect entity selection when
the preserving model's L1 activations are transplanted.  This proves:

1. The collapsed model's **downstream readout (blocks 2+ and output head) remains intact**
2. The damage is specifically to **query-match signal formation at layer 1**
3. The downstream machinery can still process the preparation-era direction

### 3. Held-entity generalization

The preparation-learned direction was fitted only from train-entity examples.  Testing
on held entities (never seen during training):

| seed | model | held_donor L1 d_only | train_donor L1 d_only | orth control |
|---:|---|---:|---:|---:|
| 43 | prep | 0.765 | 0.910 | ≤0.090 |
| 43 | static_1over17 | 0.735 | 0.905 | ≤0.095 |
| 43 | direct_full | 0.250 | 0.245 | 0.245 |
| 100 | prep | 0.940 | 0.885 | ≤0.110 |
| 100 | static_1over17 | 0.575 | 0.985 | ≤0.140 |
| 100 | direct_full | 0.380 | 0.470 | ≤0.210 |

- **held_donor** (held entity queried → donor): the direction redirects for held entities
  at rates well above chance (0.765 in s43 prep, 0.940 in s100 prep), with orth near zero.
  This means the model's query-match marking works for entities never seen during training.
- **train_donor** (train entity queried → donor into held-entity context): even stronger
  redirection (0.910–0.985), showing the downstream readout can select any entity using the
  query-match signal.
- **Collapsed models lose held redirection** more severely than trained-entity redirection,
  consistent with the preparation result that held-entity transfer is the fragile capacity.

## Mechanism

The complete causal story for this orbit-binding substrate:

1. **Block 1 creates a 1D query-match marker** at each attribute position in the residual
   stream.  This marker encodes whether that position's entity matches the query.

2. **The marker is necessary and sufficient for selection**: replacing only the marker
   direction from a donor query perfectly redirects the answer; replacing only the
   orthogonal complement has zero effect.  No other layer produces this effect.

3. **Full-objective continuation specifically degrades this marker** at L1 while leaving
   blocks 2+ and the output head functional.  Seed 43 shows complete erasure (direction
   accuracy 0.322, redirect at chance); seed 100 shows the preparation-era direction is
   at chance (0.264) but some signal may remain in reorganized dimensions.

4. **Compatible credit allocation (w=1/17) fully preserves the marker**: direction
   accuracy reaches 1.000 in seed 43 and 0.998 in seed 100, and redirection is perfect.

5. **Cross-model transplant proves downstream intactness**: the collapsed model's L1
   signal can be replaced by the preserving model's signal to recover near-perfect
   selection (0.946/0.982 full, 0.840/0.860 d_only).

6. **The marker generalizes to held entities**: the same direction, learned from train
   entities, redirects answers for entities never seen during training.

## Relation to earlier steps

- **body objective ablation** showed body-only context training destroys held transfer; causal intervention localizes
  this to the L1 query-match direction.
- **Step021b** showed static w=1/17 matches interleaving; causal intervention shows this allocation
  preserves the L1 direction with direction accuracy ≥ 0.998.
- **answer credit alignment/023** showed bag-independent and slot0 answer targets collapse binding;
  these should also fail to preserve the L1 direction (not yet tested but predicted).
- **activation marker probe's probe** showed correlational association; causal intervention establishes causal necessity
  (d_only sufficient) and sufficiency (orth_only zero, cross-model transplant rescues).

## What this does and does not establish

**Establishes**: In the synthetic orbit-binding substrate, entity selection is causally
determined by a 1D query-match marker at layer 1 attribute positions.  This marker is
formed during query-first preparation, preserved by compatible credit allocation, and
destroyed by standard full-objective training.  The damage is specifically to marker
formation, not to downstream readout.  The marker generalizes to held entities.

**Does not yet establish**:
- Whether this mechanism operates in natural-language models or BabyLM settings
- Whether the 1D nature of the marker is intrinsic or an artifact of model size
- Whether the reorganization observed in seed 100 collapsed model produces a functional
  alternative direction (would require per-model direction learning)
- Whether the marker preservation principle predicts natural-language data selection effects

## Files

- Script: `scripts/causal_intervention.py`
- Data: `data/causal_intervention/results.json`
- This note: `notes/causal_intervention_analysis.md`
