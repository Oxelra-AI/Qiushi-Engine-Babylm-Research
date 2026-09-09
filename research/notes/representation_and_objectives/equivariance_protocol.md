# equivariance protocol — Paired-World Role Equivariance Protocol

## Scientific proposition

**Claim**: When entities participate in independently sourced events that assign
asymmetric roles (winner/loser, champion/runner-up, higher/lower ranked),
training on paired worlds where roles are genuinely reversed — with an explicit
representation-matching (equivariance) loss across different surface realizations
of the same role assignment — enables cross-predicate transfer of the role-binding
operation, whereas standard sequence classification does not.

**Why this matters for data-efficient learning**: Under limited data, a learner
that extracts the abstract role-assignment operation from a few predicate families
and transfers it to unseen predicates has formed more generalizable, composable,
and reusable knowledge per training token than one that memorizes
predicate-specific argument conventions.

## Formal structure

### Paired worlds
For entity pair (A, B), two independently sourced contexts:
- C₁: event where R(A,B) holds (A wins, A ranked higher, A is champion)
- C₂: event where R(B,A) holds (B wins, B ranked higher, B is champion)

### Four-cell labels (deterministic)
- (C₁, "A bears role₊") → ENTAILED
- (C₁, "B bears role₊") → NOT_ENTAILED
- (C₂, "A bears role₊") → NOT_ENTAILED
- (C₂, "B bears role₊") → ENTAILED

### Predicate families
Each four-cell can be realized with multiple predicate types:
- Winner-first: "A defeated B", "A beat B", "A overcame B", "A prevailed over B"
- Loser-first: "A lost to B", "A was beaten by B", "A fell to B", "A was defeated by B"

**Training predicates**: {defeated, beat, lost_to, was_beaten}
**Held predicates**: {overcame, prevailed, fell_to, was_defeated}

Each set has 2 winner-first + 2 loser-first predicates.

### State predicates (independently attested)
From ranking database (separate from match results):
- "A was ranked higher than B" (verified from independent weekly ranking snapshot)
- "A had a better ranking than B"
These add event-to-state families beyond direct outcome.

## Training objectives

### Standard multi-predicate training
$$\mathcal{L}_{\text{std}} = -\sum_{(x_i, y_i) \in \mathcal{D}_{\text{train}}} \left[ y_i \log f_\theta(x_i) + (1-y_i) \log(1 - f_\theta(x_i)) \right]$$

where each $x_i$ = (context, hypothesis) uses training predicates only.

### Equivariance training
$$\mathcal{L}_{\text{equiv}} = \mathcal{L}_{\text{std}} + \lambda \cdot \mathcal{L}_{\text{match}}$$

$$\mathcal{L}_{\text{match}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \|r_\theta(x_i) - r_\theta(x_j)\|^2$$

where $\mathcal{P}$ is the set of equivariance pairs: $(i, j)$ such that $x_i$ and
$x_j$ describe the same world with the same hypothesis but different context predicates.
$r_\theta$ is the intermediate representation (e.g., GRU hidden state).

**Key insight**: The matching loss forces the model to learn that "A defeated B"
and "B lost to A" produce equivalent role representations when describing the
same world state. This is a stronger constraint than just getting both labels
right — it requires predicate-independent role encoding.

## Cross-predicate transfer evaluation

### Evaluation conditions
1. **Held family, train predicates**: family generalization baseline
2. **Train family, held predicates**: predicate transfer (key test)
3. **Held family, held predicates**: full transfer (hardest)
4. **Held context predicates, train hypothesis predicates**: context generalization

### Comparison protocol
| Model | Data | Loss | Transfer test |
|---|---|---|---|
| Single-pred | defeated only | CE | All held preds |
| Multi-pred standard | 4 train preds | CE | Held preds |
| Multi-pred equivariant | 4 train preds | CE + λ·match | Held preds |

### Success criteria
- **Positive signal**: equivariant > standard on held predicates by > 5pp
- **Moderate signal**: equivariant > standard by 2-5pp
- **Null**: no meaningful difference → bottleneck is predicate semantics, not objective
- **Negative**: equivariant < standard → equivariance loss hurts generalization

## Connections and boundaries

### Connection to rawtoken bridge screen and route judgment bottleneck
The latent occurrence-role assignment problem (mention→entity, event→affected entity,
query→entity) requires exactly this operation: binding the right role to the right
entity from diverse surface forms. Cross-predicate transfer tests whether the
abstract binding operation has been learned, rather than memorized per predicate.

### Connection to compact-view finding
The DeBERTa compact-view result (related experiments) showed that faithful data
transformation helps one architecture but doesn't transfer. The equivariance
approach addresses this by making the learning objective itself enforce
transferable role representations, rather than relying on data structure alone.

### Boundaries
1. **Predicate novelty**: A randomly initialized model has no prior knowledge of
   held predicate semantics. Transfer depends on structural similarity (same
   syntactic position, similar template). This is a known limitation.
2. **Domain scope**: Sports outcomes are one relation family. Genuine generality
   requires event-to-state, temporal, and non-sports domains.
3. **Model scale**: Results on a tiny GRU may not predict behavior at DeBERTa or
   larger scale. Positive results motivate scaling; null results don't prove
   impossibility.
4. **Paired-world requirement**: The equivariance loss requires genuine role
   reversals in the training data. Not all domains provide these naturally.

### Evidence chain to general principle
1. ✓ Paired worlds at scale with deterministic labels
2. ✓ Shortcut resistance (BoW, linear sequence, score ablation)
3. → Cross-predicate transfer comparison
4. → Independently attested event-to-state families (ranking pilot)
5. → Multi-domain generalization (sports + non-sports)
6. → Scale-up to DeBERTa/RoBERTa
7. → Movement on natural EWoK/Entity conditional-reversal items (rawtoken bridge screen and route judgment panel)

babylm2026 live surface are established. generation slice quality are in progress. packet local semantic view control are future work
that should only proceed if 3-4 show positive results.

## Artifacts
- Cross-predicate benchmark: `data/cross_predicate_equivariance/`
- Ranking-attested pilot: `data/ranking_attested_pilot/`
- paired world pilot result families: `data/paired_world_pilot/`
- paired world stress teacher and route stress tests: `data/paired_world_sequence_teacher/`
- rawtoken bridge screen and route judgment EWoK bridge panel: `data/ewok_natural_bridge_panel/`
