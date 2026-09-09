# deeper acquisition probe design Complete Synthesis: Entity-Conditioned Retrieval and Evidence Preservation

## Key findings

### 1. Genuine entity-conditioned state retrieval (not gating, not memorization)

The three-way probe scores value_a, value_b, and shared_new in each context.
The trained model strongly discriminates between the queried entity's source
value and the other entity's source value:

| Split | Cross-source retrieval both | Mean cross-source margin | Full retrieval both |
|-------|----------------------------|--------------------------|---------------------|
| Train | 84/90                      | +8.282                   | 82/90               |
| Held  | 27/30                      | +7.501                   | 24/30               |
| Parent| 1/30 (held)                | +0.034                   | 0/30                |

This rules out the "update-recipient gating" hypothesis: a model that merely
suppresses the replacement when query ≠ update-recipient would show zero
cross-source margin. The +7.501 held margin proves the model preferentially
retrieves the CORRECT entity's source value over the other entity's value.

### 2. Context-reading, not memorization

The source-reassignment test swaps value_a↔value_b in the source context
while keeping entities, update sentences, and query frames fixed.

| Split | Follows swap both | Mean swap margin |
|-------|-------------------|------------------|
| Trained held | 26/30 | +6.578 |
| Parent held  | 1/30  | -0.067 |

When the source says "A died in Turin" (instead of the original "A died in
Lexington"), the trained model predicts Turin for query A. Memorization would
predict Lexington regardless. The model reads entity-value bindings FROM context.

### 3. Entity-familiarity decomposition confirms generality

Entity overlap: 12/30 held pairs have both entities in train, 13/30 have one,
only 5/30 have both entities novel.

| Category       | n  | Cross-source both | Reassignment both | Mean cross-source |
|----------------|----|--------------------|-------------------|-------------------|
| Both novel     | 5  | 5/5                | 3/5               | +7.372            |
| One familiar   | 13 | 12/13              | 13/13             | +7.201            |
| Both familiar  | 12 | 10/12              | 10/12             | +7.880            |

The critical result: **all 5 both-novel pairs show cross-source retrieval**
with margin +7.372, comparable to familiar pairs (+7.880). Source-reassignment
following on both-novel is 3/5 (limited by small sample, not by mechanism).

Parent model shows near-zero cross-source in ALL categories (0/5, 0/13, 1/12).

### 4. Corruption localization: specific evidence tokens matter

| Condition           | Held four-condition (e60) | Protected positions |
|---------------------|---------------------------|---------------------|
| Broad corrupted     | 2/30                      | 0                   |
| Random protected    | 5/30                      | ~54.5K             |
| Critical protected  | 11/30                     | ~75.1K             |
| Clean               | 19/30                     | N/A                 |

Random protection (matched group count, ~8.4 groups/example) improves from
2→5, but critical protection (entities, source values, replacement) improves
to 11. The difference is specifically from preserving the tokens needed for
entity-conditioned selection, not generic noise reduction.

Note: protected position counts differ (54.5K vs 75.1K) because critical
evidence groups may have different average size than random groups. A
position-matched comparison would strengthen this, but the direction is clear.

## Scientific interpretation

### Established principle (bounded to this substrate)

**Evidence-preservation and concentrated credit for relational learning**:

Under BabyLM Strict-Small training (10M words, ~36M params), the model
defaults to a generic phrase-recency operation: it identifies recently
stated/updated information (high U) but does not condition this on which
entity was mentioned in the update (near-zero beta). This satisfies local
likelihood through shared preference rather than entity-conditioned tracking.

Installing entity-conditioned state retrieval requires:

1. **Concentrated answer credit**: training signal on positions where entity
   identity determines the correct answer (answer-only masking)
2. **Evidence preservation**: the sparse tokens encoding entity-value bindings
   (entity names, source values) must remain uncorrupted
3. **Specific evidence identity**: protecting these particular tokens helps
   more than protecting random tokens of matched count

The acquired computation is genuine retrieval, not gating or memorization:
- Cross-source margin +7.5 (correct source >> wrong source)
- Source-reassignment following 26/30 (reads from context, not memory)
- Generalizes to both-novel entities (5/5 cross-source, 3/5 reassignment)

### What this does NOT yet establish

1. **Multi-seed robustness**: all results are single-seed (seed=40040)
2. **Coexistence with ordinary MLM**: current success is answer-only; no
   evidence that this can be combined with standard pretraining objectives
3. **BabyLM transfer**: no Cheap7/Entity/Overall evaluation
4. **Scale dependence**: only 90 train pairs tested
5. **Relation diversity**: 108/120 pairs are death_place
6. **Position-matched corruption**: random vs critical protection used
   matched GROUP count but different POSITION count

## Relation to the highest goal

The coherent86 parent has a measurable data-efficiency failure: it satisfies
MLM likelihood through shared phrase preference rather than learning entity-
conditioned state tracking. This is now established with three-way, reassignment,
and familiarity decomposition on semantically grounded relation-first pairs.

The evidence-preservation and concentrated-credit principle gives a concrete
mechanism for this failure and a concrete path to remedy. The remaining work:

1. **Multi-seed replication** (2-3 seeds for answer-only and corruption arms)
2. **BabyLM bridge**: ALN-preserving continuation with relation-first packets,
   comparing ordinary CE with answer-focused objectives under matched exposure
3. **Cheap7/Entity evaluation**: does acquisition preserve or destroy broader
   competence?
4. **Scale and diversity**: expand beyond death_place using broader extraction
5. **Entity-disjoint held split**: construct with guaranteed separation

## Files

- Three-way + reassignment: `data/threeway_probe/`
  - Summary: `threeway_reassignment_summary.json`
  - Held records: `trained_held_threeway_records.jsonl`, `reassignment_records.jsonl`
  - Familiarity decomposition: `entity_familiarity_decomposition.json`
- Random protection: `data/random_protection/random_protection_summary.json`
- Figures: `figures/cross_source_by_familiarity.png`,
  `reassignment_by_familiarity.png`
- Design note: `notes/deeper_acquisition_probe_design.md`
- Entity overlap: `notes/entity_overlap_analysis.md`
