# orthogonal route screen summary: Orthogonal Route Screen Plan

## Context

After fastpath closure and ordinary84 endpoint closed the frozen private fast-path as a learning principle, a genuinely different route is needed for stable context-conditioned competence under the 10M-word budget. Three orthogonal contenders were identified (earlier analysis):

1. **Route 1**: Change learning experience (procedural/action-outcome data)
2. **Route 2**: Change entity/event/time-binding computation (architecture)
3. **Route 3**: Change cross-context learning signal (corpus-derived target-shared pairs)

This historical plan summarizes the screening evidence and the proposed follow-up comparisons. Proposed experiments are not completed results.

## Route 1: Procedural/Action-Outcome Data — CLOSED

### Hypothesis
More ordered procedural text (action→state-change sequences) in the training stream teaches the model to track entity states through multi-step processes, improving EWoK and GlobalPIQA interaction.

### Cheap screen result
- **Extraction**: From 64,740 corpus rows, 1,142 passages matched temporal+action heuristics
- **Quality filter**: After removing dialogue transcripts, URLs, non-prose, and fragments, only **19 strict prose passages** remain (1,530 words)
- **Root cause**: The legal 10M pool is dominated by informational snippets and CHILDES dialogue, not clean multi-step procedural prose
- **Interference ladder evidence**: Models already saturate explicit_final=1.0 and consistent_prior=1.0 on the interference ladder (ewok domain link); the deficit is specifically about state inference *under contradiction*, not about general procedural tracking

### Decision
**Route 1 is closed.** The legal corpus contains insufficient clean procedural structure, and models already handle non-contradictory sequential state tracking. Adding more procedural data within the 10M budget would displace other useful text without addressing the measured interference-under-contradiction deficit.

---

## Route 2: Representation Probe (Proposed)

### Hypothesis
Context-conditioned binding information IS formed in intermediate hidden layers but cannot be extracted by the MLM head. An architectural modification (explicit state-update computation, auxiliary readout pathway, or entity-slot mechanism) could make this information usable at the output layer.

### What it tests
The probe answers the foundational question: **Is the bottleneck at representation formation or at readout?**

- If the probe succeeds at some layers where the MLM head fails → the information exists but is inaccessible → architecture change worth trying
- If the probe fails at all layers → the information is never formed → need data/signal change (Route 3), not architecture change
- If the probe succeeds at early/middle layers but fails at late layers → information is overwritten → need a protection mechanism during the forward pass

### Design

**Stimulus**: Use the existing 800-frame frozen interference ladder (`experiments/archive/representation_and_objectives/data/interference_ladder`, content SHA `4135fafc...`). Focus on three conditions:
- `contradict_bare` (80 frames) — all models fail at 0.0 crossed success (the hard case)
- `last_event` (80 frames) — varies 26× across trajectories (discriminating)  
- `explicit_final` (80 frames) — all mature models at 1.0 (positive control)

Additionally, use the 2,039 common Route 3 cross-context items where chck82 fails (crossed_success=False). These provide natural corpus-derived items where the model has the context but fails to use it.

**Checkpoints** (5 models spanning quality range):
1. `experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022/hf_model/chck_100M` (legal16k base)
2. `experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M` (legal40k 8×480)
3. `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_50M` (scale1.75 early)
4. `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M` (scale1.75 best)
5. `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M` (scale1.75 terminal)

**Procedure**:
1. For each checkpoint, forward-pass each stimulus through the model
2. At each of the 8 transformer layers, extract the hidden representation at the [MASK] position
3. For the interference ladder: collect representations for all frames in each condition
4. For each (condition, layer, checkpoint), train a logistic regression probe:
   - Features: hidden representation (dim 480 for legal16k, 480 for scale1.75)
   - Labels: correct target (0) vs. foil target (1) — binary classification
   - Use 5-fold cross-validation
   - Report mean accuracy
5. For the cross-context pairs: same procedure but classify correct-context vs. wrong-context assignment

**Decision criteria**:
- If any (contradict_bare, layer) probe exceeds 65% accuracy for ≥2 checkpoints → **Route 2 positive**: binding info exists but is inaccessible. Design an explicit state-update pathway.
- If all (contradict_bare, layer) probes are at chance (45-55%) for all checkpoints → **Route 2 negative**: the information is never formed. Commit to Route 3.
- If probes succeed at L1-L4 but fail at L5-L8 → **overwrite pattern**: the information forms early but is erased. Design a protected readout or residual connection.

**Cost**: ~30 min GPU (5 checkpoints × 3 conditions × forward passes), ~5 min CPU (probe training). No training, no data generation, fully reversible.

---

## Route 3: Cross-Context Signal (Proposed)

### Hypothesis
Standard MLM training provides insufficient signal for context-dependent property prediction because the model can satisfy most masked-word predictions using local co-occurrence patterns and frequency priors. Explicitly constructing and training on crossed property-context pairs where target priors cancel forces the model to use broader context for prediction, directly improving EWoK-style interaction accuracy.

### Cheap screen result (no-training, existing checkpoints only)

**Extraction**: From the legal 10M pool, extracted 2,396 cross-context pairs:
- 1,361 antonym-pair pairs (19 property antonym pairs, e.g. hot/cold, open/closed)
- 1,035 same-entity pairs (76 entities with ≥2 different physical states)
- 2,039 common items scorable across all three tested checkpoints

**Three-checkpoint panel on 2,039 common items:**

| Checkpoint | Crossed Success | Mean Δ | Δ>0 frac | Known cheap7 |
|---|---|---|---|---|
| chck82 (scale1.75) | 23.6% | 1.177 | 74.1% | 43.959 |
| legal16k_base100 | 24.9% | 1.243 | 70.6% | ~40.86 |
| legal40k_8x480_100 | 19.0% | 1.364 | 69.1% | ~41.14 |

**Interpretation**: The surface is deeply unsaturated (19–25% crossed success) despite 69–74% positive paired delta. This means models often correctly assign one context but fail the other — exactly the interaction deficit measured by EWoK.

**Family structure (chck82, 2,039 items):**

| Family | n | Crossed | Mean Δ | Character |
|---|---|---|---|---|
| deep/shallow | 11 | 0.0% | -0.311 | Hard: strong one-directional prior |
| strong/weak | 89 | 0.0% | 0.905 | Hard: positive delta but 0% crossed |
| smooth/rough | 27 | 0.0% | -0.549 | Hard: wrong-direction prior |
| clean/dirty | 10 | 0.0% | -0.215 | Hard |
| eyes | 190 | 4.7% | 0.466 | Near-zero: context barely helps |
| hot/cold | 78 | 6.4% | 0.714 | Near-zero |
| soft/hard | 94 | 55.3% | 1.915 | Good: context-sensitive |
| open/closed | 98 | 49.0% | 0.599 | Good: context-sensitive |
| new/old | 91 | 47.3% | 2.237 | Good: context-sensitive |
| house | 48 | 56.2% | 2.425 | Good: entity context helps |

**Scientific structure**: There are three distinct regimes:
1. **Zero-crossed families** (0%): The model has an extremely strong prior that overrides all context. Training must teach the model that context can override these priors.
2. **Intermediate families** (5-30%): The model uses context sometimes but not reliably. Training should strengthen this signal.
3. **High-crossed families** (>40%): The model already uses context well. These are training controls.

**Antonym vs same-entity split:**
- Antonym pairs: chck82 crossed 25.9%, legal16k 29.1%, legal40k 16.3%
- Same-entity: chck82 crossed 20.7%, legal16k 19.3%, legal40k 22.5%
- Different trajectories have different strengths, confirming the measure is route-sensitive

### Training data construction after the probe
If Route 3 is selected for training:
1. Construct training blocks from the antonym pairs: place sentence_a (with target_a) and sentence_b (with target_b) as adjacent items in the training stream
2. Standard MLM masking applies, but the proximity of contrasting examples creates a stronger learning signal for context-dependent prediction
3. Use the family structure as a curriculum: start with high-crossed families (soft/hard, open/closed) where the model already shows some context sensitivity, then progressively add harder families
4. Expand the pair set: the current extraction uses simple regex; a more sophisticated extractor could yield 5-10× more pairs from the same corpus

### Decision criteria (from the no-training screen)
- The screen surface is unsaturated (19-25% crossed), trajectory-discriminating, and family-structured → **Route 3 is viable**
- If a 4M-word cross-context replay moves crossed success on the hard families by >5 absolute points without macro cheap7 collapse → **Route 3 positive**: cross-context signal is a real learning principle
- If crossed success does not move, or cheap7 collapses → **Route 3 negative**: the deficit is not addressable by training signal alone

---

## Recommendation: Routes 2 and 3

### Priority order
1. **Route 2 first** (representation probe): 30 min GPU, fully decisive. Determines whether the bottleneck is representation or readout.
2. **Route 3 preparation**: Filter and verify the cross-context training set while Route 2 runs.
3. **Route 3 training (conditional)**: Only after Route 2 results are interpreted. If Route 2 shows info never formed → Route 3 training is the right next step. If Route 2 shows info exists but is inaccessible → architecture change is the right next step (but Route 3 training could still complement it).

### Implementation sequence
1. Run the Route 2 representation probe on the 5 specified checkpoints using the interference ladder and cross-context items
2. Record probe accuracy at each (condition, layer, checkpoint) and produce a clear diagnostic
3. Based on the probe result, proceed to Route 3 training pilot or architecture modification design
4. For Route 3 pilot: construct a 4M-word cross-context replay from the extracted pairs, train from chck_80M or chck_82M, and measure crossed success on the same items plus cheap7

---

## Files

| File | Contents |
|---|---|
| `data/orthogonal_route_extraction/route3_all_cross_context.jsonl` | 2,396 cross-context pairs (antonym + same-entity) |
| `data/orthogonal_route_extraction/route3_antonym_cross_context.jsonl` | 1,361 antonym-pair items |
| `data/orthogonal_route_extraction/route3_same_entity_cross_context.jsonl` | 1,035 same-entity items |
| `data/orthogonal_route_extraction/route1_procedural_passages.jsonl` | 500 raw procedural candidates (mostly dialogue) |
| `data/route_screen_analysis/route1_strict_procedural_passages.jsonl` | 19 strict prose procedural passages |
| `data/route_screen_analysis/route_screen_analysis.json` | Full analysis with common-item comparison |
| `data/cross_context_scores/chck82.json` | Per-item scores for chck82 |
| `data/cross_context_scores/legal16k_base100.json` | Per-item scores for legal16k base |
| `data/cross_context_scores/legal40k_8x480_100.json` | Per-item scores for legal40k 8×480 |
| `data/orthogonal_route_extraction/extraction_summary.json` | Extraction summary |
| `scripts/extract_route_candidates.py` | Extraction script |
| `scripts/score_cross_context_pairs.py` | MLM scoring script |
| `scripts/analyze_route_screens.py` | Analysis script |
