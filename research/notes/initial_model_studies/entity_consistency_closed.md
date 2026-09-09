# entity consistency closed — Entity Mention Consistency closed as scaling route

## Verdict: controlled negative, do not scale to 100M

Two paired 20M seeds were run with identical geometry, corpus, tokenizer, WWM, and auxiliary hyperparameters, differing only in `extra_init_seed`/`train_rng_seed`. Both passed training-side validation (matched manifests, pair telemetry, distinct checkpoints, lower structured aux loss). But task-score replication failed decisively.

### Seed comparison at 20M (consistency minus shuffled_pair)

| Column | Seed1 | Seed2 |
|---|---:|---:|
| BLiMP | +2.04 | **−0.51** |
| Supplement | −0.98 | −0.17 |
| Entity | +0.39 | +0.21 |
| COMPS | −0.03 | +0.43 |
| GlobalPIQA mean | +3.92 | +1.97 |
| Reading mean | −0.21 | **−1.025** |
| Six-column sum | +5.135 | **+0.905** |
| Guard (Supp+BLiMP+Reading) | +0.85 | **−1.705** |

Seed2 fails all predeclared replication criteria: BLiMP direction reversed, Reading collapsed, guard columns deeply negative, six-column sum shrank by 82%. The seed1 signal was not a stable mechanism effect.

### Interpretation

Entity Mention Consistency (repeated same-form content-word InfoNCE) is better understood as representation smoothing / lexical consistency regularization than entity-state tracking. Its apparent gains were partly shuffled_pair harm and partly single-seed noise. It does not constitute a data-efficient learning principle that simultaneously advances Entity, EWoK, and GlobalPIQA while guarding grammar and Reading.

### Proposed follow-up tests

1. **Absolute WWM baseline at 10M/20M**: evaluate ordinary WWM DeBERTa-v2 at matched exposure to distinguish mechanism gain from exposure trajectory.
2. **Per-item mechanism analysis**: GlobalPIQA flip analysis on seed1 predictions to understand what changed.
3. **Genuinely different auxiliary mechanism**: move beyond repeated-form consistency to a mechanism targeting cross-sentence state/relation binding or world-knowledge structure, not just lexical smoothing.

Evidence: `data/entity_consistency_available_coordinate_comparison.json`, `data/seed2_entity_consistency_available_coordinate_comparison.json`, `data/seed2_entity_consistency_20m_training_validation.json`.
