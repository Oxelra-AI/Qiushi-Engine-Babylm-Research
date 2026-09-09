# fineweb source benchmark motif overlap stratified live-FineWeb blueprint

This is CPU-only preparation for a possible next FineWeb study. It creates no training corpus, no generated text, and no model score.

Current v3/v5 union: 17558 rows / 402719 words / 3974 docs.

Selected current-scan seed: `experiments/archive/representation_and_objectives/data/stratified_live_fineweb_blueprint/live_fineweb_stratified_current_seed_300k.jsonl` with 12437 rows / 300005 words / 3577 docs.

## Current union by source class

```json
{
  "v3_relation_context_not_v5": 208049,
  "v5_self_contained_fact": 151403,
  "v5_self_contained_fact_outside_v3_doccap8": 43267
}
```

## Selected seed by focus

```json
{
  "physical_spatial_object": 67521,
  "causal_temporal_process": 157758,
  "social_entity_state": 23882,
  "other_expository_relation": 46041,
  "definition_taxonomic_fact": 4803
}
```

Baseline16k token probe for selected seed: 1.516001 tokens/word; seq256 visible fraction 1.0; rows over 256 tokens 0.

Route use: if FineWeb continues, use this as a seed logic and source inspection object, not as a substitute for the required four-arm materialization and real training/evaluation. The scaled family still needs exact 10M/100M accounting, matched natural/control arms, faithful-view acceptance measurement, and trainer-level token exposure measurement.

Summary JSON: `experiments/archive/representation_and_objectives/data/stratified_live_fineweb_blueprint/stratified_live_fineweb_blueprint.json`
