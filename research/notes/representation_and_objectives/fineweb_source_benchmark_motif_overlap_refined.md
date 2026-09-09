# fineweb source benchmark motif overlap refined FineWeb source/benchmark motif overlap

This repairs the earlier motif pass by using field-aware benchmark text: EWoK uses concepts/contexts/targets rather than metadata; COMPS removes nonce/template tokens; Entity and GlobalPIQA use prompts/options. It is still source analysis only.

| pool | rows | words | EWoK row % | Entity row % | COMPS row % | GPIQA row % | causal row % | physical row % | spatial row % | social row % | exact eval 7gram rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cached_fineweb_seqsafe96_block | 21916 | 1753280 | 94.40 | 12.52 | 98.87 | 96.78 | 40.23 | 10.61 | 27.75 | 44.80 | 0 |
| official_lengthmatched_control_block | 21916 | 1753280 | 95.01 | 18.08 | 97.77 | 96.37 | 23.19 | 13.21 | 36.46 | 59.81 | 0 |
| live_v3_relation_rich_doccap8 | 15583 | 359452 | 40.90 | 1.84 | 68.00 | 50.66 | 20.62 | 2.90 | 9.93 | 12.85 | 0 |
| live_v5_self_contained_doccap8 | 9098 | 194670 | 36.65 | 1.58 | 65.94 | 47.27 | 22.04 | 2.73 | 9.44 | 8.03 | 0 |

## Comparisons

### cached_fineweb_vs_official_control

Task term-density ratio: `{"COMPS": 1.0664, "EWoK": 0.8639, "Entity": 0.7221, "GlobalPIQA": 0.8791}`

Relation term-density ratio: `{"causal": 1.9953, "physical_dynamic": 0.7402, "object_property": 1.5267, "spatial_state": 0.676, "social_agent": 0.6669, "temporal_process": 1.2515}`

### live_v3_vs_cached_fineweb

Task term-density ratio: `{"COMPS": 0.9134, "EWoK": 0.8448, "Entity": 0.8509, "GlobalPIQA": 0.85}`

Relation term-density ratio: `{"causal": 1.3297, "physical_dynamic": 0.7928, "object_property": 0.8667, "spatial_state": 0.9996, "social_agent": 0.6423, "temporal_process": 1.7903}`

### live_v5_vs_cached_fineweb

Task term-density ratio: `{"COMPS": 0.9401, "EWoK": 0.8238, "Entity": 0.8634, "GlobalPIQA": 0.8561}`

Relation term-density ratio: `{"causal": 1.5276, "physical_dynamic": 0.8177, "object_property": 0.9649, "spatial_state": 1.0293, "social_agent": 0.4127, "temporal_process": 1.868}`

### live_v3_vs_live_v5

Task term-density ratio: `{"COMPS": 0.9715, "EWoK": 1.0255, "Entity": 0.9856, "GlobalPIQA": 0.993}`

Relation term-density ratio: `{"causal": 0.8705, "physical_dynamic": 0.9695, "object_property": 0.8982, "spatial_state": 0.9712, "social_agent": 1.5562, "temporal_process": 0.9584}`

## Route implication

After removing metadata and nonce/template tokens, the cached seqsafe96 FineWeb block still does not look like a clean lexical match to every deficit column: term-density ratios cached/control are {"COMPS": 1.0664, "EWoK": 0.8639, "Entity": 0.7221, "GlobalPIQA": 0.8791}. It is much richer in causal/object/temporal markers but weaker in physical/spatial/social markers: {"causal": 1.9953, "physical_dynamic": 0.7402, "object_property": 1.5267, "spatial_state": 0.676, "social_agent": 0.6669, "temporal_process": 1.2515}. This makes the repaired seqsafe96 interpretive foundation downstream result especially informative: broad gains would mean source breadth carries more than these simple surface motifs; flat or harmful movement would not falsify cleaner stratified FineWeb, because the cached block is noisy and misbalanced. The live v3/v5 pools have lower benchmark-term density than the cached block but higher causal/temporal density; a future live source+view family should deliberately mix v3 relation-rich context with v5 self-contained facts and not optimize solely for isolated factual sentences.

JSON with samples: `experiments/archive/representation_and_objectives/data/fineweb_source_benchmark_overlap_refined/fineweb_source_benchmark_motif_overlap_refined.json`
