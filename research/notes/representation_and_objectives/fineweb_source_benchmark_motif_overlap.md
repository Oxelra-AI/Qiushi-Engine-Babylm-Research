# fineweb source benchmark motif overlap FineWeb source/benchmark motif overlap

CPU-only source analysis for interpreting the repaired seqsafe96 run and preparing a possible later four-arm FineWeb family. It does not measure model competence.

## Pool summary

| pool | rows | words | EWoK row hit % | Entity row hit % | COMPS row hit % | GPIQA row hit % | causal row % | physical row % | object-property row % | eval 7-gram rows >=1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cached_fineweb_seqsafe96_block | 21916 | 1753280 | 94.50 | 14.23 | 99.63 | 96.99 | 39.12 | 9.71 | 23.14 | 0 |
| official_lengthmatched_control_block | 21916 | 1753280 | 95.39 | 20.24 | 99.09 | 96.62 | 22.50 | 11.38 | 16.19 | 0 |
| live_v3_relation_rich_doccap8 | 15583 | 359452 | 40.69 | 1.88 | 77.18 | 51.20 | 19.76 | 2.66 | 8.02 | 0 |
| live_v5_self_contained_doccap8 | 9098 | 194670 | 36.37 | 1.63 | 75.02 | 47.75 | 21.21 | 2.54 | 8.38 | 0 |

## Main comparisons

### cached_fineweb_vs_official_control

Task hit-density ratio a/b: `{"COMPS": 1.1076, "EWoK": 0.8512, "Entity": 0.7221, "GlobalPIQA": 0.8837}`

Relation hit-density ratio a/b: `{"causal": 1.9936, "physical_dynamic": 0.7937, "object_property": 1.6658, "spatial_state": 0.6161, "social_agent": 0.6987, "temporal_process": 1.4512}`

### live_v3_vs_cached_fineweb

Task hit-density ratio a/b: `{"COMPS": 0.9205, "EWoK": 0.8377, "Entity": 0.8509, "GlobalPIQA": 0.8509}`

Relation hit-density ratio a/b: `{"causal": 1.315, "physical_dynamic": 0.7984, "object_property": 0.8477, "spatial_state": 0.9663, "social_agent": 0.6529, "temporal_process": 1.6604}`

### live_v5_vs_cached_fineweb

Task hit-density ratio a/b: `{"COMPS": 0.9477, "EWoK": 0.8156, "Entity": 0.8634, "GlobalPIQA": 0.8563}`

Relation hit-density ratio a/b: `{"causal": 1.5169, "physical_dynamic": 0.8395, "object_property": 0.9472, "spatial_state": 0.988, "social_agent": 0.4861, "temporal_process": 1.7017}`

### live_v3_vs_live_v5

Task hit-density ratio a/b: `{"COMPS": 0.9713, "EWoK": 1.0272, "Entity": 0.9856, "GlobalPIQA": 0.9938}`

Relation hit-density ratio a/b: `{"causal": 0.8669, "physical_dynamic": 0.951, "object_property": 0.8949, "spatial_state": 0.978, "social_agent": 1.3431, "temporal_process": 0.9757}`

## Interpretation for the next route

Cached FineWeb and official length-matched control differ in benchmark-motif exposure: EWoK ratio 0.8512, Entity 0.7221, COMPS 1.1076, GlobalPIQA 0.8837. Relation marker ratios cached/control: causal 1.9936, physical_dynamic 0.7937, object_property 1.6658, spatial_state 0.6161. Use these numbers only to interpret source substrates.  A positive seqsafe96 interpretive foundation score would still be downstream evidence; a lexical advantage here only suggests why a later stratified v3/v5 mixture might be worth materializing.  Any rows with repeated exact benchmark 7-gram overlap must be excluded before future training materialization, even when the overlap appears formulaic.

Full JSON with samples: `experiments/archive/representation_and_objectives/data/fineweb_source_benchmark_overlap/fineweb_source_benchmark_motif_overlap.json`
