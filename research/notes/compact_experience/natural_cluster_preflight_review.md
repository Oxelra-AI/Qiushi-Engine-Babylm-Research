# natural cluster preflight review review of natural entity-relation cluster preflight

`scripts/find_natural_entity_relation_clusters.py` was written and smoke-tested, then run on the full official Strict-Small training text into staging, not as a training materializer:

- smoke: `staging/processing/natural_cluster_smoke/cluster_preflight_summary.json`
- full: `staging/processing/natural_cluster_full/cluster_preflight_summary.json`
- full sample: `staging/processing/natural_cluster_full/clusters_sample.jsonl`

The full metadata-only extraction read only official training text and reports 158,523 sentences with anchors, 21,772 candidate clusters, and enough candidate words for a 2% or 4% low-dose intervention (2% prefix: 2,355 clusters / 199,971 words; 4% prefix: 5,140 clusters / 399,971 words). This makes the natural evidence-cluster route feasible as an object to inspect, but not yet safe or strong enough for training.

Scientific cautions from samples:

1. A naïve content-diversity ranking can pull long Gutenberg/Simple-Wiki passages and repeated named-entity narratives rather than balanced child-scale evidence.
2. Some candidate clusters may concentrate sensitive, violent, or identity-related material even if the official corpus was previously decontaminated. A selection process that densifies such material can change its effective exposure, so future materialization should explicitly filter or cap sensitive anchors/topics and record the effect.
3. Anchors of type `rep:*` are noisier than capitalized names, quoted titles, and numbers; many repeated common words (for example `right`, `women`, `urban`) can indicate discourse filler or topic concentration rather than clean entity-relation evidence.
4. The current extractor identifies local shared-anchor complementarity but does not yet build the required controls: anchor-only shuffle, original-repeat, untouched-order, and optional no-shared-anchor local-packing control.
5. The held-out same-block third sentence is available for 3,435 clusters and should become the first internal diagnostic: if a true cluster does not improve masked prediction or loss margin on its held-out sentence versus controls in a short run, it should not receive a 100M wave.

Route judgment: natural entity-relation clusters are a plausible next mechanism because they attack the weak EWoK/COMPS/GlobalPIQA region through real corpus-internal complementary evidence rather than more Qwen-pair exposure. The proposed construction turns the extractor into a quality-filtered metadata artifact and internal diagnostic/control materializer. It should not yet train 100M.
