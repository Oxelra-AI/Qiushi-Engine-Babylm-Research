# natural cluster preflight review construction spec: natural entity-relation cluster route

## Purpose

Build a low-cost, tightly controlled mechanism test for whether locally packed, natural, complementary evidence from the official training corpus can recover the clean-Qwen weak region (EWoK/COMPS/GlobalPIQA) while preserving Supplement, Entity, Reading, SuperGLUE, and BLiMP. This is not a cap-length continuation and not a repeat of same-window Qwen pair exposure.

## Starting asset

`experiments/archive/compact_experience/scripts/find_natural_entity_relation_clusters.py` compiles and has already produced metadata-only cluster candidates in staging:

- `data/external/cluster_preflight_summary.json`
- `data/external/clusters_sample.jsonl`
- `data/external/cluster_preflight_summary.md`

Full extraction found 21,772 clusters and enough words for 2%/4% low-dose arms. The staging outputs should be copied into `data/natural_entity_relation_clusters/` once the current no-AoA SWA task releases the data write path.

## Necessary refinement before materialization

1. Quality-filter anchors:
   - prefer `cap:*`, `quote:*`, and number anchors with clear local relation evidence;
   - cap or downweight `rep:*` anchors unless they are proper/common nouns with low discourse-filler risk;
   - exclude anchors dominated by fillers such as `right`, `thing`, `people`, `time`, `said` unless an explicit relation pattern is present.

2. Source/topic balance:
   - cap per-source cluster word share, because the first full ranking is dominated by Gutenberg and Simple Wiki;
   - cap repeated articles/books/blocks so the low-dose intervention does not densify a single narrative or sensitive topic;
   - record source and block distributions for selected clusters and all controls.

3. Sensitive-content protection:
   - future materialization must not densify violent, sexually explicit, identity-targeted, or toxic clusters even if present in the official corpus;
   - use corpus-internal lexical filters and manual sample review, then record how many clusters were removed and why.

4. Held-out internal diagnostic:
   - prefer clusters with a held-out same-block third sentence;
   - build a fixed training-internal diagnostic file containing held-out sentences and matched same-anchor different-block negatives;
   - measure masked loss on anchors, relation/predicate neighborhoods, and content words. No downstream BabyLM eval labels or official AoA/CDI words are used.

## First experimental arms after metadata proof

All arms must share the same row-length plan, tokenizer, model init, batch/LR-time, WWM settings, word budget, and checkpoint policy. Start with 10-20M exposure, not 100M.

- E1 true cluster: same-block shared-anchor complementary sentences packed into a local 256-token row.
- E2 anchor-only shuffle: keep anchor string/source/length/frequency but replace one cluster sentence with a sentence from another block or document sharing the anchor.
- E3 original-repeat: repeat one anchor sentence or equal-length excerpt, matching anchor/mask exposure.
- E4 untouched-order inventory: same selected sentences remain in their original positions, testing whether local packing is the active factor.
- Optional E5 no-shared-anchor local pack: same length/source packing but no anchor sharing, testing generic local-packing effects.

## Promotion logic

Before any 100M wave, E1 must beat E2/E3/E4 on the internal held-out diagnostic and show favorable early no-AoA movement in W=(EWoK, COMPS, GlobalPIQA) while preserving R=(Supplement, Entity, Reading) and BLiMP. If E1 is indistinguishable from E2, the effect is anchor frequency or topic; if indistinguishable from E3, it is repetition; if indistinguishable from E4, same-window packing does not add value.
