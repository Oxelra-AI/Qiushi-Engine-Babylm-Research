# anchor matched controls — existing small-route evidence synthesis

## Purpose

Before treating the anchor matched controls transition substrate as a new route, this note consolidates older small relation/binding route evidence from INITIAL_MODEL_STUDIES. The point is to preserve what has already been learned: relation-focused data can move EWoK, but earlier forms often failed their own mechanism probes or harmed GlobalPIQA and other columns.

## Main evidence

### INITIAL_MODEL_STUDIES earlier analysis legal BSM 1M
- Evidence: `research/notes/initial_model_studies/legal_bsm_1m_eval.md`
- Comparison: bsm_coherent_20pct minus bsm_swapped_20pct at ~1M.
- Positive movement: BLiMP +1.94, Entity +0.73, EWoK +0.55.
- Negative movement: Supplement -2.80, GlobalPIQA_parallel -3.89, GlobalPIQA_nonparallel -3.00, Reading -0.68.
- Mechanism reading: binding probes had 0.000 both-correct for all arms, so column movement did not show learned entity-conditioned binding.
- Consequence: do not scale exact BSM replacement without matched-update/reference and mechanism probe.

### INITIAL_MODEL_STUDIES earlier analysis paired BSM 4M trace
- Evidence: `research/notes/initial_model_studies/4m_trace_eval.md`
- Comparison: bsm_paired_coherent minus bsm_paired_swapped at 4M.
- Positive movement: EWoK +3.09, BLiMP +0.33.
- Negative movement: Entity -0.04, GlobalPIQA_parallel -1.95, GlobalPIQA_nonparallel -2.00, Reading -0.025; coherent minus matched reference GlobalPIQA mean -4.93.
- Mechanism reading: continuous binding probes remained at 0.000 both-correct across available checkpoints.
- Consequence: corpus-derived relation material can move EWoK, but may directly harm GlobalPIQA unless conditional-world structure is cleaner and matched controls are used.

### INITIAL_MODEL_STUDIES corrected tokenizer official vectors and route decision WikiAuto pair adjacency
- Evidence: `research/notes/initial_model_studies/pair_vs_shuffle_1m_profile.md`
- Comparison: source-rewrite adjacent minus shuffled at 1M.
- Positive movement: mean EWoK +1.27 and Entity +0.69.
- Negative movement: mean Supplement -1.40, BLiMP -0.10, COMPS -0.33, Reading nearly flat.
- Mechanism reading: same-source adjacency has some relation/entity signal but not broad endpoint strength.
- Consequence: supports same-proposition recurrence as a mechanism but not a standalone SOTA route.

### INITIAL_MODEL_STUDIES relation bias trainer preflight/67 active cross-view
- Evidence: `research/notes/initial_model_studies/route_after_crossview_negative.md`
- Comparison: active cross-view adjacent minus shuffled.
- Positive movement: Entity +0.085 only.
- Negative movement: mean EWoK -0.69 and broad columns not improved.
- Mechanism reading: source-side exact-overlap anchor prediction with visible simplification did not produce a reliable useful signal.
- Consequence: avoid route variants that only mask overlap anchors or rely on pair adjacency without richer relation supervision.

## Scientific conclusion

Previous small relation/binding/proposition-pair routes repeatedly produced EWoK or entity signals while failing the intended binding/interaction mechanism or harming GlobalPIQA/Supplement/Reading. The anchor matched controls transition substrate therefore must first be used only in a cheap matched treatment-vs-control probe, with GlobalPIQA readouts preserved, not directly scaled to full 100M training.

The clean next use of the anchor matched controls transition substrate, if needed after the running FW endpoints, is a small shared-coordinate treatment-vs-anchor-control probe that reads both official fast/compatible scores and the EWoK/GlobalPIQA relational readouts. It should not be a direct 100M commitment.

## Files

- summary JSON: `experiments/archive/representation_and_objectives/data/existing_smallroute_synthesis/existing_smallroute_synthesis.json`
- findings CSV: `experiments/archive/representation_and_objectives/data/existing_smallroute_synthesis/existing_smallroute_findings.csv`
