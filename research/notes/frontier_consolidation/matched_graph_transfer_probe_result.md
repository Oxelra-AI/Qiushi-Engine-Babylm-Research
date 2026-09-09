# matched graph transfer probe result matched graph-transfer probe result

## Scientific purpose

The graph packet v0 construction graph packet was not an acceptable trigger for a 4M exposure screen because `graph_compact` preserved relation words in 50/50 rows while `ordinary_compact` dropped them. A frozen NLL win could therefore be lexical reuse rather than relational structure. matched graph transfer probe result tested whether a stricter matched diagnostic set could be built where:

- the queried answer word and its opposite are absent from every context view;
- structured, neutral, and reversed views have comparable length and generation style;
- the reversed view changes only the intended temporal/causal/state/polarity/consequence edge;
- transfer is measured across the query wording rather than by copying an exposed relation token.

No pretraining, HF upload, or leaderboard submission was performed.

## Construction attempts

### v1: direct constraints

Files:
- Prompts: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_generation_prompts.jsonl`
- Outputs: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_generation_outputs.jsonl`
- Validation: `research/documents/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_validation_summary.md`

Result: only **1/30** generated examples passed. The dominant failure was that Qwen used the answer words inside the context views:

- `view_structured-contains-after`: 9
- `view_reversed-contains-before`: 9
- `view_structured-contains-because`: 5
- `view_reversed-contains-despite`: 5

This directly confirms the failure mode: a naive graph-preserving generation prompt naturally leaks the exact relation word that the frozen model is later asked to predict.

### v2: held-out clue lexicalizations

Files:
- Prompt builder: `experiments/archive/frontier_consolidation/scripts/prepare_matched_probe_prompts_v2.py`
- Validator: `experiments/archive/frontier_consolidation/scripts/validate_matched_probe.py`
- Prompts: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_generation_prompts_v2.jsonl`
- Outputs: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_generation_outputs_v2.jsonl`
- Accepted: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_accepted_v2.jsonl`
- Validation summary: `research/documents/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_validation_summary_v2.md`

The repaired prompt explicitly asked for context clue words such as `later/earlier`, `since/although`, `approved/refused`, while holding out target pairs such as `after/before`, `because/despite`, `accepted/rejected`. Result: **10/30** passed hard lexical/length/overlap checks.

Accepted family mix:

- entity_state_update: 3
- event_temporal_causal: 3
- polarity_contrast_event: 3
- social_belief_report: 1
- quantity_change_compare: 0

Accepted pair mix:

- because/despite: 4
- after/before: 3
- accepted/rejected: 2
- true/false: 1

Validation metric ranges on accepted examples:

- structured/neutral length: mean 18.8/17.8 words
- structured/reversed length: mean 18.8/18.9 words
- content Jaccard structured-neutral: mean 0.629, min 0.350
- content Jaccard structured-reversed: mean 0.833, min 0.684
- character similarity structured-reversed: mean 0.948

The packet is still small and semantically uneven. Several accepted examples have strong query priors or awkward query forms, e.g. the social-belief accepted/rejected example uses an unnatural query suffix. The validation prevented direct answer-word exposure, but it did not establish a high-quality, family-balanced relational dataset.

## Frozen forward scores

Scorer: `experiments/archive/frontier_consolidation/scripts/score_matched_probe.py`

### chck82

Files:
- Summary: `research/documents/frontier_consolidation/data/matched_graph_transfer_probe/frozen_chck82_scores_v2/matched_probe_frozen_score_summary.md`
- Paired effects: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/frozen_chck82_scores_v2/matched_probe_paired_effects.csv`

On the 10 accepted examples, protected `chck_82M` showed:

- query-only target-vs-distractor delta: **+3.3781** mean, target preferred in 9/10
- neutral delta: **+1.9693** mean, target preferred in 10/10
- structured delta: **+2.7805** mean, target preferred in 8/10
- reversed delta: **+1.2903** mean, target preferred in 6/10

Paired effects:

- structured minus neutral margin lift: **+0.8111**, bootstrap CI **[-0.1612, +1.7344]**, P(mean≤0)=0.0502
- structured minus reversed sensitivity: **+1.4902**, CI **[+0.7880, +2.2169]**
- neutral minus structured target-NLL improvement: **+1.5806**, CI **[+0.5667, +2.6869]**
- reversed minus structured target-NLL improvement: **-0.1610**, CI **[-0.7518, +0.3382]**

The large query-only and neutral priors mean the target/opposite word pair often encodes the answer without needing the context view. The structured/reversed margin difference is real in this tiny set, but it is not enough to justify training because the examples do not isolate the late official relation/state erosion phenomenon.

### chck100 comparison

Files:
- chck100 score summary: `research/documents/frontier_consolidation/data/matched_graph_transfer_probe/frozen_chck100_scores_v2/matched_probe_frozen_score_summary.md`
- 82-vs-100 comparison: `research/documents/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_82_100_comparison/matched_probe_82_100_comparison.md`

If this diagnostic captured the competence preserved at 82M and eroded by 100M, chck82 should dominate chck100 on structured lift/sensitivity. It did not. chck100 was essentially equal or slightly stronger:

- structured-neutral lift: chck82 **+0.8111**, chck100 **+0.8635**, 82-100 **-0.0523**
- structured-reversed sensitivity: chck82 **+1.4902**, chck100 **+1.5110**, 82-100 **-0.0209**
- neutral-minus-structured target-NLL improvement: chck82 **+1.5806**, chck100 **+1.6996**, 82-100 **-0.1190**
- reversed-minus-structured target-NLL improvement: chck82 **-0.1610**, chck100 **+0.0113**, 82-100 **-0.1724**

This is decisive against using the graph packet v0 construction/166 graph packet as an H100 training route. The matched probe measures a small synthetic/query-prior/local connective phenomenon that is not aligned with the protected 82M-vs-100M relation/state loss.

## Route conclusion

The current graph-preserving packet route does **not** support a 4M exposure screen.

What was learned:

1. Graph-conditioned generation can force relation-word preservation, but naive versions leak the exact relation vocabulary and become lexical reuse.
2. A held-out clue-word prompt can construct a small number of lexically controlled examples, but yield is low (10/30), family coverage is incomplete, and query priors remain large.
3. The accepted matched probe does not distinguish the protected 82M model from the degraded 100M continuation; therefore it is not measuring the official late-loss relation/state phenomenon.

Scientific interpretation: the route has not produced an explicit, verified `G` object that transfers across names, lexicalizations, and graphs without shortcut cues. Continuing to pretraining from this packet would replay the closed source-conditioning/lexical-compaction pattern under a new name.

## Next work suggested

Do not launch graph-packet H100 training from graph packet v0 construction/166 evidence. A useful next role is critical route review: decide whether to abandon graph-preserving generated views entirely, or rebuild the idea only with a more formal controlled construction where answer priors are balanced, graph instances are held out, and the 82M-vs-100M official loss is predicted before any training.

An independent route-2 hidden-readout/factorial causal test also failed: apparent identity readout did not become a family-general state-update component, and patch/remove probes remained at 0.5 held-out-family accuracy. This is evidence against treating synthetic relational readouts as general trainable mechanisms without stronger transfer tests.
