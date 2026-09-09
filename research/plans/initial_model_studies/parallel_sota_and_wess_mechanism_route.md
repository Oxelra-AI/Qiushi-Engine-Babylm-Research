# babylm scoreboard — Parallel SOTA scoreboard and repaired WESS mechanism route

## Scientific constraints

1. **Efficiency and output discipline.** Mechanism work must remain tied to the BabyLM Strict-Small official 9-column Overall objective. The comparison must retain a protected internal best and keep every candidate accountable to 9/9 evaluation, not only local probes or partial columns.

2. **Do not scale WESS from v3 yet.** The v3 `wess_shuffled` branch did not truly destroy entity addressing. It used a consistent permutation for writing and reading, so high accuracy is expected. Its low raw slot-swap score was computed in the external entity coordinate rather than the model's internal permuted coordinate. Therefore v3 proves that WESS works with text-derived span representations under gold routing, but it does not yet test learned addressing or robustness to address corruption.

3. **Next mechanism work before BabyLM integration.** Add comparisons that genuinely destroy entity addressing and remove gold spans. If WESS still shows long-range overwrite generalization and slot-level intervention response, then DeBERTa/MLM integration can proceed in parallel.

## Current official 9/9 scoreboard

Durable files:
- `scripts/update_babylm_scoreboard_revision_182.py`
- `data/babylm_scoreboard.json`
- `notes/babylm_scoreboard.md`

Current protected internal best:
- `protected_internal_debertav2_8x480_wwm_100M`
- Overall = **40.5269**
- Complete 9/9 columns: BLiMP 66.76, Supplement 59.88, EWoK 52.19, Entity 22.62, COMPS 52.19, SuperGLUE 68.0218, GlobalPIQA 35.635, Reading 7.62, AoA -0.1745.

Current locally verified public leader:
- `go76dof/wwm_curriculum_simplification_40k`
- Overall = **41.8011** (leaderboard nominal 41.80)
- Gap protected-to-leader = **1.2742 Overall** = **11.4677 summed score points**.

Protected-minus-leader column gaps:
- BLiMP -0.440
- Supplement +3.840
- EWoK -3.880
- Entity -5.830
- COMPS -1.380
- SuperGLUE -1.768
- GlobalPIQA -4.030
- Reading +2.195
- AoA -0.174

Partial-coordinate warning:
- S1 known-seven sum = 296.525; needs SuperGLUE+AoA > 79.685 to exceed leader, about +11.838 over protected SG+AoA. Not a credible shortcut.
- S2 known-seven sum = 290.205; needs SuperGLUE+AoA > 86.005, about +18.158 over protected SG+AoA. Not a route.

Operational rule: every BabyLM-scale run must be inserted into the scoreboard. If it has fewer than nine columns, record the exact missing-column sum needed to beat the leader; do not treat partial columns as SOTA progress.

## Repaired interpretation of WESS v3

What v3 supports:
- WESS with gold entity/state spans and gold entity routing solves nonce state updates from text-rendered events (`dax moves to lup.`) where endpoint and step-supervised Transformers fail on long/overwrite splits.
- Gold-routed WESS has strong slot-swap transfer (0.904) and write-ablation reversion (0.996), giving causal evidence of slot-mediated state.

What v3 does not support:
- It does not show learned span detection.
- It does not show learned entity-to-slot addressing.
- The `wess_shuffled` high accuracy is not surprising, because write and read share the same consistent shuffled mapping. It is not a true address-breaking control.
- The low raw slot-swap score for shuffled WESS must be recomputed in the internal permuted coordinate before interpretation.
- It is not yet evidence that BabyLM-scale MLM integration will improve official Entity/EWoK/GlobalPIQA.

## Next WESS mechanism experiment: v4 address/span challenge

### Required controls

1. **Gold route WESS** — baseline from v3: gold spans + gold entity-to-slot route.
2. **Consistent-permutation WESS with corrected intervention** — same as v3 shuffled, but compute slot-swap transfer in the model's internal permuted coordinate. This clarifies that consistent remapping preserves the algorithm.
3. **Eventwise random route WESS** — for each event, write the participant to a random slot independent of query/read mapping. This truly destroys persistent entity addressing; accuracy and slot interventions should collapse if entity indexing is required.
4. **Wrong-entity write WESS** — write each event update to a different entity's slot while reading the queried entity's gold slot. Should fail on overwrite/long and slot interventions.
5. **No-bottleneck WESS** — allow endpoint readout to attend to the full encoded text in addition to slots. If it matches WESS but fails slot interventions, it is a bypass; if it improves while slots weaken, the bottleneck was essential.
6. **Learned span WESS** — remove gold entity/state span positions. Add lightweight token classifiers or attention heads to select event entity/state positions from text. Compare to gold-span WESS.
7. **Span-shuffled learned/gold control** — give correct text but shuffle state span assignment within each event batch; should fail if WESS depends on correct extracted state.
8. **Parameter-matched Transformer** — add capacity to endpoint Transformer comparable to WESS and check whether long/overwrite remains weak.

### Measurements

Keep five seeds and the same split family: iid, new entity, new state, new entity+state, long, heavy overwrite. Add:
- slot-swap transfer in external and internal coordinates;
- last-write ablation reversion;
- eventwise selective write accuracy: after each event, only the participant slot should change answer-compatible state;
- persistence of nonparticipating slots;
- entity-renaming equivariance;
- state-renaming/candidate permutation equivariance;
- learned span accuracy if span heads are used.

### Decision rule for BabyLM integration

Proceed toward DeBERTa/MLM integration only if:
- gold-span WESS remains strong;
- true address-destruction controls collapse;
- learned-span WESS keeps most of the long/overwrite and intervention signal, or the failure localizes clearly to span detection rather than state tracking;
- no-bottleneck/parameter-matched Transformer controls fail to reproduce causal slot dependence.

## Parallel SOTA work

The next Execute steps should not only run micro-world experiments. In parallel, prepare the BabyLM candidate path so a successful mechanism can quickly become an official 9/9 coordinate:

- identify the S1/8x480 trainer branch where a WESS auxiliary head could be added without breaking HF save/load;
- define word-counted synthetic episode mixture sizes (e.g., 5%, 10%, 15% of 10M) with equal-word WWM-only and address-destroyed controls;
- plan a 10M screen with fast columns plus full EWoK/Entity/GlobalPIQA/Reading, followed by full 9/9 only for candidates that move the target cluster;
- update `data/babylm_scoreboard.json` after every full candidate and preserve protected best.

The immediate next highest-value execution is v4 mechanism repair. If v4 passes, launch DeBERTa/MLM integration in parallel with official-score pipeline preparation.
