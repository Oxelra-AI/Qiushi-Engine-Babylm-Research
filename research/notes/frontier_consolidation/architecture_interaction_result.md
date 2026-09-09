# DeBERTa compact × positional-score interaction: observed result and common-copy disambiguation

## Scientific question

The completed analysis includes the mature four-cell readout launched at earlier analysis:

\[
I_{\text{nodis}-\text{full}}=(\text{compact}-\text{repeat})_{\text{no\_disentangle\_abs}}-(\text{compact}-\text{repeat})_{\text{full p2c,c2p}}.
\]

The purpose is to determine whether the compact semantic second-view data effect in DeBERTa depends on the DeBERTa disentangled positional score package (`pos_att_type=[p2c,c2p]`) rather than treating compact views as an architecture-general data principle. initialization parity and interaction interpretation established that the plain `no_disentangle_abs` contrast is valid but bundled: removing the 32 positional projection tensors shifts the construction RNG stream (53/140 same-named tensors differ at init) and reduces capacity from 34,467,424 to 30,773,344 parameters.

## Training completion and integrity

Completed training arms:

- `full_p2c_c2p_abs/repeat` and `no_disentangle_abs/repeat`.
- `no_disentangle_abs/compact`.

Hardened integrity output: `experiments/archive/frontier_consolidation/data/architecture_interaction_integrity_final/architecture_interaction_integrity.{json,md}`.

All four selected-eval cells are valid and ready:

- full compact: existing legal compliant tokenizer retrain status compact reference, 100M exposure, 2,529 updates, 34,467,424 params, `pos_key_proj=16`, `pos_query_proj=16` at 80M/100M.
- full repeat: new earlier analysis repeat, 100M exposure, 2,529 updates, 34,467,424 params, `pos_key_proj=16`, `pos_query_proj=16`.
- no-disentangle compact: new earlier analysis compact, 100M exposure, 2,529 updates, 30,773,344 params, `pos_key_proj=0`, `pos_query_proj=0`.
- no-disentangle repeat: new earlier analysis repeat, 100M exposure, 2,529 updates, 30,773,344 params, `pos_key_proj=0`, `pos_query_proj=0`.

All use the legal spatial repair route status tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`, exact compact/repeat stream SHAs, seeds 43/43022/43023, WWM p=0.15, AdamW lr0.001, batch256, seq256, `lr_total_steps=2529`.

## Selected official-compatible panel

Panel: `experiments/archive/frontier_consolidation/data/architecture_interaction_selected_panel_final/architecture_interaction_selected_panel_summary.json`.

The first selected-panel run timed out after completing all 80M rows and starting 100M. A no-force rerun reused completed rows and finished the missing 100M rows. Final panel has 8 rows and 0 problems.

### 80M

Full DeBERTa compact-minus-repeat:

- cheap7 -0.1414 (GlobalPIQA -3.485 masks stable positives)
- cheap6_no_GlobalPIQA +0.4158
- cheap5_no_GlobalPIQA_Reading +0.4200
- EWoK_plus_Entity_sum +1.9100
- Supplement +0.3200, EWoK +1.2700, Entity +0.6400, COMPS +0.3400

No-disentangle compact-minus-repeat:

- cheap7 -0.6793
- cheap6_no_GlobalPIQA -0.7067
- cheap5_no_GlobalPIQA_Reading -0.6620
- EWoK_plus_Entity_sum -2.5000
- Supplement -0.5100, EWoK -2.6300, Entity +0.1300, COMPS +0.3500

Interaction `nodis - full`:

- cheap6_no_GlobalPIQA -1.1225
- cheap5_no_GlobalPIQA_Reading -1.0820
- EWoK_plus_Entity_sum -4.4100
- EWoK -3.9000
- Supplement -0.8300
- Entity -0.5100
- GlobalPIQA +2.9700 (not a stable-family rescue)

### 100M

Full DeBERTa compact-minus-repeat:

- cheap7 -0.3900 (again GlobalPIQA -3.500 masks stable positives)
- cheap6_no_GlobalPIQA +0.1283
- cheap5_no_GlobalPIQA_Reading +0.0680
- EWoK_plus_Entity_sum +0.7700
- Supplement +0.2100, Entity +1.6300, COMPS +0.4800, EWoK -0.8600

No-disentangle compact-minus-repeat:

- cheap7 -0.0429
- cheap6_no_GlobalPIQA -0.1308
- cheap5_no_GlobalPIQA_Reading -0.0660
- EWoK_plus_Entity_sum -1.0600
- Supplement +0.0800, Entity -0.0900, COMPS +0.6100, EWoK -0.9700

Interaction `nodis - full`:

- cheap6_no_GlobalPIQA -0.2592
- cheap5_no_GlobalPIQA_Reading -0.1340
- EWoK_plus_Entity_sum -1.8300
- Entity -1.7200
- Reading -0.8850
- GlobalPIQA +3.9850 (again opposite to stable-family interpretation)

### Mean over 80M/100M

- full compact-minus-repeat: cheap6_no_GlobalPIQA +0.2721; cheap5 +0.2440; EWoK_plus_Entity_sum +1.3400.
- no-disentangle compact-minus-repeat: cheap6_no_GlobalPIQA -0.4188; cheap5 -0.3640; EWoK_plus_Entity_sum -1.7800.
- interaction `nodis - full`: cheap6_no_GlobalPIQA -0.6908; cheap5 -0.6080; EWoK_plus_Entity_sum -3.1200.

The score-column readout therefore shows that the stable compact-minus-repeat movement seen in full DeBERTa does not survive removal of the p2c/c2p positional-score package. This is not a useful endpoint result and not a leaderboard action; it is mechanism evidence.

## Item/subtask intervals

Pair intervals: `experiments/archive/frontier_consolidation/data/architecture_interaction_pair_intervals`.
Four-cell item bootstrap: `experiments/archive/frontier_consolidation/data/architecture_interaction_four_cell_bootstrap`.

All item analyses have 170,722 common items, zero left/right-only items, and zero loader warnings. Reading is skipped for item-flip analyses because it is continuous.

Important item-level readings:

- At 80M, EWoK+Entity four-cell interaction is strongly negative at item level: -1.764 pp with item interval [-2.871, -0.499]; cluster interval crosses zero [-3.894, 0.229].
- At 100M, EWoK+Entity four-cell interaction remains negative: -1.340 pp with item interval [-2.598, -0.284]; cluster interval crosses zero [-3.155, 0.603].
- Stable_five item interaction is weak/mixed: 80M -0.066 pp interval crossing zero; 100M +0.486 pp item interval positive but cluster interval crossing zero. This means the broad raw item pool does not by itself prove a uniform collapse; the decisive score-column pattern is concentrated in official EWoK/Entity-style families and in stable-family aggregates.
- Supplement raw items move positively for no-disentangle even when official Supplement score interaction is negative at 80M and only slightly negative at 100M; as in earlier steps, BabyLM score aggregation and raw item movement are not interchangeable.

## Correct interpretation

The observed earlier analysis four-cell result is a real architecture-coordinate result:

1. In full DeBERTa, compact-minus-repeat is stable-positive after removing GlobalPIQA at both 80M and 100M.
2. In plain no-disentangle, compact-minus-repeat is stable-negative/near-zero on the same readout.
3. The interaction is especially negative on EWoK/Entity-style relation/state columns.
4. The positive GlobalPIQA interaction has the opposite sign and cannot rescue the stable-family interpretation.

Because initialization parity and interaction interpretation found initialization drift and capacity change, this result should be stated as: **the compact semantic second-view data effect is strongly tied to the full DeBERTa positional-score coordinate, but the plain no-disentangle contrast does not yet isolate the p2c/c2p score terms as the only cause.** It implicates the removed positional-score package together with capacity, gradient path, score composition, and initialization/optimization basin.

## Common-copy disambiguation launched

Common-copy control training was launched to address the initialization confound in the observed result.

Launched control runs:

- Common-copy no-disentangle compact DeBERTa training, run dir `experiments/archive/frontier_consolidation/training/runs/commoncopy_nodis_compact_deberta100M_seed43022`.
- Common-copy no-disentangle repeat DeBERTa training, run dir `experiments/archive/frontier_consolidation/training/runs/commoncopy_nodis_repeat_deberta100M_seed43022`.

These use the verified `common_copy_nodis_trainer_wrapper.py`, source full `p2c,c2p`, target `pos_att_type=[]`, legal compact/repeat 100M streams, legal tokenizer, seeds 43/43022/43023, WWM 0.15, batch256/seq256, AdamW lr0.001, 100M exposure, 10M checkpoints.

Common-copy removes the RNG-stream drift by copying all 140 common tensors from the same full DeBERTa init into the no-disentangle target while still omitting the 32 positional projection tensors. Capacity remains 30,773,344, so the parameter-count difference remains part of the interpretation.

Decision reading after common-copy:

- If common-copy no-disentangle still collapses compact-minus-repeat on stable families, the evidence becomes much stronger that the removed p2c/c2p positional-score pathway itself is load-bearing for the DeBERTa compact-view effect.
- If common-copy no-disentangle recovers full-like compact-minus-repeat, then the earlier analysis collapse was mostly an initialization/basin artifact rather than positional-score necessity.
- If common-copy is intermediate, the effect is shared between positional-score path removal and optimization basin/capacity.

## Related Scientific Evidence

The paired-world route remained a possible subsequent research direction after the DeBERTa positional adjudication, not an immediate BabyLM training object.

Relevant evidence:

- RoBERTa HS/LS/HD/LD factorial: selected cheap6 appeared positive (+0.9717), but item reading showed EWoK+Entity essentially zero/negative and the aggregate was driven by BLiMP-island/COMPS composition; not a transferable contextual-anchor principle.
- Memory closure: four raw-token memory variants failed; hard-coordinate memory works when roles are supplied, so the bottleneck is latent occurrence-role assignment from raw text, not the update/read operation itself.
- Cross-data binding comparison agrees binding weakness is generic DeBERTa at this scale and not compact-specific.
- The same-world source/bridge role-family core is real but tiny: 11/42 retained source↔bridge families, not enough for training.
- A broader source-attested paired-world substrate contains deterministic role reversals in independent event records (tennis 35,715 reversed unordered pairs; BWF 6,162; LaLiga 149; football-data many more). This overcomes the same-world sparsity but has shortcut and relation-family risks: score-visible sports outcomes may be solved by numeric comparison rather than role binding, and `defeated` is one relation family.

The 42.1210 score target had been met. A generalizable, transferable sample-efficient learning principle, algorithm, architecture, or theory remained an open scientific objective.

## Planned Adjudication

After the common-copy initialization records and 100M metrics become available, the planned comparison uses the same 80M/100M selected panel and item bootstrap with common-copy no-disentangle replacing plain no-disentangle. This adjudication remained a prerequisite for broader claims about role-equivariant paired worlds, latent occurrence-role assignment or DeBERTa-specific compact-view dependence.
