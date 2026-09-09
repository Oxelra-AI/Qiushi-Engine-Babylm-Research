# counterfactual micro world v3 static repair — Counterfactual micro-world v3 result and route boundary

Status: **MEASURED_AND_INTERPRETED**

This step built and scored a static repair of the counterfactual micro world freeze micro-world before using any checkpoint output. The repair was necessary because v2 contained a static linguistic artifact: the `near_far` subtype used one query template `is {target} from`, so the `near` alternative became ungrammatical (`near from`). v3 removes `near_far` and preserves subtype-compatible slot classes in renamed controls.

## Frozen object

- Freezer: `experiments/archive/representation_and_objectives/scripts/freeze_counterfactual_micro_world_v3.py`
- Frames: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3/counterfactual_micro_world_v3_frames.jsonl`
- Renamed controls: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3/counterfactual_micro_world_v3_renamed_controls.jsonl`
- Manifest: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3/counterfactual_micro_world_v3_manifest.json`
- Content SHA256: `ed60eb1ca541aae7b55dcaf99a66798d42b489ab85dcbd49ac5156a3b0d043bd`
- Frame SHA256: `88346ddbabfe06ab18e628c1a1f2ba4184ac7dd67f0b8e07220a022f1dbb0817`
- Renamed SHA256: `3bf8e5d390fdc1078997f0fbe907cdd3fce6b851519dee048f4817c4438aa17f`
- Pool SHA256: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
- Static quality: 400 frames, 100 each in `A_spatial_direct`, `B_transfer_role`, `C_reported_relation`, `D_state_update_order`; all target spans one token under legal16k and legal40k; no `near/far` surface artifact phrases remain; 198/400 frames have target prior ratio <=4.

## Scoring artifacts

- Scorer: `experiments/archive/representation_and_objectives/scripts/score_counterfactual_micro_world_v3.py`
- Full panel summary: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3_scores/micro_world_v3_panel_summary.json`
- Panel note: `research/notes/representation_and_objectives/micro_world_v3_panel_interpretation.md`
- Deeper interpretation: `experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3_scores/micro_world_v3_deeper_interpretation.json`
- Deeper note: `research/notes/representation_and_objectives/micro_world_v3_deeper_interpretation.md`
- D-state surface ablation script: `experiments/archive/representation_and_objectives/scripts/d_state_surface_ablation.py`
- D-state ablation summary: `experiments/archive/representation_and_objectives/data/d_state_surface_ablation/d_state_surface_ablation_summary.json`
- D-state ablation note: `research/notes/representation_and_objectives/d_state_surface_ablation.md`

## What v3 establishes

1. **The simple A/B/C templates are mechanically coherent but saturated in mature models.** On the mature legal16k/scale1.75/FineWeb models, `A_spatial_direct` and `C_reported_relation` are essentially 1.0 crossed-sign success, and `B_transfer_role` is usually 0.98–1.0 except legal40k variants. Cross-family shuffled contexts are low and renamed controls are stable. Thus the scorer is not merely reading target priors for these families.

2. **v3 rejects the sparse20 turnover artifact in one useful sense.** `mlm_only_20M` scores 0.2375 crossed success, but `coupled_aligned_20M` scores only 0.0050 and `coupled_shuffled_20M` 0.0150. The coupled models reduce full-EWoK stable-failure counts through turnover, yet they do not improve v3. This supports the aligned shuffled full ewok residual slices closure: coupled sparse20 did not form the intended crossed-context binding representation.

3. **v3 is not a reliable proxy for full official EWoK interaction strength.** Across the 11 targets with existing full-EWoK four-cell references, v3 crossed metrics have weak relation to EWoK accuracy and the wrong sign against stable-failure fraction: higher v3 crossed success often comes with more EWoK stable failures. The legal40k 8x480 and 12x384 checkpoints have better full-EWoK accuracy than scale1.75 80/100M but lower v3 success. This means v3 is confounded by tokenizer/template familiarity and broad maturity and must not be used as a training selector.

4. **The D-state family reveals a specific interference phenomenon, not a clean general binding measure.** All mature checkpoints score 0.0 on full v3 D contexts while solving explicit final-state forms. Full v3 D contexts look like: `The bag was closed. Kate closed the bag. Then Kate opened the bag. The bag is now {target}.` The model strongly chooses the repeated earlier state rather than the final updated state. This is a real and reproducible phenomenon, but it is narrower than full EWoK/GlobalPIQA binding.

## D-state surface ablation

The D-state ablation scored 19 existing checkpoints on four context forms:

- `full_v3`: original redundant context with initial state, same-state event, and final opposite event.
- `initial_plus_last`: initial state plus final event.
- `last_event_only`: only the final event.
- `explicit_final_state`: direct final state sentence (`The bag became open.`).

Key result:

- Mature models solve `explicit_final_state` at 1.0 crossed success.
- Mature legal16k models often solve `last_event_only` partly or strongly: seed43022 has 0.81 at 80M and 0.89 at 100M; scale1.75 is only 0.25–0.48 across 77–100M; legal40k and FineWeb allocation variants are lower.
- All mature models fail `full_v3` and `initial_plus_last` at 0.0 crossed success.
- The coupled sparse20 models are broadly damaged: they fail even `explicit_final_state` at 0.0 crossed success.

Scientific reading: the v3 D failure is not absence of lexical state semantics. It is an interference/overwrite failure: earlier state evidence overwhelms a later update, especially when the earlier state is repeated. This is close to a selective state-memory problem, but the current v3 object is too narrow and too template-dominated to justify training.

## Route boundary

The counterfactual micro world v3 static repair strategist note warned not to force the measure to rank chck_82M near the top. The evidence supports that warning. The chck_82M endpoint remains the practical broad-score SOTA candidate, but it is not a binding solution. v3 should not be tuned to recover that ranking, and no training should be launched directly from v3.

The useful scientific object extracted from v3 is more precise:

> Existing MLM-trained small models can learn direct relation words, recipient role statements, reported direct relations, and explicit final states, yet they often fail to let a later context update override an earlier conflicting state. The failure is especially strong under redundant repeated earlier-state evidence. This suggests the representation problem may be selective overwrite under interference rather than pairwise alternative choice alone.

## Next research work

The next step should deepen the representation hypothesis before new training:

1. Build a cleaner **interference-controlled state-update measurement** that varies only the amount and type of prior conflicting evidence while keeping the final update fixed: no prior state, prior state only, same-state repetition, distractor action, two-step contradictory update, and reported/believed update.
2. Score existing checkpoints first, including the legal16k/scale1.75/legal40k/FineWeb/coupled panels. The measurement should separate true final-update use from repeated-state copying and from target priors.
3. If a stable, official-surface-relevant signal appears, connect it to full EWoK rows involving material dynamics, temporal/order updates, and entity tracking before any training proposal.
4. If it does not connect to official surfaces, return to a broader representation-formation theory rather than changing templates to favor a known checkpoint.

No endpoint artifact was changed, no training was launched, and `chck_82M` remained untouched.
