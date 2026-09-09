# binding transfer and composition prediction binding transfer, private-phase design, and next prediction

## What the first official-compatible binding readout shows

The exposure-matched single-frame binding private phase from earlier analysis is a strong in-format assignment learner but a poor official Entity/BabyLM transfer endpoint.

Evidence:

- In-format local readout, ep25, alpha 1.0: `research/documents/relation_learning/data/binding_alpha_no_context_strata_ep25/summary.md` reports 92/200 full-context paired joint successes and weak no-context performance.
- Official-compatible cheap7/Entity readout:
  - `research/documents/relation_learning/data/binding_eval_readout_alpha075_100/summary.md`
  - alpha 0.75: cheap7 41.1579, -2.8016 vs chck_82M and -3.0236 vs coherent86; Entity 26.32, -2.00 vs coherent86.
  - alpha 1.0: cheap7 40.2200, -3.7394 vs chck_82M and -3.9614 vs coherent86; Entity 23.23, -5.09 vs coherent86.
- Operation strata at `research/documents/relation_learning/data/binding_eval_readout_alpha075_100/entity_strata/summary.md` reveal the structure hidden by the aggregate:
  - alpha 0.75 gains a narrow relevant-update face relative to chck_82M: rel_ge1 +0.27 pct, rel_ge1_postrel_ops0 +2.05, rel_ge3_postrel_ops0 +1.62, stale_available_not_gold +1.72.
  - the same alpha damages retention/no-update behavior: rel_eq0 -11.23 pct, rel_eq0_irrelevant_ops_gt0 -12.69, rel_eq0_irrelevant_ops_4to6 -21.36.
  - alpha 1.0 amplifies the damage and loses even most relevant-update structure.

Scientific reading: answer-only single-frame binding did not refute the mechanism; it instantiated it too narrowly. The private branch learned a visible-context packet readout that can help when the official item asks for the last relevant update with no later irrelevant operation, but it also applies update-like behavior where official Entity demands retention. This is the same relation/form specificity already seen in the VIEW/REPEAT family: a practiced readout transfers only when the deployment relation and surface make the same variable uniquely useful.

## binding transfer and composition prediction repair of the private-phase composition trainer

The active trainer was rewritten at `experiments/archive/relation_learning/scripts/private_binding_composition_trainer.py` and smoke-tested at `experiments/archive/relation_learning/data/private_binding_composition_smoke`.

Key design change:

1. Binding rows now receive answer CE only on the answer span.
2. On the same answer-masked binding input, a private-off/private-on KL term is applied to every non-answer token, so the private branch is leashed on source, update, entity names, query frame words, punctuation, and special tokens while still being free to change the answer slot.
3. The default binding rows are now the deterministic frame-varied rows from `experiments/archive/relation_learning/data/frame_varied_recombination_rows` when present. The default `reference_mixed` frame schedule samples a reference-sized number of A/B pairs per epoch across five train-seen frames, so 25 frame-varied epochs are charged close to the 25 single-frame epoch budget instead of multiplying exposure by five.

Smoke evidence:

- `train_config.json` records `status=PRIVATE_BINDING_COMPOSITION_CONFIG`, `binding_train_rows=.../recombination_train_frame_seen.jsonl`, `binding_frame_schedule=reference_mixed`, and `objective=ordinary suffix CE/KL plus binding answer CE with private-off KL on every non-answer binding token`.
- `training_log.jsonl` contains binding updates with non-answer KL positions (249 and 262 in the smoke) and small but nonzero `binding_nonanswer_kl_loss`, showing that the leash term is active.

The repair applies to composition runs that had not yet begun training. Results carrying an older `PRIVATE_BINDING_COMPOSITION_CONFIG`, or failed runs, remain unsuitable for the corrected comparison; a corrected rerun would require the same word accounting.

## Frame wording check

`research/documents/relation_learning/data/frame_phrase_entity_overlap/summary.md` compares the eight deterministic binding query frames to official Entity prompts. None of the frames uses official container-operation words such as box, contains, move, remove, put, or nothing, and the official tail prompt examples are all `Box N contains`. Thus the frame-varied rows are not near copies of the official Entity query surface; they test relation/form robustness rather than benchmark-shaped template memorization.

## Working prediction for the next private-phase arms

Single-frame versus frame-varied at matched charged words is the private-phase analogue of exact recurrence versus varied restatement:

- Single-frame answer credit should remain strongest on the practiced local frame, especially at high private amplitude.
- Frame-varied answer credit plus non-answer KL should reduce the format cue, improve unseen-frame local readout relative to single-frame, and lessen official rel_eq0 damage by forcing the private branch to keep the non-answer context close to the slow path.
- Coherent replay plus binding should hold more of coherent86's BLiMP/Supplement/GlobalPIQA/Reading profile than answer-only binding. If it also preserves the alpha0.75 relevant-update gains, it becomes the first structurally plausible route to improve the 42.1210 reference; if not, the result still maps which finite-budget private practice buys which competence profile.
- Binding continuation from coherent86 is expected to be more fragile than joint composition because it changes an already specialized private branch without replay pressure, but it directly tests whether the coherent branch can absorb answer-slot credit without losing its broad surface.

Immediate next measurements when tasks deliver:

1. After the alpha0.5 official evaluation completes, run the same readout script with alpha0.5 included.
2. For each composition endpoint, inspect `scientific_metrics.json` for `PRIVATE_BINDING_COMPOSITION_CONFIG` and frame/default rows before trusting the result.
3. Evaluate cheap7/Entity and parse operation strata against both chck_82M and coherent86.
4. For the local binding readout, parse pair IDs by frame to separate train-seen and eval-unseen frames; this is required to decide whether frame variation actually changed form transfer rather than only aggregate in-format performance.
