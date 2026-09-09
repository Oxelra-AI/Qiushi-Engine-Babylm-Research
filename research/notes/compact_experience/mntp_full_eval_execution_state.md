# mntp full eval execution state — MNTP auxiliary full evaluation execution state

## Direct experiment being measured

Run under evaluation:

- Training run: `training/runs/mlm_mntp_aux015_seed43022/`
- Endpoint: `hf_model/chck_100M`
- Intervention: MLM-primary same-corruption token-shift MNTP auxiliary, target aux/MLM gradient contribution ratio `0.15`, full WWM-MLM stream preserved on every batch
- Matched clean-Qwen factors: 8×480 DeBERTa-v2-style architecture, 16k tokenizer, seed43022 coordinate, clean-Qwen same-window data, 100M word exposure, 100 checkpoint ladder

Verified before launch:

- `hf_model/` has 100 `chck_*M` checkpoint directories.
- `chck_100M/` has `config.json`, `model.safetensors`, tokenizer files.
- `scientific_metrics.json` has status `MLM_MNTP_AUXILIARY_MANIFEST`, `actual_steps=2515`, `actual_word_exposure=100000000`, `parameter_count=34467424`, `mntp_variant=token_shift`, `aux_target_ratio=0.15`, `saved_checkpoints` length 100.

## Evaluation launch

Submitted background task:

- Command: `CUDA_VISIBLE_DEVICES=0 bash scripts/launch_full_eval.sh`
- CWD: `experiments/archive/compact_experience`
- Output root: `data/mlm_mntp_full_eval/`
- Intended payload: `data/mlm_mntp_full_eval/per_target/mlm_mntp_aux015_100M.json`

The launch script initially expected a different status string, but the real training manifest is complete. mntp full eval execution state patched the preflight to accept the actual manifest status and verify completion by steps, exact exposure, and checkpoint count.

The launch script also had a post-evaluation import typo (`compute_official_overall`) that is not part of the canonical evaluator path. mntp full eval execution state patched it to `compute_overall_from_tasks`. The decisive result remains the evaluator payload written by `custom_endpoint_full_eval.py`/`full_overall_eval_runner.py`.

## Result interpretation guard

Interpretation constraint: if MNTP is negative, do not automatically promote the old Phase2 12×384/40k/LAMB package. That earlier recipe changed depth/width, tokenizer/vocab, optimizer/LR, sequence schedule, masking schedule, and data state together. It is not an isolated clean-Qwen × architecture experiment.

The next route should be selected from the actual execution state column structure:

- If MNTP crosses 41.8: verify endpoint integrity, replicate, and prepare official-compatible submission materials.
- If MNTP improves over clean-Qwen but remains below 41.8: identify which columns improved and run one matched strengthening/replication; do not broaden into a bundled recipe.
- If MNTP is flat/negative: close this exact same-stack auxiliary construction as a SOTA route and choose one isolated next intervention, such as tokenizer isolation, minimal representation change on the current recipe, or a compact matched data-by-recipe screen.

Helper prepared:

- `scripts/analyze_mntp_full_eval.py` reads the final payload and compares execution state to clean-Qwen, causal15, and the visible leader, saving `data/mntp_eval_interpretation.json` when run after the payload exists.
- `plans/post_mntp_route_logic.md` records the mechanism-first route logic and supersedes overconfident fallback language from execution state.
