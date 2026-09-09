# ewok item flip audit — EWoK item-flip relation analysis

This CPU analysis reconstructs official EWoK correctness from four existing prediction files and the pristine official EWoK gold rows. It uses the official scoring rule: a row is correct when the selected prediction equals `Context1 + " " + Target1`. The prediction files store selected alternatives only, so they support item-flip and metadata localization but not log-likelihood margin analysis.

Key result:

- Total EWoK items: 7,618.
- Micro treatment×seed interaction: -2.389 percentage points, close to the official macro interaction -2.462 from the same-coordinate 2×2.
- Worst official domains: material-dynamics (-17.532), physical-dynamics (-10.833), spatial-relations (-8.367), physical-interactions (-5.935), social-relations (-2.132), quantitative-properties (-0.955).
- Direct/dynamic relation groups are especially unstable: direct material-dynamics, direct physical-dynamics, direct physical-interactions, material context-difference, concept-swap targets, and some variable-swap groups.
- Frequent negative correctness patterns include `0110` (clean430 wrong, reinvest430 right, clean431 right, reinvest431 wrong), `0100`, `1110`, `0111`, and `0010`; pattern order is clean430, reinvest430, clean431, reinvest431.

Next use:

- Use `experiments/archive/representation_and_objectives/data/ewok_item_flip_audit/ewok_negative_interaction_examples.csv` as the first target list for margin rescoring and relation-feature inspection.
- For true margins, instrument `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_pipeline/sentence_zero_shot/compute_results.py`: `compute_mlm_results` builds per-candidate summed log-probs and passes them to `rank_and_evaluate`, but only selected alternatives are saved. Export candidate scores before the argmax.

Full machine-readable result: `experiments/archive/representation_and_objectives/data/ewok_item_flip_audit/ewok_item_flip_audit.json`.
