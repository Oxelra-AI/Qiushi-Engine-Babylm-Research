# dualview pending panel interim interim synthesis while dual-view cheap7 panel is pending

## State entering dualview pending panel interim

The repaired aligned/shuffled 20M dual-view trainings and exact `mlm_only` reference exist. The aligned/shuffled arms are matched in training accounting and differ only by source correspondence; `mlm_only` is the exact no-aux adapter scaffold under the corrected trainer and 100M LR horizon.

Outstanding evaluation tasks from dualview panel readiness and budget:

- Aligned cheap7: pending.
- Shuffled cheap7: pending.
- MLM-only cheap7: pending.

Completed MLM-only control: exact `mlm_only` 20M cheap7 = **39.78642857142857**, +0.122857 over spatial repair route status 20M. Scores: BLiMP 59.54, Supplement 58.43, EWoK 50.10, Entity 18.39, COMPS 50.70, GlobalPIQA 32.665, Reading 8.68. Canonical summary: `data/dualview_mlm_only_20m_summary/dualview_mlm_only_20M_summary.json`. This means the aligned arm must beat a nontrivial exact adapter-scaffold control, not only spatial repair route status.

## New dualview pending panel interim CPU/mechanism work

### Parameter geometry

`scripts/dualview_parameter_geometry.py` was written and run. Output: `data/dualview_parameter_geometry/dualview_parameter_geometry.{json,md}`.

Key numbers:

- aligned vs shuffled: stock_all rel-L2 0.474050, cosine 0.887713; adapter_all rel-L2 0.540217, cosine 0.851250.
- aligned vs mlm_only: stock_all rel-L2 0.600789, cosine 0.812845; adapter_all rel-L2 0.544752, cosine 0.876818.
- shuffled vs mlm_only: stock_all rel-L2 0.603046, cosine 0.811332; adapter_all rel-L2 0.582204, cosine 0.864048.
- adapter_up RMS: aligned 0.03158, shuffled 0.03418, mlm_only 0.01323.

Interpretation: the dual-view aux signal makes the adapter substantially different from `mlm_only`, but aligned/shuffled are both far from `mlm_only` in stock parameters by 20M. This is not direct functional evidence; it means the current construction must be read as a train-time system comparison with budget substitution and co-adaptation, not as a pure isolated private-head probe.

### Complete panel analysis script

`scripts/dualview_complete_panel_analysis.py` was written and AST-checked. It should be run after all three cheap7 payloads exist. It will:

1. run the validated sentinel comparison with spatial repair route status base and aligned/shuffled/mlm_only candidates;
2. combine score summaries with the training-audit, budget-substitution audit, training-trajectory comparison, and dualview pending panel interim parameter geometry;
3. output `data/dualview_complete_panel_analysis/dualview_panel_complete.{json,md}`.

Intended command:

```bash
python -B experiments/archive/frontier_consolidation/scripts/dualview_complete_panel_analysis.py --label dualview_panel_complete
```

### independent_review verifier and generator insights

independent_review verifier integration: . Generator branches: , `...sub2_generator.md`.

Most important verifier correction: the lower aligned aggregate aux loss is expected because the conditioned view contains the true source at scoring time. It does **not** prove the source free transfer synthesis source-free transfer object. The decisive next functional checks are:

1. per-view auxiliary NLL split: conditioned vs source-free, aligned vs shuffled vs mlm_only;
2. adapter-off evaluation on the same auxiliary views, testing whether differences remain in the stock backbone;
3. budget-matched `mlm_only` at exactly the dual-view main prefix (481 updates / 19.021584M main words) if score comparison is ambiguous;
4. cheap7/sentinel score panel, still necessary for competence value.

Generator branches suggested serious successor variants if current construction is weak:

- auxiliary-only/frozen-stock adapters so main MLM never traverses changing adapters;
- word-budget-neutral pair substitution or activation transport that reuses main source-free rewrite activations instead of charging duplicate rewrite views;
- sparse auxiliary allocation to high-uncertainty, source-token-absent changed spans to reduce auxiliary charge;
- source-derived stop-gradient targets rather than source as a second input view;
- adapter/off or interpolation checks to separate private-branch gain from stock co-adaptation.

### Per-view functional-probe machinery

`scripts/dualview_aux_function_probe.py` was written and repaired. It replays the trainer's deterministic batches/masks and computes separate conditioned and source-free auxiliary NLLs for aligned/shuffled/mlm_only checkpoints on the same auxiliary units. It can also disable adapters at inference.

A tiny 10-batch pilot after cache repair succeeded and only validates the script, not the mechanism. Pilot output: `data/dualview_aux_function_probe/pilot10_aligned/pilot10_aligned.{json,md}`.

The pending adapter-on and adapter-off probes use the following outputs:

- Adapter-on full per-view probe: `data/dualview_aux_function_probe/full_adapter_on/full_adapter_on.json`.
- Adapter-off comparison follows the adapter-on result: `data/dualview_aux_function_probe/full_adapter_off/full_adapter_off_adapter_off.json`.

These probes are analysis of existing checkpoints, not training. They may reveal whether any aligned benefit is in source-free prediction or only in conditioned views.

## Budget-matched no-aux control not yet launched

The trainer saved only 1M checkpoint intervals and final checkpoints; `mlm_only` has `chck_19M`, `chck_20M`, and `final`, but not an exact 481-update / 19,021,584-word checkpoint matching the dual-view main-prefix state. A cheap control can be run by launching `dual_view_corrected_trainer.py --mode mlm_only --max_word_exposure 19021584 --checkpoint_words 19021584` under the same seeds/config. This should take less than the completed 20M `mlm_only` run and would isolate dual-view mechanism from the last 978,416 main words displaced in the current aligned-vs-mlm_only comparison.

Do **not** launch it automatically if the aligned/shuffled/mlm_only panel already decisively rejects the current construction. Launch it if aligned beats shuffled but is near or below `mlm_only`, or if the score interpretation hinges on budget substitution.

## Current decision logic

After aligned and shuffled evaluations finish:

1. inspect their summary JSONs and payload paths;
2. run `dualview_complete_panel_analysis.py`;
3. interpret:
   - aligned vs shuffled: correspondence-specific competence at fixed aux budget;
   - aligned vs mlm_only: competence under real legal budget substitution;
   - fragile families: whether it repeats scale1.75/U256 EWoK/Reading/Supplement losses;
   - per-view probes: whether the source-free transfer object actually exists in the trained model.

Current best complete legal endpoint is still scale1.75 100M Overall 41.57074653643003, below the 41.8 target. No 100M dual-view run is justified until the 20M score and functional evidence are favorable.

## dualview pending panel interim auxiliary signal potential audit

`scripts/aux_signal_potential_audit.py` was written and run. Output: `data/aux_signal_potential_audit/aux_signal_potential_audit.{json,md}`.

Main result: the current broad WWM-driven auxiliary target inventory is diluted relative to the non-copy edit phenomenon that Steps122-123 actually supported. Across 12,155 legal pairs, only 14.43% of alphanumeric rewrite token pieces are both changed and source-token-absent; 17.29% of alphanumeric word groups contain such pieces. A sparse utility ranking by changed-source-absent pieces per charged auxiliary word is steep: the top 20% of pairs use only 18.63% of full auxiliary charge but capture 38.62% of changed-source-absent pieces; the top 40% use 39.33% charge and capture 65.73% of the signal.

Interpretation: if aligned beats shuffled but loses to `mlm_only`, a likely repair is not to extend the broad auxiliary arm unchanged, but to concentrate auxiliary exposure on high-utility, source-token-absent changed spans and reduce duplicate rewrite/source charge. This remains benchmark-independent and uses only legal corpus structure plus frozen-source/rewrite edit analysis.

## dualview pending panel interim aligned and mlm_only cheap7 results

Two of the three cheap7 panel results are now known:

- exact `mlm_only`: cheap7 **39.78642857142857**, +0.122857 vs spatial repair route status 20M. It is a nontrivial adapter-scaffold control, not the negative adapter matched horizon plan approximation. Summary: `data/dualview_mlm_only_20m_summary/dualview_mlm_only_20M_summary.json`.
- broad dual-view `aligned`: cheap7 **38.76285714285714**, -0.900714 vs spatial repair route status 20M and -1.023571 vs exact `mlm_only`. Summary: `data/dualview_aligned_20m_summary/dualview_aligned_20M_summary.json`.

Partial sentinel comparison: `data/dualview_sentinel_compare/aligned_mlm_only_partial.{json,md}`. The aligned failure is broad rotation, not a small uncertainty: BLiMP -2.96, Supplement -3.69, Entity -0.74, Reading -1.145 vs spatial repair route status; EWoK +0.38 and GlobalPIQA +1.91 are the only column gains. Against `mlm_only`, aligned loses cheap7 by -1.0236; `mlm_only` recovers BLiMP (+2.81), Supplement (+6.67), Entity (+0.48), COMPS (+0.50), Reading (+1.155), while aligned retains EWoK (+1.01) and GlobalPIQA (+3.44).

This already fails the predeclared requirement that aligned beat the exact no-aux reference before any long extension. The remaining shuffled score is still useful for attribution (whether true correspondence is better than false correspondence at fixed aux budget), but cannot rescue the current broad auxiliary construction as a 100M route.

## Sparse pair-data variants built

`scripts/build_sparse_aux_pair_data.py` was written and run. Outputs: `data/sparse_aux_pair_data/sparse_aux_pair_data_summary.{json,md}` and filtered pair-data files:

- `top05`: `data/sparse_aux_pair_data/top05/sparse_aux_pair_data_top05.json`
- `top10`: `data/sparse_aux_pair_data/top10/sparse_aux_pair_data_top10.json`
- `top20`: `data/sparse_aux_pair_data/top20/sparse_aux_pair_data_top20.json`
- `top40`: `data/sparse_aux_pair_data/top40/sparse_aux_pair_data_top40.json`
- `top60`: `data/sparse_aux_pair_data/top60/sparse_aux_pair_data_top60.json`

Deterministic WWM replay forecast under the 20M charged cap:

- all pairs: 481 updates, 19.022M main words, 0.967M aux words, 137,874 aux targets.
- top05: 504 updates, 19.930M main words, 0.0397M aux words, 6,220 aux targets.
- top10: 503 updates, 19.890M main words, 0.0862M aux words, 13,004 aux targets.
- top20: 501 updates, 19.812M main words, 0.1836M aux words, 27,056 aux targets.
- top40: 495 updates, 19.575M main words, 0.3884M aux words, 55,250 aux targets.
- top60: 490 updates, 19.378M main words, 0.5939M aux words, 83,280 aux targets.

These files make a cheaper successor test possible if the remaining shuffled/per-view evidence says the source-correspondence signal is real but the broad dose is budget-negative.
