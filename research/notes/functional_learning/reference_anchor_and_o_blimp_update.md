# reference anchor and o blimp update reference-anchor clarification and O62065 BLiMP update

Created: 2026-09-08T10:02:20.437442+00:00

## CDI reference repair

The earlier analysis prose has been repaired so the earlier analysis CDI values are not read as immediate-parent preservation. In `combined_compact.csv` / `cdi_endpoint_rank_candidate_family.md`, the CDI endpoint bank is a 96-word probe anchored at `chck82_slow_scale1p75`:

- scale0.60 dense64: ΔNLL 0.034171, Δrank 27.39 vs chck82.
- clean64: ΔNLL -0.014410, Δrank -22.11 vs chck82.
- coherent86 itself: ΔNLL -0.076645, Δrank -68.11 vs chck82.

Therefore clean64 remains worse than coherent86 on this CDI bank by +0.062234 NLL and +46.00 rank. The scale-control conclusion is only the within-bank contrast: scale0.60 dense nearly matches clean source movement but is worse than clean by +0.048582 NLL and +49.50 rank.

## Dense-common scope repair

Dense-common CE/rank gains are familiar-row state-specific fit measurements: earlier analysis fixed 169 packed Qwen training-row examples and 1003 common support tokens. The values support the state-selective anchoring mechanism, but do not stand in for held-out transfer or official score components.

## O62065 BLiMP admission

O62065 BLiMP completed at raw score 68.51926840166846, report `AVERAGE ACCURACY` 68.52, with adapter-equipped `FrozenSlowPrivateDebertaV2ForMaskedLM`, 36,458,592 parameters, 995,584 private parameters, and evaluation return code 0. Rerunning strict admission in `experiments/archive/functional_learning/data/strict_split_eval_admission_after_o_blimp` admits BLiMP 68.52. O62065 still lacks ['COMPS', 'SuperGLUE'] and therefore has no Overall yet.

Files updated:

- `experiments/archive/functional_learning/data/mechanism_trade_revision/mechanism_trade_revision_data.json`
- `research/documents/functional_learning/data/mechanism_trade_revision/mechanism_trade_revision.md`
- `experiments/archive/functional_learning/drafts/A01_stageIII_integration_patch.tex`
- `experiments/archive/functional_learning/drafts/chinese_report_integration_bundle/A01_stageIII_integration_patch.tex`
- `experiments/archive/functional_learning/drafts/chinese_report_integration_bundle/mechanism_trade_revision.md`
- `experiments/archive/functional_learning/data/reference_anchor_clarification/reference_anchor_clarification.json`
