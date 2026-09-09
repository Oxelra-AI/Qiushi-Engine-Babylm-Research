# compact core full loss anatomy compact-core complete-surface loss anatomy

CPU-only comparison of the inherited clean-Qwen full endpoint and frontier_consolidation compact_view_core full endpoint. It uses existing official-compatible prediction/report files only; no model training, generation, or GPU evaluation was run.

## Overall arithmetic
- clean-Qwen Overall: 41.344291; compact_view_core Overall: 39.987652.
- Compact_view_core if its AoA column were exactly 0: 41.397350; this still does not reach 41.8.
- Component deltas compact minus clean: BLiMP +0.310, Supplement -0.990, EWoK +1.070, Entity +2.090, COMPS +0.400, SuperGLUE -1.407, GlobalPIQA -1.485, Reading +0.490, AoA -12.687
- With compact_view_core's other measured columns fixed, SuperGLUE+AoA would need 72.525 for Overall 41.8; actual compact SuperGLUE+AoA is 56.214.

## Supplement: not a broad syntactic collapse
- Equal-UID Supplement: clean 62.84, compact 61.85, delta -0.99.
  - qa_congruence_easy: clean 70.31, compact 62.50, delta -7.81; lost 8, gained 3, n=64.
  - qa_congruence_tricky: clean 49.09, compact 45.45, delta -3.64; lost 20, gained 14, n=165.
  - turn_taking: clean 64.64, compact 68.21, delta +3.57; lost 9, gained 19, n=280.
  - subject_aux_inversion: clean 80.76, compact 82.49, delta +1.73; lost 226, gained 293, n=3867.
  - hypernym: clean 49.41, compact 50.59, delta +1.19; lost 83, gained 93, n=842.
- Interpretation: compact_core improves subject-auxiliary inversion, turn-taking, and hypernym slices, but loses both QA-congruence slices. The lost QA examples are mainly simple who/what/where answer-type and animacy/object-role discriminations, which are close to child-directed dialogue and event-role evidence; this points away from a generic grammar failure and toward losing some pragmatic/action-role support when official source rows are replaced by FineWeb packets.

## GlobalPIQA: few flips against clean, but still far below the visible leader
- GlobalPIQA_parallel: clean 25.24 (26/103), compact 24.27 (25/103), delta -0.97; lost 9, gained 8.
  - weakest slices in compact-vs-clean: category=object_properties -50.0, inspiration=glenberg_robertson -33.3, inspiration=english_piqa -14.3, category=spatial -11.5.
- GlobalPIQA_nonparallel: clean 48.00 (48/100), compact 46.00 (46/100), delta -2.00; lost 11, gained 9.
  - weakest slices in compact-vs-clean: category=(none) -2.0, inspiration=(none) -2.0.
- Interpretation: compact_core is not catastrophically worse than clean-Qwen on GlobalPIQA; it loses only one net parallel example and two net nonparallel examples. But both endpoints are weak on practical physical/time/counting questions relative to the target surface, so small packet swaps are unlikely to solve SOTA unless they also preserve the other columns.

## SuperGLUE: loss localizes to entailment/paraphrase classification
- rte: clean 69.784, compact 64.029, delta -5.755; lost 23, gained 15, n=139.
- mrpc: clean 86.275, compact 81.863, delta -4.412; lost 16, gained 7, n=204.
- boolq: clean 68.379, compact 67.829, delta -0.550; lost 171, gained 162, n=1635.
- wsc: clean 61.538, compact 61.538, delta +0.000; lost 8, gained 8, n=52.
- multirc: clean 68.276, compact 68.358, delta +0.083; lost 184, gained 186, n=2424.
- mnli: clean 60.065, compact 60.452, delta +0.387; lost 520, gained 539, n=4908.
- qqp: clean 77.843, compact 78.239, delta +0.396; lost 1510, gained 1590, n=20215.
- Interpretation: the SuperGLUE mean drop is dominated by RTE and MRPC, while MNLI/QQP/MultiRC are flat-to-slightly positive and WSC unchanged. This again argues against a universal finetuning degradation; the current FineWeb compact overlay seems to damage a narrow equivalence/entailment calibration that clean-Qwen previously retained.

## Route implication
- The compact core full eval projection thresholds failure should not be reduced to AoA. The complete losses are structured: QA answer-type/action-role support, practical commonsense still weak, and RTE/MRPC sentence-pair calibration.
- Because the overlay replaced 423,520 official clean-Qwen words (including CHILDES/OpenSubtitles/Gutenberg/SimpleWiki slices) with only 353,945 source+compact FineWeb pair words plus 69,575 neutral top-up words in compact_core, a plausible failure mechanism is source-distribution displacement rather than the compact-view mechanism alone.
- A higher-value redesign, if the pending AoA/reinvest evidence permits, should preserve the clean-Qwen official source distribution much more tightly and move information-density inside the existing Qwen-pair budget: compact or mix-resolution the near-length Qwen second views, then reinvest saved words into additional official/practical/event-rich packets. This would test the density principle without paying the observed cost of replacing a broad official slice.

Machine-readable JSON: `experiments/archive/frontier_consolidation/data/compact_core_full_loss_anatomy/compact_core_full_loss_anatomy.json`
