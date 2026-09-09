# compact core full eval projection thresholds compact-core full-evaluation result and route decision

## Full official-compatible result (decisive)

`compact_view_core` (medium risk-hard clean-Qwen row-holdout overlay, DeBERTa-v2 8x480, baseline16k, fixed seq256, WWM, seed43/43022/43023, 100M exposure), full official-compatible evaluation, endpoint chck_100M:

| column | compact_view_core (full) | compact_experience clean-Qwen (full) | delta |
|---|---:|---:|---:|
| BLiMP | 67.15 | 66.84 | +0.31 |
| Supplement | 61.85 | 62.84 | -0.99 |
| EWoK | 51.26 | 50.19 | +1.07 |
| Entity | 27.85 | 25.76 | +2.09 |
| COMPS | 52.18 | 51.78 | +0.40 |
| GlobalPIQA | 35.135 | 36.62 | -1.485 |
| SuperGLUE | 68.9012 | 70.3086 | -1.407 |
| Reading | 8.25 | 7.76 | +0.49 |
| AoA | -12.687 | 0.0 | -12.687 |
| **Overall** | **39.9877** | **41.3443** | **-1.3566** |
| NLP_average | 52.0466 | 52.0484 | -0.0018 |
| Human_like_average | -2.2186 | 3.88 | -6.099 |

Evidence: `experiments/archive/frontier_consolidation/data/density_full_eval/per_target/compact_view_core.json`; summary `experiments/archive/frontier_consolidation/data/density_full_eval/density_full_eval_summary.json`.

SuperGLUE per-task: boolq 67.83, multirc 68.36, rte 64.03, wsc 61.54, mrpc 81.86, qqp 78.24, mnli 60.45; mean 68.9012. AoA raw correlation -0.126873, leaderboard -12.687, official_aoa_done, submit_ready.

## Scientific reading

1. The density-view mechanism is real on the NLP surface: compact same-source semantic views produce EWoK +1.07, Entity +2.09, COMPS +0.40, BLiMP +0.31, Reading +0.49 over the inherited clean-Qwen anchor, while NLP_average is essentially tied (-0.0018). The density eval repair fast screen's +2.13 equal7 was mostly Supplement fast overshoot; full Supplement is -0.99, so the fast Supplement value (65.6) did not survive the full split (61.85).

2. The candidate fails as SOTA because of AoA. clean-Qwen has AoA 0.0 (non-significant correlation mapped to 0), but compact_view_core has a *significant negative* fitted model-AoA/child-AoA correlation (-0.1269 -> -12.687 leaderboard). This single Human-like column removes ~1.41 Overall on its own and sinks the artifact to 39.9877.

3. This is the same structural weakness seen historically: several non-clean-Qwen controls (official_lengthmatched -15.70, shuffled -12.79, original_dup -13.32) also produce large negative AoA, while clean-Qwen aligned pairs uniquely land at 0.0. The compact FineWeb view overlay, despite preserving all qwen_pair_packed rows, pushed the acquisition-timing correlation significantly negative again.

## Timely-termination judgment

Decision criterion: stop a route once results show it cannot beat the current best full acceptance even after completing unmeasured parts. compact_view_core's unmeasured parts are now measured; SuperGLUE and AoA are both worse than clean-Qwen, and AoA is significantly negative. As-is, compact_view_core is 1.36 below the inherited best. The reinvest extension shares the identical compact transform and neutral overlay structure, so it is expected to inherit the same negative-AoA mechanism; a full reinvest evaluation is unlikely to beat 41.3443 through the AoA column, though a low-cost check remains justified because its NLP surface is slightly higher.

**Do not** continue by simply swapping doses, renaming arms, or adding more compact sources on the same negative-AoA substrate.

## The real next scientific question

The mechanism improved the NLP surface at tied NLP_average but broke acquisition timing. The generalizable-learning question is now: **why does redundancy-reduced same-source view injection turn the AoA correlation significantly negative, and can the density benefit be kept while restoring AoA to at least clean-Qwen's neutral 0.0?**

Concrete, low-cost next actions before more 100M training:
1. Passively measure the AoA trajectory mechanism on the existing compact_view_core checkpoint ladder that already ran: which words/bands drive the negative correlation, and how compact-view exposure changed early-vs-late acquisition order relative to clean-Qwen. This is CPU/GPU-light because checkpoints and the AoA surprisal outputs already exist under the full-eval AoA outputs.
2. Compare compact_view_core AoA word-curve fits against clean-Qwen's to see whether the compact transform accelerated abstract/late words or delayed concrete/early words (the historical failure mode).
3. Only if a legal, benchmark-independent adjustment plausibly restores AoA without erasing the NLP gain should another 100M training arm be spent.

The compact-view density principle remains a genuine NLP-surface finding worth reporting, but it is not yet a SOTA candidate. The route continues by understanding and repairing the AoA perturbation, not by abandoning the density mechanism and not by minor dosing on the same substrate.
