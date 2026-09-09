# lamb update allocation closure — LAMB update-allocation screen and closure

## Why this was the right bounded test

After Muon hidden-update geometry produced a large 20M gain but reversed at mature exposure, targeted masking closure and lamb opening opened LAMB as a different whole-learning-system optimizer mechanism. The scientific hypothesis was not "try the 41.8 leader's optimizer" as a recipe race. It was: preserve Adam-like per-tensor update directions while reallocating update magnitude by layer/tensor trust ratios, potentially handling heterogeneous DeBERTa-v2 parameter scales without Muon's hidden-subspace rotation.

An 80M commitment requires both broad cheap-column movement and measured trust-ratio/layer-update behavior supporting the intended LAMB mechanism. The following low-cost comparisons test those conditions:

1. complete/partial 20M-scale real training already launched in targeted masking closure and lamb opening,
2. cheap official-compatible columns only, no SuperGLUE/AoA,
3. a fixed-batch update-allocation probe measuring trust ratios, local Adam-direction preservation, effective step multipliers, and hidden spectra.

## Training outcomes

Completed and partial LAMB controls:

- LAMB lr=0.007 completed 20M cleanly in `training/runs/lamb_lr007_seed43022_20M/`. It consumed exactly 20,000,000 words with exact first loss 9.837543, but final loss was 6.8976 and tail20 mean loss 6.8812.
- LAMB lr=0.005 timed out at the 900s runtime cap after saving checkpoints through `chck_14M` in `training/runs/lamb_lr005_seed43022_20M/`. It did not scientifically diverge, but at the last logged earlier analysis / 13.525M words its loss was 4.7883; spatial repair route status nearest loss was 4.0233.

CPU dynamics summary: `data/lamb_training_dynamics/lamb_training_dynamics.md`.

Key matched-loss readout:

| run | last words | loss | nearest spatial repair route status loss | Δloss |
|---|---:|---:|---:|---:|
| LAMB lr=0.005 partial | 13.525M | 4.7883 | 4.0233 | +0.7649 |
| LAMB lr=0.007 complete | 20.000M | 6.8976 | 3.7556 | +3.1420 |
| Muon wd-matched 20M | 20.000M | 3.5528 | 3.7556 | -0.2028 |

## Cheap-column results

Because the lr=0.005 run timed out before trainer cleanup, it lacked `scientific_metrics.json`. I did not edit the original run. I created a truthful shadow directory solely for evaluation:

- `training/runs/lamb_lr005_seed43022_partial14M_shadow/`
- `hf_model/chck_14M` symlinks to the original saved checkpoint.
- `scientific_metrics.json` explicitly records `partial_shadow_for_evaluation: true`, a timed-out outcome, and the last logged training state.

Evaluation helpers:

- `scripts/eval_lamb_arm_20m.py`
- `scripts/merge_lamb_20m_eval.py`

Merged result: `data/lamb_20M_eval_merged/lamb_20M_eval_merged.md`.

Exact spatial repair route status 20M baseline is from `data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json`; Muon matched-decay comparison is from `data/muon_wdmatched_20M_eval/muon_wdmatched_20M_comparison.json`.

| arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δcheap7 vs spatial repair route status 20M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial repair route status legal AdamW 20M | 59.69 | 55.45 | 50.73 | 18.65 | 50.26 | 34.195 | 8.67 | 39.6636 | +0.0000 |
| Muon wd-matched 20M | 61.15 | 58.10 | 50.70 | 20.90 | 50.45 | 38.605 | 7.26 | 41.0236 | +1.3600 |
| LAMB lr=0.005, chck_14M partial-shadow | 53.13 | 49.10 | 49.71 | 17.36 | 49.72 | 34.725 | 7.33 | 37.2964 | -2.3671 |
| LAMB lr=0.007, chck_20M | 53.74 | 49.95 | 48.94 | 17.14 | 50.00 | 33.25 | 7.14 | 37.1657 | -2.4979 |

This is not a narrow relation/entity redistribution like Muon. It is broad underlearning or destructive update allocation: BLiMP/Supplement fall by 5.5-6.6, Entity falls by 1.3-1.5, EWoK falls by 1.0-1.8, and Reading also falls.

## Update-allocation mechanism readout

Probe script: `scripts/lamb_update_allocation_probe.py`.
Output: `data/lamb_update_allocation_probe/lamb_update_allocation_probe.md`.

The probe reconstructs the first spatial repair route status WWM batch and computes a local first-step gradient readout. It cannot reconstruct historical optimizer moment buffers from HF checkpoints, so it measures the intended update-allocation mechanism rather than exact stored LAMB states.

Key results:

| model | fixed-batch loss | trust p50 | clipped tensors | full-vector cos vs uniform AdamW | LAMB/AdamW full step norm | multiplier p50 | core hidden stable rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact init | 9.8220 | 1.0000 | 0 | 0.4433 | 0.28 | 5.00 | 152.97 |
| spatial repair route status 20M | 3.6980 | 0.0251 | 0 | 0.8192 | 0.19 | 0.13 | 43.96 |
| LAMB lr=0.005 chck_14M | 4.6839 | 0.0211 | 0 | 0.7448 | 0.17 | 0.11 | 61.18 |
| LAMB lr=0.007 chck_20M | 6.8260 | 10.0000 | 130 | 0.7943 | 0.26 | 70.00 | 7.20 |

Interpretation:

- lr=0.007 is closed both behaviorally and mechanistically. It hit the trust-ratio clamp in 130/170 tensors, applied many 70x nominal tensor multipliers relative to spatial repair route status AdamW, nearly froze the MLM head in the local probe, and collapsed core hidden stable rank to 7.20. This is not the intended gentle Adam-trajectory-preserving allocation.
- lr=0.005 is mechanistically cleaner than lr=0.007: no trust-ratio clipping and moderately broader hidden spectra (stable rank 61.18 vs spatial repair route status 43.96). But it is still behaviorally too weak at `chck_14M` and undertrained by loss. Its available checkpoint loses six of seven cheap columns and cheap7 by -2.367 versus spatial repair route status 20M. There is no basis to spend a second full 20M completion run, let alone 80M.

## Scientific conclusion

The current LAMB implementation/parameterization is closed as a SOTA route. The negative result is scientifically useful because it separates update-allocation mechanisms:

- Muon strongly improved early cheap7 but rotated the mature trajectory away from order-sensitive entity/relation competence.
- LAMB at lr=0.007 destroys optimization behavior through trust-ratio saturation and hidden-spectrum collapse.
- LAMB at lr=0.005 avoids clamp collapse but still underlearns and is far below spatial repair route status on early official-compatible behavior.

This evidence does **not** support automatically launching a generic 0.25→0.15 masking schedule. That would return to a weakened schedule/masking family without a new mechanism. The next work should integrate latest evidence and refine the whole-system mechanism comparison before the next major GPU investment.

Best fully legal complete endpoint remains spatial repair route status compact-view-reinvest, Overall 41.2578. The active goal, legal BabyLM Strict-Small Overall SOTA, remains unmet.
