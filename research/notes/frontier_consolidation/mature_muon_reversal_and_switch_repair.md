# mature muon reversal and switch repair — mature matched-decay Muon reversal and switch repair

## Prior Scientific Position

The active line was hidden-matrix update geometry on the protected legal compact-view-reinvestment substrate. earlier analysis showed a large 20M gain with Muon on 48 attention/FFN matrices, but bundled the gain with stronger hidden-matrix decoupled shrinkage. muon wdmatched causal plan and pvdm challenge matched the Muon shrinkage to spatial repair route status and preserved/improved the 20M signal:

- matched-decay Muon 20M cheap7: 41.0236 vs spatial repair route status 20M 39.6636, Δ +1.3600
- columns: BLiMP +1.46, Supplement +2.65, EWoK -0.03, Entity +2.25, COMPS +0.19, GlobalPIQA +4.41, Reading -1.41
- hidden attention/FFN spectra remained broad: stable-rank mean 121.29, top8 energy 0.0607

This justified a bounded mature readout but did not justify a 100M endpoint before 70/80M evaluation.

## Related Experimental Evidence

Dense PVDM and sparse relation-auxiliary designs were closed or deferred because their local relation signal was largely visible-context lookup or target redistribution. Muon is a distinct hidden-update-geometry comparison and must be interpreted separately. After the mature 70M/80M result, unchanged continuation to 100M was stopped.

## Mature evaluation result

Files:

- evaluation summary: `data/muon_wdmatched_mature_eval/muon_wdmatched_mature_comparison.{json,md}`
- 70M target JSON: `data/muon_wdmatched_mature_eval/per_target/muon_lr008_wd00125_70M.json`
- 80M target JSON: `data/muon_wdmatched_mature_eval/per_target/muon_lr008_wd00125_80M.json`
- mechanism summary: `data/muon_mature_mechanism/muon_mature_mechanism_summary.{json,md}`
- report anatomy: `data/continuous_muon_anatomy/continuous_muon_report_anatomy.{json,md}`

### 70M

| column | spatial repair route status | matched-decay Muon | Δ |
|---|---:|---:|---:|
| BLiMP | 65.39 | 65.30 | -0.09 |
| Supplement | 59.31 | 59.97 | +0.66 |
| EWoK | 50.47 | 51.38 | +0.91 |
| Entity | 26.98 | 25.60 | -1.38 |
| COMPS | 51.82 | 51.64 | -0.18 |
| GlobalPIQA | 35.55 | 38.105 | +2.555 |
| Reading | 8.74 | 6.725 | -2.015 |
| cheap7 | 42.6086 | 42.6743 | +0.0657 |

### 80M

| column | spatial repair route status | matched-decay Muon | Δ |
|---|---:|---:|---:|
| BLiMP | 66.11 | 66.35 | +0.24 |
| Supplement | 60.66 | 60.43 | -0.23 |
| EWoK | 51.01 | 50.72 | -0.29 |
| Entity | 27.06 | 24.17 | -2.89 |
| COMPS | 51.93 | 51.52 | -0.41 |
| GlobalPIQA | 35.58 | 39.065 | +3.485 |
| Reading | 8.29 | 7.26 | -1.03 |
| cheap7 | 42.9486 | 42.7879 | -0.1607 |

This closes the unchanged continuous matched-decay Muon route as a 100M endpoint candidate. The 20M gain mostly became an early acceleration/redistribution rather than a broad mature advantage.

## Mechanism interpretation

The intended geometry did **not** collapse at mature exposure. In `muon_mature_mechanism_summary.json`, matched-decay Muon remained far broader than spatial repair route status at matched checkpoints:

- 20M: stable-rank mean Δ +77.33, entropy-rank Δ +50.83, top8-energy Δ -0.0726, Frobenius-norm Δ +4.61
- 70M: stable-rank mean Δ +65.64, entropy-rank Δ +75.28, top8-energy Δ -0.0748, Frobenius-norm Δ +7.53
- 80M: stable-rank mean Δ +65.19, entropy-rank Δ +75.34, top8-energy Δ -0.0745, Frobenius-norm Δ +7.51

Muon also had lower train loss at 80M (2.3978 vs spatial repair route status 2.5526). Thus the failure is not numerical instability, insufficient training, or loss collapse. It is a mature competence redistribution caused by keeping orthogonalized hidden updates active late in training.

Report-level anatomy at 80M:

- Entity losses are strongest on long operation chains: regular_5_ops -17.02, ambiref_5_ops -8.94, move_contents_4_ops -5.95, move_contents_5_ops -4.31.
- EWoK losses are relation/transform sensitive: active-passive -16.66, variable_swap -6.66, spatial-relations -6.33, social-interactions -5.44, physical-relations -3.55.
- Reading drops both eye tracking (-1.31) and self-paced (-0.75).
- GlobalPIQA improves, especially through item flips on physical plausibility.

Interpretation: Muon broadens matrix spectra and improves broad plausibility/low-depth surfaces, but continuous late Muon appears to keep rotating hidden subspaces instead of consolidating stable entity-state and order-sensitive relation credit. Lower MLM loss does not mean better BabyLM behavior.

## Repair tested

The repair follows directly from the mature reversal: use Muon only as an early representation-broadening phase, then switch hidden matrices back to AdamW for late consolidation. This is not an unrelated route. It tests whether the useful early orthogonalized updates can be retained while avoiding late relation/tracking damage.

Implemented script:

- `scripts/muon_to_adamw_switch_trainer.py`

Control properties:

- exact spatial repair route status legal compact-view corpus and tokenizer
- exact DeBERTa-v2 8x480 architecture, WWM, seeds, batch, sequence length, scheduler, and checkpoint structure
- 48 attention/FFN matrices use Muon before the switch; all other tensors use AdamW throughout
- after switch, the same hidden matrices use AdamW at the spatial repair route status scheduled LR/decay
- scheduler sees base AdamW LR for hidden tensors; during Muon mode active hidden LR is internally scaled by 8 to match the muon wdmatched causal plan and pvdm challenge Muon LR0.008
- Muon weight decay remains 0.00125 so active Muon lr * wd = 1e-05, equal to spatial repair route status base AdamW shrinkage

Smoke test:

- command produced first loss 9.837543487548828, exact 48/122 grouping, switch log at optimizer steps 1 and 6 for a 5-step smoke, and a loadable checkpoint at `training/runs/muon_switch_smoke_1M/hf_model/chck_1M`.
- grouping report: `data/muon_switch_smoke/grouping_report.json`

Launched bounded 80M screens:

- `Muon20M-to-AdamW 80M screen GPU0`, run dir `training/runs/muon20toadamw_seed43022_80M`, switch_after_steps=506.
- `Muon40M-to-AdamW 80M screen GPU1`, run dir `training/runs/muon40toadamw_seed43022_80M`, switch_after_steps=1012.

These are admitted because continuous Muon had a strong early gain but a mature reversal, and the switch variants distinguish whether late continuous orthogonalization is the damaging factor. They should be evaluated only at mature 70M/80M cheap columns first. A 100M/full official endpoint is warranted only if a switch arm is broadly above spatial repair route status at 80M and especially if Entity/Reading recover without destroying the GlobalPIQA/Supplement/EWoK gains.

Prepared evaluator:

- `scripts/eval_muon_switch_mature.py`

It can run nonconflicting part-target evaluations by column group and merge them against spatial repair route status and continuous-Muon references. Suggested parallel use after checkpoints exist:

```bash
python -B experiments/archive/frontier_consolidation/scripts/eval_muon_switch_mature.py --run --arm muon20 --ckpt 80M --gpu 0 --columns BLiMP Supplement EWoK Entity
python -B experiments/archive/frontier_consolidation/scripts/eval_muon_switch_mature.py --run --arm muon20 --ckpt 80M --gpu 1 --columns COMPS GlobalPIQA_parallel GlobalPIQA_nonparallel Reading
python -B experiments/archive/frontier_consolidation/scripts/eval_muon_switch_mature.py --summarize
```

For four arm/checkpoint combinations, split by arm/checkpoint/column group across available GPUs without writing the same part target concurrently.

## Current judgment

The best fully legal complete endpoint remains spatial repair route status Overall 41.2578. Continuous matched-decay Muon is closed as an endpoint route. Update geometry is not closed globally: the observed pattern gives a specific repair hypothesis with a direct test already running. If the switch arms fail to recover Entity/Reading and cheap7 remains below spatial repair route status, this update-geometry family should not be prolonged by more switch times or naming changes without a new mechanism explaining the structured high-operation/entity-state damage.


---

## mature muon reversal and switch repair switch-screen result and route decision (appended)

The two bounded 80M Muon→AdamW switch trainings completed cleanly (both exact 80M words, loss_last 2.4305 and 2.4210, checkpoints chck_10M..chck_80M). Official-compatible cheap-column merge is in `data/muon_switch_actual_merged/actual_switch_eval_merged.{json,md}`.

### Mature cheap7 (vs spatial repair route status 42.9486 at 80M, 42.6086 at 70M)

| ckpt | arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δ spatial repair route status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70M | Muon20→AdamW | 66.04 | 61.58 | 51.39 | 22.94 | 51.95 | 37.62 | 7.970 | 42.7843 | +0.1757 |
| 80M | Muon20→AdamW | 66.38 | 61.98 | 50.08 | 23.59 | 52.23 | 37.605 | 7.770 | 42.8050 | -0.1436 |
| 70M | Muon40→AdamW | 65.59 | 61.94 | 50.46 | 23.98 | 51.62 | 35.15 | 7.505 | 42.3207 | -0.2879 |
| 80M | Muon40→AdamW | 65.98 | 61.68 | 50.22 | 25.22 | 51.66 | 34.135 | 7.570 | 42.3521 | -0.5964 |

No switch arm exceeds spatial repair route status at 80M. The Muon20→AdamW arm's mild 70M advantage (+0.1757) still collapses to -0.1436 at 80M, mirroring the continuous-Muon reversal. The most damaging surface is Entity (−3.47 at 80M for Muon20→AdamW, −1.84 for Muon40→AdamW), the same high-operation entity-state credit that continuous Muon damaged. Supplement is durably positive (+1.0 to +1.3) but cannot offset Entity/EWoK/Reading.

### Geometry: breadth vs behavior are decoupled

`data/switch_geometry/switch_geometry_readout.md`:

- After the AdamW phase, spectra reconcentrated toward spatial repair route status. Switch20_80M stable-rank Δ/init = -90.36 (spatial repair route status -110.17, continuous Muon -44.98); switch40_80M -66.33. So late AdamW largely undoes Muon's static broadening.
- Yet both switch endpoints are far from spatial repair route status in weight space: reference_80M→switch20_80M flat cosine 0.361, →switch40_80M 0.304; leading-subspace angles 72–77°.
- Continuous Muon→switch40_80M flat cosine 0.915 (angle ~34°): the Muon40 arm mostly stays on the continuous-Muon trajectory even after switching, because 40M of Muon already fixed the subspace.

### Scientific conclusion (update-geometry family)

The mature reversal is not explained by static spectral breadth (continuous Muon keeps it, switch arms lose it, both fail Entity) nor by the late optimizer alone (AdamW after switch does not recover Entity/relation surfaces). The structured entity-state and relation-credit damage is tied to the **early orthogonalized-update trajectory itself**: once early Muon has rotated hidden subspaces onto a different manifold (subspace angle to spatial repair route status 72–77°), later AdamW cannot restore the order-sensitive tracking credit that spatial repair route status develops.

This closes the Muon hidden-update orthogonalization family — continuous and early-only schedules — as a route to a legal Overall SOTA on the current 8×480 WWM compact-view system. The durable, transferable finding is a mechanism: on this small DeBERTa MLM, calibrated hidden-matrix orthogonalization reliably lifts BLiMP/Supplement/plausibility (GlobalPIQA) early but rotates competence away from multi-step entity tracking and several relation contrasts, and this competence tradeoff is trajectory-locked rather than spectral or late-optimizer-driven.

Do not spend further GPU on more Muon switch times, doses, or matrix subsets without a genuinely different mechanism that explains and repairs the high-operation entity-state damage. The best fully legal complete endpoint remains spatial repair route status Overall 41.2578.

### Next research direction (evidence-derived, not a guess)

Continuous Muon and both switch arms share a signature: Supplement improves while high-operation Entity, spatial/physical/social relations and active-passive performance decline. This suggests a bottleneck in order-sensitive relational/entity-state credit that compact-view allocation, masking variants, RTD and update geometry have not resolved. Independent target-geometry tests found visible-context lookup rather than conditional prediction. A proposed next comparison is masked-given-pivot relational prediction requiring multi-step state composition, calibrated against that lookup confound on the protected legal compact-view substrate.
