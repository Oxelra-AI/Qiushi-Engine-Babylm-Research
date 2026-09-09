# Loss trajectory and relation-margin phase dynamics during corrected-tokenizer retraining

This note preserves CPU-only evidence produced while the two H100 corrected-tokenizer retrains from earlier analysis continue. It does **not** replace the required 100M completion and pristine official evaluation.

## Active corrected-tokenizer retrains

At the time of this analysis, the corrected-tokenizer retrains were still running. Both were around 55M word exposure:

- seed43022: earlier analysis, 55,163,230 cumulative words, loss 2.7433, checkpoints through `chck_56M` at that moment.
- seed43122: earlier analysis, 55,163,230 cumulative words, loss 2.7710, checkpoints through `chck_55M` at that moment.

These are training-progress signals only. Endpoint competence remains unresolved until `chck_100M` exists and the prepared official-coordinate evaluation controller is run.

## 1. MLM training loss is not a reliable seed selector

Script: `experiments/archive/representation_and_objectives/scripts/loss_trajectory_vs_overall.py`
Output: `experiments/archive/representation_and_objectives/data/loss_trajectory_vs_overall/loss_trajectory_vs_overall.json`

Known old inherited-tokenizer reinvest endpoints:

- old-tokenizer seed43022: Overall 42.0331347900748
- old-tokenizer seed43122: Overall 41.24823958912208
- downstream delta seed43022 minus seed43122: +0.784895 Overall

Loss trajectory comparison on the same old-tokenizer pair:

- all-step mean loss delta, seed43022 minus seed43122: +0.003586
- last-50-step mean loss delta, seed43022 minus seed43122: +0.000543
- 10M window deltas are tiny and sign-flipping.

Thus the seed that wins downstream by nearly 0.785 Overall does **not** have lower MLM training loss. Training loss cannot be used as a seed-selection rule for this route. If seed selection becomes necessary, it must use official-compatible downstream probes, not broad MLM loss.

The in-progress corrected-tokenizer loss curves likewise do not yet justify a seed preference: through ~58.8M words, seed43022 minus seed43122 window loss deltas are small and changing sign (`50-60M`: -0.0415). These loss values are useful for run health, not endpoint prediction.

## 2. Corrected-tokenizer `chck_50M` focused EWoK relation probe

Script: `experiments/archive/representation_and_objectives/scripts/strictsmalltok_midtrain_ewok_margin_probe.py`
Output JSON: `experiments/archive/representation_and_objectives/data/strictsmalltok_midtrain_ewok_margin_probe/strictsmalltok_chck_50M_focus553_margins.json`
Output CSV: `experiments/archive/representation_and_objectives/data/strictsmalltok_midtrain_ewok_margin_probe/strictsmalltok_chck_50M_focus553_pair_rows.csv`

This reused the ewok tokenizer eval interface audit official-compatible EWoK MLM margin scorer on the same 553-row focused subset enriched for old inherited-tokenizer relation instability. It used already-saved corrected-tokenizer `chck_50M` checkpoints and CPU only.

Focused subset result at `chck_50M`:

- strict-small-tokenizer seed43022: 40.1447% micro accuracy on the focused 553 rows; selected-domain macro 42.9972.
- strict-small-tokenizer seed43122: 56.6004% micro accuracy; selected-domain macro 55.8399.
- seed margin Pearson correlation: 0.6202.
- opposite-sign fraction: 0.3599.
- seed43122 minus seed43022 margin delta mean: +0.8545.

Compared to old inherited-tokenizer endpoint margins from ewok tokenizer eval interface audit:

- correlation strict50 seed43022 with old100 seed43022: 0.6309
- correlation strict50 seed43122 with old100 seed43122: 0.4092
- correlation strict50 seed-delta with old100 seed-delta: -0.1677
- sign agreement with corresponding old endpoint margins: about 0.56 for each seed

Interpretation: the compliant tokenizer coordinate has already changed early focused EWoK relation dynamics. On the old-instability rows, the corrected-tokenizer 50M direction is reversed relative to the old-tokenizer endpoint: seed43122 is much stronger than seed43022 on these rows. This is not a leaderboard score and cannot predict final Overall, but it prevents importing the old seed43022 advantage as a belief about the compliant coordinate.

## 3. Old-tokenizer intermediate checkpoints: relation polarization becomes predictive late

Script: `experiments/archive/representation_and_objectives/scripts/oldtok_checkpoint_ewok_phase_probe.py`
Output JSON: `experiments/archive/representation_and_objectives/data/oldtok_checkpoint_ewok_phase_probe/oldtok_checkpoint_focus553_phase_probe.json`
Output CSV: `experiments/archive/representation_and_objectives/data/oldtok_checkpoint_ewok_phase_probe/oldtok_checkpoint_focus553_pair_rows.csv`

The same 553-row focused EWoK probe was run on old inherited-tokenizer reinvest checkpoints at 50M and 80M for both seeds.

Old-tokenizer focused subset:

| Coordinate | seed43022 | seed43122 | seed43022 advantage |
| --- | ---: | ---: | ---: |
| oldtok `chck_50M` | 50.0904 | 43.9421 | +6.1483 |
| oldtok `chck_80M` | 60.7595 | 34.3580 | +26.4014 |
| oldtok `chck_100M` endpoint (ewok tokenizer eval interface audit) | 66.1844 | 30.5606 | +35.6239 |

Delta-structure comparisons:

- oldtok 50M -> oldtok 100M: seed-delta correlation 0.4273, sign agreement 0.6239.
- oldtok 80M -> oldtok 100M: seed-delta correlation 0.9385, sign agreement 0.9241.
- oldtok 50M -> oldtok 80M: seed-delta correlation 0.4406, sign agreement 0.6094.

Interpretation: in the old coordinate, focused relation-margin polarization was only weakly shaped at 50M but had largely formed by 80M. Therefore a 50M focused EWoK probe is not by itself a reliable endpoint seed selector; an 80M-style focused probe is more informative in the old coordinate. Whether the corrected-tokenizer coordinate follows the same late-shaping behavior must be measured from its own checkpoints, not inferred from old-tokenizer runs.

## Consequence for the active SOTA route

1. The two corrected-tokenizer 100M retrains remain decisive. Do not stop them, package inherited-tokenizer artifacts, or launch a new expensive route before their official-coordinate evaluations exist.
2. The 42.033 inherited-tokenizer endpoint remains mechanism evidence but cannot be submitted because its tokenizer was trained from out-of-budget Strict-100M data.
3. MLM loss cannot pick the winning seed; broad loss differences are tiny and in the wrong direction for the known old-tokenizer winner.
4. If corrected-tokenizer full endpoints are seed-sensitive, seed choice/stability work should use official-compatible downstream probes. For EWoK relation dynamics, a focused 80M probe is much more informative than 50M in the old coordinate.
5. The corrected-tokenizer `chck_50M` result is encouraging for seed43122 relation rows but not an endpoint prediction. It shows only that the compliant representation changed early relation geometry and that old seed43022 superiority should not be assumed.

## 4. Corrected-tokenizer 70M and 80M focused EWoK probes

After the initial 50M probe, `chck_70M` and then `chck_80M` became available for both corrected-tokenizer retrains. I ran the same CPU-only official-compatible focused EWoK margin probe on those saved checkpoints.

Additional outputs:

- `experiments/archive/representation_and_objectives/data/strictsmalltok_midtrain_ewok_margin_probe/strictsmalltok_chck_70M_focus553_margins.json`
- `experiments/archive/representation_and_objectives/data/strictsmalltok_midtrain_ewok_margin_probe/strictsmalltok_chck_80M_focus553_margins.json`

Focused 553-row trajectory under the compliant tokenizer:

| corrected-tokenizer checkpoint | seed43022 | seed43122 | seed43022 advantage |
| --- | ---: | ---: | ---: |
| 50M | 40.1447 | 56.6004 | -16.4557 |
| 70M | 51.5371 | 45.3888 | +6.1483 |
| 80M | 51.8987 | 46.4738 | +5.4250 |

For comparison, old inherited-tokenizer trajectory on the same rows:

| old-tokenizer checkpoint | seed43022 | seed43122 | seed43022 advantage |
| --- | ---: | ---: | ---: |
| 50M | 50.0904 | 43.9421 | +6.1483 |
| 80M | 60.7595 | 34.3580 | +26.4014 |
| 100M endpoint | 66.1844 | 30.5606 | +35.6239 |

Corrected 80M seed-delta versus old 100M endpoint seed-delta:

- correlation: -0.0837
- sign agreement with old endpoint margins: seed43022 0.6329, seed43122 0.6600
- corrected 80M same-row opposite-sign fraction between seeds: 0.3797

Interpretation update: the corrected-tokenizer coordinate does not reproduce the old inherited-tokenizer late relation-polarization profile. At 50M it was reversed; by 70M/80M seed43022 recovers a modest advantage, but the profile remains far less polarized than the old 80M/100M profile and the seed-delta correlation with the old endpoint is near zero/slightly negative. This is not enough to predict Overall, but it strengthens the main rule: the compliant tokenizer has changed relation-learning dynamics, so final SOTA judgment must come from the corrected 100M official evaluation, not from the inherited-tokenizer seed history.
