# probe results interpretation integrated probe interpretation

This note integrates the probe results interpretation held-out copy/rewrite probes and Entity cue ablations with the copy and relevant update result official Entity relevant-update split. All values below are late means over 80M/90M/100M checkpoints for the two existing seeds; no seed43222 result has been read here.

## Arm-level late orderings

| seed | held-out copy gain | held-out rewrite gain | rewrite nonoverlap gain | Entity rel0 official acc | Entity rel>=2 official acc | stale-item full margin |
|---:|---|---|---|---|---|---|
| 43022 | R(4.186) > V(4.014) > C(3.687) | V(4.554) > C(3.942) > R(3.449) | V(1.896) > C(1.212) > R(0.460) | R(48.237) > C(39.455) > V(38.546) | V(27.460) > C(22.608) > R(20.106) | V(0.911) > C(0.388) > R(0.087) |
| 43122 | R(4.289) > V(3.931) > C(3.621) | V(4.777) > C(4.010) > R(3.374) | V(2.166) > C(1.272) > R(0.229) | R(48.388) > C(39.303) > V(39.239) | V(26.277) > C(22.063) > R(20.608) | V(0.862) > C(0.532) > R(-0.195) |

## Cue-ablation arm effects on stale-non-gold update items

`no_initial` and `no_last_update` are changes in gold-over-stale margin after removing the queried-box initial clause or the last relevant update sentence. Positive no_initial means removing the stale initial clause helps; negative no_last_update means removing the update hurts.

| seed | role | full margin | no_initial effect | no_last_update effect | no_all_updates effect | rel>=3 full margin | rel>=3 no_initial | rel>=3 no_last |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 43022 | V | +0.911 | -0.391 | -0.783 | -1.556 | +1.277 | -0.381 | -0.787 |
| 43022 | C | +0.388 | +0.199 | -0.510 | -1.704 | +0.597 | +0.102 | -0.501 |
| 43022 | R | +0.087 | +0.231 | -0.511 | -2.488 | +0.273 | +0.051 | -0.359 |
| 43122 | V | +0.862 | -0.117 | -0.709 | -1.512 | +1.179 | -0.128 | -0.705 |
| 43122 | C | +0.532 | +0.090 | -0.552 | -2.034 | +0.791 | -0.011 | -0.512 |
| 43122 | R | -0.195 | +0.498 | -0.417 | -3.157 | +0.087 | +0.311 | -0.239 |

## Scientific interpretation

- The held-out natural-copy probe closes the main copy-side ambiguity from copy and relevant update result: REPEAT has the largest copy gain on text rows not used to train any arm. VIEW also exceeds CLEAN, so the most accurate statement is not that exact repetition alone creates copying, but that both nonbaseline source+companion interventions improve natural context copying and exact copied companions improve it most. This distinction matters for the principle: the copy competence is not merely memorizing the MAX packets, but the trade-off with state updating is strongest for REPEAT.
- The held-out rewrite-conditioning probe supplies the missing positive quantity for VIEW. On unselected compact pairs, VIEW has the largest source-conditioned rewrite-token gain in both seeds, and the gap is largest on non-overlap rewrite tokens where exact token copying cannot explain the benefit. The ordering V > C > R says varied restatement buys content-conditioned use of an earlier span, while exact copied fragments are worst for that quantity under the same budget.
- The Entity cue ablation is mechanistically sharper than an aggregate loss fit. On stale-non-gold update items, VIEW has the largest full-context gold-over-stale margin and REPEAT the smallest. Removing the queried-box initial clause helps REPEAT relative to VIEW, consistent with stale initial over-anchoring in REPEAT; removing the last relevant update hurts VIEW more than REPEAT, consistent with VIEW relying more on update evidence. Removing all relevant updates hurts REPEAT even more, which is not clean evidence for update reading; it leaves the stale clause as the dominant remaining cue and should be read as a vulnerability of REPEAT under contradictory evidence rather than as VIEW being less update-dependent.
- Together with copy and relevant update result, the supported candidate principle has become a pair of checkpoint-measured competences bought by different experience structures under the same finite budget: exact natural recurrence most strengthens copy/use-earlier-span behavior, while nonidentical restatement most strengthens content-conditioned use of an earlier span and improves multi-update state discrimination. The positive VIEW side is now directly measured, but the current evidence remains two-seed until seed43222 checkpoints complete.
