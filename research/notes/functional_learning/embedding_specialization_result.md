# 020 mechanism synthesis result: held-symbol binding and tied embedding drift

This note is generated from `data/embedding_specialization/results.json`. It tests whether full-objective continuation loses held-symbol binding because tied output/input rows for unseen entity tokens drift, or because the contextual selector/readout itself specializes to familiar entities.

## Branch-level final summary

| branch | trained strong | held preserved | final train top4 | final held top4 | restored held top4 | restore Δtop4 | held B | held sel | blocked top4 | held row L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tied_carry_full | 0/1 | 0/1 | 0.333 | 0.146 | 0.146 | +0.000 | -0.005 | +0.001 | 0.333 | 0.006 |
| tied_carry_freeze_held | 0/1 | 0/1 | 0.333 | 0.146 | 0.146 | +0.000 | -0.005 | +0.001 | 0.333 | 0.000 |
| tied_reset_full | 0/1 | 0/1 | 0.281 | 0.188 | 0.188 | +0.000 | -0.006 | +0.002 | 0.281 | 0.014 |

## Per-seed final summary

| seed | branch | prep held4 | final train4 | final held4 | restored held4 | held B | held sel | held row L2 | blocked train4 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | tied_carry_full | 0.125 | 0.333 | 0.146 | 0.146 | -0.005 | +0.001 | 0.006 | 0.333 |
| 42 | tied_carry_freeze_held | 0.125 | 0.333 | 0.146 | 0.146 | -0.005 | +0.001 | 0.000 | 0.333 |
| 42 | tied_reset_full | 0.125 | 0.281 | 0.188 | 0.188 | -0.006 | +0.002 | 0.014 | 0.281 |

## Reading guide

A selective row-drift explanation is supported only if held-row restoration or held-row freezing preserves held-query binding while trained-entity binding remains strong. If freezing or untieing rows does not preserve held behavior, then the loss is more likely in the contextual network or RWT readout specialization. Blocked q→context evaluation tracks whether the prospective marker route remains behaviorally required after continuation.
