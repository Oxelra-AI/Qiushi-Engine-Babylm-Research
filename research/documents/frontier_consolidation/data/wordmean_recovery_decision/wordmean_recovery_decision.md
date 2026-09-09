# earlier analysis word-mean recovery decision

Recovered minimum official-compatible cheap-column evaluation for the completed 70M/80M word-mean checkpoints; no SuperGLUE or AoA; not a complete endpoint.

Recovered because: The managed wordmean screen and substrate constraints evaluator waited indefinitely for 60000 MiB free on GPU1 while GPU0 was free; earlier analysis reran only the two required cheap-column evaluations into an isolated output root and cancelled the obsolete stuck task.

Question: Should selected-whole-word-group mean MLM continue to 100M/full evaluation, or stop and yield the H100 to the prepared support-floored tokenizer screen?

| exposure | wordmean mean7 | tokenmean mean7 | clean mean7 | Δ wm-token | Δ wm-clean | Δ BLiMP | Δ Supp | Δ EWoK | Δ Entity | Δ COMPS | Δ GPIQA | Δ Reading |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 | 42.6043 | 42.6086 | 41.3164 | -0.0043 | +1.2879 | -0.7800 | -0.3100 | -1.8000 | -1.0500 | +0.7700 | +4.0700 | -0.9300 |
| 80 | 42.7029 | 42.9486 | 41.6021 | -0.2457 | +1.1007 | -1.5600 | -1.7100 | -1.8100 | -1.1500 | +0.5100 | +4.5250 | -0.5250 |

## Decision

- continue word-mean: `False`
- stop word-mean without LR retune: `True`
- next expensive action: launch exactly one init-matched minfreq50 support-floor 80M screen, then evaluate 70M/80M cheap columns
- reason: Word-mean does not broadly strengthen the legal compact-view reinvestment trajectory. At 80M it improves only COMPS and GlobalPIQA while damaging BLiMP, Supplement, EWoK, Entity, and Reading; at 70M mean7 is flat only because a large GlobalPIQA gain masks losses in EWoK, Entity, BLiMP/Reading and near-flat Supplement.

This is a route decision from cheap columns only. It does not change the best legal endpoint or create a submission candidate.

Full JSON: `experiments/archive/frontier_consolidation/data/wordmean_recovery_decision/wordmean_recovery_decision.json`
