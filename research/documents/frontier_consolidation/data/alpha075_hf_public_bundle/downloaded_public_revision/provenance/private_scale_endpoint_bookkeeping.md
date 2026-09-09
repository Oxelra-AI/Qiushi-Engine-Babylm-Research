# coherence margin signal isolation private-scale endpoint bookkeeping

The private-scale mechanism was already closed in private scale endpoint vs mechanism synthesis: alpha scaling is an amplitude-controlled redistribution over the coherent86 private residual, with COMPS/BLiMP-dominated churn and a small GlobalPIQA few-example component, not a general relation/state retention principle. coherence margin signal isolation only repaired endpoint arithmetic and local carriers from already completed official-compatible payloads.

## Alpha0.5

Previously repaired in reopen structured context margin route:

- cheap7: 44.17785714285714
- SuperGLUE: 69.78975515726182
- Overall(AoA0): 42.11497279525131
- delta vs protected chck82: +0.1724916278653268
- local truthful carrier: `data/truthful_private_scale_carriers/coherent86_alpha0p5/all_full_preds_truthful_coherent86_alpha0p5_mlm.json`
- carrier SHA256: `2f4dc195261062ee34010386c74de514133b5888276ba93b27951e731c723a58`

## Alpha0.75

The official-compatible SuperGLUE run itself completed; the wrapper failed only on the same stale missing `cheap7` key after evaluation. The arithmetic was repaired with `scripts/repair_private_scale_superglue_summary.py` without rerunning SuperGLUE.

- repaired summary: `data/private_scale_superglue_summary_alpha0p75/coherent86_private_scale_0p75_sg_retry_superglue_summary.{json,md}`
- cheap7: 44.18142857142857
- SuperGLUE: 69.81922238969935
- Overall(AoA0): 42.1210247099666
- delta vs protected chck82: +0.17854354258061278
- local truthful carrier: `data/truthful_private_scale_carriers/coherent86_alpha0p75/all_full_preds_truthful_coherent86_alpha0p75_mlm.json`
- carrier SHA256: `40181994810e21bc823474a3e4ac84c8eb42213e03904d36a60a4698477d1994`
- validator: PASS under current strict-small validator
- model: `training/runs/coherent86_private_scale_0p75/hf_model/final`
- model safetensors SHA256: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040c4c6203343b15c8`? See manifest for exact value; it should match coherent86 base weights with config scale changed.

Important correction: the manifest records the exact model SHA as `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`.

## Status

No upload and no leaderboard submission were performed. Alpha0.75 is now the strongest local endpoint carrier numerically among the alpha endpoints, but the protected public fallback remains `chck82` rank-1 Overall 41.94, and the scientific interpretation of the alpha family remains endpoint redistribution rather than a data-efficient learning principle.
