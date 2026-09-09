# babysteps public method reading component-level gap decomposition (decisive strategic evidence)

## Sources
- Live leaderboard parsed: `experiments/archive/representation_and_objectives/data/babylm2026_live_surface/leaderboard_parsed.json` (fetched 2026-08-29T07:29Z)
- Our best trusted local coordinate: `qwen_clean_aligned` in `experiments/archive/compact_experience/data/control_eval_summary.json`, Overall 41.34429066479573
- Official scoring: `experiments/archive/compact_experience/scripts/babylm_official_scoring.py` — Overall = unweighted mean of 9 columns (BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE, GlobalPIQA, Reading, AoA). Each Overall point = column point / 9.

## Frontier update
- **strict-small** leader (our track) is still `wwm_curriculum_simplification_40k` (go76dof) at **Overall 41.8**.
- `go76dof/spangeo05_simplification` at Overall **41.93** is on the **strict** track (_component_id 5), NOT strict-small. Not our direct target, but same author/family and confirms simplification-data family dominates.
- Second strict-small: `RecGPT-10M` at 41.53.

## Exact per-column decomposition: leader (41.8) minus ours (41.34)

| Column | Ours | Leader | Δ(leader−ours) | Overall contribution (Δ/9) |
|---|---:|---:|---:|---:|
| BLiMP | 66.84 | 67.20 | +0.36 | +0.040 |
| Supplement | 62.84 | 56.01 | **−6.83** | **−0.759** (we WIN big) |
| EWoK | 50.19 | 56.07 | **+5.88** | **+0.653** (we LOSE big) |
| Entity | 25.76 | 28.45 | +2.69 | +0.299 (lose) |
| COMPS | 51.78 | 53.57 | +1.79 | +0.199 (lose) |
| SuperGLUE | 70.31 | 69.79 | −0.52 | −0.058 (we win) |
| GlobalPIQA | 36.62 | 39.67 | +3.05 | +0.339 (lose) |
| Reading | 7.76 | 5.42 | −2.34 | −0.260 (we win) |
| AoA | 0.0 | 0.0 | 0 | 0 |
| **Overall** | **41.34** | **41.80** | | **+0.46** |

## Decisive interpretation
- Our deficit is ENTIRELY in the world-knowledge / commonsense-reasoning cluster: EWoK (−0.65 Overall), GlobalPIQA (−0.34), Entity (−0.30), COMPS (−0.20). Cluster deficit ≈ **−1.49 Overall**.
- We already DOMINATE the leader on Supplement (+0.76 Overall), Reading (+0.26), SuperGLUE (+0.06). Advantage ≈ **+1.09 Overall**. Net −0.46.
- **We do not need to match the leader everywhere.** To beat 41.8 we need to gain roughly **+0.5 Overall in the EWoK/GlobalPIQA/Entity/COMPS cluster while preserving our Supplement/Reading edge.** That is only ~+4.5 EWoK-points-equivalent spread across the cluster, or a mix.

## Consequence for route selection
- The active same-source semantic-view contrast (SimpleWiki simplification/paraphrase of the SAME propositions, 8.2% of corpus) CANNOT add world knowledge. Its plausible reach on EWoK/GlobalPIQA/Entity/COMPS is near zero by construction. Even a clean positive result is unlikely to move the deficit cluster.
- The lever that matches the deficit is **broader factual/world-knowledge experience**: exactly the FineWeb source-by-rewrite question (broader factual sentences, optionally coupled to faithful simplifications). This is the route that can plausibly close −0.46.
- RISK to manage: adding factual FineWeb material must NOT collapse our Supplement (+6.83) and Reading (+2.34) advantages. The leader's Supplement is only 56.01 — its factual data HURT Supplement/Reading. So the design must ADD world knowledge on top of the conversational/developmental corpus that gives us Supplement/Reading strength, rather than replacing it wholesale like the leader did.

## Next-experiment thesis (to validate against frontier_consolidation and the pending no-AoA result)
Add a modest, quality-screened broader-factual FineWeb block (single-document, seq-safe, ~1.5-2M words) to the protected clean-Qwen corpus, KEEPING the developmental/conversational majority intact, under the protected 8x480/16k/WWM/AdamW recipe, with a matched official control. Target: move EWoK+GlobalPIQA+Entity+COMPS up ~+3-5 points combined while holding Supplement>60 and Reading>7. If that holds, Overall > 41.8.
