# orthogonal route screen summary: Orthogonal Route Cheap Screen — Summary

## Analysis Scope

Constructed and executed cheap no-training screens for three orthogonal routes to stable context-conditioned competence, as requested by earlier analysis.

## Route 1: Procedural Data — CLOSED

Extracted 1,142 procedural passage candidates from the legal 10M pool. After strict quality filtering (removing CHILDES dialogue, URLs, non-prose, fragments), only **19 clean prose passages** remain. The legal corpus lacks sufficient clean multi-step procedural structure. Combined with the interference ladder evidence (models already saturate consistent_prior=1.0 and explicit_final=1.0), more procedural data cannot address the measured deficit.

## Route 2: Representation Probe — RECOMMENDED

Designed a concrete probe specification using the existing 800-frame interference ladder and the Route 3 cross-context items. The probe extracts hidden representations at each transformer layer (L1–L8) and trains a logistic regression to predict the correct state/property. This answers the foundational question: does binding information EXIST in the representation but fail to reach the MLM head?

**Cost**: ~30 min GPU (5 checkpoints × forward passes), ~5 min CPU (probe training). No training, fully reversible.

**Key checkpoint paths for probe:**
1. `experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022/hf_model/chck_100M`
2. `experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M`
3. `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_50M`
4. `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M`
5. `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M`

## Route 3: Cross-Context Signal — RECOMMENDED

Extracted 2,396 cross-context pairs from the legal pool (1,361 antonym-pair + 1,035 same-entity). Scored 2,039 common items on three existing checkpoints.

**Three-checkpoint panel (2,039 common items):**
| Checkpoint | Crossed Success | Mean Δ | Δ>0 |
|---|---|---|---|
| chck82 (scale1.75) | 23.6% | 1.177 | 74.1% |
| legal16k_base100 | 24.9% | 1.243 | 70.6% |
| legal40k_8x480_100 | 19.0% | 1.364 | 69.1% |

**Family structure**: Three regimes exist:
1. Zero-crossed (0%): deep/shallow, strong/weak, smooth/rough, clean/dirty — extreme prior overrides all context
2. Intermediate (5–30%): hot/cold, wet/dry, bright/dark, eyes — context helps sometimes
3. High-crossed (>40%): soft/hard, open/closed, new/old — model already context-sensitive

This is a **strongly unsaturated, trajectory-discriminating, family-structured surface** that directly measures the same deficit as EWoK (context-dependent alternative selection). Models fail on 75–81% of crossed items despite positive global delta.

## Key files

- Plan: `plans/orthogonal_route_screen_plan.md`
- Cross-context pairs: `data/orthogonal_route_extraction/route3_all_cross_context.jsonl`
- Scores: `data/cross_context_scores/chck82.json`, `legal16k_base100.json`, `legal40k_8x480_100.json`
- Analysis: `data/route_screen_analysis/route_screen_analysis.json`
- Scripts: `scripts/extract_route_candidates.py`, `score_cross_context_pairs.py`, `analyze_route_screens.py`
