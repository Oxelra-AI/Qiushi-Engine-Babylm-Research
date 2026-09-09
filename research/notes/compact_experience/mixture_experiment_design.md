# mixture experiment design — Official + Aligned Mixture Experiment Design

## Scientific rationale

paired alignment accounting audit established:
- Coherent adjacent text (paraphrase or hard-negative) teaches Entity Tracking strongly
  - ALIGNED vs INITIAL_MODEL_STUDIES baseline: Entity +7.38 (29.16 vs 21.78)
  - Effect is from local relation/entity recurrence, not specifically paraphrase
- But 100% paired data sacrifices BLiMP/Supplement/COMPS vs official corpus
  - ALIGNED equal-7 ≈ baseline equal-7 (redistribution, not net gain)
- Official corpus gives strong syntax/diagnostic scores (BLiMP 67.34, Supplement 65.2)

**Hypothesis:** An optimal mixture of official (syntax/diagnostic) + paired-aligned (entity/coherent) data can exceed BOTH pure endpoints on Overall, because:
1. Entity is one of 9 equally-weighted columns → large Entity gain has major Overall impact
2. Syntax/diagnostic breadth from official data may survive partial dilution
3. If the dose-response is concave (diminishing returns), interior optimum exists

## Experimental design

**Arms:** (all exactly 10M pool, 100M exposure, 10 passes, legal ≤10M/≤10epochs)
| Arm | Official examples | Aligned examples | Official words | Aligned words |
|-----|------------------:|------------------:|---------------:|--------------:|
| INITIAL_MODEL_STUDIES baseline (0%) | 62,500 | 0 | 10,000,000 | 0 |
| mix_25pct | 46,875 | 15,625 | 7,500,000 | 2,500,000 |
| mix_50pct | 31,250 | 31,250 | 5,000,000 | 5,000,000 |
| mix_75pct | 15,625 | 46,875 | 2,500,000 | 7,500,000 |
| ALIGNED (100%) | 0 | 62,499 | 0 | 9,999,840 |

**Recipe:** identical to inherited INITIAL_MODEL_STUDIES baseline
- DeBERTa-v2, 8 layers, hidden 480, 8 heads, intermediate 1920 (34.5M params)
- baseline16k tokenizer
- WWM, mask_prob 0.15
- batch 256, seq_len 256
- AdamW: LR 1e-3, weight_decay 0.01, warmup 0.05
- seed 43

**Data sources:**
- Official: BabyLM-community/BabyLM-2026-Strict-Small (bnc_spoken, childes, gutenberg, open_subtitles, simple_wiki, switchboard)
- Aligned: WikiLarge (Apache-2.0) + SynCSE-partial-NLI (MIT), packed as [orig_i][simp_i][orig_j][simp_j]... within 160-word sequences

**Selection method:** Random subset without replacement, seed 43. Official examples maintain natural source-proportional distribution. Pool shuffled before expansion.

## Expected analysis

1. Dose-response curve: plot each column score vs aligned fraction (0%, 25%, 50%, 75%, 100%)
2. Identify optimal mixture fraction (max equal-7 or full 9-column Overall)
3. Check for nonlinearity: is the optimum interior or at an endpoint?
4. If best mixture > baseline, it becomes the data foundation for next experiments

## Artifacts

- Materializer: `scripts/mixture_materializer.py`
- Pools: `data/mixture/{official,mix_25pct,mix_50pct,mix_75pct}_pool.jsonl`
- Training files: `data/mixture/training_files/{mix_25pct,mix_50pct,mix_75pct}_100M.jsonl`
- Launcher: `scripts/launch_mixture_training.sh`
- Evaluator: `scripts/eval_mixture_fast.py`
- Summary: `data/mixture/mixture_summary.json`

## Relation to SOTA path

The mixture experiment tests DATA composition alone on the fixed backbone. Even if the best mixture improves Overall by +0.1 to +0.3, that alone likely won't reach 41.8. The full path to SOTA likely requires combining:
1. Optimal data mixture (this experiment)
2. Tokenizer upgrade (16k → 40k SentencePiece)
3. Optimizer change (AdamW → LAMB, LR 0.007)
4. Possible capacity reallocation (8×480 → 12×384)

The leader achieved 41.8 with ALL four changes simultaneously. This experiment isolates the data composition dimension.
