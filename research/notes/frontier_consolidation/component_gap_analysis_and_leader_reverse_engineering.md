# component gap analysis and leader reverse engineering — Component Gap Analysis and Leader Reverse Engineering

## Summary

Our best result (clean-Qwen same-window pairs, DeBERTa-v2 8×480, seed43022) achieves Overall 41.3443. The leader `wwm_curriculum_simplification_40k` achieves 41.8. The gap is 0.46 Overall = 4.14 total column points.

## Component-by-component comparison

| Component       | Leader  | Ours    | Gap    | Direction |
|-----------------|---------|---------|--------|-----------|
| BLiMP           | 67.20   | 66.84   | -0.36  | behind    |
| Supplement      | 56.01   | 62.84   | **+6.83**  | **ahead** |
| EWoK            | 56.07   | 50.19   | **-5.88**  | behind    |
| Entity          | 28.45   | 25.76   | -2.69  | behind    |
| COMPS           | 53.57   | 51.78   | -1.79  | behind    |
| GlobalPIQA      | 39.67   | 36.62   | -3.05  | behind    |
| SuperGLUE       | 69.79   | 70.31   | **+0.52**  | **ahead** |
| Reading         | 5.42    | 7.76    | **+2.34**  | **ahead** |
| AoA             | 0.00    | 0.00    | 0.00   | tied      |
| **Overall**     | **41.80** | **41.34** | **-0.46** | |

## Where points must come from

We're **ahead** on Supplement (+6.83), Reading (+2.34), SuperGLUE (+0.52) = total +9.69
We're **behind** on EWoK (-5.88), GlobalPIQA (-3.05), Entity (-2.69), COMPS (-1.79), BLiMP (-0.36) = total -13.77

Net: -4.08 points (≈ -0.45 Overall, matching the gap)

**Biggest deficit: EWoK (-5.88)**. This single component accounts for 42% of our total deficit.

## Leader reverse engineering

### Model: `go76dof/wwm_curriculum_simplification_40k`
- **Architecture**: DeBERTa-v2, hidden_size=384, intermediate=1280, 12 layers, 12 heads, 34.7M params
- **Tokenizer**: 40,000 vocab SentencePiece BPE, trained on simplification pairs
- **Optimizer**: LAMB, lr=0.007, cosine schedule
- **Masking curriculum**: WWM epochs 1-7, Token masking epochs 8-10
- **Sequence length curriculum**: 64 → 256 (batch inversely scaled)
- **Data**: FineWeb simplification pairs (`go76dof/Fineweb_simplification_pairs`), 9,999,969 words
- **Data format**: Original FineWeb sentence + simplified rewrite (blank-line separated pairs)
- **Training**: 10 epochs

### Key insight: same principle as our clean-Qwen approach
The leader uses meaning-preserving paired rewrites (simplification from FineWeb) with DeBERTa-v2 WWM. Our approach uses Qwen-generated paraphrases of BabyLM corpus with DeBERTa-v2 WWM. **Both are same-window aligned-rewrite methods.**

### Innovation differences (leader vs us)

| Feature              | Leader                     | Ours                        | Novel for us? |
|----------------------|----------------------------|-----------------------------|---------------|
| Masking curriculum   | WWM→Token at epoch 7       | WWM throughout              | **YES**       |
| Seq length curriculum| 64→256                     | Fixed 256                   | **YES**       |
| Optimizer            | LAMB, lr=0.007             | AdamW (default lr)          | Tested in COMPACT_EXPERIENCE |
| Architecture         | 12×384 (34.7M)             | 8×480                       | **YES**       |
| Tokenizer vocab      | 40k                        | 16k                         | Tested in COMPACT_EXPERIENCE |
| Data source          | FineWeb simplification     | BabyLM + Qwen paraphrase   | **YES**       |
| Transformation       | Complex→Simple (compress)  | Near-length paraphrase      | **YES**       |

## AoA opportunity analysis

**Critical observation**: All top entries except two have AoA=0.0 (no checkpoint submission).

| Model                        | AoA    | Overall | Impact   |
|------------------------------|--------|---------|----------|
| deberta-base-75k-sam_ext-s1  | **+22.9** | 40.62   | +2.54 Overall |
| BabySteps_MurphysLaw-10M    | -15.1  | 40.86   | -1.68 Overall |
| All others                   | 0.0    | various | neutral  |

**If we achieved AoA=+22.9 on our current model, Overall would jump to ~43.9.**
Even AoA=+5 gives +0.56 Overall, pushing us to 41.9.

AoA requires: (1) saving checkpoints at milestones chck_1M...chck_9M, chck_10M...chck_100M (19 total); (2) correct surprisal format; (3) training dynamics that correlate with human AoA norms.

## Tail averaging opportunity

BabySteps_MurphysLaw uses "tail average over last 20% of training steps" which improved SuperGLUE by ~1.6 points. This is a zero-cost post-training operation.

## Files

- Leader model card: `research/documents/compact_experience/data/babylm2026_surface/hf_repo_surfaces/go76dof__wwm_curriculum_simplification_40k/README.md`
- Leader details: `research/notes/compact_experience/hf_top_repo_surface.md` (§1)
- Top20 scores: `experiments/archive/compact_experience/data/babylm2026_surface/strict_small_top20.json`
- Our best: `experiments/archive/compact_experience/data/full_eval/full_eval_summary.json`
