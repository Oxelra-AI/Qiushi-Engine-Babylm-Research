# leader analysis and route pivot — Leader Method Analysis and Strategic Route Pivot

## Leader's Complete Method (go76dof/wwm_curriculum_simplification_40k, Overall 41.8)

**Source**: HuggingFace model card, confirmed public

| Component | Leader's Choice |
|-----------|----------------|
| Architecture | DeBERTa-v2, 12 layers, hidden 384, intermediate 1280, 12 heads, **34.7M params** |
| Data | FineWeb-Edu simplification pairs (`go76dof/Fineweb_simplification_pairs`), **9,999,969 words** |
| Data Format | Original FineWeb sentence → simplified rewrite, blank-line separated pairs |
| Tokenizer | 40k SentencePiece BPE, trained on the simplification pairs corpus |
| Optimizer | LAMB, cosine schedule, max LR 0.007 |
| Epochs | 10 (→ 100M word exposure) |
| Seq Length Curriculum | 64 → 256 (short first, long later) |
| Masking Curriculum | WWM epochs 1–7, token-level masking epochs 8–10 |
| Author | Shaoxiang Wu |

**Critical insight**: The leader does NOT use the official BabyLM corpus at all. They replace it
entirely with FineWeb-Edu simplification pairs. This is legal under BabyLM 2026 Strict-Small
rules (confirmed by the entry being accepted on the official leaderboard).

## Why the Leader Beats Us

### Component-wise gap (clean-Qwen 41.34 → leader 41.8 = -0.46)

| Column | Ours | Leader | Gap | Explanation |
|--------|------|--------|-----|-------------|
| BLiMP | 66.84 | 67.2 | -0.36 | Slight syntax benefit from curriculum? |
| Supplement | 62.84 | 56.01 | **+6.83** | Our Qwen paraphrases give linguistic diversity |
| EWoK | 50.19 | 56.07 | **-5.88** | FineWeb is factually rich → world knowledge |
| Entity | 25.76 | 28.45 | -2.69 | More entities in web text vs child speech/fiction |
| COMPS | 51.78 | 53.57 | -1.79 | Compositional semantics from factual content |
| GlobalPIQA | 36.62 | 39.67 | -3.05 | Physical world knowledge from web text |
| SuperGLUE | 70.31 | 69.79 | +0.52 | Essentially tied |
| Reading | 7.76 | 5.42 | +2.34 | Our diverse linguistic forms help psycholinguistics |
| AoA | 0.0 | 0.0 | 0 | Both zero |

### Root cause
The official BabyLM corpus (CHILDES 28%, Gutenberg 26%, OpenSubtitles 23%, SimpleWiki 15%,
BNC Spoken 8%, Switchboard 0.2%) is primarily conversational/literary. It lacks the dense
factual/encyclopedic content that EWoK, Entity Tracking, COMPS, and GlobalPIQA test.

FineWeb-Edu is high-quality educational web content: science, history, geography, technology,
society—exactly what these world-knowledge benchmarks measure.

## Scientific Opportunity: Hybrid Factual-Linguistic Data

### Hypothesis
Combining factual-rich FineWeb content (for world knowledge) with linguistic diversity via
paraphrasing (for syntax/semantics) should produce training data that improves BOTH
the world-knowledge columns AND the linguistic columns simultaneously.

### Theoretical ceiling (best-of-both-models estimate)
If we achieve the leader's EWoK/Entity/COMPS/GlobalPIQA AND preserve our Supplement/Reading:
(67.2 + 62.84 + 56.07 + 28.45 + 53.57 + 39.67 + 70.31 + 7.76 + 0) / 9 = **42.87**

Even achieving 50% of the gap closure gives ~42.3, well above 41.8 SOTA.

### Supporting literature
1. "Beyond Repetition" (Roque & Velasco 2025): Simplified data outperforms repeated text.
   Smaller models benefit from simple→complex curriculum.
2. "Rethinking Text Complexity" (Velasco & Roque 2025): Simplified text helps BLiMP/PIQA
   but hurts EWoK/Entity. The PAIRS approach preserves both by providing both versions.
3. Our clean-Qwen paraphrase approach: Demonstrated +6.83 on Supplement through linguistic
   diversity, but doesn't add new factual content.

## Route Design: FineWeb Simplification + Paraphrase Hybrid

### Phase 1: Reproduce Leader Baseline (Fast Verification)
- Download `go76dof/Fineweb_simplification_pairs` from HuggingFace
- Train with leader's exact recipe (12×384, LAMB 0.007, 40k, curriculum)
- Verify we can match ~41.8 locally
- This validates infrastructure and establishes a controlled baseline

### Phase 2: Hybrid Data Innovation
- Generate paraphrases of the FineWeb-Edu originals using Qwen-3 (approved model)
- Create "triplet" data: original + simplified + paraphrased
- Within 10M word budget, test mixtures:
  - 50% original + 50% simplified (leader's approach)
  - 33% original + 33% simplified + 33% paraphrased (balanced triplet)
  - 40% simplified + 40% paraphrased + 20% original (diversity-weighted)

### Phase 3: Architecture & Curriculum Optimization
- Test both 12×384 (leader) and 8×480 (our proven backbone) on hybrid data
- Test curriculum variations:
  - Leader's: seq 64→256, WWM→token
  - Alternative: include paraphrases in curriculum progression
  - Difficulty-based: simple→complex based on sentence complexity scores

### Phase 4: Evaluation and Submission
- Full official-compatible 9-column evaluation
- Multi-seed verification
- Checkpoint ladder for AoA
- Submit to leaderboard

## Proposed Comparisons

1. ✅ Document leader method and gap analysis (this note)
2. Check if leader's dataset is downloadable via HuggingFace
3. Verify BabyLM 2026 rules allow FineWeb data (confirmed by leader's acceptance)
4. Design the data preparation pipeline (Qwen-3 generation prompts)
5. Set up the 12×384 + LAMB training configuration
6. Plan the fast initial experiment (download leader data → train → evaluate)

## Key Constraints Verified
- ≤10M words training data ✓ (leader uses 9,999,969)
- ≤100M word exposure (10 epochs) ✓
- Approved model families for generation: Qwen 2.5/3/3.5 up to 9B ✓
- No teacher distribution leakage ✓ (only text generation, not logit distillation)
- Required intermediate checkpoints for AoA ✓ (standard checkpoint ladder)

## Risk Assessment
- **Low risk**: Leader's exact recipe is public and proven at 41.8
- **Medium risk**: Our hybrid approach might trade Supplement for EWoK without net gain
- **Mitigation**: Phase 1 establishes baseline; Phase 2 is incremental innovation
- **Upside**: If paraphrases ADD to simplification rather than replace, could reach 42+
