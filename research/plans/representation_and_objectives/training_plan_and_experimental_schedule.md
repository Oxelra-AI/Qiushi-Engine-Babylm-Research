# leader analysis and route pivot — Complete Training Plan for 41.8+ SOTA Attempt

## Situation Summary

- **Target**: Overall ≥ 41.8 (current SOTA: wwm_curriculum_simplification_40k at 41.80)
- **Our best**: clean-Qwen seed43022 at Overall 41.34 (8×480 DeBERTa-v2, AdamW, 16k, official corpus)
- **Gap**: -0.46 concentrated in EWoK (-5.88), GlobalPIQA (-3.05), Entity (-2.69), COMPS (-1.79)
- **Advantage**: Supplement (+6.83), Reading (+2.34), SuperGLUE (+0.52)
- **Leader's key innovation**: Uses FineWeb-Edu simplification pairs (10M words of factual web content) instead of official BabyLM corpus
- **Our response**: Generate factual content with Qwen3.5-9B + apply leader's curriculum innovations

## Pending Generation

- Qwen3.5-9B factual generation (50k prompts, dual H100)
  - Shard 0: 20k expansion prompts (max 200 tokens) → ~3-4M words factual content
  - Shard 1: 30k simplification/paraphrase prompts (max 90 tokens) → ~2-3M words
  - ETA: ~30-60 minutes on dual H100

## Experimental Schedule

### Experiment 1: Leader's Training Recipe on Existing Data (FASTEST TEST)

**Purpose**: Test whether leader's training innovations (curriculum, LAMB, architecture) 
alone can close the gap, even without FineWeb data.

**Configuration (matching leader exactly except data)**:
- Architecture: DeBERTa-v2, **12 layers, hidden 384, intermediate 1280, 12 heads** (~34.7M)
- Data: Existing clean-Qwen aligned 10M corpus (`experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl`)
- Optimizer: **LAMB**, max LR **0.007**, cosine schedule
- Epochs: 10 (100M word exposure)
- Sequence length curriculum: **64 → 256** (increase over training)
- Masking curriculum: **WWM epochs 1-7, token masking epochs 8-10**
- Tokenizer: 16k baseline (since data is same, keep tokenizer matched)
- Checkpoints: every 1M words (1M-10M) then every 10M (10M-100M) for AoA

**Expected outcome**: If leader's training recipe helps, expect improvement over 41.34.
Unlikely to match 41.8 without FineWeb-level factual data, but will tell us how much
the curriculum + LAMB contribute independently.

**Estimated time**: ~4-5 hours on single H100.

### Experiment 2: Factual Enriched Corpus + Leader's Recipe (MAIN SHOT)

**Purpose**: Full reproduction of leader's approach using our Qwen-generated factual data.

**Configuration**:
- Architecture: DeBERTa-v2, 12 layers, hidden 384, intermediate 1280, 12 heads (~34.7M)
- Data: Newly assembled factual corpus from Qwen generation (target: 10M words)
  - ~4M words of factual expansions (Qwen-enriched SimpleWiki)
  - ~1.5M words of simplifications
  - ~1.5M words of paraphrases (our innovation over the leader)
  - ~3M words from CHILDES + Gutenberg (developmental + narrative diversity)
- Tokenizer: **40k SentencePiece BPE** trained on the new corpus
- Optimizer: LAMB, LR 0.007, cosine
- Epochs: 10
- Curriculum: seq 64→256, masking WWM→token
- Checkpoints: Full ladder for AoA

**Expected outcome**: Should approach or match 41.8 on EWoK/Entity/COMPS/GlobalPIQA.
Paraphrases should preserve our Supplement/Reading advantage. Could reach 42+.

**Estimated time**: ~30-60 min corpus assembly + ~4-5 hours training.

### Experiment 3: AoA Enhancement (COMPLEMENTARY)

**Purpose**: Even modest AoA (5-10 points) adds 0.56-1.11 to Overall.

**Key insight**: AoA measures age-of-acquisition progression across checkpoints.
Models that learn easy/common words first and complex words later score positive AoA.
The leader has AoA=0 (no pattern). If we can engineer a training progression that
produces developmentally-like acquisition order, we get free Overall points.

**Mechanism**: 
- Curriculum from simple→complex content naturally produces AoA pattern
- CHILDES content first (child-level vocabulary), then factual content (adult vocabulary)
- Checkpoint ladder at every 1M words captures this progression

**Configuration**: Same as Exp 2, but with explicit CONTENT curriculum:
- Epochs 1-3: Start with CHILDES + simple factual content
- Epochs 4-7: Mix in complex factual content
- Epochs 8-10: Full corpus including complex paraphrases

This is coordinated with frontier_consolidation who is also exploring AoA approaches.

### Experiment 4: Architecture Comparison (IF TIME PERMITS)

**Purpose**: Determine if 12×384 (leader, deeper) vs 8×480 (ours, wider) matters.

**Configuration**: Run Exp 2's data with both architectures, compare.

## Training Script Requirements

The existing `masking_curriculum_trainer.py` from COMPACT_EXPERIENCE already supports:
- ✅ DeBERTa-v2 with configurable layers/hidden/heads
- ✅ wwm_to_token masking mode
- ✅ WWM fixed and token fixed modes
- ✅ Cosine LR schedule
- ✅ Official BabyLM data loading

Needs addition:
- ❌ LAMB optimizer (pytorch_optimizer or custom implementation)
- ❌ Sequence length curriculum (start at 64, increase to 256)
- ❌ Custom tokenizer support (40k SentencePiece)
- ❌ Custom data format support (pair-structured corpus)
- ❌ Content curriculum (different data ordering per epoch)

## File Locations

- Prompts: `experiments/archive/representation_and_objectives/training/data/factual_prompts_shard*.jsonl`
- Generation outputs (when ready): `experiments/archive/representation_and_objectives/training/runs/factual_*/generations.jsonl`
- Corpus assembly: `experiments/archive/representation_and_objectives/training/scripts/assemble_factual_corpus.py`
- Training corpus (when assembled): `experiments/archive/representation_and_objectives/training/data/training_corpus/factual_training_corpus.train`
- Existing clean-Qwen data: `experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl`
- Existing masking curriculum trainer: `experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py`
- Leader analysis: `research/notes/representation_and_objectives/leader_analysis_and_route_pivot.md`

## Related Representation Comparison

- frontier_consolidation is working on: curriculum innovations on existing data + AoA mechanism
- representation_and_objectives (us) is working on: factual data generation + 12×384 architecture
- Key shared insight: Leader's success is primarily DATA-driven (FineWeb vs official corpus)
- Complementary: frontier_consolidation tests curriculum alone; we test data + curriculum together

## Key Decision Points

1. **After generation completes**: Assemble corpus, check word count, verify content quality
2. **After Exp 1 finishes**: If curriculum alone helps (>41.5), invest in curriculum tuning
3. **After Exp 2 finishes**: If factual data helps (>41.8), iterate on data composition
4. **AoA**: If AoA is achievable without NLP damage, it's the fastest path to 42+

## Proposed Scientific Comparisons

1. Wait for generation task `s2_t38_tool1` to complete
2. Run corpus assembly script
3. Train 40k SentencePiece tokenizer on new corpus
4. Write/adapt training script with LAMB + curriculum
5. Launch Experiment 1 (quick test with existing data)
6. Launch Experiment 2 (main attempt with factual data)
7. Evaluate with full 9-column official pipeline
