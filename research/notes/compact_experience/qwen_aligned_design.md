# Official-source Qwen-aligned data experiment design

## Scientific Rationale

The mix25 evidence (mixture experiment design) established that low-dose (25%) semantically aligned data
mixed with official BabyLM text produces a reproducible broad-competence improvement:
- Fixed-init replication: equal7_fast +0.7029, weighted_fast_proxy +0.7033
- Full official Overall: 41.4803 vs inherited 40.7028 (+0.7775)
- Mechanism: broad-competence regularization/coverage, NOT Entity-specific recurrence

The earlier mix25 experiments used SynCSE/WikiLarge external pairs, which were not retained in the proposed final data-source design. The proposed official-source version uses only:
- Official BabyLM Strict-Small training text as source
- Local Qwen3.5-9B as a rewrite generator (no weights/tokenizer/hidden states exposed)
- Every generated word counted within the 10M budget

## Design

### Data Generation Pipeline
1. Extract ~45,000 sentences (12-50 words) from official 62,500-row pool
   - Balanced across all 6 sources (bnc_spoken, childes, gutenberg, open_subtitles, simple_wiki, switchboard)
   - Filtered for quality: complete sentences, adequate content, no heavy fillers
   
2. Generate meaning-preserving rewrites using local Qwen3.5-9B
   - Prompt: "Rewrite...to convey the same meaning using different words and sentence structure"
   - Temperature 0.7 for diverse but coherent output
   - Max 80 new tokens per rewrite
   
3. Validate rewrites:
   - Length: 5-80 words
   - Ratio: within 3x of source length
   - No meta-responses or generation artifacts
   
4. Pack validated pairs into 160-word training rows:
   - Format: [orig1][rewrite1][orig2][rewrite2]...
   - Creates within-window semantic adjacency visible to MLM attention
   - Target: ~25% of 62,500 total rows (adjustable based on available pairs)

### Training Configuration
- Architecture: DeBERTa-v2 8×480 (proven strongest backbone)
- Tokenizer: baseline16k (same as all prior strong results)
- Optimizer: AdamW, LR 1e-3, warmup 6%, weight decay 0.01
- Masking: WWM fixed 15% (proven strongest mode)
- Sequence: fixed 256, batch 256
- Seeds: extra_init_seed=43022, train_rng_seed=43023 (matches fixed-init replication)
- Checkpoints: every 1M words (chck_1M...chck_100M for complete AoA)
- Total exposure: 100M words (10 passes over 10M pool)

### Experimental Arms
1. **official_only**: All 10M words from official pool, no intervention
2. **qwen_aligned**: ~25% paired rows + ~75% official rows, same 10M total

### Key Differences from Prior mix25
- Source text: ALL from official BabyLM (vs. external SynCSE/WikiLarge)
- Generator: local Qwen3.5-9B (vs. pre-existing external datasets)
- No tokenizer/weight/hidden-state transfer to submission model
- Full checkpoint cadence for official AoA computation
- Matched official-only control under identical conditions

### Success Criterion
- If qwen_aligned - official_only ΔOverall ≥ +0.25: proceed to seed replication and SOTA attempt
- If ΔOverall < +0.25: mechanism is data-source-dependent, need alternative route

## Files
- `scripts/extract_and_prompt.py` — sentence extraction and prompt creation
- `scripts/validate_and_materialize.py` — validate rewrites, create 10M corpora
- `scripts/launch_training.sh` — parallel training on 2 H100s
- `scripts/eval_full_official.py` — full nine-column evaluation
- `data/qwen_aligned/` — all data artifacts
- `training/data/rewrite_prompts.jsonl` — prompts for Qwen generation
- `training/runs/qwen_rewrites_full/` — generation outputs
