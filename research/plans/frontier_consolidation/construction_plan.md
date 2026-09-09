# Construction Plan: Leader-Recipe Replication and Enhancement

## Research Situation

Our best: Overall 41.34 (clean-Qwen same-window pairs, DeBERTa-v2 8×480, AdamW, WWM fixed, seq256).
Leader: Overall 41.80 (FineWeb simplification pairs, DeBERTa-v2 12×384, LAMB lr=0.007, WWM→Token curriculum, seq 64→256).

The leader uses the **same principle** (meaning-preserving paired rewrites + DeBERTa-v2 WWM) but with 6 specific differences. The fastest path to 41.8+ is to test each difference systematically.

## Priority-ordered improvement tracks

### Track A: Training recipe upgrade (HIGHEST PRIORITY — can test immediately)
Test the leader's training innovations on our existing clean-Qwen 10M data, which isolates training recipe effects from data effects.

**Modifications to masking_curriculum_trainer.py:**
1. Add LAMB optimizer support (`--optimizer lamb`)
2. Add explicit `--intermediate_size` argument (leader uses 1280, not 384×4=1536)

**Experiment A1: Full leader recipe on our clean-Qwen data**
```
python masking_curriculum_trainer_v2.py \
  --output_dir ... \
  --example_jsonl <clean_qwen_10M.jsonl> \
  --masking_curriculum wwm_to_token \
  --switch_frac 0.7 \
  --mask_prob_start 0.15 --mask_prob_end 0.15 \
  --seq_len_schedule "0.0:64,0.7:256" \
  --max_seq_length 256 \
  --optimizer lamb \
  --learning_rate 0.007 \
  --weight_decay 0.01 \
  --n_layer 12 --hidden_size 384 --n_head 12 \
  --intermediate_size 1280 \
  --max_word_exposure 100000000 \
  --example_pool_words 10000000 \
  --batch_size 256 \
  --checkpoint_words 1000000 \
  --seed 43022
```

**Experiment A2: Leader recipe on our 8×480 architecture (isolate architecture effect)**
Same as A1 but with `--n_layer 8 --hidden_size 480 --n_head 8 --intermediate_size 1920`

**Experiment A3: Only curriculum (no LAMB, no architecture change)**
Same as A2 but with `--optimizer adamw --learning_rate 1e-3`

**Decision logic:**
- A1 vs A2 → measures architecture effect (12×384 vs 8×480)
- A2 vs A3 → measures LAMB effect
- A3 vs baseline → measures curriculum effect alone
- If A1 or A2 closes gap to ≤0.2 of leader, proceed to full eval
- If <0.3 improvement, also need data track

### Track B: AoA implementation (HIGH PRIORITY — potential +2.5 Overall)
The only entry with positive AoA (22.9) is `deberta-base-75k-sam_ext-s1`. If we can achieve even AoA=+5, that's +0.56 Overall.

**Requirements:**
1. Save 19 checkpoints at: 1M, 2M, ..., 9M, 10M, 20M, 30M, ..., 100M word exposures
2. Each checkpoint as HF model under `hf_model/chck_{X}M/`
3. Run `eval_aoa` from the official evaluation pipeline
4. Ensure surprisal keys match the current accepted format

**Implementation:**
- Our trainer already saves checkpoints via `--checkpoint_words 1000000`
- Need to verify checkpoint naming matches `chck_1M`, `chck_2M`, etc.
- Need to verify the HF model format is compatible with official AoA eval
- Need to run AoA evaluation separately

**Quick test:** Run AoA eval on existing COMPACT_EXPERIENCE checkpoints to check format compatibility.

### Track C: Data enhancement (MEDIUM PRIORITY — after Track A results)
If curriculum alone doesn't close the gap, the data difference matters.

**Option C1: Simplification pairs from BabyLM corpus**
- Instead of near-length paraphrases, generate complex→simple rewrites
- This compresses the generated text, allowing more unique pairs
- Aligns with the information-efficient second-view hypothesis

**Option C2: FineWeb simplification pairs**
- The leader uses FineWeb as source data (not BabyLM)
- FineWeb has richer world knowledge → explains better EWoK/GlobalPIQA
- But may hurt our Supplement advantage (which comes from linguistic diversity)

**Option C3: Hybrid data**
- BabyLM corpus pairs for linguistic diversity (maintains Supplement edge)
- FineWeb pairs for world knowledge (improves EWoK/GlobalPIQA)
- Requires careful budget allocation within 10M words

### Track D: Tail averaging (LOW PRIORITY)
COMPACT_EXPERIENCE tested tail SWA on 7 zero-shot columns: essentially neutral (delta ≈ -0.06).
BabySteps_MurphysLaw reports +1.6 SuperGLUE from tail averaging. Since our SuperGLUE is already above the leader (70.31 vs 69.79), this is low priority.

## Execution order

1. **Now (component gap analysis and leader reverse engineering Explore)**: Write enhanced trainer, experiment config, research notes
2. **exposure dynamics audit (Execute)**: Run Track A experiments (A1, A2, A3) on 2 GPUs
3. **exposure dynamics audit parallel**: Run AoA format test on existing checkpoints
4. **repaired pilot eval**: Evaluate A experiments, decide on data track
5. **mechanism preservation summary+**: Full eval on best candidate, data enhancement if needed

## Quick discriminating pilot design

Before committing to full 100M-exposure runs, run 10M-exposure (1 epoch) pilots:
- ~15-20 min each on H100
- Compare final loss and fast-eval proxy metrics
- This tells us which recipe changes actually help before burning hours on full runs

## Files to produce

1. `scripts/masking_curriculum_trainer_v2.py` — enhanced trainer with LAMB
2. `scripts/lamb_optimizer.py` — LAMB optimizer (copy from COMPACT_EXPERIENCE)
3. `notes/component_gap_analysis_and_leader_reverse_engineering.md` — done
4. `plans/construction_plan.md` — this file
5. Experiment launch scripts (in Execute phase)

## Key metrics

For 10M pilot: loss_last, BLiMP-fast, Supplement-fast, EWoK-fast, Entity-fast
For full eval: all 9 columns (BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, SuperGLUE, Reading, AoA)
Target: Overall ≥ 41.8
