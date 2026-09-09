# paired continuation training done — Paired continuation from INITIAL_MODEL_STUDIES chck_70M

## Experimental design

**Problem**: b256 fixedseq pair eval raw showed that loading chck_70M and continuing with token-level masking
cannot be compared against the INITIAL_MODEL_STUDIES original chck_100M because:
1. No optimizer state is available in chck_70M (only model + tokenizer)
2. A single continuation arm confounds optimizer restart with granularity change

**Matched control**: Run BOTH arms from the same 70M weights with identical
optimizer restart, LR phase, and data order. Only the masking granularity differs.

### Shared conditions (both arms):
- Model weights: INITIAL_MODEL_STUDIES earlier analysis `chck_70M` (DeBERTa-v2 8×480, 34.5M params)
- Optimizer: fresh AdamW (lr=1e-3, wd=0.01, betas=(0.9, 0.98))
- LR schedule: cosine(warmup=122, total=2442) advanced to earlier analysis → starts at LR≈0.000227
- Data: exact post-70M example subsequence from INITIAL_MODEL_STUDIES manifest (187,500 examples = 30M words)
- Batch: 256, seq: 256, mask_prob: 0.15
- Masking RNG seed: 43
- Steps: 733

### Treatment:
- Arm A (control): mask_mode = wwm (whole-word masking)
- Arm B (treatment): mask_mode = token (token-level masking)

### Interpretation protocol:
1. **Primary signal**: token_score - wwm_score on each column isolates the granularity effect
2. **Secondary**: compare both arms to INITIAL_MODEL_STUDIES earlier analysis original chck_100M (which ran fixed WWM
   for the full 100M without optimizer restart) to estimate the restart penalty
3. If token - wwm is broadly positive → token-level granularity genuinely helps in late training
4. If token - wwm is broadly negative → the simple switch is harmful regardless of restart
5. If mixed (gains on some columns, losses on others) → conditional granularity is needed

### Checkpoints saved:
- chck_80M, chck_90M, chck_100M for each arm

### Background task:
- Expected duration: ~85 minutes
- Both GPUs active simultaneously

### Evaluation plan:
Run `eval_paired_continuation.py --mode all` on both arms + INITIAL_MODEL_STUDIES reference
using BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading (earlier analysis-style).
