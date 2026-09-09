# crossview denoising design — Cross-View Conditional Denoising Experiment

## Scientific question
Does the clean-Qwen same-window paired-experience benefit come from genuine **semantic 
cross-view reconstruction** (recovering meaning from a partner's different-but-equivalent 
expression), or merely from **lexical redundancy** (copying identical tokens visible in 
the partner view)?

## Controlled design
Shared anchors (normalized words common to original and rewrite) *locate* aligned regions.
The scientifically decisive targets are non-identical paraphrastic predicates/attributes 
whose recovery requires the partner view — not the anchors themselves.

### Arms (all matched: 15% WWM budget, same corpus/tokenizer/model/optimizer/seed43022)
| Arm | Pair-row masking | What it tests |
|-----|------------------|---------------|
| `baseline` | Standard random WWM | Clean-Qwen recipe control |
| `anchor_copy` | Preferentially mask content anchors (shared content words, one view) | Model can potentially copy from visible identical token in partner |
| `semantic_xview` | Preferentially mask non-anchors (paraphrastic targets, one view) | Model must reconstruct from partner's different expression |
| `partner_shuffle` | Same targeting as semantic_xview, partner rewrite replaced by random other pair's rewrite | Breaks cross-view correspondence; gain should vanish |

### Key mechanistic controls
1. **View selection**: each pair row randomly masks ONE view (original or rewrite) while 
   keeping the partner view intact. This forces directional inference.
2. **Budget-exact matching**: base Bernoulli selection is drawn once; priority arms only 
   swap non-target→target words of matched subword length. Total masked tokens identical 
   row-by-row across arms.
3. **Partner shuffle**: identical targeting logic but partner text is replaced by 
   length-matched derangement (37,592 of 37,594 pairs changed). If semantic_xview gains 
   vanish under shuffle, the mechanism is confirmed as cross-view semantic correspondence.

### Structure statistics (from 80k-word smoke, representative)
- Pair rows: ~19% of corpus rows by count
- Anchor fraction in pair rows: 68.3% (most content is shared)
- Content anchor fraction: 33.0% (proper nouns, specific verbs, numbers)
- Non-anchor (paraphrastic target) fraction: 31.7%
- After partner shuffle: anchor fraction drops to 14.1%, content anchors to 0.3%

## Screen parameters (20M/b256)
- Corpus: qwen_aligned_100M.jsonl, first 128,762 rows (exact 20M words)
- Tokenizer: baseline16k
- Model: DeBERTa-v2 8×480, 34,467,424 parameters
- Batch: 256, seq: 256, LR: 1e-3, cosine with warmup 5%, lr_total_steps=2515
- Seed: 43022 (identical model initialization across all arms)
- Checkpoints: every 1M words (chck_1M through chck_20M)

## Decision tree
1. **If semantic_xview > baseline on Entity/EWoK without Supplement/Reading collapse:**
   - And anchor_copy ≤ baseline: confirms semantic reconstruction is the mechanism
   - Promote semantic_xview to full 100M/b256 run with complete nine-column measurement
   - Launch partner_shuffle as mechanism confirmation arm
   
2. **If anchor_copy > baseline but semantic_xview ≤ baseline:**
   - Lexical copying is the mechanism, not semantic reconstruction
   - Investigate whether copy pressure improves general representation
   - Consider reduced-dose anchor masking or anchor-prediction auxiliary
   
3. **If both treatments ≤ baseline:**
   - Priority-targeted masking itself is not the mechanism
   - The same-window benefit comes from data-level coexistence/diversity, not masking geometry
   - Redirect to corpus composition, pair mixing ratio, or architecture interventions

4. **If semantic_xview ≈ anchor_copy > baseline:**
   - Both benefit from some form of view-conditioned reconstruction
   - The distinguishing factor is not semantic vs lexical
   - Need finer dissection (rare-word anchors only, entity anchors only, etc.)

## Files
- Trainer: `scripts/crossview_denoising_trainer.py`
- Launch: `scripts/launch_crossview_20M_screen.sh`
- Eval: `scripts/launch_crossview_noaoa_eval.sh`
- Summarizer: `scripts/summarize_crossview_noaoa.py`
- Runs: `training/runs/crossview_{baseline,anchor_copy,semantic_xview}_20M_seed43022/`

## Connection to prior evidence
- Clean-Qwen Overall 41.3443 came from same-window correspondence (clean qwen control interpretation and next mechanism controls)
- Separated-pair lost 1.78 Overall → adjacency in same attention window matters
- Duplication lost 0.35 Overall → generated second view adds beyond repetition
- This experiment dissects WHAT the model learns from that second view: copying shared 
  content (lexical redundancy) vs inferring meaning from different expressions (semantic)
