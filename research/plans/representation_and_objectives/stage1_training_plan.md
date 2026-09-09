# stage0 assessment: Stage 1 training plan for S/G marginal factorization screen

## Purpose

Test whether semantic selection (which content words from the source are retained)
produces measurably different masked-denoising quality compared to generic extractive
compression at matched content/function ratio. This is the factorization's second
decision point after the Stage 0 audit passed.

## Arms

- **S (semantic extractive)**: source + rewrite-guided order-preserving source extract
- **G (generic extractive)**: source + content-balanced generic source extract

Both arms share the same filler (9,576,489 words), pair-block geometry (423,511 words
across 3,005 rows), tokenizer, and source text. Only the side-view words differ.

## Training streams

- S: `experiments/archive/representation_and_objectives/data/training_streams/arm_s_100M_stream.jsonl`
  SHA256: `f14fd6812c97e0b4f5809848f79895820e0d7fe65be208783c8c2c3f9c5d44f7`
- G: `experiments/archive/representation_and_objectives/data/training_streams/arm_g_100M_stream.jsonl`
  SHA256: `0d736ecba50e41a146a394ff8bb137df6ba1414d2aab1ec4dbbbfc862dd746c7`

## Training configuration (matches historical triangle)

```
python3 -B experiments/archive/representation_and_objectives/scripts/gradient_checkpointed_masking_curriculum_trainer.py \
  --output_dir experiments/archive/representation_and_objectives/training/runs/arm_{s,g}_20M_screen \
  --example_jsonl experiments/archive/representation_and_objectives/data/training_streams/arm_{s,g}_100M_stream.jsonl \
  --example_jsonl_label arm_{s,g}_extract_reinvest \
  --tokenizer_path experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M \
  --tokenizer_label legal16k \
  --max_word_exposure 20000000 \
  --example_pool_words 100000000 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --warmup_fraction 0.06 \
  --weight_decay 0.01 \
  --lr_total_steps 2529 \
  --seed 43 \
  --extra_init_seed 43022 \
  --train_rng_seed 43023 \
  --masking_curriculum wwm_fixed \
  --mask_prob_start 0.15 \
  --mask_prob_end 0.15 \
  --seq_length 256 \
  --max_seq_length 256 \
  --hidden_size 480 \
  --n_layer 8 \
  --n_head 8 \
  --checkpoint_words 1000000
```

Key settings:
- **lr_total_steps=2529**: preserves the 100M-equivalent LR schedule at any horizon
- **max_word_exposure=20000000**: short screen (20M words ≈ 578 updates)
- **batch_size=256**: full optimizer batch via activation checkpointing
- **Seeds 43/43022/43023**: matches historical triangle exactly

## Launch plan

Run S and G arms in parallel on the two H100s:
- cuda:0 → S arm (20M)
- cuda:1 → G arm (20M)

Expected time: ~25 min per arm based on crossview v2 20M result timings at similar scale.

## Readout

### Primary signal: overall training loss comparison
Compare final MLM loss at 20M between S and G. If the difference is negligible (< 0.01),
the semantic selection question is not resolvable at this horizon.

### Secondary signal: cheap7 evaluation (only if loss differs)
If loss differs meaningfully, run the fast evaluation battery at 20M checkpoints
to check whether the loss difference maps to downstream competence.

### Interpretation
- **S ≈ G at 20M**: Semantic selection doesn't differentiate; the compact-view benefit
  comes from reformulation/coverage rather than which content words are selected.
  Consider extending to 40M or adding C arm before closing.
- **S > G at 20M**: Semantic selection matters even with matched content/function ratio.
  Proceed to 100M endpoint evaluation and item-level transition analysis.
- **G > S at 20M**: Generic coverage beats semantic targeting. Surprising but informative;
  check whether G's slightly better position uniformity drives this.

## Relation to historical triangle

The existing triangle has 100M endpoints:
- compact_view_reinvest: equal7 44.2886
- adjbreak_reinvest: equal7 43.0557
- compact_repeat_reinvest: equal7 41.8721

S and G represent a factorization of the compact view's advantage:
- If S ≈ compact_view at 100M: semantic selection suffices, reformulation is secondary
- If S ≈ adjbreak: the adjbreak gain was already semantic selection
- If S ≈ repeat: selection alone doesn't help; the gain requires reformulation

These comparisons are only valid at 100M and should not be made from the 20M screen.

## Files

- Assessment: `notes/stage0_assessment.md`
- Corpora: `data/marginal_corpora_v2/marginal_corpora_csg_v2.jsonl`
- Streams: `data/training_streams/`
- Audit: `data/marginal_corpora_v2/marginal_corpora_v2_audit.json`
