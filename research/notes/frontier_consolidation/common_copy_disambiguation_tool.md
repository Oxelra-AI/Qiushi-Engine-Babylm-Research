# common copy disambiguation tool — Common-copied no-disentangle trainer wrapper (disambiguation tool, verified, not launched)

## Why

initialization parity and interaction interpretation established that the running earlier analysis four-cell interaction

    I = (compact - repeat)_full_DeBERTa  -  (compact - repeat)_no_disentangle_abs

is a **valid** architecture-coordinate data-treatment interaction (initialization
cancels additively inside each compact-minus-repeat), but it is **bundled**: the
`no_disentangle_abs` cell is not "full minus only c2p/p2c score terms" — omitting the
32 positional projection tensors shifts the torch construction RNG stream, so 53/140
same-named tensors differ at init, and capacity drops 34,467,424 → 30,773,344.

Therefore, if the delivered earlier analysis result is scientifically important but ambiguous
(e.g. moderate attenuation of compact-minus-repeat that could be blamed on init/capacity
rather than the removed score channel), the clean disambiguation is a no-disentangle
run whose every common tensor is copied from the **same full DeBERTa initialization**.

## Tool

`scripts/common_copy_nodis_trainer_wrapper.py`

- Imports the exact compliant tokenizer retrain status/COMPACT_EXPERIENCE `masking_curriculum_trainer.py` and monkeypatches
  **only** `build_model`. Everything else — data loading, WWM p=0.15 masking, AdamW,
  cosine schedule, train_rng reset, checkpointing every 10M, `scientific_metrics.json` —
  is the unchanged base loop. A real run therefore still produces the same metrics the
  architecture interaction readout repair integrity reader validates.
- Inside the patched `build_model`, after the base trainer has reset RNG to the normal
  init seeds and before it resets `train_rng_seed`, it: builds the full `p2c,c2p` source,
  builds the `[]` target, copies every same-shaped common tensor from source→target,
  returns the target. The source is discarded (init donor only).
- `--common-copy-init-dry-run` writes an initialization record without entering training.

## CPU dry-run verification

Both legal streams, spatial repair route status tokenizer (SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`),
exact compliant tokenizer retrain status recipe (seeds 43/43022/43023, WWM 0.15, batch256/seq256, lr0.001, warmup0.06,
lr_total_steps 2529, checkpoint_words 10M, 100M exposure):

- compact: `data/common_copy_wrapper_compact_dryrun/common_copy_initialization.json`
- repeat:  `data/common_copy_wrapper_repeat_dryrun/common_copy_initialization.json`

Results (identical structure for both arms):
- before copy: common 140, **nonexact 53**, source keys 172, target keys 140, 32 source-only
  tensors are exactly the `pos_key_proj`/`pos_query_proj` per-layer weights+biases.
- after copy: common 140, **nonexact 0** (140/140 exact).
- target parameter count **30,773,344** (matches the running no-disentangle cell).
- initial-logit gap full source vs common-copied target: mean_abs **≈0.00243–0.00248**,
  i.e. only the removed positional-score path remains, as intended.

## Exact future command (only if warranted)

Run once per data arm (GPU0/GPU1 in parallel), no `--common-copy-init-dry-run`:

    python3 -B scripts/common_copy_nodis_trainer_wrapper.py \
      --common-copy-source-pos-att-type p2c,c2p --common-copy-target-pos-att-type '' \
      --example_jsonl <legal compact|repeat 100M jsonl> \
      --example_jsonl_label common_copy_nodis_<arm> \
      --output_dir training/runs/common_copy_nodis_abs_<arm>_deberta100M_seed43022 \
      --tokenizer_path data/compliant_tokenizer \
      --tokenizer_label compliant16k_reinvest10M \
      --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 --deberta_pos_att_type '' \
      --seed 43 --extra_init_seed 43022 --train_rng_seed 43023 \
      --batch_size 256 --seq_length 256 --max_seq_length 256 \
      --learning_rate 0.001 --warmup_fraction 0.06 --weight_decay 0.01 --lr_total_steps 2529 \
      --masking_curriculum wwm_fixed --mask_prob_start 0.15 --mask_prob_end 0.15 \
      --checkpoint_words 10000000 --max_word_exposure 100000000 --example_pool_words 100000000 \
      --num_workers 0 --log_every 50 --dynamics_trace_every 200

This removes the RNG-stream confound (common init). Capacity/parameter-count difference
(−3,694,080) still remains and must be stated. `c2p_only` vs `p2c_only` (initialization parity and interaction interpretation: bit-identical
init, 32,620,384 params) remains the clean directional lens for the narrower question.

## Boundary

CPU-only build verification. No H100 training launched. No selected evaluation, SuperGLUE,
AoA, upload, or leaderboard submission. The two earlier analysis tasks were not polled or disturbed.
Launch this wrapper only if the delivered earlier analysis interaction is important but ambiguous.
