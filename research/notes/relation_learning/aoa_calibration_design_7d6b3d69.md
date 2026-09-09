# aoa calibration design AoA calibration design — corrected and complete

## Experimental design

Three 30M-word training arms, all adapter-faithful (AdapterDebertaV2ForMaskedLM,
adapter bottleneck 128, scale 1.75, 35,463,008 parameters), sharing seed 43112
(different from v4's 43022) and identical hyperparameters except where noted:

1. **Control** (running at this stage): v4-order stream, uniform 0.15 WWM masking
   - `aoa_control/`
   - max_word_exposure=30,000,000 (exact match to stream)

2. **Schedule arm** (running at this stage): reordered stream (CHILDES-enriched early),
   uniform 0.15 WWM masking
   - `aoa_schedule_arm/`
   - max_word_exposure=30,000,051 (exact match to stream, 51-word excess)
   - Same seed/init/LR/masking as control; only row ordering differs

3. **Enrichment arm** (pending GPU): v4-order stream, enrichment-weighted WWM masking
   - `aoa_enrichment_arm/` — not yet launched
   - `enrichment_masking_wrapper.py` hooks masking probabilities into the
     proven earlier analysis adapter wrapper
   - Same seed/init/LR/model as control; only masking probabilities differ
   - alpha_start=1.5, linear taper to 0 by 0.67×2529=1694 steps
   - Corpus-occurrence-weighted normalization: expected mask rate = 0.15 exactly

## Key measurements

**arm − control** = pure intervention effect (same seed, same model init)
**control − v4 ladder at matched checkpoints** = seed/trainer noise floor

### β regression
For each CDI word w and checkpoint c:
  Δsurprisal_{w,c} = a_c + β × Δcredit_{w,c} + ε_{w,c}

where a_c are per-checkpoint intercepts that absorb global competence shifts,
and β is the common exposure→surprisal slope across arms.

Schedule arm: Δcredit = 0.15 × (cumul_occ_schedule − cumul_occ_v4)
Enrichment arm: Δcredit = cumul_enrichment_credit − 0.15 × cumul_occ_v4
  (time-varying because alpha tapers)

## Legality note

The enrichment z-scores are derived SOLELY from Strict-Small corpus statistics:
  z_t = (log_freq_CHILDES_t − log_freq_whole_stream_t) / normalizing_std

This measures how over- or under-represented a token is in the CHILDES portion
relative to the full training corpus. It is a corpus-internal frequency ratio.

NO child AoA labels, CDI acquisition ages, or external child-development data
enter the training recipe. The CDI word list is used only for EVALUATION
(matching the official AoA evaluator's procedure), never for training.

The oracle schedules tested in earlier analysis (which used child AoA labels for row
ordering) are illegal training recipes and serve only as upper bounds on the
measurement instrument. The legal enrichment arm uses only corpus frequencies.

## Preregistered decision gates

After β is measured from both 30M arms:

1. **Recalibrated legal prediction r > 0.15**: Authorize 100M×2 preregistered run
   with AoA primary and all eight endpoint columns within coherent bands
2. **0.112 < r < 0.15**: At most one exploratory 100M seed  
3. **Both arms r < 0.112**: Close AoA route; reassess dense or new credit mechanism

β concordance across arms confirms "acquisition order follows cumulative credited
exposure" as a unified principle. β discordance reveals which lever (timing vs credit)
is operative and informs the 100M design.

## Artifacts

- Streams: `experiments/archive/relation_learning/data/aoa_calibration_prep`
- Corpus freq: `experiments/archive/relation_learning/data/corpus_token_freq/corpus_token_freq.json`
- Enrichment weights: `experiments/archive/relation_learning/data/aoa_calibration_prep/token_enrichment_weights.json`
- Enrichment wrapper: `experiments/archive/relation_learning/scripts/enrichment_masking_wrapper.py`
- Exposure diffs: `experiments/archive/relation_learning/scripts/precompute_exposure_differences.py`

## Enrichment arm launch command (ready for the first free GPU)

```bash
python -B experiments/archive/relation_learning/scripts/enrichment_masking_wrapper.py \
  --adapter_bottleneck 128 --adapter_enabled 1 --adapter_scale 1.75 --gpu <GPU> \
  --enrichment_weights experiments/archive/relation_learning/data/aoa_calibration_prep/token_enrichment_weights.json \
  --enrichment_corpus_freq experiments/archive/relation_learning/data/corpus_token_freq/corpus_token_freq.json \
  --enrichment_alpha_start 1.5 --enrichment_taper_frac 0.67 \
  --example_jsonl experiments/archive/relation_learning/data/aoa_calibration_prep/v4_order_30M.jsonl \
  --example_jsonl_label aoa_enrichment_arm_seed43112 \
  --output_dir experiments/archive/relation_learning/data/aoa_enrichment_arm \
  --tokenizer_path experiments/archive/frontier_consolidation/data/compliant_tokenizer \
  --tokenizer_label compliant16k_reinvest10M \
  --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 \
  --seed 43 --extra_init_seed 43112 --train_rng_seed 43113 \
  --batch_size 256 --seq_length 256 --max_seq_length 256 \
  --learning_rate 0.001 --warmup_fraction 0.06 --weight_decay 0.01 \
  --masking_curriculum wwm_fixed --mask_prob_start 0.15 --mask_prob_end 0.15 \
  --checkpoint_words 1000000 --max_word_exposure 30000000 \
  --lr_total_steps 2529 --num_workers 0 --log_every 1 --dynamics_trace_every 100
```

Verify immediately after launch:
- stdout.log shows adapter_scaled_model_built with total_params=35463008
- enrichment_mask events show eff_rate near 0.15
- earlier analysis loss near 9.8
