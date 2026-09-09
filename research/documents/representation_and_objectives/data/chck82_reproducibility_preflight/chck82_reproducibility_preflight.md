# earlier analysis chck_82M reproducibility preflight

Status: **PASS**
Original chck_82M model SHA256: `93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3`
endpoint sweep and dualview route Overall: **41.94225421386936**; SuperGLUE: **69.7602879248243**; cheap7: **43.96**
10M pool SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`; words/rows: 10000000/64740
100M stream SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`; words/rows: 100000000/647400
Tokenizer vocab maps identical: **True**
Fixed selection rule: evaluate `chck_82M` only; no reproduction checkpoint search.
Reproduction run dir: `experiments/archive/representation_and_objectives/training/runs/adapter128_scale1p75_from_corpus_fixed82M_seed43022`

Training command (GPU0):
```bash
python \
  -B \
  experiments/archive/frontier_consolidation/scripts/adapter_scaled_trainer.py \
  --adapter_bottleneck \
  128 \
  --adapter_enabled \
  1 \
  --adapter_scale \
  1.75 \
  --gpu \
  0 \
  --example_jsonl \
  experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl \
  --example_jsonl_label \
  repro_adapter128_scale1p75_matched_100Mhorizon_fixed82M \
  --example_jsonl_meta \
  experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json \
  --output_dir \
  experiments/archive/representation_and_objectives/training/runs/adapter128_scale1p75_from_corpus_fixed82M_seed43022 \
  --tokenizer_path \
  experiments/archive/frontier_consolidation/data/compliant_tokenizer \
  --tokenizer_label \
  compliant16k_reinvest10M \
  --hidden_size \
  480 \
  --n_layer \
  8 \
  --n_head \
  8 \
  --ffn_mult \
  4 \
  --seed \
  43 \
  --extra_init_seed \
  43022 \
  --train_rng_seed \
  43023 \
  --batch_size \
  256 \
  --seq_length \
  256 \
  --max_seq_length \
  256 \
  --learning_rate \
  0.001 \
  --warmup_fraction \
  0.06 \
  --weight_decay \
  0.01 \
  --masking_curriculum \
  wwm_fixed \
  --mask_prob_start \
  0.15 \
  --mask_prob_end \
  0.15 \
  --checkpoint_words \
  1000000 \
  --max_word_exposure \
  100000000 \
  --lr_total_steps \
  2529 \
  --num_workers \
  0 \
  --log_every \
  100 \
  --dynamics_trace_every \
  500
```

JSON: `experiments/archive/representation_and_objectives/data/chck82_reproducibility_preflight/chck82_reproducibility_preflight.json`
