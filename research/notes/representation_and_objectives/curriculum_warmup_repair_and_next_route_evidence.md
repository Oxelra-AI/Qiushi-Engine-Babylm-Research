# curriculum warmup repair and next route evidence — curriculum warmup repair and next-route evidence

## Why the curriculum parity probe run was cancelled

The curriculum parity probe trainer itself passed a strong fixed-256 parity probe against the trusted legal40k legal40k accum training completion trainer, but the launch command still differed from the matched baseline in one hyperparameter that affects the entire learning-rate path:

- legal40k accum training completion legal40k baseline manifest (`data/legal40k_accum_training/legal40k_accum_train_manifest_seed43022.json`) records `warmup_fraction=0.06`.
- curriculum parity probe submitted command ( parity probe/turn_0020/tool_0001.json`) used `--warmup_fraction 0.05`.

Because the active scientific question is whether word-boundary sequence-length curriculum alone improves the fixed-256 legal40k baseline, this difference would mix curriculum with a different LR schedule. I cancelled managed task `s134_t20_tool1` after it reached only phase 0 (~12.3M words). This is not evidence that curriculum is weak; it is an attribution repair.

Useful trace from the cancelled run:

- The run successfully constructed the corrected geometry: 647,400 rows, 100,000,000 charged words.
- Phase geometry with `min_chunk_tokens=1`: 64-token phase 499,280 chunks / 1,951 steps over 20M words; 128-token phase 400,830 chunks / 1,566 steps over 30M words; 256-token phase 323,700 chunks / 1,265 steps over 50M words; total 1,223,810 chunks / 4,782 steps.
- No tail drops were recorded.
- Model config was correct: 45,826,720 params, `pos_att_type=['p2c','c2p']`, `max_relative_positions=256`, `pad_token_id=3`, `bos=1`, `eos=2`.
- The cancelled task used warmup=239 (=5% of 4,782); the baseline-matched run must use warmup=286 (=6% of 4,782).

## Relaunched interpretable curriculum endpoint

Corrected curriculum training was launched:

```text
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python3 experiments/archive/representation_and_objectives/scripts/curriculum_trainer_c2bfc720.py \
  --example_jsonl experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl \
  --tokenizer_path experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k \
  --max_word_exposure 100000000 --checkpoint_words 10000000 \
  --curriculum '64:20000000,128:50000000,256:100000000' \
  --output_dir experiments/archive/representation_and_objectives/training/runs/curriculum_adamw_8x480_legal40k_seed43022_trainrng43023_warmup006 \
  --batch_size 256 --micro_batch_size 64 --learning_rate 0.001 --warmup_fraction 0.06 --weight_decay 0.01 \
  --init_seed 43022 --train_rng_seed 43023 --seed 43 --gpu 0 --amp 0 --log_every 100 --min_chunk_tokens 1 --pretokenize_log_every 50000
```

This is the first fully matched 8×480 AdamW word-boundary curriculum comparison against legal40k accum training completion fixed-256 baseline, apart from the intended sequence-length/step-cadence change. It should be read against:

- matched fixed-256 legal40k baseline cheap7: 43.107851816373675
- visible leader cheap7: ~43.77
- full Overall frontier: visible 41.80, but with our SuperGLUE/AoA profile a crossing would likely need cheap7 closer to 43.96–44.00.

Do not launch 12×384 curriculum until this result is read.

## Contaminated LAMB endpoint

The old lamb curriculum route decision LAMB+curriculum 8×480 run completed. Its artifacts confirm it is a contaminated joint-intervention endpoint, not clean LAMB or curriculum evidence:

- `scientific_metrics.json`: optimizer=LAMB, lr=0.007, warmup=0.06, total_steps=4,935, amp=true, params=42,133,600.
- `hf_model/chck_100M/config.json`: `pad_token_id=0`, `bos/eos=null`, `pos_att_type=null`, `max_relative_positions=-1`, params differ from the 45.8M matched legal40k coordinate.
- Phase geometry differs from the repaired trainer: 484,784 / 410,964 / 367,320 chunks (token slicing with drops/restarts) vs repaired 499,280 / 400,830 / 323,700 chunks.

The comparison launched cheap7 evaluation of this contaminated endpoint only to learn whether the joint package accidentally produces an actionable score. Interpret any score strictly under the lamb config audit contamination frame.

## Parallel-route source evidence gathered while GPUs run

Knowledge retrieval found primary sources for possible next mechanisms if the corrected curriculum endpoint does not cross:

1. `Exploring smaller batch sizes for a high-performing BabyLM model architecture` (`Knowledge/objects/papers/Exploring-smaller-batch-sizes-for-a-high-performing-BabyLM-model-archite--0839ca2870d7--3605ccc2ff57/object.md`, citation `\cite{loiciga2025exploring}`): studies batch-size/gradient-accumulation effects in BabyLM; useful because sequence curriculum changes optimizer-step cadence and early batch token geometry.
2. `FORGETTER with forgetful hyperparameters and recurring sleeps...` (`Knowledge/objects/papers/FORGETTER-with-forgetful-hyperparameters-and-recurring-sleeps-can-contin--f5a1fc3e23da--44b67c93bc36/object.md`, citation `\cite{rui2025forgetter}`): periodic optimizer/non-weight-state resets with high weight decay and small batches; relevant because Muon/LAMB evidence suggests trajectory and state variables matter.
3. Active curriculum/hybrid pretraining retrieval mostly returned the findings report and related curriculum sources, not yet a directly inspected method.

Do not turn these into H100 runs before reading the real source sections and designing cheap tests. The immediate next decision still depends on the corrected-curriculum and joint-intervention results.
