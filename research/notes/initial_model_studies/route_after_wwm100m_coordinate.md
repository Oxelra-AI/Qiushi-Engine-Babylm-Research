# route after wwm100m coordinate — Route after the first real WWM100M coordinate

## What the 100M coordinate changed

The validated full-cycle protected WWM run is now a real evidence object:

- run: `training/runs/babylm_fullcycle_wwm_seed42_100M/`
- model: BERT MLM, 8 layers, hidden 256, 8 heads, 10.73M parameters
- data: official 10M words, 10 epoch-wise passes, 100M word exposure
- objective: WWM, baseline16k, fixed length 256
- full-cycle validation: 1221 steps, loss 9.76→5.50, 100 loadable checkpoints

Available official full-eval coordinate from wwm100m available official coordinate:

| column/task | WWM100M score |
|---|---:|
| BLiMP | 55.92 |
| BLiMP Supplement | 52.03 |
| Entity Tracking | 16.76 |
| COMPS | 51.66 |
| GlobalPIQA parallel | 17.48 |
| GlobalPIQA nonparallel | 48.00 |
| GlobalPIQA mean | 32.74 |
| Reading eye | 11.68 |
| Reading self-paced | 3.82 |
| EWoK | missing locally |
| AoA | not run yet |
| (Super)GLUE | not run yet |

This ends the route “small 8x256 BERT + ordinary WWM + more exposure will catch SOTA.” Full exposure did not close the main gaps. Entity remains 16.76 versus the visible top row 28.45, BLiMP is 55.92 versus 67.20, and GlobalPIQA mean is 32.74 versus 39.67. The GlobalPIQA parallel score (17.48) is especially low while nonparallel is 48.0.

But this does **not** prove the gap is “DeBERTa backbone alone.” The leader bundles at least larger capacity, DeBERTa-v2, data/curriculum/simplification, tokenizer, and optimizer changes. Earlier 1M parameter-matched DeBERTa-v2 at ~10.7M did not show a broad gain over BERT-WWM. The next route must separate capacity from the DeBERTa-v2 component at a SOTA-class scale.

## Interpretation checks

Four scientific constraints follow:

1. A/B comparison can estimate:

   - capacity effect: 10.7M BERT → ~34M BERT;
   - 34M DeBERTa-v2 package effect: ~34M BERT → ~34M DeBERTa-v2.

   It cannot by itself identify a single “disentangled attention” cause, because DeBERTa-v2 changes relative position, content-position interaction, parameter allocation, and implementation details together.

2. Parameter count, non-embedding count, sequence length, tokenizer, mask realization, epoch shuffles, optimizer schedule, and checkpoint/evaluation protocol must be matched and recorded. Word exposure alone does not equal compute exposure.

3. 256-token truncation is substantial: the WWM100M tokenization summary has 29.2% examples truncated at length 256. Length 512 may matter for Entity/GlobalPIQA, but changing length at the same time as architecture would mix context retention with model mechanism. Initial A/B should keep length fixed at 256 for attribution, then a common 512 length comparison can be run if needed.

4. GlobalPIQA parallel 17.48 must be probed without training before making it a training target. It may reflect the MLM scorer’s interaction with candidate order, token length, joint option format, or normalization. Official scoring must remain unchanged, but the research should inspect raw margins, answer length, option order, and parallel/nonparallel differences.

## Feasible ~34M arms with current trainer

The full-cycle trainer already supports `--model_type bert` and `--model_type deberta_v2`.

Parameter probe: `scripts/probe_34m_param_configs.py`.

Best immediate arms:

| arm | model_type | layers | hidden | heads | head dim | total params | embedding params | non-embedding params | interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A: BERT-capacity | bert | 8 | 512 | 8 | 64 | 34,151,936 | 8,388,608 | 25,763,328 | isolates most of the capacity jump from 10.7M BERT under same WWM/data/tokenizer/length |
| B: DeBERTa-scale | deberta_v2 | 8 | 480 | 8 | 60 | 34,467,424 | 7,864,320 | 26,603,104 | tests a ~34M DeBERTa-v2 package at similar total and non-embedding scale |

The A/B geometry is not identical (512 vs 480 hidden) because the HF DeBERTa-v2 and BERT parameter allocation differ. Therefore the result should be written as “capacity vs 34M DeBERTa-v2 package,” not a pure disentangled-attention result.

If B strongly beats A, the next mechanism-dissection comparison should include one or both of:

- geometry-matched 8x480 BERT (30.53M) vs 8x480 DeBERTa-v2 (34.47M), to test same hidden/layer layout with a parameter difference recorded;
- DeBERTa-v2 with relative attention disabled or with only relative-position pieces changed, to locate which DeBERTa component carries the effect.

Do not spend this first 34M full-scale run on 40k tokenizer, simplification pairs, LAMB, or curriculum. Those are leader-bundle factors and can be added after we know whether capacity and DeBERTa-v2 close any of the measured gaps.

## Exact first 34M comparison

Use the same official corpus and full-cycle setup as state probe and fullcycle wwm launch/74:

- official corpus revision already used in full-cycle trainer
- 100,000,000 word exposure = 10 passes over 10M official words
- WWM, `mask_prob=0.15`
- baseline16k tokenizer
- fixed `seq_length=256`, `max_seq_length=256`, `max_position_embeddings=512`
- same seed/init/RNG convention, first seed 42/456/789
- same epoch shuffle schedule where possible
- same `batch_size=512` unless memory requires reduction; if batch changes, keep 1221 optimizer updates and record effective words/update
- same `learning_rate=0.001` and cosine schedule over 1221 steps for the first controlled run
- checkpoint every 1M words

Proposed full-scale A/B comparison: one arm per H100 on two GPUs.

A scientific arguments for `experiments/archive/initial_model_studies/training/scripts/babylm_masked_train_fullcycle.py` (configured time limit: 12000 seconds):

```text
  --output_dir experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_bert8x512_wwm_seed42_100M \
  --max_word_exposure 100000000 --example_pool_words 10000000 --checkpoint_words 1000000 \
  --model_type bert --hidden_size 512 --n_layer 8 --n_head 8 --ffn_mult 4 \
  --tokenizer_label baseline16k --tokenization_summary_limit 0 \
  --mask_mode wwm --mask_prob 0.15 \
  --seq_length 256 --max_seq_length 256 --max_position_embeddings 512 \
  --batch_size 512 --lr_total_steps 1221 --learning_rate 0.001 \
  --seed 42 --extra_init_seed 456 --train_rng_seed 789 --log_every 25
```

B scientific arguments for `experiments/archive/initial_model_studies/training/scripts/babylm_masked_train_fullcycle.py` (configured time limit: 12000 seconds):

```text
  --output_dir experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M \
  --max_word_exposure 100000000 --example_pool_words 10000000 --checkpoint_words 1000000 \
  --model_type deberta_v2 --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 \
  --position_buckets 256 --max_relative_positions 256 --deberta_relative_attention true --deberta_pos_att_type p2c,c2p \
  --tokenizer_label baseline16k --tokenization_summary_limit 0 \
  --mask_mode wwm --mask_prob 0.15 \
  --seq_length 256 --max_seq_length 256 --max_position_embeddings 512 \
  --batch_size 512 --lr_total_steps 1221 --learning_rate 0.001 \
  --seed 42 --extra_init_seed 456 --train_rng_seed 789 --log_every 25
```

Before launching both at full scale, run a very small compile/load smoke if the 34M DeBERTa memory path has not yet been checked. Do not spend another step on 1M route screening for these arms; the goal is to create real full-scale comparators.

## Complete the WWM100M coordinate in parallel

Do not let EWoK block the other missing columns.

Immediate available columns:

- AoA can be run from `full_eval/aoa/cdi_childes.json` with the official AoA script or direct Python equivalent.
- (Super)GLUE can be run from `full_eval/glue_filtered/*` using `scripts/eval_finetuning.sh`. This will be slower and should be a background task; save raw `results/` and parsed aggregate.
- EWoK full remains missing because local `full_eval/ewok_filtered/` is empty; official `download_evals.py` did not populate it, and `ewok-core/ewok-core-1.0` is gated. Continue searching official sources, but if it remains unavailable, keep using fast EWoK only as an explicitly marked interim source, not a full-eval substitute.

## GlobalPIQA parallel probe before designing C

Before training an arm specifically for GlobalPIQA, analyze existing WWM100M predictions and reports:

- inspect `predictions.json` for parallel and nonparallel;
- compute candidate margin by item;
- compare answer length and token count for correct vs chosen option;
- check whether prediction follows candidate position or label ordering;
- compare sum log-probability vs per-token normalized scoring if raw fields allow it;
- bucket errors by option length, overlap with prompt, and answer position.

If the 17.48 parallel score is mostly a scorer-format interaction, the scientific target is format-robust ranking, not “physical commonsense” alone. Official scoring must not be changed, but the mechanism should be designed against the real failure mode.

## Arm C should not be launched blindly yet

The proposed C mechanisms are:

1. feedback entity/role binding in the main attention path;
2. structured entity-relation span masking plus cross-view consistency on official data;
3. within-sequence object-file slots if binding can be learned but repeated updates fail.

The immediate C candidate, once A/B and the GlobalPIQA probe are underway, is **structured entity-relation span masking plus consistency** on the A or B 34M backbone. It targets repeated entities, pronouns, predicate-argument spans, and relation-preserving format changes using official corpus text only. It is distinct from the failed routes:

- not persistent cross-document memory;
- not small-BERT char/morph side channels;
- not WikiAuto adjacency;
- not low-budget hand-coded state stories;
- not a copy of leader simplification/curriculum/tokenizer.

Construction of C depends on whether the GlobalPIQA probe localizes parallel collapse to labels, format or length. The first execution priority is A/B full-scale plus coordinate closure.

## Next immediate work

1. Proposed validation: 34M smoke/load checks for A and B if needed, followed by full-cycle A/B 100M training, ideally concurrently.
2. Required measurements: AoA and (Super)GLUE for the validated WWM100M baseline, independently of pending EWoK evaluation.
3. Proposed diagnostic: the GlobalPIQA prediction-level probe on WWM100M parallel/nonparallel outputs.
4. Interpretation requires A/B first coordinates or completed AoA/SuperGLUE; partial columns alone cannot establish a SOTA route.
