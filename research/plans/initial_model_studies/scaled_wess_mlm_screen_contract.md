# wess mlm transfer multiseed results — Scaled WESS-MLM short-budget screen contract

## Why this screen is justified

wess mlm transfer positive signal produced the first DeBERTa-v2 MLM bridge signal: `wess_gold` moved correct-vs-counterfactual log-odds and slot interventions while no-address/destroyed controls stayed at zero. wess mlm transfer multiseed results replicated this across three seeds:

- `wess_gold` mean binding log-odds: **+1.2056 ± 0.7074**;
- controls: ~0 (`plain_mlm` -0.0007, `no_address` +0.0028, random +0.0002, wrong +0.0006);
- `wess_gold` swap delta: **+2.0513**, ablation delta **+1.6287**;
- controls: ~0;
- shortcut pair heuristics: 0% pair accuracy.

This proves persistent entity-indexed slot state can causally steer DeBERTa MLM predictions in controlled counterfactual text. The next question is whether this can scale from log-odds to top-1 binding and official-relevant scores without damaging the protected model's strengths.

## This is not yet the final 100M candidate

The next run is a **short-budget screen**: large enough to test the mechanism under a real BabyLM-style training regime, but still cheap enough to iterate. It should not be treated as SOTA unless it is evaluated and entered into the scoreboard. The protected best remains DeBERTa-v2 8×480 WWM 100M, Overall 40.5269; public leader is 41.8011.

## Model and training scale

Use the existing BabyLM trainer infrastructure and baseline16k tokenizer.

Preferred backbone:
- S1 shape: DeBERTa-v2 12×384, 12 heads, intermediate 1280, max position 1024, baseline16k tokenizer.
- If runtime requires, first run a 6×384 or 8×384 smoke, but the main screen should be S1-shaped because S1 is the known GlobalPIQA-favorable shape.

Budget:
- 10M cumulative word exposure for the primary screen, matching the S1 exact 10M contract as much as possible.
- Track unique corpus words, synthetic words, cumulative exposure, epochs, optimizer steps, masked-token count, and episode ratio separately.

Mixture:
- Official BabyLM text backbone plus counterfactual WESS episodes.
- Start with 10% synthetic episode words and 90% official words.
- If there is time, compare 5%, 10%, 20% episode ratios; otherwise 10% is the first screen.
- Every synthetic word counts toward the 10M corpus/exposure budget.

Masking:
- Standard WWM on official text, matching S1 as closely as possible.
- On synthetic episodes, always include the binding-critical query target in a controlled fraction of examples; other masks follow WWM/token masking so the model still trains generally.

## Arms for the primary 10M screen

Minimum matched arms:

1. **plain_mlm_mixed** — same mixed corpus and masking, no slot module. Controls for episode text alone.
2. **no_address_memory** — same gold spans, state access, recurrence/projection/fusion, but one shared memory or unordered pool; no persistent entity address.
3. **wess_gold_address** — persistent entity slots with correct route.
4. **wess_eventwise_random** — same module, random eventwise writes; true address destruction.

Optional if resources permit:
5. **wess_wrong_entity** — wrong entity writes;
6. **official_only_s1_10M rerun** if exact baseline drift is suspected, though S1 10M already exists.

All arms must share seed/init/tokenizer/data order as much as possible. The decisive comparison is `wess_gold_address` vs `plain_mlm_mixed` and `no_address_memory`; random routing must remove the WESS-specific effect.

## Primary screen measurements

### Mechanism measurements

On the fixed counterfactual paired suite and held-out variants:

- example accuracy;
- pair accuracy;
- correct-vs-counterfactual log-odds;
- recency/distance stratified scores;
- unseen entity-state combination and unseen state scores;
- directed slot swap log-odds delta;
- last-write ablation log-odds delta;
- no-address and random-route intervention deltas.

Success target before official expansion:
- `wess_gold_address` pair accuracy meaningfully above controls (not just log-odds), ideally >0.20 on the hard fixed pair suite and clearly positive on held-out variants;
- log-odds remains > controls by a large margin;
- interventions remain direction-specific.

### Official-compatible short measurements

At 10M, evaluate at least:

- Entity Tracking (official-compatible local scorer);
- EWoK full local scorer;
- GlobalPIQA parallel/nonparallel mean;
- BLiMP/Supplement/COMPS if available from fast evaluator;
- official-text MLM validation loss;
- Reading if fast enough.

This screen does not need AoA/SuperGLUE unless it looks promising, but any promoted 100M candidate must eventually run all 9 columns.

Target movement:
- Entity should improve relative to `plain_mlm_mixed` and S1 10M, or at least not regress while binding pair accuracy increases.
- EWoK/COMPS/GlobalPIQA should not move sharply negative.
- BLiMP/Supplement/Reading must be monitored because the protected best relies on them.

## Promotion rule

Promote to full 100M candidate only if the 10M screen shows:

1. WESS-gold top-1 binding/pair accuracy improvement over no-address and plain mixed;
2. address-destroyed arm loses the effect;
3. directed interventions remain strong;
4. official Entity/EWoK/GlobalPIQA or at least Entity-local proxy moves positive or non-negative with clear mechanism evidence;
5. ordinary MLM and grammar/reading proxies do not show severe damage.

If WESS-gold only improves log-odds but not top-1 after the 10M screen, increase episode ratio or loss weight once before abandoning; do not jump to 100M. If no-address also improves, the effect is content retrieval/fusion, not entity addressing; redesign. If official columns degrade, reduce episode ratio or integrate WESS as auxiliary without overtraining on synthetic text.

## Engineering notes

Build by adapting:
- `training/scripts/babylm_masked_train_leadershape.py` for tokenizer, DeBERTa config, word accounting, WWM, checkpoints, and evaluator compatibility;
- `scripts/wess_mlm_transfer_pilot.py` for pair generation, WESS/no-address/destroyed routing, fusion, interventions;
- `data/wess_mlm_transfer_multiseed.json` for baseline bridge metrics.

Outputs to save:
- run configs and word accounting manifests;
- per-arm mechanism JSON;
- checkpoint paths;
- local official-score JSONs;
- scoreboard update file for any candidate with full/partial official columns.

The goal is to quickly answer whether the new role-binding mechanism can become an official Overall improvement, not to accumulate another local-only result.

## Exact S1 10M reference configuration to match

Reference run: `training/runs/babylm_leadershape_s1_10M_aligned_micro128/`.

Command recovered from `command.txt`:

```bash
python3 experiments/archive/initial_model_studies/training/scripts/babylm_masked_train_leadershape.py \
  --output_dir experiments/archive/initial_model_studies/training/runs/babylm_leadershape_s1_10M_aligned_micro128 \
  --max_word_exposure 10000000 \
  --example_pool_words 10000000 \
  --checkpoint_words 10000000 \
  --model_type deberta_v2 \
  --hidden_size 384 \
  --n_layer 12 \
  --n_head 12 \
  --intermediate_size 1280 \
  --position_buckets 256 \
  --max_relative_positions 256 \
  --deberta_relative_attention true \
  --deberta_pos_att_type p2c,c2p \
  --tokenizer_label baseline16k \
  --tokenization_summary_limit 0 \
  --mask_mode wwm \
  --mask_prob 0.15 \
  --seq_length 256 \
  --max_seq_length 256 \
  --max_position_embeddings 512 \
  --batch_size 256 \
  --micro_batch_size 128 \
  --lr_total_steps 2442 \
  --learning_rate 0.001 \
  --warmup_fraction 0.05 \
  --seed 42 \
  --extra_init_seed 456 \
  --train_rng_seed 789 \
  --log_every 25
```

Observed runtime for the reference S1 10M run was about 474 seconds on the available GPU setup. The WESS screen should use this as the primary short-budget geometry whenever feasible. If a smaller smoke run is needed first, it should only validate implementation and should not replace the S1-shaped screen.

The WESS short-budget arms should reuse the S1 tokenizer, shape, sequence length, WWM probability, learning-rate schedule, batch/microbatch geometry, and seed plan as closely as possible. The synthetic episode mixture changes the example stream; therefore every arm must record exact official words, synthetic episode words, cumulative exposure, optimizer steps, and masked binding-target count so the comparison remains interpretable.
