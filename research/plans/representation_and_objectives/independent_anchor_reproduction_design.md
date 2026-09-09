# earlier analysis: Independent-Anchor Reproduction Design

## Scientific Purpose

Test whether the stability–plasticity separation (coherent legal replay training only a private pathway from an immutable anchor function) recurs beyond the seed43022/chck_82M anchor as **retained-plus-new competence** rather than another macro redistribution.

## Anchor Selection

**chck_80M** from the same scale-1.75 seed43022 trajectory.
- Actual exposure: 80,034,368 words
- Path: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_80M`
- Architecture: DeBERTa-v2 8×480, adapter128 scale1.75, legal16k tokenizer
- Rationale: tests trajectory-generality (not 82M-specific), uses existing checkpoint (no new pretraining), exposure-matched ordinary continuation exists (chck_84M)
- Limitation: same seed/trajectory, not fully trajectory-independent; a positive result here tests whether the effect generalizes along the trajectory, not across trajectories

## Four Arms

| Arm | Description | Source | Training | Total Exposure |
|-----|-------------|--------|----------|----------------|
| A (coherent80) | Coherent private replay from frozen chck_80M | chck_80M + coherent legal suffix | ~170s private-only | ~84,027,168 |
| B (shuffled80) | Spanbreak private replay from frozen chck_80M | chck_80M + shuffled legal suffix | ~170s private-only | ~84,027,168 |
| C (ordinary84) | Ordinary continuation = chck_84M | scale-1.75 ladder checkpoint | none | 84,028,405 |
| D (anchor80) | Private-off reference = chck_80M | scale-1.75 ladder checkpoint | none | 80,034,368 |

Exposure match: coherent80/shuffled80 vs ordinary84 differ by ~1,119 words (0.001%).

## Training Parameters (same as layerwise exchange readout and gradient anatomy except anchor)

- Endpoint: chck_80M (NOT chck_82M)
- initial_consumed_words: 80,034,368
- skip_rows: 518,144 (2024 batches × 256 = batch-aligned)
- max_tail_charged_words: 3,992,918
- full_cap_words: 100,000,000
- batch_size: 256, seq_length: 256
- learning_rate: 0.001, warmup: 0.06, weight_decay: 0.01
- mask_prob: 0.15 (WWM)
- private_adapter_bottleneck: 128, private_adapter_scale: 1.0
- neutral_lambda: 1.0 (KL against frozen slow path)
- train_rng_seed: 43023
- Trainable: private_adapter_only (995,584 params frozen slow: 35,463,008)

## Predefined No-Training Readouts (after training, before interpretation)

### R1: Model Identity
- SHA256 of model.safetensors for coherent80 and shuffled80
- Non-private parameter equality to chck_80M (bitwise safetensors comparison)
- Training metrics: updates count, total consumed words, first/final loss, neutral loss

### R2: Cheap7 on All Four Arms
Official-compatible cheap7 (BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading) on all four arms.

### R3: Task-Balanced Retained-Plus-New Competence

For each discrete evaluation column c ∈ {BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA_nonparallel, Reading}:

- **Retention rate(model, c)** = #{items: anchor_correct AND model_correct} / #{items: anchor_correct in c}
- **Discovery rate(model, c)** = #{items: anchor_wrong AND model_correct} / #{items: anchor_wrong in c}

Compute for coherent80, shuffled80, and ordinary84.

**Pass criterion:** coherent80 has retention_rate ≥ ordinary84 in ≥ 5/7 columns AND discovery_rate ≥ ordinary84 in ≥ 4/7 columns.

**Fail criterion:** coherent80 retention < ordinary84 in ≥ 3/7 columns, OR coherent80 discovery < ordinary84 in ≥ 4/7 columns.

### R4: Label-Free Private-Benefit Signature

For each discrete evaluation item i:
- **Change zone**: items where coherent80 and anchor80 make different top predictions
- **Retention zone**: items where they agree

Within the **change zone**:
- **Coherent change accuracy** = fraction of change-zone items where coherent80 is correct
- **Anchor change accuracy** = fraction of change-zone items where anchor80 is correct
- **Private benefit** = coherent_change_accuracy − anchor_change_accuracy

Compare to ordinary84:
- **Ordinary change zone**: items where ordinary84 and anchor80 differ
- **Ordinary change accuracy** = fraction of ordinary-change-zone items where ordinary84 is correct
- **Ordinary anchor accuracy** = fraction of same items where anchor80 is correct

**Pass criterion:** coherent80 private_benefit > 0 AND coherent80 private_benefit > ordinary84 analogous benefit (evidence that private pathway changes are selectively beneficial).

**Fail criterion:** coherent80 private_benefit ≤ 0 OR coherent80 private_benefit ≤ ordinary84 analogous benefit.

## Decision Criteria (predeclared)

### CONTINUE (justify deepening the fast-path principle)
ALL of:
1. coherent80 cheap7 > anchor80 cheap7 AND coherent80 cheap7 > ordinary84 cheap7
2. R3 task-balanced criterion passes
3. R4 label-free private-benefit criterion passes

### STOP (end fast-path instantiation, redirect to representation-level mechanism search)
ANY of:
1. coherent80 cheap7 ≤ ordinary84 cheap7
2. R3 task-balanced criterion fails
3. R4 private-benefit ≤ 0

## Cost Estimate

- Training: 2 × ~170s = ~340s total GPU time (parallel on 2 GPUs: ~170s wall)
- Evaluation: 4 × ~20min cheap7 = ~80 min total GPU (parallel: ~40 min wall)
- Readout: CPU-only, ~5 min
- Total wall time: ~45 min

## Compliance

- No leaderboard submission or HF repo modification
- All outputs are local to experiments/archive/representation_and_objectives
- Protected chck_82M and coherent86 endpoints remain untouched
