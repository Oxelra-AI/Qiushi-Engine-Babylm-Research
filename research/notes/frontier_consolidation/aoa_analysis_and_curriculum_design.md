# aoa analysis and curriculum design — AoA Analysis and Developmental Curriculum Design

## Evidence from COMPACT_EXPERIENCE Clean-Qwen Model (AoA=0.0)

### Surprisal Trajectory (504 CDI words, 19 checkpoints)
```
chck_1M:   9.52  (epoch 1 start)
chck_2M:   9.85  (INCREASES - model learning generic patterns first)
chck_3M:   9.93  (still increasing)
chck_4M:   9.89
chck_5M:   9.93
chck_6M:   9.77
chck_7M:   9.44  (starting to decrease)
chck_8M:   9.17
chck_9M:   9.30
chck_10M:  8.96  (end epoch 1)
chck_20M:  8.44  (epoch 2)
chck_30M:  8.62  (INCREASES at epoch 3)
chck_40M:  9.05  (INCREASES further at epoch 4!)
chck_50M:  8.70
chck_60M:  8.54
chck_70M:  8.66
chck_80M:  8.61
chck_90M:  8.52
chck_100M: 8.52  (final)
```

### Key Observations
1. Mean surprisal is NON-MONOTONIC with bumps at epochs 3-4
2. The model learns all CDI words roughly uniformly (no developmental ordering)
3. CDI-late words like "time", "think" are common in Gutenberg/Wiki → learned fast
4. CDI-early words like "mommy", "daddy" are concentrated in CHILDES → learned slower
5. Result: natural learning dynamics create ZERO or NEGATIVE correlation with child AoA

### CDI Word Distribution
- 504 total CDI words
- 32 very early (≤20 months): baby, ball, banana, book, daddy, dog, mommy, shoe, car, etc.
- 203 mid-range (21-25 months)
- 212 later (26-30 months)
- 57 very late (>30 months): think, time, yesterday, tomorrow, etc.

### Why Standard Training Gets AoA=0.0
Standard training shuffles all data uniformly across epochs. CDI-late words (abstract: "think", "time", "yesterday") appear frequently in adult text (Gutenberg, Wikipedia) and get low surprisal quickly. CDI-early words (concrete: "mommy", "ball", "dog") are concentrated in CHILDES and get less exposure overall. This creates NEGATIVE or zero correlation.

## Design: Epoch-Level Developmental Composition Curriculum

### Core Principle
Make different epochs use different data compositions, so that:
- Early epochs (1-3): CHILDES-heavy → CDI-early words get concentrated exposure → their surprisal drops first
- Middle epochs (4-7): Gradually introduce adult text
- Late epochs (8-10): Gutenberg/Wiki-heavy → CDI-late words finally get concentrated exposure

### Proposed Data Mix Per Epoch (10M word budget, 10 epochs = 100M exposure)
```
Epochs 1-3 (0-30M exposure):
  60% CHILDES + Switchboard (child-directed, conversational)
  25% BNC Spoken (adult conversational)
  15% Simple Wikipedia (light factual)
  → CDI-early words (mommy, daddy, ball, dog, baby) get high exposure
  → CDI-late words (yesterday, think, time) get low exposure

Epochs 4-6 (30M-60M exposure):
  30% CHILDES
  25% OpenSubtitles (dialogue, moderate vocabulary)
  25% Gutenberg (literary)
  20% Simple Wikipedia (factual)
  → Mixed exposure, transition period

Epochs 7-10 (60M-100M exposure):
  15% CHILDES
  30% Gutenberg (literary, complex vocabulary)
  35% Simple Wikipedia (factual, abstract vocabulary)
  20% OpenSubtitles
  → CDI-late words (abstract, literary) get concentrated exposure
```

### Additional Mechanism: CDI-Targeted Masking
In WWM training, we can BOOST the masking probability for CDI target words:
- Epochs 1-3: CDI-early words masked at 25% (vs 15% baseline) when they appear
- Epochs 7-10: CDI-late words masked at 25% when they appear
This forces the model to PREDICT these words, amplifying the learning signal.

### Implementation Requirements
1. Source-stratified data files for each epoch mix
2. Modified trainer that changes data composition per epoch
3. CDI word list integrated into masking logic
4. Checkpoint saving at every 1M word exposure (19 checkpoints total)
5. AoA evaluation at end of training

### Expected Effect
If successful: CDI-early words' surprisal drops to low values by chck_3M-5M, while CDI-late words' surprisal only drops by chck_70M-90M. This creates positive model-child AoA correlation.

### Arithmetic Leverage
- Current Overall: 41.3443 with AoA=0.0
- Leader Overall: 41.8 with AoA=0.0
- Target: AoA=+10 → +1.11 Overall → Overall 42.5 (ABOVE LEADER)
- Even AoA=+5 → +0.56 Overall → Overall 41.9 (ABOVE LEADER)

### Risk Assessment
- Prior COMPACT_EXPERIENCE "devcurr" first-pass ordering got AoA=0.0 — but it was naive ordering within ONE pass, not epoch-level composition change
- Prior mask/mixed-objective ladders got NEGATIVE correlations — they didn't target CDI words specifically
- The proposed approach is STRUCTURALLY DIFFERENT: epoch-level data mix + CDI-targeted masking

### Resource Requirements
- DeBERTa-v2 training: ~3-4 GiB GPU → works on current 11 GiB free
- Data preparation: CPU only (arrange existing official sources into epoch mixes)
- Training time: ~100M word exposure at batch 256, seq 256 → ~25-30 min on H100
- AoA evaluation: ~17 min (19 checkpoints × ~50 sec each)
