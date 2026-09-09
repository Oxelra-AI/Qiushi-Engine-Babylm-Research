# early prediction calibration: Route recommendation after legal-tokenizer failure

## Scientific situation

The legal 16k tokenizer compact_view_reinvest endpoint scored 40.70/41.02 (two seeds),
~1.0–1.3 below the visible 41.8 leader. Mean losses vs inherited-tokenizer mechanism
coordinate: Supplement −5.09, EWoK −3.08, SuperGLUE −1.86, Entity −1.55.
Gains: GlobalPIQA +1.98, Entity +0.38 (seed43122 only), COMPS +0.27.

## Early-Prediction Calibration

From semantic-view trajectory (treatment vs packet-local control, 10 checkpoints × 7 tasks):

| Signal | 10M | 20M | 40M | 70M | 80M | 100M |
|--------|-----|-----|-----|-----|-----|------|
| eq7 delta | +0.40 | **−0.35** | +1.72 | −0.12 | +0.26 | +0.17 |
| Cross-task delta Pearson | −0.06 | 0.08 | −0.57 | **0.74** | **0.94** | — |

- First reliable delta Pearson > 0.7: **chck_70M**
- EWoK seed-delta: old 50M→100M correlation 0.427, old 80M→100M correlation **0.938**
- Corrected-tokenizer EWoK dynamics were DIFFERENT from inherited (seed ordering reversed at 50M)

**Conclusion**: 10–20M screens are invalid for route selection. The minimum informative
checkpoint for cross-task delta prediction is ~70M; for EWoK-specific stability, ~80M.
Any candidate run must be committed through staged continuation, not early-kill.

## Representation Map

### Training pool (10M words, exact same compact_view_reinvest pool)

| Tokenizer | Vocab | tok/word | over_seq256 | unk |
|-----------|-------|----------|-------------|-----|
| old_inherited_16k | 16384 | 1.4789 | 15,883 | 12 |
| legal_a01_16k | 16384 | 1.4669 | 15,143 | 0 |
| legal_byte_bpe_24k | 24576 | 1.4281 | 13,039 | 0 |
| legal_byte_bpe_32k | 32768 | 1.4066 | 11,991 | 0 |
| legal_byte_bpe_40k | 40000 | 1.3943 | 11,407 | 0 |

40k vs legal 16k: 5.0% fewer tokens per word; 3,736 fewer truncated rows (15,143→11,407).

### Evaluation text token ratios (40k / legal_16k)

| Family | Ratio | Interpretation |
|--------|-------|----------------|
| BLiMP | 0.898 | 10% shorter |
| EWoK | 0.911 | 9% shorter (our largest legal-16k loss domain) |
| Supplement | 0.945 | 5.5% shorter (our second largest loss domain) |
| COMPS | 0.937 | 6.3% shorter |
| GlobalPIQA | 0.951 | 5% shorter |
| SuperGLUE | 0.923 | 7.7% shorter |
| Entity | 0.999 | unchanged |

### Critical observation: old_inherited vs legal_16k eval ratios

| Family | Ratio | Known score delta |
|--------|-------|-------------------|
| EWoK | 1.032 | −3.08 (legal worse despite shorter text) |
| Supplement | 1.003 | −5.09 (legal much worse despite similar length) |
| SuperGLUE | 0.996 | −1.86 (legal worse despite slightly shorter) |

Score losses between inherited and legal 16k are NOT explained by token length.
The inherited tokenizer was trained on 10× more text (100M vs 10M words), producing
a vocabulary that better captures patterns relevant to downstream tasks.

### WWM grouping

ALL byte-BPE tokenizers show exactly 1.000 groups/word. The word-start prefix (Ġ)
perfectly preserves whitespace word boundaries. Changing vocabulary size changes
WITHIN-word subword granularity but not WWM unit count. This is a fundamental
difference from SentencePiece tokenizers that may group tokens differently.

## Route analysis

### Route A: Legal 40k byte-BPE (RECOMMENDED first experiment)

**Hypothesis**: A 40k vocabulary on the same 10M pool recovers some of the
inherited tokenizer's quality advantage through vocabulary breadth, while also
reducing sequence truncation by 25%.

**Why it's the best single-factor test**: Only the tokenizer changes vs the legal
16k runs. Same pool, same stream, same seeds, same architecture. The calibration
shows we can't evaluate cheaply, but the tokenizer change is the most mechanistically
grounded route: EWoK and Supplement are the biggest losses, and 40k shortens those
evaluation families by 9% and 5.5%, making more content visible per prediction.

**Risk**: Model size increases by ~11M parameters (40k×480 embedding vs 16k×480).
This is a confound. But the DeBERTa-v2 architecture scales the embedding table
linearly with vocab; there's no clean way to hold model size constant while
changing vocabulary.

**Approximate parameter budget**:
- 8×480 with 16k: ~32M total (~7.9M embedding)
- 8×480 with 40k: ~43M total (~19.2M embedding)
- Leader 12×384 with 40k: ~34.7M total (~15.4M embedding)

### Route B: 12×384 with 40k (leader-like architecture)

Second priority. Matches the leader's architecture more closely and has similar
total parameter count. Requires validating trainer support for 12-layer config.
Changes tokenizer AND architecture simultaneously, complicating interpretation.

### Route C: Sequence/masking curriculum on legal 16k

Lower priority. Tests whether training procedure (64→256 sequence curriculum,
WWM→token masking transition) recovers score without vocabulary change. The masking
trainer already supports this. Lower risk, potentially lower ceiling.

## Recommended experiment: Legal 40k byte-BPE compact_view_reinvest

### Specification

- **Tokenizer**: `experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k`
  - SHA-256: `94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758`
  - Vocab: 40,000; zero ByteLevel alphabet gaps; zero unk on training pool
  - Trained on exact 10M pool SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`

- **Training data**: Unchanged compact_view_reinvest 100M stream
  - SHA-256: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
  - 647,400 rows, exactly 100,000,000 whitespace words

- **Architecture**: DeBERTa-v2, 8 layers, 480 hidden, 8 heads, FFN×4
  - `vocab_size=40000` (changed from 16384)
  - All other architectural params unchanged

- **Training**: AdamW, LR 0.001, warmup 0.06, weight decay 0.01, fixed WWM 0.15,
  batch 256, seq_len 256, data-order seed 43
  - Run 1: init_seed 43022, train_seed 43023
  - Run 2: init_seed 43122, train_seed 43123

- **Checkpoints**: every 1M words through 10M, every 10M through 100M

- **Evaluation**: Full pristine official coordinate for BOTH seeds
  - Using hardened earlier analysis collation script
  - 7,618-row EWoK, 19×8,005 AoA, direct changed block overlap ancestry current official GlobalPIQA files

### Expected outcomes

| If 40k mean Overall ≥ 41.8 | Compliant legal SOTA candidate. Package and submit. |
| If 40k mean Overall 41.3–41.8 | Partial recovery confirmed; consider architecture change or curriculum. |
| If 40k mean Overall ≤ 41.3 | Vocabulary capacity not the main bottleneck; pivot to Route C or combined approach. |

### Decision protocol

Do NOT evaluate partial surfaces at 10–20M and use them to kill or select.
Both seeds must run to 100M and receive full pristine official evaluation.
Use 70M focused EWoK probes as staging diagnostic only (direction, not decision).

## Artifacts

- Calibration: `experiments/archive/representation_and_objectives/data/early_prediction_calibration/early_prediction_calibration.json`
- Representation map: `experiments/archive/representation_and_objectives/data/fast_representation_map/fast_representation_map.json`
- 40k tokenizer: `experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k`
- This note: `research/notes/representation_and_objectives/route_recommendation.md`
