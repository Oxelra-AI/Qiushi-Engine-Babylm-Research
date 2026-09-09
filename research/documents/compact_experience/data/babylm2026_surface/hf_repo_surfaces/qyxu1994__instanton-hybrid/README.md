---
license: mit
language:
  - en
tags:
  - babylm
  - babylm-2026
  - strict-small
  - gpt-bert
  - masked-lm
  - causal-lm
  - sample-efficient-pretraining
pipeline_tag: fill-mask
---

# instanton-hybrid

A 40.2M-parameter **GPT-BERT masked–causal hybrid** trained entirely from
scratch on the official **BabyLM 2026 Strict-Small** corpus (≤10M words).

Part of the `instanton` family — small models trained in minutes-to-hours,
submitted to the [BabyLM 2026 Challenge](https://babylm.github.io/).

| | |
|---|---|
| Parameters | 40,170,648 |
| Architecture | GPT-BERT (8 layers, 512 hidden, 8 heads, 1706 FFN) |
| Vocabulary | 16,384 (byte-level BPE, trained only on the in-budget corpus) |
| Training data | Official BabyLM 2026 Strict-Small, 9,972,806 words |
| Word exposure | 10 epochs ≈ 99.7M words (165.3M subword tokens) |
| Pretrained weights used | **None** — random init, per challenge rules |

## Training objective

Following [Charpentier & Samuel (2024), *BERT or GPT: why not both?*](https://aclanthology.org/2024.conll-babylm.24/),
the model is trained on a 50/50 blend of masked- and causal-LM objectives.
Upstream splits the two objectives **across DDP ranks**; this model was trained
on a single GPU, so the blend is reproduced in-process: half the micro-batches
of each optimizer step come from a masked dataset and half from a causal
dataset, gradients accumulate, then one step is taken.

## Recipe

LAMB, max lr 0.007, cosine schedule with 1.6% warmup / 1.6% cooldown, weight
decay 0.1, grad clip 2.0, z-loss 1e-4, mask ratio 0.30→0.15, sequence length
ramped 128→256→512 at 70%/90% of training, 16,384 tokens per step, seed 42,
bf16. ~2 hours on one A100-80GB.

## Checkpoints

Nineteen word-exposure revisions are published as git branches, as required by
the challenge: `chck_1M` … `chck_9M`, `chck_10M`, `chck_20M` … `chck_100M`,
plus `main` (final). Load any of them with `revision=`.

## Usage

The repository ships a vendored modelling wrapper, so `trust_remote_code=True`
is required. It loads under `transformers` 4.51.x; **5.x is not supported**.

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tok = AutoTokenizer.from_pretrained("qyxu1994/instanton-hybrid", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(
    "qyxu1994/instanton-hybrid", trust_remote_code=True)          # or revision="chck_10M"
```

`AutoModelForCausalLM` also works and is what the age-of-acquisition evaluation
uses. Everything else should be scored as a **masked LM** (pseudo-log-likelihood
/ `mntp`), which is how the numbers below were produced.

## Evaluation

Official BabyLM 2026 pipeline, `mntp` backend for zero-shot / GLUE / reading and
`causal` for AoA.

| Task | Score |
|---|---|
| BLiMP | 72.13 |
| BLiMP supplement | 60.86 |
| EWoK | 50.15 |
| Entity tracking (filtered) | 19.87 |
| COMPS | 52.96 |
| Global-PIQA | 37.14 |
| (Super)GLUE | 67.58 |
| Reading | 6.35 |
| Age of acquisition | 0 (r = −0.11, p = 0.081) |
| **Leaderboard NLP average** | **51.53** |
| **Leaderboard Overall** | **40.78** |

For comparison, the best causal decoder in this family (27M-parameter GPT-2,
same 10M-word budget) reaches BLiMP 67.29 and GLUE 63.01 — the hybrid objective
is worth roughly +4.8 BLiMP and +4.6 GLUE.

### Known limitation

Like every model in this family, `instanton-hybrid` correlates **negatively**
with children's age-of-acquisition ordering, so it scores 0 on that benchmark.
This is a property of the model rather than of the measurement: the checkpoint
sweep is complete (19/19, 152,095 surprisal observations) and tokenizer
fragmentation was ruled out (215 of 237 scored CDI words are single tokens, and
controlling for subword count leaves the correlation unchanged).

## Intended use

A research artifact for studying sample-efficient language acquisition. It is
small, trained on <10M words, and is **not** suitable for deployment or for
generating text people will rely on.

## Citation

Data mixture and recipe are described in the accompanying BabyLM 2026 report,
*Reweighting Child-Directed and Conversational Data for Sample-Efficient BabyLM
Pretraining*. The architecture is from Charpentier & Samuel (2024).
