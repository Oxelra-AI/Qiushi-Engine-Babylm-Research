# hf top repo surface — HuggingFace surfaces for top 2026 Strict-Small rows

This note preserves lightweight public model-card/API surfaces for top BabyLM 2026 Strict-Small leaderboard rows. It is route-grounding evidence, not a model evaluation.

## 1. wwm_curriculum_simplification_40k — `go76dof/wwm_curriculum_simplification_40k`
Leaderboard: Overall 41.8, NLP 52.97, Human-like 2.71.
API status: 400; files listed: 0.
README excerpt:
```text
---
language:
- en
license: other
library_name: transformers
pipeline_tag: fill-mask
tags:
- babylm
- babylm-2026
- strict-small
- deberta-v2
- masked-language-modeling
- whole-word-masking
- simplification
- sample-efficient-pretraining
datasets:
- go76dof/Fineweb_simplification_pairs
---

# Model Card for WWM Curriculum Simplification 40k

A 34.7M-parameter DeBERTa-style masked language model trained on 10M words of FineWeb simplification pairs for the BabyLM 2026 strict-small track.

## Table of Contents

- [Model Details](#model-details)
  - [Model Description](#model-description)
- [Uses](#uses)
- [Training Details](#training-details)
  - [Training Data](#training-data)
  - [Hyperparameters](#hyperparameters)
  - [Training Procedure](#training-procedure)
  - [Size and Checkpoints](#size-and-checkpoints)
- [Evaluation](#evaluation)
  - [Testing Data and Metrics](#testing-data-and-metrics)
  - [Results](#results)
- [Technical Specifications](#technical-specifications)
  - [Model Architecture and Objective](#model-architecture-and-objective)
  - [Software](#software)
- [Limitations](#limitations)
- [Citation](#citation)
- [Model Card Authors](#model-card-authors)

## Model Details

### Model Description

This model is a BabyLM 2026 strict-small submission trained with meaning-preserving simplification pairs and a whole-word-masking curriculum.

The model is trained on original FineWeb sentences paired with simplified rewrites. During masked language model pretraining, these paired examples expose the model to two aligned ways of expressing similar content. The goal is to improve sample efficiency under the 10M-word BabyLM strict-small budget.

- Developed by: Shaoxiang Wu
- Model type: masked language model
- Architecture: DeBERTa-v2 style encoder
- Language(s): English
- Model repo: `go76dof/wwm_curriculum_simplification_40k`
- Training data: `go76dof/Fineweb_simplification_pairs`
- License: other

## Uses

This is a pretrained masked language model. It can be used for:

- BabyLM-style zero-shot evaluation with masked-token scoring.
- Fine-tuning on classification or sentence-pair understanding tasks by adding a task head.
- Research on small-data language model pretraining.
- Research on meaning-preserving rewrite pairing, simplification, and whole-word-masking curricula.

This model is not intended as a general-purpose production language model. It is small and trained on only 10M words.

Example loading code:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "go76dof/wwm_curriculum_simplification_40k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
```

## Training Details

### Training Data

The model was trained on FineWeb sentence-level simplification pairs:

- Dataset: `go76dof/Fineweb_simplification_pairs`
- Training file: `FineWeb_simplification_pairs.train`
- Size: 9,999,969 whitespace-counted words
- Format: original FineWeb sentence f
```

## 2. RecGPT-10M — `Serdar404/RecGPT-10M`
Leaderboard: Overall 41.53, NLP 52.4, Human-like 3.46.
API status: 400; files listed: 0.
README excerpt:
```text
---
language:
- en
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- babylm
- babylm-2026
- strict-small
- custom_code
---

# BabyLM Challenge 2026 submission | RecGPT-10M

RecGPT-10M is a 34.17M-parameter recursive causal language model trained for the BabyLM 2026 Strict-Small track. It was trained for 10 epochs on a custom 10M-word English corpus using a 32,768-token BPE vocabulary.

The model applies a shared Transformer block recursively for 16 iterations. Its hidden size is 768, embedding size is 192, and feed-forward intermediate size is 12,288. Training used Muon for the recursive block and AdamW for the embedding-related parameters, with a token batch size of 32,768 and sequence length 256.

## Usage

This repository contains custom Transformers code, so loading requires `trust_remote_code=True`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Serdar404/RecGPT-10M"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```

The model is intended for scoring text as a causal language model. KV-cache generation is not currently implemented.

## BabyLM 2026 evaluation

Final-checkpoint results before leaderboard collation:

| Evaluation | Score |
|---|---:|
| BLiMP | 73.11 |
| BLiMP Supplement | 61.73 |
| EWoK | 52.62 |
| Entity Tracking | 16.59 |
| COMPS | 55.43 |
| GlobalPIQA | 40.68 |
| (Super)GLUE | 66.64 |
| NLP Average | 52.40 |

Intermediate Strict-Small checkpoints are published as Hub revisions named `chck_1M` through `chck_100M` using the official BabyLM checkpoint schedule.

## Resources

- Training code: https://github.com/serdardoesml/bblm26-recgpt
- Dataset construction: https://github.com/serdardoesml/bblm26-dataset
- Evaluation fork: https://github.com/serdardoesml/babylm-eval

## Limitations

This is a small research model trained under the BabyLM data constraint. It is not intended for production deployment, factual question answering, or safety-critical use. Its outputs may contain inaccuracies or undesirable content inherited from its training data.

```

## 3. Wordpiece-24-4 — `Wector1/Wordpiece-24-4`
Leaderboard: Overall 41.31, NLP 52.85, Human-like 0.92.
API status: 400; files listed: 0.
No README.md fetched.

## 4. Bb26_claim_compagg_v22_v15_relay_synp050_synkeep010_s0 — `gyaudhw/bb26_claim_compagg_v22_v15_relay_synp050_synkeep010`
Leaderboard: Overall 40.94, NLP 51.81, Human-like 2.92.
API status: 400; files listed: 0.
No README.md fetched.

## 5. bb26_claim_compagg_v22_v15_relay_synp050_synkeep010_s0 — `gyaudhw/bb26_claim_compagg_v22_v15_relay_synp050_synkeep010_s0`
Leaderboard: Overall 40.93, NLP 51.79, Human-like 2.92.
API status: 400; files listed: 0.
No README.md fetched.

## 6. BabySteps_MurphysLaw-10M-mixed — `svsatheesh/BabySteps_MurphysLaw-10M-mixed`
Leaderboard: Overall 40.86, NLP 53.59, Human-like -3.72.
API status: 400; files listed: 0.
README excerpt:
```text
---
language:
- en
license: mit
library_name: transformers
tags:
- babylm
- babylm-2026
- strict-small
- gpt-bert
- muon
- adamuon
- sample-efficient-pretraining
datasets:
- babylm/babylm-2026-strict-small
pipeline_tag: fill-mask
---

# BabySteps MurphysLaw 10M (mixed)

**BabyLM 2026 Challenge — strict-small track submission (Team BabySteps)**

A GPT-BERT hybrid masked/causal language model trained on the official BabyLM 2026
strict-small corpus (10M words), using the **AdaMuon** optimizer with a tuned learning
rate and tail-weight averaging over the final 20% of training.

Our central finding: after data-centric experimentation (curriculum-ordered
corpora, story and textbook corpora, synthetic data generation, teacher-supervised
objectives), **none of those interventions beat a faithful GPT-BERT reconstruction trained
on the official corpus with a Muon-family optimizer and a careful learning-rate sweep.**

## Model details

| | |
|---|---|
| Architecture | GPT-BERT (hybrid masked + causal objective, single transformer stack) |
| Parameters | 39,337,540 |
| Hidden size | 384 |
| Layers | 12 |
| Attention heads | 6 |
| Intermediate size | 1280 |
| Vocabulary | 16,384 (BabyLM GPT-BERT baseline BPE) |
| Training sequence length | 128 (architecture supports 512) |
| Position embeddings | relative, bucket size 32 |
| Precision | bf16 |

## Training

| | |
|---|---|
| Corpus | Official BabyLM 2026 **strict-small** (10M words), unmodified |
| Objective | mixed (causal_per_masked = 1, masked_per_causal = 0) |
| Optimizer | **AdaMuon** (matrix params) + AdamW-style auxiliary (embeddings, norms, head) |
| Learning rate | 0.02 (matrix) / 0.006 (auxiliary) |
| Schedule | linear warmup (1.6%) + two-phase decay |
| Weight decay | 0.1 · grad clip 2.0 · z-loss 1e-4 |
| Steps × tokens/step | 9,914 × 16,384 (≈10 epochs) |
| Seed | 42 |
| Final weights | **tail average over the last 20% of training steps** |
| Hardware | 1 × NVIDIA RTX 5090, ~30 mins training time |

Tail averaging left zero-shot scores essentially unchanged (±0.3 BLiMP) but improved
finetuned SuperGLUE by ≈1.6 points on the development twin of this configuration.

## Checkpoints

Intermediate checkpoints are published as **branches** following the challenge's naming
convention, covering the strict-small milestone schedule (1M–10M words in 1M steps, then
10M–100M in 10M steps — 19 checkpoints total):

```python
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(
    "svsatheesh/BabySteps_MurphysLaw-10M-mixed",
    revision="chck_7M",           # any of chck_1M ... chck_100M
    trust_remote_code=True,
)
```

## Usage

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("svsatheesh/BabySteps_MurphysLaw-10M-mixed")
model = AutoModelForMaskedLM.from_pretrained(
    "svsatheesh/BabySteps_MurphysLaw-10M-mixed", trust_remote_code=True
)
```

`trust_remote_code=True` is required — the GPT-BERT modeling code shi
```

## 7. instanton-hybrid — `qyxu1994/instanton-hybrid`
Leaderboard: Overall 40.78, NLP 51.53, Human-like 3.18.
API status: 400; files listed: 0.
README excerpt:
```text
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
| **Leaderboard
```

## 8. bb26_claim_agg_synonly_s0 — `gyaudhw/bb26_claim_agg_synonly_s0`
Leaderboard: Overall 40.67, NLP 51.46, Human-like 2.91.
API status: 400; files listed: 0.
No README.md fetched.

## 9. deberta-base-75k-sam_ext-s1 — `augustinian-babylm/deberta-base-75k-sam_ext-s1`
Leaderboard: Overall 40.62, NLP 48.74, Human-like 12.19.
API status: 400; files listed: 0.
README excerpt:
```text
---
license: apache-2.0
language: en
tags: [babylm, babylm-2026, strict-small, deberta, vision-initialized, synthetic-grounding]
---
# deberta-base-75k-sam_ext-s1 — BabyLM 2026 strict-small

DeBERTa-v3-base, identical recipe to
[`deberta-base-75k-sam`](https://huggingface.co/augustinian-babylm/deberta-base-75k-sam),
with **extended vision initialization**: in addition to 21,134 tokens
seeded from real grounding data (Flickr30k Entities, RefCOCO/g/+, THINGS;
SAM ViT-B features), 737 further tokens covering 1,151 concrete
zero-support words were seeded via a synthetic pipeline (LLM-written
scene descriptions → SDXL-Turbo images ×3 → OWLv2 open-vocabulary
detection → SAM features pooled in detected boxes, through the same
extraction code). Total seeded: 21,871/75,000. The synthetic pipeline
affects **embedding initialization only** — no synthetic images or
descriptions enter the training text.

**Training data**: custom ~9.9M-word corpus (`bb24.train`) from our BabyLM
2024 submission ([Edman et al. 2024, "Are BabyLMs Second Language
Learners?"](https://aclanthology.org/2024.conll-babylm.14/)) — a mixture
of LLM-synthesized paraphrase/contrastive data (SynCSE-partial; Zhang et
al. 2021) and portions of the official BabyLM corpus (Simple Wikipedia,
Gutenberg, Switchboard). Within the strict-small 10M-word budget.

On a purpose-built visual-property benchmark, the synthetically grounded
words show a seeded-like advantage over the non-extended model in 3/3
training seeds; real-grounded words and general linguistic performance
are unaffected.

Intermediate checkpoints: `chck_1M` … `chck_100M`, `earlier analysis` …
`earlier analysis`; `main` = final. Code, benchmark, analyses:
https://github.com/bylinina/augustinian_babylm

```

## 10. bb26_claim_compagg_clean_union_s1 — `gyaudhw/bb26_claim_compagg_clean_union_s1`
Leaderboard: Overall 40.59, NLP 51.3, Human-like 3.09.
API status: 400; files listed: 0.
No README.md fetched.

## 11. leviosa-baseline-medium — `GorkaUrbizu/baseline-medium`
Leaderboard: Overall 40.55, NLP 51.39, Human-like 2.6.
API status: 400; files listed: 0.
No README.md fetched.

## 12. bb26_claim_compagg_w025_synp_w025_s0 — `gyaudhw/bb26_claim_compagg_w025_synp_w025_s0`
Leaderboard: Overall 40.52, NLP 51.14, Human-like 3.33.
API status: 400; files listed: 0.
No README.md fetched.

Full JSON summary: `experiments/archive/compact_experience/data/babylm2026_surface/hf_repo_surfaces/summary.json`
