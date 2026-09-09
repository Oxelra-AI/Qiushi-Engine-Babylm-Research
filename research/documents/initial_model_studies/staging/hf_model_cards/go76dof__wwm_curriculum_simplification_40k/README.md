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
- Format: original FineWeb sentence followed by simplified rewrite

Each pair is stored as an original sentence followed by its simplified rewrite, with blank lines separating pairs.

Example:

```text
Enlightenment thinkers proposed that human reason coupled with empirical study of the physical world would lead to progress---namely, the advancement of science and the improvement of the human condition.
Enlightenment thinkers believed that using reason and studying the world would lead to scientific progress and better living conditions.
```

The tokenizer is a 40k SentencePiece BPE tokenizer trained for this data condition:

- `tokenizer/FineWeb_simplification_pairs_40k.model`
- `tokenizer/FineWeb_simplification_pairs_40k.vocab`

### Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Architecture | DeBERTa-v2 style encoder |
| Hidden size | 384 |
| Intermediate size | 1280 |
| Number of layers | 12 |
| Number of attention heads | 12 |
| Dropout | 0.1 |
| Vocabulary size | 40,000 |
| Objective | Masked language modeling |
| Optimizer | LAMB |
| Learning rate schedule | cosine |
| Max learning rate | 0.007 |
| Training epochs | 10 |
| Sequence length curriculum | 64 -> 256 |
| Masking curriculum | WWM7 -> Token3 |

### Training Procedure

The model uses a whole-word-masking curriculum:

- Epochs 1-7: whole-word masking, where all subword pieces belonging to a selected word are masked together.
- Epochs 8-10: standard token-level masking.

The sequence length also follows a curriculum. Training starts with shorter sequences and later switches to longer sequences, while batch size is scaled inversely with sequence length.

### Size and Checkpoints

The model has 34,677,952 parameters. The final checkpoint corresponds to the end of the 10-epoch training run on the 10M-word strict-small training budget.

## Evaluation

This model was evaluated with the BabyLM 2026 strict-small evaluation pipeline. The reported submission scores are leaderboard scores for the strict-small track.

### Testing Data and Metrics

The evaluation includes:

- BLiMP and BLiMP Supplement: linguistic minimal pair accuracy.
- EWoK: world-knowledge minimal pair accuracy.
- Entity Tracking: accuracy on tracking entity states through text.
- COMPS: compositional generalization.
- GlobalPIQA: average of parallel and non-parallel GlobalPIQA splits.
- Reading: average of eye-tracking and self-paced reading scores.
- (Super)GLUE: mean fine-tuning score across the BabyLM GLUE-style tasks.

### Results

| Metric | Score |
|---|---:|
| Overall Average | 41.80 |
| NLP Average | 52.97 |
| Human-like Average | 2.71 |
| BLiMP | 67.20 |
| BLiMP Supplement | 56.01 |
| EWoK | 56.07 |
| Entity Tracking | 28.45 |
| COMPS | 53.57 |
| GlobalPIQA | 39.67 |
| (Super)GLUE | 69.79 |
| Reading | 5.42 |
| AoA | 0.00 |

The strongest gains in our experiments were observed on EWoK, Entity Tracking, and GLUE-style fine-tuning. In our BabyLM 2026 strict-small submission, this 10M-word model exceeded the official 100M-word GPT-2 baseline on EWoK, Entity Tracking, GlobalPIQA, and (Super)GLUE.

## Technical Specifications

### Model Architecture and Objective

The model is a DeBERTa-v2 style encoder trained with the masked language modeling objective.

Important configuration values:

- `architectures`: `DebertaV2ForMaskedLM`
- `model_type`: `deberta-v2`
- `hidden_size`: 384
- `intermediate_size`: 1280
- `num_hidden_layers`: 12
- `num_attention_heads`: 12
- `vocab_size`: 40000
- `relative_attention`: true
- `pos_att_type`: `p2c`, `c2p`

### Software

The model is compatible with Hugging Face Transformers and PyTorch.

## Limitations

This model is trained only on English text and only under a 10M-word pretraining budget. It is intended for research use rather than production deployment.

The training data is derived from web text and automatically generated simplifications. It may contain noise, omissions, or changes in nuance inherited from either the source text or the rewrite process.

The model card reports BabyLM evaluation results, but some hidden-task results, especially GlobalPIQA and AoA, can be unstable across seeds or evaluation settings.

## Citation

If you use this model, please cite the corresponding BabyLM paper or repository once available.

## Model Card Authors

Shaoxiang Wu
