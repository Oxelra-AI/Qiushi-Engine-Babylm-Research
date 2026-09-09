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

`trust_remote_code=True` is required — the GPT-BERT modeling code ships with the repo
(`modeling_gpt_bert.py`, `configuration_gpt_bert.py`). The model can be loaded as
`AutoModelForMaskedLM` or `AutoModelForCausalLM`; all reported results use **MNTP**
(masked-next-token-prediction) evaluation, which outperformed pure-causal and pure-masked
evaluation on the aggregate in our sweeps.

## Results (official BabyLM 2026 evaluation pipeline, MNTP)

| Task | Score |
|---|---|
| BLiMP | 71.59 |
| BLiMP Supplement | 63.93 |
| EWoK | 51.94 |
| Entity Tracking | 27.95 |
| COMPS | ≈54.3 |
| Global PIQA | ≈35 |
| SuperGLUE (finetuned avg) | ≈69.9 |
| AoA correlation (CDI) | −0.151 |

SuperGLUE finetuning used the official pipeline at learning rate 8–9e-5, seed 44.
SuperGLUE convention: F1 for MRPC and QQP, accuracy elsewhere.

## Training data

Only the official BabyLM 2026 strict-small corpus was used — no custom data, no synthetic
data augmentation, no teacher models, no external pretraining. The model is fully compliant
with the strict-small track's data budget.

## Limitations

This is a deliberately small model trained on a developmentally plausible data budget. It is
a research artifact for studying sample-efficient pretraining, not a general-purpose language
model: it has limited world knowledge, produces low-quality free-form generation, and inherits
whatever biases exist in the BabyLM corpus (CHILDES, Project Gutenberg, OpenSubtitles, Simple
Wikipedia, BNC, Switchboard). It should not be deployed in any user-facing application.

Reported scores are from a single training run.

## Citation

```bibtex
@inproceedings{satheesh2026babysteps,
  title  = {BabySteps at BabyLM 2026: Optimizer Choice and Hyperparameter Tuning
            Beat Data Interventions for Sample-Efficient GPT-BERT Pretraining},
  author = {Satheesh, Siddhi and Murillo, Jorge},
  booktitle = {Proceedings of the BabyLM Challenge 2026},
  year   = {2026}
}
```

Built on GPT-BERT (Charpentier & Samuel, 2024), Muon (Jordan et al., 2024), and
AdaMuon (Si et al., 2025).
