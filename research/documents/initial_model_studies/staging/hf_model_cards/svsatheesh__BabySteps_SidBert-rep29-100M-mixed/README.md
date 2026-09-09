---
license: mit
language:
- en
tags:
- babylm
- babylm-2026
- gpt-bert
- sample-efficient-pretraining
- muon
datasets:
- babylm/babylm_100M_2026
pipeline_tag: fill-mask
library_name: transformers
---

# BabySteps SidBert 100M (BabyLM 2026, strict track)

A GPT-BERT hybrid masked/causal language model trained on the official BabyLM 2026
**strict** corpus (100M words), submitted by team **BabySteps** to the 2026 BabyLM Challenge.

The central finding behind this model: under a fixed strong architecture at BabyLM scales,
**optimizer choice and learning-rate tuning outperformed every data-centric intervention we tried**: curriculum-ordered corpora, textbook and story corpora, synthetic data generation,
and teacher-supervised auxiliary losses. This model is a faithful GPT-BERT reconstruction trained
with the Muon optimizer on the unmodified official corpus, trained on a single GPU.

## Model details

| | |
|---|---|
| Architecture | GPT-BERT (hybrid masked + causal objective) |
| Parameters | ~125M |
| Hidden size / layers / heads | 768 / 12 / 12 |
| Vocabulary | 16,384 (BabyLM GPT-BERT baseline BPE) |
| Training sequence length | 128 |
| Objective | mixed (causal_per_masked = 1) |
| Optimizer | Muon (lr 0.005), auxiliary AdamW-style lr 0.007 |
| Schedule | linear warmup + two-phase decay |
| Steps × tokens/step | 12,330 × 131,072 (~1.6B tokens, ~12 epochs) |
| Precision | bf16 |
| Seed | 42 |
| Hardware | 1 × NVIDIA RTX 5090 (data-parallel) |
| Evaluation mode | MNTP |

## Training data

The official BabyLM 2026 **strict** corpus (100M words) — no custom, synthetic, or augmented data,
and no teacher models.

## Results (official BabyLM 2026 evaluation pipeline, MNTP)

| Task | Score |
|---|---|
| BLiMP | 79.53 |
| BLiMP Supplement | 71.86 |
| EWoK | 56.36 |
| Entity Tracking | 24.52 |
| COMPS | 59.41 |
| Global PIQA | 40.67 |
| (Super)GLUE (finetuned) | 73.03 |

SuperGLUE per task: BoolQ 71.0 · MNLI 68.9 · MRPC (F1) 91.3 · MultiRC 68.1 · QQP (F1) 77.4 · RTE 70.5 · WSC 63.5
Finetuning recipe: lr 2e-5, seed 42, batch size 4, 10 epochs.

## Checkpoints

Intermediate checkpoints are published as branches following the BabyLM naming convention
`chck_{N}M`, covering the required milestone schedule (1M–10M, 10M–100M, 100M–1000M words seen).

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained(
    "svsatheesh/BabySteps_SidBert-100M-mixed", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("svsatheesh/BabySteps_SidBert-100M-mixed")

# a specific milestone checkpoint:
early = AutoModelForMaskedLM.from_pretrained(
    "svsatheesh/BabySteps_SidBert-100M-mixed", revision="chck_10M", trust_remote_code=True
)
```

The model can also be loaded with `AutoModelForCausalLM`, since GPT-BERT is trained under both objectives.

## Reproducibility note

We trained several models with **identical configuration and identical seed**, differing only in GPU
execution. Their scores span a full point of aggregate performance (seven-suite macro average
56.56–57.91, mean 57.31, sd 0.37), with per-suite ranges of 1.8 (BLiMP), 3.6 (BLiMP Supplement),
5.4 (entity tracking), 7.9 (Global PIQA) and 1.7 (SuperGLUE) points. Training is not bit-reproducible
under bf16 with non-deterministic CUDA kernels, and small evaluation suites amplify the resulting
differences. **Point-scale differences between models at this scale should be interpreted with this
variance in mind.** We additionally observe that two-GPU data-parallel runs of this configuration
score systematically ~1 BLiMP point below single-GPU runs. This model is a single-GPU run.

## Limitations

Trained on 100M words of English only. Not instruction-tuned, not aligned, and not intended for
production use — it is a research artifact for studying sample-efficient pretraining. Outputs may
be ungrammatical, factually wrong, or reflect biases present in the BabyLM corpus.

## Citation

```bibtex
@inproceedings{satheesh2026babysteps,
  title     = {BabySteps at BabyLM 2026: Optimizer Choice and Hyperparameter Tuning
               Beat Data Interventions for Sample-Efficient GPT-BERT Pretraining},
  author    = {Satheesh, Siddhi and Murillo, Jorge},
  booktitle = {Proceedings of the BabyLM Challenge at EMNLP 2026},
  year      = {2026}
}
```

Built on GPT-BERT (Charpentier & Samuel, 2024) and the Muon optimizer (Jordan et al., 2024).
