---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: fill-mask
tags:
- babylm
- babylm-2026
- strict-small
- qiushi-engine
- data-efficient-learning
---

# Qiushi-Engine-Frontier-Advancement

**Stage I model in Qiushi Engine's BabyLM Strict-Small research program.**

Frontier Advancement -> Principle Discovery -> Principle-Guided Frontier Advancement.

This English masked language model pairs source passages with concise rewrites, allowing more source passages within a fixed 10-million-word corpus. It uses a DeBERTa-v2 encoder with residual adapters: small trainable layers added to the encoder. A second adapter is added after pretraining and trained during a short continuation.

The two releases connect three research stages: training a limited-data language
model, studying how it learns from paired texts and retains learned behavior, and
using those findings to develop a continuation method tested against matched controls.

Companion model: [Qiushi-Engine-Principle-Guided-Frontier-Advancement](https://huggingface.co/leslie721007/Qiushi-Engine-Principle-Guided-Frontier-Advancement).

## Model

| Property | Value |
| --- | --- |
| Architecture | DeBERTa-v2 with two sequential residual adapter paths |
| Parameters, including masked-language-model head | 36,458,592 |
| Encoder parameters | 36,210,368 |
| Second adapter parameters | 995,584 |
| Layers / hidden size / attention heads | 8 / 480 / 8 |
| Adapter bottleneck / scales | 128 / 1.75 and 0.75 |
| Tokenizer | 16,384-entry byte-level BPE |
| Corpus budget | 10,000,000 counted words in 64,740 rows |
| Recorded cumulative word exposure | 86,005,295 |
| Local full-evaluation Overall | **42.0240** |

## Use

```python
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer

repo = "leslie721007/Qiushi-Engine-Frontier-Advancement"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True)
encoder = AutoModel.from_pretrained(repo, trust_remote_code=True)
```

The custom code is required: both adapter paths are part of the model. The encoder
entry point supports downstream fine-tuning. The included `requirements.txt` records
the tested model-loading environment. For a fixed experiment, use the same Hub commit
for tokenizer, model and encoder.

## Full Evaluation

Local results from the full BabyLM evaluation. SuperGLUE uses the custom encoder
with both adapters and fine-tuning seed 42.

| Component | Score |
| --- | ---: |
| BLiMP | 68.5100 |
| Supplement | 63.6400 |
| EWoK | 50.0200 |
| Entity | 28.3200 |
| COMPS | 52.0500 |
| SuperGLUE | 68.9457 |
| GlobalPIQA | 38.5650 |
| Reading | 8.1650 |
| AoA | 0.0000 |
| Overall | **42.0240** |

Overall is the arithmetic mean of the nine components. Age of Acquisition (AoA)
was evaluated across 18 checkpoints and scored 0.

## Research Comparison

| Method | Continuation seed 62064 | Continuation seed 62065 |
| --- | ---: | ---: |
| Stage I parent | 42.0240 | 42.0240 |
| Ordinary continuation | 42.0926 | 42.1159 |
| Dense input masking, sparse target supervision | 42.2025 | 42.1789 |
| With ordinary-input preservation | 42.2464 | 42.2317 |

Both continuation runs start from the same Stage I model. The final method improves
Overall over ordinary continuation in each run; individual task scores vary.
[METHOD.md](METHOD.md) explains the training changes and what the comparisons measure.

## Artifacts

- `EVALUATION.json`: exact local component scores and model identity.
- `METHOD.md`, `TRAINING.md`, `DATA.md`: design, recorded training and data provenance.
- `CHECKPOINTS.json`: shared intermediate checkpoints and actual exposure counts.
- `evaluation/`: complete final prediction files, AoA measurements and Fast metrics.
- `submission/`: full and full-plus-Fast official-format prediction bundles.
- `VALIDATION.json`: package loading and save/reload tests.

Intended uses are masked-token prediction, sentence scoring and encoder fine-tuning.
The experiments cover English language modeling under the BabyLM Strict-Small budget.

## License

Model weights and original code: [Apache-2.0](LICENSE). Data sources retain their original licenses; see [DATA.md](DATA.md).
