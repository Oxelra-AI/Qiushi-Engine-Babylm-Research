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
