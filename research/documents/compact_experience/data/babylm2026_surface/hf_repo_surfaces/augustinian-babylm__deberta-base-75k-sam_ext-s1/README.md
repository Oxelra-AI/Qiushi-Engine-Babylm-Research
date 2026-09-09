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
