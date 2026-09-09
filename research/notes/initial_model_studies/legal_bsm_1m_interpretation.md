# legal bsm 1m eval — interpretation of legal BSM 1M three-arm screen

Evidence:

- Full JSON: `data/legal_bsm_1m_eval.json`
- Score note: `notes/legal_bsm_1m_eval.md`
- Trained roots:
  - `training/runs/legal_bsm_1m_official_control_seed42/hf_model`
  - `training/runs/legal_bsm_1m_bsm_coherent_20pct_seed42/hf_model`
  - `training/runs/legal_bsm_1m_bsm_swapped_20pct_seed42/hf_model`

## Main finding

At ~1M words from scratch, the legal BSM replacement corpus did **not** create the intended entity-conditioned binding behavior.

All three arms scored 0.000 pair-level both-correct on every binding-switch probe:

- train templates;
- held-out entities;
- held-out values;
- held-out templates;
- order-flipped contexts;
- natural templates.

The same-value preference stayed 1.0 or near 1.0. This reproduces the bsm density persistence early-state failure in a legal from-scratch setting: early DeBERTa does not form binding from this short exposure, even when about 20% of corpus words are generated binding examples with targeted answer masks.

## Official fast scores

| model | BLiMP | Supplement | Entity | EWoK | GlobalPIQA mean | Reading |
|---|---:|---:|---:|---:|---:|---:|
| official_control | 54.31 | 48.80 | 17.99 | 48.73 | 37.225 | 6.47 |
| bsm_coherent_20pct | 55.29 | 47.60 | 19.85 | 51.00 | 33.265 | 6.13 |
| bsm_swapped_20pct | 53.35 | 50.40 | 19.12 | 50.45 | 36.71 | 6.81 |

Coherent minus official: Entity +1.86 and EWoK +2.27, but GlobalPIQA mean −3.96, Supplement −1.20, Reading −0.34. This is also not attributable to binding alone, because BSM arms had about 3× more optimizer updates, shorter row packing, and targeted answer-mask training.

Coherent minus swapped, the cleaner relation-consistency contrast, is mixed: BLiMP +1.94, Entity +0.73, EWoK +0.55, but Supplement −2.80, GlobalPIQA mean −3.445, Reading −0.68. Since binding probes remain exactly zero, these deltas do not show that consistent binding has been learned.

## Route implication

Do not scale this exact 20% legal BSM recipe to 20M or 100M as if it has passed the key mechanism test. The run shows that the corpus/training machinery works and that BSM replacement changes official columns, but it does not show entity-conditioned binding formation at 1M.

Before larger training, the next research must explain early formation failure. Two useful next directions are:

1. matched-update official-experience reference: make an official-only short-row/targeted-mask arm with the same number of optimizer updates and similar row lengths as BSM arms, so packing and update opportunity are separated from content; and
2. longer developmental timing test: run coherent and swapped BSM to 10M or 20M only if paired with a matched-update reference and binding probes at intermediate checkpoints, to see whether binding appears after a mature linguistic substrate begins to form.

The result does not end the BSM route, but it blocks scaling the current recipe without a cleaner explanation.
