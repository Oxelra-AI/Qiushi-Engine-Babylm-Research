# debertav2 official40k smoke — DeBERTa-v2 official40k smoke

Evidence JSON: `experiments/archive/initial_model_studies/data/debertav2_official40k_smoke.json`

The official40k smoke training/save/load succeeded. The first note-writing attempt failed only because it looked for `kept_tokens_per_word`; the correct tokenization summary key is `kept_tokens_per_whitespace_word`.

| property | value |
|---|---:|
| tokenizer length | 40000 |
| model | DebertaV2ForMaskedLM |
| params | 45,826,720 |
| embedding params | 19,200,000 |
| non-embedding params | 26,626,720 |
| 20k smoke steps | 32 |
| loss first→last | 10.6119→6.7070 |
| kept tokens / whitespace word | 1.19765 |
| truncated frac | 0.000000 |
| load checks | root and `chck_1M` load with `PreTrainedTokenizerFast` + `DebertaV2ForMaskedLM` |

Interpretation: same 8×480 DeBERTa backbone with official40k preserves non-embedding transformer capacity (~26.6M) but increases total params through embeddings. This smoke supports the full official40k run launched in debertav2 official40k smoke, with the explicit interpretation boundary that the intervention is tokenizer/segmentation plus embedding-capacity, not pure tokenization.
