# Data and Tokenization

The model uses a fixed 10,000,000-word English pool, including selected generated
companion text. This budget counts word occurrences in the corpus, not distinct
vocabulary items. Repeated presentations are counted separately in the training budget.

The pool combines child-directed speech, dialogue, subtitles, fiction, encyclopedic
text, and source passages paired with Qwen-generated rewrites. The compact rewrites
of FineWeb-Edu passages were generated with Qwen/Qwen3.5-9B at temperature 0.1,
at most 80 new tokens, batch size 64 and bfloat16 precision. Accepted text pairs
were selected for the training pool. The language model was initialized randomly;
Stage III uses the Stage I model as its preservation teacher.

The selected FineWeb-derived block contains 12,155 source/rewrite pairs packed into
3,005 rows, totaling 423,511 words. A further 9 words of ordinary text complete the
fixed block budget. Shorter rewrites permit 2,061 additional source passages compared
with the longer-rewrite control, within the same word budget.
`DATA_MANIFEST.json` gives the exact corpus fingerprint and source totals.

The 16,384-entry byte-level BPE tokenizer is inherited unchanged by both model
generations. The released tokenizer was trained on the counted 10M-word pool.

Source materials retain their original licenses. FineWeb-Edu is distributed under
ODC-By 1.0 and Common Crawl terms; Qwen/Qwen3.5-9B uses Apache-2.0. Evaluation
datasets are available through the official evaluation repository.

## Sources

- [BabyLM 2026 rules](https://babylm.github.io/guidelines.html)
- [BabyLM FAQ](https://babylm.github.io/faqs.html)
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
