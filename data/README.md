# Data and Reconstruction

Training uses an English pool of 10,000,000 counted words across 64,740 rows, covering child-directed language, dialogue, subtitles, literature, encyclopedic text and selected source-restatement pairs. Pool size and cumulative word presentations are separate budgets; repeated presentations must still be counted.

The [published data description](../models/frontier/DATA.md) and [structured manifest](../models/frontier/DATA_MANIFEST.json) record composition and generation settings. Compact FineWeb restatements use Qwen/Qwen3.5-9B with temperature 0.1, at most 80 new tokens, batch size 64 and bfloat16. These settings come from the pinned model's published documentation, not inference from the repository layout. Stage III reuses existing paired text. Its preservation teacher is this study's first-generation model, distinct from the external text-generation model.

Original construction, selection and filtering programs, generation templates, tokenizer-training code and construction records are included. See the [data reconstruction chain](RECONSTRUCTION.md) for specific entry points and exact input identities. A generator model name is not a complete reconstruction recipe, and the code license does not grant redistribution rights for all source or evaluation text.

`experiments/archive/` retains the original adjacency of constructors and their results; this directory provides a common reading entry point. Source-data licenses are not subsumed by the code repository's license. The [third-party material guide](../reproducibility/THIRD_PARTY.md) records these distinctions separately.

The local project includes realized training streams, paired and split-window inputs, selection records and item-level measurements; see the [program-to-material coverage map](../evidence/material_coverage.json). Files marked `local_only` remain at their scientific paths for inspection but are excluded from Git commits and distribution packages while source licenses and attribution requirements are resolved. Public-facing changes affect identifiers and paths only, not training text, numerical values, pair relationships or realized order.
