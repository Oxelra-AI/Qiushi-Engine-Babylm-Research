# Legal-40k Representation, Accumulation and Tokenizer Identity

Status: construction and one-update validation completed; the two full training outcomes were not yet available in this record.

After legal-16k compact-view-reinvestment scores of 40.704 and 41.024, the 40k byte-BPE route tested a representation change, not vocabulary size alone. It also changed segmentation, WWM subword target geometry and embedding-table size, while preserving corpus, 10M pool, 100M stream, depth/width, optimizer, masking schedule and seed identities. The key desired recovery was Supplement/EWoK without sacrificing GlobalPIQA, Entity or COMPS. Earlier 10M-to-20M signs had been unreliable.

Both direct batch-256 attempts failed before useful training through memory exhaustion. The repair used four microbatches of 64 per effective 256-row batch, full-batch mask construction and masked-token-weighted loss accumulation. Data order, row grouping, optimizer-update count, learning-rate schedule and checkpoint exposure were retained. It was explicitly not bit-identical to a hypothetical full-batch forward because dropout execution changed.

The one-update pilot consumed 256 rows and 39,370 words, with vocabulary 40,000, 45,826,720 parameters, 8,247 masked tokens and effective mask rate 0.1528. The saved checkpoint name was not evidence of 1M actual exposure.

Checkpoint serialization changed the raw tokenizer JSON digest without changing vocabulary, special-token IDs, tokenizer length or sample encodings; the sample had zero unknown tokens. Identity checks therefore allowed documented serialization equivalence while rejecting actual tokenizer or model-configuration drift. Raw-file hash inequality alone was insufficient evidence of a changed scientific tokenizer.
