# Historical Reference Factorization

Status: archival comparison anchor used in route design, not a newly verified ranking.

The recorded reference model, `go76dof/wwm_curriculum_simplification_40k`, reported Overall 41.80, NLP average 52.97 and human-like average 2.71. Its component vector was BLiMP 67.20, Supplement 56.01, EWoK 56.07, Entity 28.45, COMPS 53.57, GlobalPIQA 39.67, SuperGLUE 69.79, Reading 5.42 and AoA 0.00.

The method description separated several potentially interacting factors: 34,677,952 parameters; a 12-layer, hidden-384, intermediate-1280, 12-head DeBERTa-v2-style encoder; 40k SentencePiece BPE; paired original/simplified FineWeb sentences totaling 9,999,969 words; LAMB with cosine scheduling and maximum learning rate 0.007; ten epochs; sequence lengths 64 to 256; WWM in epochs 1-7 followed by token masking in epochs 8-10.

This factorization motivated controlled representation, geometry, sequence and objective comparisons. It did not justify attributing the reference score to any one ingredient or treating an unavailable training corpus as reproduced. The score anchor was explicitly provisional pending stronger direct evidence, and the local legal-40k comparisons remained unevaluated at the time of the record.
