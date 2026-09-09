# In-Corpus Adult Prose as a Substitution Discriminator

Status: corpus materialized; its training/evaluation results were not yet available.

The in-corpus alternative replaced 3,005 developmental/speech rows with unused Gutenberg/SimpleWiki material. It changed 442,987 words, rho 0.044299, while retaining exact 10M-pool and 100M-exposure geometry. It contained no FineWeb text and used the same tokenizer, seed and recipe.

This separated an out-of-corpus-content explanation from an adult-prose/distribution explanation. The planned comparisons were in-corpus versus the corresponding FineWeb substitution and matched clean data, together with quarter/half/full-dose and register contrasts.

At this stage the adult-prose-removal MAX arm and quarter/half-dose trainings had completed their checkpoint ladders; their existence was not equivalent to a completed integrated downstream comparison. Full-dose and childspeech-removal training remained incomplete, and in-corpus training was pending.

The open questions were whether adult-prose removal cost more than developmental/speech removal, whether unused in-corpus adult prose reproduced the FineWeb benefit, where dose onset or saturation occurred, and whether a predeclared register-sign/magnitude prediction held. Register divergence was not evidence for a reusable event-role coordinate.
