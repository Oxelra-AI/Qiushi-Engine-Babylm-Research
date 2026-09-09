# Original Compact-versus-Breadth Estimand

Status: historical pretraining design; the sliced breadth construction was subsequently superseded by a whole-sentence repair.

The intended comparison held 10M unique-pool words, 100M repeated exposure, row word sequence, source composition, pass order and the shared 16k tokenizer fixed. Only 318,851 FineWeb companion words differed: shorter restatements of the same propositions versus independent source sentences. A compact advantage would favor aligned recurrence; a breadth advantage would favor additional experience. Similar scores would leave literal source repetition as the next attribution control.

Preflight counted 64,183 rows per 10M pool and 641,830 rows per 100M stream. The recipe was DeBERTa-v2 8x480, 8 heads, FFN multiplier 4, effective batch 256 through microbatch 64, sequence length 256, AdamW learning rate 0.001, warmup 0.06, weight decay 0.01, WWM 0.15, data-order seed 43, initialization seed 43022 and training seed 43023.

This design initially treated equal row lengths as adequate matching. The later sentence-boundary audit showed why that was insufficient: slicing independent sentences introduced a coherence confound. Accordingly, the original readiness assessment is historical, not authorization to substitute its sliced breadth stream for the repaired comparator.
