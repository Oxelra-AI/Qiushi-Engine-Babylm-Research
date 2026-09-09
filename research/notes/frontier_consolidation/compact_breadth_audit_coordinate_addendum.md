# Compact-versus-Breadth Audit Coordinate

Status: historical pretraining audit and decision criteria; not an observed treatment effect.

The comparison shared 9,186,995 filler words and 494,154 FineWeb source words. Only the 318,851 companion words differed between compact same-proposition recurrence and independent breadth. The audited pools had 64,183 rows and 10,000,000 words; repeated streams had 641,830 rows and 100,000,000 words.

The audit named the row-block whole-sentence breadth stream, whose hash began `1b98269f`, not the separately constructed interleaved breadth stream. These variants must not be silently conflated. The shared 16,384-token tokenizer used 9,681,149 training words, excluding rewrite text from its FineWeb component. Pass-order seed was 10289931; model/data/training seeds were 43/43022/43023.

The whole-sentence repair removed all source-sentence splits from 5,785 manipulated rows and used 11,942 independent sentences from 6,021 documents with a cap of 6 per document. Companion source hashes had zero overlap with compact-pair source hashes.

This audit reported compact/breadth visible-token counts 14,449,442/14,411,503 and WWM-group counts 9,790,415/9,794,415, with 16,294/15,907 rows longer than 256 tokens. These are the values in this audit, not interchangeable measurements for every breadth layout.

Exact 7/8/10-gram match counts were 21/6/0 for common sources, 7/1/0 for compact rewrites and 20/6/0 for breadth companions. Zero 10-gram overlap did not eliminate every contamination concern.

The historical screening rule used compact-minus-breadth cheap7 at least +0.3 or at most -0.3 for directional evidence, with values inside that interval motivating literal repetition attribution. Stable trajectory/family evidence was still necessary; the thresholds did not convert preflight into a result.
