# Whole-Sentence and Interleaved Breadth Corrections

Status: completed construction/audit; downstream results were not available in this design record.

The original breadth arm concatenated independent sentences and sliced the stream to compact-row budgets. Of 5,785 rows, 5,780 contained a split sentence; 5,548 began mid-source and 5,548 ended mid-source. Of 13,836 independent sentences, 5,546 spanned rows. Thus apparent compact benefits could reflect coherence rather than same-proposition recurrence.

The whole-sentence repair retained exactly 10,000,000 words in 64,183 rows, including a common 813,005-word FineWeb block: 494,154 common-source words and 318,851 independent companion words. It selected 11,942 whole sentences from 6,021 documents, with at most 6 uses per document. All 5,785 manipulated rows had zero cross-row sentence splits. Row-word sequence and ten-pass order still matched compact.

A row-block layout nevertheless moved common-source positions. Interleaving the same independent sentences between common sources reduced mean absolute source-start displacement from 21.386 to 6.496 words and median per-row maximum displacement from 42 to 13 words. Both streams still had 100,000,000 words and 641,830 rows. Under the shared tokenizer, compact/interleaved breadth had visible tokens per word 1.4449/1.4412, rows over 256 tokens 16,294/15,907 and WWM groups per pass 9,790,415/9,794,364.

The interleaved whole-sentence arm became the preferred comparator. A broad compact advantage would still not distinguish paraphrastic alignment from reinforcement of the same propositions; literal source repetition remained necessary for that attribution. Opposing family-level movements are a trade-off, not an aggregate null. Narrow overlap-adjacent or truncation-related gains cannot establish reusable knowledge formation. Stable late trajectories and complete task vectors were required.
