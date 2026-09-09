# Cached Source Coherence and the Breadth Control

Status: historical construction and audit evidence; neither candidate was trained in this record.

The motivating comparison was semantic views versus packet-local repetition under the same 8x480, 16k-tokenizer, whole-word-masking (WWM), AdamW recipe. Lower MLM loss alone could not establish transferable competence. A broader factual-source candidate therefore required a matched construction rather than an uncontrolled corpus substitution.

The first cached FineWeb-Edu candidate combined 1,656,800 clean-Qwen pair words, 3,000,000 FineWeb words and 5,343,200 identical official-tail words. Its control used official text with the same FineWeb row-length sequence. Both arms contained 64,381 rows and exactly 10M words; shared paired and tail blocks were identical.

However, the 18,750 FineWeb rows included 11,032 single-document rows, 7,473 two-document rows and 245 three-document rows. Thus 7,718 rows, or 41.16%, mixed documents. This threatened a coherence-sensitive interpretation even with matched word counts.

The cleaner construction retained 1,765,120 single-document FineWeb words in 11,032 rows from 5,587 documents, together with the same 1,656,800 paired words and 6,578,080 identical tail words. The non-tail prefix fraction was 0.342192; both arms still had 64,381 rows and all required materialization checks passed. A fragment heuristic nevertheless indicated that single-document chunks were not necessarily complete article paragraphs.

This retained a source-breadth hypothesis with fewer packing confounds. It did not establish FineWeb quality, an Entity gain or a downstream improvement. The same-source semantic-view control remained necessary before selecting a broader-source training route.
