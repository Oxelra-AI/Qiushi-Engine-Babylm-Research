# baseline wwm available coordinate manifest — official corpus adjacency audit

Evidence JSON: `experiments/archive/initial_model_studies/data/official_corpus_adjacency_audit.json`

Raw dir: `experiments/archive/initial_model_studies/data/reconstruct_tmp/raw_dataset`

| file | lines | sentences | within-line adjacent sentence pairs | mean sentences/line | mean sentence words |
|---|---:|---:|---:|---:|---:|
| bnc_spoken.train.txt | 64843 | 48609 | 58 | 0.75 | 15.16 |
| childes.train.txt | 531305 | 407489 | 2097 | 0.77 | 6.36 |
| gutenberg.train.txt | 59714 | 139090 | 82411 | 2.33 | 18.41 |
| open_subtitles.train.txt | 384524 | 298840 | 5831 | 0.78 | 7.04 |
| simple_wiki.train.txt | 58695 | 101364 | 58727 | 1.73 | 14.06 |
| switchboard.train.txt | 2892 | 2085 | 26 | 0.72 | 11.06 |

Total within-line adjacent sentence pairs: **149150**

Line breaks in recovered official raw files appear to be the only durable document/utterance boundary available locally. Adjacent sentence pairs within a line are plausible for same-document continuity; cross-line adjacency should be treated source-specifically and audited before use.

First 20 example pairs are in the JSON only to avoid clutter.
