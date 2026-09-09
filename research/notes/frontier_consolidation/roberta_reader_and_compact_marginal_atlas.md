# roberta reader and compact marginal atlas: RoBERTa result-reader prepared + compact/repeat marginal atlas

## Context

The full 100M RoBERTa compact-versus-repeat arms remain in progress. This analysis did not inspect active training directories or perform new training, SuperGLUE or AoA evaluation.

The scientific need before those results arrive is to prevent two forms of drift:

1. reading the RoBERTa result from a single volatile family or single checkpoint; and
2. compressing the natural compact-vs-repeat data marginal into one alleged factor such as word order, relative position, or source-absent labels.

## Result-reader asset

Script: `scripts/roberta_transfer_result_reader.py`

Self-test output:

- `data/roberta_transfer_result_reader/reader_selftest.json`
- `data/roberta_transfer_result_reader/reader_selftest.md`

The self-test reads only the completed compact order result and roberta transfer decision scaffold and the old two-row RoBERTa smoke metrics plus synthetic selected-score payloads. It does not touch the active training directories.

Self-test facts:

- compact order result and roberta transfer decision scaffold exists and reports `ok=true`.
- Matched 100M compact/repeat pair has 30,050 text-different rows and 0 word mismatches.
- Legal-tokenizer natural-treatment difference over 100M from the roberta transfer pair scaffold ready audit: compact has +240,000 active/candidate BPE tokens, -1,960 word groups, and +320 truncated rows versus first-N repeat. This is part of the data treatment, not something to remove in the transfer test.
- Synthetic stable-positive and volatile-carried payloads route to different readings, so the reader can separate stable-family transfer from a GlobalPIQA/Reading-carried bump.

When the RoBERTa result arrives, run the reader in normal mode and inspect `reader_report.json` before interpreting. The reader verifies:

- exact 100,000,000 counted words;
- 2,529 training steps;
- 30,528,064 RoBERTa parameters;
- vocab size 16,384 and spatial repair route status legal tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`;
- seeds 43/43022/43023;
- batch256, seq256, `lr_total_steps=2529`, fixed WWM p=0.15;
- all `chck_10M`...`chck_100M` checkpoint directories;
- selected-score availability and late-band compact-minus-repeat means over `chck_60M`...`chck_100M`.

Primary scientific reading remains: late-band `cheap6_no_GlobalPIQA`, `cheap5_no_GlobalPIQA_Reading`, `EWoK_plus_Entity`, Supplement, Entity, and COMPS. Do not promote a GlobalPIQA/Reading-only movement. A positive stable-family late result would justify an independent-seed paired replication; a neutral/negative stable-family result bounds the effect in this tested RoBERTa coordinate but does not name a single architectural cause.

## Natural compact-vs-repeat marginal atlas

Script: `scripts/natural_compact_repeat_mechanism_atlas.py`

Outputs:

- `data/natural_compact_repeat_mechanism_atlas/mechanism_atlas.json`
- `data/natural_compact_repeat_mechanism_atlas/mechanism_atlas.md`
- `data/natural_compact_repeat_mechanism_atlas/pair_atlas.csv`

Inputs:

- `data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl`, SHA `6d0ac85dec1718e5f5663d09123e2f62a23f7cceb0b491b83a34f7bfca59d32d`, 12,155 compact pairs.
- legal spatial repair route status tokenizer, tokenizer JSON SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`.

Main findings over the 12,155 underlying compact-pair objects:

- Source words / compact view words: 261,803 / 161,708; ratio 0.6177.
- Compact and first-N repeat are exactly matched in whitespace word count for each underlying pair view. When encoded as individual view texts for this atlas, compact has 247,878 legal-tokenizer active BPE tokens versus 216,589 for first-N repeat: +31,289 active BPE tokens. The exact packed training-row audit remains the roberta transfer pair scaffold ready value: +24,000 active/candidate tokens per 10M pass (+240,000 over 100M) because packing, source+view concatenation, and seq256 truncation change the tokenized count.
- Mean compact coverage of source content positions: 67.35%.
- Mean first-N repeat coverage of source content positions: 59.81%.
- Mean compact-minus-repeat coverage delta: +7.54%; 55.10% of pairs have positive delta.
- Mean compact tail-content coverage after the repeat length: 70.59%; 97.22% of pairs recover at least one tail content item.
- Compact content fraction: 64.74% versus repeat 48.84%.
- 75.86% of pairs contain at least one source-absent compact content word; source-absent words are 17.11% of compact content totals.
- 96.29% of pairs contain at least one tail-only compact content word; tail-only words are 32.55% of compact content totals.

Source-position decile result:

| source decile | compact coverage | repeat coverage | compact-repeat delta |
|---:|---:|---:|---:|
| 0 | 68.28% | 100.00% | -31.72% |
| 1 | 62.63% | 100.00% | -37.37% |
| 2 | 62.06% | 100.00% | -37.94% |
| 3 | 63.18% | 99.68% | -36.50% |
| 4 | 63.52% | 94.58% | -31.06% |
| 5 | 64.08% | 69.58% | -5.50% |
| 6 | 63.90% | 31.75% | +32.15% |
| 7 | 65.96% | 7.48% | +58.49% |
| 8 | 70.88% | 0.11% | +70.77% |
| 9 | 76.32% | 0.00% | +76.32% |

## Scientific interpretation

The natural compact-vs-repeat marginal is not just fluent order, not just source-absent labels, not just source-wide coverage, and not just BPE token count. It is a coupled data operation:

- it compresses each source into fewer whitespace words;
- at fixed whitespace word count versus first-N repeat, it reintroduces content from much later source positions;
- it raises content density;
- it creates a modest but widespread source-absent content population;
- under the legal tokenizer it gives more active/candidate BPE pieces for the same word budget in the actual packed training stream (+24,000 per 10M pass; +240,000 over 100M), while the underlying individual pair-view atlas shows the same direction (+31,289 BPE pieces before packed-row truncation effects);
- the saved words are reinvested in additional legal source diversity in the 10M pool.

This atlas is not downstream evidence by itself. Its value is to preserve what exactly the pending RoBERTa result tests. If RoBERTa is positive on stable late families, the next step is independent-seed paired replication of this coupled marginal. If RoBERTa is neutral/negative on stable late families, the next step is not to assert a single missing mechanism; it is to decompose the coupled marginal with constructions that preserve as many of these measured quantities as possible while changing one scientific ingredient.
