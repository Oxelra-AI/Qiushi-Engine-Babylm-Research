# compliance control interpretation and eval harness — SuperGLUE primary-metric patch regression
Reference target: `compact_view_reinvest` using old-tokenizer seed43022 results.

- Expected current-coordinate SuperGLUE from pristine collate: 71.036049528253.
- Direct primary-metric mean: 71.036049528253.
- Harness patched mean: 71.036049528253.
- Legacy accuracy-only mean: 71.381071472243.
- Accuracy-only minus primary: 0.345021943990.
- Regression pass: True.

This validates that the compliant-endpoint evaluation harness no longer summarizes SuperGLUE in the inherited accuracy-only coordinate. Final pristine collation still remains authoritative.
