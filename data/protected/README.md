# EWoK Diagnostics

`ewok_tokenizer_interface.zip` contains the complete tokenizer audit JSON and
its 7,618-row table, including target sentences and token sequences. The JSON
uses scientific tokenizer identities instead of local paths. Counts, item order,
tokenizations and measured values are unchanged; the full CSV is byte-identical
to the recorded table.

EWoK materials are CC BY 4.0. Its [upstream terms](https://github.com/ewok-core/ewok/blob/main/TERMS_OF_USE.txt)
require password-protected distribution to reduce accidental training-data
ingestion; the public password is documented there. Do not redistribute the
decrypted benchmark text as ordinary repository files. Attribution: Ivanova,
Sathe, Lipkin and colleagues, *Elements of World Knowledge*, 2024.

The adjacent research archive exposes aggregate statistics, row identities and
numeric comparisons as ordinary JSON/CSV; text-bearing fields remain in this
protected file. These are evaluation-interface diagnostics, not model training
data and not measurements of model competence.

Extract a local working copy and enter the upstream password when asked:

```bash
unzip data/protected/ewok_tokenizer_interface.zip -d data/local/ewok_tokenizer_interface
```

The separate global-orbit construction audit retains its counts, order hashes,
alias statistics, substitutions and 30 example identities. Original corpus
excerpts are identified by their SHA-256 and reconstructed from the corresponding
input corpus, whose redistribution terms remain with its providers.

`ewok_research_diagnostics.zip` additionally preserves 29 existing text-bearing
research results at their repository-relative paths. Its adjacent JSON index
records the complete and numeric-only file identities. Extract into a separate
working directory; the ordinary repository copies intentionally contain text
hashes instead of benchmark sentences.

```bash
unzip data/protected/ewok_research_diagnostics.zip -d data/local/ewok_diagnostics
```
