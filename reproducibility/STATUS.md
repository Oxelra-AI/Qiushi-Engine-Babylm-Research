# Release contents

The research collection accompanies the Chinese and English technical reports,
both dated 8 September 2026. The two model generations, three research stages,
and 74 scientific topics provide its reading structure.

## Included

- Both 58-page report PDFs and their editable LaTeX sources.
- Eight shared English-language vector figures, 40 shared references, and the
  numerical tables used by both reports.
- Both final model packages, including weights, tokenizers, loading code,
  training descriptions and evaluation identities.
- Original research programs, configurations, data construction, numerical
  results, scientific notes and experiment plans.
- Topic reading guides, claim-to-evidence links, source mappings and corrections.

The report PDFs match the final local builds. The English arXiv source archive
is exported separately from the same report source; it is not another manuscript.

## Verification

The 10 September 2026 release checks cover public-file selection, sensitive
identifier and credential patterns, relative links, Python syntax, figure text,
model files, result arithmetic and recorded material identity. Navigation and
material-selection tests pass. Source mapping and coverage are summarized in
[evidence/material_coverage.json](../evidence/material_coverage.json).

The reports agree on 66 cross-reference labels, ten generated numerical tables,
40 cited references and 74 topic entries. Figure enlargement in the English
edition changes presentation only. Historical scientific conclusions retain
their recorded status, including conditional findings and later corrections;
release checks do not represent a new training or evaluation run.

## Licenses and distribution

Original code and configurations use Apache-2.0. Reports, figures, scientific
notes, plans and other written documentation use CC BY 4.0. Existing model and
third-party notices are retained. See [LICENSE](../LICENSE) and
[third-party requirements](THIRD_PARTY.md).

The 572 materials marked `local_only` are not distributed. Their source and
reconstruction information remain documented in the [data guide](../data/README.md).
Generated build directories and superseded drafts are excluded. Git LFS stores
model weights; the public repository starts from the selected research snapshot
rather than the internal development history.

See [build and packaging instructions](README.md) for local report and archive
commands, and the [research reading guide](../research/reading_paths.md) for the
scientific material.
