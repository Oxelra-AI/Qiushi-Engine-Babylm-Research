# English technical report

[Read the report](qiushi-engine-babylm-report-en.pdf): **Data-Efficient Language
Modeling: From Frontier Advancement to Principle-Guided Model Improvement**.

The report follows Qiushi Engine's autonomous research on BabyLM 2026
Strict-Small through frontier-model construction, principle discovery, and
principle-guided improvement. Seven main sections develop the research setting,
three stages, independent findings, discussion, and conclusion. Appendices
provide the algorithms, complete results, research-material links, and 74-topic
catalog. The report is dated 8 September 2026.

This English edition follows the [Chinese report](../zh/qiushi-engine-babylm-report-zh.pdf).
Both read the same [results](../../results/), cite the same bibliography, and use
the same eight [English-language vector figures](../zh/figures/README.md).
References are numbered by first citation. Both editions use one shared
[bibliography](../references.bib), with the same cited entries and numbering.
Scientific distinctions, including continuation versus fine-tuning seeds,
complete-model comparisons versus component attribution, and corrected
interpretations, are maintained in both languages.

The English layout uses neutral headings, compact paragraph and figure spacing,
a two-line title in one type size, and a single correspondence line. The blue
Qiushi Engine footer and shared figure palette retain the report's visual identity.
Scientific figures are centered at 180 mm width to improve label readability;
their text and lines remain vector graphics.

## Edit and build

Edit `main.tex` and `chapters/`. Typography and author information are in
`latex/`; the shared bibliography is `../references.bib`. `render.py` generates
English tables from the repository's result files. Numerical results should be
edited only at their shared source, not in generated tables.

From the repository root:

```sh
make report-en
```

The build requires Python 3 with NumPy and Matplotlib, XeLaTeX, BibTeX, Poppler
utilities, Latin Modern fonts, and TeX Gyre Heros. It regenerates shared figures
and language-specific tables, compiles the PDF, and checks citations, links,
fonts, cross-references, and bilingual numerical agreement. Intermediate files
stay in `build/` and `tables/`. No model training or evaluation runs are started.

## Standalone TeX source

For a portable source archive, first build the report, then run:

```sh
python3 reports/en/package_source.py --output /path/outside/repository/report-source.zip
```

The archive contains only the required TeX files, figures, logo, tables,
bibliography, and compiled bibliography. Its independent XeLaTeX build must
produce the same PDF text as the repository build. It needs no parent-directory
files, Python, network access, or model weights to compile.

For arXiv, select XeLaTeX and review the server-generated PDF before submission.
Fonts are loaded by their TeX Live filenames, following the
[arXiv compilation guidance](https://info.arxiv.org/help/texlive.html).
Creating this local archive does not submit or publish it. A shorter
journal- or conference-specific manuscript remains a separate document.

## Research materials

- [Model packages and usage](../../models/README.md)
- [Methods](../../methods/README.md) and [training instructions](../../reproducibility/TRAINING.md)
- [Claims and evidence](../../evidence/claim_evidence_map.md)
- [Research reading paths](../../research/reading_paths.md) and [topic materials](../../research/materials.md)
- [Research notes](../../research/notes/README.md) and [experimental plans](../../research/plans/README.md)

Each language directory provides one current report PDF and its editable source.
Repository release status is tracked separately in
[STATUS.md](../../reproducibility/STATUS.md).
