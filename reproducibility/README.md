# Building and Reproducing the Research

## Reading and Editing the Report

```bash
make report
make check
```

`make report` rebuilds figures, tables and the Chinese PDF from the shared result data. `make check` checks file references, result calculations and public content. Both use only the CPU; neither trains models nor calls external inference services.

Requirements are Python 3.12, Matplotlib, NumPy, XeLaTeX, BibTeX, Poppler, and the Noto CJK, TeX Gyre Heros and Latin Modern fonts. Python packages are recorded in [requirements-report.txt](requirements-report.txt). Existing compatible dependencies can be used directly; for a new installation, create an isolated virtual environment.

## Using the Models

The [model guide](../models/README.md) provides a local loading example. Source packages without weights can use `make models` to fetch the corresponding public revisions. Model and report environments are recorded separately; compiling the report does not require GPU training dependencies.

Original training and mechanism programs, data constructors, executed configurations and research notes are archived. Start with the [training guide](TRAINING.md), [core programs](../experiments/CORE_PROGRAMS.md) and [research material map](../research/materials.md). Loadable model packages and successful static source checks do not establish that the complete historical training has been independently rerun.

## Packaging

Packages use the reviewed public file selection and the licenses stated in
[LICENSE](../LICENSE). Install Git LFS before cloning if you need the model weights.

```bash
make package
make package-with-models
```

The first command produces a research source package containing both report PDFs
and their editable sources; the second also includes the available model weights.
Outputs go to `dist/`. Packaging excludes intermediate build files, superseded
drafts and materials marked `local_only`. These commands create local archives;
they do not upload content or create a release tag.

Update numerical results only in `results/` and report text only in the corresponding report source. Code and document references are relative to the repository root or the referring file, with no dependency on original server paths.

The English research catalogue is maintained in `results/research_catalog.tsv`.
Its Chinese wording is kept in `reports/zh/data/research_catalog.tsv` for the
Chinese report. Both catalogues use the same 74 topic identifiers and section
references; neither duplicates the numerical result tables.
