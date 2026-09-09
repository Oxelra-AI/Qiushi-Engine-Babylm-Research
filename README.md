<img src="assets/qiushi-engine-logo.png" alt="Qiushi Engine" width="220">

# Data-Efficient Language Modeling

**From Frontier Advancement to Principle-Guided Model Improvement**  
Qiushi Engine's end-to-end autonomous research on BabyLM 2026 Strict-Small.

[English report](reports/en/qiushi-engine-babylm-report-en.pdf) · [Chinese report](reports/zh/qiushi-engine-babylm-report-zh.pdf) · [Models](models/README.md) · [Research materials](research/README.md) · [Getting started](#getting-started) · [Citation](#citation)

**How can a language model learn more from the text it already has?** A passage
can occur in training without teaching the model how to use it in a new context.
Likewise, recovering performance on familiar inputs need not restore the ability
to apply the same computation to unfamiliar inputs. Understanding these
distinctions matters when more data is not an available solution.

Qiushi Engine pursued this question through a sustained autonomous research
program: developing a frontier model, investigating learning mechanisms, and
using the resulting understanding to design a stronger model. Literature
research, hypothesis formation, implementation, training, evaluation and revision
ran throughout all three stages. The result is both a model advance and an
experimental account of how limited experience shapes context use.

This repository brings together **two model generations, the complete Chinese
and English reports, and research materials organized around 74 scientific
topics**. It includes the programs and measurements behind the results, along
with the hypotheses, experimental plans, analyses and corrections through which
the research developed.

## Why BabyLM?

[BabyLM](https://babylm.github.io/) studies sample-efficient language learning
under data budgets inspired by human language development. Its 2026 edition is a
shared task and workshop at EMNLP. The Strict-Small setting used here limits the
corpus to **10 million words** and cumulative training exposure to **100 million
word presentations**.

Under these limits, architecture, text construction, tokenization, prediction
objectives and training schedules all compete for the same finite experience.
Adding another pass over familiar text consumes a budget that could instead
support a different learning opportunity. This makes the setting useful for
studying how training choices affect what a model actually learns.

Evaluation spans grammatical knowledge, contextual understanding, entity
tracking, commonsense reasoning, downstream language tasks and human learning
behavior. **Overall averages nine component scores; it is not a single accuracy
percentage.** The reports explain each component and the
[evaluation guide](evidence/evaluation_provenance.md) links the scoring protocol
to the released results.

## Three stages of research

| Stage | Central question | Research contribution |
| --- | --- | --- |
| [I · Frontier Advancement](research/stage1_frontier.md) | How can a fixed word budget support a stronger language model? | Compact restatements and budget reinvestment increase paired-text coverage; residual incremental learning builds the first-generation model and provides a foundation for controlled study. |
| [II · Principle Discovery](research/stage2_principles.md) | What does a training experience teach, and when can the learned computation still be used? | Relation, window, supervision and internal-signal interventions distinguish context-use behavior, transfer conditions and retention after further learning. |
| [III · Principle-Guided Frontier Advancement](research/stage3_improvement.md) | Can those findings guide a better training method? | Dense input masking, sparse supervision and preservation on ordinarily masked inputs improve the model, tested against continuation controls with complete evaluation. |

Each stage develops its own questions, methods, experiments and conclusions.
Later work draws on earlier models and techniques while also revising earlier
explanations. The [research reading paths](research/reading_paths.md) follow
these connections through the original scientific documents.

### A data-efficient learning principle

> Organize limited experience around the contextual information and relationships
> needed for prediction. Design visible information, supervised targets and
> functional preservation separately, and test whether the intended capability
> is learned, works on new inputs and remains useful after further training.

Three experimental findings give this principle concrete meaning:

- **Training relations shape context use.** Exact repetition and aligned
  restatement produce different source-use behavior, with effects depending on
  the target relation. Separating each pair across training windows weakens the
  effect while retaining that condition's texts. See the
  [relation experiments](methods/relation_learning.md).
- **Familiar performance and reusable computation can diverge.** In controlled
  tasks, familiar-query accuracy can recover while unseen symbols no longer
  access the same learned computation. Internal-signal interventions help
  establish which computations affect the answers. See the
  [functional-access studies](methods/functional_access.md).
- **Acquiring and preserving capabilities require distinct choices.** Removing
  local clues from a restatement changes the information available for
  prediction; choosing supervised positions changes the learning objective.
  Preservation on ordinarily masked inputs then constrains changes to existing
  predictions. See the [complete training method](methods/principle_guided_training.md).

Together, these studies connect specific learning conditions to a practical
design. Natural-text experiments, controlled mechanism studies and full-model
comparisons provide complementary evidence; the reports explain the conditions
under which each result holds.

## Two models, one continuous research program

| Model | Public Overall | Downloads and documentation |
| --- | ---: | --- |
| Qiushi-Engine-Frontier-Advancement | 42.02 | [GitHub package](models/frontier/README.md) · [Hugging Face](https://huggingface.co/leslie721007/Qiushi-Engine-Frontier-Advancement/tree/5eb20f9c5088f40183269bb2c97381711aea0143) |
| Qiushi-Engine-Principle-Guided-Frontier-Advancement | **42.25** | [GitHub package](models/principle_guided/README.md) · [Hugging Face](https://huggingface.co/leslie721007/Qiushi-Engine-Principle-Guided-Frontier-Advancement/tree/ce7eabf0dfbd3d1393670f41f610bdccbdbe45d1) |

The second generation achieves the **highest Overall in the report's
8 September 2026 public Strict-Small snapshot**. The
[public comparison](results/leaderboard_comparison.csv) includes these two models
and eight submissions by other publishers. Both models have approximately
36 million parameters and are masked language models, with weights, tokenizers
and custom loading code supplied in this repository and on Hugging Face.

Continuation controls establish how much the new training design adds beyond
ordinary continued training:

| Training strategy | Seed 62064 Overall | Seed 62065 Overall |
| --- | ---: | ---: |
| Ordinary continuation | 42.0926 | 42.1159 |
| Dense masking + sparse supervision | 42.2025 | 42.1789 |
| Dense masking + sparse supervision + ordinary-input preservation | **42.2464** | **42.2317** |

Both seeds start from the same first-generation model, whose local Overall is
42.0240. Dense masking with sparse supervision exceeds ordinary continuation at
equal cumulative word exposure. The full method improves further, with an
additional 517,332 word presentations and extra computation for preservation.
These comparisons establish the value of the training strategies; they do not
isolate every component's causal contribution. The
[complete result table](results/training_strategy_comparison.csv) includes all
nine components, the dense-supervision control and the original numerical
precision, including tasks that decline.

## Research that informs its next advance

The program also provides a concrete case of **Research RSI: recursive
self-improvement of the research process**. Its defining connection is that
scientific understanding, method innovations and experimental experience from
earlier research change subsequent questions and designs. New experiments then
test and refine those judgments.

Here, the first-generation model is both a research outcome and a foundation for
mechanism studies; those studies inform the second-generation training method.
The reports examine this completed research cycle and its relation to autonomous
science. Improvement of general research ability across independent tasks
remains a separate question. The
[research decisions](evidence/research_timeline.md) and
[claim-to-evidence map](evidence/claim_evidence_map.md) make the actual connections
available for study.

## Read the research

The scientific documents themselves live in `research/`: **1,533 notes and
analyses, 108 experimental plans, and 1,590 measurement records and technical
documents**. The three stage narratives connect the main findings; dedicated
topics preserve independent studies of representations, relation graphs, sparse
anchors, learning dynamics, optimization and measurement. Executable experiments,
inputs, numerical results and model packages have their own directories.

| What you need | Where to start |
| --- | --- |
| The complete argument | [English report](reports/en/qiushi-engine-babylm-report-en.pdf), [Chinese report](reports/zh/qiushi-engine-babylm-report-zh.pdf), [scientific synthesis](research/scientific_guide.md) |
| The reasoning behind an experiment | [Selected reading paths](research/reading_paths.md): questions, competing explanations, decisive comparisons and corrections |
| Original analyses and route decisions | [Research notes](research/notes/README.md), grouped by scientific subject and linked to their original files |
| Hypotheses and experiment designs | [Experiment plans](research/plans/README.md), including proposed controls and criteria; plans are distinct from completed results |
| A specific finding or independent branch | [74-topic material map](research/materials.md), with separate pages for each question, its code, inputs and evidence |
| Measurement details and technical records | [Supporting documents](research/documents/README.md), including checkpoint measurements and data documentation |
| Unfamiliar notation | [Terms and experimental conditions](research/terms.md) |

The selected reading paths explain the main research decisions. The research
folders also contain the less prominent analyses and independent branches.
Each source document has one maintained location; its historical statements
remain distinct from subsequent corrections and final conclusions.

## Getting started

The two PDF reports can be read directly from the links above. To obtain code,
research materials and model weights locally, install Git LFS, then run:

```bash
git lfs install
git clone https://github.com/Oxelra-AI/Qiushi-Engine-Babylm-Research.git
cd Qiushi-Engine-Babylm-Research
```

Start with the [model loading example](models/README.md#local-use) for inference
or encoder use. These are masked language models rather than chat models.
Report-building dependencies are listed separately from model dependencies in
the [reproducibility guide](reproducibility/README.md).

```bash
make report-en   # Rebuild the English report
make report      # Rebuild the Chinese report
make check       # Check files, links, model packages and recorded results
```

Report builds use the shared result tables and figures; they do not retrain or
evaluate the models.

### Follow an experiment or reuse a method

- **Check a claim:** [evidence map](evidence/claim_evidence_map.md), [model lineage](evidence/model_lineage.md), [evaluation provenance](evidence/evaluation_provenance.md).
- **Use a model:** [local packages and fixed revisions](models/README.md), including custom encoder code and tokenizer.
- **Inspect an implementation:** [core programs](experiments/CORE_PROGRAMS.md), [training and data](reproducibility/TRAINING.md), [original program dependencies](experiments/entrypoints.json).
- **Rebuild navigation:** `make navigation`; indexes are derived from existing records, not generated scientific findings.
- **Prepare an offline package:** `make package` includes reports and research sources; `make package-with-models` also includes local model weights.

## Repository layout

```text
reports/          Chinese and English reports: editable LaTeX sources and PDFs
research/         Three stages, reading paths, notes, plans and topic evidence
methods/          Compact views, residual learning, relations and training design
experiments/      Original training/mechanism programs, configurations and results archive
results/          Canonical numerical tables shared by reports and analysis
models/           Two versioned model packages, code, tokenizers and weights
data/             Data construction, provenance and redistribution information
evidence/         Claims, research decisions, model lineage and source mapping
reproducibility/  Build requirements, verification and release status
assets/           Original logo and visual conventions
tools/            Build support, data checks, model fetching and packaging
```

This repository contains technical reports and their supporting research assets.
Numerical report tables read directly from `results/`; report text is maintained
in `reports/en/` and `reports/zh/`. Shared data and figures keep both editions
aligned while allowing language-specific typography.

## Release contents and licenses

The repository includes both reports and their editable sources, two final model
packages, and research programs, configurations, construction records, results
and scientific notes organized across 74 topics. The [release summary](reproducibility/STATUS.md)
describes the included materials and completed checks. The
[data guide](data/README.md) explains dataset access, reconstruction and
redistribution conditions.

Original code uses **Apache-2.0**. Reports, scientific figures, research notes and
other written documentation use **CC BY 4.0**. Model packages retain their
published Apache-2.0 notices. Third-party materials retain their original terms;
the team name and logo do not carry a trademark grant. See [LICENSE](LICENSE)
and [third-party notices](reproducibility/THIRD_PARTY.md).

## Citation

If you use the models, methods or research materials, please cite the technical
report. The title, author list and report date are recorded in
[CITATION.cff](CITATION.cff), which also supports GitHub's **Cite this repository**
feature. The report PDFs contain the corresponding-author contact details.
