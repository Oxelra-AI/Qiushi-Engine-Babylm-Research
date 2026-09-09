# Language Learning from Limited Experience

This research connects three scientific questions: how to build stronger language models under a limited text budget, which relationships within that experience change learning, and whether new relational learning can coexist with existing predictive functions.

This directory contains the research documents themselves: **1,533 notes and analyses, 108 experimental plans, and 1,590 measurement records and technical documents**.
The folders below contain those documents, grouped by scientific purpose.

```text
research/
  stage1_frontier.md      How the first-generation model was developed
  stage2_principles.md    Mechanisms of relation learning, supervision, and functional access
  stage3_improvement.md   How those findings informed the second-generation training method
  notes/                 Questions, hypotheses, interpretations, research decisions, and syntheses
  plans/                 Competing explanations, experiment designs, controls, and decision criteria
  documents/             Measurement records, data descriptions, and technical appendices
  materials/             Documents, programs, and results linked by scientific question
```

For example, start with the [argument and evidence for relation learning](notes/relation_learning/argument_map_relation_typed_composition.md),
the [mechanistic analysis of functional reach](notes/functional_learning/interface_reach_synthesis.md),
the [three-way control design for compact rewrites](plans/representation_and_objectives/compact_view_triangle_protocol.md),
or the [preservation design and decision criteria](notes/functional_learning/preservation_experiment_design_and_evidence.md).
These records retain the understanding at the time they were written; the curated guides and stage documents present subsequent corrections alongside them.

1. [Frontier model development](stage1_frontier.md): compact rewrites, budget reallocation, and residual learning.
2. [Learning mechanisms](stage2_principles.md): source relationships, supervision allocation, and access to learned functions on unseen inputs.
3. [Principle-guided model improvement](stage3_improvement.md): densely masked inputs, sparse supervision, preservation on ordinary inputs, and paired model comparisons.

The [scientific overview](scientific_guide.md) explains how the studies connect. The [curated reading paths](reading_paths.md) link key questions, hypotheses, control designs, result analyses, and revised interpretations to the original documents. The [glossary](terms.md) explains experimental abbreviations and conditions.

## Original Notes, Plans, and Evidence

| Material | Contents and purpose |
| --- | --- |
| [Research notes and analyses](notes/README.md) | Original problem analyses, research decisions, interpretations of controls, and stage syntheses, grouped by research area |
| [Experimental plans and protocols](plans/README.md) | Competing explanations, controlled variables, and decision criteria specified before experiments; plans remain distinct from measured results |
| [Research topics and materials](materials.md) | Original programs, inputs, configurations, results, and interpretations organized around 74 scientific questions |
| [Measurement records and technical appendices](documents/README.md) | Supplementary data descriptions, checkpoint measurements, and implementation details, counted separately from core scientific findings |
| [Research catalog](catalog.md) | A quick reference to supported results, conditional conclusions, untested candidates, and withdrawn interpretations |

Research writing is collected here; executable programs and experimental inputs remain in `experiments/`, authoritative numerical tables in `results/`, method descriptions in `methods/`, and models in `models/`.
Each item has one maintained location. The number of topics is not a count of independent innovations, and historical judgments do not supersede later corrections.

## Questions, Methods, and Evidence

| Scientific question | Method | Main evidence |
| --- | --- | --- |
| How can the same word budget cover more paired examples? | [Compact views](../methods/compact_views.md) | [Pair and word counts](../results/compact_budget.csv) |
| Does the model learn repeated content or relationships to the source? | [Relation learning](../methods/relation_learning.md) | [Source advantage across three seeds](../results/relation_context_use.csv) |
| Does performance on familiar inputs indicate access to learned functions through unseen symbols? | [Functional access](../methods/functional_access.md) | [Behavioral measurements and internal interventions](../results/interface_reach.csv) |
| What can a trainable residual branch add to a frozen parent model? | [Residual learning](../methods/residual_learning.md) | [Late-training controls](../results/late_consolidation_controls.csv), [scale comparisons](../results/private_scale_sweep.csv) |
| Can new predictive experience improve the complete model? | [Acquisition and preservation training](../methods/principle_guided_training.md) | [Consistent nine-metric comparison](../results/training_strategy_comparison.csv) |
| Does the same preservation coefficient impose the same constraint? | [Preservation mechanism analysis](notes/relations_and_preservation.md#dense-masks-sparse-targets-and-ordinary-state-preservation) | [State-specific readouts](../results/state_preservation_readouts.csv), [gradient magnitudes](../results/preservation_gradient_diagnostic.csv) |

## Syntheses and Report

- [Models, objectives, and experience construction](notes/models_and_experience.md): architecture, masking, corpus organization, and measurement boundaries.
- [Representations, objectives, and incremental learning](notes/frontier_methods.md): compact budgets, frozen parent models, relation graphs, and independent controls.
- [Relations, functional access, and selective preservation](notes/relations_and_preservation.md): source use, internal interventions, and complete-model comparisons.

The Chinese-language technical report discusses [frontier model development](../reports/zh/chapters/03_frontier.tex), [learning mechanisms](../reports/zh/chapters/04_principles.tex), [model improvement](../reports/zh/chapters/05_guided_improvement.tex), and [independent research results](../reports/zh/chapters/06_research_lineage.tex) in separate chapters. Studies of relation graphs, sparse anchors, identity representations, entity reading and writing, budget substitution, optimization, and measurement have scientific value in their own right; they do not automatically provide causal explanations for the final score gains.
