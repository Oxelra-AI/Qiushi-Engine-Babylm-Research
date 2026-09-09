# Stage 3: Principle-Guided Model Improvement

[Three stages](README.md) | [Original notes](notes/README.md) | [Experimental plans](plans/README.md) | [Glossary](terms.md)

This stage turns the questions from [Stage 2](stage2_principles.md) into concrete training
operations: making the available source more important to prediction while limiting the harm
new learning can cause to ordinary language functions. It reuses the
[first-generation model](../models/frontier/README.md) and existing paired text, without
pretraining a new base model or incorporating every mechanistic study into the final recipe.
Its outcomes include both a second-generation model and a method for measuring input changes,
supervision allocation, and preservation costs separately.

## Input Masking and Prediction Targets Are Different Choices

A **source** is the original text in a pair; a **rewrite** is its corresponding alternative wording.
The **input mask** determines which words the model cannot see. **Targets or labels** determine
which positions contribute their correct answers to the prediction loss.
A masked word need not receive supervision: masked but unscored positions still change the
clues available for predicting other targets.
Here S denotes a sparse set of positions and M a denser set. Each pair of symbols specifies
the input-mask set first and the supervision-target set second.

Ordinary whole-word masking changes both visible clues and training targets, so "masking more"
alone does not identify which factor matters.
The [original dense-mask, sparse-supervision program](../experiments/archive/functional_learning/scripts/densemask_sparselabel_train.py)
identifies eligible content-word groups in the rewrite, retains the sparse-label sampling rule,
and expands the actual mask set. Dense masking has a count limit and includes every label
position. It does not unconditionally erase the entire rewrite, and the source remains visible.

| Condition on paired text | What is hidden in the input | Which positions contribute to the focused loss | Factor distinguished |
| --- | --- | --- | --- |
| (S,S) | Sparsely selected positions | The same sparse set | Input reference for focused training |
| (M,S) | A denser set of rewrite content | The original sparse labels | Fewer local clues without broader label coverage |
| (M,M) | A denser set of rewrite content | A denser label set | Broader supervision coverage than the preceding row |

Unpaired rows continue to use ordinary 15% whole-word masking. Within each update, the acquisition
loss averages the focused and ordinary targets separately, then combines them as
**0.15 * focused cross-entropy + 0.85 * ordinary cross-entropy**.
The coefficient 0.15 is not a text proportion; this is not equivalent to pooling all targets
and taking one mean. The [method description](../methods/principle_guided_training.md) and
[actual configuration](../experiments/configs/stage3.json) retain these definitions.
Only the 995,584 parameters in the existing second residual branch are updated; no third branch is added.

(M,S) retains most of the change in source response seen with dense supervision, showing that
adding all labels is not necessary for that change. But a larger source gap could reflect a
confidence shift or a deterioration under the wrong-source condition.
The [original confidence analysis and specification for subsequent comparisons](notes/functional_learning/temperature_confidence_scale_and_densemask_interpretation.md)
therefore examines both temperature-adjusted loss and candidate rankings.
The [source-readout program](../experiments/archive/functional_learning/scripts/temperature_source_readout.py)
distinguishes probability-scale effects from actual ranking changes.
This is a diagnostic analysis; it does not alter official scoring.

## From the Historical Preservation Proposal to the Final Implementation

Acquisition brings local gains alongside costs to ordinary prediction and some tasks,
motivating preservation of existing functions on more ordinary inputs.
Here, the **parent** is the complete first-generation model, including its learned residual
branches. It is frozen as the **teacher**, and the **student** starts continued training from
that same model. The teacher is not an earlier base model with the added residual branch disabled.
The first generation already used KL; this stage studies how preservation is combined with new acquisition.

The [historical preservation design](notes/functional_learning/preservation_experiment_design_and_evidence.md)
records the motivation and criteria for success and failure at the time, but writes the
divergence as `KL(student || parent)` and estimates extra exposure at approximately 500K words.
**Neither is the final method definition.**
The [subsequent correction to the controls and their interpretation](notes/functional_learning/preservation_controls_and_corrected_interpretation.md)
establishes the actual direction as `KL(parent || student)` and identifies differences in random
sequences and dropout as confounds in early comparisons. The original plan must be read together
with that correction, not used to override the implemented conditions.

The final [preservation trainer](../experiments/archive/functional_learning/scripts/clean_preservation_train.py)
adds an ordinary whole-word-masked presentation of the same paired row, constraining the teacher
and student probability distributions at its target positions.
The code computes KL from student log probabilities and teacher probabilities; both the
coefficient and temperature are 1. Preservation forward passes disable dropout and restore the
random-number state so that the extra computation does not alter subsequent random sequences
in the acquisition branch. The source text remains in the input.
This is preservation on an ordinarily masked input, not "distillation without source evidence."

The [actual training configuration](../experiments/archive/functional_learning/data/clean_preservation_lambda1_eval_full80/train_config.json)
and [completed-run statistics](../experiments/archive/functional_learning/data/clean_preservation_lambda1_eval_full80/train_summary.json)
record 80 updates, 3,162,742 acquisition words, and
**517,332 additional words presented to the student for preservation**.
Including the parent's 86,005,295 words, the complete recipe totals 89,685,369 words;
acquisition-only and ordinary continued training total 89,168,037 words.
Teacher forward passes add computation as well, which the fraction of trainable parameters
does not account for. Reusing the same text does not mean exposure is unchanged.
The [input construction and actual training prefix](../data/RECONSTRUCTION.md) explain the distinction.

## Does the Complete Model Improve on Ordinary Continued Training?

Ordinary continued training tests whether simply learning longer from the same text is sufficient.
(M,M) tests the need for dense supervision; the complete recipe then tests the overall strategy
with preservation on ordinary inputs added.
The table below reports Overall under a consistent evaluation protocol from the
[complete nine-metric results](../results/training_strategy_comparison.csv).
The two columns vary the continued-training seed but **share one pretrained parent model**;
they are not independent pretraining replicates.

| Training strategy | Continued-training seed 62064 | Continued-training seed 62065 |
| --- | ---: | ---: |
| Shared parent model | 42.0240 | Same parent model |
| Ordinary whole-word-masking continued training | 42.0926 | 42.1159 |
| Densely masked input, dense supervision (M,M) | 42.1491 | 42.1684 |
| Densely masked input, sparse supervision (M,S) | 42.2025 | 42.1789 |
| (M,S) plus ordinary-input preservation | 42.2464 | 42.2317 |

(S,S) has separate screening and mechanistic measurements but is absent from this complete
nine-metric table. Ordinary continued training cannot stand in for that single-factor control.
The complete recipe exceeds ordinary continued training by approximately 0.154 / 0.116 Overall
points across the two seeds, and acquisition-only training by approximately 0.044 / 0.053 points.
This supports a strategy improvement conditional on the shared parent model.
It does not imply that every component metric improves or that the incremental gain from
preservation has been isolated as an effect of KL itself.
The [subsequent endpoint-interpretation correction](notes/functional_learning/endpoint_policy_interpretation_corrections.md)
distinguishes early incomplete evaluations from later complete results, and scores obtained with
the historical loading procedure from current scores with the complete branch retained.

The representative model is **Qiushi-Engine-Principle-Guided-Frontier-Advancement**, with Overall
displayed publicly as 42.25. The [model package](../models/principle_guided/README.md) and
[model method record](../models/principle_guided/METHOD.md) correspond to the complete recipe.
The second continued-training run repeats the same strategy for comparison; it is not a third
public model generation.

## What Improved, and What Remains Unattributed?

Source interventions show only a small loss improvement with the correct source, while costs
under unrelated or missing sources are often larger.
Greater source sensitivity therefore cannot directly be called broader or more accurate
contextual understanding. The
[original decomposition of learning gains and losses](notes/functional_learning/evidence_decomposition_and_missing_controls.md)
preserves this diagnostic process. Its then-pending endpoint evaluations should be read in
light of later corrections and the consistent table above.

The [preservation readouts](../results/state_preservation_readouts.csv) show that the complete
recipe reduces prediction drift on ordinary inputs while retaining more of the acquisition
effect under densely masked inputs. Comparisons at fixed, shared positions use familiar training
rows, not transfer to unseen text; those positions also cannot simply be treated as the acquisition
label positions. The [measurement-scope correction](notes/relation_learning/mechanism_readout_scope_corrections.md)
preserves this distinction.

Applying preservation to densely masked inputs suppresses acquisition more strongly, but the
positions being constrained and the effective constraint strength were not independently matched.
The [shared-position gradient diagnostic](../results/preservation_gradient_diagnostic.csv),
measured on 1,003 identical targets, finds a preservation-gradient norm approximately 11.835
times that of the ordinary condition. The complete branches also use different target sets.
Equal coefficients do not imply equal strength.
Together with the complete recipe's additional student presentations and computation, these
differences mean that the evidence supports a joint acquisition-and-preservation strategy.
It does not yet isolate the net contribution of teacher information, preservation positions,
or KL itself.

This stage completes a test from a learning question to training operations and then model
evaluation, without equating controlled tasks directly with natural language.
Its reusable outcomes are separately inspectable input, supervision, and preservation designs,
and a second-generation model with explicit budgets and component-level tradeoffs.
Questions about constraint strength, extra presentations, and transfer across parent models remain open.
Other supported results, failed constructions, and untested directions are linked in the
[research catalog](catalog.md) and [scientific guide](scientific_guide.md).
