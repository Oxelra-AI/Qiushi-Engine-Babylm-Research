# Figure Estimands and Interpretation Limits

Scientific status: retained scientific content from figure-caption revisions. Suggested captions are not additional experimental evidence.

## Relation Studies

- Target-class crossing is a two-category comparison within each panel, not a three-category continuous curve. The exact overlap definition belongs to its probe and must not be replaced by a generic seen/unseen-symbol definition.
- Entity's horizontal axis is the count of relevant changes to the queried entity, 0-5, not training progress.
- Locality comparisons alternate R-C, RS-C, V-C, and VS-C. Local arms use three seeds and split arms two; each error bar uses its own available-seed standard deviation.
- C/HV/HM/V/R are discrete relation-mixture arms measured at seed43022. The figure alone does not establish a continuous saturation curve, a fitted knee, or a replication interval.
- The natural-reach comparison concerns compact and Wikipedia overlap/nonoverlap target classes. It is not a synthetic binding experiment on familiar versus unseen symbols. The two experiments can motivate related hypotheses but cannot share an evidence label.

## Acquisition and Preservation

The source-specificity panels both show outcomes: temperature-refitted and rank-based source readouts. Neither is a plotted prospective prediction. The prior directional prediction should be described separately from its later test.

The common-support figure reports three separate quantities: dense-state CE gain, rank gain, and teacher distance on 1,003 positions from 169 familiar rows. It is not a two-dimensional optimization region and is not held-out generalization evidence. The ordinary/acquisition/preservation ladder reports either Overall or component-specific increments; these estimands must remain distinct. A green-positive/red-negative heatmap must keep that sign convention.

The proposed two-bar CDI caption called its zero the first-generation model. That label is not adopted here: the attenuation evidence explicitly uses the historical 82M checkpoint for its 96-word CDI bank and records residual damage relative to the direct 86M teacher. The exact bank and reference must be retained before interpreting a negative NLL delta as recovery. The caption proposal cannot override numerical provenance.

## AoA Timing

The 30M-word figure summarizes Pearson correlations, not an age-versus-order scatter plot: control r=-0.050, p=0.407; reordered schedule r=+0.207, p=0.009; enrichment masking r=+0.024, p=0.695. For words fitted in both arms, the schedule-minus-control acquisition-order shift correlates with observed child acquisition age (r=+0.330, p=5.4e-5). This is distinct from each arm's own model-child correlation. These findings concern one seed and 30M-word exposure; they do not replace the final approximately 89.7M-word endpoint measurements.

Related evidence: [prospective readout definitions](relation_locality_prospective_readouts.md), [attenuation references](../functional_learning/attenuation_and_preservation_rationale.md), [AoA design and implementation distinction](aoa_masking_design_vs_implementation.md).
