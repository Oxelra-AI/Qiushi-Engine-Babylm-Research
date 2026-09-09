# Functional Access and Preservation

"Functional reach" describes which new objects and queries can use a learned computation. It is distinct from remembering answers to familiar questions. A controlled four-choice relation task measures familiar and unseen symbols separately and intervenes on the internal signals involved in selection.

Ordinary continuation, static relation-supervision weights and alternating supervision produce different familiar/unseen performance patterns. Internal replacement, erasure and position rotation test the signals' functional roles. The selection rate for a designated option after rotation is not accuracy on the original question; the two must not be combined into a single comparison of accuracy gains.

The [shared result table](../results/interface_reach.csv) retains conditions, units and measurements; [Chapter 3 of the report](../reports/zh/chapters/04_principles.tex) explains the internal interventions and counterexamples. These results come from a specific controlled task and do not establish the same internal mechanism throughout the final 36M model.

The implication for later training design is a measurement requirement: after performance on old questions recovers, test whether new inputs can still use the capability, not just whether the model's outputs remain broadly close to its earlier outputs.
