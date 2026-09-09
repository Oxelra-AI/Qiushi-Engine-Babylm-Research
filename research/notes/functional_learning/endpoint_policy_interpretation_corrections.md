# Endpoint Policy Interpretation Corrections

Scientific status: corrected measurement interpretation; no new scoring.

The ordinary, dense-mask/sparse-label, and preserved endpoints form a matched policy ladder. A partially evaluated endpoint does not have an Overall score: the missing components must not be imputed from another seed, loading path, or shortened evaluation. The later complete ordinary seed62065 endpoint was 42.1159198215161.

Several precise corrections matter for reading that ladder:

- Dense-mask/sparse-label seed62065 SuperGLUE is 68.93486456221126, not 69.02.
- The local WSC loading-compatibility checks are 61.54 for the starting package and 63.46153846153846 for the preserved package. These checks are not replacements for the corresponding main-table components.
- Historical stripped-loader Overall 42.1210 and repaired-loader Overall 42.023967991315104 are different measurement coordinates. They cannot be combined into a single improvement calculation.
- The preserved endpoints have Overall 42.246412332209445 and 42.231731132658. Seeds 62064 and 62065 are continuation controls from the same trained starting model, not independent from-scratch replications.
- The clean-minus-acquisition residual is +0.044 and +0.053 at the precision used in the scientific notes. It belongs to the combined preservation policy, not an isolated KL term: an extra student presentation and the magnitude of parameter movement also differ.

The intermediate record correctly left an incomplete ordinary endpoint unscored. Its later completion changes the availability of the matched contrast, not the identity of any already measured endpoint.

Evidence: [complete admission record](../../../experiments/archive/functional_learning/data/strict_split_eval_admission_o_complete/strict_split_eval_admission.json), [two-seed synthesis](../relation_learning/twoseed_synthesis_and_provenance.md), [preservation controls](preservation_controls_and_corrected_interpretation.md).
