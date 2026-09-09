# Attenuation and State-Selective Preservation

Scientific status: conditional mechanism interpretation, with explicit corrections to broader causal claims.

Three interventions answer different questions. Inference-time scaling changes the entire inherited private-adapter bank. Matched rollback attenuates only the exact dense-mask/sparse-label update relative to its starting model. Preservation training constrains predictions on a chosen input and masking state. None substitutes for either of the others.

## Whole-Bank Scaling

At private scale 0.60, the acquisition endpoint's Qwen source movement was 0.112284, close to the preserved endpoint's 0.108668 on that readout. On the same 96-word CDI bank, relative to the historical 82M slow checkpoint, the scaled endpoint had +0.034171 NLL and +27.39 rank; the preserved endpoint had -0.014410 NLL and -22.11 rank. Matching source-response amplitude did not reproduce the preserved endpoint's CDI advantage.

That reference is not the direct preservation teacher. The 86M starting model was already at -0.076645 NLL and -68.11 rank on this historical-reference bank. Preservation therefore retained +0.06223 NLL and +46.00 rank of damage relative to the direct starting model. A negative historical-reference delta must not be described as full recovery to the teacher.

## Matched Update Rollback

Rollback selected by ordinary-text KL gave alpha=0.775, ordinary calibration KL 0.013624, and Qwen source-specificity movement 0.090475. The preserved endpoint gave 0.013327 and 0.106818; the acquisition-only endpoint gave source movement 0.152371. Bounded update movement explains a substantial part of the repair. This does not contradict the whole-bank scaling result: the interventions change different parameter objects.

## Preservation State

On 169 familiar packed rows and 1,003 common-support positions, acquisition-only had dense-common CE gain 0.757052 and rank gain 132.848. Ordinary-state preservation retained 0.682168 and 128.344, while ordinary-rendering KL fell from 0.024486 to 0.009411. Dense-state preservation instead retained only 0.158699 CE gain and 34.825 rank gain despite low teacher distance.

These are state-specific fits to familiar rows, not held-out transfer results. They support a state-dependent tradeoff, not a unique causal role for the teacher trajectory. Equal nominal lambda is not equal effective constraint strength, and the additional student presentation remains a control requirement.

Evidence: [matched rollback](../../documents/functional_learning/data/frozen_bank_rollback_source_control_full/frozen_bank_rollback_source_control.md), [whole-bank scale comparison](../../documents/relation_learning/data/shrinkage_source_cdi_probe/shrinkage_source_cdi_probe.md), [common-support measurements](../../../experiments/archive/functional_learning/data/common_support_endpoint_probe_with_densecorr/common_support_endpoint_probe.json), [preservation admission limits](../relation_learning/preservation_admission_standard.md).
