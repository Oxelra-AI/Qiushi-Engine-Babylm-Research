# endpoint resolved dualview hardsurface detached-private hard-surface readout

Status: **DETACHED_PRIVATE_HARD_SURFACE_READOUT_DONE**

This is a posthoc mechanism readout on existing checkpoints. It does not train a model and does not change official submission scoring.

## Targets
- `sep_sparse20_shuffled_20M` (separated_shuffled_alignment): ready=True model=`experiments/archive/frontier_consolidation/training/runs/sep_sparse20_shuffled_20M_seed43022/hf_model/final`
- `coupled_sparse20_aligned_20M` (coupled_true_alignment): ready=True model=`experiments/archive/frontier_consolidation/training/runs/coupled_sparse20_aligned_20M_seed43022/hf_model/final`

## EWoK full ewok interaction synthesis stable-failure subset
- `sep_sparse20_shuffled_20M`: n=1471 acc=0.21210061182868797 saved_wrong=1159 stable_failure=1032 stable_frac_all=0.7015635622025833 within_both_positive_wrong_frac=0.0008628127696289905 interaction_wrong_mean=-1.6099947531106122
- `coupled_sparse20_aligned_20M`: n=1471 acc=0.512576478585996 saved_wrong=717 stable_failure=430 stable_frac_all=0.29231815091774305 within_both_positive_wrong_frac=0.005578800557880056 interaction_wrong_mean=-0.06628491602204169

## Interpretation boundary
- A positive aligned-vs-MLM delta alone is not enough: the route only supports a correspondence-learning mechanism if aligned beats shuffled and coupled controls on fixed natural hard surfaces, not merely on aggregate cheap7 or broad early-training columns.
- This readout intentionally uses fixed fw globalpiqa relevant substrate/full ewok interaction synthesis hard surfaces and existing checkpoints; it must not be used to tune official examples or define a submission-time scoring rule.

JSON: `experiments/archive/representation_and_objectives/data/detached_private_ewok_stable_subset_remaining/detached_private_hard_surface_summary.json`
