# endpoint resolved dualview hardsurface detached-private hard-surface readout

Status: **DETACHED_PRIVATE_HARD_SURFACE_READOUT_DONE**

This is a posthoc mechanism readout on existing checkpoints. It does not train a model and does not change official submission scoring.

## Targets
- `mlm_only_20M` (mlm_only): ready=True model=`experiments/archive/frontier_consolidation/training/runs/dualview_mlm_only_20M_seed43022/hf_model/final`
- `sep_sparse20_aligned_20M` (separated_true_alignment): ready=True model=`experiments/archive/frontier_consolidation/training/runs/sep_sparse20_aligned_20M_seed43022/hf_model/final`
- `sep_sparse20_shuffled_20M` (separated_shuffled_alignment): ready=True model=`experiments/archive/frontier_consolidation/training/runs/sep_sparse20_shuffled_20M_seed43022/hf_model/final`
- `coupled_sparse20_aligned_20M` (coupled_true_alignment): ready=True model=`experiments/archive/frontier_consolidation/training/runs/coupled_sparse20_aligned_20M_seed43022/hf_model/final`

## GlobalPIQA fixed hard subset
- `mlm_only_20M`: parallel=22.33009708737864 hard52_acc=3.8461538461538463 hard52_mean_top_minus_correct=1.622895339167021 hard52_rank_counts={'4': 18, '3': 25, '2': 7, '1': 2}
- `sep_sparse20_aligned_20M`: parallel=22.33009708737864 hard52_acc=7.6923076923076925 hard52_mean_top_minus_correct=1.53736199530253 hard52_rank_counts={'4': 16, '2': 15, '1': 4, '3': 17}
- `sep_sparse20_shuffled_20M`: parallel=20.388349514563107 hard52_acc=1.9230769230769231 hard52_mean_top_minus_correct=1.6378646220456103 hard52_rank_counts={'4': 17, '2': 12, '3': 22, '1': 1}
- `coupled_sparse20_aligned_20M`: parallel=22.33009708737864 hard52_acc=9.615384615384615 hard52_mean_top_minus_correct=1.443233044866988 hard52_rank_counts={'4': 18, '2': 13, '3': 16, '1': 5}
  - delta `sep_aligned_minus_mlm_only` parallel: acc=0.0 hard52_acc=3.8461538461538463 hard52_mean_margin=-0.08553334386449096 hard52_rank1=2
  - delta `sep_aligned_minus_sep_shuffled` parallel: acc=1.9417475728155331 hard52_acc=5.769230769230769 hard52_mean_margin=-0.10050262674308041 hard52_rank1=3
  - delta `sep_aligned_minus_coupled_aligned` parallel: acc=0.0 hard52_acc=-1.9230769230769225 hard52_mean_margin=0.09412895043554204 hard52_rank1=-1
  - delta `sep_shuffled_minus_mlm_only` parallel: acc=-1.9417475728155331 hard52_acc=-1.9230769230769231 hard52_mean_margin=0.014969282878589452 hard52_rank1=-1
  - delta `coupled_aligned_minus_mlm_only` parallel: acc=0.0 hard52_acc=5.769230769230768 hard52_mean_margin=-0.179662294300033 hard52_rank1=3

## Interpretation boundary
- Separated aligned versus MLM-only GlobalPIQA_parallel delta is 0.0 points; hard52 accuracy delta is 3.8461538461538463 points and hard52 mean top-minus-correct delta is -0.08553334386449096 nats.
- True alignment versus shuffled control on GlobalPIQA_parallel has delta 1.9417475728155331 points; hard52 accuracy delta 5.769230769230769 and hard52 mean-margin delta -0.10050262674308041 nats.
- A positive aligned-vs-MLM delta alone is not enough: the route only supports a correspondence-learning mechanism if aligned beats shuffled and coupled controls on fixed natural hard surfaces, not merely on aggregate cheap7 or broad early-training columns.
- This readout intentionally uses fixed fw globalpiqa relevant substrate/full ewok interaction synthesis hard surfaces and existing checkpoints; it must not be used to tune official examples or define a submission-time scoring rule.

JSON: `experiments/archive/representation_and_objectives/data/detached_private_hard_surface_readout/detached_private_hard_surface_summary.json`
