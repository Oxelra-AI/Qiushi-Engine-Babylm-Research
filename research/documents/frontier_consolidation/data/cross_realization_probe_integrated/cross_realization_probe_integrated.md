# cross realization probe resolution cross-realization probe integration

Merged events: 4500 / common 4500 / union 4500
Event sets identical: True; metadata matched: True

## All-event arm means
- full_compact_100M: compact_adv=0.189814, source_adv=0.163787, compact_minus_source=-0.026027, rep_A=0.273255, n=4500, pieces=7715
- drop_abs_100M: compact_adv=0.177078, source_adv=0.154530, compact_minus_source=-0.022548, rep_A=0.275394, n=4500, pieces=7715
- drop_copied_word_100M: compact_adv=0.181702, source_adv=0.170591, compact_minus_source=-0.011112, rep_A=0.270169, n=4500, pieces=7715
- repeat_100M: compact_adv=0.150132, source_adv=0.147918, compact_minus_source=-0.002214, rep_A=0.268713, n=4500, pieces=7715
- adjbreak_100M: compact_adv=0.145007, source_adv=0.162707, compact_minus_source=0.017699, rep_A=0.270120, n=4500, pieces=7715

## Decision-bearing all-event contrasts
- drop_abs_minus_repeat compact_adv: mean=0.026947, boot95=[-0.039176,0.092307], P>0=0.809
  - drop_abs_minus_repeat rep_A: mean=0.006681, boot95=[0.003240,0.010146], P>0=1.000
- drop_copied_word_minus_repeat compact_adv: mean=0.031571, boot95=[-0.032543,0.089683], P>0=0.837
  - drop_copied_word_minus_repeat rep_A: mean=0.001456, boot95=[-0.002090,0.004757], P>0=0.792
- full_minus_drop_abs compact_adv: mean=0.012736, boot95=[-0.047267,0.070997], P>0=0.667
  - full_minus_drop_abs rep_A: mean=-0.002139, boot95=[-0.005138,0.000920], P>0=0.080
- full_minus_drop_copied_word compact_adv: mean=0.008111, boot95=[-0.043374,0.062002], P>0=0.628
  - full_minus_drop_copied_word rep_A: mean=0.003087, boot95=[-0.000038,0.006339], P>0=0.971
- full_minus_repeat compact_adv: mean=0.039682, boot95=[-0.023679,0.103036], P>0=0.886
  - full_minus_repeat rep_A: mean=0.004543, boot95=[0.001334,0.007934], P>0=0.996
- adjbreak_minus_repeat compact_adv: mean=-0.005124, boot95=[-0.061539,0.053736], P>0=0.456
  - adjbreak_minus_repeat rep_A: mean=0.001407, boot95=[-0.001956,0.004763], P>0=0.793

## Eval-set compact-advantage contrasts
- doc_disjoint_all_accepted: drop_abs-repeat mean=0.106346, boot95=[0.013561,0.205364], P>0=0.987; full-drop_abs mean=-0.018312, boot95=[-0.107942,0.069343], P>0=0.341; full-drop_copied mean=-0.001082, boot95=[-0.085030,0.081282], P>0=0.451
- doc_disjoint_quality: drop_abs-repeat mean=0.112785, boot95=[-0.031914,0.256473], P>0=0.936; full-drop_abs mean=0.010237, boot95=[-0.135255,0.161217], P>0=0.542; full-drop_copied mean=0.010419, boot95=[-0.114344,0.139435], P>0=0.536
- source_disjoint_quality: drop_abs-repeat mean=-0.048048, boot95=[-0.153901,0.047338], P>0=0.152; full-drop_abs mean=0.030109, boot95=[-0.059890,0.123923], P>0=0.727; full-drop_copied mean=0.015471, boot95=[-0.069198,0.094725], P>0=0.595
- train_fixed_probe: drop_abs-repeat mean=-0.058686, boot95=[-0.192158,0.077728], P>0=0.169; full-drop_abs mean=0.040601, boot95=[-0.081763,0.158596], P>0=0.742; full-drop_copied mean=0.008853, boot95=[-0.117808,0.139573], P>0=0.561

## Scientific reading
This is local saved-checkpoint inference, not endpoint evidence. A full-compact-unique same-sequence target-complementarity reading would require full_compact to exceed both deletion arms on the cross-realization compact-context and representation interactions. If drop_abs/drop_copied remain above repeat and full adds little, the result instead supports compact input/contextual enrichment around shared/copied content as the local object, while downstream selected scores still decide whether that object matters for BabyLM competence.

JSON: `experiments/archive/frontier_consolidation/data/cross_realization_probe_integrated/cross_realization_probe_integrated.json`
CSV: `experiments/archive/frontier_consolidation/data/cross_realization_probe_integrated/all_event_arm_contrasts.csv`
