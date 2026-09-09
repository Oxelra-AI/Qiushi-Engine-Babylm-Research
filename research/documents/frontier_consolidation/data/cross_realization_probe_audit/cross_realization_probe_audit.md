# cross realization probe resolution cross-realization probe audit

Events: 4500, pairs: 2009, docs: 1021
Losses cover all events: True; common loss events: 4500
Counterfactual target-absence flag all true: True; BPE-identity flag all present: True
Target/counterfactual doc overlaps: 0; pair overlaps: 0

## Balance
compact-counterfactual context tokens mean: 0.205556, p95 10.0, min/max -33.0/31.0
abs compact-counterfactual relpos delta mean: 0.000752, p95 0.0

## Robust all-event contrasts
- drop_abs_minus_repeat compact_adv pair: mean=0.026947, pair_id95=[-0.040906,0.090991], P>0=0.799, clusters=2009
  - drop_abs_minus_repeat compact_adv doc: mean=0.026947, doc_id95=[-0.034510,0.088882], P>0=0.819, clusters=1741
  - drop_abs_minus_repeat rep_A pair: mean=0.006681, pair_id95=[0.003282,0.010263], P>0=1.000, clusters=2009
  - drop_abs_minus_repeat rep_A doc: mean=0.006681, doc_id95=[0.003048,0.010213], P>0=0.999, clusters=1741
- drop_copied_word_minus_repeat compact_adv pair: mean=0.031571, pair_id95=[-0.027136,0.089372], P>0=0.864, clusters=2009
  - drop_copied_word_minus_repeat compact_adv doc: mean=0.031571, doc_id95=[-0.024364,0.092962], P>0=0.853, clusters=1741
  - drop_copied_word_minus_repeat rep_A pair: mean=0.001456, pair_id95=[-0.001698,0.004850], P>0=0.818, clusters=2009
  - drop_copied_word_minus_repeat rep_A doc: mean=0.001456, doc_id95=[-0.001758,0.004740], P>0=0.784, clusters=1741
- full_minus_drop_abs compact_adv pair: mean=0.012736, pair_id95=[-0.046339,0.069389], P>0=0.671, clusters=2009
  - full_minus_drop_abs compact_adv doc: mean=0.012736, doc_id95=[-0.049584,0.068792], P>0=0.662, clusters=1741
  - full_minus_drop_abs rep_A pair: mean=-0.002139, pair_id95=[-0.005312,0.000977], P>0=0.099, clusters=2009
  - full_minus_drop_abs rep_A doc: mean=-0.002139, doc_id95=[-0.005505,0.001006], P>0=0.110, clusters=1741
- full_minus_drop_copied_word compact_adv pair: mean=0.008111, pair_id95=[-0.043802,0.059521], P>0=0.604, clusters=2009
  - full_minus_drop_copied_word compact_adv doc: mean=0.008111, doc_id95=[-0.046838,0.060923], P>0=0.612, clusters=1741
  - full_minus_drop_copied_word rep_A pair: mean=0.003087, pair_id95=[-0.000154,0.006295], P>0=0.967, clusters=2009
  - full_minus_drop_copied_word rep_A doc: mean=0.003087, doc_id95=[-0.000020,0.006244], P>0=0.973, clusters=1741
- full_minus_repeat compact_adv pair: mean=0.039682, pair_id95=[-0.021966,0.100469], P>0=0.894, clusters=2009
  - full_minus_repeat compact_adv doc: mean=0.039682, doc_id95=[-0.019115,0.101698], P>0=0.907, clusters=1741
  - full_minus_repeat rep_A pair: mean=0.004543, pair_id95=[0.001277,0.007900], P>0=0.996, clusters=2009
  - full_minus_repeat rep_A doc: mean=0.004543, doc_id95=[0.001363,0.007763], P>0=0.997, clusters=1741
- adjbreak_minus_repeat compact_adv pair: mean=-0.005124, pair_id95=[-0.060265,0.052783], P>0=0.427, clusters=2009
  - adjbreak_minus_repeat compact_adv doc: mean=-0.005124, doc_id95=[-0.057459,0.049081], P>0=0.419, clusters=1741
  - adjbreak_minus_repeat rep_A pair: mean=0.001407, pair_id95=[-0.001933,0.004901], P>0=0.783, clusters=2009
  - adjbreak_minus_repeat rep_A doc: mean=0.001407, doc_id95=[-0.002102,0.004839], P>0=0.783, clusters=1741

JSON: `experiments/archive/frontier_consolidation/data/cross_realization_probe_audit/cross_realization_probe_audit.json`
