# temperature confidence scale and densemask interpretation source-margin decomposition

This decomposes dense and sparse source-readout changes into correct-source support, wrong/absent-source degradation, and rank/order movement. It uses already-scored temperature confidence scale and densemask interpretation rows and is not official scoring.

## sparse_focus_seed62064
- Qwen Δspecific T1=0.0016756124773787116, Tfit=0.0020557124912738848, Δrank=-5.413194444444444
- Qwen condition ΔNLL T1: correct=-0.07786673564877775, wrong=-0.07619112317139903, view-only=-0.0798788860719651
- Qwen condition Δrank: correct=-2.6544791666666665, wrong=-8.067673611111111, view-only=-7.616562499999999
- Common swing ΔT1=-0.04880538066228234, ΔTfit=-0.04505195023048483, Δrank=2.2390476190476343; both-correct=21/25, both-rank=21/25
- Common condition Δmargin T1: {'source_original': 0.008812616439092702, 'source_altered': -0.057617997101375044, 'no_source': 0.01827490556807744}

## dense_focus_seed62064
- Qwen Δspecific T1=0.1743143003092458, Tfit=0.15194935254894923, Δrank=69.89072916666667
- Qwen condition ΔNLL T1: correct=-0.032093717929803665, wrong=0.14222058237944213, view-only=0.08131169229538902
- Qwen condition Δrank: correct=9.318125, wrong=79.20885416666667, view-only=44.43524305555556
- Common swing ΔT1=0.6033858109372003, ΔTfit=0.538546733387879, Δrank=115.20476190476192; both-correct=21/25, both-rank=20/25
- Common condition Δmargin T1: {'source_original': 0.2636575908604122, 'source_altered': 0.339728220076788, 'no_source': 0.02023888815016975}

## dense_focus_seed62065
- Qwen Δspecific T1=0.17833735039871598, Tfit=0.15540364948243626, Δrank=71.86121527777777
- Qwen condition ΔNLL T1: correct=-0.03286968881264329, wrong=0.1454676615860727, view-only=0.08365386076375013
- Qwen condition Δrank: correct=9.160763888888889, wrong=81.02197916666667, view-only=45.253923611111105
- Common swing ΔT1=0.5960553641546338, ΔTfit=0.531859248904955, Δrank=116.42809523809525; both-correct=21/25, both-rank=20/25
- Common condition Δmargin T1: {'source_original': 0.25727188519069116, 'source_altered': 0.33878347896394273, 'no_source': 0.02088270028432217}

## Seed agreement
{
  "qwen_source_triplet_seed_agreement": {
    "delta_specific_advantage_T1_vs_parent": {
      "n": 480,
      "pearson": 0.9996993636067567,
      "mean_abs_diff": 0.013404183732345677,
      "same_sign_fraction": 0.9875
    },
    "delta_specific_advantage_Tfit_vs_parent": {
      "n": 480,
      "pearson": 0.9996902656374643,
      "mean_abs_diff": 0.011763521408041289,
      "same_sign_fraction": 0.9854166666666667
    },
    "delta_specific_rank_advantage_vs_parent": {
      "n": 480,
      "pearson": 0.9996737035292319,
      "mean_abs_diff": 3.441875000000001,
      "same_sign_fraction": 0.9916666666666667
    },
    "delta_total_rank_advantage_vs_parent": {
      "n": 480,
      "pearson": 0.9994717010638379,
      "mean_abs_diff": 3.217222222222223,
      "same_sign_fraction": 0.9979166666666667
    }
  },
  "common_source_follow_seed_agreement": {
    "delta_source_follow_swing_T1_vs_parent": {
      "n": 25,
      "pearson": 0.9998910778425595,
      "mean_abs_diff": 0.0150590404726211,
      "same_sign_fraction": 1.0
    },
    "delta_source_follow_swing_Tfit_vs_parent": {
      "n": 25,
      "pearson": 0.9998905143070206,
      "mean_abs_diff": 0.01372953597988392,
      "same_sign_fraction": 1.0
    },
    "delta_source_follow_rank_swing_vs_parent": {
      "n": 25,
      "pearson": 0.9999323148133281,
      "mean_abs_diff": 3.7766666666666695,
      "same_sign_fraction": 1.0
    }
  }
}

## Interpretation
- **qwen_dense_mechanism**: Dense's Qwen source-specific margin is not a pure correct-source likelihood gain. In both dense seeds, correct-source NLL improves slightly while wrong-source and view-only NLL/ranks worsen substantially; the correct-vs-wrong advantage therefore reflects increased penalty for wrong evidence as well as small correct-evidence support.
- **rank_order_component**: Dense changes ranks/order on source probes, so the effect is not reducible to a global positive temperature scale. But rank advantage also increases mainly because wrong-source target ranks worsen more than correct-source ranks, not because every correct-source target rank improves.
- **method_design**: The next method should preserve rank-changing evidence discrimination while reducing broad rank/NLL degradation on CDI, grammar, and supervised transfer. Dense-mask/sparse-label should therefore be evaluated on condition-wise and rank-wise decomposition, not only aggregate margins.
