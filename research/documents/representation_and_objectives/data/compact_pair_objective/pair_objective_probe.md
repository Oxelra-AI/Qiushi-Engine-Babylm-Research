# compact directional internal result pair-objective probe

Run root: `experiments/archive/representation_and_objectives/training/runs/directional_fork_compact_seed43022`
Models: ff, fr, prefix_forward, prefix_reverse, rf, rr

## Learning changes
- second forward epoch after forward prefix: {'side_loss': {'prefix': 6.31740143525556, 'final': 5.962672632541409, 'final_minus_prefix_loss': -0.35472880271415086, 'lower_is_better': True}, 'copied_loss': {'prefix': 6.278633145554203, 'final': 5.9172397911956205, 'final_minus_prefix_loss': -0.3613933543585821, 'lower_is_better': True}, 'noncopied_loss': {'prefix': 6.492629701605409, 'final': 6.168023915399043, 'final_minus_prefix_loss': -0.3246057862063667, 'lower_is_better': True}}
- reverse epoch after forward prefix: {'side_loss': {'prefix': 5.906994153789066, 'final': 5.55264537620384, 'final_minus_prefix_loss': -0.3543487775852263, 'lower_is_better': True}, 'copied_loss': {'prefix': 6.043836602660247, 'final': 5.701365173116262, 'final_minus_prefix_loss': -0.342471429543985, 'lower_is_better': True}, 'noncopied_loss': {'prefix': 5.696427388485558, 'final': 5.323802302795217, 'final_minus_prefix_loss': -0.37262508569034125, 'lower_is_better': True}}
- second reverse epoch after reverse prefix: {'side_loss': {'prefix': 5.865448547105254, 'final': 5.546715212343858, 'final_minus_prefix_loss': -0.3187333347613963, 'lower_is_better': True}, 'copied_loss': {'prefix': 6.051569878153456, 'final': 5.693963285308689, 'final_minus_prefix_loss': -0.3576065928447676, 'lower_is_better': True}, 'noncopied_loss': {'prefix': 5.579053740843809, 'final': 5.320136758968128, 'final_minus_prefix_loss': -0.25891698187568046, 'lower_is_better': True}}
- forward epoch after reverse prefix: {'side_loss': {'prefix': 6.415609967967863, 'final': 5.975503804111481, 'final_minus_prefix_loss': -0.4401061638563819, 'lower_is_better': True}, 'copied_loss': {'prefix': 6.366806941553896, 'final': 5.928284176681021, 'final_minus_prefix_loss': -0.4385227648728751, 'lower_is_better': True}, 'noncopied_loss': {'prefix': 6.6361941045656145, 'final': 6.18893115713798, 'final_minus_prefix_loss': -0.4472629474276344, 'lower_is_better': True}}

## Interactions

{
  "available_final_arms": {
    "ff": true,
    "fr": true,
    "rr": true,
    "rf": true
  },
  "status": "complete",
  "forward": {
    "side_loss": {
      "values": {
        "ff": 5.962672632541409,
        "fr": 6.073824161553043,
        "rr": 6.09856192792306,
        "rf": 5.975503804111481
      },
      "oneway_avg_ff_rr": 6.030617280232235,
      "mixed_avg_fr_rf": 6.024663982832262,
      "mixed_minus_oneway_loss": -0.005953297399972968,
      "lower_is_better": true,
      "direction_matched_second_epoch_effects": {
        "RF_minus_FF_for_forward_second_epoch": 0.012831171570072186,
        "FR_minus_RR_for_reverse_second_epoch": -0.024737766370017233
      },
      "retention_or_cross_exposure_effects": {
        "FR_minus_RR": -0.024737766370017233,
        "RF_minus_FF": 0.012831171570072186
      }
    },
    "copied_loss": {
      "values": {
        "ff": 5.9172397911956205,
        "fr": 6.018303426389702,
        "rr": 6.0391428694631335,
        "rf": 5.928284176681021
      },
      "oneway_avg_ff_rr": 5.978191330329377,
      "mixed_avg_fr_rf": 5.973293801535361,
      "mixed_minus_oneway_loss": -0.0048975287940162815,
      "lower_is_better": true,
      "direction_matched_second_epoch_effects": {
        "RF_minus_FF_for_forward_second_epoch": 0.011044385485400099,
        "FR_minus_RR_for_reverse_second_epoch": -0.020839443073431774
      },
      "retention_or_cross_exposure_effects": {
        "FR_minus_RR": -0.020839443073431774,
        "RF_minus_FF": 0.011044385485400099
      }
    },
    "noncopied_loss": {
      "values": {
        "ff": 6.168023915399043,
        "fr": 6.324771578735613,
        "rr": 6.367129323654957,
        "rf": 6.18893115713798
      },
      "oneway_avg_ff_rr": 6.267576619527,
      "mixed_avg_fr_rf": 6.256851367936797,
      "mixed_minus_oneway_loss": -0.010725251590203655,
      "lower_is_better": true,
      "direction_matched_second_epoch_effects": {
        "RF_minus_FF_for_forward_second_epoch": 0.020907241738937543,
        "FR_minus_RR_for_reverse_second_epoch": -0.042357744919343965
      },
      "retention_or_cross_exposure_effects": {
        "FR_minus_RR": -0.042357744919343965,
        "RF_minus_FF": 0.020907241738937543
      }
    }
  },
  "reverse": {
    "side_loss": {
      "values": {
        "ff": 5.6295743143655,
        "fr": 5.55264537620384,
        "rr": 5.546715212343858,
        "rf": 5.602914923796911
      },
      "oneway_avg_ff_rr": 5.588144763354679,
      "mixed_avg_fr_rf": 5.577780150000375,
      "mixed_minus_oneway_loss": -0.010364613354303387,
      "lower_is_better": true,
      "direction_matched_second_epoch_effects": {
        "RF_minus_FF_for_forward_second_epoch": -0.02665939056858857,
        "FR_minus_RR_for_reverse_second_epoch": 0.005930163859981796
      },
      "retention_or_cross_exposure_effects": {
        "FR_minus_RR": 0.005930163859981796,
        "RF_minus_FF": -0.02665939056858857
      }
    },
    "copied_loss": {
      "values": {
        "ff": 5.702110072686076,
        "fr": 5.701365173116262,
        "rr": 5.693963285308689,
        "rf": 5.698022140937375
      },
      "oneway_avg_ff_rr": 5.698036678997383,
      "mixed_avg_fr_rf": 5.699693657026819,
      "mixed_minus_oneway_loss": 0.0016569780294357628,
      "lower_is_better": true,
      "direction_matched_second_epoch_effects": {
        "RF_minus_FF_for_forward_second_epoch": -0.004087931748701479,
        "FR_minus_RR_for_reverse_second_epoch": 0.007401887807573004
      },
      "retention_or_cross_exposure_effects": {
        "FR_minus_RR": 0.007401887807573004,
        "RF_minus_FF": -0.004087931748701479
      }
    },
    "noncopied_loss": {
      "values": {
        "ff": 5.5179596792548775,
        "fr": 5.323802302795217,
        "rr": 5.320136758968128,
        "rf": 5.456568382679848
      },
      "oneway_avg_ff_rr": 5.419048219111502,
      "mixed_avg_fr_rf": 5.390185342737532,
      "mixed_minus_oneway_loss": -0.028862876373970003,
      "lower_is_better": true,
      "direction_matched_second_epoch_effects": {
        "RF_minus_FF_for_forward_second_epoch": -0.06139129657502984,
        "FR_minus_RR_for_reverse_second_epoch": 0.003665543827088946
      },
      "retention_or_cross_exposure_effects": {
        "FR_minus_RR": 0.003665543827088946,
        "RF_minus_FF": -0.06139129657502984
      }
    }
  },
  "copied_vs_noncopied_interaction": {
    "forward": -0.005827722796187373,
    "reverse": -0.030519854403405766
  }
}

JSON: `experiments/archive/representation_and_objectives/data/compact_pair_objective/pair_objective_probe.json`
