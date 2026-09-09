# comparative query binding pilot comparative query-binding pilot

Summary JSON: `experiments/archive/representation_and_objectives/data/comparative_query_binding_pilot/comparative_query_binding_summary.json`
Rows CSV: `experiments/archive/representation_and_objectives/data/comparative_query_binding_pilot/comparative_query_binding_rows.csv`
Hard examples CSV: `experiments/archive/representation_and_objectives/data/comparative_query_binding_pilot/comparative_query_binding_hard_examples.csv`

This pilot repairs the prior malformed entity-exchange frames by keeping the natural comparative sentence and both alternatives visible, then asking which alternative is the more/less (or inverse) property-bearing option.

## Counts
{
  "rows": 26,
  "scale_specific_failures": 0,
  "base_specific_failures": 1,
  "source_counts": {
    "childes": 2,
    "cleanqwen_fineweb_compact_view_reinvest": 12,
    "open_subtitles": 5,
    "qwen_pair_packed": 4,
    "simple_wiki": 1,
    "gutenberg": 1,
    "bnc_spoken": 1
  }
}

## Scale
{
  "all": {
    "n": 26,
    "both_correct_frac": 0.038461538461538464,
    "M_pair": {
      "n": 26,
      "mean": -0.3889734051548518,
      "median": -0.09308682382106781,
      "p10": -1.940943717956543,
      "p90": 0.24760127067565918,
      "min": -2.161276698112488,
      "max": 0.5095475316047668
    },
    "margin1": {
      "n": 26,
      "mean": -1.973851165519311,
      "median": -0.6961919292807579,
      "p10": -6.9327967166900635,
      "p90": 2.0304601192474365,
      "min": -12.09918475151062,
      "max": 4.050250172615051
    },
    "margin2": {
      "n": 26,
      "mean": 1.584877760364459,
      "median": 0.5818583965301514,
      "p10": -2.251568555831909,
      "p90": 6.031466260552406,
      "min": -4.073350667953491,
      "max": 10.099315881729126
    }
  },
  "development": {
    "n": 20,
    "both_correct_frac": 0.05,
    "M_pair": {
      "n": 20,
      "mean": -0.3241728536784649,
      "median": -0.09705197811126709,
      "p10": -0.9841179549694061,
      "p90": 0.3188297748565674,
      "min": -1.9998688697814941,
      "max": 0.5095475316047668
    },
    "margin1": {
      "n": 20,
      "mean": -2.0836272560060025,
      "median": -0.6961919292807579,
      "p10": -6.9327967166900635,
      "p90": 2.0304601192474365,
      "min": -12.09918475151062,
      "max": 4.050250172615051
    },
    "margin2": {
      "n": 20,
      "mean": 1.7594544023275376,
      "median": 0.5784761980175972,
      "p10": -2.251568555831909,
      "p90": 6.791320085525513,
      "min": -4.073350667953491,
      "max": 10.099315881729126
    }
  },
  "heldout": {
    "n": 6,
    "both_correct_frac": 0.0,
    "M_pair": {
      "n": 6,
      "mean": -0.6049752434094747,
      "median": -0.05874611437320709,
      "p10": -2.161276698112488,
      "p90": 0.029708147048950195,
      "min": -2.161276698112488,
      "max": 0.2258916199207306
    },
    "margin1": {
      "n": 6,
      "mean": -1.6079308638970058,
      "median": -0.7321740090847015,
      "p10": -5.7636027336120605,
      "p90": 0.9920957684516907,
      "min": -5.7636027336120605,
      "max": 2.1634856164455414
    },
    "margin2": {
      "n": 6,
      "mean": 1.002955620487531,
      "median": 0.7462046146392822,
      "p10": -1.9375939965248108,
      "p90": 3.413939118385315,
      "min": -1.9375939965248108,
      "max": 4.156920433044434
    }
  },
  "by_rel": {
    "higher": {
      "n": 1,
      "both_correct_frac": 0.0,
      "M_pair": {
        "n": 1,
        "mean": -0.06571835279464722,
        "median": -0.06571835279464722,
        "p10": -0.06571835279464722,
        "p90": -0.06571835279464722,
        "min": -0.06571835279464722,
        "max": -0.06571835279464722
      }
    },
    "less": {
      "n": 5,
      "both_correct_frac": 0.2,
      "M_pair": {
        "n": 5,
        "mean": -0.306463885307312,
        "median": -0.05794340372085571,
        "p10": -2.161276698112488,
        "p90": 0.5095475316047668,
        "min": -2.161276698112488,
        "max": 0.5095475316047668
      }
    },
    "more": {
      "n": 20,
      "both_correct_frac": 0.0,
      "M_pair": {
        "n": 20,
        "mean": -0.42576353773474696,
        "median": -0.11981044709682465,
        "p10": -1.606682300567627,
        "p90": 0.2258916199207306,
        "min": -1.9998688697814941,
        "max": 0.34159040451049805
      }
    }
  }
}

## Base
{
  "all": {
    "n": 26,
    "both_correct_frac": 0.0,
    "M_pair": {
      "n": 26,
      "mean": -0.12479626548548158,
      "median": -0.015049226582050323,
      "p10": -0.7741284370422363,
      "p90": 0.19874334335327148,
      "min": -1.1251392364501953,
      "max": 0.2598012685775757
    },
    "margin1": {
      "n": 26,
      "mean": -1.9554732866370335,
      "median": -1.2360002025961876,
      "p10": -6.336339712142944,
      "p90": 1.3076478093862534,
      "min": -9.689639568328857,
      "max": 3.0311511158943176
    },
    "margin2": {
      "n": 26,
      "mean": 1.8306770211515517,
      "median": 1.0689126159995794,
      "p10": -1.3137344121932983,
      "p90": 5.981675781309605,
      "min": -3.003329873085022,
      "max": 8.564500331878662
    }
  },
  "development": {
    "n": 20,
    "both_correct_frac": 0.0,
    "M_pair": {
      "n": 20,
      "mean": -0.10860971179790795,
      "median": 0.010867320001125336,
      "p10": -0.29525065422058105,
      "p90": 0.22824397683143616,
      "min": -1.1251392364501953,
      "max": 0.2598012685775757
    },
    "margin1": {
      "n": 20,
      "mean": -2.2021455482114107,
      "median": -1.638860285282135,
      "p10": -6.336339712142944,
      "p90": 1.3076478093862534,
      "min": -9.689639568328857,
      "max": 3.0311511158943176
    },
    "margin2": {
      "n": 20,
      "mean": 2.0935358364135026,
      "median": 1.453389398753643,
      "p10": -1.3137344121932983,
      "p90": 6.041089057922363,
      "min": -3.003329873085022,
      "max": 8.564500331878662
    }
  },
  "heldout": {
    "n": 6,
    "both_correct_frac": 0.0,
    "M_pair": {
      "n": 6,
      "mean": -0.17875144444406033,
      "median": -0.03671290911734104,
      "p10": -0.7741284370422363,
      "p90": -0.006086602807044983,
      "min": -0.7741284370422363,
      "max": 0.009746149182319641
    },
    "margin1": {
      "n": 6,
      "mean": -1.1332324147224426,
      "median": -0.8238567933440208,
      "p10": -3.9485673904418945,
      "p90": -0.11263465881347656,
      "min": -3.9485673904418945,
      "max": 1.3076478093862534
    },
    "margin2": {
      "n": 6,
      "mean": 0.9544809702783823,
      "median": 0.8040228839963675,
      "p10": -1.3137344121932983,
      "p90": 2.169512704014778,
      "min": -1.3137344121932983,
      "max": 3.174438953399658
    }
  },
  "by_rel": {
    "higher": {
      "n": 1,
      "both_correct_frac": 0.0,
      "M_pair": {
        "n": 1,
        "mean": -0.08473583310842514,
        "median": -0.08473583310842514,
        "p10": -0.08473583310842514,
        "p90": -0.08473583310842514,
        "min": -0.08473583310842514,
        "max": -0.08473583310842514
      }
    },
    "less": {
      "n": 5,
      "both_correct_frac": 0.0,
      "M_pair": {
        "n": 5,
        "mean": -0.24166882634162903,
        "median": -0.2847612053155899,
        "p10": -0.7741284370422363,
        "p90": 0.22824397683143616,
        "min": -0.7741284370422363,
        "max": 0.22824397683143616
      }
    },
    "more": {
      "n": 20,
      "both_correct_frac": 0.0,
      "M_pair": {
        "n": 20,
        "mean": -0.09758114689029754,
        "median": 0.001829773187637329,
        "p10": -0.2653151610866189,
        "p90": 0.19874334335327148,
        "min": -1.1251392364501953,
        "max": 0.2598012685775757
      }
    }
  }
}

## Scale minus base M_pair
{
  "n": 26,
  "mean": -0.2641771396693702,
  "median": -0.009808667004108429,
  "p10": -1.0334668457508087,
  "p90": 0.19498680625110865,
  "min": -1.3871482610702515,
  "max": 0.6035909801721573
}

This readout is informative only if base both_correct is nonzero and there are corpus-distributed scale-specific failures with well-formed examples. It is a probe, not a training object by itself.
