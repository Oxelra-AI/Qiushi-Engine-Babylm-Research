# fullctx aux budget audit — masked alternative consequence calibration

Checkpoint: standard80_margin16 `experiments/archive/representation_and_objectives/training/runs/standard_legacy_70M_to_80M_seed43022/hf_model/chck_80M` runtime=293.83 device=cuda

## Coverage

{'batches': 16, 'raw_events': 10791, 'naturally_masked_events': 1603, 'capped_events': 657, 'used_events': 657, 'masked_tokens': 138379, 'used_spatial': 128, 'used_negation': 128, 'used_physical_change': 128, 'used_temporal': 128, 'used_causal_connector': 128, 'used_comparative': 17}

Negative match levels: {'0': 650, '1': 4, '2': 3}

## Margin true target vs matched alternative

{
  "ALL": {
    "n": 657,
    "mean": 6.793857949418745,
    "median": 6.171998500823975,
    "std": 5.108607622893831,
    "stderr": 0.1993057655861305,
    "p05": -0.15396385192871062,
    "p95": 16.375116348266594,
    "success_gt0": 0.9406392694063926
  },
  "causal_connector": {
    "n": 128,
    "mean": 6.730156193487346,
    "median": 6.033740997314453,
    "std": 5.535142944014604,
    "stderr": 0.4892421388186996,
    "p05": -1.165210247039795,
    "p95": 16.38171348571777,
    "success_gt0": 0.921875
  },
  "comparative": {
    "n": 17,
    "mean": 5.755999032188864,
    "median": 4.537377834320068,
    "std": 5.318259032567059,
    "stderr": 1.289867278568775,
    "p05": -0.043121910095214766,
    "p95": 15.68107986450195,
    "success_gt0": 0.9411764705882353
  },
  "negation": {
    "n": 128,
    "mean": 6.967118659988046,
    "median": 6.559218645095825,
    "std": 4.86244513444244,
    "stderr": 0.4297834909639729,
    "p05": 0.48093190193176283,
    "p95": 16.17604184150695,
    "success_gt0": 0.9609375
  },
  "physical_change": {
    "n": 128,
    "mean": 6.887984497472644,
    "median": 6.435843467712402,
    "std": 4.601291681214894,
    "stderr": 0.40670056875053767,
    "p05": 0.40007379055023196,
    "p95": 15.415334701538082,
    "success_gt0": 0.9609375
  },
  "spatial": {
    "n": 128,
    "mean": 7.060165190137923,
    "median": 6.280632972717285,
    "std": 5.352074266649048,
    "stderr": 0.473061000920195,
    "p05": 0.10967715382575995,
    "p95": 17.02800798416137,
    "success_gt0": 0.953125
  },
  "temporal": {
    "n": 128,
    "mean": 6.461705843452364,
    "median": 6.1772544384002686,
    "std": 5.072871498975581,
    "stderr": 0.4483827296266999,
    "p05": -0.9859245538711545,
    "p95": 16.101167678833004,
    "success_gt0": 0.90625
  }
}

## Gradient

{
  "ratio_to_mlm": {
    "n": 0
  },
  "cosine_with_mlm": {
    "n": 0
  }
}

Files: `experiments/archive/representation_and_objectives/data/masked_alt_consequence_calibration_margin16/masked_alt_consequence_calibration_summary.json`

