# earlier analysis clean seed62065 threshold profile

Clean seed62065 has full official zero-shot/Reading and measured AoA; SuperGLUE is still pending.

## Known non-SuperGLUE components

{
  "BLiMP": 68.22,
  "Supplement": 63.29,
  "EWoK": 49.72,
  "Entity": 29.45,
  "COMPS": 52.14,
  "GlobalPIQA": 40.05,
  "Reading": 8.195,
  "AoA": 0.0
}

Known-component sum: `311.065`; mean over 8 known components: `38.883125`.

## Required SuperGLUE thresholds

- Equal `coherent86` Overall requires SuperGLUE `67.15071192183592`.
- Equal `dense_seed62064` Overall requires SuperGLUE `68.27682007430639`.
- Equal `dense_seed62065` Overall requires SuperGLUE `68.45080432073166`.
- Equal `clean_pres_lambda1_eval_seed62064` Overall requires SuperGLUE `69.15271098988501`.

## Hypothetical reference SuperGLUE values

{
  "if_clean65_superglue_equals_coherent86": {
    "superglue": 68.94571192183594,
    "overall": 42.22341243575955,
    "delta_vs_coherent86": 0.19944444444444542,
    "delta_vs_clean_seed62064": -0.022999896449896085
  },
  "if_clean65_superglue_equals_dense_seed62064": {
    "superglue": 68.5318200743064,
    "overall": 42.177424452700706,
    "delta_vs_coherent86": 0.15345646138560198,
    "delta_vs_clean_seed62064": -0.06898787950873952
  },
  "if_clean65_superglue_equals_dense_seed62065": {
    "superglue": 68.80580432073164,
    "overall": 42.20786714674796,
    "delta_vs_coherent86": 0.1838991554328544,
    "delta_vs_clean_seed62064": -0.03854518546148711
  },
  "if_clean65_superglue_equals_clean_pres_lambda1_eval_seed62064": {
    "superglue": 69.047710989885,
    "overall": 42.234745665542775,
    "delta_vs_coherent86": 0.21077767422767124,
    "delta_vs_clean_seed62064": -0.011666666666670267
  }
}

## Scientific reading

Clean seed62065 has already replicated the seed62064 zero/Reading trade shape on the official-sized tasks: grammar/knowledge remain below coherent86, Entity and GlobalPIQA remain above it. The pending SuperGLUE value decides whether this becomes a complete official replication, because AoA is measured zero and the remaining components are fixed.
