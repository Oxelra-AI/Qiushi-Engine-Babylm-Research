# earlier analysis AoA endpoint surprisal profile

Endpoint-only CDI surprisal profile. This is not AoA: measured AoA requires full shared-ancestry plus endpoint trajectory and platform curve scoring.

Common context rows: `8005`; words: `485`.

## Endpoint delta summaries (positive = higher/worse surprisal than coherent86)

### context_delta_summary

- dense62064_minus_coherent: mean `0.18664098016412778`, median `0.14072608947753906`, SE `0.004037233284963792`, improved fraction `0.27770143660212365`.
- dense62065_minus_coherent: mean `0.19084816981691036`, median `0.14567089080810547`, SE `0.004085360810475681`, improved fraction `0.2770768269831355`.
- dense62065_minus_dense62064: mean `0.004207189652782578`, median `0.0035772323608398438`, SE `0.00022772024604105948`, improved fraction `0.41998750780762023`.

### word_mean_delta_summary

- dense62064_minus_coherent: mean `0.1973093510657894`, median `0.17665863037109375`, SE `0.00921374543636738`, improved fraction `0.12371134020618557`.
- dense62065_minus_coherent: mean `0.20195009375530842`, median `0.1845783233642578`, SE `0.009399538502785748`, improved fraction `0.1154639175257732`.
- dense62065_minus_dense62064: mean `0.0046407426895190475`, median `0.004269456863403321`, SE `0.0007196599156040968`, improved fraction `0.3917525773195876`.

## Dense seed agreement

{
  "context_delta_pearson": 0.998498193603169,
  "word_mean_delta_pearson": 0.9972092166626757,
  "context_same_sign_fraction_excluding_zero": 0.9813866333541537,
  "word_same_sign_fraction_excluding_zero": 0.979381443298969
}

## CDI word-level correlations

{
  "corr_d64_delta_with_cdi_mean_16_30m": 0.06578039972951966,
  "corr_d65_delta_with_cdi_mean_16_30m": 0.0598900810181204,
  "corr_d64_delta_with_cdi_final_30m": 0.008376593263854813,
  "corr_d65_delta_with_cdi_final_30m": 0.0003738118977784563
}
