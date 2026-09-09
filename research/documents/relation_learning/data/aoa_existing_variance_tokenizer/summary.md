# earlier analysis AoA existing evidence and tokenizer geometry analysis

Created UTC: 2026-09-08T03:19:50Z

## 1. Existing full-scale AoA movements

### earlier analysis masking-priority family

| arm | official AoA/correlation | p | fitted words |
|---|---:|---:|---:|
| mask_uniform_control_90M | -0.17702799558034168 | 0.017437959470825378 | 180 |
| mask_uniform_control_100M | -0.15302646626566574 | 0.04028084757463022 | 180 |
| mask_evidence_visible_90M | -0.15456019530556697 | 0.04112423283475058 | 175 |
| mask_evidence_visible_100M | -0.14666542859729043 | 0.05277182009746163 | 175 |
| mask_inverse_priority_95M | -0.1719406337063024 | 0.02289203126670519 | 175 |
| mask_inverse_priority_100M | -0.18344781331440352 | 0.015392915069751814 | 174 |

Key differences:
- uniform_100M_minus_90M: 0.024002
- evidence_visible_100M_minus_uniform_100M: 0.006361
- evidence_visible_90M_minus_uniform_90M: 0.022468
- inverse_priority_100M_minus_uniform_100M: -0.030421
- inverse_priority_100M_minus_evidence_visible_100M: -0.036782
- max_minus_min: 0.036782

The evidence-visible masking arm improves the 100M uniform control by only about 0.006361; inverse-priority worsens it by -0.030421. The whole earlier analysis family spans 0.036782.

### REPRESENTATION_FRONTIER_STUDIES density AoA anchors

| record | AoA/correlation | p | fitted words |
|---|---:|---:|---:|
| representation_frontier_studies_step017_compact_view_core | -0.12687286024554456 | 0.07870730227036446 | 193 |
| representation_frontier_studies_step018_density_compact_repeat_core | -0.18899206801927268 | 0.015051313739058825 | 165 |
| representation_frontier_studies_step018_density_near_view | 0.0 | None | 197 |
| representation_frontier_studies_step018_density_near_repeat | 0.0 | None | 188 |

### Positive-threshold scale

For n=225 fitted words and two-sided p=0.1, the positive correlation boundary is r=0.109937. Coherent86/v4 has r=-0.034854, so the required movement is 0.144791.
This movement is 2.03 times the sample sd of de-duplicated stored local AoA scores, 5.35 times the sd of nonzero stored scores, 3.94 times the entire earlier analysis masking-priority spread, and 22.8 times the 100M evidence-visible versus uniform difference.

## 2. Tokenizer and fit-set geometry

### Official v4 fitted set

- model_child_all_official_ok: n=225, Pearson r=-0.03485418140387646, p=0.6030225874932932, Spearman r=-0.06471765697429414
- model_child_single_token_official_ok: n=209, Pearson r=-0.039774885936871854, p=0.5674597903904498, Spearman r=-0.07884615384615386
- model_child_multi_token_official_ok: n=16, Pearson r=-0.3881641024717556, p=0.1373599764158028, Spearman r=-0.44117647058823534
- subword_len_vs_model_aoa_official_ok: n=225, Pearson r=0.12458386297567255, p=0.06209395805877064, Spearman r=0.17017392158018635
- char_len_vs_model_aoa_official_ok: n=225, Pearson r=0.23410065304007285, p=0.0003981797597904841, Spearman r=0.17400322929856007
- whole_stream_logfreq_vs_model_aoa_official_ok: n=225, Pearson r=-0.6213974018475322, p=2.000143377908363e-25, Spearman r=-0.7023904184217881
- subword_len_vs_child_aoa_all_child_ok: n=406, Pearson r=0.08152818737128834, p=0.10091905033716486, Spearman r=0.10795185573131419
- char_len_vs_child_aoa_all_child_ok: n=406, Pearson r=0.11188845495824522, p=0.024157021313303054, Spearman r=0.12808242112849477

OLS R2 for official fitted model AoA:
- model_aoa_from_logfreq: 0.38613473102286333
- model_aoa_from_subword_len: 0.015521138913941246
- model_aoa_from_char_len: 0.05480311575378882
- model_aoa_from_logfreq_plus_subword: 0.5024142080333498
- model_aoa_from_logfreq_plus_char: 0.38634510840803327
- model_aoa_from_logfreq_plus_subword_plus_char: 0.5057312032863598

### Subword length table

| subword length | child-ok words | model-fit words | fit rate | mean child AoA | mean model AoA | status counts |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 386 | 209 | 0.541 | 24.3881351062464 | 7.039950131052422 | {'threshold_above_upper_asymptote': 116, 'flat_or_zero_amplitude': 44, 'ok': 209, 'model_fit_exception': 8, 'aoa_after_last_checkpoint': 6, 'threshold_below_lower_asymptote': 2, 'aoa_before_first_checkpoint': 1} |
| 2 | 16 | 13 | 0.812 | 26.208914037479083 | 7.287118265871564 | {'ok': 13, 'threshold_above_upper_asymptote': 2, 'flat_or_zero_amplitude': 1} |
| 3 | 4 | 3 | 0.750 | 24.04363177343636 | 7.103152068418549 | {'flat_or_zero_amplitude': 1, 'ok': 3} |

### Refit variants

| variant | n | Pearson r | p | Spearman r |
|---|---:|---:|---:|---:|
| official_record_all | 225 | -0.03485418140387646 | 0.6030225874932932 | -0.06471765697429414 |
| official_record_single_token | 209 | -0.039774885936871854 | 0.5674597903904498 | -0.07884615384615386 |
| official_record_multi_token | 16 | -0.3881641024717556 | 0.1373599764158028 | -0.44117647058823534 |
| official_refit_all | 225 | -0.03485418140387646 | 0.6030225874932932 | -0.06471765697429414 |
| official_refit_single_token | 209 | -0.039774885936871854 | 0.5674597903904498 | -0.07884615384615386 |
| official_refit_multi_token | 16 | -0.3881641024717556 | 0.1373599764158028 | -0.44117647058823534 |
| per_piece_all | 225 | -0.034857115783438836 | 0.602992053529287 | -0.06471765697429414 |
| per_piece_single_token | 209 | -0.039774885936871854 | 0.5674597903904498 | -0.07884615384615386 |
| per_piece_multi_token | 16 | -0.38824268406244716 | 0.13727349213350334 | -0.44117647058823534 |
| progress_norm_all | 225 | -0.034710199953491505 | 0.6045216480230818 | -0.06477243994943112 |
| progress_norm_single_token | 209 | -0.039624805058564014 | 0.5689236899981432 | -0.07891187759608813 |
| progress_norm_multi_token | 16 | -0.3882004194396614 | 0.1373200025372047 | -0.44117647058823534 |

## Interpretation for the ongoing AoA calibration

The existing 100M masking-priority family shows that a legal masking-priority channel can move the official AoA correlation, but prior movements are much smaller than the shift needed to make v4 positive and they can also move in the wrong direction. The current 30M beta measurement remains valuable because it isolates the child-enrichment credit hypothesis; however, a positive beta should be read together with this full-scale prior rather than converted mechanically into 100M spending.

The tokenizer analysis separates two possible explanations. If token count or length explains more fitted model-AoA variance than corpus frequency, then AoA is dominated by evaluation geometry; if not, the dominant measured variable remains frequency/credit. The tables above make that comparison directly.
