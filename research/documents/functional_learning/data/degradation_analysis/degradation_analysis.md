# effective input repaired contrast result analysis of topology phase1 result wrong-constraint degradation

Existing topology phase1 result full-graph and adversarial-h1 prediction files were joined to the raw eval substrate. This asks whether wrong h1-incident constraints degrade held nonidentical content use below the full-graph condition, across active/passive voice and held objects/names.

## State readout by relation

| relation | n | full acc | adversarial acc | acc delta | full margin | adversarial margin | margin delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| h0_dax | 96 | 1.000 | 1.000 | 0.000 | 11.963 | 12.020 | 0.057 |
| h1_mep | 96 | 1.000 | 0.000 | -1.000 | 11.938 | -20.804 | -32.742 |
| h2_norp | 96 | 1.000 | 1.000 | 0.000 | 12.605 | 20.841 | 8.236 |
| h3_ziv | 96 | 1.000 | 1.000 | 0.000 | 15.912 | 0.917 | -14.995 |

## h1 state readout by voice

| relation | voice | n | full acc | adversarial acc | margin delta |
|---|---|---:|---:|---:|---:|
| h1_mep | active | 48 | 1.000 | 0.000 | -32.555 |
| h1_mep | passive | 48 | 1.000 | 0.000 | -32.930 |

## h1 state readout by held object

| object | n | full acc | adversarial acc | margin delta |
|---|---:|---:|---:|---:|
| badge | 8 | 1.000 | 0.000 | -32.742 |
| booklet | 16 | 1.000 | 0.000 | -32.742 |
| drum | 16 | 1.000 | 0.000 | -32.742 |
| flute | 8 | 1.000 | 0.000 | -32.742 |
| hammer | 16 | 1.000 | 0.000 | -32.742 |
| scarf | 8 | 1.000 | 0.000 | -32.742 |
| shell | 16 | 1.000 | 0.000 | -32.742 |
| ticket | 8 | 1.000 | 0.000 | -32.742 |

## Eval comparison by edge

| edge | n | full acc | adversarial acc | acc delta | full margin | adversarial margin | margin delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| h0_dax->h1_mep | 32 | 1.000 | 0.000 | -1.000 | 9.654 | -13.517 | -23.170 |
| h0_dax->s_give | 32 | 1.000 | 1.000 | 0.000 | 11.084 | 13.328 | 2.245 |
| h0_dax->s_receive | 32 | 1.000 | 1.000 | 0.000 | 10.956 | 11.633 | 0.678 |
| h1_mep->h3_ziv | 32 | 1.000 | 0.000 | -1.000 | 11.121 | -13.373 | -24.494 |
| h1_mep->s_give | 32 | 1.000 | 0.000 | -1.000 | 11.142 | -7.157 | -18.300 |
| h1_mep->s_receive | 32 | 1.000 | 0.000 | -1.000 | 11.080 | -4.892 | -15.972 |
| h2_norp->h3_ziv | 32 | 1.000 | 1.000 | 0.000 | 9.629 | 13.373 | 3.745 |
| h2_norp->s_give | 32 | 1.000 | 1.000 | 0.000 | 9.631 | 7.158 | -2.473 |
| h2_norp->s_receive | 32 | 1.000 | 1.000 | 0.000 | 9.624 | 4.892 | -4.733 |
| h3_ziv->h0_dax | 32 | 1.000 | 1.000 | 0.000 | 11.037 | 12.328 | 1.291 |
| h3_ziv->s_give | 32 | 1.000 | 1.000 | 0.000 | 7.660 | 13.028 | 5.368 |
| h3_ziv->s_receive | 32 | 1.000 | 1.000 | 0.000 | 14.733 | 11.518 | -3.215 |
| s_give->h0_dax | 32 | 1.000 | 1.000 | 0.000 | 11.084 | 13.328 | 2.245 |
| s_give->h1_mep | 32 | 1.000 | 0.000 | -1.000 | 11.142 | -7.157 | -18.300 |
| s_give->h2_norp | 32 | 1.000 | 1.000 | 0.000 | 9.631 | 7.158 | -2.473 |
| s_give->h3_ziv | 32 | 1.000 | 1.000 | 0.000 | 7.660 | 13.028 | 5.368 |
| s_receive->h0_dax | 32 | 1.000 | 1.000 | 0.000 | 10.956 | 11.633 | 0.678 |
| s_receive->h1_mep | 32 | 1.000 | 0.000 | -1.000 | 11.080 | -4.892 | -15.972 |
| s_receive->h2_norp | 32 | 1.000 | 1.000 | 0.000 | 9.624 | 4.892 | -4.733 |
| s_receive->h3_ziv | 32 | 1.000 | 1.000 | 0.000 | 14.733 | 11.518 | -3.215 |

## Eval comparison aggregated by h1 involvement

| h1 involved | n | full acc | adversarial acc | acc delta | margin delta |
|---|---:|---:|---:|---:|---:|
| False | 448 | 1.000 | 1.000 | 0.000 | 0.055 |
| True | 192 | 1.000 | 0.000 | -1.000 | -19.368 |

## Interpretation

Adversarial h1-incident constraints drive h1 state readout below the full-graph condition on every held h1 object and in both active/passive voices, while h0/h2/h3 remain correct. Eval comparison edges involving h1 also fall from full-graph accuracy 1.0 to 0.0 with large negative margin shifts, whereas non-h1 edges remain correct. This is the coordinate-transport analogue of degradation from wrong/exact constraints: corrupted relational content does not merely fail to help; it installs a systematically wrong h1 coordinate on held nonidentical renderings. It is still not a token-NLL copy/content probe and should not be equated with the RoBERTa nonoverlap content result.
