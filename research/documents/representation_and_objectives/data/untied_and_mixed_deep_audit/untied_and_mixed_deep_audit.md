# span matcher review and next span discovery learned-untied and mixed-surface deep audit

## State bridge-sign response

| pair | n | opposite sign | same sign | same prediction | plus acc | minus acc | mean |d+| | mean |d-| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| learned_shared_graph | 256 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 18.214 | 17.449 |
| learned_shared_direct | 256 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 17.980 | 17.862 |
| learned_shared_unchanged | 512 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 15.317 | 15.317 |
| learned_untied_graph | 256 | 0.562 | 0.438 | 0.438 | 0.969 | 0.469 | 5.822 | 1.317 |
| learned_untied_direct | 256 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 16.883 | 17.822 |
| learned_untied_unchanged | 512 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 15.317 | 15.317 |

## Comparison bridge-sign response

| pair | n | plus acc | minus acc | same pred | same product sign | opposite product sign | plus margin | minus margin | plus abslogit | minus abslogit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| learned_shared_heldheld | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 13.794 | 13.615 | 13.794 | 13.615 |
| learned_shared_mixed | 512 | 0.500 | 0.500 | 1.000 | 1.000 | 0.000 | -0.042 | -0.399 | 0.629 | 7.875 |
| learned_untied_heldheld | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 12.654 | 12.654 | 12.654 | 12.654 |
| learned_untied_mixed | 512 | 0.500 | 0.500 | 1.000 | 1.000 | 0.000 | -0.023 | -0.023 | 1.918 | 1.918 |
| oracle_shared_heldheld | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 13.809 | 13.473 | 13.809 | 13.473 |
| oracle_shared_mixed | 512 | 0.750 | 0.500 | 0.250 | 0.250 | 0.750 | 0.001 | -0.304 | 0.034 | 1.528 |

## Mixed surface summary by run

| run | n | acc | pred_true | label_true | signed margin | abs logit | product sign acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned_shared_plus | 512 | 0.500 | 0.500 | 0.500 | -0.042 | 0.629 | 0.500 |
| learned_shared_minus | 512 | 0.500 | 0.500 | 0.500 | -0.399 | 7.875 | 0.500 |
| learned_untied_plus | 512 | 0.500 | 0.500 | 0.500 | -0.023 | 1.918 | 0.500 |
| learned_untied_minus | 512 | 0.500 | 0.500 | 0.500 | -0.023 | 1.918 | 0.500 |
| oracle_shared_plus | 512 | 0.750 | 0.500 | 0.500 | 0.001 | 0.034 | 0.750 |
| oracle_shared_minus | 512 | 0.500 | 0.500 | 0.500 | -0.304 | 1.528 | 0.500 |

## Reviewer reading

Learned shared_trunk has complete row-paired state-coordinate reversal for graph and direct changed rows, while unchanged rows stay same-signed. Learned untied has direct-anchor reversal but only partial graph-state reversal; this is not the same causal fingerprint even though aggregate graph_same changes from 0.969 to 0.469. The comparison pathway in learned untied is exactly sign-invariant on heldheld and mixed rows, showing the bridge sign did not alter its comparison coordinate. Mixed held-seen accuracy remains chance for learned shared and learned untied, with bs- learned shared showing large absolute logits but wrong/arbitrary relation-family alignment.
