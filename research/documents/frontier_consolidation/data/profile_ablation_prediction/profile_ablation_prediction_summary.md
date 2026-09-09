# distribution proximity prediction profile ablation prediction

CPU-only ablation of the committed profile-distance prediction before MAX-register scores.

## Fits

| ablation | beta | Pearson | Spearman | RMSE | exEntity pred | cheap6 pred |
|---|---:|---:|---:|---:|---:|---:|
| profile_all | 14.2848 | 0.3856 | 0.4714 | 0.4884 | 0.7483 | 0.8175 |
| no_transcript_format | 17.5925 | 0.3721 | 0.4214 | 0.5119 | 0.6581 | 0.7492 |
| no_transcript_no_punct | 13.8899 | 0.3430 | 0.3786 | 0.6827 | 0.3805 | 0.4700 |
| function_only | 4.5930 | 0.0345 | 0.0393 | 0.6927 | 0.2402 | 0.2976 |
| shape_punct_func_no_format | 9.3484 | 0.1498 | 0.0250 | 0.5622 | 0.4832 | 0.5244 |
| abstract_shape_only | 18.5545 | 0.5503 | 0.6036 | 0.4979 | 0.6586 | 0.6672 |

The no_transcript_format row is the key robustness check: if it remains positive, the predicted register effect is not just a CHILDES transcript-marker artifact. The no_transcript_no_punct and function_only rows show how much of the calibration survives under still stricter structural abstractions.

JSON: `experiments/archive/frontier_consolidation/data/profile_ablation_prediction/profile_ablation_prediction_summary.json`
