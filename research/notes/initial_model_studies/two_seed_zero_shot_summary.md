# two seed zero shot summary two-seed zero-shot structured-experience summary

Input: `experiments/archive/initial_model_studies/data/2x2_two_seed_zero_shot_aggregate.json`

## Mean effects across seeds 42 and 43

| contrast | blimp | supplement | entity_tracking | ewok | comps | GlobalPIQA_mean | Reading_mean | NLP_mean_no_superglue_aoa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| structured data under WWM (B-A) | +1.690 | -0.825 | -0.095 | +1.745 | +0.620 | +0.965 | +0.110 | +0.601 |
| structured data under AMLM (D-C) | +2.220 | -1.190 | -0.830 | +0.535 | +0.050 | -1.000 | -0.030 | -0.035 |
| AMLM on official data (C-A) | -0.470 | -0.210 | +0.580 | +0.785 | +0.135 | +1.953 | +0.120 | +0.413 |
| AMLM on structured data (D-B) | +0.060 | -0.575 | -0.155 | -0.425 | -0.435 | -0.012 | -0.020 | -0.223 |
| interaction | +0.530 | -0.365 | -0.735 | -1.210 | -0.570 | -1.965 | -0.140 | -0.636 |

## Per-seed values for central contrasts

### structured data under WWM (B-A)
| column | seed42 | seed43 | mean |
|---|---:|---:|---:|
| blimp | +1.680 | +1.700 | +1.690 |
| supplement | -1.010 | -0.640 | -0.825 |
| entity_tracking | +0.410 | -0.600 | -0.095 |
| ewok | +1.130 | +2.360 | +1.745 |
| comps | +0.760 | +0.480 | +0.620 |
| GlobalPIQA_mean | -0.515 | +2.445 | +0.965 |
| Reading_mean | +0.220 | +0.000 | +0.110 |
| NLP_mean_no_superglue_aoa | +0.382 | +0.821 | +0.601 |

### structured data under AMLM (D-C)
| column | seed42 | seed43 | mean |
|---|---:|---:|---:|
| blimp | +2.270 | +2.170 | +2.220 |
| supplement | -1.060 | -1.320 | -1.190 |
| entity_tracking | -0.870 | -0.790 | -0.830 |
| ewok | -1.350 | +2.420 | +0.535 |
| comps | +0.250 | -0.150 | +0.050 |
| GlobalPIQA_mean | -1.985 | -0.015 | -1.000 |
| Reading_mean | +0.040 | -0.100 | -0.030 |
| NLP_mean_no_superglue_aoa | -0.386 | +0.316 | -0.035 |

### AMLM on official data (C-A)
| column | seed42 | seed43 | mean |
|---|---:|---:|---:|
| blimp | -0.660 | -0.280 | -0.470 |
| supplement | -0.230 | -0.190 | -0.210 |
| entity_tracking | +0.900 | +0.260 | +0.580 |
| ewok | +1.310 | +0.260 | +0.785 |
| comps | +0.310 | -0.040 | +0.135 |
| GlobalPIQA_mean | +2.445 | +1.460 | +1.953 |
| Reading_mean | +0.155 | +0.085 | +0.120 |
| NLP_mean_no_superglue_aoa | +0.604 | +0.222 | +0.413 |

### AMLM on structured data (D-B)
| column | seed42 | seed43 | mean |
|---|---:|---:|---:|
| blimp | -0.070 | +0.190 | +0.060 |
| supplement | -0.280 | -0.870 | -0.575 |
| entity_tracking | -0.380 | +0.070 | -0.155 |
| ewok | -1.170 | +0.320 | -0.425 |
| comps | -0.200 | -0.670 | -0.435 |
| GlobalPIQA_mean | +0.975 | -1.000 | -0.012 |
| Reading_mean | -0.025 | -0.015 | -0.020 |
| NLP_mean_no_superglue_aoa | -0.164 | -0.282 | -0.223 |

### interaction
| column | seed42 | seed43 | mean |
|---|---:|---:|---:|
| blimp | +0.590 | +0.470 | +0.530 |
| supplement | -0.050 | -0.680 | -0.365 |
| entity_tracking | -1.280 | -0.190 | -0.735 |
| ewok | -2.480 | +0.060 | -1.210 |
| comps | -0.510 | -0.630 | -0.570 |
| GlobalPIQA_mean | -1.470 | -2.460 | -1.965 |
| Reading_mean | -0.180 | -0.100 | -0.140 |
| NLP_mean_no_superglue_aoa | -0.769 | -0.504 | -0.636 |

