# causal intervention: causal donor-query activation intervention

## Design

Same context, two queries A and B. Donor run (query A) extracts hidden
states at ATTR_POS after each layer. Recipient run (query B) patches those
states. If the answer redirects from B's attribute to A's attribute, the
component carries usable entity-selection information.

- **full**: replace entire hidden state at ATTR_POS
- **d_only**: replace only the prep-learned query-match direction component
- **orth_only**: replace only the orthogonal complement (control)

## Seed 43

### Behavior after continuation
| model | held_top4 | held_b | held_sel | train_top4 |
|---|---:|---:|---:|---:|
| prep | 0.859 | +5.567 | +0.679 | 1.000 |
| direct_full | 0.281 | +0.040 | -0.011 | 0.238 |
| static_1over17 | 0.789 | +7.261 | +0.744 | 1.000 |

### Direction accuracy (prep-learned, tested per model)
| model | L0 | L1 | L2 |
|---|---:|---:|---:|
| prep | 0.230 | 0.972 | 0.962 |
| direct_full | 0.280 | 0.322 | 0.318 |
| static_1over17 | 0.278 | 1.000 | 0.996 |

### Train-entity donor-query redirection
**prep** clean: A=0.998, B=1.000
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.002 | 0.998 | -7.669 |
| L0_full | 0.000 | 1.000 | -7.425 |
| L0_orth_only | 0.000 | 0.998 | -7.411 |
| L1_d_only | 0.998 | 0.000 | +7.445 |
| L1_full | 0.998 | 0.000 | +7.573 |
| L1_orth_only | 0.002 | 0.998 | -7.558 |
| L2_d_only | 0.000 | 1.000 | -7.665 |
| L2_full | 0.000 | 1.000 | -7.665 |
| L2_orth_only | 0.000 | 1.000 | -7.665 |

**direct_full** clean: A=0.230, B=0.260
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.244 | 0.262 | -0.008 |
| L0_full | 0.234 | 0.268 | -0.014 |
| L0_orth_only | 0.234 | 0.272 | -0.013 |
| L1_d_only | 0.242 | 0.258 | -0.007 |
| L1_full | 0.242 | 0.260 | -0.007 |
| L1_orth_only | 0.244 | 0.258 | -0.007 |
| L2_d_only | 0.244 | 0.260 | -0.007 |
| L2_full | 0.244 | 0.260 | -0.007 |
| L2_orth_only | 0.244 | 0.260 | -0.007 |

**static_1over17** clean: A=1.000, B=1.000
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.000 | 1.000 | -10.403 |
| L0_full | 0.000 | 1.000 | -10.367 |
| L0_orth_only | 0.000 | 1.000 | -10.364 |
| L1_d_only | 1.000 | 0.000 | +9.409 |
| L1_full | 1.000 | 0.000 | +10.361 |
| L1_orth_only | 0.000 | 1.000 | -9.570 |
| L2_d_only | 0.000 | 1.000 | -10.400 |
| L2_full | 0.000 | 1.000 | -10.400 |
| L2_orth_only | 0.000 | 1.000 | -10.400 |

### Held-entity redirection (held_donor)
**prep**
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.085 | 0.915 | -5.934 |
| L0_full | 0.080 | 0.920 | -5.855 |
| L0_orth_only | 0.080 | 0.920 | -5.859 |
| L1_d_only | 0.765 | 0.055 | +4.755 |
| L1_full | 0.780 | 0.060 | +4.757 |
| L1_orth_only | 0.090 | 0.910 | -5.839 |
| L2_d_only | 0.085 | 0.915 | -5.941 |
| L2_full | 0.085 | 0.915 | -5.941 |
| L2_orth_only | 0.085 | 0.915 | -5.941 |

**direct_full**
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.250 | 0.250 | +0.025 |
| L0_full | 0.265 | 0.245 | +0.008 |
| L0_orth_only | 0.265 | 0.240 | +0.006 |
| L1_d_only | 0.250 | 0.250 | +0.024 |
| L1_full | 0.250 | 0.250 | +0.024 |
| L1_orth_only | 0.250 | 0.250 | +0.023 |
| L2_d_only | 0.250 | 0.250 | +0.023 |
| L2_full | 0.250 | 0.250 | +0.023 |
| L2_orth_only | 0.250 | 0.250 | +0.023 |

**static_1over17**
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.075 | 0.925 | -8.696 |
| L0_full | 0.075 | 0.925 | -8.734 |
| L0_orth_only | 0.075 | 0.925 | -8.732 |
| L1_d_only | 0.735 | 0.100 | +5.975 |
| L1_full | 0.765 | 0.080 | +6.526 |
| L1_orth_only | 0.095 | 0.905 | -7.765 |
| L2_d_only | 0.075 | 0.925 | -8.694 |
| L2_full | 0.075 | 0.925 | -8.694 |
| L2_orth_only | 0.075 | 0.925 | -8.694 |

### Held-entity redirection (train_donor)
**prep**
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.055 | 0.790 | -4.822 |
| L0_full | 0.070 | 0.785 | -4.855 |
| L0_orth_only | 0.070 | 0.785 | -4.848 |
| L1_d_only | 0.910 | 0.090 | +5.776 |
| L1_full | 0.915 | 0.085 | +5.880 |
| L1_orth_only | 0.055 | 0.790 | -4.812 |
| L2_d_only | 0.055 | 0.800 | -4.816 |
| L2_full | 0.055 | 0.800 | -4.816 |
| L2_orth_only | 0.055 | 0.800 | -4.816 |

**direct_full**
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.250 | 0.280 | -0.011 |
| L0_full | 0.255 | 0.260 | -0.028 |
| L0_orth_only | 0.255 | 0.270 | -0.029 |
| L1_d_only | 0.245 | 0.280 | -0.011 |
| L1_full | 0.245 | 0.280 | -0.011 |
| L1_orth_only | 0.245 | 0.280 | -0.012 |
| L2_d_only | 0.245 | 0.280 | -0.012 |
| L2_full | 0.245 | 0.280 | -0.012 |
| L2_orth_only | 0.245 | 0.280 | -0.012 |

**static_1over17**
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.085 | 0.755 | -6.505 |
| L0_full | 0.090 | 0.760 | -6.537 |
| L0_orth_only | 0.090 | 0.760 | -6.543 |
| L1_d_only | 0.905 | 0.095 | +7.802 |
| L1_full | 0.925 | 0.075 | +8.732 |
| L1_orth_only | 0.095 | 0.745 | -5.948 |
| L2_d_only | 0.085 | 0.760 | -6.506 |
| L2_full | 0.085 | 0.760 | -6.506 |
| L2_orth_only | 0.085 | 0.760 | -6.506 |

### Cross-model transplant (preserving → collapsed)
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.252 | 0.252 | -0.008 |
| L0_full | 0.240 | 0.266 | +0.008 |
| L1_d_only | 0.840 | 0.064 | +2.389 |
| L1_full | 0.946 | 0.024 | +4.806 |
| L2_d_only | 0.244 | 0.260 | -0.007 |
| L2_full | 0.244 | 0.260 | -0.007 |

## Seed 100

### Behavior after continuation
| model | held_top4 | held_b | held_sel | train_top4 |
|---|---:|---:|---:|---:|
| prep | 0.938 | +7.230 | +0.910 | 1.000 |
| direct_full | 0.477 | +1.520 | +0.192 | 0.527 |
| static_1over17 | 0.555 | +4.051 | +0.498 | 1.000 |

### Direction accuracy (prep-learned, tested per model)
| model | L0 | L1 | L2 |
|---|---:|---:|---:|
| prep | 0.260 | 0.980 | 0.926 |
| direct_full | 0.264 | 0.264 | 0.264 |
| static_1over17 | 0.264 | 0.998 | 0.574 |

### Train-entity donor-query redirection
**prep** clean: A=1.000, B=1.000
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.000 | 1.000 | -9.377 |
| L0_full | 0.000 | 1.000 | -9.278 |
| L0_orth_only | 0.000 | 1.000 | -9.282 |
| L1_d_only | 1.000 | 0.000 | +9.029 |
| L1_full | 1.000 | 0.000 | +9.263 |
| L1_orth_only | 0.000 | 1.000 | -9.200 |
| L2_d_only | 0.000 | 1.000 | -9.374 |
| L2_full | 0.000 | 1.000 | -9.374 |
| L2_orth_only | 0.000 | 1.000 | -9.374 |

**direct_full** clean: A=0.550, B=0.436
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.172 | 0.438 | -2.034 |
| L0_full | 0.182 | 0.438 | -2.025 |
| L0_orth_only | 0.186 | 0.436 | -2.026 |
| L1_d_only | 0.530 | 0.140 | +1.926 |
| L1_full | 0.538 | 0.138 | +2.262 |
| L1_orth_only | 0.174 | 0.438 | -1.753 |
| L2_d_only | 0.176 | 0.436 | -2.035 |
| L2_full | 0.176 | 0.436 | -2.035 |
| L2_orth_only | 0.176 | 0.436 | -2.035 |

**static_1over17** clean: A=1.000, B=1.000
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.000 | 1.000 | -11.493 |
| L0_full | 0.000 | 1.000 | -11.439 |
| L0_orth_only | 0.000 | 1.000 | -11.451 |
| L1_d_only | 1.000 | 0.000 | +10.714 |
| L1_full | 1.000 | 0.000 | +11.482 |
| L1_orth_only | 0.000 | 1.000 | -10.791 |
| L2_d_only | 0.000 | 1.000 | -11.502 |
| L2_full | 0.000 | 1.000 | -11.502 |
| L2_orth_only | 0.000 | 1.000 | -11.502 |

### Held-entity redirection (held_donor)
**prep**
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.105 | 0.895 | -5.981 |
| L0_full | 0.100 | 0.900 | -5.863 |
| L0_orth_only | 0.105 | 0.895 | -5.860 |
| L1_d_only | 0.940 | 0.025 | +7.415 |
| L1_full | 0.940 | 0.025 | +7.523 |
| L1_orth_only | 0.110 | 0.890 | -5.782 |
| L2_d_only | 0.105 | 0.895 | -5.979 |
| L2_full | 0.105 | 0.895 | -5.979 |
| L2_orth_only | 0.105 | 0.895 | -5.979 |

**direct_full**
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.220 | 0.495 | -1.549 |
| L0_full | 0.215 | 0.505 | -1.558 |
| L0_orth_only | 0.210 | 0.500 | -1.563 |
| L1_d_only | 0.380 | 0.200 | +1.547 |
| L1_full | 0.370 | 0.200 | +1.624 |
| L1_orth_only | 0.210 | 0.480 | -1.245 |
| L2_d_only | 0.210 | 0.500 | -1.554 |
| L2_full | 0.210 | 0.500 | -1.554 |
| L2_orth_only | 0.210 | 0.500 | -1.554 |

**static_1over17**
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.010 | 0.990 | -11.011 |
| L0_full | 0.000 | 1.000 | -10.951 |
| L0_orth_only | 0.000 | 1.000 | -10.952 |
| L1_d_only | 0.575 | 0.145 | +4.487 |
| L1_full | 0.580 | 0.105 | +5.325 |
| L1_orth_only | 0.015 | 0.985 | -10.020 |
| L2_d_only | 0.010 | 0.990 | -11.007 |
| L2_full | 0.010 | 0.990 | -11.007 |
| L2_orth_only | 0.010 | 0.990 | -11.007 |

### Held-entity redirection (train_donor)
**prep**
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.025 | 0.935 | -7.559 |
| L0_full | 0.025 | 0.935 | -7.638 |
| L0_orth_only | 0.020 | 0.945 | -7.634 |
| L1_d_only | 0.885 | 0.115 | +5.745 |
| L1_full | 0.890 | 0.110 | +5.938 |
| L1_orth_only | 0.025 | 0.940 | -7.447 |
| L2_d_only | 0.025 | 0.940 | -7.553 |
| L2_full | 0.025 | 0.940 | -7.553 |
| L2_orth_only | 0.025 | 0.940 | -7.553 |

**direct_full**
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.195 | 0.385 | -1.681 |
| L0_full | 0.185 | 0.395 | -1.684 |
| L0_orth_only | 0.190 | 0.390 | -1.685 |
| L1_d_only | 0.470 | 0.215 | +1.217 |
| L1_full | 0.500 | 0.205 | +1.522 |
| L1_orth_only | 0.190 | 0.410 | -1.602 |
| L2_d_only | 0.185 | 0.400 | -1.682 |
| L2_full | 0.185 | 0.400 | -1.682 |
| L2_orth_only | 0.185 | 0.400 | -1.682 |

**static_1over17**
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.105 | 0.575 | -5.295 |
| L0_full | 0.120 | 0.570 | -5.194 |
| L0_orth_only | 0.120 | 0.570 | -5.213 |
| L1_d_only | 0.985 | 0.015 | +9.952 |
| L1_full | 0.990 | 0.010 | +10.976 |
| L1_orth_only | 0.140 | 0.575 | -4.481 |
| L2_d_only | 0.105 | 0.575 | -5.316 |
| L2_full | 0.105 | 0.575 | -5.316 |
| L2_orth_only | 0.105 | 0.575 | -5.316 |

### Cross-model transplant (preserving → collapsed)
| layer_mode | redirect | natural | margin |
|---|---:|---:|---:|
| L0_d_only | 0.182 | 0.434 | -1.987 |
| L0_full | 0.180 | 0.414 | -1.336 |
| L1_d_only | 0.860 | 0.066 | +3.784 |
| L1_full | 0.982 | 0.008 | +6.506 |
| L2_d_only | 0.176 | 0.436 | -2.035 |
| L2_full | 0.176 | 0.436 | -2.035 |

## Interpretation

Redirect rate measures whether the model's answer follows the donor query
rather than the natural query. High redirect with d_only but not orth_only
establishes that the query-match direction causally determines the answer.
Collapsed models with moderate direction accuracy but low redirect rate
indicate the signal exists but is no longer functionally used.
Cross-model transplant tests whether the collapsed model's downstream
readout can still process the preserving model's query-match signal.
