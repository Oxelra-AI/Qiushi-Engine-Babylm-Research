# raw span identity routing result no-name-vocabulary span discovery analysis

This analysis compares lexical-available raw span discovery design outputs with raw span identity routing result no-name-vocab causal isolation. State correctness is recomputed from saved candidate-score pairs.

## Run-level metrics

| run | exists | vocab | train_state | train_cmp | graph_same | unchanged | hh_closure | mixed | eval cand>0.5 | eval other>0.5 | cand>other | other>cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lexical_shared_plus | True | 76 | 1.000 | 0.958 | 0.375 | 0.672 | 0.500 | 0.492 | 0.344 | 0.344 | 0.781 | 0.781 |
| lexical_shared_minus | False | None | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| lexical_untied_plus | True | 76 | 1.000 | 0.656 | 0.375 | 0.672 | 0.500 | 0.500 | 0.531 | 0.531 | 0.750 | 0.750 |
| lexical_untied_minus | False | None | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| noname_shared_plus | True | 60 | 1.000 | 0.542 | 0.500 | 0.906 | 0.500 | 0.492 | 0.438 | 0.438 | 0.875 | 0.875 |
| noname_shared_minus | True | 60 | 0.778 | 0.917 | 0.594 | 0.906 | 0.578 | 0.488 | 0.062 | 0.062 | 0.594 | 0.594 |
| noname_untied_plus | False | None | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| noname_untied_minus | False | None | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| noname_oracle_shared_plus | True | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| noname_oracle_shared_minus | True | 60 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Eval state-choice accuracy by family

### lexical_shared_plus
| family | n queries | acc | mean target margin |
|---|---:|---:|---:|
| direct_anchor | 192 | 0.625 | 2.397 |
| graph_transfer | 192 | 0.396 | -1.056 |
| unchanged | 384 | 0.672 | 7.345 |

### lexical_untied_plus
| family | n queries | acc | mean target margin |
|---|---:|---:|---:|
| direct_anchor | 192 | 0.604 | 0.108 |
| graph_transfer | 192 | 0.438 | -1.671 |
| unchanged | 384 | 0.672 | 7.345 |

### noname_shared_plus
| family | n queries | acc | mean target margin |
|---|---:|---:|---:|
| direct_anchor | 192 | 0.750 | 4.870 |
| graph_transfer | 192 | 0.542 | -0.457 |
| unchanged | 384 | 0.906 | 10.320 |

### noname_shared_minus
| family | n queries | acc | mean target margin |
|---|---:|---:|---:|
| direct_anchor | 192 | 0.438 | -1.096 |
| graph_transfer | 192 | 0.521 | 0.012 |
| unchanged | 384 | 0.906 | 10.320 |

### noname_oracle_shared_plus
| family | n queries | acc | mean target margin |
|---|---:|---:|---:|
| direct_anchor | 192 | 1.000 | 12.522 |
| graph_transfer | 192 | 1.000 | 11.929 |
| unchanged | 384 | 1.000 | 18.393 |

### noname_oracle_shared_minus
| family | n queries | acc | mean target margin |
|---|---:|---:|---:|
| direct_anchor | 192 | 0.000 | -18.676 |
| graph_transfer | 192 | 0.000 | -18.973 |
| unchanged | 384 | 1.000 | 18.393 |


## Bridge-sign pair metrics

### lexical_shared
Missing paired predictions: missing state predictions for one or both signs

### lexical_untied
Missing paired predictions: missing state predictions for one or both signs

### noname_shared
| family | n rows | opposite frac | same frac | mean |d+| | mean |d-| |
|---|---:|---:|---:|---:|---:|
| direct_anchor | 384 | 0.354 | 0.646 | 7.204 | 4.143 |
| graph_transfer | 384 | 0.312 | 0.688 | 4.048 | 0.014 |
| unchanged | 768 | 0.000 | 1.000 | 10.576 | 10.576 |

Comparison coordinates by suite:
| suite | n | d1 opposite | d2 opposite | product same | plus correct | minus correct |
|---|---:|---:|---:|---:|---:|---:|
| heldheld_unseen_edge_closure | 128 | 0.516 | 0.406 | 0.422 | 0.500 | 0.578 |
| mixed_held_seen_orientation | 512 | 0.352 | 0.340 | 0.465 | 0.492 | 0.488 |

### noname_untied
Missing paired predictions: missing state predictions for one or both signs

### noname_oracle_shared
| family | n rows | opposite frac | same frac | mean |d+| | mean |d-| |
|---|---:|---:|---:|---:|---:|
| direct_anchor | 384 | 1.000 | 0.000 | 12.522 | 18.676 |
| graph_transfer | 384 | 1.000 | 0.000 | 11.929 | 18.973 |
| unchanged | 768 | 0.000 | 1.000 | 18.393 | 18.393 |

Comparison coordinates by suite:
| suite | n | d1 opposite | d2 opposite | product same | plus correct | minus correct |
|---|---:|---:|---:|---:|---:|---:|
| heldheld_unseen_edge_closure | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| mixed_held_seen_orientation | 512 | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 |

