# raw span identity routing result matcher failure audit

Rows are changed-event eval rows with candidate_index=0. `rel_ok` asks whether the true name channel beats the opposite name channel; `wins_gate` asks whether it also beats the neither channel. The latter is the actual routing condition for replacing word embeddings by candidate/other embeddings.

## Overall

| run | n | cand rel ok | other rel ok | cand wins gate | other wins gate | both wins gate | cand >0.5 | other >0.5 | mean cand_neither | mean other_neither |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lexical_shared_plus | 384 | 0.781 | 0.781 | 0.500 | 0.500 | 0.062 | 0.344 | 0.344 | 0.331 | 0.331 |
| lexical_untied_plus | 384 | 0.750 | 0.750 | 0.594 | 0.594 | 0.188 | 0.531 | 0.531 | 0.223 | 0.223 |
| noname_shared_plus | 384 | 0.875 | 0.875 | 0.531 | 0.531 | 0.250 | 0.438 | 0.438 | 0.454 | 0.454 |
| noname_shared_minus | 384 | 0.594 | 0.594 | 0.062 | 0.062 | 0.000 | 0.062 | 0.062 | 0.658 | 0.658 |

## By suite

### lexical_shared_plus
| suite | n | both rel ok | both wins gate | cand wins gate | other wins gate |
|---|---:|---:|---:|---:|---:|
| cross_template_state_readout | 128 | 0.562 | 0.062 | 0.500 | 0.500 |
| paired_state_conservation | 256 | 0.562 | 0.062 | 0.500 | 0.500 |

### lexical_untied_plus
| suite | n | both rel ok | both wins gate | cand wins gate | other wins gate |
|---|---:|---:|---:|---:|---:|
| cross_template_state_readout | 128 | 0.500 | 0.188 | 0.594 | 0.594 |
| paired_state_conservation | 256 | 0.500 | 0.188 | 0.594 | 0.594 |

### noname_shared_plus
| suite | n | both rel ok | both wins gate | cand wins gate | other wins gate |
|---|---:|---:|---:|---:|---:|
| cross_template_state_readout | 128 | 0.750 | 0.250 | 0.531 | 0.531 |
| paired_state_conservation | 256 | 0.750 | 0.250 | 0.531 | 0.531 |

### noname_shared_minus
| suite | n | both rel ok | both wins gate | cand wins gate | other wins gate |
|---|---:|---:|---:|---:|---:|
| cross_template_state_readout | 128 | 0.188 | 0.000 | 0.062 | 0.062 |
| paired_state_conservation | 256 | 0.188 | 0.000 | 0.062 | 0.062 |

