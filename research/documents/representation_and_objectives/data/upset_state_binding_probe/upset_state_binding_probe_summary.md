# role coordinate anchor and state probe upset-balanced ranking-state substrate probes

The contemporaneous state substrate defeats the event-winner shortcut. Explicit rank numbers remain the intended state evidence: numeric parsing solves state rows, while rank-number ablation tests whether sparse text baselines were reading the state or exploiting residual wording.

## Transparent parsers

| mode | parser | acc | event | state_at_time | state_later |
|---|---|---:|---:|---:|---:|
| event_only | event_winner | 1.000 | 1.000 | nan | nan |
| event_only | first_state_entity | nan | nan | nan | nan |
| event_only | numeric_rank | nan | nan | nan | nan |
| state_at_time_only | event_winner | 0.500 | nan | 0.500 | nan |
| state_at_time_only | first_state_entity | 0.500 | nan | 0.500 | nan |
| state_at_time_only | numeric_rank | 1.000 | nan | 1.000 | nan |
| state_later_only | event_winner | 0.563 | nan | nan | 0.563 |
| state_later_only | first_state_entity | 0.563 | nan | nan | 0.563 |
| state_later_only | numeric_rank | 1.000 | nan | nan | 1.000 |
| event_plus_state_at_time | event_winner | 0.750 | 1.000 | 0.500 | nan |
| event_plus_state_at_time | first_state_entity | 0.500 | nan | 0.500 | nan |
| event_plus_state_at_time | numeric_rank | 1.000 | nan | 1.000 | nan |
| event_plus_state_later | event_winner | 0.781 | 1.000 | nan | 0.563 |
| event_plus_state_later | first_state_entity | 0.563 | nan | nan | 0.563 |
| event_plus_state_later | numeric_rank | 1.000 | nan | nan | 1.000 |
| full_two_state | event_winner | 0.688 | 1.000 | 0.500 | 0.563 |
| full_two_state | first_state_entity | 0.532 | nan | 0.500 | 0.563 |
| full_two_state | numeric_rank | 1.000 | nan | 1.000 | 1.000 |

## TF-IDF logistic held-family accuracy

| mode | field | held acc | event | state_at_time | state_later | untouched |
|---|---|---:|---:|---:|---:|---:|
| event_only | text | 0.495 | 0.495 | nan | nan | nan |
| event_only | text_canon | 0.468 | 0.468 | nan | nan | nan |
| event_only | text_canon_no_numbers | 0.480 | 0.480 | nan | nan | nan |
| state_at_time_only | text | 0.594 | nan | 0.594 | nan | nan |
| state_at_time_only | text_canon | 0.491 | nan | 0.491 | nan | nan |
| state_at_time_only | text_canon_no_numbers | 0.491 | nan | 0.491 | nan | nan |
| state_later_only | text | 0.600 | nan | nan | 0.600 | nan |
| state_later_only | text_canon | 0.557 | nan | nan | 0.557 | nan |
| state_later_only | text_canon_no_numbers | 0.557 | nan | nan | 0.557 | nan |
| event_plus_state_at_time | text | 0.613 | 0.501 | 0.532 | nan | 1.000 |
| event_plus_state_at_time | text_canon | 0.603 | 0.502 | 0.505 | nan | 1.000 |
| event_plus_state_at_time | text_canon_no_numbers | 0.629 | 0.574 | 0.499 | nan | 1.000 |
| event_plus_state_later | text | 0.629 | 0.500 | nan | 0.572 | 1.000 |
| event_plus_state_later | text_canon | 0.625 | 0.502 | nan | 0.560 | 1.000 |
| event_plus_state_later | text_canon_no_numbers | 0.645 | 0.559 | nan | 0.553 | 1.000 |
| full_two_state | text | 0.613 | 0.502 | 0.567 | 0.576 | 1.000 |
| full_two_state | text_canon | 0.583 | 0.498 | 0.486 | 0.556 | 1.000 |
| full_two_state | text_canon_no_numbers | 0.582 | 0.498 | 0.486 | 0.553 | 1.000 |

## State-at-time query inside event+state context only

| field | held acc |
|---|---:|
| text | 0.583 |
| text_canon | 0.491 |
| text_canon_no_numbers | 0.491 |

Summary JSON: `experiments/archive/representation_and_objectives/data/upset_state_binding_probe/upset_state_binding_probe_summary.json`
