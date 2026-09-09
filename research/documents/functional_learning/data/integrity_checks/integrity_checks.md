# held fitted direction test integrity checks

Parent: `models/frontier`

- parent SHA256: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`
- private tensor keys: 48; missing private keys on load: 0; unexpected: 0
- executed private scales after load: [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

## Runs

- `standard_carrier_anchor` exists=True total=100000000 updates=354 first_main=2.490510255098343 first_neutral=0.001525211046100594
- `parent_anchor` exists=True total=100000000 updates=354 first_main=2.490510255098343 first_neutral=-9.374094679746175e-10
- `no_kl` exists=True total=100000000 updates=354 first_main=2.490510255098343 first_neutral=0.0

## First-update pairing vs clean d component ablation standard

- `parent_anchor` equalities: {'source_row_start': True, 'source_row_end': True, 'macro_words': True, 'tail_words': True, 'total_consumed_words': True, 'main_targets': True, 'schedule_index': True, 'lr': True, 'main_loss': True}; neutral losses 0.001525211046100594 vs -9.374094679746175e-10
- `no_kl` equalities: {'source_row_start': True, 'source_row_end': True, 'macro_words': True, 'tail_words': True, 'total_consumed_words': True, 'main_targets': True, 'schedule_index': True, 'lr': True, 'main_loss': True}; neutral losses 0.001525211046100594 vs 0.0
