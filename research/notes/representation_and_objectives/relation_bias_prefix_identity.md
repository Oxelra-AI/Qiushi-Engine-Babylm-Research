# relation bias execution replay relation-bias prefix identity

No-model check comparing the exact 10M pool against the first 10M charged words consumed from the 100M launch stream.

- Pool SHA matched expected: `True`; stream SHA matched expected: `True`.
- Pool rows/words: `64740` / `10000000`; stream prefix rows/words: `64740` / `10000000`.
- Row-object multiset equal: `True`; source words equal: `True`.
- Stream-only row count: `0`; pool-only row count: `0`.

## Interpretation

- The 100M launch stream first 10M charged words are exactly a row-object multiset permutation of the audited 10M pool; relation bias execution replay full-pool relation-dose statistics describe the actual first training pass, though not its sequence order.

JSON: `experiments/archive/representation_and_objectives/data/relation_bias_prefix_identity/relation_bias_prefix_identity.json`
