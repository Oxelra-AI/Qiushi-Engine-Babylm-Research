# orbit pair audit and slot reuse update post-build audit of global-compatible orbit pair

Rows/words: orig 64740/10000000, per 64740/10000000, stable 64740/10000000

Changed rows: per 16909, stable 16909; changed row mismatch 0; changed key support mismatch 0

## Alias persistence

Stable: 10649 keys, multi-alias keys 0, P_same=1.0000, mean H(A|K)=0.0000

Per-row: 10649 keys, multi-alias keys 3444, P_same=0.1199, mean H(A|K)=0.5749, same-as-stable events 254/25839=0.0098

## Repeated-key dose

Events on repeated keys: 18638/25839 = 0.7213; keys repeated/singleton 3448/7201

## Inverse alias collisions in stable arm

Aliases: 3173; aliases with multiple source keys: 1979; events on them: 21481; mean H(K|A)=1.1311

## EWoK lexical overlap

Transformed keys in relational EWoK lexicon: 5 keys / 21 key-events

Transformed keys in any EWoK lexicon: 6 keys / 22 key-events

Changed rows with >=1 relational EWoK token in context: 16793/16909; with any EWoK token: 16879/16909

Top transformed keys that are themselves in the relational EWoK lexicon:

[('fatima', 6), ('ali', 5), ('carmen', 4), ('jesse', 4), ('mohammed', 2)]

JSON: `experiments/archive/representation_and_objectives/data/global_compatible_orbit_pair/global_orbit_pair_postbuild_audit.json`
