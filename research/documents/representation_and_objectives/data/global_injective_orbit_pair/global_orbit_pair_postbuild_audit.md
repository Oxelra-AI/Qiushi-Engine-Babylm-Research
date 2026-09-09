# orbit pair audit and slot reuse update post-build audit of global-compatible orbit pair

Rows/words: orig 64740/10000000, per 64740/10000000, stable 64740/10000000

Changed rows: per 16686, stable 16686; changed row mismatch 0; changed key support mismatch 0

## Alias persistence

Stable: 10555 keys, multi-alias keys 0, P_same=1.0000, mean H(A|K)=0.0000

Per-row: 10555 keys, multi-alias keys 3446, P_same=0.0038, mean H(A|K)=0.5816, same-as-stable events 39/25525=0.0015

## Repeated-key dose

Events on repeated keys: 18418/25525 = 0.7216; keys repeated/singleton 3448/7107

## Inverse alias collisions in stable arm

Aliases: 10555; aliases with multiple source keys: 0; events on them: 0; mean H(K|A)=0.0000

## EWoK lexical overlap

Transformed keys in relational EWoK lexicon: 5 keys / 21 key-events

Transformed keys in any EWoK lexicon: 6 keys / 22 key-events

Changed rows with >=1 relational EWoK token in context: 16571/16686; with any EWoK token: 16656/16686

Top transformed keys that are themselves in the relational EWoK lexicon:

[('fatima', 6), ('ali', 5), ('carmen', 4), ('jesse', 4), ('mohammed', 2)]

JSON: `experiments/archive/representation_and_objectives/data/global_injective_orbit_pair/global_orbit_pair_postbuild_audit.json`
