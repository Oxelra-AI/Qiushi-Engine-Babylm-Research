# earlier analysis special-token CE gap

This measures the frozen chck_82M slow path with private adapters disabled. Positive gap means adding official [CLS]/[SEP]-style special tokens raises masked-MLM CE on the same text rows.

## coherent_leash
Rows `256`, words `39553`, source `experiments/archive/relation_learning/data/held_coherent_sets/held_coherent_leash_256rows.jsonl`.
- without specials: CE `2.355182`, targets `8451`, target ratio `0.148378`, non-special tokens `56956`
- with specials: CE `2.432736`, targets `8349`, target ratio `0.146930`, non-special tokens `56823`
- special-minus-no-special CE gap: `0.077554` (intermediate)
- truncation without/with specials: `63`/`67` rows, lost tokens `1377`/`1510`

## coherent_readout
Rows `256`, words `39320`, source `experiments/archive/relation_learning/data/held_coherent_sets/held_coherent_readout_256rows.jsonl`.
- without specials: CE `2.340766`, targets `8135`, target ratio `0.146413`, non-special tokens `55562`
- with specials: CE `2.350964`, targets `8194`, target ratio `0.147743`, non-special tokens `55461`
- special-minus-no-special CE gap: `0.010198` (few-hundredths)
- truncation without/with specials: `49`/`51` rows, lost tokens `939`/`1040`

## isolated_first_macro
Rows `3819`, words `39553`, source `experiments/archive/relation_learning/data/isolated_replay_streams/isolated_all_replay_3992800w.jsonl`.
- without specials: CE `3.456768`, targets `8632`, target ratio `0.148041`, non-special tokens `58308`
- with specials: CE `3.755711`, targets `8640`, target ratio `0.148184`, non-special tokens `58306`
- special-minus-no-special CE gap: `0.298943` (tenths_or_larger)
- truncation without/with specials: `1`/`1` rows, lost tokens `25`/`27`

JSON: `experiments/archive/relation_learning/data/special_token_ce_gap/special_token_ce_gap.json`
