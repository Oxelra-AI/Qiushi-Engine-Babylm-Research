# relation bias execution replay relation-biased masking execution replay

CPU-only replay of the exact relation bias trainer preflight masking code on the 100M launch stream prefix. No optimization, model forward, GPU, or official evaluation text.

## Prefix identity and integrity

- 100M stream SHA matched expected: `True`; 10M pool SHA matched expected: `True`; tokenizer SHA matched expected: `True`.
- Prefix text sequence equal: `False`; row multiset equal: `False`; stream prefix rows/words `10000` / `1544605`, pool rows/words `10000` / `1507476`.
- Relation dataset matched base input/attention/word_group for first 256 rows: `True`.

## Realized masking replay

- Mask seeds replayed: `[43023, 43123]`.
- Relation selected-group fraction base -> boost2: `0.109144` -> `0.220130`.
- Relation selected-token fraction base -> boost2: `0.095623` -> `0.195182`.
- Realized boost/base selected-group ratio: `1.000142`; selected-token ratio: `0.986858`.
- Expected boost selected-group multiplier vs uniform: `1.000000`; expected token multiplier: `0.986387`.
- Boost clipped rows fraction: `0.000000`.
- Zero-selected row fraction base/boost: `0.000000` / `0.000000`.
- Partial selected groups base/boost: `0` / `0`.
- Special random replacements base/boost: `8` / `5`.

## Interpretation

- The checked 100M launch stream prefix subset is not the same row multiset as the compared 10M rows; first sequence mismatch 0, stream-only rows 8477, pool-only rows 8477. Full-prefix equality is required before a relation run is interpreted.
- Realized boost-2 masks over the replayed mask seeds moved relation selected groups from 0.1091 to 0.2201 and relation selected tokens from 0.0956 to 0.1952.
- Realized selected group count under boost/base ratio was 1.0001; target-token ratio was 0.9869.
- Boost expected group multiplier against uniform was 1.000000; expected token multiplier was 0.986387; row clipping fraction was 0.000000.
- Zero-selected rows occurred in base fraction 0.000000 and boost fraction 0.000000; partial selected groups were base=0, boost=0.
- Observed random-like replacements can include special IDs in both paths: base total 8, boost total 5 over the replayed masks. This is inherited from the base corruption policy rather than unique to relation bias.
- Realized source-class boost dose ranges from 0.1212 (simple_wiki) to 0.2688 (bnc_spoken), consistent with broad but source-dependent pressure.

## Files

- JSON: `experiments/archive/representation_and_objectives/data/relation_bias_execution_replay/relation_bias_execution_replay.json`
- Step records: `experiments/archive/representation_and_objectives/data/relation_bias_execution_replay/replay_step_records.jsonl`
