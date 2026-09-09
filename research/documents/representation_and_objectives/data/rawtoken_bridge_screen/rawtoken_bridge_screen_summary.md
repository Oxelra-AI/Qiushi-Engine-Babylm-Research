# rawtoken bridge screen and route judgment raw-token bridge screen

## Metric table

| run | held | aff | unaff | paraphrase | multi | write_perm_delta_held | final_loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| vanilla_base20 | 0.578125 | 0.6302083333333334 | 0.5260416666666666 | 0.5911458333333334 | 0.484375 | None | 0.10365734905004502 |
| rawmem_base20 | 0.5546875 | 0.7135416666666666 | 0.3958333333333333 | 0.4817708333333333 | 0.515625 | -0.0026041666666666297 | 0.09592729955911636 |
| rawmem_noevent_base20 | 0.53125 | 0.515625 | 0.546875 | 0.4947916666666667 | 0.5 | 0.0 | 0.12842080150047938 |
| lexmem_base20 | 0.5598958333333334 | 0.640625 | 0.4791666666666667 | 0.3958333333333333 | 0.484375 | 0.0 | 0.06814005663990974 |
| lexmem_noevent_base20 | 0.515625 | 0.5260416666666666 | 0.5052083333333334 | 0.4036458333333333 | 0.5625 | 0.0 | 0.03801449044793844 |
| vanilla_aug20 | 0.5677083333333334 | 0.7395833333333334 | 0.3958333333333333 | 0.34375 | 0.9479166666666666 | None | 0.050678417578330096 |
| lexrec_aug20 | 0.5 | 0.515625 | 0.484375 | 0.4895833333333333 | 0.90625 | 0.0026041666666666297 | 0.040345664227940614 |

## Deltas

```json
{
  "rawmem_base20_minus_vanilla_base20": {
    "held_recomb": -0.0234375,
    "paraphrase": -0.10937500000000006,
    "multi_event": 0.03125,
    "held_recomb_affected": 0.08333333333333326,
    "held_recomb_unaffected": -0.13020833333333331
  },
  "rawmem_noevent_base20_minus_vanilla_base20": {
    "held_recomb": -0.046875,
    "paraphrase": -0.09635416666666669,
    "multi_event": 0.015625,
    "held_recomb_affected": -0.11458333333333337,
    "held_recomb_unaffected": 0.02083333333333337
  },
  "lexmem_base20_minus_vanilla_base20": {
    "held_recomb": -0.01822916666666663,
    "paraphrase": -0.19531250000000006,
    "multi_event": 0.0,
    "held_recomb_affected": 0.01041666666666663,
    "held_recomb_unaffected": -0.046874999999999944
  },
  "lexmem_noevent_base20_minus_vanilla_base20": {
    "held_recomb": -0.0625,
    "paraphrase": -0.18750000000000006,
    "multi_event": 0.078125,
    "held_recomb_affected": -0.10416666666666674,
    "held_recomb_unaffected": -0.02083333333333326
  },
  "lexrec_aug20_minus_vanilla_aug20": {
    "held_recomb": -0.06770833333333337,
    "paraphrase": 0.14583333333333331,
    "multi_event": -0.04166666666666663,
    "held_recomb_affected": -0.22395833333333337,
    "held_recomb_unaffected": 0.08854166666666669
  }
}
```

## Interpretation
- compositional update test design's metadata-assisted learned memory is only an integration ceiling: it was supplied hard slot/state/event/query coordinates.
- Naive soft-slot raw memory overfits training templates but does not outperform vanilla on held/paraphrase and write-permutation barely changes results; event write is not being used as selective entity update.
- Lexical identity grouping and lexical recurrent memory also fail to produce write-sensitive held recombination; lexrec reaches multi-event only because non-held multi-event training makes overwrite learnable by ordinary vanilla too.
- No tested metadata-free raw-token interface satisfies the bridge precondition (held recombination >0.70, paraphrase robustness, multi-event overwrite, write-sensitive identity update). Frozen DeBERTa+memory natural transfer is therefore not scientifically interpretable yet.

## Paraphrase audit

```json
{
  "eval_file": "experiments/archive/representation_and_objectives/data/rawtoken_paraphrase_eval/eval.jsonl",
  "eval_sha256": "2c57cc2ac614468746b6352a8a010c7fa4281a5e9e16e75536f526324d420d5e",
  "paraphrase_rows": 384,
  "proper_quartet_groups": 96,
  "proper_group_size_counts": {
    "4": 96
  },
  "proper_answer_set_size_counts": {
    "2": 96
  },
  "proper_all_groups_have_mixed_answers": true
}
```
