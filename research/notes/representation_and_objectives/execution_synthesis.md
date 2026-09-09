# execution synthesis — execution synthesis: mature FW evidence and next mechanism pressure

## Core scientific update

This step read the completed FW 100M checkpoints without launching new H100 work. The important result is not a simple compact-over-breadth ordering. The row-block independent-breadth arm creates real movement on the hard GlobalPIQA_parallel preference surface, but it does so while damaging broader capability.

### Mature 100M GlobalPIQA

From `experiments/archive/representation_and_objectives/data/fw_globalpiqa_margin_reader/globalpiqa_margin_reader_results.json` and synthesized in `research/notes/representation_and_objectives/fw_100m_globalpiqa_mature_synthesis.md`:

| arm | GlobalPIQA_parallel | GlobalPIQA_nonparallel | aggregate | fw globalpiqa relevant substrate hard52 acc | hard52 mean top-minus-correct |
|---|---:|---:|---:|---:|---:|
| compact | 24.27 | 53.00 | 38.64 | 3.85 | 1.717 |
| row-block breadth | 29.13 | 45.00 | 37.06 | 5.77 | 1.433 |

Row-block breadth minus compact:

- parallel +4.85
- nonparallel -8.00
- aggregate GlobalPIQA -1.57
- hard52 mean top-minus-correct -0.284 nats
- hard52 rank1 +1 row
- hard52 small-wrong-margin rows within 0.50 nats +4 rows

Thus row-block breadth is the first completed arm that moves the deep hard-parallel preference surface in the desired direction, but it does not carry that movement together with compact-like broad GlobalPIQA.

### SOTA arithmetic

Compact 100M GlobalPIQA (38.64) is essentially the same as its 70M cheap value (38.605), so the 70M cheap7 score is a useful arithmetic anchor for the completed compact endpoint. Compact 70M cheap7 = 42.657 would require SuperGLUE+AoA = 77.60 to reach Overall 41.80. Row-block breadth would require SuperGLUE+AoA = 81.98. Known complete local values are far lower: compliant SuperGLUE is about 67.08–68.23, and the non-submission inherited-tokenizer model reached 71.04. Therefore the completed compact and row-block breadth arms are not SOTA-plausible through known score geometry, even though they are scientifically useful mechanism evidence.

## Row complementarity

`experiments/archive/representation_and_objectives/scripts/fw_globalpiqa_row_complementarity.py` reads only existing 100M per-row CSVs and records that parent `hf_model` files equal `hf_model/chck_100M` for `model.safetensors`, `tokenizer.json`, and `config.json`, so the 100M readout used the intended final weights.

Output summary (`research/notes/representation_and_objectives/fw_globalpiqa_row_complementarity.md`):

- GlobalPIQA_parallel: compact 24.27, breadth 29.13, exact-label oracle union 34.95; compact-only correct 6, breadth-only correct 11, both correct 19, both wrong 67.
- fw globalpiqa relevant substrate hard52 subset: compact correct 2, breadth correct 3, both correct 1, compact-only 1, breadth-only 2, both wrong 48; breadth improves rank on 20 hard rows and lowers margin on 30 hard rows.
- GlobalPIQA_nonparallel: compact 53.00, breadth 45.00, exact-label oracle union 60.00; compact-only correct 15, breadth-only correct 7, both correct 38, both wrong 40.

The arms are partly complementary at the row level, but the complementarity is not yet usable because it is split between models. Row-block breadth repairs some compact-wrong parallel rows while damaging more nonparallel rows; compact preserves nonparallel but keeps the deep hard52 weakness.

## independent_review-supported mechanism reading

independent_review generator/verifier integrations (`data/external/independent_review01_generator1_integration.md`, `data/external/independent_review01_verifier1_integration.md`) support the following interpretation:

1. Compact recurrence and row-block independent coverage should be treated as two incomplete mechanisms, not as a winner/loser pair.
2. The desired object is a single training mechanism that keeps compact's broad capability while moving the hard relation ranks/margins.
3. The next free measurement is row-level complementarity, which this step completed for GlobalPIQA.
4. The pending interleaved breadth arm remains essential because it separates independent coverage from row-block arrangement.
5. The 100M EWoK four-cell reader is needed because 70M EWoK gains may be domain-weighted surface movement rather than repair of the stable conditional-reversal rows.
6. If the desired combination does not appear, do not extend FW allocation variants. Use the anchor matched controls ultra-clean transition/control assets only as a small instrument to study a stronger mechanism, not as a direct full-scale corpus route.

independent_review also emphasized that relation repair alone is not enough for SOTA unless it lifts a broader representation that also improves the missing score mass, especially SuperGLUE and general language columns.

## Pending evidence

- `s106_t10_tool1`: interleaved whole-sentence breadth training. This must not be duplicated or restarted. When it delivers, first verify training identity, then run the minimum cheap/readout sequence: cheap7, GlobalPIQA margins, EWoK four-cell reader, and only then consider official full evaluation if arithmetic becomes plausible.
- `s112_t16_tool1`: CPU EWoK four-cell interaction reader for compact and row-block breadth 100M endpoints. Its output should show whether row-block breadth's EWoK movement touches the stable conditional-reversal structure or remains a domain-weighted shift.

## Next work

Do not launch new FW allocation variants from the completed arms. The next decisive work is to absorb the two pending pieces above. If interleaved shows compact-like broad scores and row-block-like hard relation margins, layout/source alternation becomes an active mechanism. If interleaved resembles compact or row-block without the desired combination, stop the FW allocation branch and use the anchor matched controls 30k transition/control probe only to understand what relation signal can be learned without sacrificing broad capability.
