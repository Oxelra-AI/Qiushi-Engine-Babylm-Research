# clbh binding repair result — Repaired CLBH binding discrimination result

## Why this test was necessary

A positive pointer boost from same-token embedding initialization would only prove a structural copy preset, not learned entity–property binding. The required test is whether the output path changes its choice when the queried entity changes while both candidate properties are present in the same context.

earlier analysis had a tokenization bug: for words such as `hat`, `tokenizer(' hat')` returned `[159, 1306]` = standalone space marker plus `Ġhat`. The old test compared the first token, so all candidates collapsed to the same token id 159.

## Repair

Script: `scripts/clbh_binding_repair.py`

Output: `training/runs/clbh_binding_repair_100k/binding_repair_results.json`

The repaired test:

- accepts a candidate only if `tokenizer(' ' + word)` is `[standalone space marker, single semantic word token]` or one matching token;
- uses the semantic word token id such as `Ġhat` or `Ġball`, not the standalone space marker;
- verifies that both candidate semantic token ids occur in the context before the mask;
- constructs paired contexts where both properties occur, but the queried entity changes.

Examples of verified tokens:

- `hat`: token `Ġhat`, id 1306, context position 8;
- `ball`: token `Ġball`, id 1074, context position 3;
- `Paris`: token `ĠParis`, id 3685;
- `London`: token `ĠLondon`, id 2705.

## Experiment

Frozen encoder/head: `training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M`

Trainable CLBH pointer only:

- pointer parameters: 61,921;
- activation: softplus raw QK scores;
- Q/K warm-started from embedding SVD;
- exposure: 100k words, 10 steps;
- loss decreased from 3.475 to 2.997;
- gate remained controlled (mean about 0.06–0.09);
- pointer was alive (`max_copy` around 147–240).

## Aggregate result

Post-training CLBH on 12 repaired binding cases:

| metric | value |
|---|---:|
| copy discrimination fraction | 0.500 |
| combined discrimination fraction | 0.500 |
| base discrimination fraction | 0.500 |
| mean copy margin correct-distractor | +0.0053 |
| mean base margin correct-distractor | -0.1017 |
| mean combined margin correct-distractor | -0.1016 |
| mean gate at mask | 0.0361 |

Shuffled control:

| metric | value |
|---|---:|
| copy discrimination fraction | 0.000 |
| mean copy margin | 0.000 |
| combined margin | same as base |

The CLBH copy path is not identical to shuffled, so it does copy actual context tokens. But the copy signal does **not** solve entity–property binding.

## Per-case interpretation

The decisive pattern is not context-conditioned binding. For many pairs, the pointer favors the same property regardless of which entity is queried:

- In `Alice found a ball. Bob found a hat`, the pointer favors `hat` over `ball` for the Bob query, but also favors `hat` over `ball` for the Alice query, where `ball` is correct.
- In `John lives in Paris. Mary lives in London`, the pointer favors `Paris` over `London` for both John and Mary directions.
- In `Sarah brought milk. Tom brought cake`, the pointer favors `milk` in both directions.
- In `north door open / south door closed`, the pointer favors `open` in both directions.

Thus the 0.5 discrimination fraction is pair asymmetry or token/position salience, not binding. The pointer sees and copies context tokens, but it does not use the queried entity to select the bound property.

## Scientific conclusion

CLBH as currently formulated is a context-copy/output-bias mechanism, not a relation-binding mechanism. It can introduce pointer evidence for tokens present in context, but under the repaired test it does not produce the required entity-conditioned switch in output preference.

The result does **not** justify scaling CLBH to 1M, seed43, or 100M. The output-copy route should stop unless a future design includes an explicit relation representation that binds entity identity to property roles rather than merely copying token identities.

The next route should move to a mechanism that changes relationship representation itself, such as soft latent entity-binding slots with role-sensitive updates, relation-factorized attention, or another architecture where entity and property representations interact before the output head. It should include the repaired binding-switch test as a mandatory pre-training or smoke test before any BabyLM-scale run.
