# route reopen conditional innovation — compact-pair innovation target mass

CPU-only; no model update, no official evaluation, no corpus/tokenizer change, no H100 work.

- changed rows loaded: `3005` of `3005`
- train SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## One-pass rewrite word-group mass
- all rewrite groups: `158110`; all rewrite tokens in groups: `241204`
- copyable groups: `131675` (0.833); tokens `203751` (0.845)
- innovation groups: `26435` (0.167); tokens `37453` (0.155)
- innovation relation-cue groups: `3500` (0.022); tokens `3599` (0.015)
- innovation content/other groups: `22935` (0.145); tokens `33854` (0.140)
- rows with at least one innovation group: `3004` (1.000); pairs with innovation: `10554` (0.869)

## Ten-pass exposure equivalent inside the fixed 100M stream
- innovation groups over ten passes: `264350`; innovation tokens over ten passes: `374530`
- relation-cue innovation groups over ten passes: `35000`; tokens: `35990`

## Top source-absent innovation normalized words
- `for`: 499
- `to`: 377
- `with`: 355
- `and`: 289
- `from`: 270
- `in`: 255
- `it`: 233
- `like`: 230
- `because`: 170
- `on`: 168
- `has`: 159
- `use`: 155
- `by`: 146
- `so`: 139
- `shows`: 136
- `as`: 133
- `that`: 131
- `while`: 129
- `is`: 128
- `but`: 126
- `them`: 122
- `says`: 122
- `or`: 114
- `must`: 114
- `have`: 113

## Top relation-cue innovations
- `to`: 377
- `with`: 355
- `from`: 270
- `in`: 255
- `because`: 170
- `on`: 168
- `while`: 129
- `but`: 126
- `causes`: 107
- `not`: 93
- `cause`: 92
- `makes`: 91
- `over`: 87
- `make`: 86
- `after`: 85
- `if`: 78
- `then`: 73
- `when`: 68
- `made`: 65
- `without`: 62
- `more`: 57
- `no`: 53
- `caused`: 49
- `during`: 46
- `only`: 40

Full JSON: `experiments/archive/frontier_consolidation/data/innovation_target_mass/innovation_target_mass.json`
