# Attribution and Material Rights

The original project sources, third-party software, model weights, natural-text
inputs and report artwork have distinct rights. A source hash records identity;
it does not confer a license.

| Material | Provenance | License / handling |
| --- | --- | --- |
| Two released models | Pinned versions in `models/manifest.json` | Existing Apache-2.0 licenses and model notices retained |
| BabyLM evaluator | `babylm-org/babylm-eval`, revision `6f825c291e2c4c78ad33b1935fd64d45f52642dc` | Apache-2.0; upstream copyright retained; local classifier and public-path modifications identified |
| Recursive-model reference | `serdardoesml/bblm26-recgpt`, revision `178566ec38540e91dc951d6b094efc89c0ee19b3` | MIT; existing license and source notices retained |
| Original research programs and notes | Original research artifacts in the file manifest | Code and software configurations: Apache-2.0; written notes, plans, reports and scientific figures: CC BY 4.0; third-party notices take precedence |
| Training and evaluation text | Original dataset providers listed in model/data records | Original source terms; no blanket relicensing or unreviewed bulk redistribution |
| EWoK text in tokenizer-interface diagnostics | Ivanova, Sathe, Lipkin and colleagues, 2024; `ewok-core/ewok` | CC BY 4.0 and [upstream terms](https://github.com/ewok-core/ewok/blob/main/TERMS_OF_USE.txt); full text in the [protected archive](../data/protected/README.md), numeric projections in ordinary files |
| Qiushi name and mark | Project assets | No trademark grant inferred from source availability |

The public archive includes original and adapted research sources. Path and
identifier changes are recorded through original/public file hashes;
the original research evidence is retained separately. The LAMB reference and
BabyLM leaderboard source snapshots have accompanying Apache-2.0 licenses and
source notices under `data/external/` and the corresponding research archive.
Their historical captures do not identify an upstream Git revision.

Scientific results, negative controls and corrected interpretations are retained
as research evidence. Historical checkpoint metadata is not a promise that every
intermediate weight is distributed. Final weights and exposed trajectories have
separate immutable download identities.
