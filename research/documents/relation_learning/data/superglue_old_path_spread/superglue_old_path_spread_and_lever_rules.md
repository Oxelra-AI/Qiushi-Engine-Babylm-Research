# earlier analysis SuperGLUE old-path spread and fixed lever rules

This record is fixed before reading the repaired format endpoints or the repaired faithful SuperGLUE outputs.

## Old stripped AutoModel SuperGLUE spread

earlier analysis/stage3 execution synthesis showed that the old AutoModel path loaded the same stock encoder for custom checkpoints. The following values are therefore observed fixed-protocol repeats of one stripped encoder, not evidence for differences among the adapter/private endpoints.

| label | SuperGLUE | role |
|---|---:|---|
| `chck82_old_stripped` | 69.7661813713118 | protected chck82 slow-adapter reference through the old AutoModel path |
| `coherent86_old_stripped` | 69.81922238969935 | coherent86 alpha0.75 private endpoint through the old AutoModel path |
| `a01_dense62064_old_stripped` | 69.81332894321184 | dense focus seed62064 through the old AutoModel path |
| `a01_dense62065_old_stripped_current_file` | 69.79564860374933 | dense focus seed62065 current file through the old AutoModel path if complete; the contemporaneous notes may still treat it as interim |

Macro spread: min `69.7661813713118`, max `69.81922238969935`, range `0.053041018387546046`, sample std `0.023818080412110133`.

Per-task spread is concentrated in the components whose downstream run is less stable; tasks with zero spread in this record should not be assumed noiseless under a different fine-tuning seed.

| task | range | values |
|---|---:|---|
| `boolq` | 0.0 | [68.56269113149847, 68.56269113149847, 68.56269113149847, 68.56269113149847] |
| `multirc` | 0.37128712871286496 | [67.20297029702971, 67.57425742574257, 67.53300330033002, 67.4092409240924] |
| `rte` | 0.0 | [63.30935251798561, 63.30935251798561, 63.30935251798561, 63.30935251798561] |
| `wsc` | 0.0 | [69.23076923076923, 69.23076923076923, 69.23076923076923, 69.23076923076923] |
| `mrpc` | 0.0 | [88.73720136518772, 88.73720136518772, 88.73720136518772, 88.73720136518772] |
| `qqp` | 0.0 | [71.51995905834187, 71.51995905834187, 71.51995905834187, 71.51995905834187] |
| `mnli` | 0.0 | [59.800325998370006, 59.800325998370006, 59.800325998370006, 59.800325998370006] |

## Fixed reading rules before new endpoint scores

- **superglue_reading**: A repaired coherent86 SuperGLUE movement whose magnitude is no larger than the old stripped-path range should be read as ordinary fixed-protocol variation. A movement outside that range is model-file fidelity evidence. Repaired coherent86 versus repaired chck82 also changes the number of trainable encoder parameters during downstream fine-tuning, so it is not evidence that coherent replay content helps SuperGLUE.
- **v5_bar**: The training contribution must beat faithful v4, where faithful v4 is coherent86 cheap7 plus repaired coherent86 SuperGLUE plus measured coherent86 AoA. The historical 42.1210 number remains the old stripped-path reference, not the bar for a new trained v5.
- **lever_entry_rule**: A lever can enter the composed candidate only when both private seeds move the same relevant columns in the same direction beyond the two-seed coherent band and no important column falls beyond its band. The composed candidate is then fixed once and trained/evaluated across two private seeds without further post-result mixing.
- **format_hierarchy**: coherent_unsplit_special is the special-token control; isolated_all adds row isolation; half_coherent_half_isolated tests context-presence conditioning. Read them in that order.
- **superglue_seed_repeat**: Before any submission, rerun SuperGLUE for the selected final endpoint under one additional fine-tuning seed; CB/COPA/WSC-like small validation tasks can be noisy even after file loading is repaired.

Old coherent86 Overall reproduced from old stripped SuperGLUE and AoA0: `42.12102470996659`. The new training target is not this number; it is faithful v4 after repaired SuperGLUE returns.

JSON: `experiments/archive/relation_learning/data/superglue_old_path_spread/superglue_old_path_spread_and_lever_rules.json`
