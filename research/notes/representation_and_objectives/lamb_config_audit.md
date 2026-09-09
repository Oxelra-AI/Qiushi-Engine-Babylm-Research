# lamb config audit — config audit for lamb curriculum route decision LAMB trainer

The lamb curriculum route decision smoke checkpoint was written by the same `create_model` code as the active full runs. It differs from the matched legal40k DeBERTa-v2 coordinate in non-optimizer ways.

## Tokenizer special ids
- tokenizer pad/mask/bos/eos/unk ids: pad=3, mask=4, bos=1, eos=2, unk=0

## Non-size config mismatches: lamb curriculum route decision smoke vs matched 12×384 AdamW depth
| field | lamb curriculum route decision smoke | matched 12×384 AdamW |
|---|---:|---:|
| pad_token_id | `0` | `3` |
| bos_token_id | `None` | `1` |
| eos_token_id | `None` | `2` |
| max_relative_positions | `-1` | `256` |
| pos_att_type | `None` | `['p2c', 'c2p']` |
| norm_rel_ebd | `layer_norm` | `None` |

Load-bearing mismatches:
- lamb curriculum route decision config uses `pad_token_id=0`, while the legal40k tokenizer pad id is 3 and matched baselines set pad id 3.
- lamb curriculum route decision config has `pos_att_type=null` and `max_relative_positions=-1`; matched baselines use `pos_att_type=[p2c,c2p]` and `max_relative_positions=256`.
- lamb curriculum route decision omits bos/eos ids in the saved config, unlike matched baselines.

Therefore the active endpoints are a joint optimizer × sequence-curriculum × chunking/masking × config-coordinate intervention. They may still be legal models, but their scores cannot separate LAMB/curriculum from these architecture/config differences.

Summary JSON: `experiments/archive/representation_and_objectives/data/lamb_config_audit/lamb_config_audit.json`
