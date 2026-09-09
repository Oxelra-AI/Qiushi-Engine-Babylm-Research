# fastpath private localization coherent86 identity and private-pathway localization

Status: **COMPLETE**

## Identity and endpoint arithmetic

- Coherent carrier model SHA: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`.
- Replay bit-identical to original: `True`.
- Local prediction validator accepted carrier: `True`.
- Non-private max diff vs protected chck82: `0.0`; private-off probe max diff: `0.0`.
- Total counted exposure: `86005295`; private parameters: `995584`.
- carrier SuperGLUE / Overall(AoA=0): `69.77796826428681` / `42.058107584920755`.
- repeat SuperGLUE / Overall(AoA=0): `69.84279617564938` / `42.06531068618327`.
- Minimum delta vs protected chck82 across the two SuperGLUE runs: `0.1156264175347701`.

## SuperGLUE subtask differences

| task | metric | companion analysis repeat | companion analysis carrier | Δ companion analysis-companion analysis | pred SHA equal |
|---|---|---:|---:|---:|---:|
| boolq | accuracy | 68.56269113149847 | 68.56269113149847 | 0.0 | True |
| mnli | accuracy | 59.800325998370006 | 59.800325998370006 | 0.0 | True |
| mrpc | f1 | 88.73720136518772 | 88.73720136518772 | 0.0 | True |
| multirc | accuracy | 67.73927392739274 | 67.28547854785478 | 0.453795379537965 | False |
| qqp | f1 | 71.51995905834187 | 71.51995905834187 | 0.0 | True |
| rte | accuracy | 63.30935251798561 | 63.30935251798561 | 0.0 | True |
| wsc | accuracy | 69.23076923076923 | 69.23076923076923 | 0.0 | True |

## Private-on decision partition (anchor private-off proxy = protected chck82)

Aggregate over common discrete official items: `{'n_common': 170722, 'anchor_correct': 98478, 'coherent_correct': 98363, 'ordinary_correct': 98596, 'shuffled_correct': 99476, 'private_gain': 3116, 'private_loss': 3231, 'private_retained_anchor_correct': 95247, 'private_remains_wrong': 69128, 'private_gain_also_ordinary': 1791, 'private_gain_not_ordinary': 1325, 'private_gain_also_shuffled': 1485, 'private_gain_not_shuffled': 1631, 'private_gain_unique_vs_ordinary_and_shuffled': 673, 'private_gain_shared_by_both_comparators': 833, 'private_loss_also_ordinary_wrong': 1777, 'private_loss_also_shuffled_wrong': 1427, 'private_loss_unique_coherent_wrong_vs_comparators': 812, 'retained_anchor_correct_ordinary_wrong': 3471, 'retained_anchor_correct_shuffled_wrong': 12920, 'retained_anchor_correct_both_comparators_wrong': 1325, 'coherent_correct_ordinary_wrong': 4796, 'coherent_correct_shuffled_wrong': 14551, 'coherent_correct_both_comparators_wrong': 1998, 'ordinary_correct_coherent_wrong': 5029, 'shuffled_correct_coherent_wrong': 15664, 'anchor_correct_pct': 57.683251133421585, 'coherent_correct_pct': 57.61589016061199, 'ordinary_correct_pct': 57.752369349000126, 'shuffled_correct_pct': 58.267827227890955, 'private_gain_pct': 1.8251894893452514, 'private_loss_pct': 1.8925504621548481, 'private_retained_anchor_correct_pct': 55.79070067126674, 'private_remains_wrong_pct': 40.491559377233166, 'retained_anchor_correct_ordinary_wrong_pct': 2.0331298836705285, 'retained_anchor_correct_shuffled_wrong_pct': 7.567858858260799, 'coherent_correct_ordinary_wrong_pct': 2.8092454399550144, 'ordinary_correct_coherent_wrong_pct': 2.945724628343154, 'coherent_correct_shuffled_wrong_pct': 8.523213176977777, 'shuffled_correct_coherent_wrong_pct': 9.175150244256745, 'coherent_minus_anchor_item_net': -115, 'coherent_minus_ordinary_item_net': -233, 'coherent_minus_shuffled_item_net': -1113}`

| column | n | anchor correct | coherent correct | ordinary correct | shuffled correct | private gains | private losses | coherent-anchor net | coherent-ordinary net | unique private gains vs both comparators | retained anchor correct while ordinary wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 40937 | 40967 | 40925 | 41388 | 761 | 731 | 30 | 42 | 169 | 784 |
| Supplement | 5218 | 3870 | 3856 | 3877 | 3792 | 37 | 51 | -14 | -21 | 13 | 47 |
| EWoK | 7618 | 3804 | 3802 | 3829 | 3836 | 147 | 149 | -2 | -27 | 35 | 159 |
| Entity | 6780 | 1891 | 1898 | 1912 | 1818 | 73 | 66 | 7 | -14 | 20 | 96 |
| COMPS | 91028 | 47900 | 47763 | 47980 | 48562 | 2095 | 2232 | -137 | -217 | 435 | 2377 |
| GlobalPIQA | 203 | 76 | 77 | 73 | 80 | 3 | 2 | 1 | 4 | 1 | 8 |

## Scientific reading

Coherent86 is a stronger practical endpoint than the protected 82M model under both SuperGLUE measurements, and the model/provenance records are internally consistent. The functional private path is real because the non-private path is unchanged and private-off matches the 82M anchor on the validation probe. But the official-item partition does not yet show a broad stability-plasticity learning principle: coherent private-on has negative all-discrete item net versus the anchor, ordinary86, and shuffled-private86, while the aggregate score improves through macro/column redistribution. The existing evidence supports one independent-anchor reproduction only if it is used to test this separation with prebuilt private-on/off and comparator readouts, not to launch longer tails or retention machinery blindly.

JSON: `experiments/archive/representation_and_objectives/data/fastpath_private_localization/fastpath_private_localization.json`
