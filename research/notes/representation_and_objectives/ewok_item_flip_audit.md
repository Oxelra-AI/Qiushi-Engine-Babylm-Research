# ewok item flip audit — EWoK item flip audit

This CPU audit reconstructs official EWoK correctness from the four existing prediction files. It does not contain log-likelihood margins; predictions store only selected alternatives.

Total EWoK items: 7618
Micro treatment×seed interaction: -2.389 percentage points (official macro interaction from the 2×2 note is -2.462).

## Worst official domains by micro interaction
- material-dynamics: DiD=-17.532, TE430=9.870, TE431=-7.662, n=770, neg2=70, neg1=214
- physical-dynamics: DiD=-10.833, TE430=7.500, TE431=-3.333, n=120, neg2=2, neg1=31
- spatial-relations: DiD=-8.367, TE430=2.653, TE431=-5.714, n=490, neg2=16, neg1=101
- physical-interactions: DiD=-5.935, TE430=4.317, TE431=-1.619, n=556, neg2=19, neg1=117
- social-relations: DiD=-2.132, TE430=0.065, TE431=-2.067, n=1548, neg2=61, neg1=338
- quantitative-properties: DiD=-0.955, TE430=0.637, TE431=-0.318, n=314, neg2=7, neg1=69
- physical-relations: DiD=-0.489, TE430=2.323, TE431=1.834, n=818, neg2=26, neg1=133
- agent-properties: DiD=1.719, TE430=-0.995, TE431=0.724, n=2210, neg2=43, neg1=354

## Worst metadata groups (n>=20)
- domain+ContextDiff {'domain': 'physical-interactions', 'ContextDiff': 'variable_swap'}: DiD=-30.000, TE430=16.667, TE431=-13.333, n=30
- domain+ContextDiff {'domain': 'physical-interactions', 'ContextDiff': 'negation'}: DiD=-25.000, TE430=25.000, TE431=0.000, n=20
- ContextType+ContextDiff+TargetDiff {'ContextType': 'direct', 'ContextDiff': 'variable_swap', 'TargetDiff': 'variable swap'}: DiD=-25.000, TE430=15.000, TE431=-10.000, n=20
- domain+ContextType {'domain': 'material-dynamics', 'ContextType': 'direct'}: DiD=-19.167, TE430=6.389, TE431=-12.778, n=360
- ContextType+ContextDiff+TargetDiff {'ContextType': 'direct', 'ContextDiff': 'material', 'TargetDiff': 'concept swap'}: DiD=-19.167, TE430=6.389, TE431=-12.778, n=360
- domain+ContextType {'domain': 'physical-dynamics', 'ContextType': 'direct'}: DiD=-18.333, TE430=18.333, TE431=0.000, n=60
- domain+ContextType {'domain': 'physical-interactions', 'ContextType': 'direct'}: DiD=-18.145, TE430=8.468, TE431=-9.677, n=248
- domain+ContextDiff {'domain': 'material-dynamics', 'ContextDiff': 'material'}: DiD=-17.532, TE430=9.870, TE431=-7.662, n=770
- domain+TargetDiff {'domain': 'material-dynamics', 'TargetDiff': 'concept swap'}: DiD=-17.532, TE430=9.870, TE431=-7.662, n=770
- domain+ContextDiff {'domain': 'physical-dynamics', 'ContextDiff': 'antonym'}: DiD=-16.667, TE430=11.111, TE431=-5.556, n=90
- domain+ContextType {'domain': 'material-dynamics', 'ContextType': 'indirect'}: DiD=-16.098, TE430=12.927, TE431=-3.171, n=410
- domain+ContextDiff {'domain': 'social-interactions', 'ContextDiff': 'variable swap'}: DiD=-15.625, TE430=0.000, TE431=-15.625, n=32

## Worst correctness-bit patterns
Pattern order is clean430, reinvest430, clean431, reinvest431.
- 0110: n=268, DiD=-200.000, TE430=100.000, TE431=-100.000
- 0100: n=398, DiD=-100.000, TE430=100.000, TE431=0.000
- 1110: n=383, DiD=-100.000, TE430=0.000, TE431=-100.000
- 0111: n=364, DiD=-100.000, TE430=100.000, TE431=0.000
- 0010: n=353, DiD=-100.000, TE430=0.000, TE431=-100.000
- 1111: n=1703, DiD=0.000, TE430=0.000, TE431=0.000
- 0000: n=1584, DiD=0.000, TE430=0.000, TE431=0.000
- 1100: n=255, DiD=0.000, TE430=0.000, TE431=0.000

## Next use
Use the negative-example CSV as a target list for margin rescoring and for compact-row relation-feature analysis. A margin audit should instrument `sentence_zero_shot/compute_results.py`: for MLM, per-candidate summed log-probs are built in `compute_mlm_results` and passed to `rank_and_evaluate`, but only the selected sentence is saved. Export `stacked_probs` or candidate scores before argmax.

JSON: `experiments/archive/representation_and_objectives/data/ewok_item_flip_audit/ewok_item_flip_audit.json`
Negative examples CSV: `experiments/archive/representation_and_objectives/data/ewok_item_flip_audit/ewok_negative_interaction_examples.csv`
Group CSV: `experiments/archive/representation_and_objectives/data/ewok_item_flip_audit/ewok_group_interactions.csv`
