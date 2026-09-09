# earlier analysis relation/state official IG probe

Selected official EWoK/Entity items: `2079`; radius `8`; device `cuda`.

## Status counts in full common set
- EWoK: {'both_correct': 3513, 'both_wrong': 3545, 'gain': 269, 'loss': 291}
- Entity: {'both_wrong': 4748, 'both_correct': 1713, 'loss': 178, 'gain': 141}

## chck82 by official flip status
- EWoK:both_correct: n=300; fullNLL mean=2.930517; IG mean=1.496653; ΔNLL100-82 mean=-0.011676; lowIG+highErr frac=0.100; worsened gold NLL frac=0.513
- EWoK:both_wrong: n=300; fullNLL mean=3.447590; IG mean=1.324980; ΔNLL100-82 mean=-0.020613; lowIG+highErr frac=0.150; worsened gold NLL frac=0.490
- EWoK:gain: n=269; fullNLL mean=3.377747; IG mean=1.349465; ΔNLL100-82 mean=-0.064953; lowIG+highErr frac=0.190; worsened gold NLL frac=0.379
- EWoK:loss: n=291; fullNLL mean=3.371507; IG mean=1.280914; ΔNLL100-82 mean=0.017808; lowIG+highErr frac=0.162; worsened gold NLL frac=0.540
- Entity:both_correct: n=300; fullNLL mean=1.875742; IG mean=2.613746; ΔNLL100-82 mean=-0.023876; lowIG+highErr frac=0.013; worsened gold NLL frac=0.430
- Entity:both_wrong: n=300; fullNLL mean=3.612755; IG mean=1.183972; ΔNLL100-82 mean=-0.093921; lowIG+highErr frac=0.130; worsened gold NLL frac=0.337
- Entity:gain: n=141; fullNLL mean=2.697897; IG mean=1.888285; ΔNLL100-82 mean=-0.226146; lowIG+highErr frac=0.021; worsened gold NLL frac=0.099
- Entity:loss: n=178; fullNLL mean=2.603983; IG mean=1.979794; ΔNLL100-82 mean=0.135753; lowIG+highErr frac=0.011; worsened gold NLL frac=0.725

## Loss minus gain contrasts
- EWoK: loss-gain fullNLL=-0.006239; loss-gain IG=-0.068550; loss-gain ΔNLL100-82=+0.082762; lowIG+highErr frac loss/gain=0.162/0.190
- Entity: loss-gain fullNLL=-0.093914; loss-gain IG=+0.091509; loss-gain ΔNLL100-82=+0.361898; lowIG+highErr frac loss/gain=0.011/0.021

## Direct reading
- If official relation/state losses were the same low-IG/high-error residual, loss items should have higher chck82 gold NLL, lower IG, and/or larger positive 100M-minus-82M gold-NLL deltas than gains. The numbers above are the bounded test of that connection.

CSV: `experiments/archive/frontier_consolidation/data/relation_state_official_ig_probe/relation_state_official_ig_items.csv`
JSON: `experiments/archive/frontier_consolidation/data/relation_state_official_ig_probe/relation_state_official_ig_summary.json`
