# mechanism separation summary Mechanism Separation Results

## Three-way comparison: base vs all-parameter-116-map vs tight-private-480-map

All results on full heldout rows (not f00 only).

### Base chck_82M (on 116-map family, 4,800 rows)
- raw accuracy: 0.2329
- shared-context pairs: 5/2400 (0.21%), both_wrong: 2057/2400 (85.71%)
- full quads: 0/1200 (0.00%)
- Interpretation: identity-blind, near chance

### All-parameter 116-map (35,463,008 params trainable, 25 epochs, lr 5e-5)
- raw accuracy: 0.3148
- shared-context pairs: 169/2400 (7.04%), both_wrong: 1058/2400 (44.08%)
- full quads: 9/1200 (0.75%)
- By k: k0 0.345, k1 0.342, k2 0.311, k3 0.280, k4 0.296
- By role: retained 0.375, updated 0.254
- train_seen 0.316, eval_unseen 0.313
- Training loss reached 0.00056 but heldout barely improved
- Interpretation: 116 maps too few even for full-model training; operator not acquired

### Tight-private 480-map (995,584 private params, frozen slow path, 25 epochs)
- raw accuracy: 0.5883
- shared-context pairs: 2646/6640 (39.85%), both_wrong: 1473/6640 (22.18%)
- full quads: 895/3320 (26.96%)
- By k: k0 0.642, k1 0.611, k2 0.598, k3 0.574, k4 0.517
- By role: retained 0.619, updated 0.558
- train_seen 0.613, eval_unseen 0.547
- **By relation: birth_year 0.873, birthplace 0.505, death_place 0.397**
- f00 readout during training: 0.942 accuracy, 268/300 pairs

## Key findings

1. **Instance diversity was the bottleneck, not frozen-private capacity.** The same 995,584-parameter private branch that failed on 116 maps succeeded spectacularly on 480 maps (26.96% full quads vs 0.75%).

2. **The private adapter CAN learn a new relational operator.** The literal-state assignment operator was acquired from 397 training maps and generalized to 83 unseen heldout maps with 54.7% accuracy.

3. **Relation-specific performance varies.** birth_year (numeric, distinctive) was learned at 87.3%, while death_place (string, more ambiguous) reached 39.7%. The operator is not uniformly strong across relation types.

4. **The k-degradation curve is modest.** k0 → k4 accuracy drops from 0.642 to 0.517 (80% retained), showing the operator handles stranger operations reasonably well.

5. **No k4_pos4 anomaly in the tight family.** The 116-map family's k4_pos4 updated_acc=0.0 was likely map-subset difficulty: the 480-map family has k4_pos4=0.539 with both updated (0.512) and retained (0.566) well above base.

## Scientific implications

The private continuation channel is not intrinsically limited to rebiasing existing readouts. When the training family provides enough semantically clean, balanced instances of the target relation, the 995K private adapter can learn a new relational operator that generalizes across unseen maps and frames.

The failure of the earlier 116-map pilot (earlier analysis) is now fully attributed to insufficient instance diversity, not capacity limitation. The failure of the all-parameter 116-map run confirms this: even 35.5M trainable parameters couldn't overcome the sparse family.

This restores the possibility that the private phase can import new competence from appropriately designed training material — the key open question is whether this controlled mechanism substrate can be made legal and whether the learned operator transfers to official evaluation.

## Files
- Base: `experiments/archive/relation_learning/data/base_readout/chck82_full/summary.json`
- All-param 116-map: `experiments/archive/relation_learning/data/allparam_mirrored_explicit_full/summary.json`
- Tight-private 480-map: `experiments/archive/relation_learning/data/tight_private_mirrored_explicit_full/summary.json`
- Tight-private checkpoint: `experiments/archive/relation_learning/data/tight_private_pilot/checkpoint`
