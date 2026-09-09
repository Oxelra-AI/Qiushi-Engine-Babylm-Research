# ewok tokenizer eval interface audit — official-compatible EWoK margin focus analysis

Source margins: `experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/official_ewok_margin_focus553.json`
Analysis JSON: `experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/official_ewok_margin_focus553_analysis.json`
Confident negative examples: `experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/official_ewok_confident_negative_interaction_examples.csv`

## Validation

The margin exporter mirrors the official MLM EWoK path and exactly matched the saved official argmax predictions on the 553-row focused subset:
- clean430: official prediction match rate 1.0
- reinv430: official prediction match rate 1.0
- clean431: official prediction match rate 1.0
- reinv431: official prediction match rate 1.0

This repairs the decision framework and clean collation plan problem: these margins are tied to the official candidate scoring for these rows, not to the earlier standalone PLL approximation.

## Main result

Focused subset rows: 553. Negative treatment-by-seed rows: 318. Pattern 0110 rows: 168.
Confident negative rows (reinv430 margin > +2 and reinv431 margin < -2): 22.
Moderate negative rows (reinv430 > +1 and reinv431 < -1): 82.
Near negative rows (both reinvest margins within ±1): 51 (0.160 of negative rows).

Negative interaction rows therefore are not mostly zero-margin coin flips. Many are moderate relation preferences with opposite sign in the two treatment seeds.

## Pattern 0110 margin scale

Pattern order is clean430, reinv430, clean431, reinv431 correctness.
- clean430: mean margin -1.672, median -0.944, abs<1 fraction 0.542, abs<2 fraction 0.696
- reinv430: mean margin 1.533, median 1.143, abs<1 fraction 0.423, abs<2 fraction 0.738
- clean431: mean margin 1.888, median 1.344, abs<1 fraction 0.363, abs<2 fraction 0.762
- reinv431: mean margin -1.642, median -1.352, abs<1 fraction 0.351, abs<2 fraction 0.708
- treatment margin shift seed430: mean 3.205, median 2.685
- treatment margin shift seed431: mean -3.530, median -2.974
- margin interaction seed431-minus-seed430: mean -6.735, median -5.638

## Worst domain-level margin interactions on the focused subset

- social-relations: n=111, negative_DiD=91, interaction mean=-6.508, median=-5.493
- material-dynamics: n=120, negative_DiD=100, interaction mean=-3.668, median=-3.291
- spatial-relations: n=66, negative_DiD=46, interaction mean=-3.420, median=-3.223
- physical-interactions: n=69, negative_DiD=49, interaction mean=-2.413, median=-2.253
- physical-dynamics: n=52, negative_DiD=32, interaction mean=-1.342, median=-0.775
- agent-properties: n=45, negative_DiD=0, interaction mean=1.125, median=1.142
- material-properties: n=45, negative_DiD=0, interaction mean=1.422, median=1.048
- social-interactions: n=45, negative_DiD=0, interaction mean=3.791, median=2.850

## Reading for the active route

This is not an endpoint score and remains enriched for old inherited-tokenizer instability. Its scientific value is to show what kind of relation failure the corrected-tokenizer endpoints should be inspected for: if a corrected seed loses EWoK relation rows, look for same moderate opposite-signed margin structure rather than assuming small random ties. The H100 retrains remain the decisive work for compliance; no new pretraining route is justified from this margin subset alone.
