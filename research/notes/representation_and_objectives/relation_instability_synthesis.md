# ewok item flip audit — compact-view seed interaction and relation-stability route

## Same-coordinate 2×2
- clean_43022 Overall: 41.347633
- clean_43122 Overall: 40.767775
- reinvest_43022 Overall: 42.033135
- reinvest_43122 Overall: 41.24824
- Treatment effect at seed43022: 0.685502 Overall.
- Treatment effect at seed43122: 0.480465 Overall.
- Two-seed average treatment effect: 0.582983 Overall.
- Treatment×seed interaction: -0.205037 Overall.

The below-leader absolute seed43122 endpoint is not a verdict against compact-view reinvestment. The treatment remains positive within both seeds. Most of the absolute drop from seed43022 to seed43122 already exists in the clean backbone; the extra treatment-specific drop is about 0.205 Overall.

## Column interaction
Worst treatment×seed column interaction:
- EWoK: -2.462123
- Entity: -0.960402
- COMPS: -0.709634
- BLiMP: -0.387923
- SuperGLUE: -0.024849
- AoA: 0.0

Favorable/offsetting interactions:
- GlobalPIQA: 1.514563
- Reading: 1.178445
- Supplement: 0.006589
- AoA: 0.0

EWoK is the main treatment-specific fragility. SuperGLUE interaction is near zero, Supplement is essentially stable, while Reading and GlobalPIQA interactions are favorable at seed43122.

## EWoK domain localization
Most negative official-domain interactions:
- material-dynamics: DiD=-17.532, TE43022=9.870, TE43122=-7.662, n=770
- physical-dynamics: DiD=-10.833, TE43022=7.500, TE43122=-3.333, n=120
- spatial-relations: DiD=-8.367, TE43022=2.653, TE43122=-5.714, n=490
- physical-interactions: DiD=-5.935, TE43022=4.317, TE43122=-1.619, n=556
- social-relations: DiD=-2.132, TE43022=0.065, TE43122=-2.067, n=1548
- quantitative-properties: DiD=-0.955, TE43022=0.637, TE43122=-0.318, n=314

Most positive official-domain interactions:
- material-properties: DiD=8.235, TE43022=7.059, TE43122=15.294, n=170
- social-interactions: DiD=6.463, TE43022=-5.782, TE43122=0.680, n=294
- social-properties: DiD=2.744, TE43022=3.963, TE43122=6.707, n=328
- agent-properties: DiD=1.719, TE43022=-0.995, TE43122=0.724, n=2210

## Route implication
Protect the seed43022 official-coordinate endpoint for submission in parallel, because it is a real above-leader coordinate with completed compliance evidence. Continue research on stabilizing compact-view relation formation, especially material/physical/spatial EWoK behavior and Entity/COMPS side effects. Do not treat adjacency-broken training as the next run: it tests pair-adjacency causality, not the observed seed interaction. A more useful next construction is a low-cost stability plan that separates (i) checkpoint/fine-tuning randomness, (ii) relation-domain sensitivity, and (iii) compact-view selection/placement properties before any new 100M training.

JSON: `experiments/archive/representation_and_objectives/data/relation_instability_synthesis/relation_instability_synthesis.json`
