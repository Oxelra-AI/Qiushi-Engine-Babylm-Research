# semantic repair frontier and refill plan — current-official EWoK 2×2 domain treatment interaction

This uses identical 7,618-row official EWoK data and saved predictions for clean-Qwen and compact_view_reinvest at seeds 43022 and 43122.

## Official macro-domain EWoK
- clean_43022: 50.6631
- clean_43122: 51.4803
- reinvest_43022: 53.5366
- reinvest_43122: 51.8917
- clean seed gap 43122-43022: +0.8173
- reinvest seed gap 43122-43022: -1.6449
- TE43022 reinvest-clean: +2.8735
- TE43122 reinvest-clean: +0.4114
- EWoK DiD (TE43122-TE43022): -2.4621

Domain bootstrap for DiD: observed -2.4621, p05 -6.2343, p50 -2.4011, p95 +1.0493, prob_negative 0.871.

## Domain-level DiD (most negative first)
- material-dynamics: DiD -17.53; TE43022 +9.87, TE43122 -7.66; clean gap +5.97, reinvest gap -11.56; n=770
- physical-dynamics: DiD -10.83; TE43022 +7.50, TE43122 -3.33; clean gap +10.00, reinvest gap -0.83; n=120
- spatial-relations: DiD -8.37; TE43022 +2.65, TE43122 -5.71; clean gap +4.08, reinvest gap -4.29; n=490
- physical-interactions: DiD -5.94; TE43022 +4.32, TE43122 -1.62; clean gap +2.70, reinvest gap -3.24; n=556
- social-relations: DiD -2.13; TE43022 +0.06, TE43122 -2.07; clean gap +2.26, reinvest gap +0.13; n=1548
- quantitative-properties: DiD -0.96; TE43022 +0.64, TE43122 -0.32; clean gap +2.87, reinvest gap +1.91; n=314
- physical-relations: DiD -0.49; TE43022 +2.32, TE43122 +1.83; clean gap -0.86, reinvest gap -1.34; n=818
- agent-properties: DiD +1.72; TE43022 -1.00, TE43122 +0.72; clean gap +0.05, reinvest gap +1.76; n=2210
- social-properties: DiD +2.74; TE43022 +3.96, TE43122 +6.71; clean gap -3.66, reinvest gap -0.91; n=328
- social-interactions: DiD +6.46; TE43022 -5.78, TE43122 +0.68; clean gap -4.42, reinvest gap +2.04; n=294
- material-properties: DiD +8.24; TE43022 +7.06, TE43122 +15.29; clean gap -10.00, reinvest gap -1.76; n=170

## ContextDiff DiD highlights
- variable_swap: DiD -30.00; TE43022 +16.67, TE43122 -13.33; n=30
- material: DiD -15.83; TE43022 +9.64, TE43122 -6.19; n=840
- variable swap: DiD -2.57; TE43022 +0.94, TE43122 -1.64; n=2564
- game: DiD +0.00; TE43022 +0.00, TE43122 +0.00; n=20
- other: DiD +0.41; TE43022 +1.02, TE43122 +1.43; n=490
- antonym: DiD +0.41; TE43022 +0.18, TE43122 +0.59; n=3374
- negation: DiD +1.05; TE43022 +4.74, TE43122 +5.79; n=190
- number: DiD +6.25; TE43022 +0.00, TE43122 +6.25; n=80

## Scientific read
- EWoK benefit survives at seed43122 in current official coordinates, but shrinks strongly: seed43122 gains only about +0.41 versus clean, while seed43022 gains +2.88.
- The treatment-specific EWoK weakness is concentrated in domain interactions, especially where the clean seed43122 baseline improved but the reinvest seed43122 did not. This supports a stability problem in how reinvested compact views interact with relation learning, not a total failure of the EWoK mechanism.
- This remains an EWoK-only result; do not use it to infer seed43122 Overall or choose training until full official vector and exact clean sparse temporal control are available.

Machine-readable output: `experiments/archive/frontier_consolidation/data/official_ewok_2x2_domain_did/official_ewok_2x2_domain_did.json`
