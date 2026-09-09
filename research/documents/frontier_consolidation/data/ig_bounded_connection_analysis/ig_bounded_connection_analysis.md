# earlier analysis bounded IG connection analysis

This note answers only the two allowed questions: matched compact-view/reinvestment connection and relation/state late-loss connection. It intentionally avoids arbitrary token mining.

## Matched clean80 → reinvest80 controls
### Common/full-pool sample
CSV: `experiments/archive/frontier_consolidation/data/ig_matched_clean80_reinvest80_common_sample/context_dependence_items.csv`; paired items `4096`
- r4: all ΔNLL(reinvest-clean)=-0.092847; lowIG+highErr Δ=-0.542597 (n=274); highIGtop20 Δ=-0.018082; partial corr(cleanIG, Δ | logfreq)=0.089865
- r8: all ΔNLL(reinvest-clean)=-0.092847; lowIG+highErr Δ=-0.530593 (n=294); highIGtop20 Δ=-0.055351; partial corr(cleanIG, Δ | logfreq)=0.091423
- r16: all ΔNLL(reinvest-clean)=-0.092847; lowIG+highErr Δ=-0.620948 (n=273); highIGtop20 Δ=-0.137481; partial corr(cleanIG, Δ | logfreq)=0.068666

### Changed compact-view block
CSV: `experiments/archive/frontier_consolidation/data/ig_matched_clean80_reinvest80_changed_block/context_dependence_items.csv`; paired items `12024`
- r4: all ΔNLL(reinvest-clean)=-0.577444; lowIG+highErr Δ=-2.143852 (n=1062); highIGtop20 Δ=-0.158085; partial corr(cleanIG, Δ | logfreq)=0.298327
- r8: all ΔNLL(reinvest-clean)=-0.577444; lowIG+highErr Δ=-2.173784 (n=1165); highIGtop20 Δ=-0.178830; partial corr(cleanIG, Δ | logfreq)=0.300550
- r16: all ΔNLL(reinvest-clean)=-0.577444; lowIG+highErr Δ=-2.148121 (n=1210); highIGtop20 Δ=-0.173579; partial corr(cleanIG, Δ | logfreq)=0.277338

## chck82 → chck100 corpus-likelihood relation
CSV: `experiments/archive/frontier_consolidation/data/context_dependence_ig_probe_chck82_chck100_256rows/context_dependence_items.csv`; paired items `1024`
- r4: all mean ΔNLL(100-82)=-0.030728; lowIG+highErr Δ=-0.092427 (n=70); not-stratum Δ=-0.026201; partial corr(IG, Δ | logfreq)=0.067024
- r8: all mean ΔNLL(100-82)=-0.030728; lowIG+highErr Δ=-0.080905 (n=74); not-stratum Δ=-0.026819; partial corr(IG, Δ | logfreq)=0.081484
- r16: all mean ΔNLL(100-82)=-0.030728; lowIG+highErr Δ=-0.064699 (n=74); not-stratum Δ=-0.028082; partial corr(IG, Δ | logfreq)=0.107427

## Official relation/state late loss
- EWoK: payload Δ=-0.975360, common=7618, flips={'both_correct': 3513, 'both_wrong': 3545, 'gain': 269, 'loss': 291}, net=-22
- Entity: payload Δ=-0.849493, common=6780, flips={'both_wrong': 4748, 'both_correct': 1713, 'loss': 178, 'gain': 141}, net=-37

## Direct scientific reading
- Reinvestment changes corpus likelihood most strongly in the changed compact-view block and disproportionately helps the low-IG/high-error stratum there; in the random/common sample the effect is small. This is a local data-mechanism likelihood signature, not the original high-IG target-allocation story.
- On the context dependence ig probe result chck82→chck100 sample, the low-IG/high-error stratum improves in NLL rather than worsening, while official EWoK/Entity scores decline. The likelihood residual therefore does not track relation/state loss in the only shared evidence currently available.
- The late-loss JSON summarizes official EWoK/Entity item flips but has no legal-corpus token identity; direct tracking would require a separate official-item forward probe. Given the negative corpus-likelihood relation, likelihood-based mining should stop unless such a probe is explicitly needed by a new mechanism.

JSON: `experiments/archive/frontier_consolidation/data/ig_bounded_connection_analysis/ig_bounded_connection_analysis.json`
