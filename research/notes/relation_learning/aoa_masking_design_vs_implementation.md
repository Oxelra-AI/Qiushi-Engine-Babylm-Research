# AoA Masking: Design Versus Implementation

Scientific status: documented prospective design plus static reading of the original scientific masking implementation. No training or evaluation was rerun.

The 30M design used the same adapter-faithful 8-layer, width-480 model, adapter bottleneck 128, scale 1.75, and initialization seed43112. The command distinguished seed43, extra initialization seed43112, and training RNG seed43113. Control used the original ordering and ordinary 0.15 WWM. The schedule arm changed row ordering, with early CHILDES enrichment. Its stream contained 30,000,051 words versus the control's 30,000,000. The enrichment arm retained the original ordering and changed masking probabilities.

## Implemented Masking Schedule

The enrichment wrapper derives token weights only from corpus-internal enrichment z-scores. With alpha_start=1.5 and taper fraction 0.67, the taper length is int(0.67 * 2529)=1694 masking calls. Its counter increments before probabilities are calculated, so the first training call uses 1.5*(1-1/1694), not exactly 1.5; alpha reaches zero at call1694. The initial diagnostic probability calculation resets this counter before training.

For token t, the pre-clamp probability is

`p_t = 0.15 * exp(alpha*z_t) / Z`,

where `Z = sum_t(freq_t * exp(alpha*z_t)) / sum_t(freq_t)`.

The implementation then clamps probabilities to [0.02, 0.60]. A word group's selection probability is the mean of its eligible tokens' probabilities. A group is selected by a Bernoulli draw; if the entire batch has no selected token, the first eligible token is selected. Selected positions use 80/10/10 mask/random/unchanged corruption.

The design's exact 0.15 claim applies to the corpus-occurrence-weighted mean before clamping. Clipping, group averaging, batch composition, the fallback, and stochastic draws mean it is not an assertion that every realized batch or final selected-token fraction is exactly 0.15. The implementation records the selected-token fraction separately. No realized numeric rate is inferred here without its stored measurement. The counter is a masking-call count; its equality to optimizer updates depends on the calling loop, not its name.

## Prospective Credit Test

The proposed regression was `Delta surprisal[w,c] = a[c] + beta * Delta credit[w,c] + error`, with checkpoint intercepts. Schedule credit was 0.15 times the cumulative occurrence difference; enrichment credit used the time-varying enriched credit minus control credit. Agreement of beta across arms was proposed as evidence for a common credited-exposure account; disagreement would distinguish timing from the masking lever.

The decision gates applied to the recalibrated prediction after beta was measured from both 30M arms. Predicted r > 0.15 supported a proposed two-seed 100M test with AoA as primary and all eight other endpoint columns within the specified coherent bands; 0.112 < r < 0.15 allowed at most one exploratory seed; both arms below 0.112 argued against continuing that route. The original design did not quantify those bands here. No later 100M run is asserted by this note. Child AoA ages were evaluation labels, not training inputs; age-oracle schedules were diagnostic upper bounds, not admissible training recipes. See the [recorded calibration design and decision gates](aoa_calibration_design_7d6b3d69.md).

Source implementation: [enrichment masking wrapper](../../../experiments/archive/relation_learning/scripts/enrichment_masking_wrapper.py). Related recorded outcomes: [AoA estimands and limits](figure_estimands_and_remaining_limits.md).
