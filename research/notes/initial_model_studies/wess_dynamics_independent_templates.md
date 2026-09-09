# wess dynamics independent templates — WESS S1-depth learning dynamics on independent templates

## Evidence

- Script: `scripts/wess_dynamics.py`
- Main result JSON: `data/wess_dynamics.json`
- Additional higher-episode seeds run inline in wess dynamics independent templates stdout: seeds 456 and 789 for the 50% episode setting.
- Preceding result: `notes/gated_residual_s1_depth_fix.md`

## Why this experiment was needed

The S1-depth learnable-residual WESS route showed a strong signal around 1000 steps in gated residual s1 depth fix, but a 2000-step continuation on the same controlled suite lost the signal. That meant the result could not be treated as a stable mechanism or used as a reason to launch a BabyLM run. The next question was the learning dynamics: does entity-indexed slot use persist under extended training and transfer to independently rendered templates, or is it a transient feature of one synthetic suite?

## wess dynamics independent templates design

The experiment changed the evidence standard in three ways:

1. **Independent held-out templates.** Training episodes use event forms such as `moved to`, `went into`, `walked to`, and `is now in`; held-out evaluation uses different forms such as `entered`, `stayed inside`, and `arrived at`, with a different query form. This prevents judging stability only on the training template family.
2. **Trajectory tracking.** Binding metrics are measured every 200 steps through 1500 steps, not only at the endpoint.
3. **Stabilization comparisons.** S1-depth 12×384 WESS is tested under 25% episode ratio baseline, encoder-freeze-after-750, and 50% episode ratio. Seeds 42 and 123 were run in the main script; seeds 456 and 789 were added for the 50% episode setting.

The held-out shortcut check remains strong: first-state, last-state, nearest-state, majority-visible, and previous-query-state predictors all have 0% pair accuracy on the independent-template pair score.

## Main findings

### 1. 25% episode ratio does not produce stable held-out binding

At 25% episode ratio, S1-depth WESS remains at essentially zero pair accuracy and near-zero log-odds on the independent-template suite for both seeds. The plain model also stays at zero. Freezing the encoder after earlier analysis does not rescue this.

This rules out the idea that the gated residual s1 depth fix instability is solved just by preserving the encoder or by watching for a lucky endpoint. At 25% episode ratio the binding gradient is too weak or too inconsistent for the 12-layer model to form durable slot-mediated behavior that transfers to independent phrasing.

### 2. 50% episode ratio can produce a persistent and transferable S1-depth signal

The 50% episode-ratio setting produces sustained held-out binding in several seeds:

| seed | held-out pair accuracy at final | held-out log-odds at final | learned residual scalar | slot swap delta | write-removal delta |
|---:|---:|---:|---:|---:|---:|
| 123 | 0.600 | +7.426 | -0.095 | +12.538 | +7.828 |
| 456 | 0.835 | +11.672 | -0.109 | +16.793 | +13.026 |
| 789 | 0.475 | +10.295 | +0.097 | +20.097 | +10.981 |

The trajectories are not single-step peaks. For seed 456, pair accuracy rises from 0.03 at earlier analysis to 0.54 at 800, 0.72 at 1000, 0.81 at 1200, and 0.83–0.84 at 1400–1500. Seed 123 similarly rises to ~0.60 by 1200–1500. Seed 789 is noisier but still reaches 0.47 by 1500 with large log-odds and intervention effects.

### 3. The effect is still optimization-sensitive

Seed 42 at the same 50% episode ratio remains weak: pair accuracy fluctuates around 0.03–0.12 and final pair accuracy is ~0.04. Thus 50% episode ratio is not a complete solution across all seeds, but it changes the dynamics from “signal washed out or absent” to “often persistent and transferable”.

The practical lesson is that WESS S1-depth learning has an early optimization threshold: if the residual scalar and slot pathway receive enough coherent binding gradient, the signal grows and persists; if not, it remains near zero or decays. The next BabyLM-facing design must make this threshold easy to cross reliably, not rely on selecting one peak.

## Scientific conclusion

The independent-template experiment tests persistence of the binding signal. The S1-depth WESS signal is not automatically stable, and the 1000-step gated residual s1 depth fix result alone was insufficient. However, persistent held-out binding is achievable under extended training on independent templates when the episode signal is strong enough. This means the remaining problem is not whether WESS can persist at S1 depth, but how to make the persistent regime seed-robust and compatible with official BabyLM data balance.

## Implications for the next work

Do not launch a full BabyLM 100M candidate yet. First make the stable regime reliable enough to justify official training.

The next best execution should test a small set of reliability improvements:

1. **Curriculum for episode ratio:** start with 50% episodes until held-out pair accuracy and intervention effects appear, then anneal toward 10–25% to reduce synthetic dominance and protect ordinary BabyLM scores.
2. **Separate learning rates:** use a larger learning rate for slot/residual parameters and smaller learning rate for the deep DeBERTa encoder, rather than freezing the encoder outright.
3. **Residual scalar warm start:** initialize the scalar with a small magnitude such as ±0.03 instead of exactly zero, or use two signed scalar channels so the model need not discover the useful sign from scratch.
4. **Periodic independent-template measurement:** train should save checkpoints and measure held-out template binding every few hundred steps; continuation should be based on persistence across independent templates, not the original training suite.
5. **Cross-seed short screen:** before full official training, require at least 2/3 or 3/4 seeds to maintain nonzero held-out pair accuracy and large slot interventions under the chosen schedule.

Once reliability is improved, integrate the same mechanism into the official-compatible BabyLM trainer with HF checkpoint saving and official Entity/EWoK/GlobalPIQA/BLiMP/Supplement/Reading evaluation. The protected internal best remains 40.5269 Overall; no WESS model has yet been added to the official 9-column scoreboard.
