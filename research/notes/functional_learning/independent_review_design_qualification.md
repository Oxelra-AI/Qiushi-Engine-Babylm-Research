# clean d component ablation independent_review qualification — what the coherent86 continuation runs establish

A independent_review verifier read the clean d component ablation plan and trainer. The useful correction is that clean d component ablation is a matched adaptation experiment from the exact coherent86 alpha0.75 tensors, not a perfectly seamless continuation of the original earlier analysis optimizer state.

## What the running pair cleanly controls

The standard and carrier-residual arms share the exact parent, continuation rows, WWM mask generator, dropout stream, trainable private-adapter set, executed private scale, neutral KL form, and cumulative schedule offset. The deterministic carrier scoring repair carries over from causal interface trajectory: private adapters are disabled, eval mode is used, and RNG is isolated. Thus the within-clean d component ablation standard-vs-carrier comparison is much cleaner than causal intervention's stochastic carrier-residual comparison.

## Main qualifications for interpretation

1. **Optimizer restart.** The parent tensors load but Adam moments and exact optimizer/RNG state from earlier analysis are absent. Damage in the standard arm can reflect restart dynamics, not only harmful data or overtraining.
2. **Gradient accumulation objective.** Like causal intervention/026, the trainer averages per-microbatch losses rather than exactly normalizing over all macro-batch targets. This keeps the real-pretraining loop feasible under memory pressure and pairs the two arms, but it is not exactly the original batch-256 earlier analysis update.
3. **Training scale.** earlier analysis learned the private tensors at scale 1.0 and alpha0.75 was post-hoc materialized; clean d component ablation trains with executed scale 0.75. A score gain could partly be scale recalibration, so alpha sweeps and adapter-direction measurements matter.
4. **Neutral KL anchor.** The KL compares current private-on to private-off carrier behavior, not to the parent private-on function. It can pull against the inherited coherent86 correction. If scores fall while KL-to-carrier shrinks or the adapter moves toward private-off behavior, the loss is not evidence that the remaining text lacks useful information.
5. **Checkpoint ladder interpretation.** Intermediate checkpoints mix exposure and annealing state. A mid-run score peak is practical evidence for selecting a candidate, but not by itself a pure word-budget law.

## Runtime validation already performed

`scripts/parent_validation.py` and `data/parent_validation/parent_validation.json` validated the exact parent SHA (`e14d757...`), no missing private-adapter keys, no unexpected keys, and executed private_adapter.scale = 0.75 in all layers. This removes a hidden parent-loading failure mode.

## Measurements needed once checkpoints exist

The most useful evaluation sequence is:

1. Common cheap7 screen for parent and checkpoint ladder, especially early 1M/2M/4M increments and final 100M.
2. Alpha sweep if continuation changes amplitude: at least 0.5, 0.75, and 1.0 for any promising or damaged endpoint.
3. Fixed-probe CE/KL/drift analysis with `scripts/model_drift_and_lm_probe.py`: parent-on CE, current-on CE, current-off CE, KL(parent-on||current), KL(carrier-off||current), and private-adapter displacement/cosine by layer.
4. If either run looks promising, repeat with a better accumulation implementation or a true batch-256 command if memory permits; otherwise do not promote the result as a general learning principle.

## How outcomes should be read

- Standard continuation improves broadly and fixed-probe drift is modest: the inherited private path remains practically plastic and the remaining legal data can add useful competence under this procedure.
- Standard continuation loses score while moving toward private-off/carrier behavior: the carrier KL is erasing useful inherited correction.
- Carrier-residual again gains EWoK/Entity/COMPS but loses Supplement/GlobalPIQA: the causal interface trajectory tradeoff is stable across starting states; raw uncertainty weighting is redistribution rather than a relation-resolvable principle.
- Carrier-residual beats standard at matched or lower drift and avoids the previous tradeoff: raw carrier uncertainty becomes conditionally useful only in the retained-state regime, requiring additional controls before theoretical elevation.
