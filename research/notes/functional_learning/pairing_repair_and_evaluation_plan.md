# causal interface trajectory pairing repair for carrier-residual credit

## Why this repair matters

The first causal intervention weighted replay was useful as a candidate model, but not a clean paired comparison. `compute_carrier_weights` disabled the private adapter and then ran the frozen carrier forward while the model was still in training mode. Because the inherited DeBERTa has dropout 0.1, this made the carrier probabilities and hence the token weights stochastic. It also consumed dropout RNG before the weighted arm's actual student forward, so a shared seed did not imply matched student-forward dropout between the standard and weighted arms.

This does not invalidate the trained causal intervention models as candidates, but it prevents interpreting the small standard-vs-carrier_residual difference as a pure effect of deterministic carrier-error weighting.

## Repair implemented

New wrapper: `experiments/archive/functional_learning/scripts/context_credit_paired_trainer.py`.

It reuses `context_credit_trainer.py` but monkey-patches `compute_carrier_weights` so the carrier-scoring forward:

- temporarily switches the model to `eval()`;
- disables the private adapter for the carrier distribution;
- uses `torch.random.fork_rng` with a fixed teacher seed, restoring CPU/CUDA RNG states on exit;
- restores the previous private-adapter enabled flags and train/eval state before the student main forward.

This removes dropout-stochastic carrier weights and prevents the teacher forward from advancing the student dropout RNG stream.

## Runs already produced

An initial wrapper smoke command was launched before the wrapper propagated the `--smoke` flag into the imported causal intervention trainer. Therefore the following directories are full 82M→86M corrected runs, despite their names:

- `training/runs/smoke_standard/` — 101 updates, 3,992,800 tail words, total 86,005,295 words. This is bit-identical to the causal intervention standard run because standard training does not invoke carrier scoring. SHA256 of alpha0.75 final model: `f3f46e30f5a7af5c95e9af394d15b47e1ea8ec9d16e385471da7f52b8e5b2059`.
- `training/runs/smoke_carrier_residual/` — 101 updates, 3,992,800 tail words, total 86,005,295 words, deterministic carrier scoring. SHA256 of alpha0.75 final model: `e3b722fab62f7a3af41b828f0e3eb3ed926cb59664a22d4fb7da17da14e5db38`.

For comparison, the original stochastic causal intervention carrier_residual alpha0.75 final SHA256 is `6a98c76354e59d6db01869aa9a9a127ebed0f1d9f4376783688c02e620a84a48`.

## Training differences seen before evaluation

At full tail:

- corrected deterministic carrier_residual final neutral KL: `0.009448`, larger than original stochastic causal intervention carrier_residual `0.007145` and standard `0.002551`.
- corrected deterministic carrier_residual final private RMS by layer: `[0.0297, 0.0274, 0.0275, 0.0285, 0.0320, 0.0331, 0.0481, 0.0583]`, again strongest in deep layers.
- standard final private RMS: `[0.0243, 0.0226, 0.0233, 0.0224, 0.0230, 0.0220, 0.0229, 0.0204]`.

Thus deterministic carrier-error credit produces an even larger private correction than the original stochastic run. Evaluation must determine whether this is better learning or merely excess residual magnitude.

## Evaluation needed

A common screen script was written at `scripts/eval_common_screen.py`. It fixes Reading by passing `--data_path evaluation_data/fast_eval/reading/reading_data.csv`, and for alpha comparisons it changes executed private-adapter scales by materializing a checkpoint with every layer's `private_adapter.scale` changed, not merely by editing config.

Current evaluation priorities:

1. Put exact `chck_82M`, exact `coherent86 alpha0.75`, causal intervention standard, causal intervention stochastic carrier_residual, and causal interface trajectory deterministic carrier_residual on the same cheap7 screen.
2. Complete Reading for the existing causal intervention arms; the earlier causal intervention `equal7` values were 6-column means and are not comparable.
3. Use small alpha sweeps only after common-baseline comparison, asking whether weighted training learned a better correction or only a larger correction.
4. If a candidate survives the exact common screen, run the complete official-compatible evaluation stack rather than continuing local tuning indefinitely.
