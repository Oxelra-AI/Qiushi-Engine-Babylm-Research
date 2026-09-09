# Minimal SGCR repairs and launch conditions

This note targets the corrected entity gate stats sources audited at the hashes recorded in
`runtime_probe.json`. It is a repair design, not an edit to Qiushi's live files.
The executable reference implementation and CPU tests are
`candidate_sgcr_module.py` and `test_candidate_sgcr.py`.

## 1. Replace decode/re-encode decomposition with exact BPE-prefix splitting

This is the primary blocker. The legal16k tokenizer loaded from `tokenizer.json`
has fixed padding to 256. More fundamentally, isolated byte-level decoding loses
context and sometimes bytes. The legal16k merge list is an exact prefix of the
legal40k merge list, and all 16,384 vocabulary ids agree, so every legal40k token
can be split exactly by recursively undoing legal40k merges until all component
ids are below 16,384.

Use `build_prefix_decomposition_map()` from `candidate_sgcr_module.py` in place
of `build_decomposition_map()`. Assert at launch that the merge-prefix and vocab
prefix properties hold, that all 40,000 mappings are nonempty, that every
component id is in `[0, 16384)`, and that the maximum component length is 7 for
the pinned tokenizers. Do not merely call `no_padding()`: unpadded isolated
decode/re-encode still disagrees with the exact map for 10,910/40,000 tokens.

## 2. Keep an exact-preserving but one-factor-live initialization

Never initialize both `component_embeddings.weight` and
`component_proj.weight` to zero. The current live repair (random component codes,
zero projection and bias) is acceptable: projection-weight gradient is live on
babylm2026 live surface and component-embedding gradient becomes live on leader analysis and route pivot. The alternative
in `candidate_sgcr_module.py` (zero component codes, nonzero orthogonal
projection, zero bias) makes the component table live on babylm2026 live surface. Either is exact
at initialization; document which factor is expected to have zero gradient on
the first step.

After constructing SGCR, reset the training RNG once more to the matched
`train_rng_seed`. This prevents the random semi-cold initialization from
advancing later dropout randomness, especially for CPU/reproducibility tests.

## 3. Retain the now-restored matched optimizer semantics

The final audited trainer now includes the legal40k accum training completion operation immediately before
`optim.step()`:

```python
torch.nn.utils.clip_grad_norm_(unique_params, 1.0)
optim.step()
```

Keep this in the launch revision and retain
`curriculum_state.get_current_mask_mode()`; the former `current_mode_label` field
does not exist.

## 4. Calibrate the uniform-gate control by training mass

The current type mean is `rho=0.4558668`, hence a residual multiplier of
`0.5441332`. The routed K=50 treatment's token-mass-weighted mean is
`rho=0.9357199`, residual `0.0642801`. Thus the current uniform control is 8.47x
stronger in first-moment residual exposure and is not an optimization-matched
control. Compute the constant over ordinary used tokens as:

```python
mass = counts[ordinary_used].float()
mean_rho = (rho[ordinary_used] * mass).sum() / mass.sum()
rho[~force_standard_mask] = mean_rho
rho[force_standard_mask] = 1.0
```

Keep all special/control ids at `rho=1` in treatment and controls.

## 5. Make checkpoint roles explicit

The clean baked model path now passes standard Hugging Face loading and exact
logit parity. Preserve the `_sgcr*` key filter. Add `try/finally` around baking
so the live base table is restored after any save error, and run the base
trainer's portable-tokenizer config repair after tokenizer saving.

`sgcr_components.pt` is not currently sufficient for training resumption because
it omits the pre-bake base word table; loading the baked `W_eff` and adding the
saved residual double-applies SGCR. At minimum store `base_word_embeddings`.
For real preemption recovery also store/load optimizer, scheduler, step/word
counters, and RNG states. Until this exists, label the file diagnostic-only and
launch only if an uninterrupted run is operationally acceptable.

## 6. Controls

There is no random-decomposition CLI path in the current trainer. If added, use
a fixed-seed, special-token-excluding permutation of whole exact decomposition
rows within exact length strata. This preserves each token's component count and
the global component/co-occurrence multiset much better than drawing arbitrary
component ids. Record the permutation seed and map hash.

Uniform-real and support-gated-random are two one-axis controls, not a complete
2x2 factorial; they cannot identify gate-by-composition interaction without the
uniform-random cell. If only one full control slot is available, run the
mass-calibrated uniform gate with the real exact decomposition first, because
the central claim is support-dependent routing rather than merely extra shared
embedding capacity.
