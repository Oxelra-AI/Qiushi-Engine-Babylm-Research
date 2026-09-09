# memory stability and next route — Memory stability reading and next route

## Why this step was needed

Entity ordering reversed between standalone 1M runs and the `chck_1M` point inside the 10M trajectory. This means the current memory-vs-dense comparison mixes at least three effects:

1. initialization / SGD / data-order variation,
2. different training-text pools,
3. the actual causal prefix-memory mechanism.

The trainer code clarifies the second effect. `iter_examples(files, max_words, words_per_example)` truncates the corpus before shuffling. Therefore:

- standalone `--max_word_exposure 1000000` uses the first 1M corpus words, then shuffles those examples;
- a 10M trajectory uses all 10M words, shuffles the full example pool, and `chck_1M` is the first ~1M words from that full shuffled pool.

So standalone 1M and 10M-trajectory `chck_1M` are not the same data condition.

## Evidence after the minimal repeat

memory stability and next route ran a second standalone prefix-1M seed (`seed=43`) for the close dense-untied vs memory pair.

Evidence JSON: `experiments/archive/initial_model_studies/data/profile_memory_control_seed43_1m.json`

| setting | model | BLiMP | Supplement | EWoK | Entity | COMPS | Reading eye | Reading SPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| prefix-1M seed42 | dense untied | 53.82 | 48.80 | 46.27 | 16.12 | 49.99 | 9.07 | 2.40 |
| prefix-1M seed42 | memory M64 | 55.17 | 48.00 | 51.91 | 17.86 | 50.05 | 8.95 | 2.32 |
| prefix-1M seed43 | dense untied | 54.81 | 52.00 | 49.45 | 15.42 | 49.34 | 8.47 | 2.62 |
| prefix-1M seed43 | memory M64 | 54.18 | 52.40 | 48.45 | 17.43 | 49.73 | 9.24 | 2.70 |

Deltas memory minus dense:

| setting | BLiMP | Supplement | EWoK | Entity | COMPS | Reading eye | Reading SPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefix-1M seed42 | +1.35 | -0.80 | +5.64 | +1.74 | +0.06 | -0.12 | -0.08 |
| prefix-1M seed43 | -0.63 | +0.40 | -1.00 | +2.01 | +0.39 | +0.77 | +0.08 |

## What looks stable now

Within standalone prefix-1M repeats, the memory model improves Entity in both seeds (+1.74, +2.01) and slightly improves COMPS in both seeds. This is the strongest currently repeated signal.

The 1M EWoK and Reading effects are not stable across these two standalone seeds. EWoK is strongly positive in seed42 but negative in seed43. Reading is slightly negative in seed42 and positive in seed43.

The 10M trajectory still matters because it reflects a fuller data mixture. At `chck_10M`, memory beats its dense-untied control on BLiMP (54.31 vs 53.49), EWoK (51.91 vs 50.73), Reading eye (12.16 vs 10.14), and Reading SPR (2.91 vs 1.84), but loses Supplement (56.80 vs 59.60), Entity (16.99 vs 17.79), and COMPS (49.58 vs 50.00). Because this has only one seed and a different data pool, it should be treated as promising trajectory evidence, not a settled mechanism effect.

## Route judgment

Do not choose entity-slot memory, curriculum, or auxiliary objective yet as if the mechanism question were settled. The immediate research problem is to separate data-pool effects from real architecture effects.

At the same time, do not discard memory. The repeated prefix-1M Entity gain means the memory route has a real enough signal to preserve. The 10M EWoK/Reading advantage means any next mechanism should keep or strengthen this broader semantic/reading behavior rather than chasing Entity alone.

## Next work

The next execution should add a data-pool control to the trainer: allow the training pool size and the training exposure length to differ. This can be done by adding an option such as `--selection_word_pool` or `--example_pool_words` that builds and shuffles a 10M pool, then trains only the first requested 1M exposure. With that, run a minimal 2x2 comparison:

1. dense-untied vs memory, seed42, prefix-1M pool (already available),
2. dense-untied vs memory, seed43, prefix-1M pool (now available),
3. dense-untied vs memory, seed42, full-10M pool but train only 1M,
4. dense-untied vs memory, seed43, full-10M pool but train only 1M.

This will show whether the Entity flip is caused mostly by different text mixture/order or by unstable model behavior.

If full-pool 1M repeats show memory preserves Reading/EWoK but not Entity, redesign memory toward sharper entity/state slots while keeping the prefix-memory signal as a reading/semantic stabilizer. If full-pool repeats recover Entity too, scale a refined memory variant to 10M and compare against dense-untied and dense6x384 controls. If neither Entity nor Reading/EWoK survives repeats, return to developmental data order or low-weight objective changes rather than scaling the current memory model.

## Additional interpretation constraints

An independent scientific check confirmed the pool/seed confound and added design constraints that must be satisfied for the 2x2 pool experiment to be interpretable:

1. **Only the repeated Entity gain is currently preservable.** Across both standalone prefix-1M seeds, memory improves Entity (+1.74, +2.01) and marginally COMPS (+0.06, +0.39). No EWoK, Reading, BLiMP, or Supplement advantage is stable at 1M. The 10M EWoK/Reading/BLiMP advantages are single-seed, single-trajectory, and the checkpoints are correlated, so they are "worth further verification" not "established." Do not claim them yet.

2. **The Entity signal is still not established as a mechanism.** Entity absolute scores are ~15-18 with no chance/untrained floor, no per-item output, and no confidence interval recorded. In the 10M-pool trajectory chck_1M, memory Entity is actually lower than dense. So the standalone repeat is necessary but not sufficient.

3. **LR-schedule confound.** Standalone 1M runs and 10M-target runs use cosine schedules over different total steps (98 vs 977), so at equal words-seen they are at different warmup/decay phases. The 2x2 pool experiment must use the same fixed LR schedule for all four 1M-exposure runs; otherwise pool and schedule are entangled. This is why the 10M-trajectory chck_1M should not be treated as a clean 1M data point.

4. **Pairing must be exact, not just same seed.** The memory module consumes RNG, so identical `--seed` does not guarantee identical shared GPT-2 core initialization or dropout streams versus dense. For a clean architecture contrast, the shared core should be copied from one saved initialization, memory params should use a separate RNG stream, and both architectures must consume identical example-ID order, packing, optimizer steps, and stop boundary.

5. **Capacity vs persistence control.** Memory is 12.339M params vs dense-untied 11.614M (~6% more). Before attributing any survived gain to state persistence, add a cheap ~12.34M non-persistent control: same extra projection/parameters but with the cross-token memory read/write disabled (e.g., memory replaced by a per-token transform with no cumulative prefix state). If that also reproduces the Entity gain, the effect is capacity/optimization, not memory.

### Concrete next Execute design

- Add trainer options to decouple candidate pool from exposure: `--example_pool_words` (build+shuffle this many words) and keep `--max_word_exposure` as the consumed length; fix one LR schedule for all four runs (e.g., schedule as if training the pool, or a fixed constant+warmup, recorded explicitly).
- Save one shared init for the GPT-2 core; load it into both dense and memory; give memory params an independent RNG stream.
- Run the 2x2x2: {dense-untied, memory-M64} x {1M-pool, 10M-pool} x {seed42, seed43}, all at 1M exposure, identical example-ID order within each pool/seed cell.
- Add the non-persistent ~12.34M capacity control at least for the cell where memory looks best.
- Record per-item eval outputs and a chance/untrained Entity floor so small deltas are interpretable.
- Statistic of interest: paired delta per cell, and pool x architecture interaction per seed.
