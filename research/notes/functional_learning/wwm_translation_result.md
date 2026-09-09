# wwm translation result: WWM Translation Diagnostic Result

## Summary

The allocation balanced interpretation allocation mechanism **does not survive translation** from concentrated
answer-only masking to standard 15% whole-word masking (WWM). All arms show
**negative** parent-relative common-target source-following g under WWM, meaning
concentrated WWM on 17-24 rows for 80 epochs pushes the parent *away* from its
inherited behavior rather than building on it.

## Numerical Comparison (3-seed means)

| Arm | Objective | base_g | aux_g | base_compact_NLL | aux_compact_NLL |
|-----|-----------|--------|-------|-------------------|-----------------|
| compact_aux_support | answer-only (allocation balanced interpretation, 6 seeds) | +0.143 | +0.128 | -0.540 | -0.610 |
| compact_aux_support | **WWM (wwm translation result, 3 seeds)** | **-0.040** | **-0.090** | **-0.149** | **-0.099** |
| current_aux_substitution | answer-only | +0.020 | +0.082 | -0.157 | -0.609 |
| current_aux_substitution | **WWM** | **-0.002** | **-0.039** | **-0.049** | **-0.095** |
| compact_base80_unspent | answer-only | +0.130 | +0.060 | -0.550 | -0.118 |
| compact_base80_unspent | **WWM** | **-0.054** | **-0.095** | **-0.140** | **-0.009** |

## What This Means

1. **The allocation balanced interpretation positive allocation signal was specific to concentrated content-word masking.**
   Under answer-only, the adapter receives all gradient from exactly the content words
   it must predict in the common-target bank. Under WWM, gradient is distributed across
   ALL positions (source + view), so the adapter's 995K parameters absorb noise from
   function words, source tokens, and padding rather than building correspondence-aligned
   representations.

2. **Surface learning is ~4x weaker under WWM.** compact_aux NLL delta goes from
   -0.540 (answer-only) to -0.149 (WWM). This is expected: WWM at 15% masks ~10%
   of tokens per epoch randomly, while answer-only masks ~35% of content-word groups
   (all in the view portion) per epoch.

3. **The relative ordering collapses.** Under answer-only, compact arms >> current
   arms on base_g. Under WWM, all arms are negative. current_aux is closest to zero
   because it minimally changes the inherited training distribution.

## What This Does NOT Show

The diagnostic tested **concentrated private-adapter training on 17-24 rows** under WWM.
This is NOT the same as **full-model legal-stream training on 14M words** with ~5K-10K
compact rows substituted. The mechanisms are different:

- Concentrated: 80 epochs on 17 rows → model overfits to these specific rows
- Legal-stream: 1 pass through 14M words → compact rows are a small perturbation in
  a diverse stream; the effect (if any) comes from accumulated text-distribution change
  across thousands of rows, not from concentrated credit on a few

The bridge experiment (related experiments) showed that relation rows under ordinary WWM in the
legal stream also failed. But that tested a DIFFERENT intervention (relation answer-credit).
Compact shortening changes the text surface while preserving correspondence; it has not
been tested in the actual legal-stream setting.

## Decision

The concentrated WWM diagnostic does not justify the legal-stream candidate by itself,
but it also does not definitively close the route because it tested the wrong setting.
The decisive test is the actual legal-stream experiment: replace ~5K-10K current rewrites
with compact versions in the 14M-word tail, train with standard WWM from coherent86, and
evaluate on Cheap7 and official coordinates.

However, the diagnostic lowers confidence in the compact-alone route. The practical v5
candidate should combine compact shortening with other principle-guided improvements
(correspondence repair, optimized data composition) rather than relying on compact
shortening alone.
