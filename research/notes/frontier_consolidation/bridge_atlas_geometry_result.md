# bridge atlas geometry result: Bridge atlas-compatible geometry — decisive route result

## Scientific question
Are source-attested bridge candidates geometrically distinct from the failed extractive arms,
or do they occupy the same geometric region despite surface restructuring?

## Method
Replicated the extractive surface and readout repair atlas methodology (monotone alignment over ALL normalized words,
gap1/skip between consecutive aligned positions in view order) on all 103 retained mechanism evidence synthesis
bridge candidates, split by bridge route recovered from sourcecopy error structural edit class (48 extraction-like, 55 transformation-
like, 8 substantive-only). Compared to atlas pooled benchmarks AND to matched natural compact
references on the same 103 pairs.

## Decisive finding: transformation-like bridge ≈ natural compact on structural geometry

### Matched comparison (same 103 pairs, same measurement)

| group | n | mean gap1 | mean skip | mean span | mean absent | mean compress |
|---|---:|---:|---:|---:|---:|---:|
| bridge transformation-like | 55 | **0.764** | **0.236** | 0.872 | 0.039 | 0.682 |
| matched natural compact    | 55 | **0.762** | **0.238** | 0.816 | 0.168 | 0.616 |
| bridge extraction-like     | 48 | 0.918 | 0.082 | 0.930 | 0.003 | 0.828 |
| matched natural compact (extract pairs) | 48 | 0.858 | 0.142 | 0.788 | 0.136 | 0.705 |

The transformation-like bridge has gap1/skip **within 0.002** of matched natural compact.
Extraction-like bridge is much more contiguous (gap1 0.918 vs compact 0.858).

### Atlas-compatible pooled comparison

| variant | pooled gap1 | pooled skip |
|---|---:|---:|
| compact (atlas, 3005 pairs) | 0.778 | 0.223 |
| bridge transformation-like (103 pairs) | **0.795** | **0.205** |
| extractive_wide (atlas, 3005 pairs) | 0.647 | 0.354 |
| extractive_balanced (atlas, 3005 pairs) | 0.528 | 0.472 |
| bridge extraction-like (103 pairs) | 0.936 | 0.064 |

Bridge transformation-like is closest to compact (distance 0.017 on pooled gap1).
Bridge extraction-like is far above all atlas variants — pure contiguous source substrings.

### What differs between bridge transform and natural compact

The ONLY major geometric differences:
1. **Source-absent content**: bridge 0.039 vs compact 0.168 (Δ = -0.129)
2. **Compression**: bridge 0.682 vs compact 0.616 (bridge ~7% less compressed)
3. **Aligned fraction**: bridge 0.829 vs compact 0.680 (bridge retains more source words)

Gap1/skip, the main structural-contiguity metric, is indistinguishable.

## Scientific interpretation

1. **Transformation-like bridge is geometrically distinct from extractive arms.**
   Distance to ext_balanced on pooled gap1 = 0.267; distance to compact = 0.017.
   Source-attested structural transformation produces compact-like data geometry,
   not polished extraction geometry.

2. **The bridge isolates novel vocabulary as the remaining difference.**
   A matched training run comparing transformation-like bridge to natural compact
   would cleanly test whether source-absent content (novel MLM prediction targets)
   is necessary for the compact learning benefit, while controlling for structural
   geometry (gap1/skip/span are essentially matched).

3. **Extraction-like bridge candidates are NOT useful for this test.**
   Their geometry (pooled gap1 0.936) is far from both compact and extractive —
   they are essentially contiguous source substrings, a different data treatment
   from both the extractive arms (which spread across the source) and compact
   (which restructure while introducing novel content).

## Scale question (unresolved)

From 250 sampled pairs, 55/250 = 22% yielded transformation-like bridge outputs.
From the full pool (12,155 pairs), this predicts ~2,674 transformation-like pairs,
potentially enough for most of the 3,005 compact block. A larger generation run
is needed to verify this yield at scale.

## Decision

The geometry condition for a matched bridge training experiment IS met:
transformation-like bridge candidates occupy compact-like structural geometry
while differing primarily on source-absent content. The next step should determine
whether enough transformation-like outputs can be produced at scale (full 12,155 pool,
3 regimes, strict structural filtering). If the yield supports filling most of the
compact block, a same-DeBERTa training comparison is justified — it would be the
cleanest available test of whether novel vocabulary is necessary for the compact
data-efficiency mechanism.

## Files
- Atlas-compatible geometry: `data/bridge_atlas_geometry/bridge_atlas_geometry.{json,md}`
- Per-pair geometry CSV: `data/bridge_atlas_geometry/per_pair_atlas_geometry.csv`
- Content-only geometry (methodologically different, do not compare to atlas): `data/bridge_geometry_comparison/`
- Scripts: `scripts/bridge_geometry_comparison.py`, `bridge_atlas_geometry.py`
