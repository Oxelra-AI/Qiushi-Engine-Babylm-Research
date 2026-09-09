# coherent86 multiarm mechanism reading — coherent86 mechanism reading via multi-arm decision partition

## What was tested
Saved official-compatible prediction payloads only (no model inference), across five
endpoints on the exact same official items:
- `anchor82` — protected submitted rank-1 chck_82M (cheap7 43.95945, Overall 41.94248)
- `ordinary86` — exposure-matched ordinary scale1.75 chck_86M (cheap7 43.77071)
- `shuffled86` — frozen-82M private tail, shuffled replay (cheap7 44.01286)
- `coherent86` — frozen-82M private tail, coherent replay, alpha=1 (cheap7 44.10643)
- `spanbreak86` — frozen-82M private tail, structure-destroyed control (cheap7 43.12143)

Artifacts:
- `data/private_scale_panel_analysis/private_scale_panel_analysis.{json,md}`
- `data/multiarm_decision_partition/multiarm_decision_partition.{json,md}`

## Decisive findings

1. **coherent86 is a reweighting endpoint, not broad added competence.**
   Versus the anchor: net **−115** discrete common items (3116 gains / 3231 losses over
   170,722), yet cheap7 **+0.147**. The positive official-column movement comes from
   Supplement (payload Δ +0.712) and GlobalPIQA (+0.487) shifting *column means*, while
   COMPS loses net −137 items and EWoK −2. The pattern is: aggregate up, decision surface redistributed.

2. **coherent86 adds almost no genuinely novel-correct decisions.**
   Against the union of all other arms (ordinary86, shuffled86, spanbreak86), coherent86
   has only **366 unique gains** vs **425 unique losses** relative to the anchor. Most of
   its "gains" are decisions other arms also flip. It is not a distinct new capability set.

3. **Ordinary continuation does not explain the edge, but a hard ensemble does not help.**
   ordinary86 is weaker (cheap7 43.77, −0.189 vs anchor). But an anchor-tie hard majority
   vote across the four non-anchor arms scores discrete-mean **49.879**, *below* the anchor
   (49.928) and far below coherent (50.096). The endpoints are not complementary in a way a
   vote can recover; coherent86's edge is a specific per-column weighting, not extra
   independent correct decisions.

4. **The cost coherent86 pays is concrete: 1454 anchor-correct items lost that ordinary86
   preserves.** A retention-oriented route would need to protect these without benchmark
   labels — and the three closed selectors already show no corpus-derived signal tracks the
   82M peak, so this is not currently achievable as a training target.

## Scientific status
- coherent86 remains a **legitimate stronger endpoint candidate** (higher Supplement,
  GlobalPIQA, and cheap7; projected Overall(AoA0) 42.058–42.065 across 3 SuperGLUE seeds,
  robust to ±0.03 SuperGLUE variance). Publicly packaged and validated
  (`leslie721007/babylm-strict-small-coherent86`), prepared for submission.
- coherent86 is **not** a validated general slow-fast learning principle. The private fast
  path redistributes the anchor's decision surface; it does not add a stable, reproducible
  new competence set beyond what other same-exposure endpoints touch.

## What this implies for next work
- The frozen-anchor coherent-replay route has now been characterized as endpoint
  redistribution. Building Arm 4 retention-KL machinery on top of a redistribution effect
  is not justified by current evidence.
- The remaining pending no-training probe (private scale alpha 0.5/0.75) can only interpolate
  between anchor and coherent86; if no intermediate alpha improves both score and item
  retention, private-scale interpolation is closed as another redistribution result.
- The genuinely open scientific problem is unchanged: continued legal-corpus training
  reallocates relation/state/commonsense competence while MLM loss improves, and no
  corpus-derived label-free signal has captured the 82M peak. A distinct mechanism — not a
  reweighting of the same 82M surface — is needed to *add* stable relation/state competence
  under the 10M-word budget.
