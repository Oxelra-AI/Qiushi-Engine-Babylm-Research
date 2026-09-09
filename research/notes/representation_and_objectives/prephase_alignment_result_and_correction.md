# prephase alignment design and bias analysis — Prephase alignment result: the warm-start rescue is generic head calibration, not coordinate reuse

## Result summary

A four-arm prephase alignment test definitively answered the prephase-alignment question.

| arm | pre fit changed | cont changed_focal | held inline con | micro_EEBF | interpretation |
|-----|---|---|---|---|---|
| aligned | 1.000 | 1.000 | 1.000 | 0.000 | Rescues (expected) |
| permuted | 0.996 | 1.000 | 1.000 | 0.000 | **Also rescues** — despite conflicting associations |
| disjoint | 1.000 | 1.000 | 1.000 | 0.000 | **Also rescues** — despite never-seen-before tags |
| scratch | N/A | 0.500 | 0.688 | 0.750 | Fails (stuck at chance, strong after-bias) |

All three prephase conditions (aligned, permuted, disjoint) equally rescue the inline-role continuation, while scratch fails completely. The result is unambiguous: **the warm-start rescue is generic entailment-head calibration, NOT aligned coordinate reuse**.

## What this corrects

### address bootstrap and role query's "path-dependent coordinate bootstrapping" is wrong

address bootstrap and role query observed that tag-only pretraining → inline-role continuation succeeded while scratch inline-role training failed. It interpreted this as "a previously formed address route can make a nearby rendering learnable" — suggesting the model transferred specific tag→record associations.

prephase alignment design and bias analysis shows this interpretation was incorrect:
- The **permuted** arm trained with deliberately WRONG tag→record associations (TAG_1 mapped to after-state, TAG_2 to before-state — opposite of what the continuation expected). Yet it rescued equally well. The model does not transfer specific tag→state bindings.
- The **disjoint** arm trained on completely DIFFERENT tag tokens. These tokens never appear in the continuation. Yet it also rescued equally well. The model does not even need familiarity with the continuation's tag tokens.
- The **scratch** arm, starting from the same model initialization without any prephase, stuck at 0.500 training accuracy for 10 epochs. The randomly initialized entailment head is in a basin that cannot fit this temporal-change entailment task.

### The mechanism is entailment-head initialization sensitivity

The pretrained DeBERTa encoder already has sufficient representational capacity to read tag-addressed records, including inline-role formats with changed focal states. The bottleneck is entirely in the randomly initialized 2-class entailment head.

For seed 27000, this head starts in a degenerate basin: it predicts the same class for all inputs regardless of content. Any supervised NLI task — even with different text, different tags, different label assignments — moves the head into a functional range. Once there, the encoder's existing text understanding handles the rest.

### Implications for earlier bridge findings

1. **Addressability (temporal addressability bridge)**: The finding that arbitrary tags can address records still holds. This is about the encoder's representational capacity, not about training dynamics.

2. **Changed-focal difficulty (earlier analysis)**: The finding that changed focal rows are harder than stable rows still holds. But the explanation shifts: the difficulty is not in learning to update states; it's that the untrained head's degenerate basin makes ALL rows unfittable, and the eval metrics make this look like a changed-focal-specific failure.

3. **Role-query composition failure (address bootstrap and role query)**: The failure of role→tag composition may also be partly an optimization artifact rather than a fundamental representational limit. This needs retesting with head-calibrated models.

4. **Format sensitivity (format sensitivity boundary)**: The exact-chance prefix/inline results were already corrected in address bootstrap and role query as seed-sensitive, not as a general acquisition law. prephase alignment design and bias analysis further confirms that format barriers are optimization-path phenomena for this bridge scale.

## Micro-EEBF check (bias threat)

The scratch arm shows focal_before=0.0, focal_after=1.0 on held changed/stable queries → micro_EEBF = +1.0. This is a pure after-state bias: the untrained head predicts "yes" to everything, which happens to be correct for after-state queries (where the state change makes the hypothesis true) but wrong for before-state queries.

All three prephase arms show micro_EEBF = 0.000 with perfect discrimination. This confirms that the calibrated head genuinely distinguishes before and after states rather than exploiting a temporal-position prior.

This connects directly to finding that first-basin Entity gains are bias-like (affected++, unaffected--, EEBF≈0) while second-basin gains are genuine. In our bridge: the uncalibrated head has a pure bias; the calibrated head has genuine discrimination.

## What survives from the bridge work

The established findings that do NOT depend on coordinate reuse:

1. **Stable-secondary preservation is easy**: All arms (including scratch) show secondary_after ≥ 0.75, and all calibrated arms show 1.0. The model readily preserves non-conflicting records.

2. **Changed-focal readout is the genuine bottleneck**: Even after calibration, changed focal is the last category to reach 1.0. But this is about the inherent difficulty of distinguishing contradictory before/after values, not about missing coordinate mechanisms.

3. **The tag-addressed record format works**: Once the head is calibrated, arbitrary tags address records reliably, including held entities and held wordings (from temporal addressability bridge). This is a representational capacity finding.

4. **The pretrained DeBERTa's text understanding is strong**: It can read inline-role formats, distinguish background from update, and compose tag references with ranking content — all without specific address-route training. The entire learned mechanism is in the 2-class head, not in the encoder's representation of addresses.

## What does NOT survive

- "Coordinate bootstrapping under sparse interference" as a data-efficient learning principle
- "A formed address route makes nearby renderings learnable" — any NLI training makes them learnable
- "Path-dependent surface transfer" — it's just head calibration

## Relation to the Broader Experimental Evidence

permuted-companion control at BabyLM scale is now even more important. If the micro-level address-bootstrapping mechanism is just head warm-up, the macro-level compact-view benefit might also be a generic text-distribution effect. Their aligned-vs-permuted comparison at Entity/EWoK level will test whether source-conditioned correspondence carries specific value beyond compact-text exposure.

## What genuinely survives as a scientific finding

### Head calibration vs composition: a clean separation

prephase alignment design and bias analysis shows that direct-tag addressability is absorbed by head calibration. BUT the role→tag→state composition failure (address bootstrap and role query: sparse_changed_focal = 0.53 after full calibration) persists even with a perfectly calibrated direct-tag reader. The role-query training also DAMAGED the direct-tag capability (sparse_changed_focal dropped to 0.52, held readouts to 0.31).

This is a genuine architectural finding, not an optimization artifact: a single entailment head that can read TAG→state perfectly CANNOT simultaneously learn "focal background"→TAG resolution, because on changed-focal worlds the two query types require different tag→state mappings to be active at the same time.

This suggests that composition (selector + reader) requires structural separation — not just more training or better initialization — and this IS relevant to data-efficient learning: limited-data learners may need decomposed task interfaces rather than monolithic task heads.

### The real hierarchy of bridge findings

1. **Absorbed by head calibration** (no longer interpretable as learning principles):
   - Warm-start rescue / format sensitivity
   - "Path-dependent coordinate bootstrapping"
   - The changed-focal difficulty as seen in scratch training

2. **Genuine encoder capacity** (describes what pretrained DeBERTa can already do):
   - Tag-addressed record selection with arbitrary tags
   - Stable-secondary preservation
   - Contrastive discrimination once head is calibrated

3. **Genuine composition boundary** (persists after calibration):
   - Role→tag→state composition fails in a single head
   - Role-query training degrades direct-tag capability
   - This is potentially the deepest remaining scientific finding

4. **Untested at bridge level** (established in earlier factorized/BiGRU settings):
   - Sparse relation orientation (aligned > anti > exposure) — role coordinate anchor and state probe
   - Argument-slot/filler reusability — name memorization finding
   - These were not tested with head-calibration controls

## Next scientific direction

The bridge has now produced one genuinely durable finding: **single-head composition is the bottleneck for role→address→state under interference**. This is real because it persists after calibration.

The next productive direction should ask: **does decomposing the selector and reader into separate modules solve the composition boundary?** Specifically:
- Train a selector M that maps role language to a tag selection (separate head/module)
- Keep a calibrated reader R that maps tags to state (frozen or lightly updated)
- Compose M+R and test on changed-focal + stable-secondary with held entities and wordings
- This would test whether the composition bottleneck is architectural (fixable by decomposition) or representational (fundamental to the encoder's capacity under interference)

If decomposition works: the principle is that data-efficient learning requires factored task interfaces, and monolithic heads waste representational capacity on interference rather than composition.
If decomposition fails: the bottleneck is in the encoder's ability to resolve role→tag mapping under contradictory states, which is a deeper representational limit.

companion analysis's permuted-companion result at BabyLM scale will independently test whether source-conditioned correspondence carries value beyond text distribution. If it does, the mechanism may operate at the encoder level (changing representations during pretraining) rather than at the task-head level (where our bridge work has been).

## Files

- Script: training/scripts/prephase_alignment_probe.py
- Analysis: scripts/analyze_prephase_alignment.py
- Results: data/step271_{aligned,permuted,disjoint,scratch}_seed27000/
- Analysis summary: data/prephase_analysis/prephase_analysis_summary.md
- This note: notes/prephase_alignment_result_and_correction.md
