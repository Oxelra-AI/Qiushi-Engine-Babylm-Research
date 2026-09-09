# relation filtered pair pools — relation four-cell calibration decision and rebuild direction

## Question

muon50 gp and relation pair synthesis produced 18,651 legal active-token two-context/two-target pairs. relation filtered pair pools tested whether this reservoir is connected to the established context-conditioned relation failure (EWoK conditional reversals and GlobalPIQA_parallel hard ranks) rather than to ordinary attested-context target recovery.

The decisive calibration was frozen-weight/no-update: for pair `(Ca,Ta),(Cb,Tb)`, score

`M = s(Ca,Ta) + s(Cb,Tb) - s(Ca,Tb) - s(Cb,Ta)`

with the target positions masked in the same 256-token active view. Same-stratum target-permuted and context-permuted nulls tested whether checkpoint separation survives permutation. Checkpoints: FW compact, row-block breadth, and interleaved breadth 100M.

## Evidence produced

Main files:

- scorer: `experiments/archive/representation_and_objectives/scripts/relation_fourcell_calibration.py`
- full calibration: `experiments/archive/representation_and_objectives/data/relation_fourcell_calibration/fourcell_calibration_summary.json`
- full scores: `experiments/archive/representation_and_objectives/data/relation_fourcell_calibration/fourcell_pair_scores.jsonl`
- synthesis: `experiments/archive/representation_and_objectives/data/relation_fourcell_synthesis/relation_fourcell_full_synthesis.json`
- synthesis note: `research/notes/representation_and_objectives/relation_fourcell_synthesis.md`
- model-free semantic audit: `experiments/archive/representation_and_objectives/data/relation_pair_pool_semantic_audit/relation_pair_pool_semantic_audit.json`
- subpool counts/materialization: `research/notes/representation_and_objectives/relation_pair_subpool_counts.md`, `research/notes/representation_and_objectives/relation_filtered_pair_pools.md`

Technical validity checks:

- Full calibration scored 18,643/18,651 pairs; only 8 lacked same-stratum donor assignment.
- Strict target-position/token verification passed for every scored context: `bad_contexts=0`.
- Donor assignment: 17,480 exact strata, 1,163 loose strata.
- It was evaluation-only; no model update or training was run.

## Main results

All-pair means:

| margin | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| true four-cell `M` | 11.1819 | 11.1662 | 11.2021 | interleaved > compact > rowblock | 0.0359 |
| target-perm null | 0.2394 | 0.2234 | 0.2268 | compact > interleaved > rowblock | 0.0160 |
| context-perm null | 0.2086 | 0.2196 | 0.1946 | rowblock > compact > interleaved | 0.0250 |

Relation-oriented deltas on all pairs:

- `rowblock - compact`: true `-0.0157`, target-null `-0.0160`, context-null `+0.0110`.
- `interleaved - compact`: true `+0.0202`, target-null `-0.0126`, context-null `-0.0140`.
- `rowblock - interleaved`: true `-0.0359`, target-null `-0.0034`, context-null `+0.0250`.

Subpool synthesis:

| subset | n | true order/range | null ranges | interpretation |
|---|---:|---|---|---|
| all | 18,643 | interleaved > compact > rowblock, range 0.036 | target 0.016, context 0.025 | tiny arm separation, not rowblock-parallel pattern |
| non-generic | 16,571 | interleaved > compact > rowblock, range 0.031 | target 0.020, context 0.020 | same pattern |
| candidate broad | 5,331 | interleaved > compact > rowblock, range 0.063 | target 0.061, context 0.052 | true no stronger than nulls |
| candidate strict | 1,569 | interleaved > compact > rowblock, range 0.050 | target 0.148, context 0.030 | target null larger |
| non-generic local10 | 1,599 | interleaved > compact > rowblock, range 0.066 | target 0.120, context 0.053 | target null larger |
| non-generic local15 | 374 | rowblock > compact > interleaved, range 0.090 | target 0.230, context 0.089 | order looks relation-like but null is stronger/equal |
| non-generic char50 | 1,219 | interleaved > compact > rowblock, range 0.084 | target 0.142, context 0.030 | target null larger |
| comparative all | 77 | rowblock > interleaved > compact, range 0.198 | target 0.371, context 0.268 | comparative not hidden positive; nulls dominate |

Model-free audit of the original pool:

- Antonym/opposition hints: only 2/18,651 pairs.
- Local target-window Jaccard mean: 0.049.
- Context word Jaccard mean: 0.104.
- Generic target rate: 11.1% either target generic, 1.3% both.
- Top targets are ordinary words capped at 32 uses: `give`, `table`, `real`, `five`, `live`, `actually`, `really`, `well`, `show`, etc.
- The stricter non-generic local-overlap ≥0.15 subpool has only 374 pairs and 1 comparative pair; its apparent rowblock ordering is not stronger than nulls.

## Scientific judgment

The broad active pair reservoir is **not** a safe relation-training objective. The huge true four-cell margin (~11 nats, ~98% positive) shows that the models strongly recover each attested target from its own sentence and strongly reject a different same-length target from another sentence. That is local/contextual coherence, not the unsaturated conditional-alternative choice that remains weak on EWoK and GlobalPIQA_parallel.

The checkpoint-separation test fails in the sense that matters for this route:

1. The true arm range is tiny relative to pair variance and only slightly above or comparable to null ranges.
2. The all-pair order is not the GlobalPIQA_parallel rowblock movement (`rowblock` strongest) and not a clean EWoK hard-interaction repair signal.
3. In stricter and comparative subpools, any relation-like rowblock ordering is matched or exceeded by target/context permutation nulls.
4. Model-free semantics show that the pool mostly pairs unrelated sentence-specific content words rather than competing alternatives in a shared relational slot.

Therefore this muon50 gp and relation pair synthesis/128 pool should **not** be used for exposure-matched H100 training. Training on it would likely repeat the closed full context pivot substitution probe/125 pattern: strengthen an already-saturated local-fit signal while leaving the hard conditional choice unresolved or damaging broad capability.

## Rebuild direction

The interaction problem remains valid, but the pair semantics must be rebuilt. The next reservoir should increase true cross-cell competition before any training:

1. **Shared-slot alternatives, not arbitrary same-frequency targets.** Mine contexts whose local windows or templates around the target are much more similar, so `s(Ca,Tb)` and `s(Cb,Ta)` are plausible enough to make `M` unsaturated.
2. **Relation-pivot opposition or contrast.** Prefer pairs where pivots encode opposite or systematically different relations (`before/after`, `more/less`, `increase/decrease`, `in/out`, `above/below`, `because/although`, `if/unless`, negated/non-negated variants) using hand-coded legal lexicons rather than pretrained supervision.
3. **Repeated target-alternative sets.** Look for target pairs recurring across multiple contexts or multiple relation families so the objective is not memorizing one sentence pair.
4. **Family-resolved density.** Comparative and physical/spatial affordance-like families must not be concealed by causal/temporal counts; comparative had only 77 pairs in the current pool and no positive calibration beyond nulls.
5. **No-update calibration first.** A rebuilt pool must show (a) lower/smaller unsaturated true margins, (b) arm separation aligned with known relation readouts, and (c) target/context permutations erasing that separation before becoming a training objective.

## Optimizer side evidence now available

The pending muon switch 40m globalpiqa margin 40M broad Muon harvest landed during relation filtered pair pools:

- AdamW40 cheap7 42.2214.
- Continuous Muon40 cheap7 41.3993; relative to AdamW, +2.23 Supplement and +3.43 EWoK but −4.94 GlobalPIQA, −2.09 Entity, −1.63 COMPS, −1.24 Reading.
- Muon20→AdamW40 cheap7 41.8586; recovers GlobalPIQA close to AdamW but remains below AdamW cheap7 and loses Entity/Reading.

muon50 gp and relation pair synthesis/128 GlobalPIQA 40M→50M synthesis shows no hard-rank repair: at 50M, AdamW/continuous-Muon/Muon20→AdamW/Muon40→AdamW GlobalPIQA_parallel is 28.16/27.18/26.21/23.30, and hard52 accuracy remains 3.85–7.69% with deep margins 1.680–1.849 nats. Existing-weight 50M broad/EWoK readouts were launched as `s128_t47_tool1` and `s128_t48_tool1`; their results should be read before closing the abrupt empty-moment switch artifacts.
