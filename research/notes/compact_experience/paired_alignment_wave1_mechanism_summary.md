# paired alignment accounting audit — Wave-1 paired-alignment result and accounting repair

## Evidence files

- Fast screen summary: `experiments/archive/compact_experience/data/paired_alignment_eval/paired_alignment_wave1_eval_summary.json`
- Fast screen note: `research/notes/compact_experience/paired_alignment_wave1_eval.md`
- Accounting audit: `experiments/archive/compact_experience/data/paired_alignment_accounting_audit.json`
- Accounting note: `research/notes/compact_experience/paired_alignment_accounting_audit.md`
- Full Entity probe: `experiments/archive/compact_experience/data/paired_alignment_full_entity/full_entity_probe.json`
- Full Entity note: `research/notes/compact_experience/paired_alignment_full_entity_probe.md`

## What happened

Wave 1 completed cleanly: ALIGNED and MISMATCHED each ran 2,442 steps to 100,000,000 word exposure, with valid `chck_10M`–`chck_100M` checkpoints and final losses:

- ALIGNED: 9.7931 → 1.9463
- MISMATCHED: 9.7903 → 3.2272

I evaluated both plus the inherited INITIAL_MODEL_STUDIES earlier analysis `chck_100M` baseline using the known-good BabyLM strict invocation. The critical fast contrast is:

| contrast | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading | equal-7 | weighted fast proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned − mismatched | -1.510 | +2.800 | +0.730 | +10.640 | +0.650 | -3.530 | -0.415 | +1.338 | +0.996 |
| aligned − INITIAL_MODEL_STUDIES baseline | -3.220 | -4.400 | +1.540 | +6.500 | -1.690 | +1.400 | -0.170 | -0.006 | -0.007 |

The full Entity probe confirms the dominant effect is real and not a fast-subset artifact:

| target | full Entity |
|---|---:|
| ALIGNED | 29.160 |
| MISMATCHED | 18.450 |
| INITIAL_MODEL_STUDIES baseline | 21.780 |

Deltas: aligned − mismatched +10.710; aligned − baseline +7.380; mismatched − baseline −3.330.

## Scientific reading

The matched within-sequence adjacency effect is real and strongly concentrated in Entity. The aggregate fast equal-7 lift over MISMATCHED (+1.338) is mostly carried by Entity, not broad improvement. Compared with the inherited baseline, ALIGNED is a redistribution: Entity/EWoK/GPIQA up, BLiMP/Supplement/COMPS down. This does not yet establish a broad frontier shift.

The large ALIGNED loss advantage must be interpreted cautiously. Meaning-matched adjacency makes masked tokens easier to predict in context; loss can reflect copyability rather than better transferable structure. The low-loss arm still loses BLiMP/Reading vs MISMATCHED and BLiMP/Supplement/COMPS vs the baseline.

Interpretation caveat: ALIGNED − MISMATCHED bundles at least two things: meaning-related local coherence and rewrite/paraphrase correspondence. Until a same-topic non-synonymous coherent adjacency arm is built, the mechanism cannot be attributed specifically to paraphrase correspondence.

## Accounting correction

The existing Wave-2 script was **not** launched. The audit confirms why:

| arm | base words | exposure words | effective passes | extra beyond 10 passes | status |
|---|---:|---:|---:|---:|---|
| aligned | 9,999,840 | 100,000,000 | 10.000160 | 1,600 | exceeds literal 10 passes |
| mismatched | 9,999,840 | 100,000,000 | 10.000160 | 1,600 | exceeds literal 10 passes |
| single_repeat | 9,999,840 | 100,000,000 | 10.000160 | 1,600 | exceeds literal 10 passes |
| single_orig | 9,551,200 | 100,000,000 | 10.469889 | 4,488,000 | strongly exceeds literal 10 passes |

Consequences:

- Wave-1 ALIGNED/MISMATCHED remains a valid matched mechanism screen because both arms share the same 1,600-word overshoot.
- None of the current 100M paired-alignment runs should be treated as official-candidate compliant under a literal ≤10-epoch interpretation.
- Existing Wave 2 must not be launched: SINGLE_ORIG would run ~10.47 passes and confound the intended breadth control.
- Any official-candidate rerun must either construct exactly 10,000,000-word pools or cap exposure at exactly 99,998,400 words and handle checkpoint naming explicitly.

## Next high-value work

The decisive next construction is the coherent-adjacency control. A semantic-relation ladder is needed:

1. **PARAPHRASE/ALIGNED** — existing arm: adjacent same-meaning rewrite.
2. **COHERENT-NONSYN** — adjacent same-topic or document-adjacent text that is not a paraphrase.
3. **MISMATCHED** — existing arm: adjacent unrelated text.

Contrasts to isolate:

- COHERENT-NONSYN − MISMATCHED = contribution of local topical/entity coherence.
- PARAPHRASE − COHERENT-NONSYN = additional contribution of same-meaning rewrite correspondence.

If PARAPHRASE ≈ COHERENT-NONSYN > MISMATCHED, the effect is coherence rather than rewrite correspondence, and the cheaper productive path becomes document-coherent windowing of legal corpus text. If PARAPHRASE > COHERENT-NONSYN > MISMATCHED, paired rewrite correspondence is a genuine data-efficiency signal worth scaling under exact epoch/legal accounting.
